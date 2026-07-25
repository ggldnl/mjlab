"""Run the diffdrive demonstration: drive and turn, alternated by a scripted
controller, composed through one bridging architecture.

A fixed-step controller runs drive for a while, switches to turn, holds turn for
a while, switches back, and repeats (see ``controller.py``). Under architecture 0
(the no-bridge baseline, a direct hand-off), turn takes over while the robot
still carries drive's forward momentum, so each 90 deg turn traces a wide arc
instead of a clean pivot. Commanding four such turns should walk a square and
return to the start; the momentum makes it drift off instead. That drift is the
failure a real bridge has to remove, and this demo is the harness to watch it,
measure it, and later drop a trained architecture in place of the naive one.

Watch it live (analytical experts, architecture 0):

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.demo

Use trained checkpoints instead of the analytical experts:

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.demo --no-analytical

Measure the square-closure error over four turns, headless:

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.demo --viewer headless --steps 840
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.skills.architectures import ARCHITECTURES
from mjlab.tasks.skills.experiments.diffdrive.controller import DiffdriveController
from mjlab.tasks.skills.experiments.diffdrive.dynamics import (
  analytical_drive,
  analytical_turn,
)
from mjlab.tasks.skills.meta import ComposedPolicy
from mjlab.tasks.skills.skill import PolicySkill, SkillPool
from mjlab.tasks.skills.utils import retrieve_latest_checkpoint


@dataclass(frozen=True)
class DemoConfig:
  # Use the analytical experts from dynamics.py. With --no-analytical, load trained checkpoints instead
  analytical: bool = True

  # Skip the architecture entirely and open a viewer that runs one skill at a
  # time, letting you pick between them to test each in isolation. Needs an
  # interactive viewer (viser or native), not headless
  debug: bool = False

  # which architecture to run (see architectures/__init__.py). Defaults to 0, no-bridge direct hand-off baseline
  architecture: int = 0

  # Viewer
  viewer: Literal["viser", "native", "headless"] = "viser"

  # The two skill tasks. They share one observation/action space, so either
  # one's env serves as the arena; this demo builds it from the drive task
  drive_task_id: str = "Mjlab-Diffdrive-Drive"
  turn_task_id: str = "Mjlab-Diffdrive-Turn"

  # Checkpoints for the non-analytical skills. None picks the latest trained
  # checkpoint for the corresponding task
  drive_checkpoint: str | None = None
  turn_checkpoint: str | None = None

  # Analytical drive cruise speed [m/s]: high enough to build the momentum a
  # naive hand-off fails to shed
  drive_speed: float = 8.0

  # Analytical turn target angle [rad]. The square test wants 90 deg
  turn_angle: float = math.pi / 2

  # Analytical turn forward speed [m/s]: nonzero makes the turn an arc rather
  # than a pivot. Kept below drive_speed so the momentum gap survives
  turn_speed: float = 0.01

  # Controller parameters: after how many steps we transition from one skill
  # to the other
  straight_steps: int = 1000
  turn_steps: int = 200

  num_envs: int = 1
  device: str | None = None

  # If headless, run for this amount of steps
  steps: int = 10


def _build_pool(cfg: DemoConfig, env: ManagerBasedRlEnv, device: str) -> SkillPool:
  if cfg.analytical:
    return SkillPool(
      [
        analytical_drive(speed=cfg.drive_speed),
        analytical_turn(target_angle=cfg.turn_angle, forward_speed=cfg.turn_speed),
      ]
    )
  drive_ckpt = cfg.drive_checkpoint or retrieve_latest_checkpoint(cfg.drive_task_id)
  turn_ckpt = cfg.turn_checkpoint or retrieve_latest_checkpoint(cfg.turn_task_id)
  return SkillPool(
    [
      PolicySkill("drive", cfg.drive_task_id, drive_ckpt, env, device),
      PolicySkill("turn", cfg.turn_task_id, turn_ckpt, env, device),
    ]
  )


def _run_headless(env: ManagerBasedRlEnv, policy: ComposedPolicy, steps: int) -> None:
  """Step the composition and report how far the robot lands from its start.

  A perfectly bridged square returns to its start (closure error ~0); a naive
  hand-off drifts, and the error is the size of that drift.
  """
  robot = env.scene["robot"]
  obs, _ = env.reset()
  start_xy = robot.data.root_link_pos_w[:, :2].clone()
  for _ in range(steps):
    obs, _, _, _, _ = env.step(policy(obs))
  end_xy = robot.data.root_link_pos_w[:, :2]
  closure = torch.linalg.norm(end_xy - start_xy, dim=-1)
  for i in range(env.num_envs):
    print(
      f"env {i}: start={start_xy[i].tolist()} end={end_xy[i].tolist()} "
      f"closure_error={closure[i].item():.3f} m"
    )


class _SkillSelector:
  """Drives every env with one skill at a time, picked interactively.

  This is the debug path: no controller, no bridge, just the pool. It runs the
  currently selected skill on the whole batch so a single expert can be watched
  in isolation, and switching selection lets a stateful skill (turn) re-arm on
  its next rising edge exactly as it would in the real composition.
  """

  def __init__(self, pool: SkillPool, num_envs: int, device: str) -> None:
    self.pool = pool
    self.num_envs = num_envs
    self.device = device
    self.index = 0
    pool.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

  @property
  def name(self) -> str:
    return self.pool.skills[self.index].name

  def select(self, index: int) -> None:
    self.index = index

  def cycle(self) -> None:
    self.index = (self.index + 1) % len(self.pool.skills)

  def __call__(self, obs) -> torch.Tensor:
    assignment = torch.full(
      (self.num_envs,), self.index, dtype=torch.long, device=self.device
    )
    return self.pool.act(obs, assignment)


def _run_debug(cfg: DemoConfig, env: ManagerBasedRlEnv, pool: SkillPool) -> None:
  if cfg.viewer == "headless":
    raise ValueError("--debug needs an interactive viewer (viser or native).")

  selector = _SkillSelector(pool, env.num_envs, env.device)
  names = [skill.name for skill in pool.skills]
  viewer_env = RslRlVecEnvWrapper(
    env, clip_actions=load_rl_cfg(cfg.drive_task_id).clip_actions
  )

  def debug_policy(obs) -> torch.Tensor:
    return selector(obs)

  if cfg.viewer == "native":
    from mjlab.viewer import NativeMujocoViewer

    print("[debug] press N in the viewer to cycle the active skill.")

    def on_key(keycode: int) -> None:
      if keycode == ord("N"):
        selector.cycle()
        print(f"[debug] active skill -> {selector.name}")

    NativeMujocoViewer(viewer_env, debug_policy, key_callback=on_key).run()
  else:
    import viser

    from mjlab.viewer import ViserPlayViewer

    server = viser.ViserServer(label="mjlab")
    with server.gui.add_folder("Debug"):
      buttons = server.gui.add_button_group("Active skill", options=names)

    @buttons.on_click
    def _(event) -> None:
      selector.select(names.index(event.target.value))

    ViserPlayViewer(viewer_env, debug_policy, viser_server=server).run()

  viewer_env.close()


def run_demo(cfg: DemoConfig) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(cfg.drive_task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  pool = _build_pool(cfg, env, device)

  # Debug: bypass the controller/bridge and test each skill on its own.
  if cfg.debug:
    _run_debug(cfg, env, pool)
    env.close()
    return

  if cfg.architecture not in ARCHITECTURES:
    raise ValueError(
      f"Unknown architecture {cfg.architecture}; registered: {sorted(ARCHITECTURES)}."
    )

  controller = DiffdriveController(
    pool, straight_steps=cfg.straight_steps, turn_steps=cfg.turn_steps
  )
  meta = ARCHITECTURES[cfg.architecture](env, pool)
  policy = ComposedPolicy(env, controller, meta)

  if cfg.viewer == "headless":
    _run_headless(env, policy, cfg.steps)
    env.close()
    return

  viewer_env = RslRlVecEnvWrapper(
    env, clip_actions=load_rl_cfg(cfg.drive_task_id).clip_actions
  )

  # The viewer's PolicyProtocol types its obs loosely; wrap the composition in a
  # plain callable, as skill.py's viewer does, so the dict-typed __call__ fits.
  def viewer_policy(obs) -> torch.Tensor:
    return policy(obs)

  if cfg.viewer == "viser":
    from mjlab.viewer import ViserPlayViewer

    ViserPlayViewer(viewer_env, viewer_policy).run()
  else:
    from mjlab.viewer import NativeMujocoViewer

    NativeMujocoViewer(viewer_env, viewer_policy).run()

  viewer_env.close()


if __name__ == "__main__":
  run_demo(tyro.cli(DemoConfig))
