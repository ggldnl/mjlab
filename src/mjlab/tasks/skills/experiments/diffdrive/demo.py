"""
Run the diffdrive demonstration: drive and turn, alternated by a scripted
controller, composed through one bridging architecture.

A fixed-step controller runs drive for a while, switches to turn, holds turn for a
while, switches back, and repeats (see controller.py). Under architecture 0 (the
no-bridge baseline, a direct hand-off), turn takes over while the robot still carries
drive's cruise speed, so its tight arc's lateral acceleration rolls the tall chassis
onto its side: it tips. That tip is the failure a real bridge has to remove, by
braking the robot into turn's low-speed regime before handing over.

Analytical experts are recommended over RL.

Watch the baseline live (analytical experts, architecture 0, no training needed):

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.demo

Watch a trained architecture instead (train it first with this experiment's
train.py). Its state is restored from a run directory; --checkpoint points at one,
and if omitted the latest trained run for that architecture is used:

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.demo --architecture 1
    uv run python -m mjlab.tasks.skills.experiments.diffdrive.demo \\
        --architecture 1 --checkpoint logs/skills/diffdrive/arch_1/<run>

Use trained RL checkpoints instead of the analytical experts (discouraged):

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.demo --no-analytical

Measure how often the robot tips, headless:

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.demo --viewer headless --steps 1500

Manually control which skill is currently active:

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.demo --debug
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.skills.architectures import ARCHITECTURES
from mjlab.tasks.skills.experiments.diffdrive import (
  DRIVE_SPEED,
  DRIVE_TASK_ID,
  EXPERIMENT_NAME,
  TURN_ANGLE,
  TURN_SPEED,
  TURN_TASK_ID,
  build_pool,
)
from mjlab.tasks.skills.experiments.diffdrive.controller import DiffdriveController
from mjlab.tasks.skills.meta import ComposedPolicy
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.utils import retrieve_latest_architecture_checkpoint


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

  # Architecture checkpoint: a run directory written by this experiment's train.py,
  # holding whatever the architecture needs to restore. None picks the latest
  # trained run for this architecture. Ignored for architecture 0, which trains nothing
  checkpoint: str | None = None

  # Viewer
  viewer: Literal["viser", "native", "headless"] = "viser"

  # The two skill tasks. They share one observation/action space, so either
  # one's env serves as the arena; this demo builds it from the drive task
  drive_task_id: str = DRIVE_TASK_ID
  turn_task_id: str = TURN_TASK_ID

  # Checkpoints for the non-analytical skills. None picks the latest trained
  # checkpoint for the corresponding task
  drive_checkpoint: str | None = None
  turn_checkpoint: str | None = None

  # Analytical drive cruise speed [m/s]: high enough that a naive hand-off to turn
  # tips the tall chassis
  drive_speed: float = DRIVE_SPEED

  # Analytical turn target angle [rad]. The demo commands 90 deg turns
  turn_angle: float = TURN_ANGLE

  # Analytical turn arc speed [m/s]: the low speed the arc is safe at. Kept below
  # drive_speed so the momentum gap (and the tip) survives
  turn_speed: float = TURN_SPEED

  # Controller parameters: after how many steps we transition from one skill
  # to the other
  straight_steps: int = 300
  turn_steps: int = 150

  num_envs: int = 1
  device: str | None = None

  # If headless, run for this many steps
  steps: int = 1500


def _build_pool(cfg: DemoConfig, env: ManagerBasedRlEnv, device: str) -> SkillPool:
  return build_pool(
    env,
    device,
    analytical=cfg.analytical,
    drive_task_id=cfg.drive_task_id,
    turn_task_id=cfg.turn_task_id,
    drive_checkpoint=cfg.drive_checkpoint,
    turn_checkpoint=cfg.turn_checkpoint,
    drive_speed=cfg.drive_speed,
    turn_angle=cfg.turn_angle,
    turn_speed=cfg.turn_speed,
  )


def _run_headless(env: ManagerBasedRlEnv, policy: ComposedPolicy, steps: int) -> None:
  """Step the composition and report how badly the robot tips.

  A naive hand-off rolls the robot over every time it turns at speed, which shows up
  as tip-over terminations and a large peak tilt. A good bridge brakes first, so it
  never tips: zero tip-overs and a small peak tilt.
  """
  robot = env.scene["robot"]
  obs, _ = env.reset()
  max_tilt = torch.zeros(env.num_envs, device=env.device)
  tips = torch.zeros(env.num_envs, device=env.device)
  for _ in range(steps):
    obs, _, terminated, _, _ = env.step(policy(obs))
    # projected_gravity_b[:, 2] is -1 upright; tilt from upright is acos(-that).
    tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    max_tilt = torch.maximum(max_tilt, tilt)
    tips += terminated.float()  # tipped_over is the only non-timeout termination
  for i in range(env.num_envs):
    print(
      f"env {i}: tip_overs={int(tips[i].item())}  "
      f"max_tilt={math.degrees(max_tilt[i].item()):.1f} deg"
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

    print("[DEBUG] press N in the viewer to cycle the active skill.")

    def on_key(keycode: int) -> None:
      if keycode == ord("N"):
        selector.cycle()
        print(f"[DEBUG] active skill -> {selector.name}")

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

    ViserPlayViewer(
      viewer_env,
      debug_policy,
      viser_server=server,
      info_provider=lambda _idx: selector.name,
    ).run()

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

  # Every architecture but the baseline carries trained state that must be restored.
  # The checkpoint is a run directory from train.py; default to the latest one.
  if cfg.architecture != 0:
    run_dir = cfg.checkpoint or retrieve_latest_architecture_checkpoint(
      EXPERIMENT_NAME, cfg.architecture
    )
    print("[INFO] selected checkpoint: ", run_dir)
    meta.load(Path(run_dir))

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

    # Show the running skill (or bridge) for the displayed env in the info box.
    ViserPlayViewer(viewer_env, viewer_policy, info_provider=policy.active_label).run()
  else:
    from mjlab.viewer import NativeMujocoViewer

    NativeMujocoViewer(viewer_env, viewer_policy).run()

  viewer_env.close()


if __name__ == "__main__":
  run_demo(tyro.cli(DemoConfig))
