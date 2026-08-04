"""Run the parkour demonstration: walk, run and jump down a corridor of obstacles.

The controller picks the skill from position alone (see controller.py): walk between
obstacles, run when the next one is far off, jump when it is close. Under architecture
0 (the no-bridge baseline, a direct hand-off) the jump is handed a robot that is still
carrying a run's momentum, and is asked to reproduce a clip that opens from a stand.
That is the failure a real bridge has to remove.

Watch the baseline (needs all three skills trained; no bridge training needed):

    uv run python -m mjlab.tasks.skills.experiments.parkour.demo

Watch a trained architecture instead. Its state is restored from a run directory;
--checkpoint points at one, and if omitted the latest trained run is used:

    uv run python -m mjlab.tasks.skills.experiments.parkour.demo --architecture 1

Count the failures, headless:

    uv run python -m mjlab.tasks.skills.experiments.parkour.demo \\
        --viewer headless --num-envs 64 --steps 3000

Manually control which skill is active, to check each one on its own:

    uv run python -m mjlab.tasks.skills.experiments.parkour.demo --debug
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.skills.architectures import build
from mjlab.tasks.skills.experiment import Experiment
from mjlab.tasks.skills.experiments.parkour import (
  ENTITY_NAME,
  EXPERIMENT_NAME,
  JUMP_TASK_ID,
  RUN_TASK_ID,
  WALK_TASK_ID,
  build_experiment,
)
from mjlab.tasks.skills.experiments.parkour.arena import CORRIDOR, parkour_arena_env_cfg
from mjlab.tasks.skills.experiments.parkour.controller import ParkourController
from mjlab.tasks.skills.meta import ComposedPolicy
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.utils import retrieve_latest_architecture_checkpoint


@dataclass(frozen=True)
class DemoConfig:
  # Skip the architecture entirely and open a viewer that runs one skill at a time,
  # letting you pick between them to test each in isolation. Needs an interactive
  # viewer (viser or native), not headless
  debug: bool = False

  # Which architecture to run (see architectures/__init__.py). Defaults to 0, the
  # no-bridge direct hand-off baseline
  architecture: int = 0

  # Architecture checkpoint: a run directory written by this experiment's train.py.
  # None picks the latest trained run. Ignored for architecture 0, which trains nothing
  checkpoint: str | None = None

  viewer: Literal["viser", "native", "headless"] = "viser"

  walk_task_id: str = WALK_TASK_ID
  run_task_id: str = RUN_TASK_ID
  jump_task_id: str = JUMP_TASK_ID

  # Skill checkpoints. None picks the latest trained one for the corresponding task
  walk_checkpoint: str | None = None
  run_checkpoint: str | None = None
  jump_checkpoint: str | None = None

  # How close to an obstacle the controller calls the jump [m]
  jump_distance: float = 0.8

  # A clear stretch at least this long is worth running [m]
  run_distance: float = 6.0

  num_envs: int = 1
  device: str | None = None

  # If headless, run for this many steps
  steps: int = 3000


def _build_experiment(
  cfg: DemoConfig, env: ManagerBasedRlEnv, device: str
) -> Experiment:
  return build_experiment(
    env,
    device,
    walk_task_id=cfg.walk_task_id,
    run_task_id=cfg.run_task_id,
    jump_task_id=cfg.jump_task_id,
    walk_checkpoint=cfg.walk_checkpoint,
    run_checkpoint=cfg.run_checkpoint,
    jump_checkpoint=cfg.jump_checkpoint,
  )


def _run_headless(env: ManagerBasedRlEnv, policy: ComposedPolicy, steps: int) -> None:
  """Step the composition and report how far down the corridor it gets.

  Two numbers say whether the composition works. Falls counts the hand-offs that went
  wrong. Obstacles cleared counts what the corridor was for: a robot that never falls
  because it never leaves the ground is not succeeding, and distance alone would not
  tell the two apart.
  """
  robot = env.scene[ENTITY_NAME]
  obs, _ = env.reset()

  start_x = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  furthest = start_x.clone()
  falls = torch.zeros(env.num_envs, device=env.device)
  max_tilt = torch.zeros(env.num_envs, device=env.device)

  for _ in range(steps):
    obs, _, terminated, _, _ = env.step(policy(obs))
    x = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    # A fall resets the env, so progress has to be latched before it is lost.
    furthest = torch.maximum(furthest, x)
    tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    max_tilt = torch.maximum(max_tilt, tilt)
    falls += terminated.float()

  obstacle_x = torch.tensor([o.x for o in CORRIDOR], device=env.device)
  cleared = (furthest.unsqueeze(1) > obstacle_x.unsqueeze(0)).sum(dim=1)

  print(f"\n{'env':>5} {'falls':>7} {'furthest':>10} {'cleared':>9} {'max tilt':>10}")
  for i in range(env.num_envs):
    print(
      f"{i:>5} {int(falls[i].item()):>7} {furthest[i].item():>9.1f}m "
      f"{int(cleared[i].item()):>4}/{len(CORRIDOR):<4} "
      f"{math.degrees(max_tilt[i].item()):>9.0f}d"
    )
  print(
    f"\nmean falls {falls.mean().item():.2f}   "
    f"mean obstacles cleared {cleared.float().mean().item():.2f} of {len(CORRIDOR)}"
  )


class _SkillSelector:
  """Drives every env with one skill at a time, picked interactively.

  The debug path: no controller, no bridge, just the pool. Selecting a skill calls
  its `reset`, so the jump re-anchors its clip to wherever the robot currently is,
  exactly as it would in the real composition.
  """

  def __init__(self, pool: SkillPool, num_envs: int, device: str) -> None:
    self.pool = pool
    self.num_envs = num_envs
    self.device = device
    self.index = 0
    self._all = torch.ones(num_envs, dtype=torch.bool, device=device)
    pool.reset(self._all)

  @property
  def name(self) -> str:
    return self.pool.skills[self.index].name

  def select(self, index: int) -> None:
    if index == self.index:
      return
    self.index = index
    self.pool.skills[index].reset(self._all)

  def cycle(self) -> None:
    self.select((self.index + 1) % len(self.pool.skills))

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
    env, clip_actions=load_rl_cfg(cfg.jump_task_id).clip_actions
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

  env_cfg = parkour_arena_env_cfg()
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  exp = _build_experiment(cfg, env, device)

  # Debug: bypass the controller/architecture and test each skill on its own.
  if cfg.debug:
    _run_debug(cfg, env, exp.pool)
    env.close()
    return

  controller = ParkourController(
    exp.pool,
    jump_distance=cfg.jump_distance,
    run_distance=cfg.run_distance,
    entity_name=ENTITY_NAME,
  )
  # Built from the same experiment training used: the checkpoint's networks are sized by
  # its view.
  meta = build(env, cfg.architecture, exp)

  # Every architecture but the baseline carries trained state that must be restored.
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
    env, clip_actions=load_rl_cfg(cfg.jump_task_id).clip_actions
  )

  # Handed over as-is rather than wrapped in a lambda: the viewer's reset button calls
  # `policy.reset`, and a wrapper hides it, leaving the composition committed to
  # whatever skill it was running before the reset.
  if cfg.viewer == "viser":
    from mjlab.viewer import ViserPlayViewer

    ViserPlayViewer(viewer_env, policy, info_provider=policy.active_label).run()
  else:
    from mjlab.viewer import NativeMujocoViewer

    NativeMujocoViewer(viewer_env, policy).run()

  viewer_env.close()


if __name__ == "__main__":
  run_demo(tyro.cli(DemoConfig, config=mjlab.TYRO_FLAGS))
