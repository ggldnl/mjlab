"""Run the cartpole demonstration: spin_up then balance, composed through one
bridging architecture.

A scripted controller runs spin_up for a while and then switches once to balance.
Under architecture 0 (the no-bridge baseline, a direct hand-off), balance takes
over at a fixed instant whether the pole is actually up and slow: it is handed
a pole still swinging through the top, its linear law drives the cart the wrong
way, and the pole falls. That fall is the failure a real bridge has to remove:
it should hold spin_up until the pole is up and slow enough for the balancer's
basin of attraction, then hand over.

Analytical experts are recommended over RL.

Watch the baseline live (analytical experts, architecture 0, no training needed):

    uv run python -m mjlab.tasks.skills.experiments.cartpole.demo

Watch a trained architecture instead (train it first with this experiment's
train.py). Its state is restored from a run directory; --checkpoint points at one,
and if omitted the latest trained run for that architecture is used:

    uv run python -m mjlab.tasks.skills.experiments.cartpole.demo --architecture 1
    uv run python -m mjlab.tasks.skills.experiments.cartpole.demo \\
        --architecture 1 --checkpoint logs/skills/cartpole/arch_1/<run>

Use trained RL checkpoints instead of the analytical experts:

    uv run python -m mjlab.tasks.skills.experiments.cartpole.demo --no-analytical

Measure how well the pole is held after the hand-off, headless:

    uv run python -m mjlab.tasks.skills.experiments.cartpole.demo --viewer headless --steps 1500

Important, manually control which skill is currently active:

    uv run python -m mjlab.tasks.skills.experiments.cartpole.demo --debug
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
from mjlab.tasks.skills.experiments.cartpole import (
  BALANCE_TASK_ID,
  BRIDGE_VIEW,
  EXPERIMENT_NAME,
  SPINUP_TASK_ID,
  build_pool,
)
from mjlab.tasks.skills.experiments.cartpole.controller import CartpoleController
from mjlab.tasks.skills.meta import ComposedPolicy
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.utils import retrieve_latest_architecture_checkpoint

# Observation indices, mirroring the cartpole env's actor obs order (see
# dynamics.py): pole cosine is upright at +1, pole velocity is the hinge rate.
_POLE_COS = 1
_POLE_VEL = 4


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

  # The two skill tasks. They share one observation/action space; the arena is
  # built from the swingup task so the pole starts hanging and spin_up has a job
  spinup_task_id: str = SPINUP_TASK_ID
  balance_task_id: str = BALANCE_TASK_ID

  # Checkpoints for the non-analytical skills. None picks the latest trained
  # checkpoint for the corresponding task
  spinup_checkpoint: str | None = None
  balance_checkpoint: str | None = None

  # Controller parameter: after how many steps spin_up hands over to balance.
  # Sized so the naive hand-off fires while the pole is still swinging
  swingup_steps: int = 200

  position_based: bool = True

  num_envs: int = 1
  device: str | None = None

  # Only used with --viewer headless: run this many steps, then report how
  # upright and how slow the pole ended up (i.e. whether balance kept it)
  steps: int = 1500


def _build_pool(cfg: DemoConfig, env: ManagerBasedRlEnv, device: str) -> SkillPool:
  return build_pool(
    env,
    device,
    analytical=cfg.analytical,
    spinup_task_id=cfg.spinup_task_id,
    balance_task_id=cfg.balance_task_id,
    spinup_checkpoint=cfg.spinup_checkpoint,
    balance_checkpoint=cfg.balance_checkpoint,
  )


def _run_headless(env: ManagerBasedRlEnv, policy: ComposedPolicy, steps: int) -> None:
  """Step the composition and report how well the pole is held at the end.

  A good hand-off leaves the pole upright (angle ~0) and slow; a naive one drops
  it, which shows up as a large final angle and, over the last stretch, little
  time spent upright.
  """
  window = min(steps, 200)
  obs, _ = env.reset()
  upright_in_window = torch.zeros(env.num_envs, device=env.device)
  for step in range(steps):
    obs, _, _, _, _ = env.step(policy(obs))
    actor = obs["actor"]
    assert isinstance(actor, torch.Tensor)
    if step >= steps - window:
      upright_in_window += (actor[:, _POLE_COS] > 0.9).float()
  actor = obs["actor"]
  assert isinstance(actor, torch.Tensor)
  final_angle = torch.acos(actor[:, _POLE_COS].clamp(-1.0, 1.0))
  final_speed = actor[:, _POLE_VEL].abs()
  upright_frac = upright_in_window / window
  for i in range(env.num_envs):
    print(
      f"env {i}: final_angle={math.degrees(final_angle[i].item()):.1f} deg  "
      f"final_pole_speed={final_speed[i].item():.2f} rad/s  "
      f"upright_fraction_last_{window}={upright_frac[i].item():.2f}"
    )


class _SkillSelector:
  """Drives every env with one skill at a time, picked interactively.

  This is the debug path: no controller, no bridge, just the pool. It runs the
  currently selected skill on the whole batch so a single expert can be watched
  in isolation.
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
    env, clip_actions=load_rl_cfg(cfg.spinup_task_id).clip_actions
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
  env_cfg = load_env_cfg(cfg.spinup_task_id, play=True)
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

  controller = CartpoleController(
    pool, swingup_steps=cfg.swingup_steps, position_based=cfg.position_based
  )
  meta = ARCHITECTURES[cfg.architecture](env, pool, BRIDGE_VIEW.resolve(env))

  # Every architecture but the baseline carries trained state that must be restored.
  # The checkpoint is a run directory from train.py; default to the latest one.
  if cfg.architecture != 0:
    run_dir = cfg.checkpoint or retrieve_latest_architecture_checkpoint(
      EXPERIMENT_NAME, cfg.architecture
    )
    meta.load(Path(run_dir))

  policy = ComposedPolicy(env, controller, meta)

  if cfg.viewer == "headless":
    _run_headless(env, policy, cfg.steps)
    env.close()
    return

  viewer_env = RslRlVecEnvWrapper(
    env, clip_actions=load_rl_cfg(cfg.spinup_task_id).clip_actions
  )

  # Handed over as-is rather than wrapped in a lambda: the viewer's reset button calls
  # `policy.reset`, and a wrapper hides it, leaving the composition committed to
  # whatever skill it was running before the reset.
  if cfg.viewer == "viser":
    from mjlab.viewer import ViserPlayViewer

    # Show the running skill (or bridge) for the displayed env in the info box.
    ViserPlayViewer(viewer_env, policy, info_provider=policy.active_label).run()
  else:
    from mjlab.viewer import NativeMujocoViewer

    NativeMujocoViewer(viewer_env, policy).run()

  viewer_env.close()


if __name__ == "__main__":
  run_demo(tyro.cli(DemoConfig))
