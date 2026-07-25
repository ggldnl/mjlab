"""Run the table tennis primitives (catch, balance, toss, hit) through one bridging
architecture.

Each primitive is trained separately and only ever sees its own reset, so at a switch
the next primitive inherits whatever mid-motion state the previous one left behind:
under architecture 0 (the no-bridge baseline) that is a state it never trained on.
That failure is what a bridge has to remove, and this demo is the harness to watch it.
There is no fixed sequence between them, so a simple `CycleController` cycles the pool
on a timer.

Look at where a task starts its ball, with its initial velocity drawn as an arrow
(needs no checkpoints):

    uv run python -m mjlab.tasks.skills.experiments.table_tennis.demo --debug-init catch

Look at the scene alone, with no skills and no env:

    uv run python -m mjlab.tasks.skills.experiments.table_tennis.demo --debug-scene

Test one primitive at a time, picking between them interactively:

    uv run python -m mjlab.tasks.skills.experiments.table_tennis.demo --debug

Watch the composition (needs trained checkpoints):

    uv run python -m mjlab.tasks.skills.experiments.table_tennis.demo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.skills.architectures import ARCHITECTURES
from mjlab.tasks.skills.experiments.table_tennis.controller import CycleController
from mjlab.tasks.skills.experiments.table_tennis.mdp import COMMAND_NAME, ball_pos
from mjlab.tasks.skills.meta import ComposedPolicy
from mjlab.tasks.skills.skill import PolicySkill, SkillPool
from mjlab.tasks.skills.utils import retrieve_latest_checkpoint

# The registered task each primitive is trained from.
PRIMITIVE_TASKS = {
  "catch": "Mjlab-TableTennis-Catch",
  "balance": "Mjlab-TableTennis-Balance",
  "toss": "Mjlab-TableTennis-Toss",
  "hit": "Mjlab-TableTennis-Hit",
}


@dataclass(frozen=True)
class DemoConfig:
  # Skip everything and just look at the scene (arm, racket, ball). Needs no
  # checkpoints, which is the point: it works before any skill has been trained
  debug_scene: bool = False

  # Look at where one task starts its ball: the arm holds the ready stance while the
  # env is reset over and over, with the ball's initial velocity drawn as an arrow and
  # the goal as a sphere. Needs no checkpoints either. One of catch/balance/toss/hit
  debug_init: str | None = None

  # For --debug-init: how many steps to let each spawn play out before resampling
  init_hold_steps: int = 60

  # Skip the architecture entirely and open a viewer that runs one skill at a time,
  # letting you pick between them to test each in isolation. Needs an interactive
  # viewer (viser or native), not headless
  debug: bool = False

  # Which architecture to run (see architectures/__init__.py). Defaults to 0, the
  # no-bridge direct hand-off baseline
  architecture: int = 0

  # Viewer
  viewer: Literal["viser", "native", "headless"] = "viser"

  # Explicit checkpoints, keyed by primitive name (e.g. --checkpoints.toss PATH).
  # Anything omitted falls back to the latest checkpoint for that primitive's task
  checkpoints: dict[str, str] | None = None

  # Controller parameter: how long CycleController holds each primitive before
  # switching to the next one
  steps_per_skill: int = 150

  num_envs: int = 1
  device: str | None = None

  # If headless, run for this many steps
  steps: int = 600


def _build_pool(cfg: DemoConfig, env: ManagerBasedRlEnv, device: str) -> SkillPool:
  """Load all four trained primitives."""
  overrides = cfg.checkpoints or {}
  skills = []
  for name, task_id in PRIMITIVE_TASKS.items():
    path = overrides.get(name) or retrieve_latest_checkpoint(task_id)
    skills.append(PolicySkill(name, task_id, path, env, device))
  return SkillPool(skills)


def _run_headless(env: ManagerBasedRlEnv, policy: ComposedPolicy, steps: int) -> None:
  """Step the composition and report how near the ball got to its target.

  The commanded target is what every skill is rewarded against, so the closest the
  ball ever comes to it is a blunt but honest score for the whole run. A bridge that
  removes the hand-off damage should improve it against architecture 0.
  """
  obs, _ = env.reset()
  command = env.command_manager.get_command(COMMAND_NAME)
  assert command is not None
  closest = torch.full((env.num_envs,), float("inf"), device=env.device)

  for _ in range(steps):
    obs, _, _, _, _ = env.step(policy(obs))
    command = env.command_manager.get_command(COMMAND_NAME)
    assert command is not None
    closest = torch.minimum(closest, torch.norm(ball_pos(env) - command, dim=-1))

  for i in range(env.num_envs):
    print(f"env {i}: closest_approach_to_target={closest[i].item():.3f} m")


class _SkillSelector:
  """Drives every env with one skill at a time, picked interactively.

  This is the debug path: no controller, no bridge, just the pool. It runs the
  currently selected skill on the whole batch so a single expert can be watched in
  isolation.
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
    env, clip_actions=load_rl_cfg(PRIMITIVE_TASKS["toss"]).clip_actions
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


def _run_scene_only(cfg: DemoConfig) -> None:
  """Compile the scene and open a raw MuJoCo viewer on it, with no policies.

  Deliberately does not build a `ManagerBasedRlEnv`: this path has to work before any
  skill exists, so it goes straight from the scene cfg to a model.
  """
  import mujoco
  import mujoco.viewer as mj_viewer

  from mjlab.scene import Scene
  from mjlab.tasks.skills.experiments.table_tennis.table_tennis_env_cfg import (
    BALL_SPAWN_POS,
    READY_JOINT_POS,
    table_tennis_scene_cfg,
  )

  scene = Scene(table_tennis_scene_cfg(num_envs=1), device="cpu")
  model = scene.compile()

  # The KUKA XML's own <option integrator="implicitfast"/> does not survive
  # MjSpec.attach(), and its actuators (kp=2000, kd=200) are stiff enough that the
  # default Euler integrator blows the joints up within milliseconds. A real env gets
  # this from MujocoCfg; this raw viewer has to set it by hand.
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

  # `EntityCfg.init_state` is applied by an env at reset, not baked into the compiled
  # model, so mirror the ready stance here or the arm shows collapsed in its zero pose.
  data = mujoco.MjData(model)
  for joint_name, angle in READY_JOINT_POS.items():
    joint_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{joint_name}"
    )
    data.qpos[model.jnt_qposadr[joint_id]] = angle
  ball_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "ball/floating_base_joint"
  )
  adr = model.jnt_qposadr[ball_id]
  data.qpos[adr : adr + 3] = BALL_SPAWN_POS
  data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
  mujoco.mj_forward(model, data)

  # Command the servos to the stance, or the arm sags the moment stepping starts.
  data.ctrl[:] = data.qpos[: model.nu]
  mj_viewer.launch(model, data)


class _HoldAndResample:
  """Holds the arm in its ready stance and re-rolls the episode every N steps.

  A zero action means "stay at the default joint targets", so the arm is a fixed
  backdrop and the only thing moving is the ball. Resetting on a timer turns a single
  spawn into a sample of the whole spawn distribution, which is what you actually want
  to eyeball. The reset is issued from inside the policy because the viewer owns the
  stepping loop; the observation it hands back is stale for one step, which does not
  matter when the action is constant.
  """

  def __init__(self, env: ManagerBasedRlEnv, hold_steps: int) -> None:
    self.env = env
    self.hold_steps = hold_steps
    self.action_dim = env.action_manager.total_action_dim
    self._t = 0

  def __call__(self, obs) -> torch.Tensor:
    del obs
    self._t += 1
    if self._t % self.hold_steps == 0:
      self.env.reset()
    return torch.zeros(self.env.num_envs, self.action_dim, device=self.env.device)


def _run_debug_init(cfg: DemoConfig) -> None:
  """Show one task's initial state: ball position, ball velocity, and goal."""
  from mjlab.tasks.skills.experiments.table_tennis.table_tennis_env_cfg import (
    TASK_ENV_CFGS,
  )

  task = cfg.debug_init
  assert task is not None
  if task not in TASK_ENV_CFGS:
    raise ValueError(f"Unknown task '{task}'; choose from {sorted(TASK_ENV_CFGS)}.")
  if cfg.viewer == "headless":
    raise ValueError("--debug-init needs an interactive viewer (viser or native).")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  # Not the play cfg: play disables episode limits, and here the spawn distribution is
  # exactly what we came to look at.
  env_cfg = TASK_ENV_CFGS[task](play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env.reset()

  policy = _HoldAndResample(env, cfg.init_hold_steps)
  viewer_env = RslRlVecEnvWrapper(env, clip_actions=None)
  print(
    f"[debug-init] '{task}': blue arrow = the ball's initial velocity, "
    f"green sphere = where the ball should end up. Respawning every "
    f"{cfg.init_hold_steps} steps."
  )

  if cfg.viewer == "native":
    from mjlab.viewer import NativeMujocoViewer

    NativeMujocoViewer(viewer_env, policy).run()
  else:
    from mjlab.viewer import ViserPlayViewer

    ViserPlayViewer(viewer_env, policy).run()
  viewer_env.close()


def run_demo(cfg: DemoConfig) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  if cfg.debug_scene:
    _run_scene_only(cfg)
    return

  if cfg.debug_init is not None:
    _run_debug_init(cfg)
    return

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  # Any primitive's env is a fine arena; toss's is as good as any.
  env_cfg = load_env_cfg(PRIMITIVE_TASKS["toss"], play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  pool = _build_pool(cfg, env, device)

  # Debug: bypass the controller and the architecture, and test each skill on its own.
  if cfg.debug:
    _run_debug(cfg, env, pool)
    env.close()
    return

  if cfg.architecture not in ARCHITECTURES:
    raise ValueError(
      f"Unknown architecture {cfg.architecture}; registered: {sorted(ARCHITECTURES)}."
    )

  controller = CycleController(pool, steps_per_skill=cfg.steps_per_skill)
  meta = ARCHITECTURES[cfg.architecture](env, pool)
  policy = ComposedPolicy(env, controller, meta)

  if cfg.viewer == "headless":
    _run_headless(env, policy, cfg.steps)
    env.close()
    return

  viewer_env = RslRlVecEnvWrapper(
    env, clip_actions=load_rl_cfg(PRIMITIVE_TASKS["toss"]).clip_actions
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
