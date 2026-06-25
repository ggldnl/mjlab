"""The differential-drive bridging experiment: robot + controller + bridge in the maze.

* the robot (DiffDrive) is sensed each tick for its reduced state
  [x, y, theta, v, omega] and driven by wheel torques; bounded torque means residual
  speed must be shed through the wheels, which is what makes the bridging problem real;
* the skills are the per-corridor cruisers (skills.corridor_skills), each with its
  own speed (corridor_speeds) so a switch must change speed too;
* the controller (CorridorController) decides when/what to switch, in this case
  positionally, at each junction, and runs the bridge across.

The default bridge is InstantBridge the do-nothing baseline: it hands straight to the next
corridor's skill the instant the robot reaches a junction. Because the robot arrives moving
along the old axis and the new (narrow) skill never steers, it keeps going straight
into the wall, leading to a crash. That failure is the whole motivation for a
real bridge: doing nothing does not work.

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.experiment
    uv run python -m mjlab.tasks.skills.experiments.diffdrive.experiment --backend mpl
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from mjlab.tasks.skills import play
from mjlab.tasks.skills.experiments.diffdrive.bridge import InstantBridge
from mjlab.tasks.skills.experiments.diffdrive.controller import CorridorController
from mjlab.tasks.skills.experiments.diffdrive.gridworld import (
  HORIZONTAL,
  GridWorld,
  build_spec,
  render,
)
from mjlab.tasks.skills.experiments.diffdrive.robot import STATE, DiffDrive
from mjlab.tasks.skills.experiments.diffdrive.skills import corridor_skills
from mjlab.tasks.skills.interfaces import Bridge

X, Y, THETA, V, OMEGA = STATE


@dataclass(frozen=True)
class Config:
  """Every tunable constant for the experiment, harvesting, training, and deployment.

  Bundled into one object so callers import a single name (CONFIG) instead of a long list.
  Fields are grouped by what reads them.
  """

  # Control rate. One control tick = timestep * decimation (0.02 s, 50 Hz). Harvesting and
  # the training env both step at this rate so their states line up.
  timestep: float = 0.005
  decimation: int = 4

  # Rollout harvesting. window_seconds sets the length of both windows; seconds, not a raw
  # tick count, keep the horizon meaningful independent of the control rate. The dataset
  # stores couples_per_junction (skill1 end, skill2 start) trajectory couples per junction.
  window_seconds: float = 1.0
  couples_per_junction: int = 100
  representative: str = "medoid"  # reduce skill2's tube to one line for inference

  # Closeness weights used by the rollout medoid when picking a representative tube.
  w_pos: float = 1.0
  w_head: float = 0.3
  w_speed: float = 0.3

  # Twist action mapping: a raw policy 2-vector (~[-1, 1]) becomes a clamped body twist.
  action_dim: int = 2
  v_offset: float = 1.0
  v_scale: float = 1.5
  omega_scale: float = 4.0
  v_min: float = -0.5
  v_max: float = 2.5
  omega_max: float = 4.0

  # Wheel-velocity servo (matches DiffDrive): torque = kv * (target - actual), clamped.
  kv: float = 3.0
  torque_limit: float = 0.6

  # Bridge training. Each episode commits to one fixed merge frame on skill2's window.
  # The executor (the policy) first drives to that frame (phase one), then follows the rest
  # of skill2 as a reference that advances once the robot is within track_tol (phase two).
  # history_len recent (v, omega) pairs feed the policy so it knows what it was just doing.
  track_tol: float = 1.0  # normalized distance at which the reference advances
  merge_tol: float = (
    1.0  # normalized distance counting as having reached the merge frame
  )
  history_len: int = 5  # recent (v, omega) pairs in the observation
  track_weight: float = 1.0  # reward for closeness to the advancing reference
  effort_weight: float = 0.01  # penalty weight on the squared action (energy)

  # Selector. A separate small net picks the merge frame from the two windows. It stays
  # dormant during a warmup phase (the executor first learns to reach any commanded frame),
  # then starts choosing and learning. warmup_steps counts control steps; warmup_target is
  # the rule that picks the frame meanwhile: random, last, mid, first, or a 0..1 fraction.
  warmup_steps: int = 6000
  warmup_target: str = "random"
  selector_hidden: tuple[int, ...] = (128, 128)
  selector_lr: float = 1.0e-3
  selector_entropy: float = 0.01  # keeps the selector exploring early on
  selector_mode: str = "B"  # B: soonest clean join. A: copy the executor's own return.

  # Option B selector reward (soonest clean join). A choice scores well when the executor
  # reaches the chosen frame quickly (cost), lands on it cleanly (mismatch), and then
  # follows the rest of skill2 (track). Picking a frame the executor never reaches is
  # penalized. Weights are tunable.
  sel_reach: float = 1.0  # bonus for reaching the chosen frame at all
  sel_cost: float = 1.0  # penalty on how long reaching took (favors soonest)
  sel_mismatch: float = 1.0  # penalty on the state mismatch when joining (favors clean)
  sel_track: float = 1.0  # bonus for then following the rest of skill2
  sel_unreachable: float = 1.0  # penalty when the chosen frame was never reached

  @property
  def control_dt(self) -> float:
    """Seconds per control tick."""
    return self.timestep * self.decimation

  @property
  def window_steps(self) -> int:
    """Control ticks per saved window."""
    return round(self.window_seconds / self.control_dt)


CONFIG = Config()


def make_bridge(
  name: str,
  *,
  checkpoint: str | None,
  world: GridWorld,
  speeds: dict[int, float],
) -> Bridge:
  """Build the bridge selected from the CLI.

  instant is the do-nothing baseline. learned loads the trained policy (the ONNX
  file exported during training) and needs a checkpoint path.
  """
  if name == "learned":
    if checkpoint is None:
      raise ValueError("--checkpoint is required for the learned bridge.")
    from mjlab.tasks.skills.experiments.diffdrive.bridge.policy import LearnedBridge

    return LearnedBridge(checkpoint, world, speeds)
  return InstantBridge()


def corridor_speeds(
  world: GridWorld, *, slow: float = 1.3, fast: float = 2.2
) -> dict[int, float]:
  """A distinct cruise speed per corridor, alternating fast/slow down the
  numbered sequence. Adjacent corridors then always differ, so a switch must change the
  robot's speed as well as its heading, extra work a real bridge has to do, and one
  more thing the do-nothing handover gets wrong.
  """
  return {
    cid: (fast if i % 2 == 0 else slow) for i, cid in enumerate(sorted(world.corridors))
  }


def start_state(world: GridWorld, *, speed: float = 1.0) -> np.ndarray:
  """Robot at the first corridor's start cell, aligned with its axis, already cruising."""
  first = min(world.corridors)
  corr = world.corridor(first)
  dx, dy = world.travel_directions()[first]
  heading = math.atan2(dy, dx)
  lo, hi = world.extent(corr)  # world span along the axis
  sign = dx + dy  # +1 or -1 (one component is 0)
  axis = (lo if sign > 0 else hi) + sign * world.half_width  # first cell's center
  mid = world.centerline(corr)  # lateral coordinate
  x, y = (axis, mid) if corr.orientation == HORIZONTAL else (mid, axis)
  return np.array([x, y, heading, speed, 0.0])


@dataclass
class Experiment:
  """Drives the diff-drive through the maze under the controller + bridge."""

  world: GridWorld
  robot: DiffDrive
  controller: CorridorController
  start: np.ndarray
  state: np.ndarray = field(init=False)
  crashed: bool = field(init=False, default=False)

  def __post_init__(self) -> None:
    self.state = self.start.copy()

  def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self.controller.reset()
    self.robot.reset(model, data, self.start)
    self.state = self.start.copy()
    self.crashed = False

  def policy(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    self.state = self.robot.sense(model, data)
    if not self.crashed and not self.world.is_free(
      float(self.state[X]), float(self.state[Y])
    ):
      self.crashed = True
    if self.crashed:
      return self.robot.actuate(model, data, np.zeros(2))  # brake at the wall
    command = self.controller.step(self.state)
    return self.robot.actuate(model, data, command)


def build_model(world: GridWorld, robot: DiffDrive) -> mujoco.MjModel:
  """Compile the world + robot into one model, ready for stepping.

  The world spec (floor, tiles, visual-only walls) plus the robot attached as a real
  body. We bump the timestep down and use an implicit integrator for stable wheel
  contact, and keep gravity (the robot must press on the floor). ``robot.bind`` caches
  its indices afterwards.
  """
  spec = build_spec(world)
  robot.attach_to(spec)
  spec.option.timestep = CONFIG.timestep
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model = spec.compile()
  robot.bind(model)
  return model


def run_episode(
  experiment: Experiment,
  model: mujoco.MjModel,
  *,
  decimation: int = 4,
  max_steps: int = 2000,
) -> np.ndarray:
  """Step physics headlessly until the robot crashes or `max_steps` elapse.

  Returns the (N, 5) reduced-state track (for the matplotlib backend).
  """
  data = mujoco.MjData(model)
  experiment.reset(model, data)
  mujoco.mj_forward(model, data)
  traj = [experiment.state.copy()]
  for _ in range(max_steps):
    ctrl = experiment.policy(model, data)
    data.ctrl[:] = ctrl
    for _ in range(decimation):
      mujoco.mj_step(model, data)
    traj.append(experiment.robot.sense(model, data).copy())
    if experiment.crashed:
      break
  return np.asarray(traj)


# Status overlay for the viewer.


def _hue(text: str, color: str) -> str:
  return f'<span style="color:{color};">{text}</span>'


def status_text(experiment: Experiment, speeds: Mapping[int, float]) -> dict[str, str]:
  """Status rows for the viewer's Info box (label -> value, values may be HTML).

  Separates *where the robot is* (corridor, by position) from *what is driving it* (the
  active skill, or the bridge while a switch is open), and flags the bridge so it is
  visible the moment it acts.
  """
  s = experiment.state
  controller = experiment.controller
  here = int(experiment.world.corridor_at(float(s[X]), float(s[Y])))
  active = controller.current
  return {
    "corridor (position)": str(here) if here else _hue("wall", "#e74c3c"),
    "active skill": f"corridor {active}",
    "bridge": _hue("ACTIVE", "#f1c40f") if controller.switching else "idle",
    "target speed": f"{speeds[active]:.2f} m/s",
    "current speed": f"{float(s[V]):.2f} m/s",
    "yaw rate": f"{float(s[OMEGA]):+.2f} rad/s",
    "position": f"({float(s[X]):.2f}, {float(s[Y]):.2f})",
    "heading": f"{math.degrees(float(s[THETA])):.0f}°",
    "outcome": _hue("crashed", "#e74c3c") if experiment.crashed else "running",
  }


def main() -> None:
  import tyro

  @dataclass
  class Args:
    bridge: Literal["instant", "learned"] = (
      "instant"  # which bridge to use (default: the do-nothing baseline)
    )
    idle: Literal["zero", "coast"] = (
      "coast"  # outside a skill's initiation set: "zero" (stop) or "coast" (keep motion)
    )
    backend: Literal["viser", "mpl"] = "viser"
    cell: float = 1.0
    # Per-corridor cruise speeds (alternating). The robot droops below these under
    # bounded torque, but fast corridors still carry more residual speed into a junction
    # than slow ones
    slow: float = 0.5
    fast: float = 1.5
    decimation: int = 4  # physics steps per control tick (control dt = 0.02 s)
    max_steps: int = 2000  # offline episode cap
    checkpoint: str | None = None  # ONNX policy path, required by the learned bridge

  args = tyro.cli(Args)
  world = GridWorld(cell=args.cell)
  robot = DiffDrive()
  speeds = corridor_speeds(world, slow=args.slow, fast=args.fast)
  bridge = make_bridge(
    args.bridge, checkpoint=args.checkpoint, world=world, speeds=speeds
  )
  skills = corridor_skills(world, speeds, idle=args.idle)
  controller = CorridorController(world, skills, bridge)
  experiment = Experiment(
    world=world,
    robot=robot,
    controller=controller,
    start=start_state(world, speed=0.0),  # rest at the first cell; the skill ramps up
  )
  model = build_model(world, robot)

  if args.backend == "mpl":
    import matplotlib.pyplot as plt

    traj = run_episode(
      experiment, model, decimation=args.decimation, max_steps=args.max_steps
    )
    ax = render(world)
    ax.plot(traj[:, X], traj[:, Y], "w-", lw=2.0, alpha=0.9)
    ax.plot(traj[0, X], traj[0, Y], "wo", ms=6)
    if experiment.crashed:
      ax.plot(traj[-1, X], traj[-1, Y], "rx", ms=14, mew=3)
    outcome = "CRASH" if experiment.crashed else "no crash"
    ax.set_title(f"{args.bridge} bridge -> {outcome}")
    plt.show()
  else:
    play.run(
      model,
      experiment.policy,
      decimation=args.decimation,
      on_reset=experiment.reset,
      status=lambda: status_text(experiment, speeds),
    )


if __name__ == "__main__":
  main()
