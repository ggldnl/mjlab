"""Per-corridor skills with an explicit initiation set.

A skill follows one straight corridor with a heading-hold policy: it commands a forward
speed and a yaw rate that rotates the robot back toward the corridor direction (like a
cartpole stabilizing to upright). It never corrects lateral offset and cannot make the
90 degree junction turn.

A skill is valid only from its initiation set: centered on the corridor, aligned with the
travel direction, at a moderate speed. Started there it sails to the exit. Started anywhere
else (the wrong heading and cross-axis momentum a junction leaves behind) it cannot recover.
Carrying the robot from a junction into the next skill's initiation set is the bridge's job,
not the skill's.

    skills = corridor_skills(world, corridor_speeds(world))
    controller = CorridorController(world, skills, bridge)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import OMEGA, THETA, V, X, Y
from mjlab.tasks.skills.interfaces import Command, State

ZERO, COAST = "zero", "coast"


def _wrap(angle: float) -> float:
  """Wrap an angle to (-pi, pi]."""
  return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class CorridorSkill:
  """One corridor's heading-hold policy together with its initiation set.

  The skill starts only from its initiation set: on the centerline, aligned with the
  travel direction, at a moderate speed. Membership is a fact about the dynamic state,
  not the cell: a state that is centered and aligned is in, a state with the wrong
  heading or cross-axis momentum (what a junction arrival looks like) is out, even at the
  same place. Position along the corridor does not enter the test, because it does not
  enter the policy: from a centered, aligned, moderate-speed state, driving straight holds
  the corridor all the way to the exit no matter how far in the robot starts.
  """

  world: GridWorld
  cid: int
  speed: float
  idle: str = ZERO  # outside the initiation set: ZERO (stop) or COAST (keep motion)
  kp: float = 0.8  # heading-hold positional gain
  kd: float = 2.0  # heading-hold derivative gain
  d_tol: float = 0.12  # max |lateral offset| from the centerline in the initiation set
  phi_tol: float = math.radians(15.0)  # max heading error in the initiation set
  speed_band: tuple[float, float] = (
    0.0,
    1.8,
  )  # start speed (rest allowed, too-fast not)
  heading: float = field(init=False)  # travel direction as an angle
  entry: tuple[int, int] = field(init=False)  # the geometric-start cell
  _active: bool | None = field(init=False, default=None, repr=False)

  def __post_init__(self) -> None:
    dx, dy = self.world.travel_directions()[self.cid]
    self.heading = math.atan2(dy, dx)
    self.entry = self.world.corridor(self.cid).entry_cell((dx, dy))
    self._active = None

  @property
  def active(self) -> bool:
    """Whether the skill has started, i.e. it began from inside its initiation set."""
    return bool(self._active)

  def reset(self) -> None:
    """Re-arm: the next call re-checks the initiation set to decide whether to start."""
    self._active = None

  def heading_error(self, theta: float) -> float:
    """Signed heading error to the corridor's travel direction."""
    return _wrap(theta - self.heading)

  def __call__(self, state: State) -> Command:
    if self._active is None:  # first call after a reset: decide whether to start
      self._active = self.initiation_set(state)
    if not self._active:  # began outside the initiation set: the skill does not drive
      if self.idle == COAST:
        return np.array([float(state[V]), 0.0])  # keep current heading and speed
      return np.zeros(2)  # brake to a stop
    phi = self.heading_error(float(state[THETA]))
    phi_dot = float(state[OMEGA])  # heading fixed, so phi_dot = theta_dot = omega
    return np.array([self.speed, -self.kp * phi - self.kd * phi_dot])

  def initiation_set(self, state: State) -> bool:
    """Whether the skill may be started from state: centered on the centerline, aligned
    with the travel direction, and at a moderate speed. Position along the corridor does
    not matter.
    """
    x, y = float(state[X]), float(state[Y])
    if abs(float(self.world.offset(self.cid, x, y))) > self.d_tol:
      return False
    if abs(self.heading_error(float(state[THETA]))) > self.phi_tol:
      return False
    lo, hi = self.speed_band
    return lo <= float(state[V]) <= hi


def corridor_skills(
  world: GridWorld, speeds: Mapping[int, float], idle: str = ZERO
) -> dict[int, CorridorSkill]:
  """One CorridorSkill per corridor, each at its target speed and idle behavior."""
  return {
    cid: CorridorSkill(world, cid, speeds[cid], idle=idle) for cid in world.corridors
  }


def main() -> None:
  """Visualize one skill on the grid world, choosing the skill and spawn in the viewer.

  Side-panel controls pick the corridor (which skill runs), the spawn cell (row, col), the
  heading offset from the corridor direction (0 = aligned), and the start speed. Changing
  any of them respawns the robot. The Info box shows whether the skill is RUNNING (started
  inside its initiation set) or IDLE (started anywhere else). The CLI flags just set the
  initial selection.

      uv run python -m mjlab.tasks.skills.experiments.diffdrive.skills
      uv run python -m mjlab.tasks.skills.experiments.diffdrive.skills --corridor 3
  """
  from dataclasses import dataclass as _dataclass

  import tyro

  from mjlab.tasks.skills import play
  from mjlab.tasks.skills.experiments.diffdrive.experiment import (
    build_model,
    corridor_speeds,
  )
  from mjlab.tasks.skills.experiments.diffdrive.robot import DiffDrive

  @_dataclass
  class Args:
    corridor: int = 1  # initial corridor whose skill to run
    idle: str = COAST  # initial idle behavior: "zero" (stop) or "coast" (keep motion)
    cell: tuple[int, int] | None = (
      None  # initial spawn (row, col); default = entry cell
    )
    speed: float = 1.0  # initial forward speed (m/s)
    heading_deg: float = 0.0  # initial heading offset from the corridor direction (deg)
    world_cell: float = 1.0  # metres per grid cell

  args = tyro.cli(Args)
  world = GridWorld(cell=args.world_cell)
  robot = DiffDrive()
  speeds = corridor_speeds(world)
  skills = corridor_skills(world, speeds)

  @_dataclass
  class Sel:
    cid: int
    idle: str
    row: int
    col: int
    heading_deg: float
    speed: float

  init_cell = args.cell if args.cell is not None else skills[args.corridor].entry
  sel = Sel(
    cid=args.corridor,
    idle=args.idle,
    row=int(init_cell[0]),
    col=int(init_cell[1]),
    heading_deg=float(args.heading_deg),
    speed=float(args.speed),
  )
  state_box = {"s": np.zeros(5)}

  def current_skill() -> CorridorSkill:
    return skills[sel.cid]

  def current_start() -> np.ndarray:
    skill = current_skill()
    x, y = (float(v) for v in world.cell_center(sel.row, sel.col))
    heading = skill.heading + math.radians(sel.heading_deg)
    return np.array([x, y, heading, sel.speed, 0.0])

  model = build_model(world, robot)

  def policy(model, data) -> np.ndarray:
    s = robot.sense(model, data)
    state_box["s"] = s
    return robot.actuate(model, data, current_skill()(s))

  def on_reset(model, data) -> None:
    skill = current_skill()
    skill.idle = sel.idle  # apply the selected idle behavior to this spawn
    skill.reset()  # re-arm: re-decide whether this spawn is a valid start
    start = current_start()
    state_box["s"] = start
    robot.reset(model, data, start)

  def status() -> dict[str, str]:
    s = state_box["s"]
    skill = current_skill()
    here = int(world.corridor_at(float(s[X]), float(s[Y])))
    return {
      "corridor (position)": str(here) if here else "wall",
      "skill": f"corridor {sel.cid}",
      "entry cell (initiation set)": str(skill.entry),
      "spawn cell": f"({sel.row}, {sel.col})",
      "state": "RUNNING" if skill.active else f"IDLE -> {sel.idle}",
      "idle behavior": sel.idle,
      "speed": f"{float(s[V]):.2f} m/s",
      "heading error": f"{math.degrees(skill.heading_error(float(s[THETA]))):+.0f} deg",
      "position": f"({float(s[X]):.2f}, {float(s[Y]):.2f})",
    }

  def gui(server, reset) -> None:
    options = [str(cid) for cid in sorted(world.corridors)]
    with server.gui.add_folder("Skill / spawn"):
      cid_dd = server.gui.add_dropdown("Corridor", options, initial_value=str(sel.cid))
      row_in = server.gui.add_number(
        "Spawn row", initial_value=sel.row, min=0, max=world.nrows - 1, step=1
      )
      col_in = server.gui.add_number(
        "Spawn col", initial_value=sel.col, min=0, max=world.ncols - 1, step=1
      )
      head_sl = server.gui.add_slider(
        "Heading offset (deg)", min=-180, max=180, step=5, initial_value=sel.heading_deg
      )
      speed_sl = server.gui.add_slider(
        "Start speed (m/s)", min=0.0, max=3.0, step=0.1, initial_value=sel.speed
      )
      snap_btn = server.gui.add_button("Snap cell to entry")

    def apply(_=None) -> None:
      sel.cid = int(cid_dd.value)
      sel.row = int(row_in.value)
      sel.col = int(col_in.value)
      sel.heading_deg = float(head_sl.value)
      sel.speed = float(speed_sl.value)
      reset()

    def snap_to_entry(_=None) -> None:
      entry = skills[int(cid_dd.value)].entry
      row_in.value, col_in.value = int(entry[0]), int(entry[1])
      apply()

    cid_dd.on_update(snap_to_entry)  # picking a corridor snaps the spawn to its entry
    snap_btn.on_click(snap_to_entry)
    for handle in (row_in, col_in, head_sl, speed_sl):
      handle.on_update(apply)

  play.run(model, policy, decimation=4, on_reset=on_reset, status=status, gui=gui)


if __name__ == "__main__":
  main()
