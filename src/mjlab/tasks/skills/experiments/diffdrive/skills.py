"""Per-corridor skills with an explicit initiation set.

A skill follows one straight corridor.
A skill is valid only from its initiation set: states at the entry cell, centered on
the corridor, aligned with the travel direction, at a moderate speed. Started there it
sails to the exit. Started anywhere else (the wrong heading and cross-axis momentum a
junction leaves behind) the non-steering policy drives straight into a wall. Carrying
the robot from a junction into the next skill's initiation set is the bridge's job, not
the skill's.

Two running modes:

  cruise  non-steering, command (v, 0). Never steers, so it holds a corridor only from
          an already aligned state. This is the literal reading of "a skill must not do
          the bridge's steering".
  hold    heading-hold, command (v, -kp * phi). Rotates back toward the corridor
          direction (like cartpole balance stabilizing to upright) but never corrects
          lateral offset and cannot make the 90 degree junction turn.

    skills = corridor_skills(world, corridor_speeds(world), mode="cruise")
    controller = CorridorController(world, skills, bridge)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from mjlab.tasks.skills.experiments.diffdrive.controller import (
  _adjacency_cell,
  travel_directions,
)
from mjlab.tasks.skills.experiments.diffdrive.gridworld import HORIZONTAL, GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import THETA, V, X, Y
from mjlab.tasks.skills.interfaces import Command, State

CRUISE, HOLD = "cruise", "hold"


def _wrap(angle: float) -> float:
  """Wrap an angle to (-pi, pi]."""
  return math.atan2(math.sin(angle), math.cos(angle))


def _entry_cell(world: GridWorld, cid: int) -> tuple[int, int]:
  """The corridor's geometric start: the cell a full traversal begins from.

  It is the corridor endpoint nearest the junction with the previous corridor, since a
  corridor is entered near its start and driven toward the next corridor at the far end.
  The first corridor has no predecessor, so it is the end opposite the exit.

  This is deliberately not where the robot arrives. The robot comes in at the junction
  corner, a different cell (and facing the wrong way), so it is never in the initiation
  set on arrival. Closing that gap is exactly the bridge's job.
  """
  if cid - 1 in world.corridors:
    cell = _adjacency_cell(world, cid, cid - 1)
    if cell is not None:
      corr = world.corridor(cid)
      d_start = abs(cell[0] - corr.start_cell[0]) + abs(cell[1] - corr.start_cell[1])
      d_end = abs(cell[0] - corr.end_cell[0]) + abs(cell[1] - corr.end_cell[1])
      return corr.start_cell if d_start < d_end else corr.end_cell
  corr = world.corridor(cid)
  dx, dy = travel_directions(world)[cid]
  if corr.orientation == HORIZONTAL:
    return corr.cells[0] if dx > 0 else corr.cells[-1]
  return corr.cells[-1] if dy > 0 else corr.cells[0]


@dataclass
class CorridorSkill:
  """One corridor's cruise policy together with its initiation set.

  TODO: update docstring, make sure disabling connection between initiation set and initial
    window is a good idea
  The skill starts only from its initiation set: the early window of the corridor (within
  `window` cells of the geometric start, measured along the travel direction), on the
  centerline, aligned, at a moderate speed. Membership is a fact about the dynamic state,
  not the exact cell: a state that is aligned and centered near the start is in, a state
  with the wrong heading or cross-axis momentum (what a junction arrival looks like) is
  out, even at the same place. The skill decides this once, on the first call after a
  reset, and latches it. Started inside the set it runs, and keeps running as it leaves
  the window. Started anywhere else it does nothing (zero twist), because the policy has
  no idea what to command from a state it was never meant to begin from. Getting the robot
  from where it arrives (the junction corner, wrong heading) into this set is the bridge's
  job, not the skill's.
  """

  world: GridWorld
  cid: int
  speed: float
  mode: str = CRUISE
  kp: float = 2.0  # heading-hold gain (hold mode only)
  # window: float = 1.5  # initiation-set length in cells, from the start along travel
  d_tol: float = 0.12  # max |lateral offset| from the centerline in the initiation set
  phi_tol: float = math.radians(15.0)  # max heading error in the initiation set
  speed_band: tuple[float, float] = (
    0.0,
    1.8,
  )  # start speed (rest allowed, too-fast not)
  heading: float = field(init=False)  # travel direction as an angle
  entry: tuple[int, int] = field(init=False)  # the geometric-start cell
  _entry_xy: tuple[float, float] = field(init=False, repr=False)  # its world center
  _active: bool | None = field(init=False, default=None, repr=False)

  def __post_init__(self) -> None:
    dx, dy = travel_directions(self.world)[self.cid]
    self.heading = math.atan2(dy, dx)
    self.entry = _entry_cell(self.world, self.cid)
    ex, ey = self.world.cell_center(*self.entry)
    self._entry_xy = (float(ex), float(ey))
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

  """
  def along_distance(self, x: float, y: float) -> float:
    # Distance from the geometric start along the travel direction, in metres.
    ex, ey = self._entry_xy
    return (x - ex) * math.cos(self.heading) + (y - ey) * math.sin(self.heading)
  """

  def __call__(self, state: State) -> Command:
    if self._active is None:  # first call after a reset: decide whether to start
      self._active = self.initiation_set(state)
    if not self._active:  # began outside the initiation set: the skill does not drive
      return np.zeros(2)
    if self.mode == HOLD:
      phi = self.heading_error(float(state[THETA]))
      return np.array([self.speed, -self.kp * phi])
    return np.array([self.speed, 0.0])

  def initiation_set(self, state: State) -> bool:
    """
    Whether the skill may be started from state: inside the early window of the
    corridor, centered, aligned, and at a moderate speed.

    Initiation set:
    - the along-corridor distance from the geometric start is within window (1.5 cells)
    + the lateral offset and heading error are within tolerance
    + speed is in band
    """

    x, y = float(state[X]), float(state[Y])
    """
    s = self.along_distance(x, y)
    if not 0.0 <= s <= self.window * self.world.cell:
      return False
    """
    if abs(float(self.world.offset(self.cid, x, y))) > self.d_tol:
      return False
    if abs(self.heading_error(float(state[THETA]))) > self.phi_tol:
      return False
    lo, hi = self.speed_band
    return lo <= float(state[V]) <= hi


def corridor_skills(
  world: GridWorld, speeds: Mapping[int, float], mode: str = CRUISE
) -> dict[int, CorridorSkill]:
  """One CorridorSkill per corridor, each at its target speed and the given mode."""
  return {
    cid: CorridorSkill(world, cid, speeds[cid], mode=mode) for cid in world.corridors
  }


def main() -> None:
  """Visualize one skill on the grid world, choosing the skill and spawn in the viewer.

  Side-panel controls pick the corridor (which skill runs), the mode, the spawn cell
  (row, col), the heading offset from the corridor direction (0 = aligned), and the start
  speed. Changing any of them respawns the robot. The Info box shows whether the skill is
  RUNNING (started inside its initiation set) or IDLE (started anywhere else, where it
  does nothing). The CLI flags just set the initial selection.

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
    mode: str = CRUISE  # initial mode: "cruise" (non-steering) or "hold" (heading-hold)
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
  skills_by_mode = {m: corridor_skills(world, speeds, mode=m) for m in (CRUISE, HOLD)}

  @_dataclass
  class Sel:
    cid: int
    mode: str
    row: int
    col: int
    heading_deg: float
    speed: float

  init_cell = (
    args.cell
    if args.cell is not None
    else skills_by_mode[args.mode][args.corridor].entry
  )
  sel = Sel(
    cid=args.corridor,
    mode=args.mode,
    row=int(init_cell[0]),
    col=int(init_cell[1]),
    heading_deg=float(args.heading_deg),
    speed=float(args.speed),
  )
  state_box = {"s": np.zeros(5)}

  def current_skill() -> CorridorSkill:
    return skills_by_mode[sel.mode][sel.cid]

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
      "skill": f"corridor {sel.cid} ({sel.mode})",
      "entry cell (initiation set)": str(skill.entry),
      "spawn cell": f"({sel.row}, {sel.col})",
      "state": "RUNNING" if skill.active else "IDLE (outside initiation set)",
      "speed": f"{float(s[V]):.2f} m/s",
      "heading error": f"{math.degrees(skill.heading_error(float(s[THETA]))):+.0f} deg",
      "position": f"({float(s[X]):.2f}, {float(s[Y]):.2f})",
    }

  def gui(server, reset) -> None:
    options = [str(cid) for cid in sorted(world.corridors)]
    with server.gui.add_folder("Skill / spawn"):
      cid_dd = server.gui.add_dropdown("Corridor", options, initial_value=str(sel.cid))
      mode_dd = server.gui.add_dropdown("Mode", (CRUISE, HOLD), initial_value=sel.mode)
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
      sel.mode = mode_dd.value
      sel.row = int(row_in.value)
      sel.col = int(col_in.value)
      sel.heading_deg = float(head_sl.value)
      sel.speed = float(speed_sl.value)
      reset()

    def snap_to_entry(_=None) -> None:
      entry = skills_by_mode[mode_dd.value][int(cid_dd.value)].entry
      row_in.value, col_in.value = int(entry[0]), int(entry[1])
      apply()

    cid_dd.on_update(snap_to_entry)  # picking a corridor snaps the spawn to its entry
    snap_btn.on_click(snap_to_entry)
    for handle in (mode_dd, row_in, col_in, head_sl, speed_sl):
      handle.on_update(apply)

  play.run(model, policy, decimation=4, on_reset=on_reset, status=status, gui=gui)


if __name__ == "__main__":
  main()
