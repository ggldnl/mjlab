"""
A controller owns the skills and the bridge, decides which skill should be active,
and uses the bridge to transition there.

For diffdrive-in-corridors the decision comes for free, positionally: follow the active
corridor's skill and, as soon as the robot reaches the boundary to the next corridor,
switch to that corridor's skill  invoking the bridge to carry it across (a side entry
the skill itself can't make).
In other experiments, the Controller could be implemented using a finite state machine,
a neural network, a planning algorithm, ...
"""

from __future__ import annotations

from collections.abc import Mapping

from mjlab.tasks.skills.experiments.diffdrive.gridworld import (
  HORIZONTAL,
  Corridor,
  GridWorld,
)
from mjlab.tasks.skills.experiments.diffdrive.robot import X, Y
from mjlab.tasks.skills.interfaces import Bridge, Command, Controller, Skill, State


def _adjacency_cell(world: GridWorld, k: int, n: int) -> tuple[int, int] | None:
  """A cell of corridor k orthogonally adjacent to corridor n (or None)."""
  others = set(world.corridor(n).cells)
  for r, c in world.corridor(k).cells:
    if any((r + dr, c + dc) in others for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
      return (r, c)
  return None


def _axis_coord(corr: Corridor, cell: tuple[int, int]) -> float:
  """Position of `cell` along the corridor's axis, increasing in the +x/+y sense."""
  r, c = cell
  return float(c) if corr.orientation == HORIZONTAL else float(-r)


def _junction_cell(world: GridWorld, k: int, nxt: int) -> tuple[int, int] | None:
  """The cell at which to hand corridor k over to nxt the turn corner.

  At an overlap junction (nxt crosses k's line) it is nxt's cell on that line,
  which the robot reaches naturally. At a side branch where k's straight path never
  enters nxt, it is k's own last cell beside nxt (past which the robot would
  hit the wall). The corner trigger covers both; "entered the next corridor"
  only covers the overlap case.
  """
  corr = world.corridor(k)
  axis = corr.orientation  # the grid index k holds constant: 0 (row) if H, 1 (col) if V
  const = corr.constant
  overlap = _adjacency_cell(world, nxt, k)  # a cell of nxt touching k
  if overlap is not None and overlap[axis] == const:
    return overlap
  return _adjacency_cell(world, k, nxt)


def junction_map(world: GridWorld) -> dict[int, tuple[tuple[int, int], int]]:
  """Per active corridor: the cell at which to switch, and the corridor to switch to.

  Corridors are numbered in traversal order, so this walks 1 -> 2 -> ... -> n.
  """
  order = sorted(world.corridors)
  out: dict[int, tuple[tuple[int, int], int]] = {}
  for k, nxt in zip(order[:-1], order[1:], strict=True):
    cell = _junction_cell(world, k, nxt)
    if cell is not None:
      out[k] = (cell, nxt)
  return out


def travel_directions(world: GridWorld) -> dict[int, tuple[int, int]]:
  """Unit travel vector per corridor, from its k-1 junction toward its k+1 one.

  Corridor k is entered from k-1 and exited toward k+1; the cruise direction
  points entry -> exit along the axis. The first/last corridor (no predecessor/successor)
  points toward/away from its one junction instead.
  """
  directions: dict[int, tuple[int, int]] = {}
  for cid, corr in world.corridors.items():
    entry = _adjacency_cell(world, cid, cid - 1) if cid - 1 in world.corridors else None
    exit_ = _adjacency_cell(world, cid, cid + 1) if cid + 1 in world.corridors else None
    coords = [_axis_coord(corr, cell) for cell in corr.cells]
    centroid = sum(coords) / len(coords)
    if entry is not None and exit_ is not None:
      delta = _axis_coord(corr, exit_) - _axis_coord(corr, entry)
    elif exit_ is not None:
      delta = _axis_coord(corr, exit_) - centroid
    elif entry is not None:
      delta = centroid - _axis_coord(corr, entry)
    else:
      delta = 1.0
    d = 1 if delta >= 0 else -1
    directions[cid] = (d, 0) if corr.orientation == HORIZONTAL else (0, d)
  return directions


class CorridorController(Controller):
  """Positional controller for the corridor world.

  Follows the active corridor's skill; at each junction it begins a bridge transition to
  the next corridor's skill and hands control over once the bridge reports `done`. The
  switching decision is carried out by the _decide() method; the switching mechanism
  (run the bridge, then activate the target) is generic.
  """

  def __init__(
    self, world: GridWorld, skills: Mapping[int, Skill], bridge: Bridge
  ) -> None:

    # Skills and bridge
    self.skills = dict(skills)
    self.bridge = bridge

    # World related stuff
    self.world = world
    self._junctions = junction_map(world)

    # Current and next skill (set by reset)
    self.current: int = 1
    self.target: int | None = None
    self.reset()

  def reset(self) -> None:
    self.current = min(self.skills)  # first corridor in the sequence
    self.target = None  # corridor we are switching to, else None
    for skill in self.skills.values():  # re-arm latched skills (e.g. CorridorSkill)
      reset = getattr(skill, "reset", None)
      if callable(reset):
        reset()

  @property
  def switching(self) -> bool:
    return self.target is not None

  @property
  def mode(self) -> str:
    """Human-readable state, for status overlays: "2" or "2 -> 3"."""
    return f"{self.current} -> {self.target}" if self.switching else str(self.current)

  def _decide(self, state: State) -> int | None:
    """
    Fires at the junction corner of the active corridor.
    This is the only experiment-specific part of the controller.
    """
    target = self._junctions.get(self.current)
    if target is None:
      return None
    cell, nxt = target
    r, c = self.world.world_to_cell(float(state[X]), float(state[Y]))
    return nxt if (int(r), int(c)) == cell else None

  def step(self, state: State) -> Command:
    if not self.switching:
      nxt = self._decide(state)
      if nxt is not None:  # reached a junction -> start bridging to the next skill
        self.target = nxt
        self.bridge.reset(self.skills[self.current], self.skills[nxt])
    if self.switching:
      command, done = self.bridge.step(state)
      if done:  # the bridge delivered us into the next corridor; hand control over
        assert self.target is not None
        self.current, self.target = self.target, None
      return command
    return self.skills[self.current](state)
