"""
Corridor grid world for the bridging study.

* Corridor -- one straight run of same-id cells, in grid coordinates
  (topology only, no metres): its orientation, its cell span, its cells.
* Grid -- the integer grid plus the corridors parsed out of it (still pure
  topology): 0 = wall, k > 0 = corridor k.
* World -- the continuous metric space laid over a Grid. One grid cell is
  a cell-metre square, so this is where the physical knobs live (cell size, wall
  dimensions) and where world <-> grid conversions and collision queries happen.

Rendering (matplotlib + MuJoCo) is kept out of the data classes, in the free functions
at the bottom.

Frame: x runs east (+column), y runs north (so grid row 0 is the top /
largest y), origin at the bottom-left corner. The robot lives only inside corridor
cells; leaving them (into a 0 cell or off the grid) is a crash.

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.world
    uv run python -m mjlab.tasks.skills.experiments.diffdrive.world --backend mpl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np
import numpy.typing as npt

from mjlab.tasks.skills import play
from mjlab.tasks.skills.interfaces import World

# Corridor orientation: the axis the run extends along.
HORIZONTAL, VERTICAL = 0, 1

# Sample grid with 7 corridors.
DEFAULT_GRID: list[list[int]] = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 5, 5, 5, 6],
  [0, 0, 0, 0, 0, 0, 4, 0, 0, 6],
  [0, 0, 0, 0, 0, 0, 4, 0, 0, 6],
  [0, 0, 0, 3, 3, 3, 4, 0, 0, 6],
  [0, 0, 0, 2, 0, 0, 0, 0, 0, 7],
  [0, 0, 0, 2, 0, 0, 0, 0, 0, 7],
  [0, 0, 0, 2, 0, 0, 0, 0, 0, 7],
  [1, 1, 1, 2, 0, 0, 0, 0, 0, 7],
  [0, 0, 0, 2, 0, 0, 0, 0, 0, 7],
]

# Distinct corridor colors (RGBA), indexed by (corridor id - 1).
_PALETTE: tuple[tuple[float, float, float, float], ...] = (
  (0.20, 0.55, 0.85, 1.0),
  (0.90, 0.45, 0.20, 1.0),
  (0.30, 0.70, 0.35, 1.0),
  (0.80, 0.30, 0.55, 1.0),
  (0.55, 0.40, 0.80, 1.0),
  (0.85, 0.70, 0.20, 1.0),
  (0.25, 0.70, 0.70, 1.0),
)


def _color(cid: int) -> tuple[float, float, float, float]:
  return _PALETTE[(cid - 1) % len(_PALETTE)]


@dataclass(frozen=True)
class Corridor:
  """One straight corridor in grid cells.

  Built from its first and last cell.
  - `orientation` is HORIZONTAL or VERTICAL;
  - `lo`/`hi` are the inclusive cell-index span along the run (columns if horizontal, rows if vertical);
  - `com` is the midpoint between `lo` and `hi`;
  - `length` is the cell count;
  - `cells` lists every (row, col) from `lo` to `hi`.
  """

  start_cell: tuple[int, int]
  end_cell: tuple[int, int]
  orientation: int = field(init=False)
  lo: int = field(init=False)
  hi: int = field(init=False)
  com: float = field(init=False)
  length: int = field(init=False)
  cells: tuple[tuple[int, int], ...] = field(init=False)

  def __post_init__(self) -> None:

    if self.start_cell is None or self.end_cell is None:
      return

    r1, c1 = self.start_cell
    r2, c2 = self.end_cell

    # orientation: 0 = horizontal, 1 = vertical
    if r1 == r2:
      orientation = 0
      lo, hi = sorted((c1, c2))
    elif c1 == c2:
      orientation = 1
      lo, hi = sorted((r1, r2))
    else:
      raise ValueError("Corridor must be straight (horizontal or vertical)")

    if orientation == 0:
      cells = [(r1, c) for c in range(lo, hi + 1)]
    else:
      cells = [(r, c1) for r in range(lo, hi + 1)]

    computed = dict(
      orientation=orientation,
      lo=lo,
      hi=hi,
      com=(lo + hi) / 2.0,
      length=hi - lo + 1,
      cells=cells,
    )
    for name, value in computed.items():
      object.__setattr__(self, name, value)

  @property
  def constant(self) -> int:
    """The grid index held fixed: the run's row if horizontal, its column if vertical."""
    return self.start_cell[self.orientation]

  def __repr__(self) -> str:
    if self.orientation == HORIZONTAL:
      return f"Corridor(row={self.constant}, cols={self.lo}->{self.hi})"
    return f"Corridor(col={self.constant}, rows={self.lo}->{self.hi})"


@dataclass(frozen=True)
class Grid:
  """The integer grid plus the corridors parsed from it (topology, no metres).

  grid[r, c] is 0 for a wall and k > 0 for corridor k; each corridor id
  must form a single straight run. No check on the goodness of the grid is
  performed, its consistency enforcement is left to the user.
  """

  grid: np.ndarray = field(default_factory=lambda: np.asarray(DEFAULT_GRID, dtype=int))
  corridors: dict[int, Corridor] = field(init=False)

  def __post_init__(self) -> None:
    grid = np.asarray(self.grid, dtype=int)
    object.__setattr__(self, "grid", grid)
    object.__setattr__(self, "corridors", self._extract_corridors(grid))

  @staticmethod
  def _extract_corridors(grid: np.ndarray) -> dict[int, Corridor]:
    out: dict[int, Corridor] = {}
    for cid in (int(v) for v in np.unique(grid) if v != 0):
      rows, cols = np.where(grid == cid)
      cells = list(zip(rows.tolist(), cols.tolist(), strict=True))
      out[cid] = Corridor(cells[0], cells[-1])
    return out

  @property
  def nrows(self) -> int:
    return int(self.grid.shape[0])

  @property
  def ncols(self) -> int:
    return int(self.grid.shape[1])

  @property
  def ncorridors(self) -> int:
    return len(self.corridors)

  def corridor(self, cid: int) -> Corridor:
    return self.corridors[cid]

  def in_grid(self, r: int, c: int) -> bool:
    return 0 <= r < self.nrows and 0 <= c < self.ncols

  def is_corridor(self, r: int, c: int) -> bool:
    """Whether cell (r, c) is a corridor (in the grid and nonzero)."""
    return self.in_grid(r, c) and bool(self.grid[r, c] != 0)

  def what_corridor(self, r: int, c: int) -> int:
    return self.grid[r, c]


@dataclass(frozen=True)
class GridWorld(World):
  """
  The continuous metric space laid over a Grid. Holds the physical parameters
  necessary for the construction of the environment.

  - `cell` is the side of each cell in meters;
  - `wall_height`/`wall_thickness` are cosmetic (visualization only).

  Grid <-> world conversions and the collision queries are vectorized over
  arbitrary leading batch dims, so a rollout can score many states at once.
  """

  grid: Grid = field(default_factory=Grid)
  cell: float = 1.0  # metres per grid cell (== corridor width)
  wall_height: float = 0.3  # visualization only
  wall_thickness: float = 0.04  # visualization only

  # Grid pass-throughs (so callers hold one object, not two).

  @property
  def corridors(self) -> dict[int, Corridor]:
    return self.grid.corridors

  def corridor(self, cid: int) -> Corridor:
    return self.grid.corridor(cid)

  @property
  def nrows(self) -> int:
    return self.grid.nrows

  @property
  def ncols(self) -> int:
    return self.grid.ncols

  # Sizes (metres).

  @property
  def half_width(self) -> float:
    return self.cell / 2.0

  @property
  def width(self) -> float:
    return self.ncols * self.cell

  @property
  def height(self) -> float:
    return self.nrows * self.cell

  # Grid <-> world.

  def cell_center(
    self, r: npt.ArrayLike, c: npt.ArrayLike
  ) -> tuple[np.ndarray, np.ndarray]:
    """World (x, y) at the center of grid cell (r, c) (vectorized)."""
    x = (np.asarray(c) + 0.5) * self.cell
    y = (self.nrows - 1 - np.asarray(r) + 0.5) * self.cell
    return x, y

  def world_to_cell(
    self, x: npt.ArrayLike, y: npt.ArrayLike
  ) -> tuple[np.ndarray, np.ndarray]:
    """Grid (r, c) containing world (x, y) (may be out of range; vectorized)."""
    c = np.floor(np.asarray(x, float) / self.cell).astype(int)
    r = self.nrows - 1 - np.floor(np.asarray(y, float) / self.cell).astype(int)
    return r, c

  def in_bounds(self, x: npt.ArrayLike, y: npt.ArrayLike) -> np.ndarray:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (x >= 0.0) & (x < self.width) & (y >= 0.0) & (y < self.height)

  # Continuous-space queries (pure numpy, vectorized over arbitrary leading dims).

  def corridor_at(self, x: npt.ArrayLike, y: npt.ArrayLike) -> np.ndarray | int:
    """Corridor id owning world (x, y); 0 for wall / off-grid."""
    inb = self.in_bounds(x, y)
    r, c = self.world_to_cell(x, y)
    rr = np.clip(r, 0, self.nrows - 1)
    cc = np.clip(c, 0, self.ncols - 1)
    cid = np.where(inb, self.grid.grid[rr, cc], 0)
    return _scalarize(cid)

  def is_free(self, x: npt.ArrayLike, y: npt.ArrayLike) -> np.ndarray | bool:
    """Whether world (x, y) is inside a corridor (True) or a wall (crash)."""
    return _scalarize(np.asarray(self.corridor_at(x, y)) != 0)

  # Corridor geometry in metres (derived from the topological corridor + cell size).

  def centerline(self, corr: Corridor) -> float:
    """Lateral world coordinate of corr's centerline (y if horizontal, else x)."""
    if corr.orientation == HORIZONTAL:
      return (self.nrows - 1 - corr.constant + 0.5) * self.cell
    return (corr.constant + 0.5) * self.cell

  def extent(self, corr: Corridor) -> tuple[float, float]:
    """World (lo, hi) span of corr along its axis (x horizontal, y vertical)."""
    if corr.orientation == HORIZONTAL:
      return corr.lo * self.cell, (corr.hi + 1) * self.cell
    return (self.nrows - 1 - corr.hi) * self.cell, (self.nrows - corr.lo) * self.cell

  def offset(self, cid: int, x: npt.ArrayLike, y: npt.ArrayLike) -> np.ndarray | float:
    """Signed lateral distance of (x, y) from corridor cid's centerline.

    Positive is north (horizontal corridor) or east (vertical). |offset| within
    half_width means on the corridor; the centerline is 0.
    """
    corr = self.corridor(cid)
    coord = y if corr.orientation == HORIZONTAL else x
    return _scalarize(np.asarray(coord, float) - self.centerline(corr))

  # Walls.

  def wall_segments(self) -> list[tuple[float, float, float, float]]:
    """Walls as (cx, cy, sx, sy) boxes: one per free-cell edge facing a wall.

    A wall is drawn on every edge of a free cell whose neighbor is solid or off the
    grid, which exactly encloses the corridors. Each wall edge borders a single free
    cell, so this yields each wall face once.
    """
    half = self.cell / 2.0
    t = self.wall_thickness
    free = self.grid.is_corridor
    segs: list[tuple[float, float, float, float]] = []
    for r in range(self.nrows):
      for c in range(self.ncols):
        if not free(r, c):
          continue
        cx, cy = (float(v) for v in self.cell_center(r, c))
        if not free(r - 1, c):  # north (+y) edge
          segs.append((cx, cy + half, half + t, t))
        if not free(r + 1, c):  # south (-y) edge
          segs.append((cx, cy - half, half + t, t))
        if not free(r, c + 1):  # east (+x) edge
          segs.append((cx + half, cy, t, half + t))
        if not free(r, c - 1):  # west (-x) edge
          segs.append((cx - half, cy, t, half + t))
    return segs


def _scalarize(a: np.ndarray) -> np.ndarray:
  """Return a Python scalar for a 0-d array, else the array unchanged."""
  a = np.asarray(a)
  return a.item() if a.ndim == 0 else a


# Visualization (kept apart from the world model).


def build_spec(world: GridWorld) -> mujoco.MjSpec:
  """Build the MuJoCo spec of the world: floor, corridor tiles and walls.

  No robot and no collisions (walls are visual only; crashes are scored analytically
  via World.is_free). Returning the uncompiled spec lets an experiment add its own
  bodies (e.g. a robot) before compiling; build_world is the compiled shortcut.
  """
  spec = mujoco.MjSpec()
  spec.option.timestep = 0.01
  wb = spec.worldbody
  cx, cy = world.width / 2.0, world.height / 2.0
  span = max(world.width, world.height)

  light = wb.add_light()
  light.pos = np.array([cx, cy, span])
  light.dir = np.array([0.0, 0.0, -1.0])

  cam = wb.add_camera()  # default orientation looks straight down (-z), up = +y
  cam.name = "top"
  cam.pos = np.array([cx, cy, 1.4 * span])

  floor = wb.add_geom()
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.pos = np.array([cx, cy, 0.0])
  floor.size = np.array([cx + world.cell, cy + world.cell, 0.1])
  floor.rgba = np.array([0.18, 0.18, 0.20, 1.0])

  def box(
    px: float, py: float, sx: float, sy: float, sz: float, pz: float, rgba
  ) -> None:
    g = wb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos = np.array([px, py, pz])
    g.size = np.array([sx, sy, sz])
    g.rgba = np.array(rgba)
    g.contype, g.conaffinity = 0, 0  # visual only

  # Corridor floor tiles, colored by id.
  tile = world.cell / 2.0 * 0.96
  for cid, corr in world.corridors.items():
    rgba = _color(cid)
    for r, c in corr.cells:
      tx, ty = (float(v) for v in world.cell_center(r, c))
      box(tx, ty, tile, tile, 0.004, 0.004, rgba)

  # Walls on every free/solid boundary.
  h = world.wall_height
  for wx, wy, sx, sy in world.wall_segments():
    box(wx, wy, sx, sy, h / 2.0, h / 2.0, (0.55, 0.55, 0.62, 1.0))

  return spec


def build_world(world: GridWorld) -> mujoco.MjModel:
  """Compile the world build_spec to a model (the viewer's world, no robot)."""
  return build_spec(world).compile()


def render(world: GridWorld, ax=None, *, centerlines: bool = True):
  """Draw the world top-down with matplotlib; returns the axes.

  Quick offline look (no viewer / MuJoCo). Corridor cells are filled and labeled by id,
  walls are black edges, centerlines are dashed.
  """
  import matplotlib.pyplot as plt
  from matplotlib.patches import Rectangle

  if ax is None:
    _, ax = plt.subplots(figsize=(world.ncols, world.nrows))

  for cid, corr in world.corridors.items():
    for r, c in corr.cells:
      x0, y0 = c * world.cell, (world.nrows - 1 - r) * world.cell
      ax.add_patch(Rectangle((x0, y0), world.cell, world.cell, color=_color(cid)))
    # Label at the corridor's first cell.
    lx, ly = (float(v) for v in world.cell_center(*corr.cells[0]))
    ax.text(lx, ly, str(cid), ha="center", va="center", weight="bold")
    if centerlines:
      lo, hi = world.extent(corr)
      mid = world.centerline(corr)
      if corr.orientation == HORIZONTAL:
        ax.plot([lo, hi], [mid, mid], "k--", lw=0.8, alpha=0.5)
      else:
        ax.plot([mid, mid], [lo, hi], "k--", lw=0.8, alpha=0.5)

  for wx, wy, sx, sy in world.wall_segments():
    ax.add_patch(Rectangle((wx - sx, wy - sy), 2 * sx, 2 * sy, color="black"))

  ax.set_xlim(-world.cell * 0.2, world.width + world.cell * 0.2)
  ax.set_ylim(-world.cell * 0.2, world.height + world.cell * 0.2)
  ax.set_aspect("equal")
  ax.set_xlabel("x (east)")
  ax.set_ylabel("y (north)")
  return ax


def main() -> None:
  from dataclasses import dataclass as _dataclass

  import tyro

  @_dataclass
  class Args:
    cell: float = 1.0
    backend: Literal["viser", "mpl"] = "viser"

  args = tyro.cli(Args)
  world = GridWorld(cell=args.cell)
  if args.backend == "mpl":
    import matplotlib.pyplot as plt

    render(world)
    plt.show()
  else:
    play.run(build_world(world), lambda model, _data: np.zeros(model.nu))


if __name__ == "__main__":
  main()
