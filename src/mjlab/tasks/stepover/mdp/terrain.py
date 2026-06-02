"""Barrier terrain for the step-over task.

Flat ground with a single thin barrier running across the path. The robot spawns
behind it facing +x and must step over the barrier one leg at a time. The
barrier height is interpolated by difficulty, so in curriculum mode row 0 is
flat ground (just walk) and higher rows raise a taller barrier.

The terrain publishes a single ``barrier`` flat-patch holding the barrier
top-center and height, which the step-over abstraction samples as its reference
(coupling the via-point height to the curriculum level automatically).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

_GROUND_RGBA = (0.45, 0.5, 0.6, 1.0)
_BARRIER_RGBA = (0.75, 0.45, 0.2, 1.0)


@dataclass(kw_only=True)
class BarrierTerrainCfg(SubTerrainCfg):
  """Flat ground with one cross-path barrier, ground top at z=0."""

  ground_thickness: float = 0.2
  """Thickness of the ground slab, in meters (top sits at z=0)."""
  barrier_height_range: tuple[float, float] = (0.0, 0.3)
  """Min and max barrier height, interpolated by difficulty, in meters. Start at
  0 so the easiest curriculum level is flat ground."""
  barrier_thickness: float = 0.1
  """Thickness (x) of the barrier, in meters."""
  spawn_setback: float = 1.0
  """Distance from the tile's near edge to the spawn, in meters."""
  spawn_distance: float = 0.4
  """Distance from the spawn to the barrier, in meters. The robot starts
  standing right in front of the barrier (no walking approach)."""
  barrier_patch_name: str = "barrier"
  """Name under which the barrier top-center/height is published."""

  def __post_init__(self) -> None:
    # Register the barrier patch so the generator pre-allocates storage for the
    # explicit patch returned by ``function``.
    self.flat_patch_sampling = {
      self.barrier_patch_name: FlatPatchSamplingCfg(
        num_patches=1,
        patch_radius=0.05,
      )
    }

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    size_x, size_y = self.size

    height = self.barrier_height_range[0] + difficulty * (
      self.barrier_height_range[1] - self.barrier_height_range[0]
    )

    geometries: list[TerrainGeometry] = []

    # Ground slab spanning the whole tile, top at z=0.
    ground = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(size_x / 2, size_y / 2, self.ground_thickness / 2),
      pos=(size_x / 2, size_y / 2, -self.ground_thickness / 2),
    )
    geometries.append(TerrainGeometry(geom=ground, color=_GROUND_RGBA))

    # Spawn standing right in front of the barrier, facing +x.
    origin = np.array([self.spawn_setback, size_y / 2, 0.0])
    barrier_x = origin[0] + self.spawn_distance

    # Barrier: a full-width slab standing on the ground (skip if degenerate).
    if height > 1e-3:
      barrier = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.barrier_thickness / 2, size_y / 2, height / 2),
        pos=(barrier_x, size_y / 2, height / 2),
      )
      geometries.append(TerrainGeometry(geom=barrier, color=_BARRIER_RGBA))

    patch = np.array([[barrier_x, size_y / 2, height]])  # Top-center.
    return TerrainOutput(
      origin=origin,
      geometries=geometries,
      flat_patches={self.barrier_patch_name: patch},
    )
