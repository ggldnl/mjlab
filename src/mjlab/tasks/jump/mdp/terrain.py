"""Gap terrain for the jump task.

Two flat platforms separated by a pit. The robot spawns on the near platform
facing +x and must clear the gap to land on the far platform. The pit makes
jumping the only solution - a policy cannot simply walk to the target.

The gap width is interpolated by difficulty, so in curriculum mode row 0 is
(nearly) flat ground and higher rows open progressively wider gaps. The terrain
also publishes a set of *landing patches* - valid target positions just past the
gap on the far platform - which the jump abstraction samples as its target. This
couples the jump distance to the gap automatically: a flat row yields a short
hop, a wide-gap row yields a long jump.
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

_PLATFORM_RGBA = (0.45, 0.5, 0.6, 1.0)
_PIT_RGBA = (0.1, 0.1, 0.12, 1.0)


@dataclass(kw_only=True)
class GapTerrainCfg(SubTerrainCfg):
  """A near platform, a gap, and a far platform, with tops at z=0."""

  near_length: float = 2.5
  """Length of the near platform along x, in meters."""
  gap_range: tuple[float, float] = (0.0, 0.6)
  """Min and max gap width, interpolated by difficulty, in meters. Start at 0 so
  the easiest curriculum level is flat ground."""
  platform_thickness: float = 0.2
  """Thickness of each platform, in meters (tops sit at z=0)."""
  floor_depth: float = 1.0
  """Depth of the pit floor below the platform tops, in meters."""
  origin_setback: float = 0.35
  """Distance back from the near platform's far edge where the robot spawns."""

  landing_patch_name: str = "landing"
  """Name under which the far-platform landing targets are published."""
  landing_offset: float = 0.1
  """Distance past the far edge of the gap where the landing zone begins."""
  landing_zone_length: float = 0.4
  """Length (x) of the landing zone on the far platform, in meters."""
  landing_lateral: float = 0.3
  """Half-width (y) of the landing zone about the patch center, in meters."""
  num_landing_patches: int = 12
  """Number of candidate landing targets sampled in the landing zone."""

  def __post_init__(self) -> None:
    # Register the landing patch set so the generator pre-allocates storage for
    # the explicit patches returned by ``function``.
    self.flat_patch_sampling = {
      self.landing_patch_name: FlatPatchSamplingCfg(
        num_patches=self.num_landing_patches,
        patch_radius=0.15,
      )
    }

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    size_x, size_y = self.size

    gap = self.gap_range[0] + difficulty * (self.gap_range[1] - self.gap_range[0])
    far_start = self.near_length + gap

    geometries: list[TerrainGeometry] = []

    def add_box(center, half) -> None:
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=center)
      geometries.append(TerrainGeometry(geom=geom, color=_PLATFORM_RGBA))

    # Near platform: x in [0, near_length], top at z=0.
    add_box(
      center=(self.near_length / 2, size_y / 2, -self.platform_thickness / 2),
      half=(self.near_length / 2, size_y / 2, self.platform_thickness / 2),
    )
    # Far platform: x in [far_start, size_x], top at z=0.
    far_len = max(1e-3, size_x - far_start)
    add_box(
      center=(far_start + far_len / 2, size_y / 2, -self.platform_thickness / 2),
      half=(far_len / 2, size_y / 2, self.platform_thickness / 2),
    )
    # Pit floor spanning the whole patch, to catch missed jumps.
    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(size_x / 2, size_y / 2, 0.05),
      pos=(size_x / 2, size_y / 2, -self.floor_depth - 0.05),
    )
    geometries.append(TerrainGeometry(geom=floor, color=_PIT_RGBA))

    # Landing targets: a band on the far platform just past the gap.
    x0 = far_start + self.landing_offset
    x1 = x0 + self.landing_zone_length
    yc = size_y / 2
    patches = np.zeros((self.num_landing_patches, 3))
    patches[:, 0] = rng.uniform(x0, x1, self.num_landing_patches)
    patches[:, 1] = rng.uniform(
      yc - self.landing_lateral, yc + self.landing_lateral, self.num_landing_patches
    )
    patches[:, 2] = 0.0

    # Spawn on the near platform, set back from the gap edge, facing +x.
    origin = np.array([self.near_length - self.origin_setback, size_y / 2, 0.0])
    return TerrainOutput(
      origin=origin,
      geometries=geometries,
      flat_patches={self.landing_patch_name: patches},
    )
