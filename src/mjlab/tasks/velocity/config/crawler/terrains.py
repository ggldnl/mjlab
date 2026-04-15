"""
Holds terrain configurations.
"""

from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ALL_TERRAINS_CFG


# Map a few available MJLab terrain presets to simple names
_TRAINING_SUB_TERRAINS = {
    "flat": ALL_TERRAINS_CFG.sub_terrains["flat"],
    "rough": ALL_TERRAINS_CFG.sub_terrains.get("random_uniform", ALL_TERRAINS_CFG.sub_terrains["flat"]),
    "obstacles": ALL_TERRAINS_CFG.sub_terrains.get("discrete_obstacles", ALL_TERRAINS_CFG.sub_terrains["flat"]),
    "stairs_up": ALL_TERRAINS_CFG.sub_terrains.get("mesh_stair_up", ALL_TERRAINS_CFG.sub_terrains["flat"]),
    "stairs_down": ALL_TERRAINS_CFG.sub_terrains.get("mesh_stair_down", ALL_TERRAINS_CFG.sub_terrains["flat"]),
}

def simple_terrain_cfg() -> TerrainEntityCfg:
    # Pick an existing terrain from MJLab config
    base_cfg = ALL_TERRAINS_CFG.sub_terrains["flat"]  # guaranteed to exist

    return TerrainEntityCfg(
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(8.0, 8.0),
            num_rows=1,
            num_cols=1,
            curriculum=False,  # explicitly disabled
            sub_terrains={
                "flat": base_cfg,
            },
            difficulty_range=(0.0, 0.0),  # no variation
            border_width=0.0,
        ),
    )


def training_terrain_cfg() -> TerrainEntityCfg:
  return TerrainEntityCfg(
    terrain_generator=TerrainGeneratorCfg(
      sub_terrains=_TRAINING_SUB_TERRAINS,
      curriculum=True,  # Robots are promoted to harder rows as they improve
      num_rows=10,
      num_cols=20,
      size=(8.0, 8.0),  # Per-tile size in meters
      border_width=20.0,
    ),
    max_init_terrain_level=4,  # New envs start on the first 5 rows only
  )


def play_terrain_cfg() -> TerrainEntityCfg:
    return TerrainEntityCfg(
        terrain_generator=TerrainGeneratorCfg(
            sub_terrains={"flat": ALL_TERRAINS_CFG.sub_terrains["flat"]},
            curriculum=False,
            num_rows=5,
            num_cols=5,
            size=(8.0, 8.0),
            border_width=10.0,
        ),
    )