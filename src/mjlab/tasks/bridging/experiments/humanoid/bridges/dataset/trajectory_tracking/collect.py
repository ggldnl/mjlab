"""Collect the trajectory tracking dataset."""

import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker import (
  TrackerCfg,
  collect,
)

__all__ = ["TrackerCfg", "collect"]


if __name__ == "__main__":
  collect(tyro.cli(TrackerCfg, config=mjlab.TYRO_FLAGS))
