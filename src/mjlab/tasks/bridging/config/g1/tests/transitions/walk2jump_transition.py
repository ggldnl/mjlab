"""Watch a dedicated walk to jump policy and request its transition in Viser.

Run from the repository root:
uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.walk2jump_transition
"""

from dataclasses import dataclass
from pathlib import Path

import tyro

import mjlab
from mjlab.tasks.bridging.baseline.play import PlayCfg, play


@dataclass(frozen=True)
class Config(PlayCfg):
  checkpoint: Path = Path("logs/baseline/walk2jump/seed_0/model.pt")
  automatic: bool = False


def main() -> None:
  cfg = tyro.cli(Config, config=mjlab.TYRO_FLAGS)
  if cfg.scene.pair != "walk2jump":
    raise ValueError("This viewer requires --scene.pair walk2jump")
  play(cfg)


if __name__ == "__main__":
  main()
