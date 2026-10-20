"""Nominal incoming skill performance, without a bridge or a parameter sweep.

Run: uv run python -m mjlab.tasks.bridging.scoring.nominal --skill jump
"""

# pyright: reportPrivateImportUsage=false

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.baseline.scene import Pair
from mjlab.tasks.bridging.scoring.metrics import RUNS, summarize
from mjlab.tasks.bridging.scoring.rollout import add_noise, measure
from mjlab.tasks.bridging.scoring.run import SceneOptions, report_header, save, setup


@dataclass
class Config(SceneOptions):
  skill: Literal["walk", "jump", "kick"] = "jump"
  transition_checkpoint: Path | None = None


@torch.inference_mode()
def run(cfg: Config, on_update: Callable[[dict], None] | None = None):
  pair = cast(
    Pair, {"walk": "jump2walk", "jump": "walk2jump", "kick": "walk2kick"}[cfg.skill]
  )
  directory = cfg.output / "nominal" / cfg.skill
  directory.mkdir(parents=True, exist_ok=False)
  scene, _, _ = setup(pair, cfg, transition_checkpoint=cfg.transition_checkpoint)
  report = None
  try:
    report = report_header(scene, cfg, "nominal", cfg.transition_checkpoint)
    save(directory, report, on_update)
    seeds = tuple(range(cfg.seed_start, cfg.seed_start + RUNS))
    scene.reset(cfg.seed_start)
    scene.nominal()
    add_noise(scene, seeds)
    print(
      f"Scoring {RUNS} parallel runs with perturbation seeds {seeds[0]} through {seeds[-1]}",
      flush=True,
    )
    rows = measure(scene, "nominal", seeds, cfg.failure_errors)
    report["cells"].append(
      {
        "distance_m": None,
        "bridge_duration_s": None,
        "transition_limit_s": None,
        "summary": summarize(rows, cfg.skill),
        "trials": rows,
      }
    )
    report["status"] = "complete"
    save(directory, report, on_update)
    print(report["cells"][0]["summary"], flush=True)
  except Exception as error:
    if report is not None:
      report["status"], report["error"] = "failed", str(error)
      save(directory, report, on_update)
    raise
  finally:
    scene.close()


if __name__ == "__main__":
  run(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
