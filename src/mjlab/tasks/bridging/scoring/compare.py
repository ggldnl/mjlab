"""Run nominal skills, both baselines and bridge sweeps into results.json.

Run: uv run python -m mjlab.tasks.bridging.scoring.compare --output data/bridging/comparison
"""

# pyright: reportPrivateImportUsage=false

import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import tyro

import mjlab
from mjlab.tasks.bridging.baseline.model import (
  TransitionPolicy,
  fingerprint,
  training_distance,
)
from mjlab.tasks.bridging.baseline.scene import Pair
from mjlab.tasks.bridging.bridges.diffusion.config import tracker_experiment
from mjlab.tasks.bridging.bridges.diffusion.execution.runtime import (
  latest_planner_checkpoint,
)
from mjlab.tasks.bridging.scoring import nominal
from mjlab.tasks.bridging.scoring import run as pairs
from mjlab.tasks.bridging.scoring.metrics import METRICS, RUNS, FailureErrors
from mjlab.tasks.bridging.scoring.run import write_json as save
from mjlab.tasks.bridging.tests.stage import find_checkpoint

PAIRS: tuple[Pair, ...] = ("walk2jump", "walk2kick", "jump2walk")
SKILLS: dict[Pair, Literal["walk", "jump", "kick"]] = {
  "walk2jump": "jump",
  "walk2kick": "kick",
  "jump2walk": "walk",
}


@dataclass
class Config:
  output: Path = Path("data/bridging/comparison")
  walk2jump_checkpoint: Path | None = None
  walk2kick_checkpoint: Path | None = None
  jump2walk_checkpoint: Path | None = None
  bridge_checkpoint: Path | None = None
  tracker_checkpoint: Path | None = None
  entry: int | None = None
  seed_start: int = 10000
  device: str = "cuda:0"
  failure_errors: FailureErrors = field(default_factory=FailureErrors)


def transition_checkpoint(pair: Pair, explicit: Path | None) -> Path:
  if explicit is not None:
    if not explicit.is_file():
      raise FileNotFoundError(explicit)
    return explicit.resolve()
  root = Path("logs/rsl_rl") / f"g1_{pair}_transition"
  found = [path for path in root.rglob("model*.pt") if path.is_file()]
  if not found:
    raise FileNotFoundError(f"No transition checkpoint under {root}")
  return max(found, key=lambda path: path.stat().st_mtime).resolve()


def normalized(skill: str, absolute: dict, reference: dict) -> dict:
  ratios = {}
  for name, direction in METRICS[skill].items():
    actual, nominal_value = absolute[name], reference[name]
    if any(not math.isfinite(value) or value < 0 for value in (actual, nominal_value)):
      raise ValueError(f"Invalid absolute score for {name}")
    if direction == "higher":
      if nominal_value == 0:
        raise ValueError(
          "Nominal ball speed is zero; cannot normalize kick performance"
        )
      ratios[name] = actual / nominal_value
    else:
      ratios[name] = (
        (1.0 if actual == 0 else 0.0)
        if nominal_value == 0
        else nominal_value / max(actual, 1e-12)
      )
  scores = {name: min(1.0, ratio) for name, ratio in ratios.items()}
  return {
    "metrics": scores,
    "score": sum(scores.values()) / len(scores),
    "ratios": ratios,
    "selection_score": sum(ratios.values()) / len(ratios),
  }


def normalized_cell(skill: str, cell: dict, reference: dict) -> dict:
  return {
    "distance_m": cell["distance_m"],
    "bridge_duration_s": cell["bridge_duration_s"],
    "absolute": cell["summary"]["mean"],
    "success_rate": cell["summary"]["success_rate"],
    **normalized(skill, cell["summary"]["mean"], reference),
  }


def selection_key(cell: dict) -> tuple:
  """Best clipped score, then unclipped ratio, success, shorter duration and distance."""
  return (
    cell["score"],
    cell["selection_score"],
    cell["success_rate"],
    -cell["bridge_duration_s"],
    -cell["distance_m"],
  )


def comparison(reports: dict) -> dict:
  result = {}
  for pair in PAIRS:
    skill = pair.split("2")[1]
    nominal_report = reports.get(f"nominal/{skill}")
    if not nominal_report or not nominal_report["cells"]:
      continue
    reference = nominal_report["cells"][0]["summary"]["mean"]
    item: dict[str, Any] = {"nominal": reference}
    # Each baseline is one request, the bridge a grid of distances and durations
    for method in ("no-op", "transition"):
      report = reports.get(f"{pair}/{method}")
      if report and report["cells"]:
        item[method.replace("-", "_")] = {
          **normalized_cell(skill, report["cells"][0], reference),
          "provisional": report["status"] != "complete",
        }
    report = reports.get(f"{pair}/bridge")
    if report and report["cells"]:
      cells = [normalized_cell(skill, cell, reference) for cell in report["cells"]]
      item["bridge"] = cells
      item["best_bridge"] = {
        **max(cells, key=selection_key),
        "provisional": report["status"] != "complete",
      }
      distances = sorted({cell["distance_m"] for cell in cells})
      item["best_bridge_by_distance"] = [
        max(
          (cell for cell in cells if math.isclose(cell["distance_m"], distance)),
          key=selection_key,
        )
        for distance in distances
      ]
    result[pair] = item
  return result


def run(cfg: Config):
  if cfg.output.exists():
    raise FileExistsError(f"Choose a new --output directory: {cfg.output}")
  pairs.SceneOptions(seed_start=cfg.seed_start, entry=cfg.entry)
  checkpoints = {}
  incompatible = []
  for pair in PAIRS:
    path = transition_checkpoint(pair, getattr(cfg, f"{pair}_checkpoint"))
    policy, saved = TransitionPolicy.load(path)
    del policy
    try:
      training_distance(saved)
    except ValueError as error:
      incompatible.append(f"{pair}: {path}\n  {error}")
      continue
    if saved["config"]["scene"]["pair"] != pair:
      raise ValueError(f"Checkpoint for {pair} belongs to a different pair")
    if cfg.entry is not None and cfg.entry not in saved["config"]["entries"]:
      raise ValueError(f"Entry {cfg.entry} was not used to train {pair}")
    if (
      cfg.seed_start < saved["training_seed_end"]
      and cfg.seed_start + RUNS > saved["config"]["seed"]
    ):
      raise ValueError(f"Scoring seeds overlap training for {pair}")
    for name in ("source_checkpoint", "target_checkpoint", "selector_path"):
      if (
        fingerprint(Path(saved["config"]["scene"][name])) != saved["fingerprints"][name]
      ):
        raise ValueError(f"{pair}: {name} differs from transition training")
    checkpoints[pair] = path
  if incompatible:
    raise ValueError(
      "Transition checkpoints without a known training placement:\n"
      + "\n".join(incompatible)
      + "\nRetrain the listed skill pairs. "
      "All three pairs are required for the complete comparison."
    )
  bridge = (cfg.bridge_checkpoint or latest_planner_checkpoint()).resolve()
  tracker = find_checkpoint(tracker_experiment("g1"), cfg.tracker_checkpoint).resolve()
  cfg.output.mkdir(parents=True)
  results: dict = {
    "protocol": "nominal_normalized_comparison_v8",
    "status": "running",
    "config": json.loads(json.dumps(asdict(cfg), default=str)),
    "normalization": {
      "lower": "nominal_mean / post_switch_mean",
      "higher": "post_switch_mean / nominal_mean",
      "score": "Mean of per-metric ratios clipped to [0, 1]",
      "selection": "Highest mean clipped ratio; ties prefer mean unclipped ratio, success rate, shorter duration, shorter distance",
      "comparison": "One shared request per pair: a timed straight walk or the first landing. Instant switch starts the incoming skill on the robot. The bridge places the entry sequence at each grid distance ahead of the robot and gets each grid duration, best overall and best duration at each distance. The walk2jump and jump2walk transitions start the incoming skill where they hand over. The walk2kick transition keeps the training scene: ball and reference fixed in the world, requested within its training distance of the entry",
      "selection_data": "Best parameters selected on these evaluation runs; exploratory result",
    },
    "checkpoints": {},
    "reports": {},
    "comparison": {},
    "active_stage": None,
  }
  destination = cfg.output / "results.json"
  save(destination, results)
  try:
    snapshot_dir = cfg.output / "checkpoints"
    snapshot_dir.mkdir()
    for pair, path in checkpoints.items():
      snapshot = snapshot_dir / f"{pair}.pt"
      shutil.copyfile(path, snapshot)
      checkpoints[pair] = snapshot
      results["checkpoints"][pair] = {
        "original": str(path),
        "snapshot": str(snapshot.resolve()),
        "sha256": fingerprint(snapshot),
      }
    for name, path in (("bridge", bridge), ("tracker", tracker)):
      results["checkpoints"][name] = {"path": str(path), "sha256": fingerprint(path)}
    save(destination, results)

    def update(key: str, report: dict):
      results["reports"][key] = report
      results["comparison"] = comparison(results["reports"])
      save(destination, results)

    for stage in ("nominal", "no-op", "transition", "bridge"):
      for pair in PAIRS:
        skill = SKILLS[pair]
        key = f"nominal/{skill}" if stage == "nominal" else f"{pair}/{stage}"
        results["active_stage"] = key
        save(destination, results)
        print(f"Scoring {key}", flush=True)
        options: dict[str, Any] = dict(
          output=cfg.output,
          device=cfg.device,
          seed_start=cfg.seed_start,
          entry=cfg.entry,
          failure_errors=cfg.failure_errors,
          transition_checkpoint=checkpoints[pair],
        )
        if stage == "nominal":
          nominal.run(
            nominal.Config(skill=skill, **options),
            on_update=lambda report, key=key: update(key, report),
          )
        else:
          if stage == "bridge":
            for name, path in (("bridge", bridge), ("tracker", tracker)):
              if fingerprint(path) != results["checkpoints"][name]["sha256"]:
                raise ValueError(f"{name} checkpoint changed during scoring")
          pairs.run(
            pair,
            pairs.Config(
              method=stage,
              bridge_checkpoint=bridge,
              tracker_checkpoint=tracker,
              **options,
            ),
            on_update=lambda report, key=key: update(key, report),
          )
    results["status"], results["active_stage"] = "complete", None
    save(destination, results)
    print(f"Saved all results to {destination}", flush=True)
  except BaseException as error:
    results["status"] = (
      "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    )
    results["error"] = str(error)
    save(destination, results)
    raise


if __name__ == "__main__":
  run(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
