"""Action discontinuity of the bridge and both baselines, on the comparison's trials.

Replays the requests of a compare.py output with its checkpoints, seeds and selected
bridge setting, and records the joint position target applied at every control step.
When the controller changes, the target jumps in one step by however much the two
policies disagree. The skill scores do not see this.

Values are joint target changes over one control step, in radians, RMS over the
joints unless stated. A seam is a step where the controller changes: the request
for the bridge and the transition policy, and the handoff. The instant switch has
one seam.

    seam_rad                 largest change at a seam
    request_jump_rad         first step after the request
    handoff_jump_rad         first incoming step
    handoff_joint_jump_rad   largest single joint change at the first incoming step
    peak_step_rad            largest change from the request to the end of the trial
    mean_step_rad            mean change over the same steps

Each ratio divides a method's mean by the nominal mean of the same value, so 1 looks
like the incoming skill's own steps. Statistics use every trial that reached the
step, failed or not. The scores of the same trials are compared with the comparison
in score_difference, which stays near zero when requests and models match.

Output in OUTPUT: results.json with every report and the discontinuity summary,
actions.csv with one row per setting, and the usual PAIR/METHOD report folders.

Run

1. Selected bridge setting and both baselines, every pair.

    uv run python -m mjlab.tasks.bridging.scoring.discontinuity --comparison data/bridging/comparison --output data/bridging/discontinuity

2. All 45 bridge settings.

    uv run python -m mjlab.tasks.bridging.scoring.discontinuity --comparison data/bridging/comparison --output data/bridging/discontinuity_grid --bridge grid

3. One pair only.

    uv run python -m mjlab.tasks.bridging.scoring.discontinuity --comparison data/bridging/comparison --output data/bridging/discontinuity_kick --pair walk2kick
"""

# pyright: reportPrivateImportUsage=false

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Literal

import tyro

import mjlab
from mjlab.tasks.bridging.baseline.model import fingerprint
from mjlab.tasks.bridging.baseline.scene import Pair
from mjlab.tasks.bridging.scoring import nominal
from mjlab.tasks.bridging.scoring import run as pair_scoring
from mjlab.tasks.bridging.scoring.compare import PAIRS, SKILLS
from mjlab.tasks.bridging.scoring.metrics import ACTION_METRICS, FailureErrors
from mjlab.tasks.bridging.scoring.run import write_json as save

STAGES = ("nominal", "no-op", "transition", "bridge")


@dataclass
class Config:
  comparison: Path = Path("data/bridging/comparison")
  output: Path = Path("data/bridging/discontinuity")
  bridge: Literal["best", "grid"] = "best"
  pair: Literal["all", "walk2jump", "walk2kick", "jump2walk"] = "all"
  device: str = "cuda:0"


def action_summary(rows: list[dict]) -> dict[str, dict[str, Any]]:
  """Mean, sample deviation and maximum of each value over the trials that have it."""
  result = {}
  for name in ACTION_METRICS:
    values = [row["action"][name] for row in rows if row["action"][name] is not None]
    result[name] = {
      "mean": fmean(values) if values else None,
      "std": stdev(values) if len(values) > 1 else None,
      "max": max(values) if values else None,
      "runs": len(values),
    }
  return result


def same_setting(a: dict, b: dict) -> bool:
  return all(
    a[name] == b[name]
    or (a[name] is not None and b[name] is not None and math.isclose(a[name], b[name]))
    for name in ("distance_m", "bridge_duration_s")
  )


def cell_summary(cell: dict, reference: dict | None, previous: dict | None) -> dict:
  action = action_summary(cell["trials"])
  return {
    "distance_m": cell["distance_m"],
    "bridge_duration_s": cell["bridge_duration_s"],
    "success_rate": cell["summary"]["success_rate"],
    "action": action,
    "ratios": {
      name: action[name]["mean"] / reference[name]["mean"]
      if reference and action[name]["mean"] is not None and reference[name]["mean"]
      else None
      for name in ACTION_METRICS
    },
    "score_difference": {
      name: value - previous["summary"]["mean"][name]
      for name, value in cell["summary"]["mean"].items()
    }
    if previous
    else None,
  }


def discontinuity(reports: dict, previous: dict, pairs: tuple[Pair, ...]) -> dict:
  result = {}
  for pair in pairs:
    skill = SKILLS[pair]
    item: dict[str, Any] = {}
    nominal_report = reports.get(f"nominal/{skill}")
    reference = (
      action_summary(nominal_report["cells"][0]["trials"])
      if nominal_report and nominal_report["cells"]
      else None
    )
    for stage in STAGES:
      key = f"nominal/{skill}" if stage == "nominal" else f"{pair}/{stage}"
      report = reports.get(key)
      if not report or not report["cells"]:
        continue
      old = previous["reports"].get(key, {"cells": []})["cells"]
      cells = [
        cell_summary(
          cell, reference, next((o for o in old if same_setting(o, cell)), None)
        )
        for cell in report["cells"]
      ]
      if stage != "bridge":
        item[stage.replace("-", "_")] = cells[0]
        continue
      # Bridge cells at the settings the comparison selected
      item["bridge"] = cells
      chosen = previous["comparison"][pair]
      best = [cell for cell in cells if same_setting(cell, chosen["best_bridge"])]
      if best:
        item["best_bridge"] = best[0]
      picks = chosen.get("best_bridge_by_distance", [])
      by_distance = [c for p in picks for c in cells if same_setting(c, p)]
      if picks and len(by_distance) == len(picks):
        item["best_bridge_by_distance"] = by_distance
    result[pair] = item
  return result


def table(summary: dict) -> list[dict]:
  rows = []
  for pair, item in summary.items():
    for method in ("nominal", "no_op", "transition", "bridge"):
      cells = item.get(method)
      for cell in cells if isinstance(cells, list) else [cells] if cells else []:
        row: dict[str, Any] = {
          "pair": pair,
          "method": method,
          "distance_m": cell["distance_m"],
          "bridge_duration_s": cell["bridge_duration_s"],
          "success_rate": cell["success_rate"],
        }
        for name in ACTION_METRICS:
          for statistic in ("mean", "std", "max"):
            row[f"{statistic}_{name}"] = cell["action"][name][statistic]
          row[f"ratio_{name}"] = cell["ratios"][name]
        rows.append(row)
  return rows


def write_table(path: Path, rows: list[dict]):
  if not rows:
    return
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def print_summary(summary: dict):
  names = ("seam_rad", "handoff_jump_rad", "peak_step_rad", "mean_step_rad")
  print("pair       method       success    seam handoff    peak    mean seam/nom")
  for pair, item in summary.items():
    for method in ("nominal", "no_op", "transition", "best_bridge"):
      cell = item.get(method)
      if cell is None:
        continue
      values = [cell["action"][name]["mean"] for name in names]
      values.append(cell["ratios"]["seam_rad"])
      text = " ".join("      -" if v is None else f"{v:7.3f}" for v in values)
      print(f"{pair:<10} {method:<12} {cell['success_rate']:7.2f} {text}")


def run(cfg: Config):
  selected: tuple[Pair, ...] = PAIRS if cfg.pair == "all" else (cfg.pair,)
  if cfg.output.exists():
    raise FileExistsError(f"Choose a new --output directory: {cfg.output}")
  previous = json.loads((cfg.comparison / "results.json").read_text(encoding="utf-8"))
  if previous["status"] != "complete":
    raise ValueError(f"The comparison is {previous['status']}, finish it first")
  checkpoints = {}
  for name, item in previous["checkpoints"].items():
    # Transition snapshots sit next to results.json, wherever the comparison moved
    path = (
      cfg.comparison / "checkpoints" / f"{name}.pt"
      if name in PAIRS
      else Path(item["path"])
    )
    if fingerprint(path) != item["sha256"]:
      raise ValueError(f"{name} checkpoint differs from the comparison: {path}")
    checkpoints[name] = path.resolve()
  for pair in selected:
    if "best_bridge" not in previous["comparison"].get(pair, {}):
      raise ValueError(f"The comparison has no bridge result for {pair}")
  settings = previous["config"]
  cfg.output.mkdir(parents=True)
  # The comparison's requests, run.py rejects them if skills or sampling differ
  sources = []
  for pair in selected:
    for source in sorted((cfg.comparison / pair).glob("source_*.pt")):
      (cfg.output / pair).mkdir(exist_ok=True)
      shutil.copyfile(source, cfg.output / pair / source.name)
      sources.append(f"{pair}/{source.name}")
  results: dict[str, Any] = {
    "protocol": "action_discontinuity_v1",
    "status": "running",
    "config": json.loads(json.dumps(asdict(cfg), default=str)),
    "comparison": {
      "path": str(cfg.comparison.resolve()),
      "protocol": previous["protocol"],
      "seed_start": settings["seed_start"],
      "entry": settings["entry"],
      "sources": sources,
    },
    "units": "Joint position target change per control step in rad, RMS over joints unless joint",
    "ratios": "Method mean over the nominal mean of the same value",
    "checkpoints": {
      name: {"path": str(path), "sha256": previous["checkpoints"][name]["sha256"]}
      for name, path in checkpoints.items()
    },
    "reports": {},
    "discontinuity": {},
    "active_stage": None,
  }
  destination = cfg.output / "results.json"
  save(destination, results)

  def update(key: str, report: dict):
    results["reports"][key] = report
    results["discontinuity"] = discontinuity(results["reports"], previous, selected)
    write_table(cfg.output / "actions.csv", table(results["discontinuity"]))
    save(destination, results)

  try:
    for stage in STAGES:
      for pair in selected:
        skill = SKILLS[pair]
        key = f"nominal/{skill}" if stage == "nominal" else f"{pair}/{stage}"
        results["active_stage"] = key
        save(destination, results)
        print(f"Scoring {key}", flush=True)
        options: dict[str, Any] = dict(
          output=cfg.output,
          device=cfg.device,
          seed_start=settings["seed_start"],
          entry=settings["entry"],
          failure_errors=FailureErrors(**settings["failure_errors"]),
          transition_checkpoint=checkpoints[pair],
        )
        on_update = lambda report, key=key: update(key, report)  # noqa: E731
        if stage == "nominal":
          nominal.run(nominal.Config(skill=skill, **options), on_update=on_update)
          continue
        if stage == "bridge":
          best = previous["comparison"][pair]["best_bridge"]
          options |= dict(
            bridge_checkpoint=checkpoints["bridge"],
            tracker_checkpoint=checkpoints["tracker"],
          )
          if cfg.bridge == "best":
            options |= dict(
              distance=best["distance_m"], duration=best["bridge_duration_s"]
            )
        pair_scoring.run(
          pair, pair_scoring.Config(method=stage, **options), on_update=on_update
        )
    results["status"], results["active_stage"] = "complete", None
    save(destination, results)
    print_summary(results["discontinuity"])
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
