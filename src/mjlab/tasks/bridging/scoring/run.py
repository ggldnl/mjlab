"""Shared scene setup and plain JSON/CSV output for the scoring commands."""

# pyright: reportPrivateUsage=false, reportPrivateImportUsage=false

import csv
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.baseline.model import (
  TransitionPolicy,
  fingerprint,
  training_distance,
)
from mjlab.tasks.bridging.baseline.scene import Pair, SceneCfg, TransitionScene
from mjlab.tasks.bridging.scoring.metrics import (
  DISTANCES,
  DURATIONS,
  METRICS,
  RUNS,
  FailureErrors,
  summarize,
)
from mjlab.tasks.bridging.scoring.rollout import (
  TRIGGERS,
  WALK_SAMPLING,
  measure,
  placement,
  request_kind,
  source_request,
)
from mjlab.tasks.bridging.selector import STATES_PATH


@dataclass
class SceneOptions:
  entry: int | None = None
  selector: Path | None = None
  source_checkpoint: Path | None = None
  target_checkpoint: Path | None = None
  walk_speed: float | None = None
  seed_start: int = 10000
  device: str = "cuda:0"
  output: Path = Path("data/bridging/scoring")
  failure_errors: FailureErrors = field(default_factory=FailureErrors)

  def __post_init__(self):
    if self.seed_start < 0 or (self.entry is not None and self.entry < 0):
      raise ValueError("Seeds and entry indices must be nonnegative")


@dataclass
class Config(SceneOptions):
  method: Literal["bridge", "transition", "no-op"] = "bridge"
  distance: float | None = None
  duration: float | None = None
  bridge_checkpoint: Path | None = None
  tracker_checkpoint: Path | None = None
  transition_checkpoint: Path | None = None

  def __post_init__(self):
    super().__post_init__()
    if self.method != "bridge" and (
      self.distance is not None or self.duration is not None
    ):
      raise ValueError(
        "Distance and duration are bridge parameters; neither baseline accepts them"
      )
    for value, grid, name in (
      (self.distance, DISTANCES, "distance"),
      (self.duration, DURATIONS, "duration"),
    ):
      if value is not None and not any(math.isclose(value, x) for x in grid):
        raise ValueError(f"{name} must be one of {grid}")
    if self.method == "transition" and self.transition_checkpoint is None:
      raise ValueError("Supply --transition-checkpoint for the transition method")


def setup(
  pair: Pair,
  cfg: SceneOptions,
  method="nominal",
  transition_checkpoint=None,
  bridge_checkpoint=None,
  tracker_checkpoint=None,
):
  policy, metadata = None, None
  if transition_checkpoint is not None:
    policy, metadata = TransitionPolicy.load(transition_checkpoint, cfg.device)
    saved = metadata["config"]["scene"]
    if saved["pair"] != pair:
      raise ValueError("The transition checkpoint belongs to another skill pair")
    values = dict(saved)
    for name in ("source_checkpoint", "target_checkpoint", "selector_path"):
      values[name] = Path(values[name])
    for name, override in (
      ("selector_path", cfg.selector),
      ("source_checkpoint", cfg.source_checkpoint),
      ("target_checkpoint", cfg.target_checkpoint),
    ):
      if override is not None:
        values[name] = override
      if fingerprint(values[name]) != metadata["fingerprints"][name]:
        raise ValueError(f"{name} does not match the trained transition")
    if cfg.entry is not None:
      values["entry"] = cfg.entry
    if values["entry"] not in metadata["config"]["entries"]:
      raise ValueError("This reference entry was not used for transition training")
    if cfg.walk_speed is not None and not math.isclose(
      cfg.walk_speed, values["walk_speed"]
    ):
      raise ValueError("Walking command differs from transition training")
    if (
      cfg.seed_start < metadata["training_seed_end"]
      and cfg.seed_start + RUNS > metadata["config"]["seed"]
    ):
      raise ValueError("Scoring seeds overlap transition training")
    scene_cfg = SceneCfg(**values)
  else:
    scene_cfg = SceneCfg(
      pair=pair,
      entry=cfg.entry or 0,
      selector_path=cfg.selector or STATES_PATH,
      source_checkpoint=cfg.source_checkpoint,
      target_checkpoint=cfg.target_checkpoint,
      walk_speed=cfg.walk_speed if cfg.walk_speed is not None else 0.8,
    )
  scene_cfg = replace(
    scene_cfg, device=cfg.device, initial_joint_noise=0.0, initial_velocity_noise=0.0
  )
  scene = TransitionScene(
    scene_cfg,
    RUNS,
    bridge="diffusion" if method == "bridge" else "no-op",
    bridge_checkpoint=bridge_checkpoint,
    tracker_checkpoint=tracker_checkpoint,
  )
  try:
    if scene.target_skill != "walk":
      length = int(scene.run.motion.motion.time_step_total_per_motion.max())
      scene.cfg = replace(scene.cfg, post_s=length * scene.env.step_dt)
  except Exception:
    scene.close()
    raise
  return scene, policy, metadata


def report_header(
  scene: TransitionScene, cfg: SceneOptions, method: str, transition_checkpoint=None
):
  files = {
    **scene.checkpoints,
    **scene.comparison_checkpoints,
    "selector": str(scene.cfg.selector_path),
  }
  if transition_checkpoint is not None:
    files["transition"] = str(transition_checkpoint)
  return {
    "protocol": "absolute_skill_performance_v5",
    "status": "running",
    "pair": scene.cfg.pair,
    "incoming_skill": scene.target_skill,
    "method": method,
    "config": json.loads(json.dumps(asdict(cfg), default=str)),
    "scene": json.loads(json.dumps(asdict(scene.cfg), default=str)),
    "seeds": list(range(cfg.seed_start, cfg.seed_start + RUNS)),
    "metric_directions": METRICS[scene.target_skill],
    "failure_rule": "Failed errors use max(measurement, configured penalty); failed kick speed is zero",
    "artifacts": {
      name: {"path": str(Path(path).resolve()), "sha256": fingerprint(Path(path))}
      for name, path in files.items()
    },
    "cells": [],
  }


def write_json(path: Path, report: dict):
  temporary = path.with_suffix(".tmp")
  temporary.write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
  )
  for attempt in range(5):
    try:
      temporary.replace(path)
      break
    except PermissionError:
      if attempt == 4:
        raise
      # Windows readers can briefly hold the previous report open
      time.sleep(0.1)


def save(
  directory: Path, report: dict, on_update: Callable[[dict], None] | None = None
):
  write_json(directory / "report.json", report)
  trials, averages = [], []
  for cell in report["cells"]:
    context = {
      name: cell[name]
      for name in ("distance_m", "bridge_duration_s", "transition_limit_s")
    }
    average = {
      **context,
      **{
        name: cell["summary"][name]
        for name in ("runs", "successes", "failures", "success_rate")
      },
    }
    for name, value in cell["summary"]["mean"].items():
      average[f"mean_{name}"] = value
      average[f"std_{name}"] = cell["summary"]["std"][name]
    averages.append(average)
    for row in cell["trials"]:
      flat = {
        **context,
        **{
          name: row[name]
          for name in (
            "seed",
            "success",
            "handoff",
            "termination",
            "transition_s",
            "post_steps",
            "post_horizon_s",
            "contact",
          )
        },
      }
      flat["failure_reasons"] = ";".join(row["failure_reasons"])
      flat["source_walk_s"] = row.get("source_walk_s")
      for name in row["measurements"]:
        flat[f"raw_{name}"] = row["measurements"][name]
        flat[f"score_{name}"] = row["scores"][name]
      for name, value in row.get("action", {}).items():
        flat[f"action_{name}"] = value
      trials.append(flat)
  for filename, rows in (("trials.csv", trials), ("averages.csv", averages)):
    if rows:
      with (directory / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
  if on_update is not None:
    on_update(report)


@torch.inference_mode()
def run(pair: Pair, cfg: Config, on_update: Callable[[dict], None] | None = None):
  directory = cfg.output / pair / cfg.method
  directory.mkdir(parents=True, exist_ok=False)
  scene, policy, metadata = setup(
    pair,
    cfg,
    cfg.method,
    cfg.transition_checkpoint,
    cfg.bridge_checkpoint,
    cfg.tracker_checkpoint,
  )
  report = None
  try:
    seeds = tuple(range(cfg.seed_start, cfg.seed_start + RUNS))
    report = report_header(scene, cfg, cfg.method, cfg.transition_checkpoint)
    bridge = cfg.method == "bridge"
    kind = request_kind(scene, cfg.method)
    # Only the walk2kick transition asks at a distance, the one it was trained at
    approach = None
    if kind == "approach":
      approach = training_distance(metadata) if metadata is not None else None
      if approach is None:
        raise ValueError("The walk2kick transition needs its fixed training distance")
    # Every method gets one request, the bridge places its entry at each distance
    distances = (
      (DISTANCES if cfg.distance is None else (cfg.distance,)) if bridge else (None,)
    )
    durations = (
      (DURATIONS if cfg.duration is None else (cfg.duration,)) if bridge else (None,)
    )
    transition_steps = metadata["config"]["max_transition_steps"] if metadata else 100
    report["incoming_reference"] = placement(scene, cfg.method)
    report["request"] = kind
    report["request_trigger"] = TRIGGERS[kind]
    report["request_placement_m"] = approach
    report["distances_m"] = list(distances) if bridge else []
    report["bridge_durations_s"] = durations
    save(directory, report, on_update)
    # Shared by every method under this output that asks for the same request
    source_path = directory.parent / f"source_{kind}.pt"
    context: dict[str, Any] = {
      "seeds": seeds,
      "request": kind,
      "scene": {
        name: getattr(scene.cfg, name)
        for name in (
          "pair",
          "entry",
          "walk_speed",
          "placement_distance",
          "walk_distance",
          "source_timeout_s",
        )
      },
      "artifacts": {
        name: report["artifacts"][name]["sha256"]
        for name in (scene.source, scene.target_skill, "selector")
      },
    }
    if kind != "landing":
      context["walk_sampling"] = WALK_SAMPLING
    if approach is not None:
      context["distance_m"] = approach
    if source_path.exists():
      recording = torch.load(source_path, map_location=cfg.device, weights_only=True)
      if recording["context"] != context:
        raise ValueError(
          "Saved source uses different skills or sampling settings; choose another --output"
        )
      source = recording["request"]
    else:
      source = source_request(scene, seeds, kind, approach)
      torch.save({"context": context, "request": source}, source_path)
    torch.save(source, directory / "source.pt")
    report["source_recording_sha256"] = fingerprint(directory / "source.pt")
    if kind != "landing":
      report["walk_sampling"] = WALK_SAMPLING
      for name in ("source_walk_s", "source_phase_offset_s", "source_start_offset_m"):
        if name in source:
          report[name] = source[name].cpu().tolist()
    print(
      f"Scoring {RUNS} parallel runs with perturbation seeds {seeds[0]} through {seeds[-1]}",
      flush=True,
    )
    for distance in distances:
      for duration in durations:
        scene.reset(cfg.seed_start)
        scene.restore_request(source)
        torch.manual_seed(cfg.seed_start)
        rows = measure(
          scene,
          cfg.method,
          seeds,
          cfg.failure_errors,
          duration,
          policy,
          transition_steps,
          distance,
        )
        if kind != "landing":
          for row, elapsed in zip(
            rows, source["source_walk_s"].cpu().tolist(), strict=True
          ):
            row["source_walk_s"] = elapsed
        cell = {
          "distance_m": distance,
          "bridge_duration_s": duration,
          "transition_limit_s": transition_steps * scene.env.step_dt
          if cfg.method == "transition"
          else None,
          "summary": summarize(rows, scene.target_skill),
          "trials": rows,
        }
        report["cells"].append(cell)
        save(directory, report, on_update)
        print(
          json.dumps({key: cell[key] for key in cell if key != "trials"}), flush=True
        )
    report["status"] = "complete"
    save(directory, report, on_update)
  except Exception as error:
    if report is not None:
      report["status"], report["error"] = "failed", str(error)
      save(directory, report, on_update)
    raise
  finally:
    scene.close()


def main(pair: Pair):
  run(pair, tyro.cli(Config, config=mjlab.TYRO_FLAGS))
