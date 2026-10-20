"""Compare a dedicated transition, direct switch and bridges on matched trials."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.baseline.metrics import (
  matched_comparisons,
  quality,
  summarize,
)
from mjlab.tasks.bridging.baseline.model import (
  TransitionPolicy,
  fingerprint,
  training_distance,
)
from mjlab.tasks.bridging.baseline.scene import SceneCfg, TransitionScene

Method = Literal["transition", "direct", "diffusion", "mixed", "imitation", "cvae"]


def load_for_scene(
  checkpoint: Path, cfg: SceneCfg
) -> tuple[TransitionPolicy, SceneCfg, dict]:
  policy, metadata = TransitionPolicy.load(checkpoint, cfg.device)
  saved = metadata["config"]["scene"]
  changes = {}
  for name in ("source_checkpoint", "target_checkpoint", "selector_path"):
    path = getattr(cfg, name)
    if path is None:
      changes[name] = Path(saved[name])
    elif Path(path).resolve() != Path(saved[name]).resolve():
      raise ValueError(f"{name} differs from the trained baseline")
  cfg = replace(cfg, **changes)
  if cfg.entry not in metadata["config"].get("entries", [saved["entry"]]):
    raise ValueError("Requested entry was not included in baseline training")
  for name, value in asdict(cfg).items():
    if (
      name
      not in (
        "device",
        "entry",
        "source_checkpoint",
        "target_checkpoint",
        "selector_path",
      )
      and value != saved[name]
    ):
      raise ValueError(
        f"scene.{name} differs from training: {value!r} != {saved[name]!r}"
      )
  for name, digest in metadata["fingerprints"].items():
    if fingerprint(Path(getattr(cfg, name))) != digest:
      raise ValueError(f"{name} contents changed since training")
  return policy, cfg, metadata


@dataclass(frozen=True)
class EvalCfg:
  checkpoint: Path
  scene: SceneCfg = field(default_factory=SceneCfg)
  methods: tuple[Method, ...] = ("transition", "direct")
  distances: tuple[float, ...] | None = None
  durations: tuple[float, ...] = (2.0,)
  seeds: tuple[int, ...] = tuple(range(10000, 10030))
  bridge_checkpoint: Path | None = None
  tracker_checkpoint: Path | None = None
  output: Path = Path("data/baseline/results.json")

  def __post_init__(self) -> None:
    if not self.methods or len(set(self.methods)) != len(self.methods):
      raise ValueError("methods must be nonempty and unique")
    if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
      raise ValueError("seeds must be nonempty, nonnegative and unique")
    for values in (self.distances, self.durations):
      if values is None:
        continue
      if (
        not values
        or len(set(values)) != len(values)
        or any(not math.isfinite(value) or value <= 0 for value in values)
      ):
        raise ValueError("distances and durations must be unique, finite and positive")


@torch.no_grad()
def trial(
  scene: TransitionScene,
  method: str,
  seed: int,
  distance: float,
  duration: float,
  policy: TransitionPolicy | None = None,
  request: dict | None = None,
) -> dict:
  scene.reset(seed)
  scene.interrupt(distance)
  if request is not None:
    scene.restore_request(request)
  scene.last_request = scene.capture_request()
  actual_distance = scene.run.distance
  elapsed = 0
  handoff = method in ("nominal", "direct")
  reason = (
    "nominal" if method == "nominal" else "direct" if method == "direct" else "timeout"
  )
  if method == "nominal":
    scene.nominal()
  elif method != "direct":
    if method == "transition":
      scene.hold_target()
      scene.run.command.stop()
    else:
      scene.open_bridge(duration)
    budget = max(1, round(duration / scene.env.step_dt))
    for tick in range(budget + 1):
      if bool(scene.failed.any()):
        reason = "robot_failed"
        break
      if method == "transition":
        if policy is None:
          raise ValueError("transition requires a policy")
        scene.hold_target()
        action, _, terminal = policy.act(scene.observe())
        if bool(terminal.all()) or tick == budget - 1:
          handoff = True
          reason = "learned_termination" if bool(terminal.all()) else "transition_limit"
          break
      else:
        action, handoff = scene.bridge_action(duration)
      if handoff:
        reason = "bridge_termination"
        break
      if tick == budget:
        break
      scene.step(action)
      elapsed += 1
  errors, within = scene.endpoint()
  if handoff:
    rms, success, action_jump = scene.successor()
  else:
    rms, success, action_jump = None, None, None
  return {
    "method": method,
    "seed": seed,
    "entry": scene.cfg.entry,
    "distance": distance,
    "actual_distance": actual_distance,
    "request_state": scene.last_request["state"][0].cpu().tolist(),
    "requested_target": scene.last_request["target"][0].cpu().tolist(),
    "budget_s": duration,
    "duration_s": elapsed * scene.env.step_dt,
    "handoff": handoff,
    "termination": reason,
    "success": bool(success[0]) if success is not None else False,
    "rms": rms[0].cpu().tolist() if rms is not None else None,
    "endpoint_errors": errors[0].cpu().tolist(),
    "within_endpoint_box": bool(within[0]),
    "action_jump": float(action_jump[0]) if action_jump is not None else None,
    "action_jump_max": float(scene.handoff_delta[0].abs().max()) if handoff else None,
    "action_jump_raw": float(scene.handoff_jump_raw[0]) if handoff else None,
    "joint_command_delta_rad": scene.handoff_delta[0].cpu().tolist()
    if handoff
    else None,
    "joint_names": scene.handoff_joint_names if handoff else None,
    "post_steps": scene.post_steps if handoff else 0,
  }


def evaluate(cfg: EvalCfg) -> dict:
  if cfg.output.exists() or cfg.output.with_suffix(".csv").exists():
    raise FileExistsError(f"Refusing to replace results at {cfg.output}")
  policy, scene_cfg, metadata = load_for_scene(cfg.checkpoint, cfg.scene)
  # Keep the evaluation seeds separate from every training and bootstrap episode
  train_cfg = metadata["config"]
  distance = training_distance(metadata)
  distances = cfg.distances or (
    (distance,) if distance is not None else tuple(train_cfg["distances"])
  )
  if (
    distance is not None
    and "transition" in cfg.methods
    and any(not math.isclose(value, distance) for value in distances)
  ):
    raise ValueError("Transition evaluation must use the fixed training distance")
  used = range(
    train_cfg["seed"],
    metadata.get(
      "training_seed_end",
      train_cfg["seed"] + train_cfg["bootstrap_batches"] + train_cfg["iterations"],
    ),
  )
  if any(seed in used for seed in cfg.seeds):
    raise ValueError("Evaluation seeds overlap training seeds")
  nominal = {}
  requests = {}
  rows: list[dict] = []
  groups = []
  comparison_artifacts = {}
  scene = TransitionScene(scene_cfg)
  try:
    channels, floors = scene.channels, scene.floors
    for distance in distances:
      for seed in cfg.seeds:
        row = trial(scene, "nominal", seed, distance, 0.0)
        if not row["success"]:
          raise RuntimeError(
            f"Nominal skill failed for seed {seed}, distance {distance}; inspect the entry and checkpoint before comparing quality"
          )
        nominal[(distance, seed)] = row
        requests[(distance, seed)] = scene.last_request
  finally:
    scene.close()
  for method in cfg.methods:
    bridge = "no-op" if method in ("transition", "direct") else method
    scene = TransitionScene(
      scene_cfg,
      bridge=bridge,
      bridge_checkpoint=cfg.bridge_checkpoint,
      tracker_checkpoint=cfg.tracker_checkpoint,
    )
    try:
      comparison_artifacts[method] = {
        name: {"path": path, "sha256": fingerprint(Path(path))}
        for name, path in scene.comparison_checkpoints.items()
      }
      for distance in distances:
        for duration in cfg.durations:
          cell = []
          for seed in cfg.seeds:
            row = trial(
              scene,
              method,
              seed,
              distance,
              duration,
              policy,
              requests[(distance, seed)],
            )
            reference = nominal[(distance, seed)]
            row["nominal_rms"] = reference["rms"]
            row["quality"] = (
              float(
                quality(
                  torch.tensor([row["rms"]], device=scene_cfg.device),
                  torch.tensor([reference["rms"]], device=scene_cfg.device),
                  floors,
                  torch.tensor([row["success"]], device=scene_cfg.device),
                )[0]
              )
              if row["handoff"]
              else None
            )
            rows.append(row)
            cell.append(row)
          group = {
            "method": method,
            "distance": distance,
            "budget_s": duration,
            **summarize(cell),
          }
          groups.append(group)
          print(json.dumps(group), flush=True)
    finally:
      scene.close()
  paired = matched_comparisons(rows)
  report = {
    "config": json.loads(json.dumps(asdict(cfg), default=str)),
    "baseline": metadata,
    "baseline_checkpoint": {
      "path": str(cfg.checkpoint.resolve()),
      "sha256": fingerprint(cfg.checkpoint),
    },
    "comparison_artifacts": comparison_artifacts,
    "channels": channels,
    "floors": floors.cpu().tolist(),
    "protocol_version": 3,
    "protocol": "Learned transition termination with upstream forced cap; native bridge termination within budget; matched successful metrics",
    "handoff_discontinuity_unit": "rad",
    "target_error_channels": list(scene.run.command.error_names),
    "nominal": list(nominal.values()),
    "trials": rows,
    "groups": groups,
    "overall": {
      method: summarize([row for row in rows if row["method"] == method])
      for method in cfg.methods
    },
    "paired": paired,
    "reachable": [group for group in groups if group["success_rate"] == 1.0],
    "reach": [
      {
        "method": method,
        "budget_s": duration,
        "max_successful_distance": max(
          (
            group["distance"]
            for group in groups
            if group["method"] == method
            and group["budget_s"] == duration
            and group["success_rate"] == 1.0
          ),
          default=None,
        ),
      }
      for method in cfg.methods
      for duration in cfg.durations
    ],
  }
  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  cfg.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
  with cfg.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  return report


def main() -> None:
  cfg = tyro.cli(EvalCfg, config=mjlab.TYRO_FLAGS)
  evaluate(cfg)
  print(cfg.output)


if __name__ == "__main__":
  main()
