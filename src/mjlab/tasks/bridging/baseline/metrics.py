"""Numerical protocol from thesis chapters 6 and 7."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import math

import numpy as np
import torch

TRACKING_CHANNELS = (
  "error_anchor_pos",
  "error_anchor_rot",
  "error_body_pos",
  "error_body_rot",
  "error_joint_pos",
  "error_joint_vel",
)
TRACKING_FLOORS = (0.01, 0.02, 0.01, 0.02, 0.05, 0.1)
WALK_CHANNELS = ("forward_velocity", "lateral_velocity", "yaw_rate")
WALK_FLOORS = (0.05, 0.05, 0.05)


def quality(
  errors: torch.Tensor,
  nominal: torch.Tensor,
  floors: torch.Tensor,
  success: torch.Tensor,
) -> torch.Tensor:
  if errors.shape != nominal.shape or errors.shape[-1] != floors.numel():
    raise ValueError("Quality channels must match")
  if not bool(torch.isfinite(nominal).all()) or bool((nominal < 0).any()):
    raise ValueError("Nominal errors must be finite and nonnegative")
  ratio = errors.clamp_min(floors) / nominal.clamp_min(floors)
  valid = success & torch.isfinite(errors).all(-1)
  return torch.where(valid, ratio.mean(-1).reciprocal(), 0.0)


def wilson(successes: int, trials: int) -> tuple[float, float]:
  if not 0 <= successes <= trials or trials < 1:
    raise ValueError("Need 0 <= successes <= trials and trials > 0")
  z = 1.959963984540054
  p = successes / trials
  denominator = 1 + z * z / trials
  center = (p + z * z / (2 * trials)) / denominator
  radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
  return max(0.0, center - radius), min(1.0, center + radius)


def summarize(rows: list[dict]) -> dict:
  valid = [row for row in rows if row["success"] and row.get("handoff", True)]
  successes = len(valid)
  scores = np.asarray([row["quality"] for row in valid], dtype=float)
  return {
    "trials": len(rows),
    "metric_trials": successes,
    "success_rate": successes / len(rows) if rows else None,
    "success_rate_ci95": wilson(successes, len(rows)) if rows else None,
    "quality_mean": float(scores.mean()) if valid else None,
    "quality_std": float(scores.std(ddof=1))
    if len(valid) > 1
    else 0.0
    if valid
    else None,
    "endpoint_rate": float(np.mean([row["within_endpoint_box"] for row in valid]))
    if valid
    else None,
    "duration_mean_s": float(np.mean([row["duration_s"] for row in valid]))
    if valid
    else None,
    "action_jump_mean_rad": float(np.mean([row["action_jump"] for row in valid]))
    if valid
    else None,
    "action_jump_max_mean_rad": float(
      np.mean([row["action_jump_max"] for row in valid])
    )
    if valid
    else None,
    "target_error_mean": np.mean(
      [row["endpoint_errors"] for row in valid], axis=0
    ).tolist()
    if valid
    else None,
    "downstream_rms_mean": np.mean([row["rms"] for row in valid], axis=0).tolist()
    if valid
    else None,
  }


def matched_comparisons(rows: list[dict]) -> list[dict]:
  """Compare only identical test cases in which both methods succeed."""

  def key(row: dict) -> tuple:
    return row["entry"], row["distance"], row["budget_s"], row["seed"]

  baseline = {key(row): row for row in rows if row["method"] == "transition"}
  if not baseline:
    return []
  reports = []
  for method in dict.fromkeys(
    row["method"] for row in rows if row["method"] != "transition"
  ):
    other = {key(row): row for row in rows if row["method"] == method}
    if baseline.keys() != other.keys():
      raise ValueError("Methods must have identical test cases for paired comparison")
    cells = list(dict.fromkeys(item[:3] for item in baseline))
    for cell in [None, *cells]:
      keys = [item for item in baseline if cell is None or item[:3] == cell]
      selected = [
        item
        for item in keys
        if all(
          row["success"] and row["handoff"] for row in (baseline[item], other[item])
        )
      ]
      left = [baseline[item] for item in selected]
      right = [other[item] for item in selected]
      for first, second in zip(left, right, strict=True):
        if first["post_steps"] != second["post_steps"]:
          raise ValueError("Matched executions must have the same downstream horizon")
        for field in ("request_state", "requested_target"):
          if (
            field in first
            and field in second
            and not np.allclose(first[field], second[field], rtol=1e-5, atol=1e-5)
          ):
            raise ValueError(f"Matched executions have different {field}")
      reports.append(
        {
          "method": method,
          "scope": "overall" if cell is None else "cell",
          "entry": cell[0] if cell else None,
          "distance": cell[1] if cell else None,
          "budget_s": cell[2] if cell else None,
          "attempted_pairs": len(keys),
          "matched_successes": len(selected),
          "retained_fraction": len(selected) / len(keys),
          "retained_seeds": [item[-1] for item in selected],
          "transition": summarize(left),
          "comparison": summarize(right),
          "quality_minus_transition": float(
            np.mean(
              [
                second["quality"] - first["quality"]
                for first, second in zip(left, right, strict=True)
              ]
            )
          )
          if selected
          else None,
          "handoff_jump_rad_minus_transition": float(
            np.mean(
              [
                second["action_jump"] - first["action_jump"]
                for first, second in zip(left, right, strict=True)
              ]
            )
          )
          if selected
          else None,
          "target_errors_minus_transition": np.mean(
            [
              np.asarray(second["endpoint_errors"])
              - np.asarray(first["endpoint_errors"])
              for first, second in zip(left, right, strict=True)
            ],
            axis=0,
          ).tolist()
          if selected
          else None,
        }
      )
  return reports
