"""Physical measurements, with explicit failure values and no normalization."""

# pyright: reportPrivateImportUsage=false

import math
from dataclasses import dataclass

import torch

RUNS = 50
DISTANCES = tuple(round(i / 10, 1) for i in range(2, 11))
DURATIONS = tuple(round(i / 10, 1) for i in range(4, 13, 2))
METRICS = {
  "walk": {"velocity_error_m_s": "lower", "yaw_rate_error_rad_s": "lower"},
  "jump": {"body_position_error_m": "lower"},
  "kick": {"forward_ball_speed_m_s": "higher"},
}
# Joint position target change per control step, in radians, see action_discontinuity
ACTION_METRICS = (
  "seam_rad",
  "request_jump_rad",
  "handoff_jump_rad",
  "handoff_joint_jump_rad",
  "peak_step_rad",
  "mean_step_rad",
)


@dataclass(frozen=True)
class FailureErrors:
  velocity_error_m_s: float = 2.0
  yaw_rate_error_rad_s: float = 2.0
  body_position_error_m: float = 1.0

  def __post_init__(self):
    if any(not math.isfinite(value) or value <= 0 for value in vars(self).values()):
      raise ValueError("Failure errors must be finite and positive")


def walk(
  linear_velocity: torch.Tensor, yaw_rate: torch.Tensor, desired_velocity: torch.Tensor
) -> dict[str, torch.Tensor]:
  return {
    "velocity_error_m_s": (linear_velocity[:, :2] - desired_velocity[:, :2]).norm(
      dim=-1
    ),
    "yaw_rate_error_rad_s": (yaw_rate - desired_velocity[:, 2]).abs(),
  }


def jump(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
  return {"body_position_error_m": (actual - reference).norm(dim=-1).mean(-1)}


def kick(
  velocity: torch.Tensor, direction: torch.Tensor, touched: torch.Tensor
) -> torch.Tensor:
  speed = (velocity[:, :2] * direction[:, :2]).sum(-1).clamp_min(0)
  return torch.where(touched, speed, 0.0)


def action_steps(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """RMS and largest joint target change per step, from (envs, steps + 1, joints)."""
  change = targets.diff(dim=1)
  return change.square().mean(-1).sqrt(), change.abs().amax(-1)


def action_discontinuity(
  rms: torch.Tensor, peak: torch.Tensor, steps: int, handoff: int | None
) -> dict[str, float | None]:
  """Joint target changes of one trial, from the request to the end of its measurement.

  rms and peak come from action_steps. Step 0 is the first action after the request,
  steps the trial's own steps and handoff the first incoming step, None without one.
  Seams are the request and the handoff, the same step for an instant switch. The
  values are described in scoring.discontinuity.
  """
  result: dict[str, float | None] = dict.fromkeys(ACTION_METRICS)
  if steps == 0:
    return result
  if steps > len(rms) or (handoff is not None and not 0 <= handoff < steps):
    raise ValueError("Trial steps exceed the recorded actions")
  own = rms[:steps]
  result["request_jump_rad"] = result["seam_rad"] = float(own[0])
  result["peak_step_rad"] = float(own.max())
  result["mean_step_rad"] = float(own.mean())
  if handoff is not None:
    result["handoff_jump_rad"] = float(own[handoff])
    result["handoff_joint_jump_rad"] = float(peak[handoff])
    result["seam_rad"] = max(float(own[0]), float(own[handoff]))
  return result


def failure_scores(
  skill: str, measured: dict, success: bool, errors: FailureErrors
) -> dict[str, float]:
  result = {}
  for name, direction in METRICS[skill].items():
    value = measured[name]
    if value is not None and (not math.isfinite(value) or value < 0):
      raise ValueError(f"Invalid measurement: {name}={value}")
    if success:
      if value is None:
        raise ValueError("A successful execution must have all measurements")
      result[name] = value
    elif direction == "higher":
      result[name] = 0.0
    else:
      result[name] = max(getattr(errors, name), value or 0.0)
  return result


def summarize(rows: list[dict], skill: str) -> dict:
  if len(rows) != RUNS or len({row["seed"] for row in rows}) != RUNS:
    raise ValueError("Each reported setting needs exactly 50 distinct runs")
  means, observed, counts, deviations = {}, {}, {}, {}
  for name in METRICS[skill]:
    values = [row["scores"][name] for row in rows]
    mean = sum(values) / RUNS
    means[name] = mean
    deviations[name] = math.sqrt(
      sum((value - mean) ** 2 for value in values) / (RUNS - 1)
    )
    raw = [
      row["measurements"][name] for row in rows if row["measurements"][name] is not None
    ]
    observed[name] = sum(raw) / len(raw) if raw else None
    counts[name] = len(raw)
  failures = {}
  for row in rows:
    for reason in row["failure_reasons"]:
      failures[reason] = failures.get(reason, 0) + 1
  return {
    "runs": RUNS,
    "successes": sum(row["success"] for row in rows),
    "failures": sum(not row["success"] for row in rows),
    "success_rate": sum(row["success"] for row in rows) / RUNS,
    "mean": means,
    "std": deviations,
    "observed_mean": observed,
    "observed_runs": counts,
    "failure_reasons": failures,
  }
