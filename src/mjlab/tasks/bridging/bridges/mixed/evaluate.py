"""Evaluate the footstep diffusion planner on held out windows.

Every window is generated three times, with footsteps from the heuristic planner,
footsteps read from the clip, and no footsteps. Metrics, averaged over windows:

    given     sole distance to the footsteps the model was given, in cm
    clip      sole distance to the clip's own footsteps, in cm
    slip      planted sole speed in xy, in cm/s
    accel     root mean square joint acceleration, in rad/s^2

Footsteps are only conditioning, so given is the check that they are used: with
clip footsteps, clip should fall well below its value without footsteps. Slip and
accel are also printed for the recorded clips, as the reference.

A second table compares the heuristic planner's steps with the clip's own steps
between A and B, grouped by root speed at A, with braking windows (B slower than
half of A) apart:

    steps      landings between A and B, clip and planner
    miss       mean absolute difference in landings per window
    interval   ticks between consecutive landings
    stride     distance between consecutive plants of the same foot, in cm

--generate False skips the diffusion model and prints only this table.

Run

1. Train a planner, see bridges.mixed.train.

2. Evaluate it.

    uv run python -m mjlab.tasks.bridging.bridges.mixed.evaluate --checkpoint logs/rsl_rl/g1_footstep_diffusion_planner/<run>/model_30000.pt
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.tasks.bridging.bridges.diffusion.config import motion_patterns
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  Windows,
  encode_pose,
  load_motions,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import GeneratedPath
from mjlab.tasks.bridging.bridges.mixed.bridge import FootstepBridge, FootstepPlan
from mjlab.tasks.bridging.bridges.mixed.footsteps import sole_frames


@dataclass
class EvaluateCfg:
  checkpoint: Path
  motions: tuple[str, ...] = ()
  holdout: int = 8
  windows: int = 256
  batch: int = 64
  sample_steps: int | None = None
  device: str = "cuda:0"
  seed: int = 0
  generate: bool = True


def slip(
  bridge: FootstepBridge, path: GeneratedPath, plan: FootstepPlan
) -> torch.Tensor:
  """Mean xy sole speed on planted frames between A and B, per window."""
  anchor = path.states[:, 0]
  position, _ = sole_frames(bridge.feet, encode_pose(path.states, anchor))
  speed = (position[:, 1:, :, :2] - position[:, :-1, :, :2]).norm(dim=-1) * bridge.fps
  start = bridge.history - 1
  frames = path.states.shape[1]
  planted = (
    plan.contact[:, start + 1 : start + frames]
    & plan.contact[:, start : start + frames - 1]
  )
  time = torch.arange(1, frames, device=speed.device)[None]
  active = (planted & (time <= path.duration[:, None])[..., None]).float()
  return (speed * active).sum((1, 2)) / active.sum((1, 2))


def acceleration(bridge: FootstepBridge, path: GeneratedPath) -> torch.Tensor:
  """Root mean square joint acceleration between A and B, per window."""
  joints = bridge.layout.joints
  position = path.states[..., 13 : 13 + joints]
  second = (position[:, 2:] - 2 * position[:, 1:-1] + position[:, :-2]) * bridge.fps**2
  time = torch.arange(1, position.shape[1] - 1, device=position.device)[None]
  active = (time < path.duration[:, None]).float()[..., None]
  return (second.square() * active).sum((1, 2)).div(active.sum((1, 2)) * joints).sqrt()


def step_statistics(
  plan: FootstepPlan, history: int, duration: torch.Tensor
) -> np.ndarray:
  """Landings, mean interval and mean stride between A and B, per window.

  NaN where a window has too few landings for an interval or a stride.
  """
  contact = plan.contact.cpu().numpy()
  places = plan.channels[..., 1:3].cpu().double().numpy()
  out = np.full((contact.shape[0], 3), np.nan)
  for row in range(contact.shape[0]):
    first = history
    last = history - 1 + int(duration[row])
    events = []
    for foot in range(2):
      flag = contact[row, :, foot]
      for tick in np.flatnonzero(flag[1:] & ~flag[:-1]) + 1:
        if first <= tick <= last:
          events.append((int(tick), foot, places[row, tick, foot]))
    events.sort(key=lambda event: event[0])
    out[row, 0] = len(events)
    if len(events) > 1:
      out[row, 1] = np.diff([event[0] for event in events]).mean()
    strides = []
    for foot in range(2):
      own = [event[2] for event in events if event[1] == foot]
      strides += [np.linalg.norm(b - a) for a, b in zip(own, own[1:], strict=False)]
    if strides:
      out[row, 2] = np.mean(strides)
  return out


def print_steps(
  clip: np.ndarray, planned: np.ndarray, start_speed: np.ndarray, end_speed: np.ndarray
) -> None:
  braking = (start_speed >= 0.5) & (end_speed < 0.5 * start_speed)
  groups = {
    "all": np.ones_like(braking),
    "A < 0.5": (start_speed < 0.5) & ~braking,
    "A 0.5-1.5": (start_speed >= 0.5) & (start_speed < 1.5) & ~braking,
    "A > 1.5": (start_speed >= 1.5) & ~braking,
    "braking": braking,
  }
  print(f"{'':11}{'windows':>8}{'steps':>14}{'miss':>7}{'interval':>14}{'stride':>14}")
  for name, picked in groups.items():
    if not picked.any():
      continue
    a, b = clip[picked], planned[picked]
    miss = np.abs(a[:, 0] - b[:, 0]).mean()

    pairs = [
      f"{np.nanmean(a[:, column]) * scale:7.1f}{np.nanmean(b[:, column]) * scale:7.1f}"
      for column, scale in ((0, 1.0), (1, 1.0), (2, 100.0))
    ]
    print(f"{name:11}{int(picked.sum()):8d}{pairs[0]}{miss:7.2f}{pairs[1]}{pairs[2]}")


def recorded(bridge: FootstepBridge, states: torch.Tensor, duration: torch.Tensor):
  start = bridge.history - 1
  return GeneratedPath(states[:, start:], duration)


@torch.no_grad()
def evaluate(cfg: EvaluateCfg) -> dict[str, dict[str, float]]:
  bridge = FootstepBridge.load(cfg.checkpoint, cfg.device, cfg.sample_steps)
  corpus = load_motions(
    cfg.motions or motion_patterns(bridge.robot, "val"),
    bridge.process.denoiser.columns,
    cfg.device,
    "all",
    cfg.holdout,
    robot=bridge.robot,
  )
  windows = Windows(
    corpus, bridge.history, bridge.future, bridge.min_steps, bridge.max_steps
  )
  torch.manual_seed(cfg.seed)
  totals: dict[str, dict[str, list[torch.Tensor]]] = {}
  steps: dict[str, list[np.ndarray]] = {"clip": [], "planner": [], "a": [], "b": []}

  def add(name: str, metric: str, value: torch.Tensor) -> None:
    totals.setdefault(name, {}).setdefault(metric, []).append(value)

  remaining = cfg.windows
  while remaining > 0:
    count = min(cfg.batch, remaining)
    remaining -= count
    states, duration = windows.states(count)
    history = states[:, : bridge.history]
    rows = bridge.history - 1 + duration
    offsets = torch.arange(bridge.future, device=states.device)
    target = states[
      torch.arange(count, device=states.device)[:, None], rows[:, None] + offsets
    ]
    anchor = history[:, -1]
    clip = bridge.clip_plan(states)
    heuristic = bridge.plan(history, target, duration)
    hidden = FootstepPlan(
      heuristic.contact, heuristic.channels, torch.zeros_like(heuristic.known)
    )
    steps["clip"].append(step_statistics(clip, bridge.history, duration))
    steps["planner"].append(step_statistics(heuristic, bridge.history, duration))
    speed_rows = torch.arange(count, device=states.device)
    steps["a"].append(anchor[:, 7:9].norm(dim=-1).cpu().numpy())
    steps["b"].append(states[speed_rows, rows, 7:9].norm(dim=-1).cpu().numpy())
    if not cfg.generate:
      continue
    reference = recorded(bridge, states, duration)
    add("recorded", "slip", slip(bridge, reference, clip))
    add("recorded", "accel", acceleration(bridge, reference))
    for name, plan in (("planner", heuristic), ("clip", clip), ("none", hidden)):
      path = bridge.generate(history, target, duration, plan=plan)
      if name != "none":
        add(name, "given", bridge.plant_error(path, plan, anchor))
      add(name, "clip", bridge.plant_error(path, clip, anchor))
      add(name, "slip", slip(bridge, path, plan if name != "none" else clip))
      add(name, "accel", acceleration(bridge, path))

  print_steps(*(np.concatenate(steps[name]) for name in ("clip", "planner", "a", "b")))
  if not cfg.generate:
    return {}
  scale = {"given": 100.0, "clip": 100.0, "slip": 100.0, "accel": 1.0}
  results = {
    name: {
      metric: float(torch.cat(values).nanmean()) * scale[metric]
      for metric, values in metrics.items()
    }
    for name, metrics in totals.items()
  }
  print(f"{'':10}{'given':>10}{'clip':>10}{'slip':>10}{'accel':>10}")
  for name, metrics in results.items():
    row = "".join(
      f"{metrics.get(metric, math.nan):10.2f}"
      for metric in ("given", "clip", "slip", "accel")
    )
    print(f"{name:10}{row}")
  return results


if __name__ == "__main__":
  evaluate(tyro.cli(EvaluateCfg, config=mjlab.TYRO_FLAGS))
