"""Mix imperfect diffusion paths into the universal tracker reference dataset.

Run

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.build_dataset
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DIFFUSION_REFERENCE_SOURCE,
  DIFFUSION_TRACKER_DATASET,
  TRACKER_DATASET,
  find_checkpoint,
  load_dataset,
  write,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.planner.bridge import (
  EXPERIMENT,
  DiffusionBridge,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.command import (
  STATE_HISTORY,
)


@dataclass(frozen=True)
class Config:
  base: Path = TRACKER_DATASET
  output: Path = DIFFUSION_TRACKER_DATASET
  planner_checkpoint: Path | None = None
  count: int = 50_000
  batch: int = 256
  holdout: int = 8
  sample_steps: int | None = None
  device: str = "cuda:0"
  seed: int = 0


def build(cfg: Config) -> Path:
  """Append generated references while preserving the recorded validation split."""
  if cfg.count < 1 or cfg.batch < 1 or cfg.holdout < 2:
    raise ValueError("count and batch must be positive; holdout must exceed one")
  torch.manual_seed(cfg.seed)
  planner_path = find_checkpoint(
    (EXPERIMENT,),
    str(cfg.planner_checkpoint) if cfg.planner_checkpoint is not None else None,
    hint=" Train the kinematic planner first.",
  )
  planner = DiffusionBridge.load(planner_path, cfg.device, cfg.sample_steps)
  data = load_dataset(cfg.base, cfg.device, "train", cfg.holdout)
  if abs(data.fps - planner.fps) > 1e-6:
    raise ValueError("planner and tracker datasets use different frame rates")
  windows = data.segments(
    planner.min_steps,
    planner.max_steps,
    before=planner.history - 1,
    after=planner.future - 1,
  )

  generated_states: list[np.ndarray] = []
  generated_envs: list[np.ndarray] = []
  generated_trajectories: list[np.ndarray] = []
  generated_frames: list[np.ndarray] = []
  made = 0
  with torch.no_grad():
    while made < cfg.count:
      count = min(cfg.batch, cfg.count - made)
      _, _, duration, positions = windows.draw(count)
      history_rows = windows.history(positions, planner.history - 1)
      offsets = torch.arange(planner.future, device=duration.device)
      target_rows = windows.order[
        positions[:, None] + duration[:, None] + offsets[None]
      ]
      paths = planner.generate(
        data.states[history_rows], data.states[target_rows], duration
      ).states
      routes = torch.cat(
        (data.states[history_rows[:, -STATE_HISTORY:]], paths[:, 1:]), dim=1
      )
      time = torch.arange(routes.shape[1], device=routes.device)[None]
      valid = time < duration[:, None] + planner.future + STATE_HISTORY - 1
      trajectory = torch.arange(made, made + count, device=paths.device)[:, None]
      env_id = trajectory.remainder(cfg.holdout - 1) + 1
      generated_states.append(routes[valid].cpu().numpy().astype(np.float32))
      generated_frames.append(
        time.expand(count, -1)[valid].cpu().numpy().astype(np.int32)
      )
      generated_trajectories.append(
        trajectory.expand_as(time.expand(count, -1))[valid]
        .cpu()
        .numpy()
        .astype(np.int32)
      )
      generated_envs.append(
        env_id.expand_as(time.expand(count, -1))[valid].cpu().numpy().astype(np.int16)
      )
      made += count
      print(f"[diffusion tracker data] {made}/{cfg.count}")

  with np.load(cfg.base, allow_pickle=False) as raw:
    base_states = np.asarray(raw["states"], dtype=np.float32)
    base_env = np.asarray(raw["env_id"], dtype=np.int16)
    base_frame = np.asarray(raw["frame"], dtype=np.int32)
    base_skill = np.asarray(raw["skill"])
    base_trajectory = np.asarray(raw["trajectory"])
    names = tuple(str(name) for name in raw["skill_names"])
    fps = float(raw["fps"])
  pairs = np.stack((base_skill, base_trajectory), axis=1)
  _, base_global = np.unique(pairs, axis=0, return_inverse=True)
  generated = np.concatenate(generated_states)
  generated_count = generated.shape[0]
  generated_skill = np.full(generated_count, len(names), dtype=base_skill.dtype)
  generated_global = np.concatenate(generated_trajectories) + base_global.max() + 1
  return write(
    cfg.output,
    states=[base_states, generated],
    env_ids=[base_env, np.concatenate(generated_envs)],
    trajectory_ids=[base_global.astype(np.int32), generated_global],
    frames=[base_frame, np.concatenate(generated_frames)],
    sources=[base_skill, generated_skill],
    names=(*names, DIFFUSION_REFERENCE_SOURCE),
    fps=fps,
    trajectory_ids_global=True,
  )


if __name__ == "__main__":
  build(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
