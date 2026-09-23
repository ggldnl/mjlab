"""Train the diffusion bridge on physically executed tracker windows.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.train
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  LOG_ROOT,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.bridge import (
  EXPERIMENT,
  checkpoint_metadata,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.data import (
  Normalizer,
  Windows,
  training_mask,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.model import (
  Denoiser,
  ModelCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.process import (
  Diffusion,
  ProcessCfg,
)


@dataclass
class TrainCfg:
  dataset: Path = DEFAULT_DATASET
  output: Path = LOG_ROOT / EXPERIMENT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  history: int = 4
  future: int = 4
  min_steps: int = 15
  max_steps: int = 60
  model: ModelCfg = field(default_factory=ModelCfg)
  process: ProcessCfg = field(default_factory=ProcessCfg)
  batch: int = 256
  iterations: int = 30_000
  learning_rate: float = 2e-4
  ema_decay: float = 0.995
  fit_batches: int = 32
  bridge_mask_probability: float = 0.5
  log_every: int = 100
  save_every: int = 2_000
  device: str = "cuda:0"
  seed: int = 0


def train(cfg: TrainCfg) -> Path:
  if cfg.batch < 1 or cfg.iterations < 1 or cfg.fit_batches < 1:
    raise ValueError("batch, iterations and fit_batches must be positive")
  if not 0 <= cfg.ema_decay < 1:
    raise ValueError("ema_decay must be in [0, 1)")
  torch.manual_seed(cfg.seed)
  data = load_dataset(cfg.dataset, cfg.device, "train")
  windows = Windows(data, cfg.history, cfg.future, cfg.min_steps, cfg.max_steps)
  print(
    f"[diffusion] {windows.segments.starts.numel()} training windows at {data.fps:g} Hz"
  )
  with torch.no_grad():
    samples = torch.cat([windows.sample(cfg.batch)[0] for _ in range(cfg.fit_batches)])
    normalizer = Normalizer.fit(samples)
  del samples
  model = Denoiser(windows.layout.width, windows.columns, cfg.model).to(cfg.device)
  process = Diffusion(model, cfg.process).to(cfg.device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
  ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
  cfg.output.mkdir(parents=True, exist_ok=True)

  def save(iteration: int) -> Path:
    checkpoint = cfg.output / f"model_{iteration}.pt"
    torch.save(
      checkpoint_metadata(
        cfg.model,
        cfg.process,
        windows.layout,
        normalizer,
        cfg.history,
        cfg.future,
        cfg.min_steps,
        cfg.max_steps,
        data.fps,
        ema,
        iteration,
      ),
      checkpoint,
    )
    return checkpoint

  running = 0.0
  for iteration in range(1, cfg.iterations + 1):
    features, duration = windows.sample(cfg.batch)
    clean = normalizer.normalize(features)
    known = training_mask(
      windows.layout,
      windows.columns,
      cfg.history,
      cfg.future,
      duration,
      cfg.bridge_mask_probability,
    )
    loss = process.loss(clean, known)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    with torch.no_grad():
      for name, value in model.state_dict().items():
        if value.is_floating_point():
          ema[name].lerp_(value, 1 - cfg.ema_decay)
        else:
          ema[name].copy_(value)
    running += loss.item()
    if iteration % cfg.log_every == 0:
      print(
        f"[diffusion] {iteration}/{cfg.iterations} loss {running / cfg.log_every:.5f}"
      )
      running = 0.0
    if iteration % cfg.save_every == 0 and iteration < cfg.iterations:
      print(f"[diffusion] saved {save(iteration)}")
  checkpoint = save(cfg.iterations)
  print(f"[diffusion] saved {checkpoint}")
  return checkpoint


if __name__ == "__main__":
  train(tyro.cli(TrainCfg, config=mjlab.TYRO_FLAGS))
