"""Fit the denoiser on the corpus. Offline, no simulator.

One iteration draws a batch of recorded windows, corrupts them to random noise levels and
asks the model for the windows they came from. Nothing about a start, a target or a
deadline appears anywhere in here, which is the point: what is being learned is what a
second of G1 motion looks like, and a crossing is a constraint applied to that afterwards.
Read the loss as a reconstruction error in normalized feature units.

The averaged weights are what gets deployed. A diffusion model's exponential moving average
is reliably better than the iterate it came from, by enough that sampling from the raw
iterate is a common way to conclude that a model failed to train when it did not.

Run

1. Build the corpus first. It is shared and lives one level up.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker

2. Train. With no arguments this reads the training split of the default corpus and writes
   checkpoints under logs/rsl_rl/g1_diffusion_bridge.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.train

3. Train on part of the corpus, to see what one clip family is worth.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.train \
      --sources "('walk1_subject1', 'jumps1_subject1')"

4. Score the result against the do nothing baseline and against the other architectures.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.evaluate \
      --bridge diffusion

5. Watch it cross.

    uv run play Mjlab-G1-Diffusion-Bridge
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  LOG_ROOT,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.data import (
  Normalizer,
  Windows,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.diffusion import (
  Process,
  ProcessCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.model import (
  Denoiser,
  ModelCfg,
)

EXPERIMENT = "g1_diffusion_bridge"
"""Log directory, beside every other architecture's runs. Not a reinforcement learning run,
and kept under the same root anyway so find_checkpoint and `uv run play` locate it without
a second convention."""


@dataclass
class TrainCfg:
  """One offline training run."""

  dataset: Path = DEFAULT_DATASET
  split: str = "train"
  sources: tuple[str, ...] | None = None
  """Clips to train on. None is all of them, which is what a general bridge wants."""

  history: int = 4
  """Ticks of past the model is conditioned on, 0.08 s at 50 Hz. One frame does not say
  which phase of a stride the robot is in, and two situations needing opposite strategies
  then look identical."""

  horizon: int = 60
  """Ticks predicted ahead, 1.2 s at 50 Hz. Matches the longest window
  BridgeCommandCfg.duration_s_range draws, so a target is always inside the horizon and
  guidance never has to aim at a column that does not exist."""

  model: ModelCfg = field(default_factory=ModelCfg)
  process: ProcessCfg = field(default_factory=ProcessCfg)

  batch: int = 256
  iterations: int = 60_000
  learning_rate: float = 2.0e-4
  ema: float = 0.995
  """Decay of the averaged weights. At this rate the average trails by about 200 steps."""

  ema_warmup: int = 1_000
  """Iterations before the average starts tracking. Averaging from step zero drags the
  deployed weights back toward the initialization for thousands of iterations."""

  fit_batches: int = 40
  """Batches drawn to measure the feature mean and deviation before training starts."""

  log_every: int = 200
  save_every: int = 5_000
  device: str = "cuda:0"
  seed: int = 0
  run: str | None = None
  """Name of the run directory. None stamps the current time."""


class Average:
  """Exponential moving average of the denoiser weights."""

  def __init__(self, model: torch.nn.Module, decay: float) -> None:
    self.decay = decay
    self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

  def update(self, model: torch.nn.Module, active: bool) -> None:
    for key, value in model.state_dict().items():
      held = self.shadow[key]
      if not active or not held.is_floating_point():
        held.copy_(value)
      else:
        held.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

  def state_dict(self) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in self.shadow.items()}


def train(cfg: TrainCfg) -> Path:
  """Fit one model and return the directory its checkpoints went to."""
  torch.manual_seed(cfg.seed)

  data = load_dataset(cfg.dataset, cfg.device, cfg.split)
  if data.previous_action is None:
    raise SystemExit(
      f"{cfg.dataset} holds no action column, and this architecture diffuses states and "
      f"actions together. Rebuild the corpus with `uv run python -m mjlab.tasks.bridging"
      f".experiments.humanoid.bridges.dataset.tracker`."
    )
  windows = Windows(data, cfg.history, cfg.horizon, cfg.sources)
  layout = windows.layout
  print(
    f"[diffusion] {len(windows)} windows of {windows.columns} ticks, "
    f"{layout.width} features, {layout.num_bodies} bodies"
  )

  sample = torch.cat([windows.sample(cfg.batch) for _ in range(cfg.fit_batches)])
  normalizer = Normalizer.fit(sample)
  del sample

  denoiser = Denoiser(layout.width, windows.columns, cfg.model).to(cfg.device)
  process = Process(denoiser, cfg.process).to(cfg.device)
  parameters = sum(p.numel() for p in denoiser.parameters())
  print(f"[diffusion] {parameters / 1e6:.1f}M parameters")

  optimizer = torch.optim.Adam(denoiser.parameters(), lr=cfg.learning_rate)
  average = Average(denoiser, cfg.ema)

  stamp = cfg.run or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  out = LOG_ROOT / EXPERIMENT / stamp
  out.mkdir(parents=True, exist_ok=True)
  print(f"[diffusion] writing to {out}")

  def save(iteration: int) -> Path:
    path = out / f"model_{iteration}.pt"
    torch.save(
      {
        "ema": average.state_dict(),
        "model": asdict(cfg.model),
        "process": asdict(cfg.process),
        "layout": asdict(layout),
        "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
        "history": cfg.history,
        "horizon": cfg.horizon,
        "fps": data.fps,
        "sources": data.names,
        "iteration": iteration,
      },
      path,
    )
    return path

  running = 0.0
  for iteration in range(1, cfg.iterations + 1):
    batch = normalizer(windows.sample(cfg.batch))
    loss = process.loss(batch, cfg.history)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
    optimizer.step()
    average.update(denoiser, iteration > cfg.ema_warmup)

    running += float(loss.detach())
    if iteration % cfg.log_every == 0:
      print(
        f"[diffusion] {iteration}/{cfg.iterations} loss {running / cfg.log_every:.5f}"
      )
      running = 0.0
    if iteration % cfg.save_every == 0 and iteration != cfg.iterations:
      print(f"[diffusion] saved {save(iteration)}")

  print(f"[diffusion] saved {save(cfg.iterations)}")
  return out


if __name__ == "__main__":
  train(tyro.cli(TrainCfg, config=mjlab.TYRO_FLAGS))
