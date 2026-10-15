"""Train the kinematic diffusion planner on retargeted motion clips.

Run:

    uv run python -m \
      mjlab.tasks.bridging.bridges.diffusion.planner.train
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.bridges.diffusion.config import (
  motion_patterns,
  planner_experiment,
)
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  Normalizer,
  Windows,
  bridge_mask,
  load_motions,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import (
  checkpoint_metadata,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.model import (
  Denoiser,
  ModelCfg,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.process import (
  Diffusion,
  FootChannels,
  PathLoss,
  ProcessCfg,
)
from mjlab.tasks.bridging.config import RobotAlias

LOG_ROOT = Path("logs") / "rsl_rl"


@dataclass
class TrainCfg:
  robot: RobotAlias = "g1"
  motions: tuple[str, ...] = ()
  output: Path | None = None
  history: int = 50
  """Frames up to A, one gait cycle at 50 Hz"""
  future: int = 25
  """Frames from B on, one step at 50 Hz"""
  min_steps: int = 15
  max_steps: int = 60
  time_scale_range: tuple[float, float] = (0.8, 1.25)
  start_xy_range: float = 0.01
  start_perturb_probability: float = 0.5
  mirror_probability: float = 0.5
  foot_channels: bool = True
  """Contacts and soles on the boundary frames, see PlannerLayout"""
  history_noise: float = 0.01
  """Joint position noise before A in rad, a stand in for simulated history"""
  history_noise_probability: float = 0.5
  holdout: int = 8
  model: ModelCfg = field(default_factory=ModelCfg)
  process: ProcessCfg = field(default_factory=ProcessCfg)
  endpoint_weight: float = 1.0
  foot_slip_weight: float = 1.0
  foot_contact_height: float = 0.05
  foot_contact_speed: float = 0.2
  contact_weight: float = 1.0
  batch: int = 256
  iterations: int = 30_000
  learning_rate: float = 2e-4
  ema_decay: float = 0.995
  fit_batches: int = 32
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
  motions = cfg.motions or motion_patterns(cfg.robot, "train")
  output = cfg.output or (
    LOG_ROOT
    / planner_experiment(cfg.robot)
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  columns = cfg.history + cfg.max_steps + cfg.future - 1
  corpus = load_motions(
    motions, columns, cfg.device, "all", cfg.holdout, robot=cfg.robot
  )
  feet = (
    FootChannels(
      cfg.robot, corpus.fps, cfg.foot_contact_height, cfg.foot_contact_speed
    ).to(cfg.device)
    if cfg.foot_channels
    else None
  )
  windows = Windows(
    corpus,
    cfg.history,
    cfg.future,
    cfg.min_steps,
    cfg.max_steps,
    cfg.time_scale_range,
    cfg.start_xy_range,
    cfg.start_perturb_probability,
    cfg.mirror_probability,
    feet,
    cfg.history_noise,
    cfg.history_noise_probability,
  )
  print(
    f"[diffusion] {windows.starts.numel()} windows from {len(corpus.names)} "
    f"kinematic clips at {corpus.fps:g} Hz"
  )
  with torch.no_grad():
    samples = torch.cat([windows.sample(cfg.batch)[0] for _ in range(cfg.fit_batches)])
    normalizer = Normalizer.fit(samples)
  del samples

  model = Denoiser(windows.layout.width, windows.columns, cfg.model).to(cfg.device)
  process = Diffusion(model, cfg.process, windows.layout).to(cfg.device)
  path_loss = PathLoss(
    cfg.robot,
    normalizer,
    windows.layout,
    cfg.history,
    corpus.fps,
    cfg.endpoint_weight,
    cfg.foot_slip_weight,
    cfg.foot_contact_height,
    cfg.foot_contact_speed,
    cfg.contact_weight,
  ).to(cfg.device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
  ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
  output.mkdir(parents=True, exist_ok=True)

  def save(iteration: int) -> Path:
    checkpoint = output / f"model_{iteration}.pt"
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
        corpus.fps,
        ema,
        iteration,
        cfg.robot,
        cfg.foot_contact_height,
        cfg.foot_contact_speed,
      ),
      checkpoint,
    )
    return checkpoint

  running = 0.0
  for iteration in range(1, cfg.iterations + 1):
    features, duration = windows.sample(cfg.batch)
    clean = normalizer.normalize(features)
    known = bridge_mask(
      cfg.batch, windows.columns, windows.layout, cfg.history, cfg.future, duration
    )
    rows = cfg.history - 1 + duration
    time = torch.arange(windows.columns, device=cfg.device)[None]
    valid = time <= rows[:, None] + cfg.future - 1
    loss = process.loss(clean, known, valid, path_loss, duration)
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
