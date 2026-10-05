"""Train the footstep conditioned diffusion planner on retargeted motion clips.

Windows are sampled and augmented exactly as for bridges.diffusion.planner.train.
Footsteps are then read from the augmented pose, so mirrored and time warped
windows get matching footsteps. Each window shows the model a random part of its
footsteps, see MaskCfg in footsteps.py. Shown footsteps get a small random offset
and timing jitter, so the model tolerates the heuristic planner's footsteps.

The footstep planner's tables (stride, step interval, swing time and landing lead
per root speed, steady and braking) and the planted height are fitted once before
training, printed, and saved in the checkpoint.

Run

1. Build the motion dataset as described in the diffusion README.

2. Train.

    uv run python -m mjlab.tasks.bridging.bridges.mixed.train --robot g1

Checkpoints are written to logs/rsl_rl/g1_footstep_diffusion_planner/<run>/.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.bridges.diffusion.config import motion_patterns
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  Normalizer,
  Windows,
  load_motions,
  pose_features,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.model import Denoiser, ModelCfg
from mjlab.tasks.bridging.bridges.diffusion.planner.process import (
  Diffusion,
  ProcessCfg,
  RobotFootKinematics,
)
from mjlab.tasks.bridging.bridges.mixed.bridge import (
  PlantLoss,
  checkpoint_metadata,
  planner_experiment,
)
from mjlab.tasks.bridging.bridges.mixed.footsteps import (
  ContactCfg,
  FootstepLayout,
  MaskCfg,
  detect_contacts,
  fit_gait,
  footstep_channels,
  jitter_timing,
  perturb_places,
  sole_frames,
  training_mask,
)
from mjlab.tasks.bridging.config import RobotAlias

LOG_ROOT = Path("logs") / "rsl_rl"


@dataclass
class TrainCfg:
  robot: RobotAlias = "g1"
  motions: tuple[str, ...] = ()
  output: Path | None = None
  history: int = 4
  future: int = 1
  min_steps: int = 15
  max_steps: int = 60
  time_scale_range: tuple[float, float] = (0.8, 1.25)
  start_xy_range: float = 0.01
  start_perturb_probability: float = 0.5
  mirror_probability: float = 0.5
  holdout: int = 8
  model: ModelCfg = field(default_factory=ModelCfg)
  process: ProcessCfg = field(default_factory=ProcessCfg)
  contact: ContactCfg = field(default_factory=ContactCfg)
  mask: MaskCfg = field(default_factory=MaskCfg)
  endpoint_weight: float = 1.0
  foot_slip_weight: float = 1.0
  plant_weight: float = 0.5
  batch: int = 256
  iterations: int = 30_000
  learning_rate: float = 2e-4
  ema_decay: float = 0.995
  fit_batches: int = 32
  log_every: int = 100
  save_every: int = 2_000
  device: str = "cuda:0"
  seed: int = 0


class FootstepWindows:
  """Planner windows with footstep channels appended."""

  def __init__(
    self, windows: Windows, feet: RobotFootKinematics, contact: ContactCfg
  ) -> None:
    self.windows = windows
    self.feet = feet
    self.contact = contact
    self.layout = FootstepLayout(windows.layout.joints)
    self.columns = windows.columns
    self.fps = windows.data.fps

  @torch.no_grad()
  def sample(
    self, count: int, mask: MaskCfg | None = None
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Features, durations and the planted flags the footstep channels use.

    With a MaskCfg the footsteps get its timing jitter and place noise.
    """
    pose, duration = self.windows.sample_pose(count)
    position, yaw = sole_frames(self.feet, pose)
    contact = detect_contacts(position, self.fps, self.contact)
    if mask is not None:
      anchor = self.windows.history - 1
      contact = jitter_timing(contact, anchor, mask.timing_jitter_probability)
    channels = footstep_channels(position, yaw, contact)
    if mask is not None:
      channels = perturb_places(channels, contact, mask.position_noise, mask.yaw_noise)
    features = torch.cat((pose_features(pose), channels.flatten(-2)), dim=-1)
    return features, duration, contact


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
  feet = RobotFootKinematics(cfg.robot).to(cfg.device)
  windows = FootstepWindows(
    Windows(
      corpus,
      cfg.history,
      cfg.future,
      cfg.min_steps,
      cfg.max_steps,
      cfg.time_scale_range,
      cfg.start_xy_range,
      cfg.start_perturb_probability,
      cfg.mirror_probability,
    ),
    feet,
    cfg.contact,
  )
  layout = windows.layout
  print(
    f"[footstep] {corpus.num_windows} windows from {len(corpus.names)} "
    f"kinematic clips at {corpus.fps:g} Hz"
  )
  samples = [windows.sample(cfg.batch) for _ in range(cfg.fit_batches)]
  features = torch.cat([sample[0] for sample in samples])
  normalizer = Normalizer.fit(features)
  contact = torch.cat([sample[2] for sample in samples])
  gait = fit_gait(
    contact,
    features[..., layout.feet].unflatten(-1, (2, -1)),
    features[..., layout.root_position][..., :2],
    corpus.fps,
  )
  print(
    f"[footstep] gait: step {gait.step_length:.3f} m, swing "
    f"{gait.swing_ticks:.1f} ticks, planted height {gait.stance_height:.3f} m"
  )
  for name in ("interval", "swing", "lead"):
    steady, braking = getattr(gait, name)
    print(f"[footstep] {name:8} by speed {gait.speeds}")
    print(f"[footstep]   steady  {tuple(round(value, 2) for value in steady)}")
    print(f"[footstep]   braking {tuple(round(value, 2) for value in braking)}")
  del samples, features, contact

  model = Denoiser(layout.width, windows.columns, cfg.model).to(cfg.device)
  process = Diffusion(model, cfg.process, layout).to(cfg.device)
  path_loss = PlantLoss(
    cfg.robot,
    normalizer,
    layout,
    cfg.history,
    corpus.fps,
    cfg.endpoint_weight,
    cfg.foot_slip_weight,
    cfg.contact.height,
    cfg.contact.speed,
    cfg.plant_weight,
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
        layout,
        normalizer,
        cfg.history,
        cfg.future,
        cfg.min_steps,
        cfg.max_steps,
        corpus.fps,
        ema,
        iteration,
        cfg.robot,
        gait,
        cfg.contact,
      ),
      checkpoint,
    )
    return checkpoint

  running = 0.0
  for iteration in range(1, cfg.iterations + 1):
    features, duration, contact = windows.sample(cfg.batch, cfg.mask)
    clean = normalizer.normalize(features)
    known = training_mask(
      layout,
      windows.columns,
      cfg.history,
      cfg.future,
      duration,
      contact,
      cfg.mask,
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
        f"[footstep] {iteration}/{cfg.iterations} loss {running / cfg.log_every:.5f}"
      )
      running = 0.0
    if iteration % cfg.save_every == 0 and iteration < cfg.iterations:
      print(f"[footstep] saved {save(iteration)}")
  checkpoint = save(cfg.iterations)
  print(f"[footstep] saved {checkpoint}")
  return checkpoint


if __name__ == "__main__":
  train(tyro.cli(TrainCfg, config=mjlab.TYRO_FLAGS))
