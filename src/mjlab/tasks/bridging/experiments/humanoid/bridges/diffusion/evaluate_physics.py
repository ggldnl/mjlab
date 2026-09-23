"""Check planner boundaries and the recorded-action physics baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.bridge import (
  DiffusionBridge,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation import (
  IMITATION_TASK_ID,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  Tolerances,
  channel_errors,
  upper_body_mask,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.resume import restore_action
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import arena, state


@dataclass
class EvalCfg:
  checkpoint: Path
  dataset: Path = DEFAULT_DATASET
  batch: int = 32
  duration: int = 40
  seed: int = 0
  device: str = "cuda:0"
  sample_steps: int | None = None


def summarize(name: str, errors: torch.Tensor, tolerances: torch.Tensor) -> None:
  success = (errors <= tolerances).all(dim=-1)
  names = (
    "root_pos_m",
    "root_ori_rad",
    "root_lin_vel_mps",
    "root_ang_vel_radps",
    "lower_joint_pos_rad",
    "lower_joint_vel_radps",
    "upper_joint_pos_rad",
    "upper_joint_vel_radps",
  )
  print(f"{name}: strict terminal success {int(success.sum())}/{len(success)}")
  for index, channel in enumerate(names):
    values = errors[:, index]
    print(
      f"  {channel}: mean={values.mean():.3f} "
      f"p90={torch.quantile(values, 0.9):.3f} "
      f"limit={tolerances[index]:.3f}"
    )


@torch.no_grad()
def evaluate(cfg: EvalCfg) -> None:
  if cfg.batch < 1:
    raise ValueError("batch must be positive")
  torch.manual_seed(cfg.seed)
  bridge = DiffusionBridge.load(cfg.checkpoint, cfg.device, cfg.sample_steps)
  if not bridge.min_steps <= cfg.duration <= bridge.max_steps:
    raise ValueError("duration outside checkpoint training range")
  dataset = load_dataset(cfg.dataset, cfg.device, "eval")
  if dataset.previous_action is None:
    raise ValueError("dataset needs previous_action")
  if dataset.num_joints != bridge.layout.joints or dataset.fps != bridge.fps:
    raise ValueError("dataset and checkpoint must have the same robot and control rate")

  columns = bridge.history + cfg.duration + bridge.future - 1
  segments = dataset.segments(columns - 1, columns - 1)
  picked = torch.randint(segments.starts.numel(), (cfg.batch,), device=cfg.device)
  offsets = torch.arange(columns, device=cfg.device)
  rows = segments.order[segments.starts[picked, None] + offsets]
  recorded = dataset.states[rows]
  actions = dataset.previous_action[rows]

  env_cfg = arena(IMITATION_TASK_ID, WALK_TASK_ID, KICK_TASK_ID)
  env_cfg.scene.num_envs = cfg.batch
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    env.reset()
    origins = env.scene.env_origins
    history = recorded[:, : bridge.history]
    target_start = bridge.history - 1 + cfg.duration
    target_sequence = recorded[:, target_start : target_start + bridge.future]
    target = target_sequence[:, 0]
    history[:, :, :3] += origins[:, None]
    target_sequence[:, :, :3] += origins[:, None]
    target = target_sequence[:, 0]
    duration = torch.full(
      (cfg.batch,), cfg.duration, device=cfg.device, dtype=torch.long
    )
    path = bridge.generate(history, target_sequence, duration)
    if not torch.equal(path.states[:, cfg.duration], target):
      raise AssertionError("planner did not preserve the target boundary")

    robot: Entity = env.scene["robot"]
    start = history[:, -1]
    joints = dataset.num_joints
    robot.write_root_state_to_sim(start[:, :13])
    robot.write_joint_state_to_sim(start[:, 13 : 13 + joints], start[:, 13 + joints :])
    restore_action(env, actions[:, bridge.history - 1])
    env.sim.forward()
    print(f"[physics] {cfg.batch} recorded replays, {cfg.duration} ticks")

    for step in range(cfg.duration):
      env.step(actions[:, bridge.history + step])
    errors = channel_errors(
      state(env), target, upper_body_mask(tuple(robot.joint_names), env.device)
    )
    limits = Tolerances().tensor(env.device)
    summarize("recorded replay", errors, limits)
  finally:
    env.close()


if __name__ == "__main__":
  evaluate(tyro.cli(EvalCfg, config=mjlab.TYRO_FLAGS))
