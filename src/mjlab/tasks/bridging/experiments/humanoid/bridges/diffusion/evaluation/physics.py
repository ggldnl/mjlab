"""Measure UniTracker arrival error on held out diffusion plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.dataset.motions import (
  DEFAULT_MOTIONS,
  Windows,
  load_motions,
  motion_files,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.execution.tracker import (
  UniTrackerExecutor,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.planner.bridge import (
  DiffusionBridge,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  Tolerances,
  channel_errors,
  upper_body_mask,
)
from mjlab.tasks.unitracker.config.g1.env_cfgs import unitree_g1_unitracker_env_cfg
from mjlab.tasks.unitracker.mdp import MotionCommandCfg


@dataclass
class EvaluatePhysicsCfg:
  checkpoint: Path
  motions: tuple[str, ...] = DEFAULT_MOTIONS
  count: int = 16
  holdout: int = 8
  sample_steps: int | None = None
  device: str = "cuda:0"
  seed: int = 0


def robot_state(robot: Entity) -> torch.Tensor:
  data = robot.data
  return torch.cat(
    (
      data.root_link_pos_w,
      data.root_link_quat_w,
      data.root_link_lin_vel_w,
      data.root_link_ang_vel_w,
      data.joint_pos,
      data.joint_vel,
    ),
    dim=-1,
  )


@torch.no_grad()
def evaluate(cfg: EvaluatePhysicsCfg) -> torch.Tensor:
  if cfg.count < 1:
    raise ValueError("count must be positive")
  bridge = DiffusionBridge.load(cfg.checkpoint, cfg.device, cfg.sample_steps)
  corpus = load_motions(
    cfg.motions,
    bridge.process.denoiser.columns,
    cfg.device,
    "eval",
    cfg.holdout,
  )
  windows = Windows(
    corpus,
    bridge.history,
    bridge.future,
    bridge.min_steps,
    bridge.max_steps,
  )
  torch.manual_seed(cfg.seed)
  states, duration = windows.states(cfg.count)
  batch = torch.arange(cfg.count, device=cfg.device)[:, None]
  offsets = torch.arange(bridge.future, device=cfg.device)
  target_rows = bridge.history - 1 + duration[:, None] + offsets
  target = states[batch, target_rows]
  plans = bridge.generate(states[:, : bridge.history], target, duration).states

  env_cfg = unitree_g1_unitracker_env_cfg(play=True)
  motion = env_cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = str(motion_files(cfg.motions)[0])
  env_cfg.events = {}
  env_cfg.rewards = {}
  env_cfg.terminations = {}
  env_cfg.metrics = {}
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    robot: Entity = env.scene["robot"]
    executor = UniTrackerExecutor(env)
    action_term = env.action_manager.get_term("joint_pos")
    assert isinstance(action_term, BaseAction)

    def rollout(
      reference: torch.Tensor, start: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
      env.reset()
      joints = bridge.layout.joints
      robot.write_root_state_to_sim(start[None, :13])
      robot.write_joint_state_to_sim(
        start[None, 13 : 13 + joints], start[None, 13 + joints :]
      )
      # TextOp's action-history observation is zero at an episode reset.  A joint-space
      # hold target is not equivalent: after division by the small action scales it can
      # put values above 3 into a channel that the policy saw reset to zero in training.
      env.action_manager.initialize_action(torch.zeros_like(env.action_manager.action))
      env.sim.forward()
      for tick in range(reference.shape[0] - 1):
        rows = (tick + torch.arange(executor.horizon, device=cfg.device)).clamp_max(
          reference.shape[0] - 1
        )
        action = executor(robot_state(robot)[:, None], reference[rows][None])
        env.step(action)
      return channel_errors(
        robot_state(robot),
        goal[None],
        upper_body_mask(tuple(robot.joint_names), env.device),
      )[0]

    planned_errors: list[torch.Tensor] = []
    reference_errors: list[torch.Tensor] = []
    for index, steps in enumerate(duration.tolist()):
      start = states[index, bridge.history - 1]
      goal = target[index, 0]
      planned_errors.append(rollout(plans[index, : steps + 1], start, goal))
      demonstrated = states[index, bridge.history - 1 : bridge.history + steps]
      reference_errors.append(rollout(demonstrated, start, goal))

    planned = torch.stack(planned_errors)
    demonstrated = torch.stack(reference_errors)
    limits = Tolerances().tensor(env.device)
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

    def summarize(label: str, result: torch.Tensor) -> None:
      success = (result <= limits).all(-1)
      print(f"{label} strict terminal success: {int(success.sum())}/{cfg.count}")
      for column, name in enumerate(names):
        values = result[:, column]
        print(
          f"  {name}: mean={values.mean():.4f} "
          f"p90={torch.quantile(values, 0.9):.4f} limit={limits[column]:.4f}"
        )

    summarize("demonstrated path", demonstrated)
    summarize("generated path", planned)
    return planned
  finally:
    env.close()


if __name__ == "__main__":
  evaluate(tyro.cli(EvaluatePhysicsCfg, config=mjlab.TYRO_FLAGS))
