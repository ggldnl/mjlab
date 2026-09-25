"""Deploy the learned universal tracker on a generated diffusion path."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, cast

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
  find_checkpoint,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker import (
  TRACKER_EXPERIMENT,
  TRACKER_TASK_ID,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.command import (
  FUTURE_OFFSETS,
  STATE_HISTORY,
  TrackerCommandCfg,
  reference_features,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.env_cfg import (
  COMMAND,
  tracker_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  Tolerances,
  channel_errors,
  upper_body_mask,
)
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse

Policy = Callable[[TensorDict], torch.Tensor]


class LearnedTrackerExecutor:
  """Build the training observation online and convert residuals to joint actions."""

  horizon = max(FUTURE_OFFSETS) + 1
  history = STATE_HISTORY

  def __init__(self, env: ManagerBasedRlEnv, policy: Policy) -> None:
    self.env = env
    self.robot = env.scene["robot"]
    self.policy = policy
    self.fps = 1.0 / env.step_dt
    self.offsets = torch.tensor(FUTURE_OFFSETS, device=env.device)
    self.upper_body = upper_body_mask(tuple(self.robot.joint_names), env.device)
    self.tolerances = Tolerances().tensor(env.device)
    self.path: torch.Tensor | None = None
    self.duration: torch.Tensor | None = None
    self.index: torch.Tensor | None = None
    self._runner: MjlabOnPolicyRunner | None = None

    action = env.action_manager.get_term("joint_pos")
    if not isinstance(action, BaseAction):
      raise TypeError("Learned tracker requires a joint action term")
    self.action_term = action
    self.joint_ids = action.target_ids
    self.residual = torch.zeros(env.num_envs, action.action_dim, device=env.device)
    self.previous_residual = torch.zeros_like(self.residual)
    self.previous_previous_residual = torch.zeros_like(self.residual)

  @classmethod
  def load(
    cls,
    env: ManagerBasedRlEnv,
    checkpoint: Path | None = None,
  ) -> LearnedTrackerExecutor:
    """Load the actor and its observation normalizer from an RSL-RL checkpoint."""
    path = find_checkpoint(
      (TRACKER_EXPERIMENT,),
      str(checkpoint) if checkpoint is not None else None,
      hint=f" Train one with `uv run train {TRACKER_TASK_ID}`.",
    )
    cfg = tracker_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cast(TrackerCommandCfg, cfg.commands[COMMAND]).dataset_path = None
    helper = ManagerBasedRlEnv(cfg, device=env.device)
    wrapped = RslRlVecEnvWrapper(helper, clip_actions=None)
    try:
      agent_cfg = load_rl_cfg(TRACKER_TASK_ID)
      runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=env.device)
      runner.load(
        str(path), load_cfg={"actor": True}, strict=True, map_location=env.device
      )
      policy = runner.get_inference_policy(device=env.device)
      executor = cls(env, policy)
      executor._runner = runner
      print(f"tracker  {path}")
      return executor
    finally:
      wrapped.close()

  def reset(self, done: torch.Tensor | None = None) -> None:
    ids: torch.Tensor | slice = slice(None) if done is None else done
    self.residual[ids] = 0.0
    self.previous_residual[ids] = 0.0
    self.previous_previous_residual[ids] = 0.0
    if done is None:
      self.path = None
      self.duration = None
      self.index = None

  def set_plan(
    self, path: torch.Tensor, duration: torch.Tensor, index: torch.Tensor
  ) -> None:
    """Receive the complete generated path and current tick from DiffusionRuntime."""
    self.path = path
    self.duration = duration
    self.index = index

  def _history(self, history: torch.Tensor) -> torch.Tensor:
    if history.shape[1] >= STATE_HISTORY:
      return history[:, -STATE_HISTORY:]
    padding = history[:, :1].expand(-1, STATE_HISTORY - history.shape[1], -1)
    return torch.cat((padding, history), dim=1)

  def _observation(self, history: torch.Tensor) -> torch.Tensor:
    if self.path is None or self.duration is None or self.index is None:
      raise RuntimeError("DiffusionRuntime has not supplied a generated path")
    states = self._history(history)
    current = states[:, -1]
    batch = torch.arange(current.shape[0], device=current.device)[:, None]
    rows = (self.index[:, None] + self.offsets[None]).clamp(max=self.path.shape[1] - 1)
    future = self.path[batch, rows]
    endpoint = self.path[
      torch.arange(current.shape[0], device=current.device), self.duration
    ]
    remaining = (self.duration - self.index).clamp(min=0)
    future_time = self.offsets[None].expand(current.shape[0], -1).float() / self.fps
    phase = (self.index.float() / self.duration.float()).clamp(0.0, 1.0)
    path = torch.cat(
      (
        reference_features(current, future).flatten(1),
        future_time,
        reference_features(current, endpoint),
        phase[:, None],
        (remaining.float() / self.fps)[:, None],
      ),
      dim=-1,
    )

    joints = self.robot.data.joint_pos.shape[1]
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + joints)
    qd = slice(q.stop, q.stop + joints)
    quaternion = states[..., 3:7].contiguous()
    gravity_w = states.new_zeros((*states.shape[:2], 3))
    gravity_w[..., 2] = -1.0
    default = self.robot.data.default_joint_pos
    default_velocity = self.robot.data.default_joint_vel
    assert default is not None and default_velocity is not None
    root_height = states[..., 2:3] - self.env.scene.env_origins[:, None, 2:3]
    return torch.cat(
      (
        path,
        root_height.flatten(1),
        quat_apply_inverse(quaternion, states[..., 7:10]).flatten(1),
        quat_apply_inverse(quaternion, states[..., 10:13]).flatten(1),
        quat_apply_inverse(quaternion, gravity_w).flatten(1),
        (states[..., q] - default[:, None]).flatten(1),
        (states[..., qd] - default_velocity[:, None]).flatten(1),
        self.residual,
        self.previous_residual,
        self.previous_previous_residual,
      ),
      dim=-1,
    )

  def __call__(self, history: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    del reference
    observation = self._observation(history)
    residual = self.policy(
      TensorDict({"actor": observation}, batch_size=[history.shape[0]])
    )
    self.previous_previous_residual[:] = self.previous_residual
    self.previous_residual[:] = self.residual
    self.residual[:] = residual

    assert (
      self.path is not None and self.duration is not None and self.index is not None
    )
    next_row = (self.index + 1).clamp(max=self.path.shape[1] - 1)
    reference_q = self.path[
      torch.arange(history.shape[0], device=history.device),
      next_row,
      ROOT_STATE_DIM : ROOT_STATE_DIM + self.robot.data.joint_pos.shape[1],
    ][:, self.joint_ids]
    scale = self.action_term.scale
    offset = self.action_term.offset
    desired = reference_q + residual * scale
    return (desired - offset) / scale

  def captured(self, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.shape[1] < 1:
      raise ValueError("target must contain B")
    errors = channel_errors(history[:, -1], target[:, 0], self.upper_body)
    return (errors <= self.tolerances).all(-1)
