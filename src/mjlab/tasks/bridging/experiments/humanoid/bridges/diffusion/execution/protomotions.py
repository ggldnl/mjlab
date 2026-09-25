"""ProtoMotions G1 BONES deployment adapter.

Download the pinned checkpoint:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.execution.protomotions
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast
from urllib.request import urlretrieve

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.unitracker.config.g1.env_cfgs import unitree_g1_unitracker_env_cfg
from mjlab.tasks.unitracker.mdp import MotionCommandCfg
from mjlab.utils.lab_api.math import quat_inv, quat_mul, yaw_quat

_REVISION = "b30eae3c7258292f3e5a7b60ecd8c9d3317e3ed3"
CHECKPOINT_URL = (
  "https://media.githubusercontent.com/media/NVlabs/ProtoMotions/"
  f"{_REVISION}/data/pretrained_models/motion_tracker/g1-bones-deploy/"
  "compiled_models/unified_pipeline.onnx"
)
DEFAULT_CHECKPOINT = Path("data/protomotions/g1_bones_unified_pipeline.onnx")
CHECKPOINT_SHA256 = "a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c"

JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
FUTURE_STEPS = (1, 2, 4, 8)


def download_checkpoint(path: Path = DEFAULT_CHECKPOINT) -> Path:
  """Download and verify the pinned official ONNX checkpoint."""
  path.parent.mkdir(parents=True, exist_ok=True)
  urlretrieve(CHECKPOINT_URL, path)
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  if digest != CHECKPOINT_SHA256:
    path.unlink()
    raise ValueError(f"ProtoMotions checkpoint hash mismatch: {digest}")
  return path


def protomotions_g1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the mjlab G1 environment with the BONES deployment control contract."""
  cfg = unitree_g1_unitracker_env_cfg(play=play)
  cfg.sim.mujoco.timestep = 0.001
  cfg.decimation = 20

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = JOINT_NAMES
  action.scale = 1.0
  action.offset = 0.0
  action.use_default_offset = False

  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.command_joint_names = JOINT_NAMES
  return cfg


def _xyzw(quaternion: torch.Tensor) -> np.ndarray:
  value = torch.cat((quaternion[..., 1:], quaternion[..., :1]), dim=-1)
  return np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)


class ProtoMotionsPolicy:
  """Run the official G1 BONES unified ONNX policy on an mjlab motion command."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    checkpoint: Path = DEFAULT_CHECKPOINT,
  ) -> None:
    import onnxruntime as ort

    if env.num_envs != 1:
      raise ValueError("The deployment checkpoint supports one mjlab environment")
    if not checkpoint.is_file():
      raise FileNotFoundError(
        f"Missing {checkpoint}. Run this module to download the checkpoint."
      )
    self.env = env
    self.robot = env.scene["robot"]
    self.command = cast(MotionCommand, env.command_manager.get_term("motion"))
    self.session = ort.InferenceSession(
      str(checkpoint), providers=["CPUExecutionProvider"]
    )
    self.output_names = [value.name for value in self.session.get_outputs()]
    expected_inputs = {
      "current_anchor_rot",
      "current_dof_pos",
      "current_dof_vel",
      "current_root_local_ang_vel",
      "historical_processed_actions",
      "mimic_future_anchor_rot",
      "mimic_future_dof_pos",
      "mimic_future_dof_vel",
    }
    if {value.name for value in self.session.get_inputs()} != expected_inputs:
      raise ValueError("Unexpected ProtoMotions ONNX input contract")

    joint_ids, _ = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
    self.joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    action_term = env.action_manager.get_term("joint_pos")
    if not isinstance(action_term, BaseAction):
      raise TypeError("ProtoMotions needs a joint action term")
    target_names = tuple(action_term.target_names)
    if set(target_names) != set(JOINT_NAMES):
      raise ValueError("Environment action joints do not match ProtoMotions")
    self.action_from_policy = torch.tensor(
      [JOINT_NAMES.index(name) for name in target_names], device=env.device
    )
    self.torso_id = self.robot.body_names.index("torso_link")
    self.motion_torso_id = self.command.cfg.body_names.index("torso_link")
    self.offsets = torch.tensor(FUTURE_STEPS, device=env.device)
    self.previous_targets = np.zeros((1, 1, len(JOINT_NAMES)), dtype=np.float32)
    self.heading_offset = torch.zeros((1, 4), device=env.device)
    self.heading_offset[:, 0] = 1.0
    self.last_step = -1

  def reset(self) -> None:
    actual = self.robot.data.body_link_quat_w[:, self.torso_id]
    reference = self.command.motion.body_quat_w[0, self.motion_torso_id][None]
    self.heading_offset = quat_mul(yaw_quat(actual), quat_inv(yaw_quat(reference)))
    self.previous_targets.fill(0.0)
    self.last_step = int(self.command.time_steps[0])

  def reference(self) -> torch.Tensor:
    """Return the current full state in the common evaluator format."""
    step = self.command.time_steps
    motion = self.command.motion
    return torch.cat(
      (
        motion.body_pos_w[step, 0] + self.env.scene.env_origins,
        motion.body_quat_w[step, 0],
        motion.body_lin_vel_w[step, 0],
        motion.body_ang_vel_w[step, 0],
        motion.joint_pos[step],
        motion.joint_vel[step],
      ),
      dim=-1,
    )[:, None]

  @torch.no_grad()
  def __call__(self, observation: torch.Tensor) -> torch.Tensor:
    del observation
    step = int(self.command.time_steps[0])
    if step < self.last_step:
      self.reset()
    self.last_step = step

    rows = (self.command.time_steps[:, None] + self.offsets).clamp_max(
      self.command.motion.time_step_total - 1
    )
    future_quat = self.command.motion.body_quat_w[rows, self.motion_torso_id]
    future_quat = quat_mul(
      self.heading_offset[:, None].expand_as(future_quat), future_quat
    )
    future_pos = self.command.motion.joint_pos[rows][..., self.joint_ids]
    future_vel = self.command.motion.joint_vel[rows][..., self.joint_ids]
    inputs = {
      "current_anchor_rot": _xyzw(self.robot.data.body_link_quat_w[:, self.torso_id]),
      "current_dof_pos": np.ascontiguousarray(
        self.robot.data.joint_pos[:, self.joint_ids].cpu().numpy(), dtype=np.float32
      ),
      "current_dof_vel": np.ascontiguousarray(
        self.robot.data.joint_vel[:, self.joint_ids].cpu().numpy(), dtype=np.float32
      ),
      "current_root_local_ang_vel": np.ascontiguousarray(
        self.robot.data.root_link_ang_vel_b.cpu().numpy(), dtype=np.float32
      ),
      "historical_processed_actions": self.previous_targets,
      "mimic_future_anchor_rot": _xyzw(future_quat),
      "mimic_future_dof_pos": np.ascontiguousarray(
        future_pos.cpu().numpy(), dtype=np.float32
      ),
      "mimic_future_dof_vel": np.ascontiguousarray(
        future_vel.cpu().numpy(), dtype=np.float32
      ),
    }
    outputs = dict(
      zip(self.output_names, self.session.run(self.output_names, inputs), strict=True)
    )
    targets = cast(np.ndarray, outputs["joint_pos_targets"])
    self.previous_targets = targets[:, None].copy()
    action = torch.as_tensor(targets, device=self.env.device)
    return action[:, self.action_from_policy]


if __name__ == "__main__":
  print(download_checkpoint())
