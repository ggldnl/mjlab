"""Pilot alternative physical continuations from shared start states.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.branch \
      --checkpoint logs/rsl_rl/g1_cvae_oracle/<run>/model_5000.pt

This collector never pairs endpoints from separate rollouts. Each retained branch is
executed by the oracle and records its own measured endpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import tyro
from tensordict import TensorDict

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.collect import (
  ORACLE_BRANCH_DATASET,
  _numpy,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle import (
  ORACLE_TASK_ID,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle.command import (
  MotionSetCommand,
  MotionSetCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle.runner import (
  OracleRunner,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  control_rate,
  state,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.build import (
  canonical,
  features,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)


class BranchMotionCommand(MotionSetCommand):
  """Rigidly align each reference continuation to a shared robot start."""

  def __init__(self, cfg: BranchMotionCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.reference_rotation = torch.zeros(self.num_envs, 4, device=self.device)
    self.reference_rotation[:, 0] = 1.0
    self.reference_translation = env.scene.env_origins.clone()

  def set_references(
    self,
    env_ids: torch.Tensor,
    phases: torch.Tensor,
    motion_ids: torch.Tensor,
    root: torch.Tensor,
  ) -> None:
    reference_pos = self.motion.body_pos_w[phases, 0]
    reference_quat = self.motion.body_quat_w[phases, 0]
    rotation = quat_mul(
      yaw_quat(root[:, 3:7]), quat_conjugate(yaw_quat(reference_quat))
    )
    self.reference_rotation[env_ids] = rotation
    self.reference_translation[env_ids] = root[:, :3] - quat_apply(
      rotation, reference_pos
    )
    self.motion_ids[env_ids] = motion_ids
    self.time_steps[env_ids] = phases
    self.motion_end_steps[env_ids] = self.motion.motion_ends[motion_ids]

  def _place(self, value: torch.Tensor, kind: str) -> torch.Tensor:
    extra = value.ndim - 2
    rotation = self.reference_rotation.reshape(
      self.num_envs, *(1 for _ in range(extra)), 4
    )
    rotation = rotation.expand(*value.shape[:-1], 4)
    if kind == "pos":
      offset = self.reference_translation.reshape(
        self.num_envs, *(1 for _ in range(extra)), 3
      )
      return offset + quat_apply(rotation, value)
    if kind == "quat":
      return quat_mul(rotation, value)
    return quat_apply(rotation, value)

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._place(self.motion.body_pos_w[self.time_steps], "pos")

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._place(self.motion.body_quat_w[self.time_steps], "quat")

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._place(self.motion.body_lin_vel_w[self.time_steps], "vel")

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._place(self.motion.body_ang_vel_w[self.time_steps], "vel")

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  @property
  def future_body_pos_w(self) -> torch.Tensor:
    return self._place(self.motion.body_pos_w[self.future_time_steps], "pos")

  @property
  def future_body_quat_w(self) -> torch.Tensor:
    return self._place(self.motion.body_quat_w[self.future_time_steps], "quat")

  @property
  def future_body_lin_vel_w(self) -> torch.Tensor:
    return self._place(self.motion.body_lin_vel_w[self.future_time_steps], "vel")

  @property
  def future_body_ang_vel_w(self) -> torch.Tensor:
    return self._place(self.motion.body_ang_vel_w[self.future_time_steps], "vel")


@dataclass(kw_only=True)
class BranchMotionCommandCfg(MotionSetCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> BranchMotionCommand:
    return BranchMotionCommand(self, env)


@dataclass
class BranchCfg:
  checkpoint: Path
  path: Path = ORACLE_BRANCH_DATASET
  groups: int = 100
  num_branches: int = 8
  horizon_steps: int = 40
  settle_steps: int = 15
  max_match_distance: float = 10.0
  max_root_error: float = 0.25
  max_foot_error: float = 0.25
  min_endpoint_separation: float = 0.10
  min_action_separation: float = 0.02
  device: str = "cuda:0"


def _candidate_phases(
  command: BranchMotionCommand,
  start: torch.Tensor,
  count: int,
  horizon: int,
  limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  motion = command.motion
  reference = torch.cat(
    (
      motion.body_pos_w[:, 0],
      motion.body_quat_w[:, 0],
      motion.body_lin_vel_w[:, 0],
      motion.body_ang_vel_w[:, 0],
      motion.joint_pos,
      motion.joint_vel,
    ),
    dim=-1,
  )
  target = features(canonical(start))
  distance = torch.linalg.vector_norm(features(canonical(reference)) - target, dim=-1)
  phases = torch.arange(len(distance), device=distance.device)
  motion_ids = torch.searchsorted(motion.motion_ends, phases, right=True)
  available = motion.motion_ends[motion_ids] - phases
  distance[available < horizon + 5] = float("inf")
  chosen_phases: list[int] = []
  chosen_ids: list[int] = []
  for phase in distance.argsort().tolist():
    if float(distance[phase]) > limit or len(chosen_phases) == count:
      break
    motion_id = int(motion_ids[phase])
    if motion_id in chosen_ids:
      continue
    chosen_phases.append(phase)
    chosen_ids.append(motion_id)
  chosen = torch.tensor(chosen_phases, device=command.device, dtype=torch.long)
  return (
    chosen,
    torch.tensor(chosen_ids, device=command.device, dtype=torch.long),
    distance[chosen],
  )


def collect(cfg: BranchCfg) -> Path:
  """Record a filtered pilot corpus with several references per shared start."""
  if not cfg.checkpoint.is_file():
    raise FileNotFoundError(cfg.checkpoint)
  if cfg.num_branches < 2 or cfg.horizon_steps < 1 or cfg.groups < 1:
    raise ValueError("Need at least two branches, one group and a positive horizon")
  env_cfg = load_env_cfg(ORACLE_TASK_ID)
  env_cfg.scene.num_envs = cfg.num_branches
  env_cfg.events = {}
  for group in env_cfg.observations.values():
    group.enable_corruption = False
  original = env_cfg.commands["motion"]
  if not isinstance(original, MotionSetCommandCfg):
    raise TypeError("Oracle task must use MotionSetCommandCfg")
  options = {field.name: getattr(original, field.name) for field in fields(original)}
  motion_cfg = BranchMotionCommandCfg(**options)
  motion_cfg.pose_range = {}
  motion_cfg.velocity_range = {}
  motion_cfg.joint_position_range = (0.0, 0.0)
  env_cfg.commands["motion"] = motion_cfg
  env_cfg.scene.sensors = (
    *env_cfg.scene.sensors,
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="netforce",
      num_slots=1,
    ),
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  agent_cfg = load_rl_cfg(ORACLE_TASK_ID)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = OracleRunner(wrapped, asdict(agent_cfg), device=cfg.device)
  runner.load(
    str(cfg.checkpoint), load_cfg={"actor": True}, strict=True, map_location=cfg.device
  )
  policy = runner.get_inference_policy(device=cfg.device)
  command = env.command_manager.get_term("motion")
  sensor = env.scene["feet_ground_contact"]
  if not isinstance(command, BranchMotionCommand) or not isinstance(
    sensor, ContactSensor
  ):
    raise TypeError("Branch motion or foot sensor is missing")

  records: dict[str, list[np.ndarray]] = {
    key: []
    for key in (
      "states",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
      "foot_contact",
      "previous_action",
      "env_id",
      "trajectory",
      "frame",
      "phase",
      "motion_id",
    )
  }
  robot = env.scene["robot"]
  origin = env.scene.env_origins
  feet = command._foot_indexes
  kept_groups = 0
  try:
    for group_index in range(cfg.groups):
      command.reference_rotation.zero_()
      command.reference_rotation[:, 0] = 1.0
      command.reference_translation[:] = origin
      obs, _ = env.reset()
      survived = True
      for _ in range(cfg.settle_steps):
        with torch.inference_mode():
          action = policy(TensorDict(obs, batch_size=[cfg.num_branches]))  # ty: ignore[invalid-argument-type]
        obs, _, terminated, truncated, _ = env.step(action)
        survived &= not bool((terminated | truncated)[0])
      if not survived:
        print(f"[goal] group {group_index + 1}: source start did not survive")
        continue
      start = state(robot)[:1].clone()
      phases, ids, match = _candidate_phases(
        command, start, cfg.num_branches, cfg.horizon_steps, cfg.max_match_distance
      )
      if len(phases) < 2:
        print(f"[goal] group {group_index + 1}: only {len(phases)} matching references")
        continue
      active = torch.arange(len(phases), device=cfg.device)
      start_world = start.expand(len(phases), -1).clone()
      start_world[:, :3] += origin[active] - origin[0]
      command.set_references(active, phases, ids, start_world)
      joints = start_world[:, 13 : 13 + command.robot_joint_pos.shape[1]]
      speeds = start_world[:, 13 + joints.shape[1] :]
      command._write_reference_state_to_sim(
        active,
        start_world[:, :3],
        start_world[:, 3:7],
        start_world[:, 7:10],
        start_world[:, 10:13],
        joints,
        speeds,
      )
      previous_action = env.action_manager.action[:1].expand(len(phases), -1).clone()
      env.action_manager.initialize_action(previous_action, active)
      env.sim.forward()
      command.update_relative_body_poses()
      env.observation_manager._obs_buffer = None
      obs = env.observation_manager.compute()
      branch_rows: dict[str, list[np.ndarray]] = {key: [] for key in records}
      valid = torch.ones(len(phases), dtype=torch.bool, device=cfg.device)

      for tick in range(cfg.horizon_steps + 1):
        if sensor.data.found is None:
          raise ValueError("Foot contact sensor has no found data")
        actual = state(robot)[: len(phases)].clone()
        actual[:, :2] -= origin[: len(phases), :2]
        body_pos = command.robot_body_pos_w[: len(phases)] - origin[: len(phases), None]
        values = {
          "states": _numpy(actual),
          "body_pos_w": _numpy(body_pos),
          "body_quat_w": _numpy(command.robot_body_quat_w[: len(phases)]),
          "body_lin_vel_w": _numpy(command.robot_body_lin_vel_w[: len(phases)]),
          "body_ang_vel_w": _numpy(command.robot_body_ang_vel_w[: len(phases)]),
          "foot_contact": _numpy((sensor.data.found[: len(phases)] > 0).float()),
          "previous_action": _numpy(env.action_manager.action[: len(phases)]),
          "env_id": np.full(len(phases), group_index, dtype=np.int32),
          "trajectory": np.arange(len(phases)) + group_index * cfg.num_branches,
          "frame": np.full(len(phases), tick, dtype=np.int32),
          "phase": _numpy(command.time_steps[: len(phases)]),
          "motion_id": _numpy(command.motion_ids[: len(phases)]),
        }
        for key, value in values.items():
          branch_rows[key].append(value)
        root_error = torch.linalg.vector_norm(
          command.robot_body_pos_w[active, 0] - command.body_pos_w[active, 0], dim=-1
        )
        foot_error = torch.linalg.vector_norm(
          command.robot_body_pos_w[active][:, feet]
          - command.body_pos_w[active][:, feet],
          dim=-1,
        ).amax(dim=-1)
        valid &= (root_error <= cfg.max_root_error) & (foot_error <= cfg.max_foot_error)
        if tick == cfg.horizon_steps:
          break
        with torch.inference_mode():
          action = policy(TensorDict(obs, batch_size=[cfg.num_branches]))  # ty: ignore[invalid-argument-type]
        obs, _, terminated, truncated, _ = env.step(action)
        valid &= ~(terminated | truncated)[: len(phases)]

      endpoints = torch.from_numpy(branch_rows["states"][-1]).to(cfg.device)
      separation = torch.linalg.vector_norm(
        endpoints[:, None, :3] - endpoints[None, :, :3], dim=-1
      )
      early_actions = (
        torch.from_numpy(
          np.stack(branch_rows["previous_action"][1 : min(6, cfg.horizon_steps + 1)])
        )
        .to(cfg.device)
        .transpose(0, 1)
      )
      action_difference = (
        (early_actions[:, None] - early_actions[None]).square().mean(dim=(-1, -2))
      ).sqrt()
      distinct = (
        (separation >= cfg.min_endpoint_separation)
        & (action_difference >= cfg.min_action_separation)
        & valid[None, :]
      ).any(dim=1)
      accepted = torch.where(valid & distinct)[0].tolist()
      if len(accepted) < 2:
        print(
          f"[goal] group {group_index + 1}: {int(valid.sum())} tracked, "
          f"{int((valid & distinct).sum())} distinct, "
          f"match distances {match.tolist()}, "
          f"max early action difference {action_difference.max():.3f}"
        )
        continue
      for branch in accepted:
        for key, values in branch_rows.items():
          records[key].append(np.stack(values)[:, branch])
      kept_groups += 1
      print(
        f"[goal] group {group_index + 1}/{cfg.groups}: {len(accepted)} branches, "
        f"match {match[accepted].min():.2f}..{match[accepted].max():.2f}"
      )
  finally:
    env.close()

  if not kept_groups:
    raise ValueError(
      "No feasible, distinct branches; inspect oracle recovery and reference matches"
    )
  output = {key: np.concatenate(values) for key, values in records.items()}
  output["skill"] = np.zeros(len(output["states"]), dtype=np.int16)
  output["skill_names"] = np.asarray(["oracle"])
  output["body_names"] = np.asarray(motion_cfg.body_names)
  output["motion_files"] = np.asarray(command.motion.motion_files)
  output["fps"] = np.asarray(control_rate(env_cfg))
  output["trajectory_ids_global"] = np.asarray(True)
  cfg.path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(cfg.path, **output)  # ty: ignore[invalid-argument-type]
  print(
    f"[goal] wrote {cfg.path}: {kept_groups} shared starts, {len(output['states'])} rows"
  )
  return cfg.path


if __name__ == "__main__":
  collect(tyro.cli(BranchCfg, config=mjlab.TYRO_FLAGS))
