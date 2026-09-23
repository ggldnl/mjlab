"""Record physically executed oracle trajectories for goal bridge training.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.collect \
      --checkpoint logs/rsl_rl/g1_cvae_oracle/<run>/model_5000.pt
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from tensordict import TensorDict

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
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
  DATASET_ROOT,
  control_rate,
  state,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

ORACLE_ROLLOUT_DATASET = DATASET_ROOT / "oracle_rollouts.npz"
ORACLE_BRANCH_DATASET = DATASET_ROOT / "oracle_branches.npz"
DEFAULT_ORACLE_DATASET = DATASET_ROOT / "oracle_goal.npz"


@dataclass
class CollectCfg:
  checkpoint: Path
  path: Path = ORACLE_ROLLOUT_DATASET
  num_envs: int = 64
  steps: int = 1000
  settle: int = 25
  max_root_error: float = 0.25
  max_foot_error: float = 0.25
  device: str = "cuda:0"


def _numpy(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy().copy()


def collect(cfg: CollectCfg) -> Path:
  """Write one row per qualified physical control step."""
  if not cfg.checkpoint.is_file():
    raise FileNotFoundError(cfg.checkpoint)
  if cfg.steps <= cfg.settle or cfg.num_envs < 1:
    raise ValueError("steps must exceed settle and num_envs must be positive")

  env_cfg = load_env_cfg(ORACLE_TASK_ID)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.events = {}
  for group in env_cfg.observations.values():
    group.enable_corruption = False
  motion_cfg = env_cfg.commands["motion"]
  if not isinstance(motion_cfg, MotionSetCommandCfg):
    raise TypeError("Oracle task must use MotionSetCommandCfg")
  motion_cfg.balanced_sampling = True
  motion_cfg.pose_range = {}
  motion_cfg.velocity_range = {}
  motion_cfg.joint_position_range = (0.0, 0.0)
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
    str(cfg.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=cfg.device,
  )
  policy = runner.get_inference_policy(device=cfg.device)
  motion = env.command_manager.get_term("motion")
  sensor = env.scene["feet_ground_contact"]
  if not isinstance(motion, MotionSetCommand) or not isinstance(sensor, ContactSensor):
    raise TypeError("Oracle motion or foot contact sensor is missing")

  rows: dict[str, list[np.ndarray]] = {
    name: []
    for name in (
      "states",
      "env_id",
      "trajectory",
      "frame",
      "phase",
      "motion_id",
      "previous_action",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
      "foot_contact",
      "valid",
    )
  }
  obs, _ = env.reset()
  age = torch.zeros(cfg.num_envs, dtype=torch.long, device=cfg.device)
  trajectory = torch.arange(cfg.num_envs, dtype=torch.long, device=cfg.device)
  env_id = torch.arange(cfg.num_envs, dtype=torch.long, device=cfg.device)
  origin = env.scene.env_origins[:, None, :]
  feet = motion._foot_indexes

  try:
    for tick in range(cfg.steps):
      with torch.inference_mode():
        action = policy(
          TensorDict(obs, batch_size=[cfg.num_envs])  # ty: ignore[invalid-argument-type]
        )
      obs, _, terminated, truncated, _ = env.step(action)
      done = terminated | truncated
      age = torch.where(done, 0, age + 1)
      trajectory = torch.where(done, trajectory + cfg.num_envs, trajectory)
      root_error = torch.linalg.vector_norm(
        motion.robot_body_pos_w[:, 0] - motion.body_pos_w[:, 0], dim=-1
      )
      foot_error = torch.linalg.vector_norm(
        motion.robot_body_pos_w[:, feet] - motion.body_pos_w[:, feet], dim=-1
      ).amax(dim=-1)
      valid = (
        (age >= cfg.settle)
        & ~done
        & (root_error <= cfg.max_root_error)
        & (foot_error <= cfg.max_foot_error)
      )
      current = state(env.scene["robot"]).clone()
      current[:, :2] -= env.scene.env_origins[:, :2]
      body_pos = motion.robot_body_pos_w - origin
      rows["states"].append(_numpy(current))
      rows["env_id"].append(_numpy(env_id))
      rows["trajectory"].append(_numpy(trajectory))
      rows["frame"].append(_numpy(age))
      rows["phase"].append(_numpy(motion.time_steps))
      rows["motion_id"].append(_numpy(motion.motion_ids))
      rows["previous_action"].append(_numpy(env.action_manager.action))
      rows["body_pos_w"].append(_numpy(body_pos))
      rows["body_quat_w"].append(_numpy(motion.robot_body_quat_w))
      rows["body_lin_vel_w"].append(_numpy(motion.robot_body_lin_vel_w))
      rows["body_ang_vel_w"].append(_numpy(motion.robot_body_ang_vel_w))
      if sensor.data.found is None:
        raise ValueError("Foot contact sensor has no found data")
      rows["foot_contact"].append(_numpy((sensor.data.found > 0).float()))
      rows["valid"].append(_numpy(valid))
      if (tick + 1) % 100 == 0:
        print(f"[goal] collected {tick + 1}/{cfg.steps} control steps")
  finally:
    env.close()

  values = {key: np.concatenate(parts) for key, parts in rows.items()}
  mask = values.pop("valid").astype(bool)
  kept = {key: value[mask] for key, value in values.items()}
  if not len(kept["states"]):
    raise ValueError("No qualified oracle rows were collected")
  kept["skill"] = np.zeros(len(kept["states"]), dtype=np.int16)
  kept["skill_names"] = np.asarray(["oracle"])
  kept["body_names"] = np.asarray(motion_cfg.body_names)
  kept["motion_files"] = np.asarray(motion.motion.motion_files)
  kept["fps"] = np.asarray(control_rate(env_cfg))
  kept["trajectory_ids_global"] = np.asarray(True)
  cfg.path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(cfg.path, **kept)
  print(f"[goal] wrote {cfg.path} ({len(kept['states'])} qualified rows)")
  return cfg.path


if __name__ == "__main__":
  collect(tyro.cli(CollectCfg, config=mjlab.TYRO_FLAGS))
