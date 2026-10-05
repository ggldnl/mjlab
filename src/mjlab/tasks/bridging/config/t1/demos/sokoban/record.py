"""Record entry windows from the trained skills for diffusion handoffs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.config.t1.demos.sokoban.arena import WALK_TASK
from mjlab.tasks.bridging.config.t1.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.tests.stage import load_policy, state
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


@dataclass
class Config:
  robot: Literal["t1", "g1"] = "t1"
  out: Path = Path("data/sokoban/t1_entries.npz")
  walk_task: str = WALK_TASK
  walk_checkpoint: Path | None = None
  push_checkpoint: Path | None = None
  frames: int = 16
  walk_warmup: int = 50
  device: str | None = None


def main(config: Config) -> None:
  from mjlab import tasks as _tasks

  del _tasks
  if config.frames < 1 or config.walk_warmup < 0:
    raise ValueError("Need positive frames and nonnegative warmup")
  device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  clips = {}
  joint_names = None
  fps = None
  push_task = PUSH_TASK_ID
  if config.robot == "g1":
    from mjlab.tasks.bridging.config.g1.skills.push import PUSH_TASK_ID as G1_PUSH

    push_task = G1_PUSH
  for name, task, checkpoint in (
    ("walk", config.walk_task, config.walk_checkpoint),
    ("push", push_task, config.push_checkpoint),
  ):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.rewards, cfg.metrics, cfg.curriculum = {}, {}, {}
    cfg.events.pop("push_robot", None)
    if name == "walk":
      twist = cfg.commands["twist"]
      assert isinstance(twist, UniformVelocityCommandCfg)
      cfg.commands["twist"] = replace(
        twist,
        resampling_time_range=(1e9, 1e9),
        gui=False,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        rel_standing_envs=0.0,
        init_velocity_prob=0.0,
      )
      standing = cfg.commands["twist"]
      assert isinstance(standing, UniformVelocityCommandCfg)
      standing.ranges.lin_vel_x = (0.0, 0.0)
      standing.ranges.lin_vel_y = (0.0, 0.0)
      standing.ranges.ang_vel_z = (0.0, 0.0)
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)
    try:
      names = tuple(env.scene["robot"].joint_names)
      frequency = 1.0 / env.step_dt
      if joint_names is not None and (names != joint_names or frequency != fps):
        raise ValueError("Skill joint layouts or control frequencies differ")
      joint_names, fps = names, frequency
      policy = load_policy(
        task, wrapped, "actor", device, checkpoint, task_runner=False
      )
      obs = wrapped.get_observations()
      frames = []
      with torch.inference_mode():
        for index in range(
          config.frames + (config.walk_warmup if name == "walk" else 0)
        ):
          if name == "push" or index >= config.walk_warmup:
            frames.append(state(env)[0].cpu().numpy().copy())
          obs, _, done, _ = wrapped.step(policy(obs))
          if bool(done.any()):
            raise RuntimeError(f"{name} failed while recording its entry")
      clips[name] = np.stack(frames)
      clips[name][:, :3] -= env.scene.env_origins[0].cpu().numpy()
    finally:
      wrapped.close()
  config.out.parent.mkdir(parents=True, exist_ok=True)
  assert joint_names is not None and fps is not None
  np.savez(config.out, **clips, joint_names=np.asarray(joint_names), fps=fps)
  print(f"Saved {config.frames} entry frames per skill to {config.out}")


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
