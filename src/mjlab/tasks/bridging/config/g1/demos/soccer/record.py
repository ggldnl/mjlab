"""Record a locomotion entry at the speed configured for jump recovery."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.config.g1.demos.soccer.arena import (
  external_twist,
  reset_robot,
)
from mjlab.tasks.bridging.config.g1.demos.soccer.config import (
  CONFIG_PATH,
  Scene,
  Settings,
)
from mjlab.tasks.bridging.tests.stage import fresh_obs, load_policy, state
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


@dataclass(frozen=True)
class Config:
  config: Path = CONFIG_PATH
  out: Path | None = None
  device: str | None = None
  seed: int = 0


def main(config: Config) -> None:
  from mjlab import tasks as _tasks

  del _tasks
  settings = Settings.load(config.config)
  policies = settings.policies
  torch.manual_seed(config.seed)
  device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = load_env_cfg(policies.locomotion_task, play=True)
  cfg.seed = config.seed
  cfg.scene.num_envs = 1
  cfg.events = {
    "reset_robot": EventTermCfg(
      func=reset_robot, mode="reset", params={"scene": Scene()}
    )
  }
  cfg.rewards, cfg.metrics, cfg.curriculum = {}, {}, {}
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  cfg.commands["twist"] = external_twist(twist)
  env = ManagerBasedRlEnv(cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  frames = []
  try:
    checkpoint = (
      Path(policies.locomotion_checkpoint) if policies.locomotion_checkpoint else None
    )
    policy = load_policy(
      policies.locomotion_task, wrapped, "actor", device, checkpoint, task_runner=False
    )
    command = env.command_manager.get_term("twist")
    assert isinstance(command, UniformVelocityCommand)
    obs = wrapped.get_observations()
    with torch.inference_mode():
      for step in range(policies.record_warmup_steps + policies.record_frames):
        command.vel_command_b[0] = command.vel_command_b.new_tensor(
          [policies.speed_after_jump, 0.0, 0.0]
        )
        if step >= policies.record_warmup_steps:
          frames.append(state(env)[0].cpu().numpy().copy())
        obs, _, done, _ = wrapped.step(policy(fresh_obs(env)))
        if (
          bool(done.any())
          or float(env.scene["robot"].data.root_link_pos_w[0, 2])
          < settings.controller.fall_height
        ):
          raise RuntimeError("Locomotion failed while recording its entry")
    clip = np.stack(frames)
    clip[:, :3] -= env.scene.env_origins[0].cpu().numpy()
    if not np.isfinite(clip).all():
      raise RuntimeError("Locomotion entry contains nonfinite states")
    output = config.out or Path(policies.locomotion_entry_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
      output,
      states=clip,
      fps=1.0 / env.step_dt,
      joint_names=np.asarray(env.scene["robot"].joint_names),
      speed=policies.speed_after_jump,
    )
    print(
      f"Saved {len(clip)} locomotion frames to {output}; mean speed {np.linalg.norm(clip[:, 7:9], axis=1).mean():.2f} m/s"
    )
  finally:
    wrapped.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
