"""Evaluate CVAE endpoint accuracy on held-out tracker windows.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.evaluate \
      --checkpoint logs/rsl_rl/g1_cvae_bridge/<run>/model_9999.pt \
      --motion-file data/lafan1_g1/motions/walk1_subject1.npz
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae import CVAE_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.command import CvaeCommand
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.runner import CvaeRunner
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import CHANNELS
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--episodes", type=int, default=512)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  if args.episodes < 1 or args.num_envs < 1:
    parser.error("--episodes and --num-envs must be positive")
  if not args.checkpoint.is_file() or not args.motion_file.is_file():
    parser.error("--checkpoint and --motion-file must exist")

  torch.manual_seed(args.seed)
  cfg = load_env_cfg(CVAE_TASK_ID, play=True)
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = str(args.motion_file)
  cfg.scene.num_envs = args.num_envs
  cfg.auto_reset = False
  cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env)
  runner = CvaeRunner(wrapped, asdict(load_rl_cfg(CVAE_TASK_ID)), device=args.device)
  runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  bridge = env.command_manager.get_term("bridge")
  assert isinstance(bridge, CvaeCommand)

  errors, durations, deadlines, falls = [], [], [], []
  obs = wrapped.get_observations()
  with torch.inference_mode():
    while sum(len(batch) for batch in errors) < args.episodes:
      obs, _, done, _ = wrapped.step(policy(obs))
      ids = done.nonzero(as_tuple=True)[0]
      if ids.numel() == 0:
        continue
      errors.append(bridge.target_errors()[ids].cpu())
      durations.append((bridge.window_steps[ids] / bridge.fps).cpu())
      deadlines.append(env.reset_time_outs[ids].cpu())
      falls.append(env.reset_terminated[ids].cpu())
      env.reset(env_ids=ids)
      obs = wrapped.get_observations()

  error = torch.cat(errors)[: args.episodes]
  duration = torch.cat(durations)[: args.episodes]
  deadline = torch.cat(deadlines)[: args.episodes]
  fall = torch.cat(falls)[: args.episodes]
  success = deadline & ~fall & (error <= bridge.tolerances.cpu()).all(dim=-1)
  print(f"eval episodes: {len(error)}  source: {args.motion_file.stem}")
  print(
    f"strict success: {success.float().mean():.1%}  "
    f"deadline: {deadline.float().mean():.1%}  falls: {fall.float().mean():.1%}"
  )
  for low, high in ((0.5, 1.0), (1.0, 1.5), (1.5, 2.01)):
    selected = (duration >= low) & (duration < high)
    if selected.any():
      print(
        f"{low:.1f}-{min(high, 2.0):.1f}s: {int(selected.sum())} episodes, "
        f"{success[selected].float().mean():.1%} strict success"
      )
  for index, name in enumerate(CHANNELS):
    values = error[:, index]
    print(
      f"{name:>18}: mean {values.mean():.3f}, "
      f"p95 {torch.quantile(values, 0.95):.3f}, "
      f"tolerance {bridge.tolerances[index]:.3f}"
    )
  wrapped.close()


if __name__ == "__main__":
  main()
