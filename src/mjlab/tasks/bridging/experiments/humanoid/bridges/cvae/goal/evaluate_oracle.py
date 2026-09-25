"""Check that the oracle can track executed routes used for DAgger.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.evaluate_oracle \
      --checkpoint logs/rsl_rl/g1_cvae_oracle/<run>/model_5000.pt
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal import (
  GOAL_CVAE_TASK_ID,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.collect import (
  DEFAULT_ORACLE_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.command import (
  GoalCommand,
  GoalCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.runner import (
  _actor_state,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  CHANNELS,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

NAMES = CHANNELS + ("foot_pos", "foot_ori", "foot_lin_vel", "foot_ang_vel")


@dataclass
class EvaluateCfg:
  checkpoint: Path
  dataset: Path = DEFAULT_ORACLE_DATASET
  num_envs: int = 64
  batches: int = 4
  split: str = "eval"
  perturb_start: bool = True
  student: bool = False
  device: str = "cuda:0"


def evaluate(cfg: EvaluateCfg) -> dict[str, float]:
  if not cfg.checkpoint.is_file():
    raise FileNotFoundError(cfg.checkpoint)
  env_cfg = load_env_cfg(GOAL_CVAE_TASK_ID)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.terminations = {}
  bridge_cfg = env_cfg.commands["bridge"]
  if not isinstance(bridge_cfg, GoalCommandCfg):
    raise TypeError("Goal CVAE task has no goal command")
  bridge_cfg.dataset_path = cfg.dataset
  bridge_cfg.split = cfg.split
  bridge_cfg.start_perturb_prob = 1.0 if cfg.perturb_start else 0.0
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  agent_cfg = load_rl_cfg(GOAL_CVAE_TASK_ID)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(GOAL_CVAE_TASK_ID)
  assert runner_cls is not None
  runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
  algorithm = cast(Distillation, runner.alg)
  if cfg.student:
    runner.load(
      str(cfg.checkpoint),
      load_cfg={"student": True, "iteration": False},
      strict=True,
      map_location=cfg.device,
    )
    policy = algorithm.student
  else:
    algorithm.teacher.load_state_dict(_actor_state(cfg.checkpoint), strict=True)
    policy = algorithm.teacher
  policy.eval()
  bridge = env.command_manager.get_term("bridge")
  if not isinstance(bridge, GoalCommand):
    raise TypeError("Goal CVAE task did not create GoalCommand")
  errors: list[torch.Tensor] = []
  success: list[torch.Tensor] = []
  try:
    for _ in range(cfg.batches):
      obs, _ = env.reset()
      policy.reset()
      done = torch.zeros(cfg.num_envs, dtype=torch.bool, device=cfg.device)
      for _ in range(bridge.max_steps):
        with torch.inference_mode():
          action = policy(
            TensorDict(obs, batch_size=[cfg.num_envs])  # ty: ignore[invalid-argument-type]
          )
        obs, _, _, _, _ = env.step(action)
        reached = (bridge.step >= bridge.window_steps) & ~done
        if reached.any():
          measured = torch.cat((bridge.target_errors(), bridge.foot_errors()), dim=-1)
          errors.append(measured[reached].detach().cpu())
          limits = torch.cat(
            (
              bridge.tolerances,
              torch.tensor(bridge.cfg.foot_tolerances, device=cfg.device),
            )
          )
          success.append((measured[reached] <= limits).all(dim=-1).detach().cpu())
          done |= reached
      if not done.all():
        raise RuntimeError("Some bridge windows did not reach their deadline")
  finally:
    wrapped.close()
  values = torch.cat(errors)
  good = torch.cat(success)
  report = {
    f"{name}_p95": torch.quantile(values[:, i], 0.95).item()
    for i, name in enumerate(NAMES)
  }
  report["strict_success"] = good.float().mean().item()
  report["windows"] = float(len(good))
  for name in NAMES:
    print(f"{name:>18}: p95={report[f'{name}_p95']:.4f}")
  print(f"strict_success={report['strict_success']:.3f} over {len(good)} windows")
  return report


if __name__ == "__main__":
  evaluate(tyro.cli(EvaluateCfg, config=mjlab.TYRO_FLAGS))
