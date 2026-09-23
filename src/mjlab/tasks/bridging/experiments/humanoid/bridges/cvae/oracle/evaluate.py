"""Evaluate oracle tracking, recovery, and causal reference use.

Example:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle.evaluate \
      --checkpoint logs/rsl_rl/g1_cvae_oracle/<run>/model_5000.pt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle import (
  ORACLE_TASK_ID,
  mdp,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle.command import (
  MotionSetCommand,
  MotionSetCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle.runner import (
  OracleRunner,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  CHANNELS,
  Tolerances,
  channel_errors,
  upper_body_mask,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends

FOOT_CHANNELS = ("foot_pos", "foot_ori", "foot_lin_vel", "foot_ang_vel")
EVAL_CHANNELS = CHANNELS + FOOT_CHANNELS
FOOT_HANDOFF_LIMITS = (0.06, 0.10, 0.30, 0.60)


def _state_errors(command: MotionSetCommand) -> torch.Tensor:
  """Measure the post-step robot against the now-current reference frame."""
  actual = torch.cat(
    (
      command.robot_body_pos_w[:, 0],
      command.robot_body_quat_w[:, 0],
      command.robot_body_lin_vel_w[:, 0],
      command.robot_body_ang_vel_w[:, 0],
      command.robot_joint_pos,
      command.robot_joint_vel,
    ),
    dim=-1,
  )
  target = torch.cat(
    (
      command.body_pos_w[:, 0],
      command.body_quat_w[:, 0],
      command.body_lin_vel_w[:, 0],
      command.body_ang_vel_w[:, 0],
      command.joint_pos,
      command.joint_vel,
    ),
    dim=-1,
  )
  bridge = channel_errors(
    actual,
    target,
    upper_body_mask(tuple(command.robot.joint_names), command.device),
  )
  feet = command._foot_indexes
  foot = torch.stack(
    (
      torch.linalg.vector_norm(
        command.robot_body_pos_w[:, feet] - command.body_pos_w[:, feet], dim=-1
      ).amax(dim=-1),
      quat_error_magnitude(
        command.robot_body_quat_w[:, feet], command.body_quat_w[:, feet]
      ).amax(dim=-1),
      torch.linalg.vector_norm(
        command.robot_body_lin_vel_w[:, feet] - command.body_lin_vel_w[:, feet],
        dim=-1,
      ).amax(dim=-1),
      torch.linalg.vector_norm(
        command.robot_body_ang_vel_w[:, feet] - command.body_ang_vel_w[:, feet],
        dim=-1,
      ).amax(dim=-1),
    ),
    dim=-1,
  )
  return torch.cat((bridge, foot), dim=-1)


def _stats(values: torch.Tensor) -> dict[str, dict[str, float]]:
  return {
    name: {
      "mean": values[:, index].mean().item(),
      "p50": torch.quantile(values[:, index], 0.50).item(),
      "p95": torch.quantile(values[:, index], 0.95).item(),
      "max": values[:, index].max().item(),
    }
    for index, name in enumerate(EVAL_CHANNELS)
  }


def _per_motion(
  errors: torch.Tensor, survived: torch.Tensor, motion_ids: torch.Tensor
) -> dict[str, dict[str, Any]]:
  result = {}
  for motion_id in motion_ids.unique(sorted=True).tolist():
    selected = motion_ids == motion_id
    result[str(motion_id)] = {
      "episodes": int(selected.sum()),
      "survival_rate": survived[selected].float().mean().item(),
      "terminal": _stats(errors[selected]),
    }
  return result


def _run_scenario(
  *,
  env: ManagerBasedRlEnv,
  wrapped: RslRlVecEnvWrapper,
  policy,
  command: MotionSetCommand,
  episodes: int,
  seed: int,
  recovery_cfg: MotionSetCommandCfg | None,
  shuffled_reference: bool,
) -> dict[str, Any]:
  command.oracle_cfg.pose_range = (
    {} if recovery_cfg is None else recovery_cfg.pose_range.copy()
  )
  command.oracle_cfg.velocity_range = (
    {} if recovery_cfg is None else recovery_cfg.velocity_range.copy()
  )
  command.oracle_cfg.joint_position_range = (
    (0.0, 0.0) if recovery_cfg is None else recovery_cfg.joint_position_range
  )
  command._sample_cursor = 0
  torch.manual_seed(seed)
  env.reset()
  obs = wrapped.get_observations()
  current_initial = _state_errors(command)
  current_motion_ids = command.motion_ids.clone()
  proprio_dim = mdp.oracle_proprioception(env, "motion").shape[-1]

  final_batches: list[torch.Tensor] = []
  initial_batches: list[torch.Tensor] = []
  survival_batches: list[torch.Tensor] = []
  motion_batches: list[torch.Tensor] = []
  length_batches: list[torch.Tensor] = []
  step_batches: list[torch.Tensor] = []
  action_squared_delta = 0.0
  action_squared = 0.0
  action_count = 0

  # The simulator resets sensor buffers in place, which is incompatible with
  # tensors created under torch.inference_mode(). no_grad still avoids autograd.
  with torch.no_grad():
    while sum(len(batch) for batch in final_batches) < episodes:
      shifted = obs.clone()
      shifted["actor"][:, proprio_dim:] = obs["actor"].roll(shifts=1, dims=0)[
        :, proprio_dim:
      ]
      actions = policy(obs)
      shifted_actions = policy(shifted)
      action_squared_delta += torch.square(actions - shifted_actions).sum().item()
      action_squared += torch.square(actions).sum().item()
      action_count += actions.numel()

      applied = shifted_actions if shuffled_reference else actions
      obs, _, done, _ = wrapped.step(applied)
      step_error = _state_errors(command)
      step_batches.append(step_error.cpu())
      ids = done.nonzero(as_tuple=True)[0]
      if ids.numel() == 0:
        continue

      final_batches.append(step_error[ids].cpu())
      initial_batches.append(current_initial[ids].cpu())
      survival_batches.append(
        (env.reset_time_outs[ids] & ~env.reset_terminated[ids]).cpu()
      )
      motion_batches.append(current_motion_ids[ids].cpu())
      length_batches.append(env.episode_length_buf[ids].cpu())
      env.reset(env_ids=ids)
      obs = wrapped.get_observations()
      current_initial[ids] = _state_errors(command)[ids]
      current_motion_ids[ids] = command.motion_ids[ids]

  final = torch.cat(final_batches)[:episodes]
  initial = torch.cat(initial_batches)[:episodes]
  survived = torch.cat(survival_batches)[:episodes]
  motion_ids = torch.cat(motion_batches)[:episodes]
  lengths = torch.cat(length_batches)[:episodes]
  steps = torch.cat(step_batches)
  return {
    "episodes": episodes,
    "motion_coverage": int(motion_ids.unique().numel()),
    "survival_rate": survived.float().mean().item(),
    "episode_length_steps": {
      "mean": lengths.float().mean().item(),
      "min": int(lengths.min()),
      "max": int(lengths.max()),
    },
    "initial": _stats(initial),
    "all_steps": _stats(steps),
    "terminal": _stats(final),
    "within_handoff_rate": None,
    "action_reference_test": {
      "rms_delta": math.sqrt(action_squared_delta / action_count),
      "relative_rms_delta": math.sqrt(
        action_squared_delta / max(action_squared, 1e-12)
      ),
    },
    "per_motion": _per_motion(final, survived, motion_ids),
    "_terminal_tensor": final,
    "_survival_tensor": survived,
  }


def _p95(result: dict[str, Any]) -> torch.Tensor:
  return torch.tensor([result["terminal"][name]["p95"] for name in EVAL_CHANNELS])


def _strip_tensors(value: Any) -> Any:
  if isinstance(value, dict):
    return {
      key: _strip_tensors(item)
      for key, item in value.items()
      if not key.startswith("_")
    }
  return value


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-files", nargs="+", default=None)
  parser.add_argument("--episodes", type=int, default=156)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--horizon-s", type=float, default=2.0)
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error("--checkpoint must exist")
  if args.episodes < 1 or args.num_envs < 2 or args.horizon_s <= 0:
    parser.error("episodes and horizon must be positive; num-envs must be at least 2")

  configure_torch_backends()
  torch.manual_seed(args.seed)
  cfg = load_env_cfg(ORACLE_TASK_ID, play=True)
  training_cfg = load_env_cfg(ORACLE_TASK_ID, play=False)
  motion_cfg = cast(MotionSetCommandCfg, cfg.commands["motion"])
  recovery_cfg = cast(MotionSetCommandCfg, training_cfg.commands["motion"])
  if args.motion_files:
    motion_cfg.motion_files = tuple(args.motion_files)
  cfg.scene.num_envs = args.num_envs
  cfg.episode_length_s = args.horizon_s
  cfg.auto_reset = False
  cfg.seed = args.seed
  motion_cfg.debug_vis = False
  motion_cfg.balanced_sampling = True
  step_dt = cfg.sim.mujoco.timestep * cfg.decimation
  motion_cfg.minimum_remaining_steps = math.ceil(args.horizon_s / step_dt) + 1

  agent_cfg = load_rl_cfg(ORACLE_TASK_ID)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = OracleRunner(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  command = cast(MotionSetCommand, env.command_manager.get_term("motion"))

  print(
    f"Evaluating {args.checkpoint} on {len(command.motion.motion_files)} clips "
    f"({args.episodes} episodes/scenario, {args.horizon_s:.2f}s windows)"
  )
  nominal = _run_scenario(
    env=env,
    wrapped=wrapped,
    policy=policy,
    command=command,
    episodes=args.episodes,
    seed=args.seed,
    recovery_cfg=None,
    shuffled_reference=False,
  )
  recovery = _run_scenario(
    env=env,
    wrapped=wrapped,
    policy=policy,
    command=command,
    episodes=args.episodes,
    seed=args.seed,
    recovery_cfg=recovery_cfg,
    shuffled_reference=False,
  )
  shuffled = _run_scenario(
    env=env,
    wrapped=wrapped,
    policy=policy,
    command=command,
    episodes=args.episodes,
    seed=args.seed,
    recovery_cfg=None,
    shuffled_reference=True,
  )

  handoff = torch.tensor(
    list(asdict(Tolerances()).values()) + list(FOOT_HANDOFF_LIMITS)
  )
  oracle_margin = handoff * 0.5
  for result in (nominal, recovery, shuffled):
    result["within_handoff_rate"] = (
      (result["_survival_tensor"] & (result["_terminal_tensor"] <= handoff).all(dim=-1))
      .float()
      .mean()
      .item()
    )
  nominal_score = (_p95(nominal) / handoff).amax().item()
  shuffled_score = (_p95(shuffled) / handoff).amax().item()
  degradation = shuffled_score / max(nominal_score, 1e-12)
  gates = {
    "nominal_survival": nominal["survival_rate"] >= 0.99,
    "nominal_half_budget_p95": bool((_p95(nominal) <= oracle_margin).all()),
    "recovery_survival": recovery["survival_rate"] >= 0.95,
    "recovery_handoff_p95": bool((_p95(recovery) <= handoff).all()),
    "reference_changes_action": nominal["action_reference_test"]["relative_rms_delta"]
    >= 0.10,
    "wrong_reference_degrades_tracking": degradation >= 1.25,
  }
  gates["overall"] = all(gates.values())

  match = re.search(r"model_(\d+)", args.checkpoint.stem)
  report = {
    "checkpoint": str(args.checkpoint.resolve()),
    "iteration": int(match.group(1)) if match else None,
    "configuration": {
      "episodes_per_scenario": args.episodes,
      "num_envs": args.num_envs,
      "horizon_s": args.horizon_s,
      "seed": args.seed,
      "channels": EVAL_CHANNELS,
      "motions": [
        {"id": index, "file": path}
        for index, path in enumerate(command.motion.motion_files)
      ],
      "handoff_limits": dict(zip(EVAL_CHANNELS, handoff.tolist(), strict=True)),
      "oracle_margin_limits": dict(
        zip(EVAL_CHANNELS, oracle_margin.tolist(), strict=True)
      ),
    },
    "nominal": _strip_tensors(nominal),
    "recovery": _strip_tensors(recovery),
    "shuffled_reference": _strip_tensors(shuffled),
    "reference_degradation_ratio": degradation,
    "gates": gates,
  }
  output = args.output or args.checkpoint.with_name(
    f"{args.checkpoint.stem}_oracle_eval.json"
  )
  output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

  print("\nOracle acceptance report")
  print(f"  overall: {'PASS' if gates['overall'] else 'FAIL'}")
  print(
    f"  nominal survival {nominal['survival_rate']:.1%}, "
    f"within handoff {nominal['within_handoff_rate']:.1%}"
  )
  print(
    f"  recovery survival {recovery['survival_rate']:.1%}, "
    f"within handoff {recovery['within_handoff_rate']:.1%}"
  )
  print(
    "  reference action delta "
    f"{nominal['action_reference_test']['relative_rms_delta']:.1%}, "
    f"wrong-reference degradation {degradation:.2f}x"
  )
  print("\nTerminal p95 (nominal / recovery / half-budget limit)")
  for index, name in enumerate(EVAL_CHANNELS):
    print(
      f"  {name:>18}: {_p95(nominal)[index]:.4f} / "
      f"{_p95(recovery)[index]:.4f} / {oracle_margin[index]:.4f}"
    )
  failed = [name for name, passed in gates.items() if not passed]
  if failed:
    print(f"\n  failed gates: {', '.join(failed)}")
  print(f"  JSON: {output.resolve()}")
  wrapped.close()


if __name__ == "__main__":
  main()
