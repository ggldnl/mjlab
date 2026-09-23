"""Compare perfect-reference tracking with recorded open-loop action replay.

Run:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.tests.experiments.tracking_ceiling
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
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
  DEFAULT_DATASET,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  CHANNELS,
  Tolerances,
  channel_errors,
  upper_body_mask,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.resume import restore_action
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import state
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

DEFAULT_CHECKPOINT = Path(
  "logs/rsl_rl/g1_cvae_oracle/2026-09-21_15-34-07/model_11500.pt"
)
DEFAULT_OUTPUT = Path("logs/benchmarks/bridge_ceiling/model_11500.json")


@dataclass
class ExperimentCfg:
  checkpoint: Path = DEFAULT_CHECKPOINT
  dataset: Path = DEFAULT_DATASET
  output: Path = DEFAULT_OUTPUT
  batch: int = 128
  duration: int = 40
  capture_steps: int = 5
  seed: int = 42
  device: str = "cuda:0"


def _write_states(robot: Entity, states: torch.Tensor, origins: torch.Tensor) -> None:
  placed = states.clone()
  placed[:, :2] += origins[:, :2]
  joints = robot.data.joint_pos.shape[1]
  robot.write_root_state_to_sim(placed[:, :13])
  robot.write_joint_state_to_sim(placed[:, 13 : 13 + joints], placed[:, 13 + joints :])


@torch.no_grad()
def _build_reference(
  env: ManagerBasedRlEnv,
  command: MotionSetCommand,
  paths: torch.Tensor,
) -> SimpleNamespace:
  """Run FK on recorded states and return one reference clip per environment."""
  robot: Entity = env.scene["robot"]
  origins = env.scene.env_origins
  body_pos: list[torch.Tensor] = []
  body_quat: list[torch.Tensor] = []
  body_lin_vel: list[torch.Tensor] = []
  body_ang_vel: list[torch.Tensor] = []
  for step in range(paths.shape[1]):
    _write_states(robot, paths[:, step], origins)
    env.sim.forward()
    indexes = command.body_indexes
    body_pos.append((robot.data.body_link_pos_w[:, indexes] - origins[:, None]).clone())
    body_quat.append(robot.data.body_link_quat_w[:, indexes].clone())
    body_lin_vel.append(robot.data.body_link_lin_vel_w[:, indexes].clone())
    body_ang_vel.append(robot.data.body_link_ang_vel_w[:, indexes].clone())

  length = paths.shape[1]
  starts = torch.arange(env.num_envs, device=env.device) * length
  lengths = torch.full((env.num_envs,), length, dtype=torch.long, device=env.device)
  joints = robot.data.joint_pos.shape[1]
  return SimpleNamespace(
    joint_pos=paths[:, :, 13 : 13 + joints].reshape(-1, joints),
    joint_vel=paths[:, :, 13 + joints :].reshape(-1, joints),
    body_pos_w=torch.stack(body_pos, dim=1).flatten(0, 1),
    body_quat_w=torch.stack(body_quat, dim=1).flatten(0, 1),
    body_lin_vel_w=torch.stack(body_lin_vel, dim=1).flatten(0, 1),
    body_ang_vel_w=torch.stack(body_ang_vel, dim=1).flatten(0, 1),
    motion_lengths=lengths,
    motion_starts=starts,
    motion_ends=starts + lengths,
    motion_files=tuple(f"dataset_{index}" for index in range(env.num_envs)),
    time_step_total=env.num_envs * length,
  )


def _stats(values: torch.Tensor) -> dict[str, dict[str, float]]:
  return {
    name: {
      "mean": values[:, index].mean().item(),
      "p50": torch.quantile(values[:, index], 0.50).item(),
      "p90": torch.quantile(values[:, index], 0.90).item(),
      "p95": torch.quantile(values[:, index], 0.95).item(),
      "max": values[:, index].max().item(),
    }
    for index, name in enumerate(CHANNELS)
  }


def _summary(
  exact: torch.Tensor,
  window: torch.Tensor,
  failed: torch.Tensor,
  limits: torch.Tensor,
) -> dict[str, Any]:
  exact_scale = (exact / limits).amax(dim=-1)
  window_scale = (window / limits).amax(dim=-1)
  survived = ~failed
  valid_exact = exact[survived]
  valid_window = window[survived]
  valid_scale = exact_scale[survived]
  if valid_exact.shape[0] == 0:
    raise RuntimeError("Every rollout failed before the endpoint")
  return {
    "survival_rate": survived.float().mean().item(),
    "exact": _stats(valid_exact),
    "best_in_capture_window": _stats(valid_window),
    "exact_capture_rate": {
      f"{scale}x": (survived & (exact_scale <= scale)).float().mean().item()
      for scale in (1, 2, 4, 8)
    },
    "window_capture_rate": {
      f"{scale}x": (survived & (window_scale <= scale)).float().mean().item()
      for scale in (1, 2, 4, 8)
    },
    "exact_worst_channel_multiple": {
      "p50": torch.quantile(valid_scale, 0.50).item(),
      "p90": torch.quantile(valid_scale, 0.90).item(),
      "p95": torch.quantile(valid_scale, 0.95).item(),
    },
  }


def _print_summary(name: str, result: dict[str, Any]) -> None:
  print(f"\n{name}")
  print(f"  survival: {result['survival_rate']:.1%}")
  print(
    "  exact capture: "
    + ", ".join(
      f"{scale} {rate:.1%}" for scale, rate in result["exact_capture_rate"].items()
    )
  )
  print(
    "  window capture: "
    + ", ".join(
      f"{scale} {rate:.1%}" for scale, rate in result["window_capture_rate"].items()
    )
  )
  print("  exact terminal errors: p50 / p90 / limit")
  limits = asdict(Tolerances())
  for channel in CHANNELS:
    values = result["exact"][channel]
    print(
      f"    {channel:>18}: {values['p50']:.4f} / {values['p90']:.4f} / "
      f"{limits[channel]:.4f}"
    )


@torch.no_grad()
def run(cfg: ExperimentCfg) -> dict[str, Any]:
  if not cfg.checkpoint.is_file():
    raise FileNotFoundError(cfg.checkpoint)
  if cfg.batch < 1 or cfg.duration < 1 or cfg.capture_steps < 0:
    raise ValueError(
      "batch and duration must be positive; capture_steps cannot be negative"
    )

  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  dataset = load_dataset(cfg.dataset, cfg.device, "eval")
  if dataset.previous_action is None:
    raise ValueError("The dataset needs previous_action for experiment 0B")
  reference_length = cfg.duration + cfg.capture_steps + 6
  segments = dataset.segments(reference_length - 1, reference_length - 1)
  picked = torch.randint(segments.starts.numel(), (cfg.batch,), device=cfg.device)
  positions = segments.starts[picked]
  offsets = torch.arange(reference_length, device=cfg.device)
  rows = segments.order[positions[:, None] + offsets]
  paths = dataset.states[rows]
  recorded_actions = dataset.previous_action[rows]

  env_cfg = load_env_cfg(ORACLE_TASK_ID, play=True)
  env_cfg.scene.num_envs = cfg.batch * 2
  env_cfg.auto_reset = True
  env_cfg.seed = cfg.seed
  motion_cfg = cast(MotionSetCommandCfg, env_cfg.commands["motion"])
  motion_cfg.debug_vis = False
  motion_cfg.balanced_sampling = False
  agent_cfg = load_rl_cfg(ORACLE_TASK_ID)
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    runner = OracleRunner(wrapped, asdict(agent_cfg), device=cfg.device)
    runner.load(
      str(cfg.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=cfg.device,
    )
    policy = runner.get_inference_policy(device=cfg.device)
    command = cast(MotionSetCommand, env.command_manager.get_term("motion"))

    paired_paths = paths.repeat(2, 1, 1)
    command.motion = cast(Any, _build_reference(env, command, paired_paths))
    command.motion_ids[:] = torch.arange(env.num_envs, device=env.device)
    command.motion_end_steps[:] = command.motion.motion_ends
    command.time_steps[:] = command.motion.motion_starts
    command.update_relative_body_poses()

    robot: Entity = env.scene["robot"]
    _write_states(robot, paired_paths[:, 0], env.scene.env_origins)
    restore_action(env, recorded_actions[:, 0].repeat(2, 1))
    env.sim.forward()
    obs = wrapped.get_observations()

    target = paths[:, cfg.duration].repeat(2, 1)
    target[:, :2] += env.scene.env_origins[:, :2]
    upper = upper_body_mask(tuple(robot.joint_names), env.device)
    limits = Tolerances().tensor(env.device)
    exact: torch.Tensor | None = None
    capture: list[torch.Tensor] = []
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    print(
      f"[ceiling] {cfg.batch} held-out windows, {cfg.duration} ticks, "
      f"{cfg.capture_steps} capture ticks"
    )
    for step in range(1, cfg.duration + cfg.capture_steps + 1):
      oracle_action = policy(obs)[: cfg.batch]
      replay_index = min(step, recorded_actions.shape[1] - 1)
      action = torch.cat((oracle_action, recorded_actions[:, replay_index]), dim=0)
      obs, _, done, _ = wrapped.step(action)
      failed |= done.bool()
      errors = channel_errors(state(env), target, upper)
      if step == cfg.duration:
        exact = errors.clone()
      if step >= cfg.duration:
        capture.append(errors)

    if exact is None:
      raise RuntimeError("The exact endpoint was not evaluated")
    capture_errors = torch.stack(capture)
    normalized = capture_errors / limits
    best_step = normalized.amax(dim=-1).argmin(dim=0)
    env_index = torch.arange(env.num_envs, device=env.device)
    best = capture_errors[best_step, env_index]
    oracle = _summary(
      exact[: cfg.batch], best[: cfg.batch], failed[: cfg.batch], limits
    )
    replay = _summary(
      exact[cfg.batch :], best[cfg.batch :], failed[cfg.batch :], limits
    )
    report = {
      "configuration": {
        **asdict(cfg),
        "checkpoint": str(cfg.checkpoint.resolve()),
        "dataset": str(cfg.dataset.resolve()),
        "output": str(cfg.output.resolve()),
        "fps": dataset.fps,
        "channels": CHANNELS,
        "limits": asdict(Tolerances()),
      },
      "experiment_0a_perfect_reference_tracker": oracle,
      "experiment_0b_recorded_action_replay": replay,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    _print_summary("0A perfect-reference tracker", oracle)
    _print_summary("0B recorded-action replay", replay)
    print(f"\nReport: {cfg.output.resolve()}")
    return report
  finally:
    wrapped.close()


if __name__ == "__main__":
  run(tyro.cli(ExperimentCfg, config=mjlab.TYRO_FLAGS))
