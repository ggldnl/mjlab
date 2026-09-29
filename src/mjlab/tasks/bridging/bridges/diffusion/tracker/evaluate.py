"""Evaluate exact endpoint precision on held-out motion windows.

Run

    uv run python -m \
      mjlab.tasks.bridging.bridges.diffusion.tracker.evaluate \
      --checkpoint logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_30000.pt
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.bridges.dataset.dataset import find_checkpoint
from mjlab.tasks.bridging.bridges.diffusion.config import (
  motion_patterns,
  tracker_experiment,
  tracker_task_id,
)
from mjlab.tasks.bridging.bridges.diffusion.tracker.command import (
  TrackerCommand,
  TrackerCommandCfg,
)
from mjlab.tasks.bridging.bridges.diffusion.tracker.env_cfg import (
  COMMAND,
  tracker_env_cfg,
)
from mjlab.tasks.bridging.bridges.imitation.command import (
  CHANNELS,
  Tolerances,
)
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

DEFAULT_OUTPUT = Path("logs/benchmarks/diffusion_tracker/evaluation.json")


@dataclass(frozen=True)
class EvaluateCfg:
  checkpoint: Path | None = None
  robot: str = "g1"
  motions: tuple[str, ...] = ()
  output: Path = DEFAULT_OUTPUT
  sources: tuple[str, ...] | None = None
  batch: int = 256
  duration_s: float = 2.0
  seed: int = 42
  device: str = "cuda:0"


def _statistics(values: torch.Tensor) -> dict[str, dict[str, float]]:
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


def _print_report(report: dict[str, Any]) -> None:
  print(f"survival:       {report['survival_rate']:.1%}")
  print(f"strict arrival: {report['strict_arrival_rate']:.1%}")
  print("worst-channel pass rate:")
  for scale, rate in report["arrival_rate"].items():
    print(f"  {scale:>3}: {rate:.1%}")
  print("terminal error: p50 / p90 / tolerance")
  limits = report["configuration"]["tolerances"]
  for name in CHANNELS:
    values = report["terminal_error"][name]
    print(
      f"  {name:>18}: {values['p50']:.4f} / {values['p90']:.4f} / {limits[name]:.4f}"
    )


@torch.no_grad()
def evaluate(cfg: EvaluateCfg) -> dict[str, Any]:
  if cfg.batch < 1 or not 0.0 < cfg.duration_s <= 2.0:
    raise ValueError("batch must be positive and duration_s must be in (0, 2]")
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  checkpoint = find_checkpoint(
    (tracker_experiment(cfg.robot),),
    str(cfg.checkpoint) if cfg.checkpoint is not None else None,
    hint=f" Train one with `uv run train {tracker_task_id(cfg.robot)}`.",
  )
  env_cfg = tracker_env_cfg(
    play=True,
    split="eval",
    motion_patterns=cfg.motions or motion_patterns(cfg.robot, "val"),
    sources=cfg.sources,
    robot=cfg.robot,
  )
  env_cfg.scene.num_envs = cfg.batch
  env_cfg.auto_reset = False
  env_cfg.seed = cfg.seed
  env_cfg.terminations = {"route_done": env_cfg.terminations["route_done"]}
  command_cfg = cast(TrackerCommandCfg, env_cfg.commands[COMMAND])
  command_cfg.duration_s_range = (cfg.duration_s, cfg.duration_s)
  command_cfg.debug_vis = False

  agent_cfg = load_rl_cfg(tracker_task_id(cfg.robot))
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=cfg.device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=cfg.device,
    )
    policy = runner.get_inference_policy(device=cfg.device)
    command = cast(TrackerCommand, env.command_manager.get_term(COMMAND))
    observation = wrapped.get_observations()
    fell = torch.zeros(cfg.batch, dtype=torch.bool, device=cfg.device)
    endpoint_ticks = int(round(cfg.duration_s / env.step_dt))
    ticks = endpoint_ticks + command.post_steps
    print(
      f"[tracker] {cfg.batch} held-out windows, {endpoint_ticks} + "
      f"{command.post_steps} post-B ticks, checkpoint {checkpoint}"
    )
    done = torch.zeros(cfg.batch, dtype=torch.bool, device=cfg.device)
    errors = None
    for tick in range(ticks):
      observation, _, done, _ = wrapped.step(policy(observation))
      fell |= env.scene["robot"].data.projected_gravity_b[:, 2] > -0.7
      if tick + 1 == endpoint_ticks:
        errors = command.target_errors().clone()
    if not bool(done.all()):
      raise RuntimeError(
        "Not every evaluation window completed its post-B continuation"
      )
    if errors is None:
      raise RuntimeError("Tracker evaluation did not reach the endpoint")

    tolerances = command.tolerances
    worst = (errors / tolerances).amax(dim=-1)
    survived = ~fell
    valid = errors[survived]
    if valid.shape[0] == 0:
      raise RuntimeError("Every evaluation rollout fell before the endpoint")
    report: dict[str, Any] = {
      "configuration": {
        **asdict(cfg),
        "checkpoint": str(checkpoint.resolve()),
        "motions": cfg.motions,
        "output": str(cfg.output.resolve()),
        "task": tracker_task_id(cfg.robot),
        "fps": 1.0 / env.step_dt,
        "tolerances": asdict(Tolerances()),
      },
      "survival_rate": survived.float().mean().item(),
      "strict_arrival_rate": (survived & (worst <= 1.0)).float().mean().item(),
      "arrival_rate": {
        f"{scale}x": (survived & (worst <= scale)).float().mean().item()
        for scale in (1, 2, 4, 8)
      },
      "worst_channel_multiple": {
        "p50": torch.quantile(worst[survived], 0.50).item(),
        "p90": torch.quantile(worst[survived], 0.90).item(),
        "p95": torch.quantile(worst[survived], 0.95).item(),
        "max": worst[survived].max().item(),
      },
      "terminal_error": _statistics(valid),
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    _print_report(report)
    print(f"report: {cfg.output.resolve()}")
    return report
  finally:
    wrapped.close()


if __name__ == "__main__":
  evaluate(tyro.cli(EvaluateCfg, config=mjlab.TYRO_FLAGS))
