"""Evaluate exact endpoint precision on held-out dynamic rollout windows.

Run

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.evaluate \
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
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  find_checkpoint,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker import (
  TRACKER_EXPERIMENT,
  TRACKER_TASK_ID,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.command import (
  TrackerCommand,
  TrackerCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.env_cfg import (
  COMMAND,
  tracker_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  CHANNELS,
  Tolerances,
)
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

DEFAULT_OUTPUT = Path("logs/benchmarks/diffusion_tracker/evaluation.json")


@dataclass(frozen=True)
class EvaluateCfg:
  checkpoint: Path | None = None
  dataset: Path = DEFAULT_DATASET
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
    (TRACKER_EXPERIMENT,),
    str(cfg.checkpoint) if cfg.checkpoint is not None else None,
    hint=f" Train one with `uv run train {TRACKER_TASK_ID}`.",
  )
  env_cfg = tracker_env_cfg(
    play=True, split="eval", dataset_path=cfg.dataset, sources=cfg.sources
  )
  env_cfg.scene.num_envs = cfg.batch
  env_cfg.auto_reset = False
  env_cfg.seed = cfg.seed
  env_cfg.terminations = {"deadline": env_cfg.terminations["deadline"]}
  command_cfg = cast(TrackerCommandCfg, env_cfg.commands[COMMAND])
  command_cfg.duration_s_range = (cfg.duration_s, cfg.duration_s)
  command_cfg.debug_vis = False

  agent_cfg = load_rl_cfg(TRACKER_TASK_ID)
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
    ticks = int(round(cfg.duration_s / env.step_dt))
    print(
      f"[tracker] {cfg.batch} held-out windows, {ticks} ticks, checkpoint {checkpoint}"
    )
    done = torch.zeros(cfg.batch, dtype=torch.bool, device=cfg.device)
    for _ in range(ticks):
      observation, _, done, _ = wrapped.step(policy(observation))
      fell |= env.scene["robot"].data.projected_gravity_b[:, 2] > -0.7
    if not bool(done.all()):
      raise RuntimeError("Not every evaluation window ended at its configured deadline")

    errors = command.target_errors()
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
        "dataset": str(cfg.dataset.resolve()),
        "output": str(cfg.output.resolve()),
        "task": TRACKER_TASK_ID,
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
