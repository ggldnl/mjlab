"""Score a trained bridge, always next to a robot that does nothing.

Run

1. Score a checkpoint against the statue baseline. The newest checkpoint is picked when
   none is named, and the path is printed.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate \
      --episodes 2048 \
      --checkpoint logs/rsl_rl/g1_bridge/<run>/model_5000.pt

2. Measure the statue on its own, before there is anything to compare it to.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate --policies "('statue',)"

3. Watch instead of read. The player draws the target as a translucent robot and leaves it
   standing, so the gap to the real robot at the deadline is the arrival error.

    uv run play Mjlab-G1-Bridge
    uv run play Mjlab-G1-Bridge --agent zero

Reading the table

    arrived            every channel inside tolerance at the deadline, over every episode
                       including the ones that ended on the floor. The honest number
    score              the smooth version the reward is built on
    reached deadline   share of episodes that got that far
    failed early       share that terminated instead
    err *              per channel medians, over the episodes that reached their deadline
                       only. An episode that fell has no arrival to be wrong about, and
                       counting it as zero would flatter a policy that falls often. Read
                       these against reached deadline

score is a blend of exponential kernels and has no absolute meaning: widen every tolerance
and a robot doing nothing approaches 1.0. It is only ever a number next to another number,
which is why the statue column is always printed. arrived stands alone because it is not a
kernel: the requirements are physical, chosen for what the next skill needs, and nothing
about them moves when the dataset is rebuilt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridge import BRIDGE_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  DEFAULT_DATASET,
  LOG_ROOT,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  CHANNELS,
  BridgeCommand,
  BridgeCommandCfg,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

COMMAND = "bridge"


@dataclass
class EvalCfg:
  checkpoint: str | None = None
  """Explicit checkpoint. Empty takes the newest under logs/rsl_rl/g1_bridge and prints
  the path, because picking by modification time has loaded the wrong policy here before."""

  episodes: int = 1024
  num_envs: int = 256
  dataset: Path = DEFAULT_DATASET
  split: str = "eval"
  device: str = "cuda:0"
  seed: int = 0

  diagnose: bool = False
  """Report how the corpus sits against the arrival requirements, instead of evaluating a
  policy. Needs no checkpoint.

  Read it, do not act on it. The requirements are what a hand-over needs, this says how
  far a robot doing nothing is from meeting them, which tells you which channels are the
  hard ones and which will read zero for a while."""

  policies: tuple[str, ...] = ("bridge", "statue")
  """Which policies to run, one column of the table each. bridge is the trained
  checkpoint, statue is a robot holding its default pose for the whole window.

  The statue alone needs no checkpoint. That is the point: the floor a trained bridge has
  to clear is measurable before there is anything to measure against it."""

  per_joint: bool = False
  """Print terminal joint position and velocity errors for every joint."""


@dataclass
class Result:
  """One rollout's worth of episodes."""

  name: str
  arrived: torch.Tensor
  score: torch.Tensor
  reached: torch.Tensor
  fell: torch.Tensor
  errors: torch.Tensor
  """(episodes, len(CHANNELS)). Rows for episodes that never reached a deadline are
  unfilled, and are excluded by reached."""
  joint_pos_errors: torch.Tensor
  joint_vel_errors: torch.Tensor


def _build(cfg: EvalCfg) -> ManagerBasedRlEnv:
  env_cfg = load_env_cfg(BRIDGE_TASK_ID, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  command = env_cfg.commands[COMMAND]
  assert isinstance(command, BridgeCommandCfg)
  command.dataset_path = cfg.dataset
  command.split = cfg.split
  # The rollout reads the arrival latched at the deadline, and an auto-reset would draw
  # the next window and clear it before step returns
  env_cfg.auto_reset = False
  return ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)


def _command(env: ManagerBasedRlEnv) -> BridgeCommand:
  term = env.command_manager.get_term(COMMAND)
  assert isinstance(term, BridgeCommand)
  return term


def _find_checkpoint(explicit: str | None) -> Path:
  if explicit:
    path = Path(explicit)
    if not path.exists():
      raise SystemExit(f"No checkpoint at {path}.")
    return path
  found = sorted(
    (LOG_ROOT / "g1_bridge").rglob("model_*.pt"), key=lambda p: p.stat().st_mtime
  )
  if not found:
    raise SystemExit(
      f"No checkpoint under {LOG_ROOT / 'g1_bridge'}. Train one with "
      f"`uv run train {BRIDGE_TASK_ID}`."
    )
  return found[-1]


def _trained(env: ManagerBasedRlEnv, cfg: EvalCfg):
  from tensordict import TensorDict

  agent_cfg = load_rl_cfg(BRIDGE_TASK_ID)
  path = _find_checkpoint(cfg.checkpoint)
  print(f"[eval] policy: {path}")

  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(BRIDGE_TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
  runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=cfg.device)
  inference = runner.get_inference_policy(device=cfg.device)

  def policy(obs):
    with torch.inference_mode():
      return inference(TensorDict(obs, batch_size=[cfg.num_envs]))

  return policy


def _statue(env: ManagerBasedRlEnv, cfg: EvalCfg):
  """Hold the default pose. The action term offsets from the default, so a zero action is
  exactly that: the do nothing baseline, not merely a weak policy."""
  shape = env.action_space.shape
  del cfg

  def policy(obs):
    del obs
    return torch.zeros(shape, device=env.device)

  return policy


def rollout(env: ManagerBasedRlEnv, policy, cfg: EvalCfg, name: str) -> Result:
  """Run windows until episodes of them have finished, recording each arrival."""
  command = _command(env)
  torch.manual_seed(cfg.seed)
  obs, _ = env.reset(seed=cfg.seed)

  arrived: list[torch.Tensor] = []
  score: list[torch.Tensor] = []
  reached: list[torch.Tensor] = []
  fell: list[torch.Tensor] = []
  errors: list[torch.Tensor] = []
  joint_pos_errors: list[torch.Tensor] = []
  joint_vel_errors: list[torch.Tensor] = []
  done_count = 0

  while done_count < cfg.episodes:
    action = policy(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    done = (terminated | truncated).nonzero().flatten()
    if done.numel():
      # Read before the reset. errors_now latches all of this at the deadline, and place
      # clears it for the next window
      arrived.append(command.arrived[done].clone())
      score.append(command.score[done].clone())
      reached.append(command.reached[done].clone())
      fell.append(terminated[done].float().clone())
      errors.append(command.final[done].clone())
      joint_pos_errors.append(command.final_joint_pos[done].clone())
      joint_vel_errors.append(command.final_joint_vel[done].clone())
      done_count += int(done.numel())
      obs, _ = env.reset(env_ids=done)
      print(f"[eval] {name}: {done_count}/{cfg.episodes}", end="\r")

  print()
  return Result(
    name=name,
    arrived=torch.cat(arrived)[: cfg.episodes],
    score=torch.cat(score)[: cfg.episodes],
    reached=torch.cat(reached)[: cfg.episodes],
    fell=torch.cat(fell)[: cfg.episodes],
    errors=torch.cat(errors)[: cfg.episodes],
    joint_pos_errors=torch.cat(joint_pos_errors)[: cfg.episodes],
    joint_vel_errors=torch.cat(joint_vel_errors)[: cfg.episodes],
  )


def report(
  results: list[Result], joint_names: tuple[str, ...], per_joint: bool
) -> None:
  """One row per number, one column per policy."""
  width = max(len(r.name) for r in results) + 2
  head = "metric".ljust(22) + "".join(r.name.rjust(width) for r in results)
  print()
  print(head)
  print("-" * len(head))

  def row(label: str, values: list[float]) -> None:
    print(label.ljust(22) + "".join(f"{v:>{width}.3f}" for v in values))

  row("arrived", [r.arrived.mean().item() for r in results])
  row("score", [r.score.mean().item() for r in results])
  row("reached deadline", [r.reached.mean().item() for r in results])
  row("failed early", [r.fell.mean().item() for r in results])
  print()
  for index, channel in enumerate(CHANNELS):
    values = []
    for r in results:
      kept = r.errors[r.reached > 0, index]
      values.append(kept.median().item() if kept.numel() else float("nan"))
    row(f"err {channel}", values)
  if per_joint:
    print()
    for index, name in enumerate(joint_names):
      pos = []
      vel = []
      for result in results:
        kept = result.reached > 0
        pos.append(
          result.joint_pos_errors[kept, index].median().item()
          if bool(kept.any())
          else float("nan")
        )
        vel.append(
          result.joint_vel_errors[kept, index].median().item()
          if bool(kept.any())
          else float("nan")
        )
      row(f"err q {name}", pos)
      row(f"err qd {name}", vel)
  print()
  print(
    "Channel errors are medians over the episodes that reached their deadline. "
    "Windows are drawn from the same distribution for every column but are not the "
    "same windows, so read the columns as distributions."
  )


def diagnose(env: ManagerBasedRlEnv, cfg: EvalCfg) -> None:
  """How far a robot that does nothing is from each arrival requirement.

  Nothing is stepped. The error at the instant a window opens is what a statue would still
  have at its deadline, so it is the do nothing baseline for every channel at once.

  There is nothing to paste back into the config. An earlier version printed a suggested
  Tolerances block at half the measured gap, which made the definition of a successful
  hand-over a function of how hard the corpus currently was: rebuild the dataset, get a
  different definition, and every number previously reported quietly changed meaning. The
  requirements are physical now and stay put. This only says where the task sits against
  them.
  """
  from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
    arrival_score,
    channel_errors,
  )

  command = _command(env)
  torch.manual_seed(cfg.seed)

  rows = []
  drawn = 0
  while drawn < cfg.episodes:
    env.reset()
    rows.append(
      channel_errors(command.state_now(), command.target, command.arms).clone()
    )
    drawn += cfg.num_envs
  gaps = torch.cat(rows)

  print()
  print(
    f"{'channel':<16}{'requires':>11}{'statue misses by':>19}{'statue meets it':>18}"
  )
  print("-" * 64)
  for index, channel in enumerate(CHANNELS):
    required = float(command.tolerances[index])
    median = gaps[:, index].median().item()
    met = (gaps[:, index] <= command.tolerances[index]).float().mean().item()
    note = ""
    if required >= median:
      note = "  <- not the hard part"
    elif required * 10.0 < median:
      note = "  <- out of reach for now"
    print(f"{channel:<16}{required:>11.2f}{median:>19.3f}{met:>17.2f}{note}")

  print()
  print(
    f"A statue scores {arrival_score(gaps, command.tolerances).mean().item():.3f} and "
    f"arrives on {(gaps <= command.tolerances).all(dim=-1).float().mean().item():.3f} "
    "of windows."
  )
  print(
    "That score is the floor a trained bridge has to clear. It is not comparable across "
    "different tolerances, which is why evaluate always prints it beside the policy."
  )


def main(cfg: EvalCfg) -> None:
  env = _build(cfg)
  try:
    if cfg.diagnose:
      diagnose(env, cfg)
      return
    builders = {"bridge": _trained, "statue": _statue}
    unknown = set(cfg.policies) - set(builders)
    if unknown:
      raise SystemExit(
        f"Unknown policies {sorted(unknown)}. Known: {sorted(builders)}."
      )
    robot = env.scene["robot"]
    report(
      [rollout(env, builders[n](env, cfg), cfg, n) for n in cfg.policies],
      tuple(robot.joint_names),
      cfg.per_joint,
    )
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(EvalCfg, config=mjlab.TYRO_FLAGS))
