"""The state bank: every state the skills were measured actually being in.

A bridge episode needs two states and a clock. This module supplies the states, and it
supplies them from the skills themselves rather than from a motion capture corpus.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.dataset.dataset
    uv run python -m ...bridge.dataset.dataset --skills "('walk','run')" --steps 800

What is not attainable by construction is the pair: two individually valid states can
still be an impossible thing to ask for in the time given. That is the command term's
problem, not this one's, and mdp/commands.py says how it is handled.

##
# What is recorded
##

One row per environment per control step, in the layout the rest of this package uses:

    root_pos (3)  root_quat (4)  root_lin_vel (3)  root_ang_vel (3)  q (J)  qd (J)

Root positions have their environment's origin subtracted off the horizontal, so the
numbers are small and mean "where in its own tile the robot was". Height is left alone,
because height above the floor is a real part of a state and the bridge has to reach it.

Rows are dropped for the first `settle` steps after any reset. mjlab resets an
environment the moment it terminates, so the step after a fall is a robot standing
freshly at its default pose, a state the skill never chose to be in, and which would
otherwise flood the bank with identical standing poses drawn from every failure. This is
the same auto-reset hazard that makes success metrics read high on a broken task; here it
would quietly turn the bank into a pile of resets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.run import RUN_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ROBOT = "robot"

ROOT_STATE_DIM = 13
"""Root position, orientation, linear velocity and angular velocity, in that order."""

DEFAULT_BANK = Path("data") / "bridge" / "rollouts.npz"

LOG_ROOT = Path("logs") / "rsl_rl"


@dataclass(frozen=True)
class SkillSpec:
  """One skill, and where its trained weights might be."""

  task: str
  experiments: tuple[str, ...]
  """Log directories to look in, best first.

  Two names per skill because the experiments were renamed from parkour_* to g1_* without
  moving logs/rsl_rl, so the checkpoints that exist are mostly under the old name. Trying
  both here is cheaper than either orphaning them or moving them."""


SKILLS: dict[str, SkillSpec] = {
  "walk": SkillSpec(WALK_TASK_ID, ("g1_walk", "parkour_walk")),
  "run": SkillSpec(RUN_TASK_ID, ("g1_run", "parkour_run")),
  "jump": SkillSpec(JUMP_TASK_ID, ("g1_jump", "parkour_jump_rsi", "humanoid_jump")),
  "kick": SkillSpec(KICK_TASK_ID, ("g1_kick", "parkour_kick")),
  "push": SkillSpec(PUSH_TASK_ID, ("g1_push", "parkour_push")),
}


@dataclass
class BankCfg:
  """How the bank is collected."""

  skills: tuple[str, ...] = ("walk", "run", "jump")
  """Which skills to record. Defaults to the three that need nothing on the floor; kick
  and push work too and are left out only because a bridge has no reason to aim at a
  state whose meaning depends on where a crate was."""

  num_envs: int = 64
  steps: int = 500
  settle: int = 25
  """Control steps discarded after every reset, per environment."""

  path: Path = DEFAULT_BANK
  device: str = "cuda:0"
  checkpoints: tuple[str, ...] = ()
  """Explicit checkpoint paths, one per entry of `skills`, when the automatic search picks
  the wrong run. Empty means search."""


def state(robot: Entity) -> torch.Tensor:
  """(N, 13 + 2J). The one place this layout is written down."""
  data = robot.data
  return torch.cat(
    [
      data.root_link_pos_w,
      data.root_link_quat_w,
      data.root_link_lin_vel_w,
      data.root_link_ang_vel_w,
      data.joint_pos,
      data.joint_vel,
    ],
    dim=-1,
  )


def find_checkpoint(spec: SkillSpec, explicit: str | None = None) -> Path:
  """The newest checkpoint of a skill, said out loud.

  Printed rather than assumed. Picking by modification time has bitten this project
  before: a newer unrelated run left in logs/ outranks the one that was meant, and the
  comparison that follows is then between two things neither of which is what was asked
  for.
  """
  if explicit:
    path = Path(explicit)
    if not path.exists():
      raise SystemExit(f"No checkpoint at {path}.")
    return path
  for experiment in spec.experiments:
    found = sorted(
      (LOG_ROOT / experiment).rglob("model_*.pt"), key=lambda p: p.stat().st_mtime
    )
    if found:
      return found[-1]
  tried = ", ".join(str(LOG_ROOT / e) for e in spec.experiments)
  raise SystemExit(
    f"No checkpoint for '{spec.task}'. Looked under {tried}. Train it with "
    f"`uv run train {spec.task}`, or pass the path in `checkpoints`."
  )


def _rollout(
  name: str, spec: SkillSpec, cfg: BankCfg, checkpoint: str | None
) -> tuple[np.ndarray, np.ndarray]:
  """Drive one skill with its own weights, in its own environment, recording every step.

  The training config rather than the play one. A play config narrows command ranges and
  switches off the noise a skill was trained under, which makes for a tidier
  demonstration and a narrower bank: what is wanted here is the spread of states the skill
  actually occupies in service, including the ones it reaches when a command sits at the
  edge of its range.
  """
  from tensordict import TensorDict

  env_cfg = load_env_cfg(spec.task)
  env_cfg.scene.num_envs = cfg.num_envs
  agent_cfg = load_rl_cfg(spec.task)

  path = find_checkpoint(spec, checkpoint)
  print(f"[bank] {name}: {path}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(spec.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
  runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=cfg.device)
  policy = runner.get_inference_policy(device=cfg.device)

  robot: Entity = env.scene[ROBOT]
  origin = env.scene.env_origins[:, :2]
  obs, _ = env.reset()

  age = torch.zeros(cfg.num_envs, dtype=torch.long, device=cfg.device)
  rows: list[torch.Tensor] = []
  keep: list[torch.Tensor] = []
  for step in range(cfg.steps):
    # Only the policy call goes in inference mode. Stepping the environment inside it
    # turns every buffer it writes into an inference tensor, and the next reset cannot
    # write them.
    with torch.inference_mode():
      action = policy(
        TensorDict(obs, batch_size=[cfg.num_envs])  # ty: ignore[invalid-argument-type]
      )
    obs, _, terminated, truncated, _ = env.step(action)
    age = torch.where(terminated | truncated, torch.zeros_like(age), age + 1)

    here = state(robot).clone()
    here[:, 0:2] -= origin
    rows.append(here)
    keep.append(age >= cfg.settle)
    if (step + 1) % 100 == 0:
      print(f"[bank] {name}: {step + 1}/{cfg.steps}")

  env.close()
  states = torch.stack(rows, dim=0).flatten(0, 1)
  valid = torch.stack(keep, dim=0).flatten(0, 1)
  # Which environment each surviving row came from, so the split can hold whole
  # environments out rather than individual frames. Consecutive frames of one rollout are
  # nearly the same state, and a frame-level split would put a row's near twin on the
  # other side of it.
  env_id = torch.arange(cfg.num_envs, device=cfg.device).repeat(cfg.steps)[valid]
  return (
    states[valid].cpu().numpy().astype(np.float32),
    env_id.to(torch.int16).cpu().numpy(),
  )


def collect(cfg: BankCfg) -> Path:
  """Record every configured skill and write one npz."""
  states: list[np.ndarray] = []
  env_ids: list[np.ndarray] = []
  skill_ids: list[np.ndarray] = []
  fps = 0.0

  for index, name in enumerate(cfg.skills):
    if name not in SKILLS:
      raise SystemExit(f"Unknown skill '{name}'. Known: {', '.join(SKILLS)}.")
    spec = SKILLS[name]
    explicit = cfg.checkpoints[index] if index < len(cfg.checkpoints) else None

    env_cfg = load_env_cfg(spec.task)
    rate = 1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)
    if fps and abs(rate - fps) > 1e-6:
      raise SystemExit(
        f"'{name}' runs at {rate:.1f} Hz and the skills before it at {fps:.1f} Hz. A bank "
        f"mixing control rates has no single meaning for a deadline counted in steps."
      )
    fps = rate

    rows, envs = _rollout(name, spec, cfg, explicit)
    states.append(rows)
    env_ids.append(envs)
    skill_ids.append(np.full(len(rows), index, dtype=np.int16))
    print(f"[bank] {name}: {len(rows)} states")

  everything = np.concatenate(states)
  cfg.path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    cfg.path,
    states=everything,
    skill=np.concatenate(skill_ids),
    env_id=np.concatenate(env_ids),
    skill_names=np.asarray(cfg.skills),
    fps=np.asarray(fps),
  )
  print(f"[bank] wrote {cfg.path} ({len(everything)} states at {fps:.0f} Hz)")
  return cfg.path


@dataclass
class Bank:
  """A loaded bank, on the device, with one slice per skill."""

  states: torch.Tensor
  """(N, 13 + 2J)."""
  skill: torch.Tensor
  """(N,) index into `names`."""
  names: tuple[str, ...]
  fps: float

  @property
  def num_joints(self) -> int:
    return (self.states.shape[1] - ROOT_STATE_DIM) // 2

  def of(self, names: tuple[str, ...] | None) -> torch.Tensor:
    """Row indices belonging to these skills, or every row when `names` is None."""
    if names is None:
      return torch.arange(self.states.shape[0], device=self.states.device)
    mask = torch.zeros_like(self.skill, dtype=torch.bool)
    for name in names:
      if name not in self.names:
        raise ValueError(f"This bank holds {self.names}, not '{name}'.")
      mask |= self.skill == self.names.index(name)
    if not bool(mask.any()):
      raise ValueError(f"No states from {names} in this bank.")
    return mask.nonzero().flatten()


def load_bank(path: Path, device: str, split: str = "train", holdout: int = 8) -> Bank:
  """Read a bank and keep one side of the split.

  One environment in every `holdout` goes to 'eval'. The split is by environment, so an
  evaluation pair is drawn from rollouts no training pair was drawn from.

  It is worth being honest about what this split does and does not buy. Both sides come
  from the same policies, so it measures whether the bridge learned the task or the
  particular pairs it saw, and not whether it transfers to a skill it has never met.
  Nothing built out of rollouts can answer that second question; the other dataset is what
  will.
  """
  if split not in ("train", "eval"):
    raise ValueError(f"split is 'train' or 'eval', not '{split}'.")
  if not path.exists():
    raise SystemExit(
      f"No bank at {path}. Build one with "
      f"`uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.dataset.dataset`."
    )

  raw = np.load(path, allow_pickle=False)
  env_id = torch.from_numpy(raw["env_id"]).to(device).long()
  held = (env_id % holdout) == 0
  mask = held if split == "eval" else ~held

  bank = Bank(
    states=torch.from_numpy(raw["states"]).to(device)[mask],
    skill=torch.from_numpy(raw["skill"]).to(device).long()[mask],
    names=tuple(str(n) for n in raw["skill_names"]),
    fps=float(raw["fps"]),
  )
  print(f"[bank] {bank.states.shape[0]} states in '{split}' from {bank.names}")
  return bank


if __name__ == "__main__":
  collect(tyro.cli(BankCfg, config=mjlab.TYRO_FLAGS))
