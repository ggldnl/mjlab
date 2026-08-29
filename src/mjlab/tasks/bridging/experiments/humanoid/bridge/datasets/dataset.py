"""What a dataset is, plus the rollout driver every source shares.

A dataset is a table of states the robot was measured being in, under physics, with a
policy holding it up. That is the only property the bridge needs: both ends of a window
are reachable because a robot reached them.

Every source builds one the same way, by driving a trained policy and writing down what
happens, so the driving lives here. A source module only says which policy, in which
environment, with what on the floor.

Row layout, one row per environment per control step:

    root_pos (3)  root_quat (4)  root_lin_vel (3)  root_ang_vel (3)  q (J)  qd (J)

Root position has the environment origin subtracted off x and y, so the numbers are small
and mean "where in its own tile". Height is untouched, since height above the floor is
part of the state the bridge has to reach.

Rows in the first `settle` steps after a reset are dropped. mjlab resets the instant an
environment terminates, so the step after a fall is a robot standing at its default pose,
and without this the dataset fills up with one identical standing pose per failure.

Three extra columns, none of them read by training today:

    source  which policy or clip the row came from
    frame   control steps since that row's episode started. Two rows of one episode are a
            start and target one robot got between, and their frame gap is a deadline it
            met. mdp/commands.py draws deadlines and checks them instead.
    goal    every command term's value at that step, side by side. What the skill was
            being asked for while it was in that state. For the selector: a walk state
            produced at 2 m/s is only an entry point for a walk about to be asked for
            2 m/s. Width is per source and means nothing across sources. Older datasets
            lack the column; readers treat absent as unknown, not as an error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

ROBOT = "robot"

ROOT_STATE_DIM = 13
"""Root position, orientation, linear velocity, angular velocity, in that order."""

DATASET_ROOT = Path("data") / "bridge"

DEFAULT_DATASET = DATASET_ROOT / "rollouts.npz"
"""The skills dataset. Every config points here unless told otherwise."""

TRACKER_DATASET = DATASET_ROOT / "tracker.npz"
"""The tracker dataset, built from a motion tracker following LAFAN1."""

LOG_ROOT = Path("logs") / "rsl_rl"


@dataclass
class RolloutCfg:
  """How much to record. Shared by every source."""

  num_envs: int = 64
  steps: int = 500
  settle: int = 25
  """Control steps discarded after every reset, per environment."""
  device: str = "cuda:0"


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


def commanded(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Every active command term's value, side by side. (N, G).

  Order and width are whatever the manager happens to hold, so a row is only comparable
  against another row of the same skill. A skill with no command term gives a zero width
  column, which is the honest answer: there is nothing to condition on.
  """
  values = [
    env.command_manager.get_command(n) for n in env.command_manager.active_terms
  ]
  present = [v for v in values if v is not None]
  if not present:
    return torch.zeros(env.num_envs, 0, device=env.device)
  return torch.cat(present, dim=-1)


def control_rate(env_cfg: ManagerBasedRlEnvCfg) -> float:
  """Hz. A deadline is counted in control steps, so a dataset has to know this."""
  return 1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)


def find_checkpoint(
  experiments: tuple[str, ...], explicit: str | None = None, hint: str = ""
) -> Path:
  """The newest checkpoint under the first of these experiments that has one.

  The caller prints the path it got. Picking by modification time has gone wrong here
  before: an unrelated newer run left in logs/ outranks the one that was meant.
  """
  if explicit:
    path = Path(explicit)
    if not path.exists():
      raise SystemExit(f"No checkpoint at {path}.")
    return path
  for experiment in experiments:
    found = sorted(
      (LOG_ROOT / experiment).rglob("model_*.pt"), key=lambda p: p.stat().st_mtime
    )
    if found:
      return found[-1]
  tried = ", ".join(str(LOG_ROOT / e) for e in experiments)
  raise SystemExit(f"No checkpoint found. Looked under {tried}.{hint}")


def record(
  task: str,
  env_cfg: ManagerBasedRlEnvCfg,
  checkpoint: Path,
  cfg: RolloutCfg,
  label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Drive one trained policy in one environment, recording every control step.

  Returns the states, the environment each row came from, how many steps into its episode
  it was, and what it was commanded to do at the time.

  The caller configures `env_cfg` first, and that is the only difference between sources: a
  skill wants its own environment untouched, a tracker wants the same environment with a
  different clip in it.

  Always the training config, never the play one. Play narrows command ranges and drops the
  noise the policy trained under, which gives a tidier demo and a narrower dataset. What is
  wanted here is the full spread of states the policy occupies in service, including the
  ones at the edge of its command range.
  """
  from tensordict import TensorDict

  env_cfg.scene.num_envs = cfg.num_envs
  agent_cfg = load_rl_cfg(task)
  print(f"[dataset] {label}: {checkpoint}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=cfg.device
  )
  policy = runner.get_inference_policy(device=cfg.device)

  robot: Entity = env.scene[ROBOT]
  origin = env.scene.env_origins[:, :2]
  obs, _ = env.reset()

  age = torch.zeros(cfg.num_envs, dtype=torch.long, device=cfg.device)
  rows: list[torch.Tensor] = []
  ages: list[torch.Tensor] = []
  goals: list[torch.Tensor] = []
  keep: list[torch.Tensor] = []
  for step in range(cfg.steps):
    # Only the policy call goes in inference mode. Stepping the env inside it marks every
    # buffer it writes as an inference tensor, and the next reset cannot write them
    with torch.inference_mode():
      action = policy(
        TensorDict(obs, batch_size=[cfg.num_envs])  # ty: ignore[invalid-argument-type]
      )
    obs, _, terminated, truncated, _ = env.step(action)
    age = torch.where(terminated | truncated, torch.zeros_like(age), age + 1)

    here = state(robot).clone()
    here[:, 0:2] -= origin
    rows.append(here)
    ages.append(age.clone())
    # Read after the step, so this is the command the policy was following when it
    # produced the state, not one drawn for the episode about to start
    goals.append(commanded(env).clone())
    keep.append(age >= cfg.settle)
    if (step + 1) % 100 == 0:
      print(f"[dataset] {label}: {step + 1}/{cfg.steps}")

  env.close()
  states = torch.stack(rows, dim=0).flatten(0, 1)
  frames = torch.stack(ages, dim=0).flatten(0, 1)
  commands = torch.stack(goals, dim=0).flatten(0, 1)
  valid = torch.stack(keep, dim=0).flatten(0, 1)
  # Which environment each surviving row came from, so load_dataset can hold whole
  # environments out rather than individual frames
  env_id = torch.arange(cfg.num_envs, device=cfg.device).repeat(cfg.steps)[valid]
  return (
    states[valid].cpu().numpy().astype(np.float32),
    env_id.to(torch.int16).cpu().numpy(),
    frames[valid].to(torch.int32).cpu().numpy(),
    commands[valid].cpu().numpy().astype(np.float32),
  )


def write(
  path: Path,
  states: list[np.ndarray],
  env_ids: list[np.ndarray],
  frames: list[np.ndarray],
  sources: list[np.ndarray],
  names: tuple[str, ...],
  fps: float,
  goals: list[np.ndarray] | None = None,
) -> Path:
  """One npz, in the layout `load_dataset` expects.

  The `skill` and `skill_names` keys keep their names even though a tracker dataset puts
  clip names in them. Renaming would orphan the datasets already on disk.

  Sources have different command widths, so `goal` is padded to the widest and `goal_dim`
  says how much of each row is real. Padded into one table rather than one array per
  source, because one row per state is the layout everything downstream indexes by.
  """
  everything = np.concatenate(states)
  path.parent.mkdir(parents=True, exist_ok=True)
  columns: dict[str, Any] = {
    "states": everything,
    "skill": np.concatenate(sources),
    "env_id": np.concatenate(env_ids),
    "frame": np.concatenate(frames),
    "skill_names": np.asarray(names),
    "fps": np.asarray(fps),
  }
  if goals is not None:
    width = max(g.shape[1] for g in goals)
    columns["goal"] = np.concatenate(
      [np.pad(g, ((0, 0), (0, width - g.shape[1]))) for g in goals]
    ).astype(np.float32)
    columns["goal_dim"] = np.asarray([g.shape[1] for g in goals], dtype=np.int16)
  np.savez(path, **columns)
  print(f"[dataset] wrote {path} ({len(everything)} states at {fps:.0f} Hz)")
  return path


@dataclass
class Dataset:
  """A loaded dataset, on the device, tagged with the source each row came from."""

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
    """Row indices belonging to these sources, or every row when `names` is None."""
    if names is None:
      return torch.arange(self.states.shape[0], device=self.states.device)
    mask = torch.zeros_like(self.skill, dtype=torch.bool)
    for name in names:
      if name not in self.names:
        raise ValueError(f"This dataset holds {self.names}, not '{name}'.")
      mask |= self.skill == self.names.index(name)
    if not bool(mask.any()):
      raise ValueError(f"No states from {names} in this dataset.")
    return mask.nonzero().flatten()


def load_dataset(
  path: Path, device: str, split: str = "train", holdout: int = 8
) -> Dataset:
  """Read a dataset and keep one side of the split.

  One environment in every `holdout` goes to 'eval'. Splitting by environment rather than by
  frame keeps evaluation pairs out of every rollout a training pair came from. Consecutive
  frames of one rollout are nearly the same state, so a frame level split would put a row's
  near twin on the other side of it.

  For the skills dataset both sides come from the same policies, so this measures whether
  the bridge learned the task or the particular pairs it saw. It says nothing about transfer
  to a skill it never met. tracker.py is what answers that.
  """
  if split not in ("train", "eval"):
    raise ValueError(f"split is 'train' or 'eval', not '{split}'.")
  if not path.exists():
    raise SystemExit(
      f"No dataset at {path}. Build one with `uv run python -m "
      f"mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.skills` "
      f"(or ...datasets.tracker)."
    )

  raw = np.load(path, allow_pickle=False)
  env_id = torch.from_numpy(raw["env_id"]).to(device).long()
  held = (env_id % holdout) == 0
  mask = held if split == "eval" else ~held

  loaded = Dataset(
    states=torch.from_numpy(raw["states"]).to(device)[mask],
    skill=torch.from_numpy(raw["skill"]).to(device).long()[mask],
    names=tuple(str(n) for n in raw["skill_names"]),
    fps=float(raw["fps"]),
  )
  print(f"[dataset] {loaded.states.shape[0]} states in '{split}' from {loaded.names}")
  return loaded
