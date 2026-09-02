"""The dataset format, plus the rollout driver every source shares.

A dataset is a table of states the robot was measured being in, under physics, with a policy
holding it up. That is the only property the bridge needs: both ends of a window are
reachable because a robot reached them.

Every source builds one the same way, by driving a trained policy and writing down what
happens, so the driving lives here. A source module only says which policy, in which
environment, with what on the floor. See datasets/tracker.py for the recipe.

Row layout
----------

One row per environment per control step:

    root_pos (3)  root_quat (4)  root_lin_vel (3)  root_ang_vel (3)  q (J)  qd (J)

Root position has the environment origin subtracted off x and y, so the numbers are small
and mean "where in its own tile". Height is untouched, since height above the floor is
part of the state the bridge has to reach.

Rows in the first `settle` steps after a reset are dropped. mjlab resets the instant an
environment terminates, so the step after a fall is a robot standing at its default pose,
and without this the dataset fills up with one identical standing pose per failure.

Four extra columns
------------------

    source      which policy or clip the row came from
    trajectory  which physical rollout. A reset always starts a new one, so two rows sharing
                this were reached without the simulator being touched in between. Unique
                within a source and not across them, since each source is recorded on its
                own; `load_dataset` pairs it with `source` to get one identifier per rollout
    frame       control steps since that row's episode started. Two rows of one trajectory
                whose frames differ by k are a start and a target one robot got between in
                k control steps, and that is the only kind of window the bridge trains on.
                `Dataset.segments` is what turns the two columns into that index
    goal        every command term's value at that step, side by side. What the skill was
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

TRACKER_DATASET = DATASET_ROOT / "tracker.npz"
"""The human motion corpus, built by driving trajectory trackers over LAFAN1 clips."""

DEFAULT_DATASET = TRACKER_DATASET
"""What every config points at unless told otherwise.

The human motion corpus, because it is the one that gives the bridge a claim to being
independent of the skill pool. A bridge trained on skill rollouts and tested on skill
hand-overs cannot distinguish having learned bridging from having learned those five
policies; nothing in that experiment separates the two.
"""

SKILLS_DATASET = DATASET_ROOT / "rollouts.npz"
"""Deprecated. The corpus built from the skill pool's own rollouts. See datasets/skills.py.

Kept loadable, and no longer the default. Pass it explicitly to reproduce an older run.
"""

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Drive one trained policy in one environment, recording every control step.

  Returns the states, the environment and physical trajectory each row came from, how many
  steps into that trajectory it was, and what it was commanded to do at the time.

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
  trajectory = torch.arange(cfg.num_envs, dtype=torch.long, device=cfg.device)
  rows: list[torch.Tensor] = []
  ages: list[torch.Tensor] = []
  trajectories: list[torch.Tensor] = []
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
    done = terminated | truncated
    age = torch.where(done, torch.zeros_like(age), age + 1)
    # A reset starts a new physical trajectory. Adding num_envs keeps every trajectory
    # identifier unique while retaining the environment identity in its remainder.
    trajectory = torch.where(done, trajectory + cfg.num_envs, trajectory)

    here = state(robot).clone()
    here[:, 0:2] -= origin
    rows.append(here)
    ages.append(age.clone())
    trajectories.append(trajectory.clone())
    # Read after the step, so this is the command the policy was following when it
    # produced the state, not one drawn for the episode about to start
    goals.append(commanded(env).clone())
    keep.append(age >= cfg.settle)
    if (step + 1) % 100 == 0:
      print(f"[dataset] {label}: {step + 1}/{cfg.steps}")

  env.close()
  states = torch.stack(rows, dim=0).flatten(0, 1)
  frames = torch.stack(ages, dim=0).flatten(0, 1)
  trajectory_ids = torch.stack(trajectories, dim=0).flatten(0, 1)
  commands = torch.stack(goals, dim=0).flatten(0, 1)
  valid = torch.stack(keep, dim=0).flatten(0, 1)
  # Which environment each surviving row came from, so load_dataset can hold whole
  # environments out rather than individual frames
  env_id = torch.arange(cfg.num_envs, device=cfg.device).repeat(cfg.steps)[valid]
  return (
    states[valid].cpu().numpy().astype(np.float32),
    env_id.to(torch.int16).cpu().numpy(),
    trajectory_ids[valid].to(torch.int32).cpu().numpy(),
    frames[valid].to(torch.int32).cpu().numpy(),
    commands[valid].cpu().numpy().astype(np.float32),
  )


def write(
  path: Path,
  states: list[np.ndarray],
  env_ids: list[np.ndarray],
  trajectory_ids: list[np.ndarray],
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
    "trajectory": np.concatenate(trajectory_ids),
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
  trajectory: torch.Tensor
  """(N,) physical rollout identity. A reset always starts a new one."""
  frame: torch.Tensor
  """(N,) control step within `trajectory`."""
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

  def segments(
    self,
    min_steps: int,
    max_steps: int,
    start_rows: torch.Tensor | None = None,
  ) -> Segments:
    """An index of every contiguous stretch of rollout long enough to be a window.

    Both ends of a window come from one trajectory, so a window is never a pair of states
    invented by putting two rollouts side by side: a robot was demonstrably in the first,
    and `steps` control ticks later it was demonstrably in the second, under physics.

    Both ends therefore also come from the same source. Restricting the start to a set of
    sources restricts the target to the same set, which is why there is one filter here and
    not two. Coverage of a posture family is a property of the corpus, not of a pairing
    rule.
    """
    if min_steps < 1 or max_steps < min_steps:
      raise ValueError("Segment bounds must satisfy 1 <= min_steps <= max_steps.")

    # Sort into (trajectory, frame) order. The recording is time-major, so rows of one
    # rollout are strided rather than adjacent, and the settle cut can shorten a rollout
    # from the front. Sorting is what makes "the row `k` ticks later" an index offset
    width = int(self.frame.max().item()) + 1
    order = torch.argsort(self.trajectory * width + self.frame)
    trajectory = self.trajectory[order]
    frame = self.frame[order]

    # A run is a maximal stretch whose frames step by one inside one trajectory. Inside a
    # run, position + k is exactly the state k control ticks later
    steps_by_one = (trajectory[1:] == trajectory[:-1]) & (frame[1:] == frame[:-1] + 1)
    opens = torch.cat(
      [
        torch.ones(1, dtype=torch.bool, device=steps_by_one.device),
        ~steps_by_one,
      ]
    )
    run = opens.long().cumsum(0) - 1
    last = torch.bincount(run).cumsum(0) - 1
    positions = torch.arange(order.numel(), device=order.device)
    available = last[run] - positions

    eligible = available >= min_steps
    if start_rows is not None:
      allowed = torch.zeros_like(eligible)
      allowed[start_rows] = True
      eligible &= allowed[order]
    if not bool(eligible.any()):
      raise ValueError(
        "No contiguous rollout segment matches the requested sources and duration range."
      )

    return Segments(
      order=order,
      starts=positions[eligible],
      available=available[eligible].clamp(max=max_steps),
      min_steps=min_steps,
    )


@dataclass
class Segments:
  """Where every legal window lives, and how to draw one.

  Kept as an index rather than a materialised table of pairs. A rollout of `L` usable steps
  contains `L * K` windows for `K` admissible durations, and writing them all down costs
  memory proportional to that product for no benefit: the duration is drawn per episode
  anyway, and drawing it fresh is what stops a 15000-iteration run from seeing the same
  frozen set of windows for its whole life.
  """

  order: torch.Tensor
  """(N,) dataset row at each position, in (trajectory, frame) order."""
  starts: torch.Tensor
  """(K,) positions a window may open at."""
  available: torch.Tensor
  """(K,) longest window each start admits, in control steps, already capped."""
  min_steps: int

  def draw(
    self, count: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`count` windows: start rows, target rows, durations in control steps, and the
    position each one opened at.

    The duration is uniform over what the chosen start actually admits, which is not the
    same as uniform over the configured range: a start near the end of its rollout only
    offers short windows. Sampling the start first and the duration inside it is what keeps
    every drawn window a demonstrated one.

    The position is returned because the two endpoints are not all a window holds. Inside a
    run, position plus k is the state k control ticks later, so the positions between a
    start and its target are the crossing the robot actually made, and `path` reads them.
    """
    device = self.starts.device
    picked = torch.randint(0, self.starts.numel(), (count,), device=device)
    position = self.starts[picked]
    span = self.available[picked] - self.min_steps + 1
    steps = self.min_steps + (torch.rand(count, device=device) * span).long().clamp(
      max=span - 1
    )
    return self.order[position], self.order[position + steps], steps, position

  def path(
    self, position: torch.Tensor, steps: torch.Tensor, span: int
  ) -> torch.Tensor:
    """The rows of the demonstrated crossing, one per control tick. (N, span + 1).

    Column k is the state k ticks after the window opened, so column 0 is the start and
    column `steps` is the target. Columns past a window's own duration repeat its target
    rather than running on into whatever follows in the rollout: nothing reads them, since
    the episode is over by then, and a row from the next stride would be a quietly wrong
    answer if anything ever did.
    """
    offsets = torch.arange(span + 1, device=position.device)
    reach = torch.minimum(offsets.unsqueeze(0), steps.unsqueeze(-1))
    return self.order[position.unsqueeze(-1) + reach]


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
      f"mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.tracker`. "
      f"(...datasets.skills builds the deprecated skill-pool corpus.)"
    )

  raw = np.load(path, allow_pickle=False)
  env_id = torch.from_numpy(raw["env_id"]).to(device).long()
  held = (env_id % holdout) == 0
  mask = held if split == "eval" else ~held

  if "trajectory" not in raw:
    raise SystemExit(
      f"{path} predates time-consistent bridge transitions. Rebuild it with the dataset "
      "collector before training this bridge."
    )

  # A trajectory identifier is unique inside one source and starts over at zero for the
  # next, because every source is recorded by its own `record` call. So two clips hold the
  # same identifiers, and `segments` sorts rows by (trajectory, frame): shared identifiers
  # interleave rows of different clips at equal frames, no adjacent pair steps by one, every
  # run collapses to a single row and nothing is long enough to be a window. Pairing the
  # identifier with its source is what makes one physical rollout one trajectory again
  skill = torch.from_numpy(raw["skill"]).to(device).long()
  trajectory = torch.from_numpy(raw["trajectory"]).to(device).long()
  trajectory = skill * (int(trajectory.max().item()) + 1) + trajectory

  loaded = Dataset(
    states=torch.from_numpy(raw["states"]).to(device)[mask],
    skill=skill[mask],
    trajectory=trajectory[mask],
    frame=torch.from_numpy(raw["frame"]).to(device).long()[mask],
    names=tuple(str(n) for n in raw["skill_names"]),
    fps=float(raw["fps"]),
  )
  print(f"[dataset] {loaded.states.shape[0]} states in '{split}' from {loaded.names}")
  return loaded
