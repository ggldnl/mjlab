"""Profile a skill: which states are worth handing it, at which goal.

Run:

    1. Train the skills. Checkpoints are found under each skill's own log directory.

       uv run train Mjlab-G1-Walk --env.scene.num-envs 4096

    2. Profile. One npz per skill under data/selector.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.profile
       uv run python -m ...selector.profile --skills "('jump',)" --steps 1500

       A policy that climbed a speed curriculum needs its real range said out loud, because
       the command term's config still holds the curriculum's first stage:

       uv run python -m ...selector.profile --skills "('run',)" \
         --sweep-range "(1.0, 4.0)"

    3. Look at the result.

       uv run python -m ...selector.viewer --skill jump

The selector answers one question: given a skill and a goal, which states are worth aiming
the bridge at. It does not know where the robot was interrupted, does not know what the
bridge costs, and does not choose. Choosing among the offers is a later component's job.

A skill's own rollouts already contain the answer. A state the skill was measured passing
through, in a run that went on to finish cleanly, is by construction a state the skill can
carry on from. So the profile is built by watching: drive the trained policy, write down
which situations it keeps finding itself in, rank them by how reliably it goes there.

##
# One goal at a time
##

The rollouts have to be honest about the conditioning. Two families need opposite handling.

The jump and the kick draw a goal once per episode and hold it, so reading the goal off
every frame is correct: a frame and its label always agree.

The velocity skills do not. `twist` resamples every three to eight seconds mid-episode, and
has modes on top of that: a share of environments stand still, a share run in heading mode
where the yaw rate is recomputed from a heading error every step. Read frame by frame, a
rollout is a smear. The robot spends a second finishing the old command while wearing the
label of the new one, the yaw rate never holds still, and states from half a dozen twists
land in one band. It shows up as candidates facing in unrelated directions.

So the velocity family is swept, not read. The command's ranges are pinned to one twist,
the mode shares are zeroed, the environment is reset, and the whole batch runs at that
twist before the next one is pinned. Every draw returns the same number, the observation
and the label are the same thing by construction, and a band is one sweep point.

##
# The same way round
##

A state is recorded relative to where its own episode started: the robot's position at that
first frame is subtracted off and its heading turned to zero, orientation and both velocity
vectors with it. Skills here are egocentric, so which way the world the robot happened to
face is not part of what it was doing, and leaving it in makes two recordings of one
situation look like two situations.

##
# Stages
##

Some skills have a beginning and an end. The jump crouches, launches, flies and lands, and
those are four different things to be handed. Clustering the band as a whole lets the long
end of the motion swallow the short interesting part: standing is tight, covered, and wins
every cluster it is offered. So a skill with a progression is split into stages first and
clustered inside each, giving every part of the motion its own quota. Walking has no such
progression and is left alone.

##
# Scoring
##

Three measured numbers, no learning and no critic.

    coverage    how many separate runs pass through this situation, not how many frames.
                Counted per frame, a slow moment wins by lasting longer rather than by
                being a place the skill reliably goes.
    clean       of the runs that passed through it, how many finished without falling.
                The safety term, measured rather than assumed.
    tightness   how alike the frames in the group are. A loose group is a bucket of
                leftovers the clustering had to put somewhere, not a situation.

    score = coverage * clean * tightness

Deliberately blunt: all three have to be decent, and any one near zero disqualifies.

Output, per band and stage, is a handful of full states in the layout
BridgeCommand.target already takes, plus the run each was lifted out of so it can be looked
at in context.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets import dataset
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.skills import SKILLS
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import reyaw
from mjlab.tasks.bridging.experiments.humanoid.selector import features
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, yaw_quat

ROBOT = "robot"

PROFILE_ROOT = Path("data") / "selector"


##
# Reading, pinning and staging each skill's goal.
##


def _named(env: ManagerBasedRlEnv, term: str) -> torch.Tensor:
  """One command term's value, with a readable failure when the term is not there."""
  value = env.command_manager.get_command(term)
  if value is None:
    raise SystemExit(
      f"This environment has no command term '{term}'. It holds "
      f"{env.command_manager.active_terms}."
    )
  return value


def _term(env: ManagerBasedRlEnv, name: str) -> Any:
  """One command term, typed loosely on purpose.

  Pinning a goal reaches into the term's own config, whose shape differs per skill. There is
  no common type to narrow to, so the alternative is an isinstance branch per skill in every
  pinning function.
  """
  term = env.command_manager.get_term(name)
  if term is None:
    raise SystemExit(
      f"This environment has no command term '{name}'. It holds "
      f"{env.command_manager.active_terms}."
    )
  return term


def _twist(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _named(env, "twist")


def _twist_prepare(env_cfg: ManagerBasedRlEnvCfg) -> None:
  """Take the command curriculum out before the environment is built.

  It rewrites the twist ranges from its own stage table starting at the very first step,
  which quietly undoes any pin. The symptom is a profile labelled with the goals it asked
  for and recorded at whatever the curriculum last set: five bands all coming out with a
  mean forward speed of zero, holding states from every speed, facing every way.
  """
  env_cfg.curriculum.pop("command_vel", None)


def _twist_sweep(
  env: ManagerBasedRlEnv, count: int, span: tuple[float, float] | None
) -> list[tuple[float, ...]]:
  """Forward speed, evenly across the range.

  One axis, not a grid over all three. A grid of any useful resolution is the cube of this
  and the rollout budget is already the expensive part. Forward speed is the axis the skills
  were trained hardest on and the one a bridge most often delivers into. Sweeping something
  else means changing this function.

  The default range is the command term's own, which is the curriculum's first stage. Right
  for a policy whose training never left it, too narrow for one that climbed: the run skill
  reaches several metres a second and would be profiled over the first metre. `sweep_range`
  is the override, and there is no way to read the answer off a checkpoint.
  """
  low, high = span if span else _term(env, "twist").cfg.ranges.lin_vel_x
  return [(float(v), 0.0, 0.0) for v in np.linspace(low, high, count)]


def _twist_pin(env: ManagerBasedRlEnv, goal: Sequence[float]) -> None:
  """Make every draw of the twist return `goal`, and switch the modes off.

  The ranges, not the buffer. A command term recomputes and resamples inside the env step,
  between the physics and the observation the policy acts on, so a value written from out
  here is one the policy has already been handed something else instead of. Collapsing the
  ranges leaves the term doing its own work with only one answer to find.
  """
  cfg = _term(env, "twist").cfg
  vx, vy, wz = goal
  cfg.ranges.lin_vel_x = (vx, vx)
  cfg.ranges.lin_vel_y = (vy, vy)
  cfg.ranges.ang_vel_z = (wz, wz)
  # The modes are why a fixed range is not enough. A standing env ignores the command, a
  # heading env overwrites the yaw rate every step from its own heading error, a forward env
  # clamps what it was given, and a world env reads it in the wrong frame
  cfg.rel_standing_envs = 0.0
  cfg.rel_heading_envs = 0.0
  cfg.rel_forward_envs = 0.0
  cfg.rel_world_envs = 0.0


def _jump_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env, "motion")
  assert isinstance(term, JumpCommand)
  return term.goal


def _jump_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env, "motion")
  assert isinstance(term, JumpCommand)
  return term.phase.squeeze(-1)


def _kick_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _named(env, "kick")


def _kick_sweep(
  env: ManagerBasedRlEnv, count: int, span: tuple[float, float] | None
) -> list[tuple[float, ...]]:
  low, high = span if span else _term(env, "kick").cfg.ranges.speed
  return [(float(v), 0.0) for v in np.linspace(low, high, count)]


def _kick_pin(env: ManagerBasedRlEnv, goal: Sequence[float]) -> None:
  cfg = _term(env, "kick").cfg
  speed, heading = goal
  cfg.ranges.speed = (speed, speed)
  cfg.ranges.heading = (heading, heading)


def _planar(goals: np.ndarray) -> np.ndarray:
  return np.hypot(goals[:, 0], goals[:, 1])


@dataclass(frozen=True)
class SkillGoal:
  """Everything the profiler needs to know about one skill's conditioning."""

  labels: tuple[str, ...]
  read: Callable[[ManagerBasedRlEnv], torch.Tensor]
  """What the skill is being asked for, per environment. (N, G)."""

  key: Callable[[np.ndarray], np.ndarray]
  """(N, G) to (N,). The one number bands are cut on, when they are not swept.

  A scalar, not the whole goal vector. The jump's goal carries a heading offset and an apex
  alongside the distance, and standardizing all four gives the two incidental ones the same
  say as the one that changes the motion. Bands then split on apex, which nobody wants."""

  key_label: str
  sweep: (
    Callable[
      [ManagerBasedRlEnv, int, tuple[float, float] | None], list[tuple[float, ...]]
    ]
    | None
  ) = None
  """Goals to pin one after another, or None to let the skill draw its own and read them."""

  pin: Callable[[ManagerBasedRlEnv, Sequence[float]], None] | None = None
  prepare: Callable[[ManagerBasedRlEnvCfg], None] | None = None
  """Anything to change before the environment exists, such as removing a curriculum that
  would fight the pin."""

  phase: Callable[[ManagerBasedRlEnv], torch.Tensor] | None = None
  """Progress through the skill in [0, 1], for skills that have a beginning and an end."""


GOALS: dict[str, SkillGoal] = {
  "walk": SkillGoal(
    ("vx", "vy", "wz"),
    _twist,
    lambda g: g[:, 0],
    "vx",
    _twist_sweep,
    _twist_pin,
    _twist_prepare,
  ),
  "run": SkillGoal(
    ("vx", "vy", "wz"),
    _twist,
    lambda g: g[:, 0],
    "vx",
    _twist_sweep,
    _twist_pin,
    _twist_prepare,
  ),
  "push": SkillGoal(
    ("vx", "vy", "wz"),
    _twist,
    lambda g: g[:, 0],
    "vx",
    _twist_sweep,
    _twist_pin,
    _twist_prepare,
  ),
  # Read rather than swept: one clip and one stretch are drawn per episode and held, so a
  # frame and its label already agree. There is no range to pin either, since the goal is
  # the clip's own travel times the stretch and not a number the config names
  "jump": SkillGoal(
    ("dx", "dy", "dyaw", "apex"), _jump_goal, _planar, "distance", phase=_jump_phase
  ),
  # Staged in the same sense the jump is, and not wired up: the kick's progression lives in
  # mdp.phase, which reports a ball rather than a fraction
  "kick": SkillGoal(("vx", "vy"), _kick_goal, _planar, "speed", _kick_sweep, _kick_pin),
}


@dataclass
class ProfileCfg:
  """How a skill is profiled."""

  skills: tuple[str, ...] = ("walk", "run", "jump")
  """Which skills to profile. Kick and push work too and want something on the floor."""

  num_envs: int = 64
  steps: int = 900
  """Control steps in total. A swept skill splits these between its sweep points."""

  settle: int = 25
  """Control steps discarded after every reset.

  mjlab resets the instant an environment terminates, so the frame after a fall is a robot
  standing at its default pose. No policy chose to be in it, and left in, it forms a large,
  tight, well covered cluster in every band of every skill."""

  device: str = "cuda:0"

  goal_bands: int = 5
  """Sweep points for a swept skill. Equal width bins of the key for a read one."""

  sweep_range: tuple[float, float] | None = None
  """Ends of the swept axis. None takes the command term's own range, which is the
  curriculum's first stage and too narrow for a policy that climbed past it."""

  stages: int = 4
  """Slices of the progression clustered separately, for skills that have one."""

  clusters: int = 10
  """Situations looked for inside a stage, before scoring throws most of them away."""

  keep: int = 4
  """Candidates kept per stage, best last."""

  play_frames: int = 400
  """Cap on the run stored beside each candidate for the viewer to play back."""

  out: Path = PROFILE_ROOT
  checkpoints: tuple[str, ...] = field(default_factory=tuple)
  """Explicit checkpoint paths, one per entry of skills. Empty means search."""

  seed: int = 0


@dataclass
class Recording:
  """One skill's rollouts, flattened to one row per environment per control step."""

  states: np.ndarray
  """(N, 13 + 2J), each relative to where its own episode started."""
  feats: np.ndarray
  """(N, F)."""
  goals: np.ndarray
  """(N, G)."""
  band: np.ndarray
  """(N,) which sweep point produced the row, or -1 when the bands are cut afterwards."""
  phase: np.ndarray
  """(N,) progress through the skill in [0, 1], or zero for skills without one."""
  run: np.ndarray
  """(N,) which episode the row came from. Unique across environments."""
  age: np.ndarray
  """(N,) control steps into that episode, so an episode can be put back in order."""
  failed: np.ndarray
  """(E,) whether episode e ended by terminating rather than by running out of time."""
  joint_names: tuple[str, ...]
  fps: float


def canonical(
  states: torch.Tensor, anchor_quat: torch.Tensor, anchor_xy: torch.Tensor
) -> torch.Tensor:
  """The same states seen from where their episode began, facing along its +x.

  A yaw rotation leaves height alone, so this removes exactly the part of the state saying
  where on the floor the robot was and which way it pointed, and nothing else. Two
  recordings of one situation come out identical rather than as two situations that happen
  to share a shape.
  """
  rotation = quat_conjugate(anchor_quat)
  out = reyaw(states, rotation)
  offset = states[:, 0:3].clone()
  offset[:, 0:2] -= anchor_xy
  out[:, 0:3] = quat_apply(rotation, offset)
  return out


def record(name: str, checkpoint: str | None, cfg: ProfileCfg) -> Recording:
  """Drive one trained skill and write down everything the profile needs.

  Close to dataset.record and deliberately not built on it. That one records states so the
  bridge gets reachable windows. This one sweeps the goal, anchors every state to its own
  episode, and also needs the situation each frame is in, its stage, which episode it
  belongs to, and how that episode ended. Threading all of that through one function would
  leave neither readable.
  """
  from tensordict import TensorDict

  spec = SKILLS[name]
  goal = GOALS[name]
  env_cfg = load_env_cfg(spec.task)
  env_cfg.scene.num_envs = cfg.num_envs
  if goal.prepare is not None:
    goal.prepare(env_cfg)
  fps = dataset.control_rate(env_cfg)
  path = dataset.find_checkpoint(
    spec.experiments,
    checkpoint,
    hint=f" Train it with `uv run train {spec.task}`, or name one in `checkpoints`.",
  )
  print(f"[selector] {name}: {path}")

  agent_cfg = load_rl_cfg(spec.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(spec.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
  runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=cfg.device)
  policy = runner.get_inference_policy(device=cfg.device)

  robot: Entity = env.scene[ROBOT]
  describe = features.Describer(robot)
  origin = env.scene.env_origins[:, :2]

  rows: list[torch.Tensor] = []
  situations: list[torch.Tensor] = []
  wanted: list[torch.Tensor] = []
  stages: list[torch.Tensor] = []
  bands: list[torch.Tensor] = []
  runs: list[torch.Tensor] = []
  ages: list[torch.Tensor] = []
  keep: list[torch.Tensor] = []
  failed: list[bool] = []
  next_id = 0

  def roll(steps: int, band: int) -> None:
    """One batch of environments, from a fresh reset, at whatever goal is pinned."""
    nonlocal next_id
    obs, _ = env.reset()
    age = torch.zeros(cfg.num_envs, dtype=torch.long, device=cfg.device)
    alive = torch.arange(next_id, next_id + cfg.num_envs, device=cfg.device)
    failed.extend([False] * cfg.num_envs)
    next_id += cfg.num_envs
    anchor_xy = robot.data.root_link_pos_w[:, :2] - origin
    anchor_quat = yaw_quat(robot.data.root_link_quat_w).clone()
    anchor_xy = anchor_xy.clone()

    for step in range(steps):
      # Only the policy call goes in inference mode. Stepping the env inside it marks
      # every buffer it writes as an inference tensor, and the next reset cannot write them
      with torch.inference_mode():
        action = policy(
          TensorDict(obs, batch_size=[cfg.num_envs])  # ty: ignore[invalid-argument-type]
        )
      obs, _, terminated, truncated, _ = env.step(action)

      here = dataset.state(robot).clone()
      here[:, 0:2] -= origin

      done = terminated | truncated
      ended = done.nonzero(as_tuple=False).squeeze(-1)
      if ended.numel():
        # The env has already reset, so this frame is the next episode's first. Close the
        # finished episodes, hand out new ids, and re-anchor on what is standing there now,
        # all before the frame is written down
        for old, fell in zip(
          alive[ended].cpu().numpy(), terminated[ended].cpu().numpy(), strict=True
        ):
          failed[int(old)] = bool(fell)
        count = int(ended.numel())
        alive[ended] = torch.arange(next_id, next_id + count, device=cfg.device)
        failed.extend([False] * count)
        next_id += count
        anchor_xy[ended] = here[ended, 0:2]
        anchor_quat[ended] = yaw_quat(here[ended, 3:7])

      age = torch.where(done, torch.zeros_like(age), age + 1)

      rows.append(canonical(here, anchor_quat, anchor_xy))
      situations.append(describe(robot).clone())
      # Read after the step, so this is the goal the policy was following when it produced
      # the state, not one drawn for an episode that has not started
      wanted.append(goal.read(env).clone())
      stages.append(
        goal.phase(env).clone()
        if goal.phase
        else torch.zeros(cfg.num_envs, device=cfg.device)
      )
      bands.append(torch.full((cfg.num_envs,), band, device=cfg.device))
      runs.append(alive.clone())
      ages.append(age.clone())
      keep.append(age >= cfg.settle)
      if (step + 1) % 100 == 0:
        print(f"[selector] {name}: band {band}, {step + 1}/{steps}")

  if goal.sweep is not None and goal.pin is not None:
    points = goal.sweep(env, cfg.goal_bands, cfg.sweep_range)
    budget = max(cfg.steps // len(points), cfg.settle + 1)
    for index, point in enumerate(points):
      print(f"[selector] {name}: pinning {goal.labels} = {point}")
      goal.pin(env, point)
      roll(budget, index)
  else:
    roll(cfg.steps, -1)

  joint_names = robot.joint_names
  env.close()

  valid = torch.stack(keep, dim=0).flatten(0, 1)

  def take(stack: list[torch.Tensor]) -> np.ndarray:
    return torch.stack(stack, dim=0).flatten(0, 1)[valid].cpu().numpy()

  return Recording(
    states=take(rows).astype(np.float32),
    feats=take(situations).astype(np.float32),
    goals=take(wanted).astype(np.float32),
    band=take(bands).astype(np.int16),
    phase=take(stages).astype(np.float32),
    run=take(runs).astype(np.int64),
    age=take(ages).astype(np.int32),
    failed=np.asarray(failed, dtype=bool),
    joint_names=joint_names,
    fps=fps,
  )


def kmeans(
  points: torch.Tensor, k: int, seed: int, iters: int = 50
) -> tuple[torch.Tensor, torch.Tensor]:
  """Plain Lloyd's algorithm with a k-means++ start. Returns labels and centres.

  Written out rather than imported: it is thirty lines against a scikit-learn dependency for
  one function. Empty clusters keep their seed centre and go on to score zero coverage,
  which is the right outcome for them.
  """
  n = points.shape[0]
  k = max(1, min(k, n))
  generator = torch.Generator(device=points.device).manual_seed(seed)
  first = torch.randint(n, (1,), generator=generator, device=points.device)
  centres = points[first].clone()
  for _ in range(k - 1):
    spread = torch.cdist(points, centres).amin(dim=1).square()
    total = float(spread.sum())
    if total <= 0.0:
      pick = torch.randint(n, (1,), generator=generator, device=points.device)
    else:
      pick = torch.multinomial(spread / total, 1, generator=generator)
    centres = torch.cat([centres, points[pick]])
  labels = torch.zeros(n, dtype=torch.long, device=points.device)
  for _ in range(iters):
    labels = torch.cdist(points, centres).argmin(dim=1)
    for j in range(k):
      here = labels == j
      if bool(here.any()):
        centres[j] = points[here].mean(dim=0)
  return labels, centres


def bands_of(rec: Recording, goal: SkillGoal, count: int) -> np.ndarray:
  """One band per frame: the sweep point it ran at, or an equal width bin of the key."""
  if rec.band.max() >= 0:
    return rec.band.astype(np.int64)
  key = goal.key(rec.goals)
  low, high = float(key.min()), float(key.max())
  if high - low < 1e-9:
    return np.zeros(key.size, dtype=np.int64)
  edges = (key - low) / (high - low) * count
  return np.clip(edges.astype(np.int64), 0, count - 1)


@dataclass
class Candidates:
  """What profiling one skill produced, ready to be written out."""

  states: np.ndarray
  feats: np.ndarray
  band: np.ndarray
  stage: np.ndarray
  score: np.ndarray
  coverage: np.ndarray
  clean: np.ndarray
  tightness: np.ndarray
  band_goal: np.ndarray
  band_key: np.ndarray
  play: np.ndarray
  play_feats: np.ndarray
  play_len: np.ndarray
  play_at: np.ndarray
  feat_mean: np.ndarray
  feat_std: np.ndarray


def profile(rec: Recording, goal: SkillGoal, cfg: ProfileCfg) -> Candidates:
  """Cluster each band, or each stage of each band, score the groups and keep the best."""
  device = torch.device(cfg.device)
  mean = rec.feats.mean(axis=0)
  std = np.maximum(rec.feats.std(axis=0), 1e-6)
  # Standardized, or the clustering is dominated by whichever column is measured in the
  # largest units. A yaw rate in rad/s and a foot height in metres are not comparable
  # distances until both are in spreads
  z_feats = torch.as_tensor((rec.feats - mean) / std, device=device)

  bands = bands_of(rec, goal, cfg.goal_bands)
  num_bands = int(bands.max()) + 1
  stages = cfg.stages if goal.phase is not None else 1
  key = goal.key(rec.goals)

  found: list[dict[str, Any]] = []
  band_goal = np.zeros((num_bands, rec.goals.shape[1]), dtype=np.float32)
  band_key = np.zeros(num_bands, dtype=np.float32)

  for band in range(num_bands):
    in_band = np.flatnonzero(bands == band)
    if in_band.size == 0:
      continue
    band_goal[band] = rec.goals[in_band].mean(axis=0)
    band_key[band] = key[in_band].mean()

    # The runs this band holds, and the earliest progress each was seen at. A stage uses
    # the second number to tell a run that skipped it from one that began after it
    eligible, which = np.unique(rec.run[in_band], return_inverse=True)
    opened = np.full(eligible.size, np.inf)
    np.minimum.at(opened, which, rec.phase[in_band])

    for stage in range(stages):
      if stages == 1:
        where = in_band
      else:
        low, high = stage / stages, (stage + 1) / stages
        slice_of = (rec.phase[in_band] >= low) & (
          rec.phase[in_band] < high
          if stage + 1 < stages
          else rec.phase[in_band] <= high
        )
        where = in_band[slice_of]
      if where.size < cfg.clusters:
        continue

      # Coverage counts the runs that could have reached this stage, not every run in the
      # band. Reference state initialization drops an episode into a random point of the
      # clip, so most runs begin past the crouch and were never candidates for reaching it.
      # Counted over the whole band the opening stage scores three percent and gets thrown
      # away, which is a fact about the seeding and not about the skill. A run that started
      # early and then fell short still counts and still penalizes the stage
      runs_here = eligible if stages == 1 else eligible[opened < high]
      labels, centres = kmeans(
        z_feats[where], cfg.clusters, cfg.seed + band * 97 + stage
      )

      scored: list[tuple[float, float, float, float, int]] = []
      for group in range(centres.shape[0]):
        rows = where[(labels == group).cpu().numpy()]
        if rows.size == 0:
          continue
        runs = np.unique(rec.run[rows])
        # Per run, not per frame: a situation the skill lingers in during one run out of
        # fifty is not one it reliably goes to
        touched = float(runs.size / max(runs_here.size, 1))
        survived = float(np.mean(~rec.failed[runs]))
        gap = torch.cdist(z_feats[rows], centres[group : group + 1]).squeeze(1)
        packed = 1.0 / (1.0 + float(gap.mean()))

        # The representative is the frame nearest the centre, from a run that finished
        # standing up. A candidate is a state the skill will be started from, and one lifted
        # out of a run that fell is not evidence of anything
        survivors = ~rec.failed[rec.run[rows]]
        pool = np.flatnonzero(survivors) if survivors.any() else np.arange(rows.size)
        nearest = pool[int(gap.cpu().numpy()[pool].argmin())]
        scored.append(
          (touched * survived * packed, touched, survived, packed, int(rows[nearest]))
        )

      # Ascending, so the viewer's slider walks from the least likely to the most
      scored.sort(key=lambda entry: entry[0])
      for total, touched, survived, packed, row in scored[-cfg.keep :]:
        found.append(
          {
            "row": row,
            "band": band,
            "stage": stage,
            "score": total,
            "coverage": touched,
            "clean": survived,
            "tightness": packed,
          }
        )

  width = rec.states.shape[1]
  count = len(found)
  play = np.zeros((count, cfg.play_frames, width), dtype=np.float32)
  play_feats = np.zeros((count, cfg.play_frames, rec.feats.shape[1]), dtype=np.float32)
  play_len = np.zeros(count, dtype=np.int32)
  play_at = np.zeros(count, dtype=np.int32)
  for index, entry in enumerate(found):
    # The run this candidate came out of, in the order it happened, so the viewer can play
    # the motion and know the candidate is somewhere in it
    rows = np.flatnonzero(rec.run == rec.run[entry["row"]])
    order = rows[np.argsort(rec.age[rows])][: cfg.play_frames]
    play[index, : order.size] = rec.states[order]
    play_feats[index, : order.size] = rec.feats[order]
    play_len[index] = order.size
    match = np.flatnonzero(order == entry["row"])
    play_at[index] = int(match[0]) if match.size else 0

  def column(field_name: str, dtype: Any) -> np.ndarray:
    return np.asarray([entry[field_name] for entry in found], dtype=dtype)

  return Candidates(
    states=np.asarray(
      [rec.states[entry["row"]] for entry in found], dtype=np.float32
    ).reshape(-1, width),
    feats=np.asarray(
      [rec.feats[entry["row"]] for entry in found], dtype=np.float32
    ).reshape(-1, rec.feats.shape[1]),
    band=column("band", np.int16),
    stage=column("stage", np.int16),
    score=column("score", np.float32),
    coverage=column("coverage", np.float32),
    clean=column("clean", np.float32),
    tightness=column("tightness", np.float32),
    band_goal=band_goal,
    band_key=band_key,
    play=play,
    play_feats=play_feats,
    play_len=play_len,
    play_at=play_at,
    feat_mean=mean.astype(np.float32),
    feat_std=std.astype(np.float32),
  )


def write(name: str, rec: Recording, found: Candidates, cfg: ProfileCfg) -> Path:
  """One npz per skill, holding everything the viewer and the bridge need."""
  path = cfg.out / f"{name}.npz"
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    path,
    skill=np.asarray(name),
    key_label=np.asarray(GOALS[name].key_label),
    goal_labels=np.asarray(GOALS[name].labels),
    feature_names=np.asarray(features.FEATURES),
    joint_names=np.asarray(rec.joint_names),
    fps=np.asarray(rec.fps),
    **{field_name: getattr(found, field_name) for field_name in vars(found)},
  )
  bands = len(set(found.band.tolist()))
  print(f"[selector] wrote {path} ({len(found.score)} candidates over {bands} bands)")
  return path


def run(cfg: ProfileCfg) -> list[Path]:
  written: list[Path] = []
  for index, name in enumerate(cfg.skills):
    if name not in SKILLS:
      raise SystemExit(f"Unknown skill '{name}'. Known: {', '.join(SKILLS)}.")
    if name not in GOALS:
      raise SystemExit(f"No goal reader for '{name}'. Add one to GOALS.")
    explicit = cfg.checkpoints[index] if index < len(cfg.checkpoints) else None
    rec = record(name, explicit, cfg)
    written.append(write(name, rec, profile(rec, GOALS[name], cfg), cfg))
  return written


if __name__ == "__main__":
  run(tyro.cli(ProfileCfg, config=mjlab.TYRO_FLAGS))
