"""Finetune a tracking skill to get back on track fast after a bad hand-over at one entry.

finetune.py widens the reset everywhere in the clip and asks the skill to track well anyway.
This asks for something else: reset into the frames around one entry point, far enough off
that the skill cannot simply continue, and pay for closing the error rather than for being
on the reference already. A skill that survives a bad hand-over and a skill that recovers
from one in half the steps are different policies, and only the second objective can tell
them apart.

Why that needs its own reward. The tracking terms are summed exp(-e^2/s^2) kernels. At the
error the bridge delivers at kick entry 3, 4.3 tolerances on the worst leg joint, every one
of them is pinned near zero, so nothing in the reward says that a smaller error is better
until the policy is already nearly back. The term added here is potential shaping,

    F = gamma * P(next) - P(now),    P(s) = -(worst tracked body's distance from reference)

which is policy invariant (Ng, Harada and Russell, 1999): the sum over an episode
telescopes to a constant, so it cannot move the converged skill, only supply gradient
during the transient where the kernels have none. That is the whole point. It pays per step
for reducing the distance and charges for growing it, at any error, including errors so
large the tracking terms have stopped reading.

Three pieces, and they are independent:

    rehearsal reset   a share of every batch resets inside the entry window instead of
                      wherever the sampler wanted, with the wide per-channel noise from
                      finetune.Scales. The rest of the batch resets normally, which is what
                      keeps the rest of the clip in the training distribution
    recovery shaping  the term above, on every env. Not gated on the rehearsals: gating it
                      would break the telescoping and it would stop being policy invariant
    grace window      motion_far relaxed at a rehearsal reset and tightened back to the
                      task's own threshold over grace_steps. Without it the reset offset
                      alone trips the termination before the policy has acted, and the
                      episode collects the -100 termination penalty for a state it was
                      handed. That teaches the skill that the entry is death

What to check:

    Curriculum/rehearsal/scale              the noise ramp, 0 to 1, reaching 1 at
                                            ramp_fraction of the run. Beside it the joint
                                            noise it currently means, in rad
    Episode_Metrics/rehearsal_share         the share of episodes that were rehearsals.
                                            Divide the two metrics below by it: they are
                                            zero on every normal episode. Above the share
                                            configured, and that is not a bug: a rehearsal
                                            ends sooner, so more of them finish inside a
                                            logging window. Both metrics average over the
                                            same episodes, so the division still holds
    Episode_Metrics/rehearsal_recovered     share of all episodes that were a rehearsal and
                                            got back under recovered_m. Over the share
                                            above, this is the recovery rate
    Episode_Metrics/rehearsal_recovery      steps to get there, censored at the episode
                                            length when it never happened. The number this
                                            run exists to move down
    Rewards/recovery                        near zero at convergence on normal episodes,
                                            positive while a rehearsal is closing. A term
                                            that stays large is a policy living off the
                                            transient, which means the window is too wide
                                            or the noise past what the skill can do
    Episode_Termination/motion_far          a large share at the end is the grace window
                                            expiring before the policy is back

Run

1. Finetune. The checkpoint defaults to the newest under the skill's own log directory, and
   the window is read from the selector entry.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.recover \
        --skill kick --entry 3

2. Compare the two policies on the hand-over they are meant to differ on. --kick-variants
   resolves the newest checkpoint of each experiment, so neither path has to be typed.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.benchmarks.kick.transitions \
        --entry 3 --switch-steps '(100,130,160)' \
        --kick-variants "('base','recover')"

   Read recovery_steps and track_error_auc per kick_label in summary.csv. Those columns
   measure what this run trains; success and ball speed say whether it cost anything.

3. Check the skill did not rot away from the entry, which is what the mixture is for.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.benchmarks.run \
        --skill kick --phases '(123,)' --seeds '(0,1,2)' \
        --scales '(1.0,2.0,4.0,6.0,8.0)' --save-traces False \
        --output logs/benchmarks/recover

One thing this does not reproduce. A reset leaves the previous action at zero and a real
hand-over leaves the outgoing policy's, which the observation carries. transitions.py
measures that gap as previous_action_error_max; nothing here closes it, because the action
that a reset could write is a guess and the recorded one belongs to the bridge.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, astuple, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  TABLE_PATH,
  EntryTable,
)
from mjlab.tasks.bridging.experiments.humanoid.skills import SKILLS
from mjlab.tasks.bridging.experiments.humanoid.skills.finetune import (
  FAR_TERMINATION,
  Scales,
  Target,
  targets,
  tracker,
  widen,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
  JumpCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import find_checkpoint
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml

Span = tuple[float, float]

TRACE_ATTR = "_recovery_trace"
"""Where the per-step tracking error lives on the env. One per env, made by the tracker."""


##
# What a rehearsal is.
##


@dataclass
class Plan:
  """The rehearsal: where in the clip, how often, and how far off.

  Held by the command term, the reward, the termination and the ramp at once, so there is
  one copy of every number and the curriculum moves it by writing scale.
  """

  window: tuple[int, int]
  """Half open [first, last) in the clip's own frames. The stretch a hand-over aims into."""
  share: float
  """Fraction of every reset batch that becomes a rehearsal."""
  narrow: Target
  """The task's own reset noise, where the ramp starts."""
  wide: Target
  """The reset noise a rehearsal ends at, from the bridge tolerances."""
  ramp_steps: int
  recovered_m: float
  """Worst tracked body within this distance of the reference counts as back on track."""
  grace_steps: int
  grace_scale: float
  scale: float = 0.0
  """Where the ramp is, 0 to 1. Written by rehearsal_ramp, read by ranges()."""

  def ranges(self) -> Target:
    """The reset a rehearsal draws from now."""
    alpha = self.scale
    pose, speeds, joints, rates = self.narrow
    to_pose, to_speeds, to_joints, to_rates = self.wide

    def blend(a: Span, b: Span) -> Span:
      return (a[0] + alpha * (b[0] - a[0]), a[1] + alpha * (b[1] - a[1]))

    return (
      {k: blend(pose.get(k, (0.0, 0.0)), v) for k, v in to_pose.items()},
      {k: blend(speeds.get(k, (0.0, 0.0)), v) for k, v in to_speeds.items()},
      blend(joints, to_joints),
      blend(rates, to_rates),
    )


class Rehearsal(JumpCommand):
  """Mixed in over whatever tracker the skill has, to reset part of a batch at the entry.

  Declared against the jump's tracker and mixed in over the skill's, which is a subclass of
  it: the kick subclasses it to carry the ball and to keep a reset off it, and a recovery
  run has to keep both. See rehearsing(), which grafts this onto the term the skill's own
  config built, ahead of that class in the method order.

  Nothing here is called on a term that was not grafted, so begin() and not __init__ sets
  up what the reset needs.
  """

  plan: Plan
  rehearsing: torch.Tensor

  def begin(self, plan: Plan) -> None:
    self.plan = plan
    self.rehearsing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # The one place the trace is created, so the bar it latches on comes from the plan and
    # no term has to be handed it a second time and agree
    setattr(self._env, TRACE_ATTR, Trace(self._env, plan.recovered_m))

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Once for the whole batch, before anything is moved. The adaptive sampler replaces its
    # failure counts on every call rather than adding to them, so resampling the two halves
    # separately would leave it holding the second half alone
    super()._resample_command(env_ids)

    picked = torch.rand(len(env_ids), device=self.device) < self.plan.share
    self.rehearsing[env_ids] = picked
    trace(self._env).reset(env_ids)
    chosen = env_ids[picked]
    if chosen.numel() == 0:
      return

    low, high = self.plan.window
    frames = torch.randint(low, high, (len(chosen),), device=self.device)
    limit = getattr(self, "motion_reset_limit", None)
    if limit is not None:
      # The kick carries this to keep a reset off the ball. A window reaching past it would
      # write the striking foot into the ball and the strike terms would pay for the
      # teleport, which is the failure KickCommand._cap_to_backswing exists to prevent
      frames = torch.minimum(frames, limit[self.motion_ids[chosen]])
    self.time_steps[chosen] = frames

    was = self._install(self.plan.ranges())
    try:
      self.place_on_reference(chosen)
    finally:
      self._install(was)
    self.update_relative_body_poses()

  def _install(self, ranges: Target) -> Target:
    """Put these reset ranges on the config, and hand back the ones that were there."""
    cfg = self.cfg
    was = (
      cfg.pose_range,
      cfg.velocity_range,
      cfg.joint_position_range,
      cfg.joint_velocity_range,
    )
    (
      cfg.pose_range,
      cfg.velocity_range,
      cfg.joint_position_range,
      cfg.joint_velocity_range,
    ) = ranges
    return was


def rehearsing(source: JumpCommandCfg, plan: Plan) -> JumpCommandCfg:
  """A copy of the skill's tracker config whose term also rehearses hand-overs.

  Both the config and the term it builds are re-classed, which is how a class chosen at
  runtime gets extended without this module naming it. Nothing about the reference, the
  observation or the reward the tracker carries changes, and the copy keeps the skill's own
  config untouched for anything else reading it.
  """
  base = type(source)

  def build(self: JumpCommandCfg, env: ManagerBasedRlEnv) -> JumpCommand:
    # The skill's own build, named rather than reached through self, which now overrides it
    term = base.build(self, env)
    if not isinstance(term, JumpCommand):
      raise TypeError("A recovery finetune needs a clip tracker")
    term.__class__ = type(
      f"{type(term).__name__}Rehearsal", (Rehearsal, type(term)), {}
    )
    rehearsal = cast(Rehearsal, term)
    rehearsal.begin(plan)
    return rehearsal

  clone = copy.copy(source)
  clone.__class__ = type(f"{base.__name__}Rehearsal", (base,), {"build": build})
  return clone


##
# The tracking error, once per step.
##


class Trace:
  """Worst tracked body's distance from the reference, this step and last, plus the latch.

  Rewards, terminations and metrics all want the same number and all run before the command
  manager, so this refreshes lazily on first access within a step, guarded by the env's step
  counter, the way the kick's phase tracker does.
  """

  def __init__(self, env: ManagerBasedRlEnv, recovered_m: float) -> None:
    n, device = env.num_envs, env.device
    self.recovered_m = recovered_m
    self._step = -1
    self.error = torch.zeros(n, device=device)
    self.previous = torch.zeros(n, device=device)
    # No previous error yet, so the first step after a reset shapes nothing. That drops one
    # constant per episode out of the telescoping sum and changes no comparison
    self.fresh = torch.ones(n, dtype=torch.bool, device=device)
    self.steps = torch.zeros(n, dtype=torch.long, device=device)
    self.recovered = torch.zeros(n, dtype=torch.bool, device=device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.fresh[env_ids] = True
    self.steps[env_ids] = 0
    self.recovered[env_ids] = False
    # The scene moved under us and the entity buffers are not refreshed until the next
    # forward, so the cached step is stale rather than readable here
    self._step = -1

  def refresh(self, env: ManagerBasedRlEnv, command: JumpCommand) -> None:
    if self._step == env.common_step_counter:
      return
    self._step = env.common_step_counter
    error = torch.norm(command.body_pos_w - command.robot_body_pos_w, dim=-1).amax(
      dim=-1
    )
    self.previous = torch.where(self.fresh, error, self.error)
    self.error = error
    self.fresh = torch.zeros_like(self.fresh)

    back = error < self.recovered_m
    self.steps = torch.where(back & ~self.recovered, env.episode_length_buf, self.steps)
    self.recovered |= back


def trace(env: ManagerBasedRlEnv) -> Trace:
  """The env's tracking error trace. Not refreshed.

  Created by the rehearsing tracker's begin(), which is the only thing that knows the bar a
  recovery is measured against. Terminations run before rewards and rewards before metrics,
  so whichever of them touches it first would otherwise have set that bar for the rest.
  """
  found = getattr(env, TRACE_ATTR, None)
  if found is None:
    raise RuntimeError(
      "No recovery trace on this environment. Its tracker was not built by "
      "recover.rehearsing, so nothing is rehearsing a hand-over."
    )
  return found


def _read(env: ManagerBasedRlEnv, command_name: str) -> Trace:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, JumpCommand)
  found = trace(env)
  found.refresh(env, command)
  return found


def _rehearsing(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  mask = getattr(command, "rehearsing", None)
  if mask is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return mask


##
# Reward, termination, metrics.
##


def recovery(env: ManagerBasedRlEnv, command_name: str, gamma: float) -> torch.Tensor:
  """Potential shaping on the tracking error: paid for closing it, charged for opening it.

  P(s) is minus the distance the worst tracked body sits from the reference, so this is
  gamma * P(next) - P(now) with the two read one step apart. Bounded, telescoping, and
  policy invariant, which is why it is safe to add to a converged reward.

  Divided back out of the manager's dt, the way the bridge's own event rewards are, because
  this is a difference between two steps and not a rate. Without it the weight would mean a
  fiftieth of what it says and the term would be too small to read against the tracking sum.
  """
  found = _read(env, command_name)
  paid = found.previous - gamma * found.error
  return paid / env.step_dt if env.cfg.scale_rewards_by_dt else paid


def motion_too_far_grace(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  grace_steps: int,
  grace_scale: float,
) -> torch.Tensor:
  """motion_too_far, with the threshold opened at a rehearsal reset and closed back down.

  Only for the rehearsals, and only for grace_steps after they start. A normal episode is
  terminated on exactly the number the task's own curriculum is holding, so nothing about
  the skill's failure definition moves.
  """
  found = _read(env, command_name)
  limit = torch.full_like(found.error, threshold)
  if grace_steps > 0:
    elapsed = env.episode_length_buf.clamp(max=grace_steps).float() / grace_steps
    opened = threshold * (grace_scale + (1.0 - grace_scale) * elapsed)
    limit = torch.where(_rehearsing(env, command_name), opened, limit)
  return found.error > limit


def rehearsal_share(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _rehearsing(env, command_name).float()


def rehearsal_recovered(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Got back under recovered_m, counted over the rehearsals and zero everywhere else."""
  found = _read(env, command_name)
  return (found.recovered & _rehearsing(env, command_name)).float()


def rehearsal_recovery(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Steps a rehearsal took to get back on track, censored at the episode length.

  Censored rather than dropped, so an episode that never recovers scores the longest time
  it could have taken instead of scoring nothing and flattering the mean.
  """
  found = _read(env, command_name)
  taken = torch.where(found.recovered, found.steps, env.episode_length_buf)
  return taken.float() * _rehearsing(env, command_name).float()


def tracking_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Worst tracked body's distance from the reference, in metres."""
  return _read(env, command_name).error


class rehearsal_ramp:
  """Open the rehearsal reset from the task's own noise to the target over the run.

  Linear in the environment step count from the iteration this run started at, which is not
  zero: the step counter is advanced before training so the task's own curricula open at the
  stage the loaded checkpoint was trained at. The origin is taken on the first call rather
  than configured, so the two cannot drift apart.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    del env
    self._plan: Plan = cfg.params["plan"]
    self._origin: int | None = None

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, plan: Plan
  ) -> dict[str, torch.Tensor]:
    del env_ids
    if self._origin is None:
      self._origin = env.common_step_counter
    elapsed = env.common_step_counter - self._origin
    plan.scale = min(elapsed / max(plan.ramp_steps, 1), 1.0)
    _, _, joints, rates = plan.ranges()
    return {
      "scale": torch.tensor(plan.scale),
      "joint_pos": torch.tensor(joints[1]),
      "joint_vel": torch.tensor(rates[1]),
    }


##
# The run.
##


@dataclass
class Config:
  skill: str = "kick"
  entry: int = 3
  """Which selector entry to rehearse. The window is centred on its frame."""
  table: Path = TABLE_PATH
  window: tuple[int, int] | None = None
  """Half open clip frames to reset into. None centres half_width either side of the entry."""
  half_width: int = 13
  """Frames either side of the entry frame, when the window is not given.

  Thirteen is a quarter second at 50 Hz on both sides. The bridge does not deliver the robot
  on an exact frame, and a skill robust on one frame and not its neighbours would be robust
  to nothing a hand-over can produce.
  """

  checkpoint: Path | None = None
  """What to finetune. The newest under the skill's own log directory when not given."""

  share: float = 0.3
  """Fraction of every reset batch that rehearses the entry.

  The rest reset the way the task always did, which is what keeps the other nine tenths of
  the clip in the training distribution. A run at 1.0 trains one moment of one clip and
  forgets the skill around it.
  """
  scales: Scales = field(default_factory=Scales)
  """How far off a rehearsal starts, per channel, in multiples of its arrival tolerance."""
  ramp_fraction: float = 0.4
  """Share of the run spent opening the rehearsal reset to the target."""

  shaping_weight: float = 10.0
  """Reward per metre of tracking error closed, summed over however long it takes.

  The term undoes the manager's dt, so this is metres and not metres a second: ten means
  closing the 0.3 m the bridge leaves at kick entry 3 is worth 3 over the whole recovery.
  Read that against what recovering half a second sooner is worth in tracking reward, which
  is the 7.5 a step the kernels pay at most over the steps saved, so about 3.75 for
  twenty-five steps. The two are deliberately the same size: large enough to be a reason to
  hurry, too small to be a reason to leave the reference and come back.
  """
  recovered_m: float = 0.2
  """Worst tracked body within this of the reference counts as back on track.

  Under the 0.45 the task's own curriculum ends motion_far at, so recovering means tracking
  again rather than merely not failing.
  """
  grace_steps: int = 25
  """Steps a rehearsal gets before motion_far is back at the task's own threshold."""
  grace_scale: float = 2.5
  """How far open motion_far is on the first step of a rehearsal."""

  iterations: int = 1500
  num_envs: int = 4096
  seed: int = 0
  device: str = "cuda:0"
  logger: str = "tensorboard"
  suffix: str = "recover"
  """Appended to the experiment name, so this logs beside the skill rather than into it."""
  log_root: Path = Path("logs") / "rsl_rl"


def window_for(cfg: Config) -> tuple[int, int]:
  """The frames a rehearsal resets into, from the flag or from the selector entry."""
  if cfg.window is not None:
    return cfg.window
  table = EntryTable.load(cfg.table)
  found = table.of(cfg.skill)
  if not 0 <= cfg.entry < len(found):
    raise SystemExit(
      f"{cfg.skill} has {len(found)} entries and entry {cfg.entry} was asked for. "
      "Pass --window to name the frames directly."
    )
  frame = found[cfg.entry].frame
  return (max(frame - cfg.half_width, 0), frame + cfg.half_width + 1)


def grace_far_termination(env_cfg: ManagerBasedRlEnvCfg, cfg: Config) -> None:
  """Swap motion_far for the version that opens at a rehearsal reset.

  The threshold itself is left alone, in the term and in the curriculum that moves it, so
  the skill's failure definition on a normal episode is the one it was trained with. The
  curriculum updates params rather than replacing them, so the two extra params survive it.
  """
  term = env_cfg.terminations.get(FAR_TERMINATION)
  if term is None:
    raise SystemExit(
      f"{cfg.skill} has no {FAR_TERMINATION} termination, so there is no tracking failure "
      "to open at a hand-over. Only tracking skills are supported."
    )
  term.func = motion_too_far_grace
  term.params["grace_steps"] = cfg.grace_steps
  term.params["grace_scale"] = cfg.grace_scale


def build(cfg: Config, gamma: float) -> tuple[ManagerBasedRlEnvCfg, int]:
  """The finetuning environment, and the step count to start its curricula at."""
  env_cfg = load_env_cfg(SKILLS[cfg.skill])
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed

  name = tracker(env_cfg)
  command = env_cfg.commands[name]
  assert isinstance(command, JumpCommandCfg)
  narrow: Target = (
    dict(command.pose_range),
    dict(command.velocity_range),
    command.joint_position_range,
    command.joint_velocity_range,
  )
  pose, speeds, joints, rates = targets(cfg.scales)
  # common_step_counter advances once per control step, so an iteration is
  # num_steps_per_env of it whatever the environment count
  steps_per_iteration = int(load_rl_cfg(SKILLS[cfg.skill]).num_steps_per_env)
  plan = Plan(
    window=window_for(cfg),
    share=cfg.share,
    narrow=narrow,
    # A rehearsal never draws narrower than the task's own reset, the way finetune.py never
    # narrows what the task already had
    wide=(
      widen(command.pose_range, pose),
      widen(command.velocity_range, speeds),
      (min(narrow[2][0], joints[0]), max(narrow[2][1], joints[1])),
      (min(narrow[3][0], rates[0]), max(narrow[3][1], rates[1])),
    ),
    ramp_steps=max(int(cfg.ramp_fraction * cfg.iterations) * steps_per_iteration, 1),
    recovered_m=cfg.recovered_m,
    grace_steps=cfg.grace_steps,
    grace_scale=cfg.grace_scale,
  )
  env_cfg.commands[name] = rehearsing(command, plan)

  env_cfg.rewards["recovery"] = RewardTermCfg(
    func=recovery,
    weight=cfg.shaping_weight,
    params={"command_name": name, "gamma": gamma},
  )
  grace_far_termination(env_cfg, cfg)

  env_cfg.metrics["rehearsal_share"] = MetricsTermCfg(
    func=rehearsal_share, params={"command_name": name}
  )
  env_cfg.metrics["rehearsal_recovered"] = MetricsTermCfg(
    func=rehearsal_recovered, reduce="last", params={"command_name": name}
  )
  env_cfg.metrics["rehearsal_recovery"] = MetricsTermCfg(
    func=rehearsal_recovery, reduce="last", params={"command_name": name}
  )
  env_cfg.metrics["tracking_error"] = MetricsTermCfg(
    func=tracking_error, params={"command_name": name}
  )

  # Past every stage of every curriculum the task carries, so a fresh run opens at the
  # values the loaded checkpoint was trained at instead of rewinding to the first stage
  start = 0
  for curriculum in env_cfg.curriculum.values():
    for stage in curriculum.params.get("stages", []):
      start = max(start, int(stage["step"]))
  start += 1

  env_cfg.curriculum["rehearsal"] = CurriculumTermCfg(
    func=rehearsal_ramp, params={"plan": plan}
  )
  return env_cfg, start


def run(cfg: Config) -> Path:
  if cfg.skill not in SKILLS:
    raise SystemExit(f"Unknown skill. Known: {', '.join(sorted(SKILLS))}.")
  if min(cfg.iterations, cfg.num_envs) < 1:
    raise SystemExit("Iterations and environment count must be positive")
  if not 0.0 < cfg.share <= 1.0:
    raise SystemExit("The rehearsal share is a fraction of every reset batch")
  if not 0.0 <= cfg.ramp_fraction <= 1.0:
    raise SystemExit("The ramp fraction is a share of the run")
  if min(astuple(cfg.scales)) < 0 or cfg.recovered_m <= 0:
    raise SystemExit("Noise scales must not be negative and the recovery bar positive")
  if cfg.grace_steps < 0 or cfg.grace_scale < 1.0:
    raise SystemExit(
      "The grace window opens the threshold, so its scale is at least one"
    )
  if cfg.half_width < 0:
    raise SystemExit("The window half width is a frame count")

  task = SKILLS[cfg.skill]
  agent = load_rl_cfg(task)
  source = find_checkpoint(agent.experiment_name, cfg.checkpoint)
  print(f"[recover] continuing {source}")

  # The shaping is only policy invariant at the discount the agent actually uses, so it is
  # read off the agent rather than declared here
  gamma = float(getattr(getattr(agent, "algorithm", None), "gamma", 0.99))
  env_cfg, start = build(cfg, gamma)
  window = window_for(cfg)
  print(f"[recover] rehearsing frames {window[0]} to {window[1] - 1} at {cfg.share:g}")

  agent = replace(
    agent,
    experiment_name=f"{agent.experiment_name}_{cfg.suffix}",
    max_iterations=cfg.iterations,
    logger=cfg.logger,
    upload_model=False,
    resume=False,
  )
  log_dir = (
    cfg.log_root / agent.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  log_dir.mkdir(parents=True)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent))
  (log_dir / "recover.json").write_text(
    json.dumps(
      {
        "config": asdict(cfg),
        "window": list(window),
        "source_checkpoint": str(source),
        "curriculum_start_step": start,
      },
      indent=2,
      default=str,
    )
  )
  print(f"[recover] logging to {log_dir}")

  torch.manual_seed(cfg.seed)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    env.common_step_counter = start
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), str(log_dir), cfg.device)
    runner.load(
      str(source), load_cfg={"actor": True, "critic": True}, map_location=cfg.device
    )
    runner.current_learning_iteration = 0
    env.common_step_counter = start
    runner.learn(num_learning_iterations=cfg.iterations, init_at_random_ep_len=True)
  finally:
    env.close()
  print(f"[recover] {log_dir.resolve()}")
  return log_dir


if __name__ == "__main__":
  run(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
