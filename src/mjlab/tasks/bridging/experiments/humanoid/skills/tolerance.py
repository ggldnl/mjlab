"""How wrong a state a skill can be handed at one entry and still get back on its clip.

The other half of the bridge measurement. bridges/evaluate.py says what a bridge delivers;
this says what a skill will accept. A hand-over works when the first fits inside the second,
and until both are measured in the same eight channels and the same units neither number
means anything on its own.

Not a survival test. A skill handed a bad state will usually stay upright: it compensates,
wanders off its reference, and finishes the motion as something else. That is a failure for
composition and a success for a fall check, which is why survival is reported here in its
own column and is never the criterion. What is measured is the skill's own tracking error,
the mean distance between its bodies and its reference's, and a displacement counts as
tolerated only while that error stays within `margin` of what the same entry produces
undisturbed.

Against the reference, not against an undisturbed sibling rollout. A strike or a landing is
chaotic and two rollouts of one entry part company within half a second, so a metric built
on the difference between them reads noise at every displacement. The undisturbed run is
still there, as env 0, and only says what the error is when nothing is wrong.

One channel is displaced at a time, over both signs and several random directions, each case
in its own env, all in parallel. A rung counts as tolerated only if every case on it holds,
and the limit is read from the bottom up: a channel that fails at 0.06 and passes again at
0.12 has not shown a safe displacement of 0.12.

Run

1. Measure an entry. Entries are the selector's, earliest first.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.tolerance \
      --skill kick --entry 3

2. Ask for a stricter or looser definition of on track. This is the knob that decides what
   the answer means, so it belongs in any quote of the result.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.tolerance \
      --skill kick --entry 3 --margin 0.02

3. Take the Tolerances block it prints into tests/entry_tolerances.py, then score a bridge
   against it.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.evaluate \
      --bridge distillation

Reading the result

    tolerates      the largest displacement of that channel every case held at. The number
                   to put in an entry profile
    tracks to      the worst tracking error on that rung, over every sign and direction
    fell           cases on that rung that terminated. Reported, never the criterion

A channel whose ladder runs out reports the top rung and says so: that is a statement about
the ladder, not about the skill.

Requires a skill that tracks a reference, since the measure is tracking error. A commanded
locomotion skill has no clip to be off, and is refused rather than given a number.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
  find_checkpoint,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp.commands import (
  CHANNELS,
  UNITS,
  arm_mask,
  channel_errors,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import resume
from mjlab.tasks.bridging.experiments.humanoid.selector.table import Entry, EntryTable
from mjlab.tasks.bridging.experiments.humanoid.skills import SKILLS
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul

REPORT_ROOT = Path("logs") / "benchmarks" / "skills"

STEPS: dict[str, float] = {
  "root_pos": 0.02,
  "root_ori": 0.05,
  "root_lin_vel": 0.10,
  "root_ang_vel": 0.15,
  "leg_joint_pos": 0.05,
  "leg_joint_vel": 0.50,
  "arm_joint_pos": 0.05,
  "arm_joint_vel": 0.50,
}
"""One rung of each channel's ladder, in that channel's own unit. Sized so the bottom rung
is finer than any bridge has ever delivered and the top is past what any skill has held."""

LADDER = (1, 2, 3, 4, 6, 8, 12, 16)
"""Rung multipliers. Coarse at the top on purpose: the interesting region is the bottom."""


@dataclass
class Config:
  skill: str = "kick"
  entry: int = 3
  """Index into that skill's selector entries, earliest first."""

  margin: float = 0.05
  """Extra tracking error, in metres, that still counts as on track.

  The definition of the answer, not a detail of it. A skill handed a displaced state always
  tracks worse; this says how much worse is still the same motion. Quote it with any
  tolerance this script produces, because 0.02 and 0.10 describe different skills.
  """

  directions: int = 3
  """Random directions per channel. Each is one displacement grown along the ladder, so a
  channel's limit is the worst of 2 x directions cases per rung."""

  settle: float = 0.3
  """Seconds the policy gets to absorb the displacement before the error is measured. A
  hand-over is allowed a transient; what is being measured is whether it ends."""

  horizon: float = 1.2
  """Seconds of rollout after the entry."""

  checkpoint: Path | None = None
  """Explicit checkpoint, or None for the newest under the skill's own experiment."""

  seed: int = 0
  device: str = "cuda:0"
  report_dir: Path = REPORT_ROOT
  write_report: bool = True


@dataclass
class Case:
  """One displaced rollout: a channel, a direction, and how far along it."""

  channel: str
  direction: int
  rung: float
  """Unsigned displacement in the channel's unit. Groups the signs and directions."""
  amount: float
  """Signed displacement actually applied."""


@dataclass
class Outcome:
  case: Case
  injected: float
  """What channel_errors scores the displacement as. Should equal rung, and is measured
  rather than assumed, because a clamp or a joint limit can eat part of it."""
  tracked: float
  """Mean body tracking error over the measured window, in metres."""
  fell: bool


##
# Displacing a state.
##


def _shape(
  channel: str, arms: torch.Tensor, generator: torch.Generator, device: str
) -> torch.Tensor:
  """A unit displacement for one channel. Scaling it walks the whole ladder.

  Root channels get a unit 3-vector, so a displacement of s reads as exactly s under the
  norm channel_errors takes. Joint channels get a vector over their own group normalised so
  its largest component is one, so a displacement of s reads as exactly s under the
  worst-joint maximum channel_errors takes. Drawn once per direction and reused across the
  rungs, which is what makes a ladder a ladder rather than eight unrelated displacements.
  """
  if channel.startswith("root"):
    vector = torch.randn(3, generator=generator, device=device)
    return vector / vector.norm().clamp(min=1e-6)
  group = arms if channel.startswith("arm") else ~arms
  vector = torch.zeros(arms.numel(), device=device)
  drawn = torch.randn(int(group.sum()), generator=generator, device=device)
  vector[group] = drawn / drawn.abs().amax().clamp(min=1e-6)
  return vector


def displace(
  base: torch.Tensor, channel: str, shape: torch.Tensor, amount: float
) -> torch.Tensor:
  """One state, moved along one channel by one amount. (1, 13 + 2J) -> (1, 13 + 2J).

  Exactly the eight quantities channel_errors measures, moved one at a time. Root
  orientation turns about the drawn axis rather than adding to the quaternion, since a
  quaternion is not a vector space and adding to one produces something that is not a
  rotation.
  """
  out = base.clone()
  joints = shape.numel() if not channel.startswith("root") else 0
  if channel == "root_pos":
    out[:, 0:3] += amount * shape
  elif channel == "root_ori":
    angle = torch.full((1,), amount, device=base.device)
    out[:, 3:7] = quat_mul(
      quat_from_angle_axis(angle, shape.unsqueeze(0)), base[:, 3:7]
    )
  elif channel == "root_lin_vel":
    out[:, 7:10] += amount * shape
  elif channel == "root_ang_vel":
    out[:, 10:ROOT_STATE_DIM] += amount * shape
  else:
    start = ROOT_STATE_DIM + (0 if channel.endswith("_pos") else joints)
    out[:, start : start + joints] += amount * shape
  return out


def ladder(cfg: Config) -> list[Case]:
  """Every case, ordered so a case's index is its env."""
  return [
    Case(channel, direction, rung * STEPS[channel], sign * rung * STEPS[channel])
    for channel in CHANNELS
    for direction in range(cfg.directions)
    for rung in LADDER
    for sign in (-1, 1)
  ]


##
# Rolling them out.
##


def _policy(task: str, checkpoint: Path, env: ManagerBasedRlEnv, device: str):
  """The skill's own actor, frozen."""
  from tensordict import TensorDict

  agent = load_rl_cfg(task)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = runner_cls(wrapped, asdict(agent), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  inference = runner.get_inference_policy(device=device)

  @torch.no_grad()
  def act(obs):
    return inference(TensorDict(obs, batch_size=[env.num_envs]))

  return act


def _fresh_obs(env: ManagerBasedRlEnv):
  """The observation as of right now, not as of the last step.

  The manager hands back a cached buffer whenever one exists, and everything here has just
  written a new robot state into the simulation. Without dropping the cache the first action
  after a displacement is computed from the state before it.
  """
  env.observation_manager._obs_buffer = None
  return env.observation_manager.compute()


def _prepare_scene(env: ManagerBasedRlEnv, skill: str) -> None:
  """Whatever a skill needs cleared before it is handed a robot mid-clip.

  Only the kick has anything: its strike latches are real observations, and one taking over
  with them still set believes it has already struck the ball. Looked up by name rather than
  by a registry, because there is exactly one of these.
  """
  if skill == "kick":
    from mjlab.tasks.bridging.experiments.humanoid.skills.kick import mdp as kick_mdp

    kick_mdp.reset_kick_phase(env)


def _place(env: ManagerBasedRlEnv, entry: Entry, targets: torch.Tensor) -> None:
  """Put every env on its own displaced copy of the entry, scene left alone.

  The entity reset and the forward are not optional: the action term holds the last action
  it applied and body poses are derived from qpos, so a robot written without them is read
  back as the robot that was there before.
  """
  robot = env.scene["robot"]
  joints = robot.num_joints
  robot.write_joint_state_to_sim(
    targets[:, ROOT_STATE_DIM : ROOT_STATE_DIM + joints],
    targets[:, ROOT_STATE_DIM + joints :],
  )
  robot.write_root_state_to_sim(targets[:, 0:ROOT_STATE_DIM])
  robot.reset(env_ids=torch.arange(env.num_envs, device=env.device))
  env.sim.forward()
  resume.rewind(env, entry.frame)
  previous = torch.as_tensor(entry.previous_action, device=env.device)
  resume.restore_action(env, previous.unsqueeze(0).expand(env.num_envs, -1))


def tracking_error(command: JumpCommand) -> torch.Tensor:
  """Mean distance between the robot's bodies and the reference's. (N,) metres.

  The skill's own measure of being on its clip, and what it is rewarded for. Being off it is
  the failure this script exists to find, and it is not the same failure as falling over.
  """
  return (command.body_pos_w - command.robot_body_pos_w).norm(dim=-1).mean(dim=-1)


@torch.no_grad()
def measure(cfg: Config) -> tuple[float, list[Outcome], Entry, Path]:
  """Roll every case out. Returns the undisturbed error, the outcomes, and what was run."""
  if cfg.skill not in SKILLS:
    raise SystemExit(
      f"Unknown skill '{cfg.skill}'. Known: {', '.join(sorted(SKILLS))}."
    )
  task = SKILLS[cfg.skill]
  entries = EntryTable.load().of(cfg.skill)
  if not entries:
    raise SystemExit(
      f"No selector entries for '{cfg.skill}'. Build them with selector.build."
    )
  if not 0 <= cfg.entry < len(entries):
    raise SystemExit(
      f"'{cfg.skill}' has {len(entries)} entries, so --entry must be 0 to "
      f"{len(entries) - 1}."
    )
  entry = entries[cfg.entry]
  checkpoint = find_checkpoint(
    (load_rl_cfg(task).experiment_name,),
    None if cfg.checkpoint is None else str(cfg.checkpoint),
    hint=f" Train it with `uv run train {task}`.",
  )

  cases = ladder(cfg)
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = len(cases) + 1
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  try:
    if "motion" not in env.command_manager.active_terms:
      raise SystemExit(
        f"'{cfg.skill}' tracks no reference, so there is no trajectory to be off. This "
        "measures tracking error, which a commanded locomotion skill does not have."
      )
    command = env.command_manager.get_term("motion")
    if not isinstance(command, JumpCommand):
      raise SystemExit(
        f"'{cfg.skill}' has a motion command that is not a clip tracker."
      )

    print(f"[tolerance] skill: {cfg.skill}, task {task}")
    print(f"[tolerance] policy: {checkpoint}")
    print(f"[tolerance] entry {cfg.entry}: {entry.name}, frame {entry.frame}")
    print(f"[tolerance] {len(cases)} displaced rollouts against one undisturbed")

    # Before anything is placed. Building it wraps the env, and RslRlVecEnvWrapper resets
    # on construction because rsl-rl does not: a policy built after the displacement resets
    # every robot back onto the reference and the whole measurement reads as a skill that
    # tolerates everything
    policy = _policy(task, checkpoint, env, cfg.device)

    # The reset anchors the clip at the env origin and places whatever the scene carries
    # against it, so winding the phase to the entry frame leaves the two consistent
    env.reset(seed=cfg.seed)
    resume.prepare(env, entry)
    _prepare_scene(env, cfg.skill)
    base = resume.target(env, entry)

    arms = arm_mask(tuple(env.scene["robot"].joint_names), env.device)
    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    shapes = {
      (channel, direction): _shape(channel, arms, generator, cfg.device)
      for channel in CHANNELS
      for direction in range(cfg.directions)
    }

    # Each case is displaced from its own env's copy of the entry, never from env 0's. The
    # copies differ by the env origin, and a robot written onto another env's origin is a
    # metre from its own reference before the displacement is counted at all
    targets = torch.cat(
      [base[:1]]
      + [
        displace(
          base[index + 1 : index + 2],
          case.channel,
          shapes[(case.channel, case.direction)],
          case.amount,
        )
        for index, case in enumerate(cases)
      ]
    )
    injected = channel_errors(targets, base, arms)
    _place(env, entry, targets)

    # An env that terminates is auto reset, so its error is frozen on the done flag rather
    # than read off a robot already living somebody else's fresh episode
    obs = _fresh_obs(env)
    error = tracking_error(command)
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    trace = []
    for _ in range(round(cfg.horizon / env.step_dt)):
      obs, _, terminated, truncated, _ = env.step(policy(obs))
      done |= terminated | truncated
      error = torch.where(done, error, tracking_error(command))
      trace.append(error)
    tracked = torch.stack(trace[round(cfg.settle / env.step_dt) :]).mean(dim=0)

    column = {channel: index for index, channel in enumerate(CHANNELS)}
    outcomes = [
      Outcome(
        case=case,
        injected=float(injected[index + 1, column[case.channel]]),
        tracked=float(tracked[index + 1]),
        fell=bool(done[index + 1]),
      )
      for index, case in enumerate(cases)
    ]
    return float(tracked[0]), outcomes, entry, checkpoint
  finally:
    env.close()


##
# Reading the ladder.
##


@dataclass
class Limit:
  channel: str
  tolerates: float
  """Largest displacement every case held at, in the channel's unit."""
  exhausted: bool
  """The ladder ran out before the skill did, so this is a floor and not a limit."""
  failed_at: float
  """The rung that broke it, or the top rung when exhausted."""
  fell_at: float
  """Share of cases on the breaking rung that terminated. Says whether the skill lost its
  trajectory or lost its footing, which are different problems."""


def limits(floor: float, outcomes: list[Outcome], cfg: Config) -> list[Limit]:
  """Read each channel's ladder from the bottom up, stopping at the first rung that fails."""
  ceiling = floor + cfg.margin
  out = []
  for channel in CHANNELS:
    rungs = sorted({o.case.rung for o in outcomes if o.case.channel == channel})
    tolerates, failed_at, fell_at, exhausted = 0.0, rungs[-1], 0.0, True
    for rung in rungs:
      group = [o for o in outcomes if o.case.channel == channel and o.case.rung == rung]
      if max(o.tracked for o in group) > ceiling or any(o.fell for o in group):
        failed_at = min(o.injected for o in group)
        fell_at = sum(o.fell for o in group) / len(group)
        exhausted = False
        break
      # The smallest of the group, so a limit understates rather than overstates
      tolerates = min(o.injected for o in group)
    out.append(Limit(channel, tolerates, exhausted, failed_at, fell_at))
  return out


def render(
  cfg: Config,
  entry: Entry,
  checkpoint: Path,
  floor: float,
  found: list[Limit],
  outcomes: list[Outcome],
) -> str:
  """The report, as markdown. The same text is printed and written to disk."""
  out: list[str] = []
  out.append(f"# Entry tolerance: {cfg.skill} entry {cfg.entry}")
  out.append("")
  for label, value in [
    ("skill", f"{cfg.skill} ({SKILLS[cfg.skill]})"),
    ("entry", f"{cfg.entry}, {entry.name}, clip frame {entry.frame}"),
    ("checkpoint", str(checkpoint)),
    ("on track means", f"tracking within {cfg.margin * 100:.0f} cm of undisturbed"),
    ("undisturbed tracking", f"{floor * 100:.1f} cm"),
    (
      "measured over",
      f"{cfg.settle:.1f} s to {cfg.horizon:.1f} s after the entry",
    ),
    ("cases", f"{len(outcomes)}, {2 * cfg.directions} per rung per channel"),
    ("scored", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
  ]:
    out.append(f"- **{label}**: {value}")

  out.append("")
  out.append("## What this entry tolerates")
  out.append("")
  out.append("| channel | tolerates | unit | broke at | of those, fell |")
  out.append("|---|---|---|---|---|")
  for limit in found:
    note = " (ladder ran out)" if limit.exhausted else ""
    broke = "-" if limit.exhausted else f"{limit.failed_at:.3g}"
    out.append(
      f"| {limit.channel} | {limit.tolerates:.3g}{note} | {UNITS[limit.channel]} "
      f"| {broke} | {limit.fell_at * 100:.0f}% |"
    )

  out.append("")
  out.append(
    "`tolerates` is the largest displacement of that channel where every sign and every "
    "direction still came back to the clip. `broke at` is the next rung up. `of those, "
    "fell` is how many of the breaking cases terminated rather than merely wandering: a "
    "low number there means the skill stayed upright and stopped doing the motion, which "
    "is the failure a survival test cannot see."
  )

  out.append("")
  out.append("## As an entry profile")
  out.append("")
  out.append("Paste into `tests/entry_tolerances.py` to score a bridge against it.")
  out.append("")
  out.append("```python")
  out.append("Tolerances(")
  for limit in found:
    out.append(f"  {limit.channel}={limit.tolerates:.3g},")
  out.append(")")
  out.append("```")

  out.append("")
  out.append("## Reading this")
  out.append("")
  out.append(
    "One channel is displaced at a time. Eight channels each tolerating their own limit "
    "does not mean all eight at once do, and nothing here measures that: a hand-over "
    "misses on every channel simultaneously. Treat these as upper bounds on a joint "
    "envelope, not as the envelope."
  )
  out.append("")
  out.append(
    "The measure is tracking error, not survival. A skill that absorbs a displacement, "
    "stays upright and finishes the motion as something else has failed here and would "
    "pass a fall check, which is the whole reason this is measured separately."
  )
  out.append("")
  return "\n".join(out)


def main(cfg: Config) -> None:
  floor, outcomes, entry, checkpoint = measure(cfg)
  found = limits(floor, outcomes, cfg)
  text = render(cfg, entry, checkpoint, floor, found, outcomes)
  print()
  print(text)
  if cfg.write_report:
    out = cfg.report_dir / f"{cfg.skill}_entry{cfg.entry}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[tolerance] wrote {out}")


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
