"""How close to the target does a bridge actually put the robot. One script, any architecture.

Draws windows from the eval split of the shared corpus, runs the chosen bridge across each
one, and reports the gap left standing at the best moment of every crossing, in the units
the gap is measured in: metres, metres per second, radians, radians per second. Beside it,
the same numbers for a robot that does nothing, because a distance means little without the
distance you get for free.

Every architecture is scored by this one script against the same corpus with the same code,
which is the reason the corpus sits in bridges/dataset rather than inside an architecture.
Two reports are comparable line by line.

Run

1. Score the newest checkpoint of an architecture. The path is printed.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.evaluate \
      --bridge distillation

2. Score a particular checkpoint.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.evaluate \
      --bridge imitation \
      --checkpoint logs/rsl_rl/g1_imitation_bridge/<run>/model_5000.pt

3. Score it the way it is actually used, taking over from a robot that is not exactly on a
   recorded state. See EvalCfg.start_noise.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.evaluate \
      --bridge distillation --start-noise 1.0

Every run also writes logs/benchmarks/<bridge>/<run>_<checkpoint>.md, one file per
architecture per checkpoint, so a score can be found again without re-running it.

Reading the report

    delivered        every channel inside its requirement at the same moment. The honest
                     number, and the only one that is not a kernel
    stayed up        crossings that did not end on the floor. Every error below is measured
                     over these only: a window that fell has no arrival to be wrong about,
                     and scoring it as a miss would flatter a policy that falls often
    median, 9 in 10  the gap left standing, per channel. Read the requirement column beside
                     them: what matters is the ratio, and the units differ per channel
    worst joints     which individual joints carry the leg and arm channel numbers. The
                     channel is a worst-joint maximum, so one bad joint sets it
    hand-over        the same delivery scored against the envelope a receiving skill can
                     resume from, and how much wider that envelope would have to be for a
                     given share of crossings to land inside it. See Handoff

The requirements come from Tolerances and are physical: they are what the next skill needs
to be handed, not a function of how hard the corpus currently is. They do not move when the
corpus is rebuilt, which is what makes two reports months apart comparable.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges import (
  DEFAULT_BRIDGE,
  BridgeKind,
  BridgeSpec,
  resolve,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  find_checkpoint,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp.commands import (
  CHANNELS,
  UNITS,
  BridgeCommand,
  BridgeCommandCfg,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import reset_policy

COMMAND = "bridge"

REPORT_ROOT = Path("logs") / "benchmarks"

DEGREES = {"rad": "deg", "rad/s": "deg/s"}
"""Angular channels are also printed in degrees. Three degrees of torso tilt is a thing a
person can picture and 0.05 rad is not."""


@dataclass
class EvalCfg:
  bridge: BridgeKind = DEFAULT_BRIDGE
  """Which architecture to score. A stub is refused by bridges.resolve before anything
  starts."""

  checkpoint: str | None = None
  """Explicit checkpoint. Empty takes the newest under the architecture's own experiment
  and prints the path, because picking by modification time has loaded the wrong policy
  here before."""

  episodes: int = 1024
  num_envs: int = 256
  dataset: Path = DEFAULT_DATASET
  split: str = "eval"
  """Which side of the corpus split. eval holds environments no training window came
  from."""

  device: str = "cuda:0"
  seed: int = 0

  start_noise: float = 0.0
  """How hard to perturb the start state off its recorded value, 0 to 1.

  Zero is the clean measurement and the default, so a report is comparable to every other
  report. One is the honest deployment measurement: at inference a bridge takes over from
  wherever the outgoing skill left the robot, never from a corpus row, and a policy scored
  only from exact recorded starts has never been asked to steer.

  Setting this also collapses the perturbation ramp to immediate. The ramp is written in
  environment steps for a training run and would stay near zero for a whole evaluation.
  """

  worst_joints: int = 6
  """How many individual joints to name, ranked by how often they are the bottleneck.

  The four joint channels are worst-joint maxima, so a channel that misses is one joint
  that misses. Naming it is the difference between "the legs are 0.3 rad out" and "the
  left knee is 0.3 rad out", and only the second one tells you what to do.
  """

  baseline: bool = True
  """Score a robot holding its default pose beside the policy. Keep it on: the channel
  errors have no absolute meaning and every number here is only ever a number next to
  another number."""

  rates: tuple[float, ...] = (0.5, 0.9)
  """Which share of crossings the hand-over section reports a profile for.

  A hand-over either lands inside what the next skill needs or it does not, so the useful
  question is not the average error but how wide the requirement would have to be for a
  given share of crossings to clear it. 0.5 is the typical crossing, 0.9 is close to the
  worst one anybody should plan around.
  """

  report_dir: Path = REPORT_ROOT
  write_report: bool = True


@dataclass
class Delivery:
  """One policy's worth of finished windows."""

  name: str
  arrived: torch.Tensor
  """(episodes,) every channel inside its requirement at one moment."""
  score: torch.Tensor
  arrival_s: torch.Tensor
  fell: torch.Tensor
  errors: torch.Tensor
  """(episodes, 8) the gap at the best moment, in the units of UNITS."""
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  """(episodes, J) per joint, at the same moment."""

  @property
  def kept(self) -> torch.Tensor:
    """Crossings that stayed on their feet. Everything per channel is measured over these."""
    return self.fell < 1

  def quantile(self, values: torch.Tensor, q: float) -> float:
    live = values[self.kept]
    return float(live.quantile(q)) if live.numel() else float("nan")


##
# Building.
##


def _build(cfg: EvalCfg, spec: BridgeSpec) -> ManagerBasedRlEnv:
  """The architecture's own play environment, pointed at the eval corpus.

  Play and not train, so the observation carries no noise and the tolerance curriculum is
  off: a report is against the fixed requirements, never against wherever a run's
  curriculum happened to have got to. For an architecture with a conditioning mask, play is
  also what forces the deployment pattern, so what is scored is the bridge problem.
  """
  env_cfg = load_env_cfg(spec.task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  command = env_cfg.commands[COMMAND]
  assert isinstance(command, BridgeCommandCfg)
  command.dataset_path = cfg.dataset
  command.split = cfg.split
  if cfg.start_noise > 0.0:
    command.start_noise = cfg.start_noise
    command.start_noise_steps = 1
  # The rollout reads each window's best moment off the command, and an auto-reset would
  # draw the next window and clear it before step returns
  env_cfg.auto_reset = False
  return ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)


def _command(env: ManagerBasedRlEnv) -> BridgeCommand:
  term = env.command_manager.get_term(COMMAND)
  assert isinstance(term, BridgeCommand)
  return term


def _trained(env: ManagerBasedRlEnv, cfg: EvalCfg, spec: BridgeSpec, path: Path):
  """The architecture's policy, loaded through its own runner.

  Through the runner and not by reading the state dict, because an architecture is free to
  train in a shape that is not the shape it deploys in. The distillation bridge is: its
  checkpoint holds a student and a teacher, and its runner is what knows the student is the
  thing that acts.
  """
  from tensordict import TensorDict

  agent_cfg = load_rl_cfg(spec.task_id)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(spec.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
  try:
    runner.load(
      str(path), load_cfg={"actor": True}, strict=True, map_location=cfg.device
    )
  except ValueError as error:
    raise SystemExit(f"[eval] {path} holds no deployable policy.\n  {error}") from error
  inference = runner.get_inference_policy(device=cfg.device)

  class Driven:
    """The loaded policy in the shape rollout drives, keeping the model reachable so its
    hidden state can be cleared between windows."""

    # Asked for rather than assumed. An architecture is free to deploy something that is
    # not an rsl-rl model at all: the diffusion bridge hands back a plain closure over a
    # sampler, which has no hidden state to clear and no attribute saying so
    is_recurrent = getattr(inference, "is_recurrent", False)

    def __call__(self, obs):
      with torch.inference_mode():
        return inference(TensorDict(obs, batch_size=[cfg.num_envs]))

    def reset(self, dones) -> None:
      inference.reset(dones)

  return Driven()


def _statue(env: ManagerBasedRlEnv):
  """Hold the default pose. The action term offsets from the default, so a zero action is
  exactly that: the do nothing baseline, not merely a weak policy."""
  shape = env.action_space.shape

  def policy(obs):
    del obs
    return torch.zeros(shape, device=env.device)

  return policy


##
# Measuring.
##


def rollout(env: ManagerBasedRlEnv, policy, cfg: EvalCfg, name: str) -> Delivery:
  """Run windows until `episodes` of them have finished, recording every arrival."""
  command = _command(env)
  torch.manual_seed(cfg.seed)
  obs, _ = env.reset(seed=cfg.seed)

  collected: dict[str, list[torch.Tensor]] = {
    key: [] for key in ("arrived", "score", "arrival_s", "fell", "errors", "q", "qd")
  }
  finished = 0

  while finished < cfg.episodes:
    action = policy(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    reset_policy(policy, terminated | truncated)
    done = (terminated | truncated).nonzero().flatten()
    if done.numel():
      # Read before the reset. advance keeps all of this current at the window's best
      # moment, and the next place clears it
      collected["arrived"].append(command.arrived[done].clone())
      collected["score"].append(command.score[done].clone())
      collected["arrival_s"].append(command.arrival_s[done].clone())
      collected["fell"].append(terminated[done].float().clone())
      collected["errors"].append(command.final[done].clone())
      collected["q"].append(command.final_joint_pos[done].clone())
      collected["qd"].append(command.final_joint_vel[done].clone())
      finished += int(done.numel())
      obs, _ = env.reset(env_ids=done)
      print(f"[eval] {name}: {finished}/{cfg.episodes} windows", end="\r")

  print()
  cut = cfg.episodes
  return Delivery(
    name=name,
    arrived=torch.cat(collected["arrived"])[:cut],
    score=torch.cat(collected["score"])[:cut],
    arrival_s=torch.cat(collected["arrival_s"])[:cut],
    fell=torch.cat(collected["fell"])[:cut],
    errors=torch.cat(collected["errors"])[:cut],
    joint_pos=torch.cat(collected["q"])[:cut],
    joint_vel=torch.cat(collected["qd"])[:cut],
  )


@dataclass
class ChannelRow:
  """One channel, ready to print. Everything already in physical units."""

  channel: str
  unit: str
  requirement: float
  median: float
  p90: float
  met: float
  baseline: float
  """The statue's median on this channel, or nan when no baseline was run."""

  @property
  def reach(self) -> float:
    """Median error over the requirement. One is the limit, whatever the unit was."""
    return self.median / self.requirement if self.requirement else float("nan")

  def also(self, value: float) -> str:
    """The same number in degrees, for the angular channels. Empty for the rest."""
    if self.unit not in DEGREES:
      return ""
    return f"{value * 180.0 / math.pi:.1f} {DEGREES[self.unit]}"


def channel_rows(
  result: Delivery, tolerances: torch.Tensor, baseline: Delivery | None
) -> list[ChannelRow]:
  rows = []
  for index, channel in enumerate(CHANNELS):
    limit = float(tolerances[index])
    inside = result.errors[result.kept, index] <= limit
    rows.append(
      ChannelRow(
        channel=channel,
        unit=UNITS[channel],
        requirement=limit,
        median=result.quantile(result.errors[:, index], 0.5),
        p90=result.quantile(result.errors[:, index], 0.9),
        met=float(inside.float().mean()) if inside.numel() else float("nan"),
        baseline=baseline.quantile(baseline.errors[:, index], 0.5)
        if baseline is not None
        else float("nan"),
      )
    )
  return rows


@dataclass
class JointRow:
  """One joint, and how much of its channel's number it is responsible for."""

  name: str
  blamed: float
  """Share of crossings where this joint is the worst in its group.

  Ranked on, rather than on the error itself, because that is the question the channel
  asks. leg_joint_pos is a maximum over the legs within each crossing, so its median is
  the median of whichever joint happened to be worst that time. A table of per joint
  medians does not add up to it and looks like a contradiction: every joint reads well
  under the channel, because no single joint is worst often enough for its own median to
  be the channel's.
  """

  median: float
  """This joint's position error across the crossings it was blamed for, in radians."""
  velocity: float
  """Its velocity error over the same crossings."""


def joint_rows(
  result: Delivery, names: tuple[str, ...], arms: torch.Tensor, count: int
) -> list[JointRow]:
  """Which joints are actually the bottleneck, and by how much when they are.

  Blame is counted inside each group, legs and arms separately, because the two are
  separate channels with separate requirements. A joint that is never the worst in its
  group contributes nothing to its channel however badly it tracks.
  """
  kept = result.kept
  pos, vel = result.joint_pos[kept], result.joint_vel[kept]
  if not pos.numel() or count <= 0:
    return []

  blame = torch.zeros(pos.shape[1], device=pos.device)
  culprit = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
  for group in (~arms, arms):
    if not bool(group.any()):
      continue
    members = group.nonzero().flatten()
    worst = members[pos[:, group].argmax(dim=1)]
    blame.index_add_(0, worst, torch.ones_like(worst, dtype=blame.dtype))
    culprit = torch.where(
      pos.gather(1, worst.unsqueeze(-1)).squeeze(-1)
      > pos.gather(1, culprit.unsqueeze(-1)).squeeze(-1),
      worst,
      culprit,
    )
  blame /= pos.shape[0]

  rows = []
  for index in torch.argsort(blame, descending=True)[:count].tolist():
    # Over the crossings this joint was worst in, not over all of them. Its typical error
    # when it is the problem is the number that explains the channel
    when = culprit == index
    sample = when if bool(when.any()) else torch.ones_like(when)
    rows.append(
      JointRow(
        name=names[index],
        blamed=float(blame[index]),
        median=float(pos[sample, index].median()),
        velocity=float(vel[sample, index].median()),
      )
    )
  return rows


##
# Will a hand-over land.
##


@dataclass
class Handoff:
  """How this delivery sits against one skill's entry requirement.

  The bridge is only ever a means: what matters is whether the policy that takes over can
  resume from where it was left. That policy tolerates some envelope of error, and the
  hand-over succeeds when the delivery lands inside it. Everything here is that comparison,
  and nothing here is a kernel or a score.
  """

  name: str
  note: str
  tolerances: torch.Tensor
  """(8,) the envelope, in physical units."""

  delivered: float
  """Share of surviving crossings inside every channel of it at one moment."""

  headroom: dict[float, float]
  """Rate to the multiple of this envelope that share of crossings actually clears.

  Computed from the worst channel of each crossing, so it is a joint statement about all
  eight at once and not eight separate marginal ones. 1.0 would mean that share of
  crossings already lands inside the envelope as written.
  """

  binding: list[tuple[str, float]]
  """Channel, and the share of crossings where it is the one furthest outside. What would
  have to be fixed, in the order it would have to be fixed."""

  marginal: dict[str, float]
  """Channel to the multiple of its own limit it clears at the first rate, ignoring the
  other seven. The gap between this and headroom is how much the joint requirement costs
  over any single one of them."""


def handoff(
  result: Delivery,
  tolerances: torch.Tensor,
  rates: tuple[float, ...],
  name: str,
  note: str,
) -> Handoff:
  """Score one delivery against one entry envelope."""
  ratio = result.errors[result.kept] / tolerances
  worst = ratio.amax(dim=-1)
  blame = ratio.argmax(dim=-1)
  return Handoff(
    name=name,
    note=note,
    tolerances=tolerances,
    delivered=float((worst <= 1.0).float().mean()) if worst.numel() else float("nan"),
    headroom={r: float(worst.quantile(r)) for r in rates},
    binding=[
      (channel, float((blame == index).float().mean()))
      for index, channel in enumerate(CHANNELS)
    ],
    marginal={
      channel: float(ratio[:, index].quantile(rates[0]))
      for index, channel in enumerate(CHANNELS)
    },
  )


def requirement_profiles(
  baseline: torch.Tensor,
) -> list[tuple[str, torch.Tensor, str]]:
  """Every envelope worth scoring against: the physical baseline, then the skill entries.

  Entries are deduplicated by value. They are currently all the same provisional profile,
  and printing seven identical rows would suggest seven measurements where there is one.
  The skills sharing a profile are named beside it instead.

  Imported here rather than at module load, so this script still runs if the entry table
  is mid-rebuild.
  """
  from mjlab.tasks.bridging.experiments.humanoid.tests.entry_tolerances import (
    ENTRY_TOLERANCES,
  )

  out = [
    (
      "hand-over baseline",
      baseline,
      "Tolerances, argued from what a hand-over costs. Not skill specific",
    )
  ]
  shared: dict[tuple[float, ...], list[str]] = {}
  for skill, frames in sorted(ENTRY_TOLERANCES.items()):
    for frame, profile in sorted(frames.items()):
      key = tuple(float(getattr(profile, channel)) for channel in CHANNELS)
      shared.setdefault(key, []).append(f"{skill} {frame}")
  for index, (key, users) in enumerate(shared.items()):
    out.append(
      (
        f"skill entry {index + 1}" if len(shared) > 1 else "skill entry",
        torch.tensor(key, device=baseline.device, dtype=baseline.dtype),
        "asked by " + ", ".join(users),
      )
    )
  return out


##
# Rendering.
##


def _headline(result: Delivery) -> list[tuple[str, str]]:
  kept = result.kept
  return [
    ("windows", f"{result.arrived.numel()}"),
    ("stayed up", f"{float((~result.fell.bool()).float().mean()) * 100:.1f}%"),
    ("delivered", f"{float(result.arrived.mean()) * 100:.1f}%"),
    (
      "time to best moment",
      f"{float(result.arrival_s[kept].median()):.2f} s" if bool(kept.any()) else "n/a",
    ),
    ("arrival score", f"{float(result.score.mean()):.3f}"),
  ]


def render(
  spec: BridgeSpec,
  path: Path,
  cfg: EvalCfg,
  result: Delivery,
  baseline: Delivery | None,
  rows: list[ChannelRow],
  joints: list[JointRow],
  conditions: list[tuple[str, str]],
  handoffs: list[Handoff],
) -> str:
  """The report, as markdown. The same text is printed and written to disk."""
  out: list[str] = []
  now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
  out.append(f"# Bridge delivery: {spec.kind}")
  out.append("")
  for label, value in [
    ("architecture", spec.kind),
    ("task", spec.task_id),
    ("checkpoint", str(path)),
    ("corpus", f"{cfg.dataset} ({cfg.split} split)"),
    ("scored", now),
  ] + conditions:
    out.append(f"- **{label}**: {value}")

  out.append("")
  out.append("## Headline")
  out.append("")
  out.append("| | |")
  out.append("|---|---|")
  for label, value in _headline(result):
    out.append(f"| {label} | {value} |")
  if baseline is not None:
    out.append(f"| statue delivered | {float(baseline.arrived.mean()) * 100:.1f}% |")
    out.append(f"| statue arrival score | {float(baseline.score.mean()):.3f} |")

  out.append("")
  out.append("## How close it gets, per channel")
  out.append("")
  out.append(
    "| channel | unit | requirement | median | median (deg) | 9 in 10 | x requirement "
    "| within requirement | statue median |"
  )
  out.append("|---|---|---|---|---|---|---|---|---|")
  for row in rows:
    out.append(
      f"| {row.channel} | {row.unit} | {row.requirement:.3g} | {row.median:.3f} "
      f"| {row.also(row.median) or '-'} | {row.p90:.3f} | {row.reach:.1f}x "
      f"| {row.met * 100:.0f}% | {row.baseline:.3f} |"
    )

  out.append("")
  out.append("## Worst joints")
  out.append("")
  if joints:
    out.append(
      "Which joint sets its group's channel, and its error on the crossings where it does."
    )
    out.append("")
    out.append("| joint | worst in its group | position error | | velocity error |")
    out.append("|---|---|---|---|---|")
    for row in joints:
      out.append(
        f"| {row.name} | {row.blamed * 100:.0f}% | {row.median:.3f} rad "
        f"| {row.median * 180.0 / math.pi:.1f} deg | {row.velocity:.2f} rad/s |"
      )
    out.append("")
    out.append(
      "The four joint channels are worst-joint maxima within a crossing, so they are set "
      "by whichever joint was worst that time. Blame spread evenly over many joints means "
      "the whole chain is loose; blame concentrated on one or two means there is something "
      "specific to fix."
    )
  else:
    out.append("No crossing stayed on its feet, so there is no arrival to attribute.")

  out.append("")
  out.append("## Will a hand-over land")
  out.append("")
  out.append(
    "A hand-over works when the delivery lands inside the envelope the next policy can "
    "resume from. `inside as written` is that, exactly. The two columns beside it say how "
    "much wider the envelope would have to be for that share of crossings to clear it, "
    "measured on the worst channel of each crossing, so it is a statement about all eight "
    "at once."
  )
  out.append("")
  rates = list(handoffs[0].headroom) if handoffs else []
  header = "| envelope | inside as written |" + "".join(
    f" needs this multiple for {r:.0%} |" for r in rates
  )
  out.append(header)
  out.append("|---" * (2 + len(rates)) + "|")
  for hand in handoffs:
    out.append(
      f"| {hand.name} | {hand.delivered * 100:.1f}% |"
      + "".join(f" {hand.headroom[r]:.1f}x |" for r in rates)
    )
  out.append("")
  for hand in handoffs:
    out.append(f"- **{hand.name}**: {hand.note}")

  for hand in handoffs:
    out.append("")
    out.append(f"### {hand.name}: what it actually delivers")
    out.append("")
    out.append(
      "| channel | unit | asked | "
      + " | ".join(f"met by {r:.0%} (all eight)" for r in rates)
      + " | this channel alone | blocks the hand-over |"
    )
    out.append("|---" * (5 + len(rates)) + "|")
    for index, channel in enumerate(CHANNELS):
      limit = float(hand.tolerances[index])
      cells = "".join(f" {limit * hand.headroom[r]:.3g} |" for r in rates)
      blocks = dict(hand.binding)[channel]
      out.append(
        f"| {channel} | {UNITS[channel]} | {limit:.3g} |"
        + cells
        + f" {limit * hand.marginal[channel]:.3g} |"
        + f" {blocks * 100:.0f}% |"
      )
    out.append("")
    worst = max(hand.binding, key=lambda pair: pair[1])
    out.append(
      f"`blocks the hand-over` is the share of crossings where that channel is the one "
      f"furthest outside. Here it is **{worst[0]}** on {worst[1] * 100:.0f}% of them, so "
      f"that is what a wider envelope would have to forgive first, and what a better "
      f"bridge would have to fix first."
    )
    out.append("")
    out.append(
      "`this channel alone` is what the bridge clears on that channel ignoring the other "
      "seven. It is always tighter than the all-eight column, and the gap between them is "
      "the price of needing every channel right at the same moment."
    )

  out.append("")
  out.append("## Reading this")
  out.append("")
  out.append(
    "Channel errors are over the crossings that stayed on their feet. A window that ended "
    "on the floor has no arrival to be wrong about, and counting it as a miss would "
    "flatter a policy that falls often, so read every error against `stayed up`."
  )
  out.append("")
  out.append(
    "`delivered` is the only number here that is not a kernel: every one of the eight "
    "channels inside its requirement at the same moment. Seven out of eight is not a "
    "hand-over. The requirements are physical and do not move when the corpus is rebuilt."
  )
  out.append("")
  out.append(
    "`x requirement` is the median over the limit, so the eight are comparable to each "
    "other whatever their units, and the largest is the one thing standing between this "
    "bridge and a delivery."
  )
  out.append("")
  return "\n".join(out)


def report_path(root: Path, spec: BridgeSpec, checkpoint: Path) -> Path:
  """logs/benchmarks/<bridge>/<run>_<checkpoint>.md. One file per architecture per
  checkpoint, with the run in the name so two runs' model_2000 do not collide."""
  run = checkpoint.parent.name
  return root / spec.kind / f"{run}_{checkpoint.stem}.md"


##
# Entry point.
##


def _conditions(env: ManagerBasedRlEnv, cfg: EvalCfg) -> list[tuple[str, str]]:
  """What the policy was asked for, read off the command rather than restated here.

  bridge_prob only exists on an architecture that has a conditioning mask, so it is read by
  name and skipped where there is none. That keeps this script free of any knowledge of
  which architectures those are.
  """
  command = _command(env)
  low, high = command.cfg.duration_s_range
  rows = [
    ("command", type(command).__name__),
    (
      "window",
      f"{low:.2g} to {high:.2g} s, patience x{command.cfg.patience_scale:.2g}",
    ),
    ("start perturbation", f"{cfg.start_noise:.2g} of full scale"),
  ]
  bare = getattr(command.cfg, "bridge_prob", None)
  if bare is not None:
    rows.append(("interior visible", f"never ({bare:.0%} of windows drawn bare)"))
  return rows


def main(cfg: EvalCfg) -> None:
  spec = resolve(cfg.bridge)
  path = find_checkpoint(
    (spec.experiment,),
    cfg.checkpoint,
    hint=f" Train one with `uv run train {spec.task_id}`.",
  )
  print(f"[eval] architecture: {spec.kind}")
  print(f"[eval] policy: {path}")

  env = _build(cfg, spec)
  try:
    conditions = _conditions(env, cfg)
    result = rollout(env, _trained(env, cfg, spec, path), cfg, spec.kind)
    baseline = rollout(env, _statue(env), cfg, "statue") if cfg.baseline else None
    command = _command(env)
    rows = channel_rows(result, command.tolerances, baseline)
    joints = joint_rows(
      result, tuple(command.robot.joint_names), command.arms, cfg.worst_joints
    )
    handoffs = [
      handoff(result, profile, cfg.rates, name, note)
      for name, profile, note in requirement_profiles(command.tolerances)
    ]
    text = render(spec, path, cfg, result, baseline, rows, joints, conditions, handoffs)
  finally:
    env.close()

  print()
  print(text)
  if cfg.write_report:
    out = report_path(cfg.report_dir, spec, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
  main(tyro.cli(EvalCfg, config=mjlab.TYRO_FLAGS))
