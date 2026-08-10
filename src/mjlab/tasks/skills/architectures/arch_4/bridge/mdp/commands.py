"""The window the bridge is asked to cross, and the reference it is scored against.

One episode is one hole. On reset the robot is teleported onto the hand-off -- the last
context frame of a window drawn from the corpus -- carrying that frame's joint angles and,
crucially, its velocities. The momentum the whole architecture exists to exploit is in
those velocities, and a reset that dropped them would be posing the robot rather than
handing it over.

The clip is then rebased onto the environment. The hand-off frame's horizontal position is
translated to the environment origin and nothing else is touched, so the reference and the
robot live in the same coordinates and every comparison downstream is a plain subtraction.
A pure translation is enough because the robot is spawned at the clip's own heading; there
is no rotation to undo.

##
# What the policy is told, and what it is not
##

It is told where it has to arrive: the future context, as poses relative to wherever the
robot is right now, refreshed every step. It is told how far through the hole it is and
how long the hole is.

It is not told what the body actually did inside the hole. That absence is the
architecture. A policy shown the reference it is being scored against learns to follow a
reference, and at inference there is no reference to follow -- only a target handed over by
the descriptor. A policy shown two ends and paid for what went between has to learn what
usually goes between, which is the part that transfers.

The reference is still read every step, by the reward. Training signal and observation are
different channels and only one of them is allowed to see the answer.

##
# The episode does not end when the hole does
##

An episode runs the hole and then the whole second unmasked window on top of it, with the
policy still driving. That second stretch is the `resume`, and it is not a cool-down.

The failure it exists to catch is the one a hole-only reward cannot see. A bridge paid
only for the frames inside the hole is paid for a sum, and a sum is indifferent to where
the shortfall lands: crossing beautifully and arriving a hand's breadth out of position
scores about the same as crossing scrappily and arriving on the mark. Only the second of
those is any use. Whatever runs next starts from the state the bridge leaves behind and
expects that state to be the one its own motion begins in, so an arrival error is not a
few lost points, it is the next skill's initial condition being wrong.

So the state at the hand-off is measured once, the moment it happens, and turned into
`hand_off_score`: one number in (0, 1] saying how much of what comes next the bridge has
actually earned the right to. Everything the resume pays is multiplied by it. A bridge
that arrives badly cannot buy the second window back by tracking it well afterwards,
because at inference it will not be the one tracking it -- and it can lose the second
window outright by falling over during it, which terminates the episode as a failure and
not as a time-out.

That is the whole coupling, and it is deliberately one-directional: the hole's own reward
is untouched, and the pressure reaches the actions inside the hole the way it should,
through the value of the state they hand over.

##
# The two halves do not have to belong together
##

A window cut out of one recording gives a pair that is coherent by construction: the same
body, one stride later, going the way it was already going. At inference the two halves are
whatever two skills the composition happens to name, and nothing guarantees any of that.
Trained only on coherent pairs, a bridge learns to continue a motion, which is a much
easier problem than the one it will be asked, and the difference does not show up until the
composition is assembled and fails.

So a share of environments get a window whose two halves were never adjacent. Three things
can be varied, independently, and each is a dial rather than a switch (see `SpliceCfg`):

- **where** the second motion begins, relative to the hand-off: further off, and turned;
- **what** it is, taken from another window entirely rather than from this one;
- **how long** the bridge has to get there.

For those environments the recorded hole is thrown away, because it led somewhere else, and
with it goes the only frame-by-frame answer there was. What remains is the arrival and the
resume, which is exactly the signal that transfers -- and `mdp/rewards.approach` is what
keeps the hole dense enough to learn from once the reference in it is gone.

Both kinds of window are drawn in the same batch, and the mix ramps: a run opens on
coherent pairs alone and walks toward mismatched ones over `SpliceCfg.warmup_steps`. That
ramp is the warm start. A policy that cannot yet keep a robot upright is in no position to
invent a transition between two unrelated motions, and starting it there asks two questions
at once and gets neither answered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.skills.architectures.arch_4.bridge import frames
from mjlab.tasks.skills.architectures.arch_4.bridge.dataset import (
  ROOT_STATE_DIM,
  BridgeDataset,
  Corpus,
  CorpusCfg,
  WindowCfg,
  build_corpus,
  rebase_states,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_error_magnitude,
  quat_from_angle_axis,
  quat_mul,
  yaw_quat,
)


class BridgeCommand(CommandTerm):
  """Draws a window per environment, teleports onto its hand-off, holds its reference."""

  cfg: BridgeCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: BridgeCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    corpus = Corpus(build_corpus(cfg.corpus), mirror=cfg.corpus.mirror)
    dataset = BridgeDataset(corpus, cfg.windows, split=cfg.split)
    if not len(dataset):
      raise ValueError(f"No windows in the '{cfg.split}' split of this corpus.")

    self.num_joints = corpus.num_joints
    self.pose_dim = frames.pose_dim(self.num_joints)
    self.past = cfg.windows.past
    self.future = cfg.windows.future
    self.gap_range = cfg.windows.gap_range
    self.max_gap = cfg.windows.gap_range[1]
    # The corpus rate, which is also the control rate. A spliced window is placed a
    # distance ahead that depends on how many frames the bridge is given, and frames only
    # become seconds through this.
    self.fps = corpus.fps

    # How many frames past the hand-off the second window is played out for. Capped at
    # what the corpus actually holds: asking for more resume than there is future context
    # would score the robot against a frame that was clamped rather than recorded, which
    # looks like the motion standing still and is not what the body did.
    self.resume = min(cfg.resume_steps, self.future - 1)
    if self.resume < 1:
      raise ValueError(
        f"resume_steps is {cfg.resume_steps} against a {self.future}-frame future "
        f"context, leaving {self.resume} frames of resume. It has to be at least 1: an "
        f"episode that stops at the hand-off never finds out whether the arrival was "
        f"usable, which is the question this task exists to answer."
      )

    # The corpus as one tensor, with a window reduced to an offset into it. Sampling is
    # then index arithmetic and the whole corpus lives on the device once.
    lengths = [corpus.length(i) for i in range(len(corpus))]
    offsets = torch.tensor(
      [sum(lengths[:i]) for i in range(len(lengths))], device=self.device
    )
    self.flat = torch.cat(corpus.states, dim=0).to(self.device)
    self.window_base = (
      torch.tensor([w.start for w in dataset.windows], device=self.device)
      + offsets[torch.tensor([w.clip for w in dataset.windows], device=self.device)]
    )
    self.window_gap = torch.tensor([w.gap for w in dataset.windows], device=self.device)
    self.num_windows = len(dataset.windows)
    print(f"[bridge] {self.num_windows} windows in the '{cfg.split}' split")

    # Per environment: the reference from the hand-off onward, already rebased, plus how
    # long this hole is and how far into it we are.
    span = self.max_gap + self.future
    self.reference = torch.zeros(
      self.num_envs, span, self.flat.shape[1], device=self.device
    )
    self.gap = torch.ones(self.num_envs, dtype=torch.long, device=self.device)

    # Which frame of the reference the robot is being scored against right now, and
    # nothing else. The whole episode is laid out on this one index:
    #
    #     0 .. gap-1     the hole, which the bridge has to invent
    #     gap            the hand-off: the state whatever runs next inherits
    #     gap+1 .. +res  the resume, the second window played out under the bridge
    #
    # It is stepped once at the end of every environment step, and once at the end of the
    # reset, which is why a freshly placed window starts at -1: the robot is standing on
    # the hand-off frame having taken no action, and the first action it takes is the one
    # that has to produce frame 0. Starting at 0 instead, which is what this did before,
    # scores every frame against the one after it and asks the bridge to be permanently
    # twenty milliseconds into the future.
    self.step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    # Latched at the hand-off and held for the rest of the episode: how usable the state
    # the bridge produced actually was, in (0, 1]. Zero until then, which nothing reads --
    # `resume_credit` is the only consumer and it is zero before the hand-off anyway -- but
    # which keeps an episode that never got there from averaging into the log as though it
    # had arrived perfectly.
    self.hand_off_score = torch.zeros(self.num_envs, device=self.device)
    self.hand_off_error = torch.zeros(self.num_envs, device=self.device)
    self.reached_hand_off = torch.zeros(self.num_envs, device=self.device)

    # Whether this environment's two halves belong together, and so whether the reference
    # inside the hole is a recording of anything. False for a spliced window: the frames
    # are still there, but they led to a continuation that has been replaced, and nothing
    # is scored against them.
    self.scored_hole = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    # How far the robot was from the arrival frame when the hole opened. Only a spliced
    # window uses it, as the bound in `lost_tracking`: with no reference to lose, the one
    # thing still worth ending an episode over is a robot going the other way.
    self.start_distance = torch.zeros(self.num_envs, device=self.device)

    # Which frames of the future context are shown as the target. A handful spread across
    # it rather than all of them: consecutive mocap frames are nearly identical and the
    # observation would be mostly repetition.
    self.target_index = torch.linspace(
      0, self.future - 1, cfg.target_samples, device=self.device
    ).long()

    # Which window each environment drew, and an override for a viewer that wants to
    # study one hole rather than take what it is given. Training leaves this alone.
    self.chosen = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.force_window: int | None = None

    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_root_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["progress"] = torch.zeros(self.num_envs, device=self.device)
    # The three numbers that say whether this is working, and they are read together.
    # `reached_hand_off` is the share of episodes that got as far as handing over at all;
    # the other two are only about those. `hand_off_error` is the root distance at the
    # instant the next skill would take over, in metres, and `hand_off_score` is the same
    # arrival read across every channel and squashed into (0, 1].
    self.metrics["reached_hand_off"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["hand_off_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["hand_off_score"] = torch.zeros(self.num_envs, device=self.device)
    # The share of episodes whose two halves were never adjacent, which is the curriculum
    # made visible. The three numbers above are a blend of both kinds of window while this
    # is between zero and one, so read them against it rather than on their own: an arrival
    # score that holds steady while this climbs is the change working.
    self.metrics["spliced"] = torch.zeros(self.num_envs, device=self.device)

  def context_before(self, env_id: int) -> torch.Tensor:
    """The frames leading up to the hand-off, rebased like the reference. (past, 13 + 2J).

    Not needed by training, which starts at the hand-off. An evaluator needs it to show
    where the robot came from, since a bridge is only meaningful against the motion that
    fed into it.
    """
    base = int(self.window_base[self.chosen[env_id]])
    context = self.flat[base : base + self.past].clone()
    hand_off = self.flat[base + self.past - 1]
    origin = self._env.scene.env_origins[env_id]
    context[:, 0:2] += (origin[:2] - hand_off[0:2]).unsqueeze(0)
    return context

  @property
  def command(self) -> torch.Tensor:
    """Where the robot has to get to, and how much of the hole is left.

    The target poses are relative to the robot's current base, so this stays a live
    instruction rather than a fact about the past: as the robot moves, what remains to be
    covered shrinks, and that is what the policy has to act on.
    """
    return torch.cat([self.target_in_base(), self.phase()], dim=-1)

  def phase(self) -> torch.Tensor:
    """Where in the episode this is: through the hole, and through the resume.

    (num_envs, 3): progress across the hole, how long the hole is, and progress through
    the second window afterwards.

    The third channel is what tells the policy the hole is behind it. Without it, the two
    stretches are indistinguishable from the inside once the first saturates -- the robot
    would be a third of the way through the motion it is supposed to be resuming and
    reading exactly the observation it read at the hand-off.
    """
    progress = (self.step.clamp(min=0).float() / self.gap.float()).clamp(max=1.0)
    resumed = ((self.step - self.gap).clamp(min=0).float() / self.resume).clamp(max=1.0)
    return torch.stack([progress, self.gap.float() / self.max_gap, resumed], dim=-1)

  def resuming(self) -> torch.Tensor:
    """Whether the hand-off is behind us and the second window is being played out.

    Strictly past it: the hand-off frame itself is the arrival being measured, not part of
    what the arrival earns.
    """
    return self.step > self.gap

  def target_in_base(self) -> torch.Tensor:
    """The future context, seen from where the robot is now. (num_envs, samples * pose)."""
    index = (self.gap.unsqueeze(1) + self.target_index.unsqueeze(0)).clamp(
      max=self.reference.shape[1] - 1
    )
    target = torch.gather(
      self.reference,
      1,
      index.unsqueeze(-1).expand(-1, -1, self.reference.shape[2]),
    )
    base_pos = self.robot.data.root_link_pos_w
    base_yaw = yaw_quat(self.robot.data.root_link_quat_w)
    return frames.encode(target, base_pos, base_yaw).flatten(start_dim=1)

  def reference_now(self) -> torch.Tensor:
    """The reference state at this step, for the reward. (num_envs, 13 + 2J).

    Clamped at both ends: below, because a freshly placed window sits at -1 until the
    first step (see `self.step`); above, because nothing should read off the end of the
    reference even though the layout says it cannot happen.
    """
    index = self.step.clamp(min=0, max=self.reference.shape[1] - 1)
    return self.reference[torch.arange(self.num_envs, device=self.device), index]

  def arrival_target(self) -> torch.Tensor:
    """The frame the next motion starts from. (num_envs, 13 + 2J).

    Fixed for the whole episode, unlike `reference_now`, which is why a reward that has no
    reference to follow can still be aimed at something.
    """
    return self.reference[torch.arange(self.num_envs, device=self.device), self.gap]

  def tracked(self) -> torch.Tensor:
    """Whether the reference at this step is a recording of anything, per environment.

    True throughout a coherent window. True from the hand-off onward for a spliced one:
    however the two halves came to be paired, the second is real motion and the robot is
    supposed to be on it. Only the hole of a spliced window is unanswerable, and that is
    the whole of what this excludes.
    """
    return self.scored_hole | (self.step >= self.gap)

  def arrival_weight(self) -> torch.Tensor:
    """A ramp from 1 at the hand-off to `cfg.arrival_weight` at the far end.

    Filling a hole plausibly and arriving somewhere else is the failure this architecture
    exists to prevent, so the two must not score the same. The ramp is what says so. It
    holds at the top through the resume rather than falling back, because the frames after
    the hand-off are the ones being got wrong when a composition breaks.
    """
    progress = (self.step.clamp(min=0).float() / self.gap.float()).clamp(max=1.0)
    return 1.0 + (self.cfg.arrival_weight - 1.0) * progress

  def tracking_weight(self) -> torch.Tensor:
    """The ramp, zeroed wherever there is no reference to track."""
    return self.arrival_weight() * self.tracked().float()

  def approach_weight(self) -> torch.Tensor:
    """The ramp, zeroed wherever there is one.

    The complement of `tracking_weight`, so between them every step of every episode is
    paid by exactly one of the two, and the policy is never left with a stretch that is
    worth nothing to cross.
    """
    return self.arrival_weight() * (~self.tracked()).float()

  def arrival_error(self) -> tuple[torch.Tensor, torch.Tensor]:
    """How wrong the robot is against the frame the next motion starts from."""
    robot = self.robot.data
    actual = torch.cat(
      [
        robot.root_link_pos_w,
        robot.root_link_quat_w,
        robot.root_link_lin_vel_w,
        robot.root_link_ang_vel_w,
        robot.joint_pos,
        robot.joint_vel,
      ],
      dim=-1,
    )
    return arrival_error(actual, self.reference_now(), self.num_joints, self.cfg)

  def resume_credit(self) -> torch.Tensor:
    """What a step of the second window is worth: nothing before it, the score during it.

    Zero inside the hole, so this pays for the resume and only the resume, and
    `hand_off_score` afterwards, so what the second window is worth is decided by the
    state it was handed. This is the term that stops a bad arrival from being a small
    deduction and makes it what it is at inference: the next skill starting from a state
    its motion does not begin in.
    """
    return torch.where(
      self.resuming(), self.hand_off_score, torch.zeros_like(self.hand_off_score)
    )

  def schedule(self) -> tuple[float, float]:
    """How mismatched the pairs are right now: what share are spliced, and by how much.

    Both ramp linearly over `SpliceCfg.warmup_steps` environment steps and then hold, so a
    run opens on the task as it stood before any of this existed and walks from there.
    Zero warmup starts at the configured values, which is what a run resuming from a
    checkpoint that already crosses coherent holes wants.
    """
    splice = self.cfg.splice
    if splice.warmup_steps <= 0:
      return splice.fraction, splice.difficulty
    alpha = min(self._env.common_step_counter / splice.warmup_steps, 1.0)
    return splice.fraction * alpha, splice.difficulty * alpha

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Draw a window per environment and put the robot on its hand-off."""
    if env_ids.numel() == 0:
      return
    count = env_ids.numel()
    if self.force_window is None:
      choice = torch.randint(0, self.num_windows, (count,), device=self.device)
    else:
      # An evaluator wants to look at one particular hole rather than whatever came up.
      choice = torch.full(
        (count,), self.force_window % self.num_windows, device=self.device
      )
    base = self.window_base[choice]
    self.chosen[env_ids] = choice

    # The window from the hand-off onward, padded to the widest hole this run allows.
    span = self.reference.shape[1]
    index = base.unsqueeze(1) + self.past + torch.arange(span, device=self.device)
    index = index.clamp(max=self.flat.shape[0] - 1)
    hand_off = self.flat[base + self.past - 1].clone()
    reference = self.flat[index].clone()
    gap = self.window_gap[choice]

    fraction, difficulty = self.schedule()
    spliced = torch.rand(count, device=self.device) < fraction
    if bool(spliced.any()):
      reference, gap = self._splice(
        choice, hand_off, reference, gap, spliced, difficulty
      )

    self.place_window(
      env_ids,
      hand_off=hand_off,
      reference=reference,
      gap=gap,
      noise=self.cfg.start_noise,
      scored_hole=~spliced,
    )

  def _splice(
    self,
    choice: torch.Tensor,
    hand_off: torch.Tensor,
    reference: torch.Tensor,
    gap: torch.Tensor,
    spliced: torch.Tensor,
    difficulty: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the second half of the marked windows with one that never followed the
    first. Returns the reference and the gaps with those rows substituted.

    Every row is computed and the marked ones are selected at the end, because branching
    on a mask costs more here than the arithmetic it saves.
    """
    count = choice.numel()
    splice = self.cfg.splice
    device = self.device

    # How long the bridge gets, drawn fresh rather than taken from the window. The
    # recorded gap is the length of a continuation that is about to be thrown away and
    # has nothing to say about the pair being built.
    low, high = self.gap_range
    gap = torch.where(
      spliced, torch.randint(low, high + 1, (count,), device=device), gap
    )

    # What the robot has to resume. Either another window's continuation, which has
    # nothing to do with what this body was just doing, or this window's own, which leaves
    # the placement below as the only thing that has changed. Both are useful and they are
    # different questions, so the mix is a dial.
    elsewhere = torch.rand(count, device=device) < splice.cross_clip
    source = torch.where(
      elsewhere, torch.randint(0, self.num_windows, (count,), device=device), choice
    )
    start = self.window_base[source] + self.past + self.window_gap[source]
    index = start.unsqueeze(1) + torch.arange(self.future, device=device)
    second = self.flat[index.clamp(max=self.flat.shape[0] - 1)].clone()

    # Where it begins. The hand-off is carrying momentum, so the place the next motion
    # would naturally start is wherever that momentum was taking the body: carry the
    # velocity forward for the length of the hole and put it there. That is the pair as it
    # would look if nothing had gone wrong and it is what difficulty zero produces, which
    # leaves the content of the second half as the only thing that has changed. Difficulty
    # moves it off that spot, by as much as a walk could cover in the frames available, and
    # turns it by as much as a body could turn in them. Both budgets scale with the gap, so
    # a window is never asked for ground it could not have covered however well it is
    # driven.
    horizon = gap.float() / self.fps
    heading = yaw_quat(hand_off[:, 3:7])
    carry = hand_off[:, 0:2] + hand_off[:, 7:9] * horizon.unsqueeze(-1)
    bearing = torch.rand(count, device=device) * (2.0 * math.pi)
    radius = (
      torch.rand(count, device=device) * splice.travel_speed * horizon * difficulty
    )
    turn = (2.0 * torch.rand(count, device=device) - 1.0) * (
      splice.turn_speed * horizon * difficulty
    )
    offset = torch.stack(
      [radius * bearing.cos(), radius * bearing.sin(), torch.zeros_like(radius)], dim=-1
    )
    axis = torch.zeros(count, 3, device=device)
    axis[:, 2] = 1.0
    second = rebase_states(
      second,
      to_pos=torch.cat(
        [carry + quat_apply(heading, offset)[:, 0:2], second[:, 0, 2:3]], dim=-1
      ),
      to_yaw=quat_mul(heading, quat_from_angle_axis(turn, axis)),
    )

    # Laid out the way everything downstream reads a reference: the arrival frame held
    # through the hole, the second window from the hand-off on, its last frame repeated to
    # the end of the buffer. Nothing scores the hole of a spliced window, and holding the
    # arrival across it means the things that do still read there -- the metrics, the
    # stray bound in `lost_tracking` -- degrade to distance from where the robot has to end
    # up, which is the only thing left in the hole that means anything.
    span = reference.shape[1]
    clock = torch.arange(span, device=device).unsqueeze(0) - gap.unsqueeze(1)
    frame = clock.clamp(0, self.future - 1).unsqueeze(-1)
    laid = torch.gather(second, 1, frame.expand(-1, -1, second.shape[-1]))
    return torch.where(spliced.view(-1, 1, 1), laid, reference), gap

  def place_window(
    self,
    env_ids: torch.Tensor,
    hand_off: torch.Tensor,
    reference: torch.Tensor,
    gap: torch.Tensor,
    noise: float = 0.0,
    scored_hole: torch.Tensor | None = None,
  ) -> None:
    """Start these environments on a window that did not have to come from the corpus.

    `hand_off` is (N, 13 + 2J), the state control arrives in. `reference` is (N, span,
    13 + 2J) starting at the frame after it, where span is the widest hole this run allows
    plus the future context. `gap` is (N,), how many frames the bridge has.

    Both are rebased here, exactly as a corpus window is: the hand-off's horizontal
    position is slid onto the environment's origin and the reference is slid with it, so
    heights, headings and every velocity survive and the reference and the robot end up in
    one coordinate system. Whoever supplies a window only has to have it internally
    consistent -- where it sits in its own world is this method's problem, not theirs.

    `scored_hole` is (N,) and says, per environment, whether the frames before `gap` are a
    recording of a body that actually went from this hand-off to this arrival. It defaults
    to true, which is what a caller supplying a window cut from one clip means. A caller
    that has pasted two motions together has no such frames and must say so, or the reward
    will score the bridge against an in-between that leads somewhere else.

    `_resample_command` is one caller. The other is an evaluation that wants to score the
    bridge on the hand-over it will actually be asked for, between two policies, which is
    a window no corpus contains (see experiments/parkour/tests/walk2jump.py).
    """
    if env_ids.numel() == 0:
      return
    span = self.reference.shape[1]
    if reference.shape[1] != span or reference.shape[2] != self.flat.shape[1]:
      raise ValueError(
        f"A window for this run is ({span}, {self.flat.shape[1]}) per environment; got "
        f"{tuple(reference.shape[1:])}."
      )

    self.gap[env_ids] = gap
    self.step[env_ids] = -1
    self.hand_off_score[env_ids] = 0.0
    self.hand_off_error[env_ids] = 0.0
    self.reached_hand_off[env_ids] = 0.0
    self.scored_hole[env_ids] = (
      torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
      if scored_hole is None
      else scored_hole
    )

    origin = self._env.scene.env_origins[env_ids]
    shift = origin[:, :2] - hand_off[:, :2]
    reference = reference.clone()
    reference[:, :, 0:2] += shift.unsqueeze(1)
    self.reference[env_ids] = reference

    root_pos = hand_off[:, 0:3].clone()
    root_pos[:, 0:2] = origin[:, :2]
    root_quat = hand_off[:, 3:7]
    root_lin_vel = hand_off[:, 7:10]
    root_ang_vel = hand_off[:, 10:ROOT_STATE_DIM]
    joint_pos = hand_off[:, ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
    joint_vel = hand_off[:, ROOT_STATE_DIM + self.num_joints :]

    # A share of episodes start off the reference. At inference the robot arrives from
    # whatever the outgoing skill left behind, never from a corpus frame, and a policy
    # that has only ever started exactly on the reference has never had to steer.
    if noise > 0.0:
      root_pos = root_pos + torch.randn_like(root_pos) * noise * 0.05
      root_lin_vel = root_lin_vel + torch.randn_like(root_lin_vel) * noise * 0.2
      root_ang_vel = root_ang_vel + torch.randn_like(root_ang_vel) * noise * 0.2
      joint_pos = joint_pos + torch.randn_like(joint_pos) * noise * 0.05
      joint_vel = joint_vel + torch.randn_like(joint_vel) * noise * 0.5

    # How far there is to go, measured from where the robot has actually been put rather
    # than from the frame it was drawn from, so the perturbation above is inside it. This
    # is the only thing a spliced hole can be judged against while it is open.
    arrival = reference[
      torch.arange(env_ids.numel(), device=self.device), gap.clamp(max=span - 1)
    ]
    self.start_distance[env_ids] = (root_pos - arrival[:, 0:3]).norm(dim=-1)

    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[:, :, 0], limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self.robot.write_root_state_to_sim(
      torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1),
      env_ids=env_ids,
    )
    self.robot.reset(env_ids=env_ids)

  def _update_command(self) -> None:
    """Advance the clock, latching the arrival on the way past it.

    This runs at the end of a step, after the reward has been paid for it, so `self.step`
    still names the step that just happened and the robot state is the one that step
    produced. When those coincide with the hand-off, this is the one moment the arrival
    can be read: the state the next skill would be starting from. It is latched rather
    than recomputed because from here on the robot is being driven through the second
    window and is no longer at the hand-off, and what the resume is worth was decided
    there.
    """
    arriving = self.step == self.gap
    if arriving.any():
      distance, error = self.arrival_error()
      score = torch.exp(-error)
      self.hand_off_error = torch.where(arriving, distance, self.hand_off_error)
      self.hand_off_score = torch.where(arriving, score, self.hand_off_score)
      self.reached_hand_off = torch.where(
        arriving, torch.ones_like(self.reached_hand_off), self.reached_hand_off
      )
    self.step += 1

  def _update_metrics(self) -> None:
    reference = self.reference_now()
    joints = reference[:, ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
    self.metrics["error_joint_pos"] = (
      (self.robot.data.joint_pos - joints).abs().mean(dim=-1)
    )
    self.metrics["error_root_pos"] = (
      self.robot.data.root_link_pos_w - reference[:, 0:3]
    ).norm(dim=-1)
    self.metrics["progress"] = (
      self.step.clamp(min=0).float() / self.gap.float()
    ).clamp(max=1.0)
    # Copies, not the buffers themselves: `CommandTerm.reset` zeroes whatever it finds in
    # `metrics` after reading it, and zeroing the latched arrival would be a live episode
    # losing the number the rest of it is scaled by.
    self.metrics["reached_hand_off"] = self.reached_hand_off.clone()
    self.metrics["hand_off_error"] = self.hand_off_error.clone()
    self.metrics["hand_off_score"] = self.hand_off_score.clone()
    self.metrics["spliced"] = (~self.scored_hole).float()


def arrival_error(
  actual: torch.Tensor,
  reference: torch.Tensor,
  num_joints: int,
  cfg: BridgeCommandCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """How wrong one state is against the frame a next motion would start from.

  Both arguments are corpus-layout rows, (N, 13 + 2J). Returns the root distance in
  metres, which is the number worth reading in a log, and a unitless total across every
  channel a skill taking over would care about: where the body is, which way it is facing,
  how fast it is going, and what the joints are doing at what rate. Each channel is
  divided by a tolerance of its own before they are averaged, since a tenth of a radian
  and a tenth of a metre per second are not the same mistake and summing them raw would
  let the loosest channel decide the answer.

  A free function rather than a method because the number has to mean the same thing in
  two places that cannot share a live environment: here, where it gates the resume during
  training, and in an evaluation scoring a hand-over between two policies from recorded
  states. Two implementations of "did it arrive" would drift apart, and the whole argument
  rests on them agreeing.
  """
  joints = reference[:, ROOT_STATE_DIM : ROOT_STATE_DIM + num_joints]
  joint_vel = reference[:, ROOT_STATE_DIM + num_joints :]

  distance = (actual[:, 0:3] - reference[:, 0:3]).norm(dim=-1)
  channels = torch.stack(
    [
      distance.square() / cfg.arrival_pos_std**2,
      quat_error_magnitude(actual[:, 3:7], reference[:, 3:7]).square()
      / cfg.arrival_ori_std**2,
      (actual[:, 7:10] - reference[:, 7:10]).square().sum(dim=-1)
      / cfg.arrival_lin_vel_std**2,
      (actual[:, ROOT_STATE_DIM : ROOT_STATE_DIM + num_joints] - joints)
      .square()
      .mean(dim=-1)
      / cfg.arrival_joint_pos_std**2,
      (actual[:, ROOT_STATE_DIM + num_joints :] - joint_vel).square().mean(dim=-1)
      / cfg.arrival_joint_vel_std**2,
    ],
    dim=-1,
  )
  return distance, channels.mean(dim=-1)


def relative_pose(
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
  pos: torch.Tensor,
  quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """A pose seen from a base frame, using heading only so tilt does not leak in."""
  yaw = yaw_quat(base_quat)
  return quat_apply_inverse(yaw, pos - base_pos), quat_mul(quat_conjugate(yaw), quat)


@dataclass(kw_only=True)
class SpliceCfg:
  """How unlike each other the two halves of a window are allowed to be.

  A window cut from one recording is a pair that goes together by construction, and a
  bridge trained only on those learns to continue a motion rather than to join two. These
  are the dials that take it apart. All of them ramp from nothing over `warmup_steps`, so
  the numbers here are where a run ends up and not where it starts.
  """

  fraction: float = 0.5
  """Share of environments whose second half never followed the first.

  Half rather than all of them, so every batch still carries windows with a recorded
  in-between. Those are the only ones with a dense frame-by-frame answer in the hole, they
  are what teaches the bridge what human motion crossing a gap looks like, and dropping
  them leaves a humanoid learning to invent a transition from a reward that only pays at
  the far end."""

  difficulty: float = 1.0
  """How far off the natural spot a spliced second half is put, as a share of the budget.

  Zero puts it exactly where the body would have been had it carried straight on at the
  speed it was already going, which is a mismatch of content and nothing else. One uses the
  whole of what `travel_speed` and `turn_speed` say is reachable in the frames available.
  Above one the placement stops being reliably feasible, which is a thing to do on purpose
  and not by default."""

  cross_clip: float = 0.7
  """Share of spliced windows whose second half is taken from a different window.

  The remainder keep their own continuation and only move, which separates the two
  questions: this one is 'the next motion is somewhere else', the other is 'the next motion
  is a different thing'. A composition asks both at once and they are worth being able to
  turn down independently when something is not working."""

  travel_speed: float = 1.2
  """Ground a body is assumed to be able to cover, in m/s, when sizing the offset.

  A brisk walk. The bridge is not restricted to it -- it can and does run -- but a budget
  drawn at a sprint would generate pairs that only a sprint could join, and most of the
  corpus is not sprinting."""

  turn_speed: float = 1.5
  """Turn a body is assumed to be able to make, in rad/s, when sizing the heading offset.

  At the longest hole and full difficulty this is most of a half turn, which is past what
  a walk can do while still going somewhere and is deliberately at the hard end."""

  warmup_steps: int = 120_000
  """Environment steps over which `fraction` and `difficulty` ramp from zero.

  Counted in vectorized steps, so a run of 15000 iterations at 24 steps each is 360000 of
  them and this default spends the first third arriving at the configured mix. Set it to
  zero to start at full strength, which is what a run resuming from a checkpoint that
  already crosses coherent holes wants: that checkpoint is the warm start, and the ramp
  exists to produce one when there is none."""


@dataclass(kw_only=True)
class BridgeCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  corpus: CorpusCfg = field(default_factory=CorpusCfg)
  windows: WindowCfg = field(default_factory=WindowCfg)
  splice: SpliceCfg = field(default_factory=SpliceCfg)

  split: str = "train"
  """Which half of the corpus this env draws from. The evaluation env uses 'eval', so a
  run is scored on subjects it was never trained on."""

  target_samples: int = 5
  """Frames of the future context shown to the policy."""

  resume_steps: int = 24
  """Frames of the second window played out past the hand-off, with the policy still
  driving and still scored.

  This is the part of training that asks the question inference asks: not 'did you fill
  the hole' but 'did you leave the robot somewhere the next thing can carry on from'. It
  used to be ten frames, a fifth of a second, which is long enough for a bad arrival to
  still look survivable -- a humanoid that has been handed the wrong momentum takes
  longer than that to fall over. The default is now the whole future context the corpus
  provides, less the hand-off frame itself, so an arrival that does not work has time to
  not work.

  Capped at `windows.future - 1`; past that there is no recorded continuation left to
  score against."""

  arrival_weight: float = 3.0
  """How much more the far end of a hole counts than the near end."""

  ##
  # What counts as arriving.
  #
  # One tolerance per channel, used only to make the channels comparable before they are
  # averaged into `hand_off_score`. Each is roughly the error at which a skill starting
  # from this state would notice: a std of error costs about two thirds of the resume.
  ##

  arrival_pos_std: float = 0.15
  """Root position [m]."""

  arrival_ori_std: float = 0.3
  """Root orientation [rad]."""

  arrival_lin_vel_std: float = 0.5
  """Root linear velocity [m/s]. The channel that matters most and is easiest to get
  wrong: a body in the right pose carrying the wrong momentum is a body that is about to
  be somewhere else."""

  arrival_joint_pos_std: float = 0.25
  """Joint positions [rad], as a root-mean-square over the joints."""

  arrival_joint_vel_std: float = 3.0
  """Joint velocities [rad/s], likewise."""

  start_noise: float = 1.0
  """Scale on the perturbation applied to the hand-off state. Zero starts every episode
  exactly on the reference."""

  def build(self, env: ManagerBasedRlEnv) -> BridgeCommand:
    return BridgeCommand(self, env)
