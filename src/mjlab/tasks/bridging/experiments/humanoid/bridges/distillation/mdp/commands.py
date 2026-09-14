"""The bridge window, with the recorded crossing exposed as a maskable constraint set.

BridgeCommand already draws a window, teleports onto its start, holds its target and scores
the arrival. It keeps the recorded crossing for the guidance reward and shows it to nobody.
This subclass shows it, twice, at two levels of detail:

    full      every interior keyframe, every channel. What the teacher reads
    masked    a random subset of the same keyframes. What the student reads

The interior only. The target is not in here: it arrives through the base command, which
carries the gap to it, the clock and the tolerance profile, and is never masked. So with
every bit off the student's observation is the imitation bridge's observation followed by a
block of zeros, which is the same question asked of the same corpus.

A keyframe is a fixed fraction of the window, `keyframes` of them evenly spaced strictly
inside it. Each carries the gap from the robot now to the recorded state at that tick, the
seconds until it, and two bits saying what of it is real:

    core      root position, orientation, both velocities, and the leg and waist joints
    arms      the shoulder, elbow and wrist joints

Masked channels are zeroed, and the bits are what tells the two cases apart. Splitting the
mask by group rather than by joint is MaskedMimic's sparse-joint conditioning cut where this
project already cuts it: CHANNELS splits root and legs from arms, SUPPORT says why.

Run

    uv run train Mjlab-G1-Distillation-Bridge --env.scene.num-envs 4096
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp.commands import (
  BridgeCommand,
  BridgeCommandCfg,
  _rot6d,
  reyaw,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

SLOT_EXTRAS = 3
"""Seconds to the keyframe, then the core bit and the arm bit."""


class MaskedBridgeCommand(BridgeCommand):
  """A bridge window whose interior is readable, in full or through a random mask.

  Everything about the window itself is the base class: the corpus, the draw, the teleport,
  the tolerance curriculum, the arrival score and every metric. Two architectures scored on
  the same numbers is the point of keeping the corpus one level up, and it only holds if
  they are scored by the same code.

  What is added is the constraint set and the mask over it, drawn once per window and held
  until the next one, the way the tolerance profile is.
  """

  cfg: MaskedBridgeCommandCfg

  def __init__(self, cfg: MaskedBridgeCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    if cfg.keyframes < 1:
      raise ValueError("keyframes must be at least 1")
    for name in ("bridge_prob", "slot_prob", "arm_prob"):
      value = getattr(cfg, name)
      if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} is a probability and must be in [0, 1]")

    self.keyframes = cfg.keyframes

    self.fractions = torch.tensor(
      [(k + 1) / (cfg.keyframes + 1) for k in range(cfg.keyframes)],
      device=self.device,
    )
    """(K,) where each keyframe sits in the window, strictly inside it.

    Fixed and evenly spaced rather than sampled, because the observation is a fixed width
    vector and a slot has to mean the same thing every step. Which slots are readable is
    what varies, and that is the mask.
    """

    self.gap_dim = 15 + 2 * self.num_joints
    """Width of one state gap: root position, a 6D rotation, both root velocities, and the
    two joint blocks. Fifteen and not thirteen because the rotation takes six numbers."""

    arms = self.arms
    legs = ~arms
    root = torch.ones(15, dtype=torch.bool, device=self.device)
    self._core_elements = torch.cat([root, legs, legs])
    self._arm_elements = torch.cat([~root, arms, arms])
    """(gap_dim,) which entries of a gap each mask bit covers. By element rather than by
    slice, because nothing orders the G1's joints with the arms contiguous."""

    self.slot_mask = torch.zeros(self.num_envs, cfg.keyframes, 2, device=self.device)
    """(N, K, 2) the core bit and the arm bit of every keyframe of the live window.

    Zeros is the deployment pattern: nothing of the interior is readable and the target is
    all there is. _open leaves it there, so a window aimed from outside through open_window
    gets that pattern without asking, and _resample_command draws over it.
    """

    self.metrics["visible_slots"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["bridge_pattern"] = torch.zeros(self.num_envs, device=self.device)

  ##
  # Reading the crossing.
  ##

  def reference_at(self, tick: torch.Tensor) -> torch.Tensor:
    """The recorded state at these control ticks. (N,) -> (N, 13 + 2J).

    BridgeCommand.reference_now is this at the current step. Same row table, same yaw and
    translation, read at a tick the caller chooses instead of at the one the clock is on.

    Undefined where has_reference is zero. Callers multiply by it rather than branching,
    which is what the guidance reward already does.
    """
    rows = self._ref_rows.gather(1, tick.clamp(min=0, max=self._span).unsqueeze(-1))
    state = (
      self.dataset.states[rows.squeeze(-1)] if self.dataset is not None else self.target
    )
    moved = reyaw(state, self._ref_rotation)
    moved[:, 0:3] = self._ref_to + quat_apply(
      self._ref_rotation, state[:, 0:3] - self._ref_from
    )
    return moved

  def gap_to(self, state: torch.Tensor) -> torch.Tensor:
    """How far the robot is from a state, in its own frame. (N, 13 + 2J) -> (N, gap_dim).

    The same encoding the base command uses for the target, applied to any state: position
    in the heading frame, orientation as a 6D rotation, both velocities in the body frame,
    joints as plain differences. One encoding for the target and for every keyframe, so the
    policy reads the interior of a window in the units it already reads its end in.
    """
    data = self.robot.data
    yaw = yaw_quat(data.root_link_quat_w)
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    qd = slice(ROOT_STATE_DIM + self.num_joints, ROOT_STATE_DIM + 2 * self.num_joints)
    return torch.cat(
      [
        quat_apply_inverse(yaw, state[:, 0:3] - data.root_link_pos_w),
        _rot6d(quat_mul(quat_conjugate(data.root_link_quat_w), state[:, 3:7])),
        quat_apply_inverse(
          data.root_link_quat_w, state[:, 7:10] - data.root_link_lin_vel_w
        ),
        quat_apply_inverse(
          data.root_link_quat_w,
          state[:, 10:ROOT_STATE_DIM] - data.root_link_ang_vel_w,
        ),
        state[:, q] - data.joint_pos,
        state[:, qd] - data.joint_vel,
      ],
      dim=-1,
    )

  @property
  def slot_ticks(self) -> torch.Tensor:
    """(N, K) which control tick of the window each keyframe sits on.

    Fractions of window_steps, which is the crossing the window asked for, not of patience.
    Patience is slack around the question and a keyframe is part of the question.
    """
    ticks = self.fractions.unsqueeze(0) * self.window_steps.unsqueeze(-1).float()
    return ticks.round().long().clamp(min=0, max=max(self._span, 0))

  @property
  def visible(self) -> torch.Tensor:
    """(N, K, 2) the live mask with windows that have no crossing forced off.

    A window aimed from outside has no recorded interior, so there is nothing for a bit to
    make readable. Enforced here rather than at the draw, so it holds for the teacher too:
    in the transition arena the teacher is as blind to the interior as the student, which
    is the truth about that arena and not something to paper over.
    """
    return self.slot_mask * self.has_reference.view(-1, 1, 1)

  def constraints(self, masked: bool) -> torch.Tensor:
    """The interior of the window. (N, K * (gap_dim + 3)).

    Slot k holds the gap to the recorded state at its tick, the seconds until that tick,
    and its two bits. With masked set, the channels a bit switches off are zeroed and the
    seconds go with them, so a slot the policy cannot read is a slot of zeros carrying two
    zeros that say so. Without it every bit reads one and nothing is zeroed, which is the
    fully conditioned view the teacher acts on.

    Seconds go negative once a keyframe is behind the robot, like the clock does, because a
    crossing that is late is a thing worth being able to see.
    """
    # The teacher's view is the mask with every bit on, which is still gated on there
    # being a crossing to read: has_reference is a fact about the window, not a mask
    everything = torch.ones_like(self.slot_mask) * self.has_reference.view(-1, 1, 1)
    bits = self.visible if masked else everything
    ticks = self.slot_ticks
    spent = self.step.float().unsqueeze(-1)
    seconds = (ticks.float() - spent) / self.fps

    slots: list[torch.Tensor] = []
    for k in range(self.keyframes):
      gap = self.gap_to(self.reference_at(ticks[:, k]))
      core, arm = bits[:, k, 0:1], bits[:, k, 1:2]
      keep = core * self._core_elements + arm * self._arm_elements
      slots.append(
        torch.cat(
          [
            gap * keep,
            seconds[:, k : k + 1] * (core + arm).clamp(max=1.0),
            core,
            arm,
          ],
          dim=-1,
        )
      )
    return torch.cat(slots, dim=-1)

  def dense_reference(self) -> torch.Tensor:
    """The recorded state for this very tick, and whether there is one. (N, gap_dim + 1).

    The teacher's tracking signal, and the reason phase one is learnable at all. Following
    a motion that is known frame by frame is a tracking problem, which this project trains
    routinely; inventing one from two endpoints is the problem the student is left with.
    """
    gap = self.gap_to(self.reference_now()) * self.has_reference.unsqueeze(-1)
    return torch.cat([gap, self.has_reference.unsqueeze(-1)], dim=-1)

  ##
  # Drawing the mask.
  ##

  @property
  def guide_scale(self) -> float:
    """One, always. The teacher never lets go of the recorded crossing.

    The base class anneals this to zero because its policy has to end up reference free,
    and it is the same policy throughout. Here the two roles are separate networks: the
    teacher is allowed to be a tracker because it is never deployed, and what has to run
    without a reference is the student, which is not trained by this reward at all.

    A teacher whose behaviour is pinned to the reference is also the better labeller. Left
    free to arrive however it liked, it would settle on one habit of its own, and the
    student would be regressing onto that habit rather than onto the demonstrations.
    """
    return 1.0

  def _sample_mask(self, count: int) -> torch.Tensor:
    """One window's mask. (count, K, 2).

    With probability bridge_prob the whole interior is off, which is the deployment
    pattern: the robot knows where it has to be and nothing about how to get there. It is
    drawn outright rather than left to fall out of K independent coin flips, because it is
    the pattern the policy is actually judged on and it has to be common enough to learn.

    Otherwise each keyframe is readable with probability slot_prob, and its arms are
    readable with probability arm_prob on top of that. Arms under core and not beside it,
    so no slot ever describes where the hands go and nothing about where the body is,
    which is not a constraint anybody would write.
    """
    core = torch.rand(count, self.keyframes, device=self.device) < self.cfg.slot_prob
    arm = core & (
      torch.rand(count, self.keyframes, device=self.device) < self.cfg.arm_prob
    )
    mask = torch.stack([core, arm], dim=-1).float()
    bare = torch.rand(count, device=self.device) < self.cfg.bridge_prob
    return mask * (~bare).view(-1, 1, 1).float()

  def _open(
    self,
    env_ids: torch.Tensor,
    duration_s: torch.Tensor,
    root_pos: torch.Tensor,
    tolerances: torch.Tensor,
  ) -> None:
    """Clear the mask along with every other latch of the window before.

    Cleared to the deployment pattern rather than to nothing in particular, so a window
    aimed from outside through open_window is already the question the bridge exists to
    answer. _resample_command draws over this for a window that came from the corpus.
    """
    super()._open(env_ids, duration_s, root_pos, tolerances)
    self.slot_mask[env_ids] = 0.0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    if env_ids.numel():
      self.slot_mask[env_ids] = self._sample_mask(env_ids.numel())

  def _update_metrics(self) -> None:
    super()._update_metrics()
    readable = self.visible[:, :, 0]
    self.metrics["visible_slots"] = readable.sum(dim=-1)
    self.metrics["bridge_pattern"] = (readable.sum(dim=-1) == 0).float()


@dataclass(kw_only=True)
class MaskedBridgeCommandCfg(BridgeCommandCfg):
  """The bridge window, plus how much of its interior the student gets to see."""

  keyframes: int = 3
  """Constraint slots inside the window, evenly spaced and strictly inside it.

  Three puts them at a quarter, a half and three quarters of the crossing. More slots is a
  denser description and a wider observation: each costs 15 + 2J + 3 numbers, 76 on the
  G1. The number that matters is not this one but how often they are switched off.
  """

  bridge_prob: float = 0.35
  """How often a window switches the whole interior off.

  The deployment pattern, and the only one the trained student is ever asked for. Drawn
  outright rather than left to K independent coin flips: at keyframes 3 and slot_prob 0.5
  it would otherwise come up one window in eight, which is not enough of the batch to
  learn the case the policy exists for.

  Play sets this to one, so watching the task is watching the bridge problem.
  """

  slot_prob: float = 0.5
  """How often each keyframe is readable, in a window that is not the bare one.

  A half spreads the draws over the whole ladder from one visible keyframe to all of them,
  which is the point: the student learns one policy covering every density of description,
  with the bare case at one end and something close to tracking at the other.
  """

  arm_prob: float = 0.5
  """How often a readable keyframe also describes the arms.

  Under the core bit, never beside it. A slot saying where the hands go and nothing about
  where the body is describes no constraint anybody would write, and SUPPORT is the
  standing statement that the arms are the half that can be given up.
  """

  def build(self, env: ManagerBasedRlEnv) -> MaskedBridgeCommand:
    return MaskedBridgeCommand(self, env)
