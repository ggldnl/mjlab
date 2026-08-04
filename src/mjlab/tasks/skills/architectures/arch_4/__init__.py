"""Architecture 4: one bridge for the whole pool, trained the way a masked token is.

Take a clip of a real body moving. Mask a window out of the middle of it, preferably
somewhere a lot happens (the body changes direction, changes pace, leaves the ground).
Train a policy to produce what goes in that window, given only the frames on either side
and never the frames inside it. What it has to learn, to be paid at all, is how one
stretch of motion connects to another: the stitch itself.

The name is an analogy and nothing more. There is no transformer here and nothing is a
token. What is borrowed is the training signal: a hole, its two edges, and a loss on
what was removed.

At inference the two edges are no longer a clip's. The frames before the hole are the
last N the robot actually produced under the skill being left, and the frames after it
are the first M of a rollout of the skill being entered. The bridge is asked the same
question it was trained on and fills the same hole, except that this one has never been
filled by anybody: the robot has to get from a run it is in the middle of to the opening
of a jump, dynamically, in the steps it is given.

Where in the next policy's rollout those M frames are taken from is a real choice and is
left open -- `profile_offset` in the config. The beginning of the skill is the literal
reading; a later window is the skill once it has settled, which for a periodic gait is
the more honest thing to aim at.

##
# How that becomes code
##

One policy, shared by every skill in the pool. There is no skill id anywhere in its
input and no per-target copy of anything: what says where a transition is headed is the
after-context, which is a recording of the next skill actually moving. A pool that grows
by a skill costs a bank of recorded frames, not a training run. Nothing here is
source-aware either -- which skill is being left arrives as the frames it produced.

Everything is expressed in the frame vector of frames.py, which is why any of this can
be trained on motion capture and used on policies: it is a body-frame description of what
the robot is doing, identical in kind whether it was measured off a clip or off the
simulator, and blind to where in the world the motion happened.

Three things are carried per transition, all fixed at the moment of the switch:

- the before-context, taken from a rolling history this class keeps every step, not only
  while bridging;
- the after-context, sampled from the profile bank the trainer recorded;
- how long the gap is, and how far through it we are. The transition ends when the gap
  runs out, so there is no hand-over decision to learn and no success function anywhere
  in this architecture -- the same simplification arch_3 makes, for a different reason.

Fresh from the constructor the policy is untrained, the profile banks are empty and the
unit conversions are the identity; train.py fills all three in place and all three go in
the checkpoint. A bridge restored without its banks has nothing to aim at, and one
restored without its conversions emits numbers that mean something else entirely.

This is also the one architecture that reads the robot directly rather than only through
the experiment's state view, because a frame is joint state and root motion. So it is the
one that has to be told which scene entity is the robot; `build` in
architectures/__init__.py passes the experiment's.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.architectures.arch_4.frames import frame_dim, robot_frame
from mjlab.tasks.skills.architectures.arch_4.networks import input_dim, masked_input
from mjlab.tasks.skills.architectures.common.nets import build_actor, obs_td
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.spaces import ActionSpace, StateSpace, Units
from mjlab.tasks.skills.view import StateView, resolve_view


class Arch4(MetaPolicy):
  """Meta policy holding one masked in-betweening bridge for the whole pool."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    view: StateView | None = None,
    *,
    entity_name: str = "robot",
    hidden_dims: tuple[int, ...] = (512, 256, 128),
    past_steps: int = 16,
    future_steps: int = 16,
    context_stride: int = 2,
    gap_steps: int = 48,
    obs_group: str = "actor",
  ) -> None:
    self.obs_group = obs_group
    self.entity_name = entity_name
    self.context_stride = max(context_stride, 1)
    self.gap_steps = gap_steps
    self.max_gap = gap_steps
    """The longest gap trained on; the policy reads a length relative to this."""

    projection = resolve_view(env, None, obs_group) if view is None else view
    self.obs_dim = projection.dim
    self.action_dim = env.action_manager.total_action_dim

    entity: Entity = env.scene[entity_name]
    self.num_joints = int(entity.data.joint_pos.shape[-1])
    self.frame_dim = frame_dim(self.num_joints)

    # Identity until training measures them, so a freshly constructed architecture is
    # callable (it just does not do anything useful yet). `frame_space` is the same
    # conversion applied to the context windows, measured over the corpus rather than
    # over the pool, which is why it is not part of `Units`.
    self.action_space = ActionSpace(self.action_dim).to(env.device)
    self.state_space = StateSpace(self.obs_dim).to(env.device)
    self.frame_space = StateSpace(self.frame_dim).to(env.device)

    self.past_steps = past_steps
    self.future_steps = future_steps
    self.hidden_dims = tuple(hidden_dims)
    self.actor = self._build(env.device, env.num_envs)
    # One bank of recorded frame windows per skill, filled by train.py.
    self.profiles: dict[int, torch.Tensor] = {}

    super().__init__(env, pool, projection)

  def _build(self, device: str, num_envs: int):
    """The actor, sized for the input `masked_input` produces."""
    width = input_dim(self.obs_dim, self.frame_dim, self.past_steps, self.future_steps)
    template = torch.zeros(num_envs, width, device=device)
    return build_actor(obs_td(template), self.action_dim, self.hidden_dims, device)

  def configure(
    self,
    past_steps: int,
    future_steps: int,
    context_stride: int,
    hidden_dims: tuple[int, ...],
  ) -> None:
    """Resize the policy to the shape a training config asks for. Called by train.py.

    The context lengths decide the width of the input layer, so unlike arch_3's schedule
    they cannot simply be assigned after the fact. An experiment overriding either of
    them on the command line would otherwise train a network built for the constructor's
    defaults, silently. The stride does not change any width, but it changes what the
    numbers in that width mean, so it travels with them.
    """
    shape = (past_steps, future_steps, max(context_stride, 1), tuple(hidden_dims))
    if shape == (
      self.past_steps,
      self.future_steps,
      self.context_stride,
      self.hidden_dims,
    ):
      return
    self.past_steps = past_steps
    self.future_steps = future_steps
    self.context_stride = max(context_stride, 1)
    self.hidden_dims = tuple(hidden_dims)
    self.actor = self._build(self.env.device, self.env.num_envs)
    self.reset()

  def adopt_units(self, units: Units) -> None:
    """Take on the pool-measured unit conversions. Called by train.py.

    Same contract as arch_1 and arch_3. The frame conversion is measured over the corpus
    instead and arrives separately.
    """
    self.action_space = units.action.to(self.env.device)
    self.state_space = units.state.to(self.env.device)

  def adopt_frame_space(self, frame_space: StateSpace) -> None:
    """Take on the frame conversion measured over the corpus. Called by train.py."""
    self.frame_space = frame_space.to(self.env.device)

  def adopt_profiles(self, profiles: dict[int, torch.Tensor]) -> None:
    """Take on the recorded windows each skill is aimed at. Called by train.py."""
    self.profiles = {
      skill_id: rows.to(self.env.device) for skill_id, rows in profiles.items()
    }

  @property
  def entity(self) -> Entity:
    return self.env.scene[self.entity_name]

  ##
  # Per-transition bookkeeping: the rolling history, and what each switch froze from it.
  ##

  @property
  def history_span(self) -> int:
    """How many steps back the before-context reaches, at this stride."""
    return self.context_stride * (self.past_steps - 1) + 1

  def reset(self) -> None:
    super().reset()
    num_envs = self.env.num_envs
    device = self.env.device
    # Every frame of the recent past, oldest first, kept deep enough that every
    # `context_stride`-th of them is the before-context the policy reads. Maintained for
    # every environment at all times, because a switch can fire on any step and its
    # context is whatever was happening just before it.
    self._history = torch.zeros(
      num_envs, self.history_span, self.frame_dim, device=device
    )
    self._history_filled = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self._past = torch.zeros(num_envs, self.past_steps, self.frame_dim, device=device)
    self._future = torch.zeros(
      num_envs, self.future_steps, self.frame_dim, device=device
    )
    self._elapsed = torch.zeros(num_envs, dtype=torch.long, device=device)

  def notify_reset(self, done: torch.Tensor) -> None:
    super().notify_reset(done)
    # The history must not cross an episode boundary: the frames before a reset belong
    # to a different rollout, and a transition early in the next episode would be
    # conditioned on a motion that never led into it.
    self._history_filled = self._history_filled & ~done
    self._elapsed = torch.where(done, torch.zeros_like(self._elapsed), self._elapsed)

  def act(self, obs: VecEnvObs, command: torch.Tensor) -> torch.Tensor:
    """The base class's step, with the rolling history updated after it.

    After rather than before: a switch firing on this step reads the frames of the steps
    that are already finished, which is exactly what the history holds at this point.
    """
    actions = super().act(obs, command)
    self._remember()
    return actions

  @torch.no_grad()
  def _remember(self) -> None:
    frame = robot_frame(self.entity)
    self._history = torch.roll(self._history, shifts=-1, dims=1)
    self._history[:, -1] = frame
    # An environment whose episode just started has no history to roll. Filling every
    # slot with its first frame reads as "nothing has changed yet", which is a motion
    # the corpus contains; a block of zeros is not.
    fresh = ~self._history_filled
    if bool(fresh.any()):
      self._history[fresh] = frame[fresh].unsqueeze(1).expand(-1, self.history_span, -1)
      self._history_filled = self._history_filled | fresh

  def begin_switch(
    self, switching: torch.Tensor, source: torch.Tensor, target: torch.Tensor
  ) -> None:
    # Source-agnostic: the skill being left arrives as the frames it produced.
    del source
    self._elapsed = torch.where(
      switching, torch.zeros_like(self._elapsed), self._elapsed
    )
    # Both edges of the hole are frozen here and held for the whole transition. The
    # before-edge especially: letting it slide would walk it into the gap, and the policy
    # was never shown a before-context made of its own output. Taking every stride-th
    # frame, ending on the newest, is the same window the corpus is read with.
    self._past[switching] = self._history[switching][:, :: self.context_stride]
    for target_id in target[switching].unique().tolist():
      rows = switching & (target == target_id)
      bank = self.profiles.get(int(target_id))
      if bank is None or bank.shape[0] == 0:
        raise ValueError(
          f"No profile recorded for skill '{self.pool[int(target_id)].name}', so there "
          f"is nothing for the bridge to stitch toward. Either this architecture was "
          f"never trained, or that skill could not be rolled when it was (see "
          f"collect_profiles in dataset.py)."
        )
      picked = torch.randint(0, bank.shape[0], (int(rows.sum()),), device=bank.device)
      self._future[rows] = bank[picked]

  @torch.no_grad()
  def bridge_step(
    self,
    obs: VecEnvObs,
    skill_actions: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del source, target  # One policy for the whole pool; the contexts say the rest
    del skill_actions  # The bridge drives on its own; neither skill acts while it does

    # No homogeneous-batch assumption anywhere here, unlike arch_1 and arch_3:
    # environments bridging toward different targets differ only in the context they
    # carry, and that is already per environment.
    full_state = obs[self.obs_group]
    assert isinstance(full_state, torch.Tensor)
    state = self.state_space.standardize(self.view(full_state))
    gap = max(self.gap_steps, 1)
    progress = (self._elapsed.float() / gap).clamp(0.0, 1.0)
    length = torch.full_like(progress, gap / max(self.max_gap, 1))

    normalized_action = self.actor(
      obs_td(
        masked_input(
          state,
          self.frame_space.standardize(self._past),
          self.frame_space.standardize(self._future),
          progress,
          length,
        )
      )
    )
    actions = self.action_space.denormalize(normalized_action)

    self._elapsed = torch.where(active, self._elapsed + 1, self._elapsed)
    # The gap is as long as it was declared to be and nothing shortens it: the transition
    # ends when it runs out, and the target skill takes over on the next step.
    handover = active & (self._elapsed >= gap)
    return actions, handover

  # The weights, the units they were fitted in, the profiles they aim at, and the shapes
  # needed to rebuild the policy before any of it can be read back.
  _CHECKPOINT = "arch_4.pt"

  def save(self, path: Path) -> None:
    torch.save(
      {
        "actor": self.actor.state_dict(),
        "action_space": self.action_space.state_dict(),
        "state_space": self.state_space.state_dict(),
        "frame_space": self.frame_space.state_dict(),
        "profiles": self.profiles,
        "past_steps": self.past_steps,
        "future_steps": self.future_steps,
        "context_stride": self.context_stride,
        "hidden_dims": self.hidden_dims,
        "gap_steps": self.gap_steps,
        "max_gap": self.max_gap,
        "entity_name": self.entity_name,
      },
      path / self._CHECKPOINT,
    )

  def load(self, path: Path) -> None:
    checkpoint = torch.load(
      path / self._CHECKPOINT, map_location=self.env.device, weights_only=False
    )
    # Shapes first: the policy has to be the one that was saved before its weights mean
    # anything, and the context lengths decide its input width.
    self.configure(
      int(checkpoint["past_steps"]),
      int(checkpoint["future_steps"]),
      int(checkpoint["context_stride"]),
      tuple(checkpoint["hidden_dims"]),
    )
    self.actor.load_state_dict(checkpoint["actor"])
    self.action_space.load_state_dict(checkpoint["action_space"])
    self.state_space.load_state_dict(checkpoint["state_space"])
    self.frame_space.load_state_dict(checkpoint["frame_space"])
    self.adopt_profiles(checkpoint["profiles"])
    # The gap is as much a part of a trained bridge as its weights: the policy reads how
    # far through one it is, so replaying it on a different length asks a question it was
    # never trained on.
    self.gap_steps = int(checkpoint["gap_steps"])
    self.max_gap = int(checkpoint["max_gap"])
    self.entity_name = str(checkpoint["entity_name"])
