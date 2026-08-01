"""Windows: stretches of recorded skill behavior around a hand-over.

This is the data layer every bridging architecture shares. It knows how to cut windows
out of a rollout and how to harvest and restore the simulator state at a cut. It does
not know what an architecture then does with them, and in particular it assumes nothing
about hand-over decisions, success tests or switch-deciders: an architecture that has no
such thing still gets its windows here.

A window is a fixed-length stretch of one skill's behavior, anchored at a hand-over.
Three roles are defined, and a skill's spec says how long each of them is for it:

- `Opening`: the first steps of a skill from its own reset. What starting it looks like,
  used when it is the skill being handed `to`.
- `Closing`: the last steps before the hand-over. What leaving it looks like, used when
  it is the skill being handed `from`.
- `Overrun`: the steps after the hand-over with that same skill still driving, i.e. what
  would have happened if control had not been taken away. Zero-length unless an
  architecture asks for it.

Each skill carries its own spec, because "how much of this behavior is worth showing"
is a property of the behavior. A skill that settles in ten steps does not need the same
window as one that takes a second to get going, and how far into a skill a hand-over
can reasonably fall is likewise its own business.

What a window records is not the whole observation but whatever the experiment's
`StateView` keeps of it, because a window is what a bridge is compared against
and the comparison is only meaningful on channels the bridge can actually do
something about. Absolute position, world-frame heading and per-episode goals are the
ones to leave out; an experiment that declares no view records everything, which is the
right default only until it turns out not to be.

TODO we need to make sure this is actually what we want.
Alongside the recordings, `collect_interrupts` harvests the simulator state at each cut
so training episodes can start there. An interrupt state is more than qpos/qvel: action
and command terms carry per-env state the simulator does not hold and the observation
does not expose (an acceleration-limited wheel command is still ramping toward a target
that lives in the action term, and `last_action` is read off the action manager).
`ManagerState` snapshots that too, and `verify_restore` proves the round trip by
recomputing the observation after a restore and comparing it against the one recorded at
capture.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.buffers import RingBuffer
from mjlab.tasks.skills.skill import Skill, SkillPool
from mjlab.tasks.skills.view import FullState, StateView

# The roles a window can play around a hand-over. Strings rather than an enum so an
# architecture can add one of its own without editing this module.
OPENING = "opening"
CLOSING = "closing"
OVERRUN = "overrun"


def as_tensor(value: object) -> torch.Tensor:
  """Narrow a `VecEnvObs`/`TensorDict` lookup result to a plain `torch.Tensor`."""
  assert isinstance(value, torch.Tensor)
  return value


def _view_of(env: ManagerBasedRlEnv, view: StateView | None, group: str) -> StateView:
  """The view to record through, defaulting to the whole observation."""
  if view is not None:
    return view
  return FullState(as_tensor(env.observation_manager.compute()[group]).shape[-1])


##
# Specs.
##


@dataclass(frozen=True)
class SkillInit:
  """The state a skill's own training started it from, applied after an env reset.

  Every skill in a pool is rolled in one shared arena, and that arena is built from one
  task's env cfg -- so `env.reset()` produces *that* task's start state for every skill,
  not each skill's own. On the cart-pole the arena is the swing-up task, whose pole
  hangs, so `balance`'s opening window was being recorded from a hanging pole: the one
  state an upright-only balancer cannot work in. Everything downstream then inherits it,
  since the opening window is exactly what a bridge is trained to reproduce.

  A skill therefore declares where it starts, and it is applied on top of the reset. The
  ranges are absolute joint targets rather than offsets from the entity default, because
  the default belongs to whichever task built the arena and is the thing being corrected.

  This is not the skill's whole start distribution, only the part that differs between
  skills. Whatever the shared reset already randomizes correctly is left alone.
  """

  entity_name: str
  """The scene entity these joints belong to."""

  joint_pos: Mapping[str, tuple[float, float]] = field(default_factory=dict)
  """Per joint name, the inclusive range its position is drawn from [rad or m]."""

  joint_vel: Mapping[str, tuple[float, float]] = field(default_factory=dict)
  """Per joint name, the inclusive range its velocity is drawn from."""

  def apply(self, env: ManagerBasedRlEnv, obs_group: str = "actor") -> VecEnvObs:
    """Write this start state into every env and return the refreshed observation.

    Mirrors the tail of `restore_interrupts`: write, forward, re-sense, recompute, so
    the observation handed back really describes the state that was just written.
    """
    entity: Entity = env.scene[self.entity_name]
    joint_pos = entity.data.joint_pos.clone()
    joint_vel = entity.data.joint_vel.clone()
    for target, ranges in ((joint_pos, self.joint_pos), (joint_vel, self.joint_vel)):
      for name, (low, high) in ranges.items():
        index = entity.find_joints([name], preserve_order=True)[0]
        target[:, index] = torch.empty((env.num_envs, 1), device=env.device).uniform_(
          low, high
        )
    entity.write_joint_state_to_sim(joint_pos, joint_vel)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    env.abstraction_manager.compute(dt=0.0)
    return env.observation_manager.compute(update_history=True)


def start_skill(
  env: ManagerBasedRlEnv, spec: SkillWindowSpec, obs: VecEnvObs
) -> VecEnvObs:
  """Reset-time hook: put the env in this skill's own start state, if it declares one."""
  return obs if spec.init is None else spec.init.apply(env)


@dataclass(frozen=True)
class SkillWindowSpec:
  """How much of one skill to record around a hand-over it takes part in.

  The lengths are independent of when the hand-over happens, deliberately. Deriving a
  window from the cut ("keep whatever came before it") makes the window length a
  side effect of the sampling, which produces windows of every length down to nothing:
  a cut three steps into a skill leaves three frames, and a cut at step zero leaves a
  single state with no history. Those are not recordings of behavior. Here the two are
  separate, and `interrupt_range` is checked against `closing` so a closing window can
  never run off the start of the episode.
  """

  opening: int = 32
  """Steps from this skill's own reset, recorded when it is the skill handed *to*."""

  closing: int = 32
  """Steps ending at the hand-over, recorded when it is the skill handed *from*."""

  overrun: int = 0
  """Steps past the hand-over with this skill still driving, i.e. what it would have
  gone on to do. Only collected if an architecture asks for the role."""

  init: SkillInit | None = None
  """Where this skill starts, when the shared arena's reset does not put it there.
  None means the arena's own reset is already right for this skill."""

  interrupt_range: tuple[int, int] = (32, 96)
  """How long this skill has been running when a hand-over happens, sampled uniformly
  per env between these two (inclusive). Widen it to cover more of the skill's life; the
  low end has to be at least `closing`."""

  def __post_init__(self) -> None:
    low, high = self.interrupt_range
    for name in ("opening", "closing", "overrun"):
      if getattr(self, name) < 0:
        raise ValueError(f"{name} cannot be negative, got {getattr(self, name)}")
    if low > high:
      raise ValueError(f"interrupt_range is empty: {self.interrupt_range}")
    if low < self.closing:
      raise ValueError(
        f"interrupt_range starts at {low}, which is less than the {self.closing}-step "
        f"closing window: a hand-over that early has no closing window to show for it. "
        f"Raise the low end to at least {self.closing}, or shorten the window."
      )

  def length(self, role: str) -> int:
    """How many steps the window for `role` spans."""
    try:
      return {OPENING: self.opening, CLOSING: self.closing, OVERRUN: self.overrun}[role]
    except KeyError:
      raise KeyError(
        f"'{role}' is not a role this spec sizes; known: "
        f"{[OPENING, CLOSING, OVERRUN]}. Add a field for it, or size it yourself."
      ) from None

  @property
  def cut_choices(self) -> int:
    """How many distinct hand-over points `interrupt_range` allows."""
    low, high = self.interrupt_range
    return high - low + 1

  def sample_cuts(self, n: int, device: str) -> torch.Tensor:
    """`n` hand-over points drawn uniformly from `interrupt_range`."""
    low, high = self.interrupt_range
    return torch.randint(low, high + 1, (n,), device=device)


class WindowPlan:
  """One `SkillWindowSpec` per skill, for one experiment.

  Keyed by skill name so an experiment can lay them out next to the pool it builds and
  read off at a glance what each skill is showing.
  """

  def __init__(
    self, specs: Mapping[str, SkillWindowSpec], default: SkillWindowSpec | None = None
  ) -> None:
    self.specs = dict(specs)
    self.default = default

  def __getitem__(self, skill: Skill | str) -> SkillWindowSpec:
    name = skill if isinstance(skill, str) else skill.name
    spec = self.specs.get(name, self.default)
    if spec is None:
      raise KeyError(
        f"No window spec for skill '{name}'; the plan covers "
        f"{sorted(self.specs)}. Add one, or give the plan a default."
      )
    return spec

  def __iter__(self) -> Iterator[str]:
    return iter(self.specs)

  def check(self, pool: SkillPool) -> None:
    """Fail now, with a useful message, rather than partway through a long run."""
    for skill in pool.skills:
      self[skill]


##
# Manager state.
##


class ManagerState:
  """Per-env state held by the action and command terms, snapshot and restored.

  Found by inspection rather than declared: every tensor attribute of every action
  term, the action manager itself, and every command term whose leading dimension is
  `num_envs` is per-env state by construction. That covers an acceleration limiter's
  current target, the action history the `last_action` observation reads, and a
  command term's sampled goal, without an architecture having to know any of them by
  name.
  """

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._owners = self._collect_owners(env)
    self._num_envs = env.num_envs
    self.fields: tuple[str, ...] = tuple(
      f"{owner_name}.{attr}"
      for owner_name, owner in self._owners
      for attr in self._per_env_attrs(owner, env.num_envs)
    )

  @staticmethod
  def _collect_owners(env: ManagerBasedRlEnv) -> list[tuple[str, object]]:
    owners: list[tuple[str, object]] = [("action_manager", env.action_manager)]
    owners += [(f"action.{n}", t) for n, t in env.action_manager._terms.items()]
    owners += [(f"command.{n}", t) for n, t in env.command_manager._terms.items()]
    return owners

  @staticmethod
  def _per_env_attrs(owner: object, num_envs: int) -> list[str]:
    return sorted(
      name
      for name, value in vars(owner).items()
      if isinstance(value, torch.Tensor) and value.shape[:1] == (num_envs,)
    )

  def shapes(self) -> dict[str, tuple[int, ...]]:
    """Per-field trailing shape, for sizing a buffer."""
    out: dict[str, tuple[int, ...]] = {}
    for owner_name, owner in self._owners:
      for attr in self._per_env_attrs(owner, self._num_envs):
        out[f"{owner_name}.{attr}"] = tuple(getattr(owner, attr).shape[1:])
    return out

  def dtypes(self) -> dict[str, torch.dtype]:
    out: dict[str, torch.dtype] = {}
    for owner_name, owner in self._owners:
      for attr in self._per_env_attrs(owner, self._num_envs):
        out[f"{owner_name}.{attr}"] = getattr(owner, attr).dtype
    return out

  def read(self, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """The current state of the envs `mask` selects."""
    out: dict[str, torch.Tensor] = {}
    for owner_name, owner in self._owners:
      for attr in self._per_env_attrs(owner, self._num_envs):
        out[f"{owner_name}.{attr}"] = getattr(owner, attr)[mask].clone()
    return out

  def write(self, values: dict[str, torch.Tensor]) -> None:
    """Overwrite every env's state from a full batch of rows."""
    for owner_name, owner in self._owners:
      for attr in self._per_env_attrs(owner, self._num_envs):
        getattr(owner, attr)[:] = values[f"{owner_name}.{attr}"]


##
# What a collection produces.
##


@dataclass
class Window:
  """One role's worth of recorded behavior, for many rollouts of one skill.

  `obs` and `action` are shaped (N, steps, dim). `valid` marks the steps that are real:
  a closing window is complete by construction, but an opening or overrun window can run
  into an episode boundary, and those steps are marked rather than kept. `cut` is where
  the hand-over fell in each rollout, which for an opening window is not meaningful and
  is left at zero.
  """

  role: str
  skill_id: int
  skill_name: str
  obs: torch.Tensor
  action: torch.Tensor
  valid: torch.Tensor
  cut: torch.Tensor

  def __len__(self) -> int:
    return int(self.obs.shape[0])

  @property
  def steps(self) -> int:
    return int(self.obs.shape[1])

  def transitions(self) -> dict[str, torch.Tensor]:
    """Flatten into (obs, action, next_obs, done) transitions.

    Only consecutive pairs whose *both* steps are valid survive, so nothing that spans
    an episode boundary gets through. `done` is always 0 for what is left, which is the
    point: these are all in-episode transitions.
    """
    obs = self.obs[:, :-1]
    next_obs = self.obs[:, 1:]
    action = self.action[:, :-1]
    keep = (self.valid[:, :-1] & self.valid[:, 1:]).reshape(-1)
    return {
      "obs": obs.reshape(-1, obs.shape[-1])[keep],
      "action": action.reshape(-1, action.shape[-1])[keep],
      "next_obs": next_obs.reshape(-1, next_obs.shape[-1])[keep],
      "done": torch.zeros(int(keep.sum()), device=self.obs.device),
    }


@dataclass
class InterruptSet:
  """Simulator states harvested at hand-over points, ready to restart episodes from.

  `state` holds everything needed to put the env back: joint state, root state (absent
  for a fixed-base entity, which `write_root_state_to_sim` rejects) and the action and
  command terms' per-env state. `obs` is the observation as it read at capture time,
  kept so a restore can be checked rather than assumed.

  `source_id` says which skill was interrupted and `cut_frac` how far through that
  skill's own hand-over range the cut fell, on a 0-to-1 scale. The fraction rather than
  the raw step count, because skills have their own ranges and only the fraction is
  comparable across them.
  """

  state: dict[str, torch.Tensor]
  obs: torch.Tensor
  source_id: torch.Tensor
  cut: torch.Tensor
  cut_frac: torch.Tensor
  has_root_state: bool

  def __len__(self) -> int:
    return int(self.obs.shape[0])

  def sample_indices(self, n: int, device: str) -> torch.Tensor:
    return torch.randint(0, len(self), (n,), device=device)


##
# Collection.
##


def collect_opening(
  env: ManagerBasedRlEnv,
  skill: Skill,
  skill_id: int,
  spec: SkillWindowSpec,
  num_windows: int,
  obs_group: str = "actor",
  view: StateView | None = None,
) -> Window:
  """Roll `skill` from its own reset and record its opening window.

  Steps taken after an episode boundary inside the window are marked invalid rather
  than kept: the env auto-resets, so a transition straddling the boundary pairs a
  terminal observation with a freshly-spawned one, which is not behavior the skill ever
  produced and is not something a discriminator should be asked to call real.
  """
  device = env.device
  num_envs = env.num_envs
  project = _view_of(env, view, obs_group)
  active = torch.ones(num_envs, dtype=torch.bool, device=device)

  obs_chunks, action_chunks, valid_chunks = [], [], []
  collected = 0
  while collected < num_windows:
    obs, _ = env.reset()
    obs = start_skill(env, spec, obs)
    skill.reset(active)
    step_obs, step_action, step_valid = [], [], []
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    for _ in range(spec.opening):
      actions = skill.act(obs, active)
      step_obs.append(project(as_tensor(obs[obs_group])))
      step_action.append(actions)
      step_valid.append(alive.clone())
      obs, _, terminated, time_out, _ = env.step(actions)
      alive = alive & ~(terminated | time_out)

    obs_chunks.append(torch.stack(step_obs, dim=1))
    action_chunks.append(torch.stack(step_action, dim=1))
    valid_chunks.append(torch.stack(step_valid, dim=1))
    collected += num_envs

  keep = num_windows
  return Window(
    role=OPENING,
    skill_id=skill_id,
    skill_name=skill.name,
    obs=torch.cat(obs_chunks)[:keep],
    action=torch.cat(action_chunks)[:keep],
    valid=torch.cat(valid_chunks)[:keep],
    cut=torch.zeros(keep, dtype=torch.long, device=device),
  )


def collect_handover(
  env: ManagerBasedRlEnv,
  skill: Skill,
  skill_id: int,
  spec: SkillWindowSpec,
  num_windows: int,
  roles: tuple[str, ...] = (CLOSING,),
  obs_group: str = "actor",
  view: StateView | None = None,
) -> dict[str, Window]:
  """Roll `skill`, cut it somewhere in its range, and record the roles asked for.

  One pass produces every role that is anchored on the same cut, so `CLOSING` (the
  steps leading up to it) and `OVERRUN` (the steps after it, with this skill still
  driving) come out of the same rollout and line up frame for frame: the overrun window
  begins exactly where the closing window ends. An architecture that wants only one of
  them pays only for that one.
  """
  unknown = set(roles) - {CLOSING, OVERRUN}
  if unknown:
    raise ValueError(
      f"collect_handover records {CLOSING} and {OVERRUN}; got {sorted(unknown)}. "
      f"An opening window comes from collect_opening."
    )
  device = env.device
  num_envs = env.num_envs
  project = _view_of(env, view, obs_group)
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  obs_dim = project.dim
  action_dim = env.action_manager.total_action_dim
  lengths = {role: spec.length(role) for role in roles}
  last_cut = spec.interrupt_range[1]

  chunks: dict[str, dict[str, list[torch.Tensor]]] = {
    role: {"obs": [], "action": [], "valid": [], "cut": []} for role in roles
  }
  collected = 0

  while collected < num_windows:
    obs, _ = env.reset()
    obs = start_skill(env, spec, obs)
    skill.reset(active)
    cut = spec.sample_cuts(num_envs, device)
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    history: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=spec.closing)
    pass_windows = {
      role: _blank_window(num_envs, lengths[role], obs_dim, action_dim, device)
      for role in roles
    }

    for t in range(last_cut + max(lengths.get(OVERRUN, 0), 0) + 1):
      cur_obs = project(as_tensor(obs[obs_group]))
      actions = skill.act(obs, active)

      if CLOSING in roles:
        # An env that already broke has a history spanning a reset, so it is left
        # entirely invalid rather than recorded as behavior the skill produced.
        hit = (cut == t) & alive
        if hit.any():
          _write_history(pass_windows[CLOSING], history, hit)

      if OVERRUN in roles:
        # The overrun window starts at the cut, so slot 0 is the very frame the
        # hand-over would have happened on.
        slot = t - cut
        inside = (slot >= 0) & (slot < lengths[OVERRUN]) & alive
        if inside.any():
          rows = inside.nonzero(as_tuple=True)[0]
          columns = slot[inside]
          pass_windows[OVERRUN]["obs"][rows, columns] = cur_obs[inside]
          pass_windows[OVERRUN]["action"][rows, columns] = actions[inside]
          pass_windows[OVERRUN]["valid"][rows, columns] = True

      history.append((cur_obs, actions))
      obs, _, terminated, time_out, _ = env.step(actions)
      alive = alive & ~(terminated | time_out)

    for role in roles:
      for name in ("obs", "action", "valid"):
        chunks[role][name].append(pass_windows[role][name])
      chunks[role]["cut"].append(cut)
    collected += num_envs

  return {
    role: Window(
      role=role,
      skill_id=skill_id,
      skill_name=skill.name,
      obs=torch.cat(chunks[role]["obs"])[:num_windows],
      action=torch.cat(chunks[role]["action"])[:num_windows],
      valid=torch.cat(chunks[role]["valid"])[:num_windows],
      cut=torch.cat(chunks[role]["cut"])[:num_windows],
    )
    for role in roles
  }


def _blank_window(
  num_envs: int, steps: int, obs_dim: int, action_dim: int, device: str
) -> dict[str, torch.Tensor]:
  return {
    "obs": torch.zeros(num_envs, steps, obs_dim, device=device),
    "action": torch.zeros(num_envs, steps, action_dim, device=device),
    "valid": torch.zeros(num_envs, steps, dtype=torch.bool, device=device),
  }


def _write_history(
  window: dict[str, torch.Tensor],
  history: deque[tuple[torch.Tensor, torch.Tensor]],
  hit: torch.Tensor,
) -> None:
  """Copy the buffered frames into the rows `hit` selects.

  The deque is full whenever this runs, because the earliest cut a `SkillWindowSpec`
  allows is a whole closing window into the episode.
  """
  assert len(history) == window["obs"].shape[1], "closing window is not full"
  for slot, (obs, action) in enumerate(history):
    window["obs"][hit, slot] = obs[hit]
    window["action"][hit, slot] = action[hit]
    window["valid"][hit, slot] = True


def collect_interrupts(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  exclude: int,
  plan: WindowPlan,
  num_interrupts: int,
  obs_group: str = "actor",
) -> InterruptSet:
  """Harvest simulator states at hand-over points, from every skill but `exclude`.

  Each pass runs one source skill from a fresh reset and cuts every env at its own point
  drawn from that skill's range, so one pass already spans the whole range of "how long
  had the source skill been running". States only: whatever windows an architecture
  wants around these cuts come from `collect_handover`, which an architecture calls for
  the roles it actually uses.
  """
  device = env.device
  num_envs = env.num_envs
  entity: Entity = env.scene[entity_name]
  num_joints = entity.data.joint_pos.shape[-1]
  has_root_state = not entity.is_fixed_base
  others = [i for i in range(len(pool)) if i != exclude]
  active = torch.ones(num_envs, dtype=torch.bool, device=device)

  manager_state = ManagerState(env)
  obs_dim = as_tensor(env.observation_manager.compute()[obs_group]).shape[-1]

  shapes: dict[str, tuple[int, ...]] = {
    "joint_pos": (num_joints,),
    "joint_vel": (num_joints,),
    "obs": (obs_dim,),
    "source_id": (),
    "cut": (),
    "cut_frac": (),
  }
  if has_root_state:
    # Root pose (3 position + 4 quaternion) and root velocity (3 linear + 3 angular).
    shapes["root_state"] = (13,)
  shapes.update(manager_state.shapes())
  dtypes = manager_state.dtypes()
  dtypes["source_id"] = torch.long
  dtypes["cut"] = torch.long

  buffer = RingBuffer(num_interrupts, device, shapes, dtypes)

  while not buffer.full:
    before_sweep = len(buffer)
    for skill_id in others:
      skill = pool[skill_id]
      spec = plan[skill]
      obs, _ = env.reset()
      obs = start_skill(env, spec, obs)
      skill.reset(active)

      cut = spec.sample_cuts(num_envs, device)
      captured = torch.zeros(num_envs, dtype=torch.bool, device=device)
      alive = torch.ones(num_envs, dtype=torch.bool, device=device)
      low, high = spec.interrupt_range
      span = max(high - low, 1)

      for t in range(high + 1):
        # An env that already broke has auto-reset, so its state belongs to a fresh
        # episode and is not a place this skill was ever interrupted.
        hit = (cut == t) & ~captured & alive
        if hit.any():
          n = int(hit.sum())
          values = {
            "joint_pos": entity.data.joint_pos[hit],
            "joint_vel": entity.data.joint_vel[hit],
            "obs": as_tensor(obs[obs_group])[hit],
            "source_id": torch.full((n,), skill_id, device=device),
            "cut": cut[hit],
            "cut_frac": (cut[hit] - low).float() / span,
          }
          if has_root_state:
            values["root_state"] = torch.cat(
              [entity.data.root_link_pose_w, entity.data.root_link_vel_w], dim=-1
            )[hit]
          values.update(manager_state.read(hit))
          buffer.add(**values)
          captured |= hit

        if t == high or bool(captured.all()):
          break
        obs, _, terminated, time_out, _ = env.step(skill.act(obs, active))
        alive = alive & ~(terminated | time_out)

    if len(buffer) == before_sweep:
      raise RuntimeError(
        f"A full sweep of {[pool[i].name for i in others]} harvested no interrupt "
        f"states: every env broke before its cut. The cut ranges in the window plan "
        f"reach further into these skills than they survive on their own."
      )

  rows = buffer.all()
  state_fields = ["joint_pos", "joint_vel", *manager_state.fields]
  if has_root_state:
    state_fields.append("root_state")
  return InterruptSet(
    state={name: rows[name] for name in state_fields},
    obs=rows["obs"],
    source_id=rows["source_id"],
    cut=rows["cut"],
    cut_frac=rows["cut_frac"],
    has_root_state=has_root_state,
  )


##
# Restore, and proving it works.
##


def restore_interrupts(
  env: ManagerBasedRlEnv,
  entity: Entity,
  interrupts: InterruptSet,
  manager_state: ManagerState,
  indices: torch.Tensor | None = None,
) -> VecEnvObs:
  """Reset every env to a sampled harvested interrupt state.

  Mirrors the tail of `ManagerBasedRlEnv.reset()` (scene write, forward, command,
  sense, abstraction, obs compute) but substitutes a direct state write for the
  event-manager-driven random reset `_reset_idx` would do, and restores the action and
  command terms alongside the physics so the env really is the state that was harvested
  (see `verify_restore`).
  """
  if indices is None:
    indices = interrupts.sample_indices(env.num_envs, env.device)
  rows = {name: value[indices] for name, value in interrupts.state.items()}

  if interrupts.has_root_state:
    entity.write_root_state_to_sim(rows["root_state"])
  entity.write_joint_state_to_sim(rows["joint_pos"], rows["joint_vel"])
  manager_state.write(rows)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.episode_length_buf[:] = 0
  # compute(dt=0.0) refreshes the command from the restored state without resampling
  # it, which is what we want: the sampled goal was restored above.
  env.command_manager.compute(dt=0.0)
  env.sim.sense()
  env.abstraction_manager.compute(dt=0.0)
  return env.observation_manager.compute(update_history=True)


def verify_restore(
  env: ManagerBasedRlEnv,
  entity: Entity,
  interrupts: InterruptSet,
  manager_state: ManagerState,
  obs_group: str = "actor",
  tolerance: float = 1e-4,
) -> tuple[float, float]:
  """Restore a batch of interrupt states and compare the observation to the recorded one.

  Returns (max absolute error, fraction of rows within `tolerance`). A large error means
  the harvested state does not fully determine the env: something per-env is not being
  carried across, and everything trained from these states is being trained on a
  different distribution than the one that was inspected. Assumes the observation is
  deterministic, which it is in the play-mode envs these experiments train in
  (observation corruption is off).
  """
  indices = interrupts.sample_indices(env.num_envs, env.device)
  obs = restore_interrupts(env, entity, interrupts, manager_state, indices)
  error = (as_tensor(obs[obs_group]) - interrupts.obs[indices]).abs()
  within = (error.amax(dim=-1) <= tolerance).float().mean()
  return float(error.max()), float(within)
