"""The data arch_1 learns from: what is collected, how it is stored, how it is checked.

Two collections feed the two training phases:

- `collect_target_windows`: the target skill's *initiation* window. The skill is let
    run from its own reset and its first `window_steps` transitions are recorded. This
    is the discriminator's "real" half in phase 1.
- `collect_interrupts`: the places a bridge may be dropped into. Every other skill in
    the pool is rolled for a random number of steps and the resulting state is
    recorded, together with the `window_steps` steps of behavior that led up to it
    (the source skill's *terminal* window). The states are what bridge-training
    episodes are reset to; the terminal windows are there so the data can be looked at
    (see inspect_windows.py) rather than trusted blind.

An interrupt state is more than qpos/qvel. Action and command terms carry per-env
state the simulator does not hold and the observation does not expose: the diffdrive
wheel command is acceleration-limited, so the target it is currently ramping toward
lives in the action term, and `last_action` in the observation is read off the action
manager. Writing back only joint and root state leaves all of that stale, and the
restored env is then *not* the state it was harvested from: the bridge is handed a
robot whose wheels are still committed to the speed the previous rollout asked for.
`ManagerState` snapshots it, and `verify_restore` proves the round trip is exact by
recomputing the observation after a restore and comparing it against the one recorded
at capture time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.skill import Skill, SkillPool


def as_tensor(value: object) -> torch.Tensor:
  """Narrow a `VecEnvObs`/`TensorDict` lookup result to a plain `torch.Tensor`."""
  assert isinstance(value, torch.Tensor)
  return value


##
# Manager state.
##


class ManagerState:
  """Per-env state held by the action and command terms, snapshot and restored.

  Found by inspection rather than declared: every tensor attribute of every action
  term, the action manager itself, and every command term whose leading dimension is
  `num_envs` is per-env state by construction. That covers the acceleration limiter's
  current target, the action history the `last_action` observation reads, and a
  command term's sampled goal, without arch_1 having to know any of them by name.
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
# Buffers.
##


class RingBuffer:
  """Fixed-capacity, overwrite-oldest storage for named tensor fields.

  Fields may have any trailing shape, so one buffer holds flat rows (an observation),
  windows (`(window_steps, obs_dim)`) and scalars side by side.
  """

  def __init__(
    self,
    capacity: int,
    device: str,
    shapes: dict[str, tuple[int, ...]],
    dtypes: dict[str, torch.dtype] | None = None,
  ) -> None:
    self.capacity = capacity
    self.device = device
    self._fields = tuple(shapes)
    dtypes = dtypes or {}
    self._data = {
      name: torch.zeros(
        (capacity, *shape), device=device, dtype=dtypes.get(name, torch.float32)
      )
      for name, shape in shapes.items()
    }
    self._size = 0
    self._next = 0

  def __len__(self) -> int:
    return self._size

  @property
  def full(self) -> bool:
    return self._size >= self.capacity

  def add(self, **values: torch.Tensor) -> None:
    """Append a batch of rows, each tensor shaped (N, *field_shape)."""
    n = next(iter(values.values())).shape[0]
    if n == 0:
      return
    idx = (torch.arange(n, device=self.device) + self._next) % self.capacity
    for name in self._fields:
      self._data[name][idx] = values[name].to(self._data[name].dtype)
    self._next = (self._next + n) % self.capacity
    self._size = min(self._size + n, self.capacity)

  def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
    if self._size == 0:
      raise RuntimeError("Cannot sample from an empty buffer")
    idx = torch.randint(0, self._size, (batch_size,), device=self.device)
    return {name: self._data[name][idx] for name in self._fields}

  def all(self) -> dict[str, torch.Tensor]:
    """Every stored row, in insertion order for a buffer that never wrapped."""
    return {name: self._data[name][: self._size] for name in self._fields}


@dataclass(frozen=True)
class ObsLayout:
  """How the flat observation vector splits into named observation terms.

  Read off the env's observation manager so anything plotting or slicing an
  observation can label the channels instead of showing bare indices.
  """

  term_names: tuple[str, ...]
  term_dims: tuple[int, ...]

  @staticmethod
  def of(env: ManagerBasedRlEnv, group: str = "actor") -> ObsLayout:
    names = tuple(env.observation_manager.active_terms[group])
    dims = tuple(
      int(np.prod(d)) for d in env.observation_manager.group_obs_term_dim[group]
    )
    return ObsLayout(names, dims)

  def slices(self) -> list[tuple[str, slice]]:
    out: list[tuple[str, slice]] = []
    start = 0
    for name, dim in zip(self.term_names, self.term_dims, strict=True):
      out.append((name, slice(start, start + dim)))
      start += dim
    return out


@dataclass
class WindowSet:
  """Fixed-length stretches of one or more skills' behavior, ready to look at.

  `obs` and `action` are shaped (N, window_steps, dim). `valid` marks the steps that
  are real (a window ending before `window_steps` steps had elapsed is left-padded).
  `skill_id` names, per row, the skill that produced the window; `offset` is the step
  the window ends at, which for an interrupt window is how long the source skill had
  been running when the bridge was dropped in.
  """

  obs: torch.Tensor
  action: torch.Tensor
  valid: torch.Tensor
  skill_id: torch.Tensor
  offset: torch.Tensor
  skill_names: tuple[str, ...]
  layout: ObsLayout
  kind: str

  def __len__(self) -> int:
    return int(self.obs.shape[0])

  def name_of(self, index: int) -> str:
    return self.skill_names[int(self.skill_id[index])]

  def to_npz(self, path: Path) -> None:
    np.savez_compressed(
      path,
      obs=self.obs.cpu().numpy(),
      action=self.action.cpu().numpy(),
      valid=self.valid.cpu().numpy(),
      skill_id=self.skill_id.cpu().numpy(),
      offset=self.offset.cpu().numpy(),
      skill_names=np.array(self.skill_names),
      term_names=np.array(self.layout.term_names),
      term_dims=np.array(self.layout.term_dims),
      kind=np.array(self.kind),
    )

  @staticmethod
  def from_npz(path: Path) -> WindowSet:
    data = np.load(path, allow_pickle=False)
    return WindowSet(
      obs=torch.from_numpy(data["obs"]),
      action=torch.from_numpy(data["action"]),
      valid=torch.from_numpy(data["valid"]),
      skill_id=torch.from_numpy(data["skill_id"]),
      offset=torch.from_numpy(data["offset"]),
      skill_names=tuple(str(n) for n in data["skill_names"]),
      layout=ObsLayout(
        tuple(str(n) for n in data["term_names"]),
        tuple(int(d) for d in data["term_dims"]),
      ),
      kind=str(data["kind"]),
    )


@dataclass
class InterruptSet:
  """Harvested bridge-training start states, plus how each one was arrived at.

  `state` holds everything needed to put the env back: joint state, root state (absent
  for a fixed-base entity, which `write_root_state_to_sim` rejects) and the action and
  command terms' per-env state. `obs` is the observation as it read at capture time,
  kept so a restore can be checked rather than assumed. `windows` is the source
  skill's terminal window for each row.
  """

  state: dict[str, torch.Tensor]
  obs: torch.Tensor
  windows: WindowSet
  has_root_state: bool

  def __len__(self) -> int:
    return int(self.obs.shape[0])

  def sample_indices(self, n: int, device: str) -> torch.Tensor:
    return torch.randint(0, len(self), (n,), device=device)


##
# Collection.
##


def collect_target_windows(
  env: ManagerBasedRlEnv,
  skill: Skill,
  skill_id: int,
  skill_names: tuple[str, ...],
  window_steps: int,
  num_windows: int,
  obs_group: str = "actor",
) -> WindowSet:
  """Roll `skill` from its own reset, recording the first `window_steps` transitions.

  Steps taken after an episode boundary inside the window are marked invalid rather
  than kept: the env auto-resets, so a transition straddling the boundary pairs a
  terminal observation with a freshly-spawned one, which is not behavior the skill
  ever produced and is not something a discriminator should be asked to call real.
  """
  device = env.device
  num_envs = env.num_envs
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  layout = ObsLayout.of(env, obs_group)

  obs_chunks: list[torch.Tensor] = []
  action_chunks: list[torch.Tensor] = []
  valid_chunks: list[torch.Tensor] = []
  collected = 0

  while collected < num_windows:
    obs, _ = env.reset()
    skill.reset(active)
    step_obs, step_action, step_valid = [], [], []
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    for _ in range(window_steps):
      cur_obs = as_tensor(obs[obs_group])
      actions = skill.act(obs, active)
      step_obs.append(cur_obs)
      step_action.append(actions)
      step_valid.append(alive.clone())
      obs, _, terminated, time_out, _ = env.step(actions)
      alive = alive & ~(terminated | time_out)

    obs_chunks.append(torch.stack(step_obs, dim=1))
    action_chunks.append(torch.stack(step_action, dim=1))
    valid_chunks.append(torch.stack(step_valid, dim=1))
    collected += num_envs

  obs_all = torch.cat(obs_chunks)[:num_windows]
  action_all = torch.cat(action_chunks)[:num_windows]
  valid_all = torch.cat(valid_chunks)[:num_windows]
  n = obs_all.shape[0]
  return WindowSet(
    obs=obs_all,
    action=action_all,
    valid=valid_all,
    skill_id=torch.full((n,), skill_id, dtype=torch.long, device=device),
    offset=torch.zeros(n, dtype=torch.long, device=device),
    skill_names=skill_names,
    layout=layout,
    kind="target initiation",
  )


def window_transitions(windows: WindowSet) -> dict[str, torch.Tensor]:
  """Flatten a `WindowSet` into (obs, action, next_obs, done) transitions.

  Only consecutive pairs whose *both* steps are valid survive, so nothing that spans
  an episode boundary reaches the discriminator. `done` is always 0 for what is left,
  which is the point: these are all in-episode transitions.
  """
  obs = windows.obs[:, :-1]
  next_obs = windows.obs[:, 1:]
  action = windows.action[:, :-1]
  keep = windows.valid[:, :-1] & windows.valid[:, 1:]
  flat = keep.reshape(-1)
  return {
    "obs": obs.reshape(-1, obs.shape[-1])[flat],
    "action": action.reshape(-1, action.shape[-1])[flat],
    "next_obs": next_obs.reshape(-1, next_obs.shape[-1])[flat],
    "done": torch.zeros(int(flat.sum()), device=windows.obs.device),
  }


def collect_interrupts(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  exclude: int,
  max_steps: int,
  num_rollouts: int,
  capacity: int,
  window_steps: int,
  obs_group: str = "actor",
) -> InterruptSet:
  """Harvest candidate bridge-training start states from every skill but `exclude`.

  Each rollout runs one source skill from a fresh reset and captures every env at its
  own random step in [0, max_steps], so one rollout already spans the whole range of
  "how long had the source skill been running". Alongside the restorable state, the
  `window_steps` steps of behavior leading up to the capture are kept.
  """
  device = env.device
  num_envs = env.num_envs
  entity: Entity = env.scene[entity_name]
  num_joints = entity.data.joint_pos.shape[-1]
  has_root_state = not entity.is_fixed_base
  others = [i for i in range(len(pool)) if i != exclude]
  active = torch.ones(num_envs, dtype=torch.bool, device=device)

  manager_state = ManagerState(env)
  layout = ObsLayout.of(env, obs_group)
  obs_dim = sum(layout.term_dims)
  action_dim = env.action_manager.total_action_dim

  shapes: dict[str, tuple[int, ...]] = {
    "joint_pos": (num_joints,),
    "joint_vel": (num_joints,),
  }
  if has_root_state:
    # Root pose (3 position + 4 quaternion) and root velocity (3 linear + 3 angular).
    shapes["root_state"] = (13,)
  shapes.update(manager_state.shapes())
  state_dtypes = manager_state.dtypes()

  shapes["obs"] = (obs_dim,)
  shapes["window_obs"] = (window_steps, obs_dim)
  shapes["window_action"] = (window_steps, action_dim)
  shapes["window_valid"] = (window_steps,)
  shapes["source_id"] = ()
  shapes["offset"] = ()
  state_dtypes["window_valid"] = torch.bool
  state_dtypes["source_id"] = torch.long
  state_dtypes["offset"] = torch.long

  buffer = RingBuffer(capacity, device, shapes, state_dtypes)

  for _ in range(num_rollouts):
    if buffer.full:
      break
    for skill_id in others:
      skill = pool[skill_id]
      obs, _ = env.reset()
      skill.reset(active)

      # A random capture step per env; step until every env has been captured.
      target_steps = torch.randint(0, max_steps + 1, (num_envs,), device=device)
      captured = torch.zeros(num_envs, dtype=torch.bool, device=device)
      history: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=window_steps)

      for t in range(max_steps + 1):
        cur_obs = as_tensor(obs[obs_group])
        hit = (target_steps == t) & ~captured
        if hit.any():
          window_obs, window_action, window_valid = _stack_history(
            history, hit, window_steps, obs_dim, action_dim, device
          )
          values = {
            "joint_pos": entity.data.joint_pos[hit],
            "joint_vel": entity.data.joint_vel[hit],
            "obs": cur_obs[hit],
            "window_obs": window_obs,
            "window_action": window_action,
            "window_valid": window_valid,
            "source_id": torch.full((int(hit.sum()),), skill_id, device=device),
            "offset": target_steps[hit],
          }
          if has_root_state:
            values["root_state"] = torch.cat(
              [entity.data.root_link_pose_w, entity.data.root_link_vel_w], dim=-1
            )[hit]
          values.update(manager_state.read(hit))
          buffer.add(**values)
          captured |= hit

        if t == max_steps or bool(captured.all()):
          break
        actions = skill.act(obs, active)
        history.append((cur_obs, actions))
        obs, _, _, _, _ = env.step(actions)

  rows = buffer.all()
  state_fields = ["joint_pos", "joint_vel", *manager_state.fields]
  if has_root_state:
    state_fields.append("root_state")
  windows = WindowSet(
    obs=rows["window_obs"],
    action=rows["window_action"],
    valid=rows["window_valid"],
    skill_id=rows["source_id"],
    offset=rows["offset"],
    skill_names=tuple(s.name for s in pool.skills),
    layout=layout,
    kind="source terminal",
  )
  return InterruptSet(
    state={name: rows[name] for name in state_fields},
    obs=rows["obs"],
    windows=windows,
    has_root_state=has_root_state,
  )


def _stack_history(
  history: deque[tuple[torch.Tensor, torch.Tensor]],
  hit: torch.Tensor,
  window_steps: int,
  obs_dim: int,
  action_dim: int,
  device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """The last `window_steps` (obs, action) pairs for the envs `hit` selects.

  Left-padded with zeros (and marked invalid) when the capture step came before
  `window_steps` had elapsed, so every row has the same shape.
  """
  n = int(hit.sum())
  have = len(history)
  window_obs = torch.zeros(n, window_steps, obs_dim, device=device)
  window_action = torch.zeros(n, window_steps, action_dim, device=device)
  window_valid = torch.zeros(n, window_steps, dtype=torch.bool, device=device)
  for i, (obs, action) in enumerate(history):
    slot = window_steps - have + i
    window_obs[:, slot] = obs[hit]
    window_action[:, slot] = action[hit]
    window_valid[:, slot] = True
  return window_obs, window_action, window_valid


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
  command terms alongside the physics so the env really is the state that was
  harvested (see `verify_restore`).
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

  Returns (max absolute error, fraction of rows within `tolerance`). A large error
  means the harvested state does not fully determine the env: something per-env is not
  being carried across, and everything trained from these states is being trained on a
  different distribution than the one that was inspected.
  """
  indices = interrupts.sample_indices(env.num_envs, env.device)
  obs = restore_interrupts(env, entity, interrupts, manager_state, indices)
  restored = as_tensor(obs[obs_group])
  expected = interrupts.obs[indices]
  error = (restored - expected).abs()
  within = (error.amax(dim=-1) <= tolerance).float().mean()
  return float(error.max()), float(within)
