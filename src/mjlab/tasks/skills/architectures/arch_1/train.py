"""Training for arch_1: everything that turns an untrained `Arch1` into a trained one.

For each target skill in the pool, in order:
- collect the target skill's own transition window (the "real" data to imitate);
- collect interrupted states from the *other* skills (where the bridge gets dropped);
- phase 1: train that skill's bridge actor by AIRL + PPO (move like the target);
- phase 2: freeze the actor and train that skill's switch-decider by Double-DQN
    (when to hand over), against an external success signal.

Only training-specific machinery lives here (a ring buffer, the two
collection passes, the interrupt-state reset, and the two loops). The reusable
inference-time networks live in networks.py, and the meta policy that holds them in
__init__.py. rsl_rl's `PPO`, `RolloutStorage`, and `MLPModel` are reused directly
rather than through `OnPolicyRunner`, because the runner owns reset scheduling and
these loops must reset training episodes to harvested interrupt states instead.

Nothing here is experiment specific: `train` takes the env, the pool, the entity to
harvest states from, and one success oracle per target skill. Each experiment owns
its own entry point that supplies those and calls it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.architectures.arch_1 import Arch1
from mjlab.tasks.skills.architectures.arch_1.networks import (
  OBS_GROUPS,
  AIRLDiscriminator,
  DiscBatch,
  DoubleDQN,
  SwitchQNetwork,
)
from mjlab.tasks.skills.skill import Skill, SkillPool

# A success oracle for the switch-decider: given the env after a bridging window,
# returns a bool per env saying whether the target skill actually took over safely.
# It is privileged and external, never read off the skill itself (see Skill).
SuccessFn = Callable[[ManagerBasedRlEnv], torch.Tensor]


def _as_tensor(value: object) -> torch.Tensor:
  """Narrow a `VecEnvObs`/`TensorDict` lookup result to a plain `torch.Tensor`."""
  assert isinstance(value, torch.Tensor)
  return value


class RingBuffer:
  """Fixed-capacity, overwrite-oldest storage for named 2D tensor fields."""

  def __init__(self, capacity: int, device: str, **shapes: int) -> None:
    self.capacity = capacity
    self.device = device
    self._fields = tuple(shapes)
    self._data = {
      name: torch.zeros((capacity, dim), device=device) for name, dim in shapes.items()
    }
    self._size = 0
    self._next = 0

  def __len__(self) -> int:
    return self._size

  def add(self, **values: torch.Tensor) -> None:
    """Append a batch of rows, each tensor shaped (N, dim)."""
    n = next(iter(values.values())).shape[0]
    idx = (torch.arange(n, device=self.device) + self._next) % self.capacity
    for name in self._fields:
      self._data[name][idx] = values[name]
    self._next = (self._next + n) % self.capacity
    self._size = min(self._size + n, self.capacity)

  def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
    if self._size == 0:
      raise RuntimeError("Cannot sample from an empty buffer")
    idx = torch.randint(0, self._size, (batch_size,), device=self.device)
    return {name: self._data[name][idx] for name in self._fields}


def collect_target_window(
  env: ManagerBasedRlEnv,
  target_skill: Skill,
  obs_dim: int,
  action_dim: int,
  window_steps: int,
  window_episodes: int,
) -> RingBuffer:
  """Roll the frozen target skill from its own reset, recording the first
  `window_steps` transitions of each episode as the discriminator's "real" data.

  If an episode terminates before `window_steps` (auto-reset), the transition into
  the next episode is still recorded as part of the window.
  TODO we might want to be sure about this
  """
  device = env.device
  num_envs = env.num_envs
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  capacity = window_steps * window_episodes * num_envs
  buffer = RingBuffer(
    capacity, device, obs=obs_dim, action=action_dim, next_obs=obs_dim, done=1
  )

  episodes = 0
  while episodes < window_episodes:
    obs, _ = env.reset()
    target_skill.reset(active)
    for _ in range(window_steps):
      cur_obs = _as_tensor(obs["actor"])
      actions = target_skill.act(obs, active)
      next_obs, _, terminated, time_out, _ = env.step(actions)
      done = (terminated | time_out).float().unsqueeze(-1)
      buffer.add(
        obs=cur_obs, action=actions, next_obs=_as_tensor(next_obs["actor"]), done=done
      )
      obs = next_obs
    episodes += num_envs

  return buffer


def collect_interrupt_states(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  exclude: int,
  max_steps: int,
  num_rollouts: int,
  capacity: int,
) -> tuple[RingBuffer, bool]:
  """Harvest candidate bridge-training start states.

  For every skill other than `exclude` (the bridge's own target), roll that skill
  from its own reset for a random number of steps in `[0, max_steps]` (per env) and
  record the resulting full simulator state. Returns the buffer and whether it holds
  a free-joint root state (False for fixed-base entities like the cartpole cart,
  which `write_root_state_to_sim` rejects).
  """
  device = env.device
  num_envs = env.num_envs
  entity = env.scene[entity_name]
  num_joints = entity.data.joint_pos.shape[-1]
  has_root_state = not entity.is_fixed_base
  others = [i for i in range(len(pool)) if i != exclude]
  active = torch.ones(num_envs, dtype=torch.bool, device=device)

  shapes = {"joint_pos": num_joints, "joint_vel": num_joints}
  if has_root_state:
    shapes["root_state"] = 13  # why though?
  buffer = RingBuffer(capacity, device, **shapes)

  for _ in range(num_rollouts):
    for skill_id in others:
      skill = pool[skill_id]
      obs, _ = env.reset()
      skill.reset(active)

      # A random capture step per env; step until every env has been captured
      target_steps = torch.randint(0, max_steps + 1, (num_envs,), device=device)
      captured = torch.zeros(num_envs, dtype=torch.bool, device=device)
      for t in range(max_steps + 1):
        hit = (target_steps == t) & ~captured
        if hit.any():
          values = {
            "joint_pos": entity.data.joint_pos[hit],
            "joint_vel": entity.data.joint_vel[hit],
          }
          if has_root_state:
            values["root_state"] = torch.cat(
              [entity.data.root_link_pose_w, entity.data.root_link_vel_w], dim=-1
            )[hit]
          buffer.add(**values)
          captured |= hit
        if t < max_steps and not bool(captured.all()):
          obs, _, _, _, _ = env.step(skill.act(obs, active))

  return buffer, has_root_state


def reset_to_interrupt_states(
  env: ManagerBasedRlEnv,
  entity: Entity,
  buffer: RingBuffer,
  has_root_state: bool,
  num_envs: int,
) -> VecEnvObs:
  """Reset every env to a freshly sampled harvested interrupt state.

  Mirrors the tail of `ManagerBasedRlEnv.reset()` (scene write, forward, command,
  sense, abstraction, obs compute) but substitutes a direct state write for the
  event-manager-driven random reset `_reset_idx` would normally do.
  """
  sample = buffer.sample(num_envs)
  if has_root_state:
    entity.write_root_state_to_sim(sample["root_state"])
  entity.write_joint_state_to_sim(sample["joint_pos"], sample["joint_vel"])
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.episode_length_buf[:] = 0
  env.command_manager.compute(dt=0.0)
  env.sim.sense()
  env.abstraction_manager.compute(dt=0.0)
  return env.observation_manager.compute(update_history=True)


def _log_prob(
  actor: MLPModel, obs: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
  """Log-probability of `actions` under the actor's current distribution at `obs`."""
  obs_td = TensorDict({"actor": obs}, batch_size=[obs.shape[0]])
  actor(obs_td, stochastic_output=True)  # Updates the distribution's params
  return actor.get_output_log_prob(actions).squeeze(-1)


def train_bridge(
  env: ManagerBasedRlEnv,
  actor: MLPModel,
  target_skill: Skill,
  target_skill_id: int,
  pool: SkillPool,
  entity_name: str,
  *,
  critic_hidden_dims: tuple[int, ...] = (64, 64),
  window_steps: int = 32,
  window_episodes: int = 64,
  interrupt_max_steps: int = 64,
  interrupt_rollouts: int = 64,
  interrupt_capacity: int = 4096,
  disc_hidden_dims: tuple[int, ...] = (100, 100),
  disc_learning_rate: float = 3e-4,
  disc_epochs: int = 1,
  disc_batch_size: int = 64,
  disc_gamma: float = 0.99,
  num_steps_per_env: int = 24,
  num_learning_epochs: int = 5,
  num_mini_batches: int = 4,
  clip_param: float = 0.2,
  gamma: float = 0.99,
  lam: float = 0.95,
  value_loss_coef: float = 1.0,
  entropy_coef: float = 0.01,
  learning_rate: float = 1e-3,
  max_grad_norm: float = 1.0,
  num_iterations: int = 100,
  log_every: int = 1,
) -> None:
  """Phase 1: train `actor` in place by AIRL + PPO to move like `target_skill`.

  The referee (discriminator) is trained to tell the actor's rollouts from the
  target skill's own transition-window rollouts; the actor is trained (PPO) to fool
  it, using the referee's confidence as reward. A throwaway critic is built here for
  PPO and discarded (only the actor is kept, in the meta policy).
  """
  device = env.device
  num_envs = env.num_envs
  action_dim = env.action_manager.total_action_dim
  entity = env.scene[entity_name]

  obs, _ = env.reset()
  obs_td = TensorDict(obs, batch_size=[num_envs])
  obs_dim = _as_tensor(obs_td["actor"]).shape[-1]

  critic = MLPModel(
    obs_td, OBS_GROUPS, "critic", 1, hidden_dims=critic_hidden_dims, activation="tanh"
  ).to(device)
  storage = RolloutStorage(
    "rl", num_envs, num_steps_per_env, obs_td, [action_dim], device
  )
  ppo = PPO(
    actor,
    critic,
    storage,
    num_learning_epochs=num_learning_epochs,
    num_mini_batches=num_mini_batches,
    clip_param=clip_param,
    gamma=gamma,
    lam=lam,
    value_loss_coef=value_loss_coef,
    entropy_coef=entropy_coef,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    device=device,
  )
  discriminator = AIRLDiscriminator(
    obs_dim, action_dim, disc_hidden_dims, disc_gamma
  ).to(device)
  disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_learning_rate)

  print(
    f"\n=== bridge -> '{target_skill.name}': AIRL + PPO, "
    f"{num_iterations} iterations over {num_envs} envs ==="
  )
  print(f"[collect] rolling '{target_skill.name}' for its transition window...")
  window = collect_target_window(
    env, target_skill, obs_dim, action_dim, window_steps, window_episodes
  )
  print(f"[collect] target window: {len(window)} transitions")
  print(f"[collect] rolling {len(pool) - 1} other skill(s) for interrupt states...")
  interrupts, has_root_state = collect_interrupt_states(
    env,
    pool,
    entity_name,
    target_skill_id,
    interrupt_max_steps,
    interrupt_rollouts,
    interrupt_capacity,
  )
  print(f"[collect] interrupt states: {len(interrupts)}")

  for iteration in range(num_iterations):
    obs = reset_to_interrupt_states(env, entity, interrupts, has_root_state, num_envs)
    obs_td = TensorDict(obs, batch_size=[num_envs])
    ppo.train_mode()
    reward_sum = 0.0
    disc_loss_sum = 0.0
    fake_obs, fake_actions, fake_next_obs, fake_done = [], [], [], []

    for _ in range(num_steps_per_env):
      actions = ppo.act(obs_td)
      next_obs, _, terminated, time_out, extras = env.step(actions)
      done = terminated | time_out
      next_obs_td = TensorDict(next_obs, batch_size=[num_envs])

      log_prob = ppo.transition.actions_log_prob
      assert log_prob is not None
      batch = DiscBatch(
        obs=_as_tensor(obs_td["actor"]),
        actions=actions,
        next_obs=_as_tensor(next_obs_td["actor"]),
        done=done,
        log_prob=log_prob.squeeze(-1),
      )
      # Reward = how convinced the referee is that this looked like the real skill
      with torch.no_grad():
        reward = discriminator.reward(batch)
      reward_sum += float(reward.mean())

      fake_obs.append(_as_tensor(obs_td["actor"]))
      fake_actions.append(actions)
      fake_next_obs.append(_as_tensor(next_obs_td["actor"]))
      fake_done.append(done)

      step_extras = dict(extras)
      step_extras["time_outs"] = time_out
      ppo.process_env_step(next_obs_td, reward, done, step_extras)
      obs_td = next_obs_td

    ppo.compute_returns(obs_td)

    # Train the referee: this rollout's transitions are the "fake" half, the target
    # skill's own transition window the "real" half. Uses the actor as it stood
    # during the rollout, so the rewards fed to PPO above are one update behind.
    fake_obs_t = torch.cat(fake_obs)
    fake_actions_t = torch.cat(fake_actions)
    fake_next_obs_t = torch.cat(fake_next_obs)
    fake_done_t = torch.cat(fake_done)
    for _ in range(disc_epochs):
      idx = torch.randint(0, fake_obs_t.shape[0], (disc_batch_size,), device=device)
      policy_batch = DiscBatch(
        obs=fake_obs_t[idx],
        actions=fake_actions_t[idx],
        next_obs=fake_next_obs_t[idx],
        done=fake_done_t[idx],
        log_prob=_log_prob(actor, fake_obs_t[idx], fake_actions_t[idx]),
      )
      expert = window.sample(disc_batch_size)
      expert_batch = DiscBatch(
        obs=expert["obs"],
        actions=expert["action"],
        next_obs=expert["next_obs"],
        done=expert["done"].squeeze(-1),
        log_prob=_log_prob(actor, expert["obs"], expert["action"]),
      )
      disc_loss = discriminator.loss(policy_batch, expert_batch)
      disc_optimizer.zero_grad()
      disc_loss.backward()
      disc_optimizer.step()
      disc_loss_sum += float(disc_loss)

    losses = ppo.update()

    if log_every and (iteration % log_every == 0 or iteration == num_iterations - 1):
      # `reward` is softplus(logit): it falls as the referee gets better and rises as
      # the bridge fools it, so read it against `disc_loss`, not as a task score. A
      # disc_loss collapsing toward 0 means the referee is winning outright and the
      # bridge has no gradient left to climb.
      print(
        f"[bridge {iteration + 1:4d}/{num_iterations}] "
        f"airl_reward {reward_sum / num_steps_per_env:7.4f} | "
        f"disc_loss {disc_loss_sum / max(disc_epochs, 1):7.4f} | "
        f"value {losses['value']:8.4f} | "
        f"surrogate {losses['surrogate']:+8.4f} | "
        f"entropy {losses['entropy']:6.3f}"
      )


def _epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
  frac = min(1.0, step / max(decay_steps, 1))
  return start + frac * (end - start)


def train_switch(
  env: ManagerBasedRlEnv,
  actor: MLPModel,
  switch: SwitchQNetwork,
  target_skill: Skill,
  target_skill_id: int,
  pool: SkillPool,
  entity_name: str,
  success_fn: SuccessFn,
  *,
  learning_rate: float = 1e-4,
  gamma: float = 0.99,
  replay_capacity: int = 100_000,
  batch_size: int = 128,
  update_target_every: int = 500,
  epsilon_start: float = 1.0,
  epsilon_end: float = 0.05,
  epsilon_decay_steps: int = 20_000,
  max_transition_steps: int = 64,
  num_iterations: int = 2000,
  warmup_steps: int = 1000,
  log_every: int = 50,
) -> None:
  """Phase 2: train `switch` in place by Double-DQN, against the frozen `actor`.

  The frozen actor drives the robot while the switch-decider chooses, per step,
  whether to hand over. `success_fn(env)` is evaluated once per window (after
  `max_transition_steps`) and is the only outcome signal: +1 for switching into a
  success, -1 for switching into a failure or for never committing (timeout).
  """
  device = env.device
  num_envs = env.num_envs
  entity = env.scene[entity_name]

  obs, _ = env.reset()
  obs_dim = _as_tensor(obs["actor"]).shape[-1]
  dqn = DoubleDQN(switch, gamma, learning_rate, device)
  replay = RingBuffer(
    replay_capacity, device, obs=obs_dim, action=1, reward=1, next_obs=obs_dim, done=1
  )

  print(
    f"\n=== switch -> '{target_skill.name}': Double-DQN, "
    f"{num_iterations} iterations over {num_envs} envs ==="
  )
  print(f"[collect] rolling {len(pool) - 1} other skill(s) for interrupt states...")
  interrupts, has_root_state = collect_interrupt_states(
    env,
    pool,
    entity_name,
    target_skill_id,
    max_transition_steps,
    max_transition_steps,
    replay_capacity,
  )
  print(f"[collect] interrupt states: {len(interrupts)}")

  all_envs = torch.ones(num_envs, dtype=torch.bool, device=device)
  total_steps = 0
  last_loss = float("nan")

  for iteration in range(num_iterations):
    obs = reset_to_interrupt_states(env, entity, interrupts, has_root_state, num_envs)
    target_skill.reset(all_envs)

    switched = torch.zeros(num_envs, dtype=torch.bool, device=device)
    has_switch = torch.zeros(num_envs, dtype=torch.bool, device=device)
    switch_obs = torch.zeros(num_envs, obs_dim, device=device)
    stay_obs, stay_action, stay_reward, stay_next_obs, stay_done = [], [], [], [], []

    for t in range(max_transition_steps):
      state = _as_tensor(obs["actor"])
      epsilon = _epsilon(total_steps, epsilon_start, epsilon_end, epsilon_decay_steps)

      with torch.no_grad():
        bridge_actions = actor(TensorDict({"actor": state}, batch_size=[num_envs]))
      target_actions = target_skill.act(obs, switched)
      actions = torch.where(switched.unsqueeze(-1), target_actions, bridge_actions)

      decisions = dqn.act(state, epsilon)
      bridging_before = ~switched
      new_switch = (decisions == 1) & bridging_before

      next_obs, _, _, _, _ = env.step(actions)
      next_state = _as_tensor(next_obs["actor"])

      record_switch = new_switch & ~has_switch
      if record_switch.any():
        switch_obs[record_switch] = state[record_switch]
        has_switch = has_switch | record_switch

      record_stay = bridging_before & ~new_switch
      is_last = t == max_transition_steps - 1
      if record_stay.any():
        n = int(record_stay.sum())
        stay_obs.append(state[record_stay])
        stay_action.append(torch.zeros(n, 1, device=device))
        stay_reward.append(torch.full((n, 1), -1.0 if is_last else 0.0, device=device))
        stay_next_obs.append(next_state[record_stay])
        stay_done.append(torch.full((n, 1), float(is_last), device=device))

      switched = switched | new_switch
      obs = next_obs
      total_steps += num_envs

    success = success_fn(env)
    final_reward = torch.where(
      success, torch.ones_like(success, dtype=torch.float32), -1.0
    )

    if has_switch.any():
      n = int(has_switch.sum())
      replay.add(
        obs=switch_obs[has_switch],
        action=torch.ones(n, 1, device=device),
        reward=final_reward[has_switch].unsqueeze(-1),
        next_obs=switch_obs[has_switch],  # Unused: done=1 zeroes the bootstrap term.
        done=torch.ones(n, 1, device=device),
      )
    if stay_obs:
      replay.add(
        obs=torch.cat(stay_obs),
        action=torch.cat(stay_action),
        reward=torch.cat(stay_reward),
        next_obs=torch.cat(stay_next_obs),
        done=torch.cat(stay_done),
      )

    warmed_up = len(replay) >= batch_size and total_steps >= warmup_steps
    if warmed_up:
      batch = replay.sample(batch_size)
      last_loss = float(
        dqn.update(
          batch["obs"],
          batch["action"].squeeze(-1).long(),
          batch["reward"].squeeze(-1),
          batch["next_obs"],
          batch["done"].squeeze(-1),
        )
      )
      if iteration % update_target_every == 0:
        dqn.sync_target()

    if log_every and (iteration % log_every == 0 or iteration == num_iterations - 1):
      # A switch rate near 1.0 with a poor success rate means it hands over
      # indiscriminately; a rate near 0.0 means it never commits and just eats the
      # -1 timeout at the end of every window.
      print(
        f"[switch {iteration + 1:5d}/{num_iterations}] "
        f"eps {epsilon:4.2f} | "
        f"switched {float(switched.float().mean()):5.1%} | "
        f"success {float(success.float().mean()):5.1%} | "
        f"loss {last_loss:8.4f} | "
        f"replay {len(replay)}" + ("" if warmed_up else " (warmup)")
      )


def train(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  meta: Arch1,
  success_fns: dict[int, SuccessFn],
) -> Arch1:
  """Train every bridge and switch-decider `meta` holds, in place, and return it.

  For each target skill: phase 1 (AIRL + PPO) trains its actor, then phase 2
  (Double-DQN) trains its switch-decider against `success_fns[target_id]`.
  """
  for target_id in range(len(pool)):
    target_skill = pool[target_id]
    actor = meta.actors[target_id]
    train_bridge(env, actor, target_skill, target_id, pool, entity_name)

    # Freeze the actor before training the switch-decider on top of it.
    actor.eval()
    for p in actor.parameters():
      p.requires_grad_(False)

    train_switch(
      env,
      actor,
      meta.switches[target_id],
      target_skill,
      target_id,
      pool,
      entity_name,
      success_fns[target_id],
    )
  return meta
