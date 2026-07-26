"""Training for arch_1: everything that turns an untrained `Arch1` into a trained one.

For each target skill in the pool, in order:
- collect the target skill's own initiation window (the "real" data to imitate);
- collect interrupt states from the *other* skills (where the bridge gets dropped);
- phase 1: train that skill's bridge actor by AIRL + PPO (move like the target);
- phase 2: freeze the actor and train that skill's switch-decider by Double-DQN
    (when to hand over), against an external success signal.

Only training-specific machinery lives here (the two loops and their metrics). What is
collected, how it is stored, and how it is checked lives in windows.py; the reusable
inference-time networks in networks.py; the meta policy that holds them in
__init__.py. rsl_rl's `PPO`, `RolloutStorage`, and `MLPModel` are reused directly
rather than through `OnPolicyRunner`, because the runner owns reset scheduling and
these loops must reset training episodes to harvested interrupt states instead.

Nothing here is experiment specific: `train` takes the env, the pool, the entity to
harvest states from, and one success oracle per target skill. Each experiment owns
its own entry point that supplies those and calls it.

A note on the success oracle, because it is easy to get silently wrong. The env
auto-resets an environment the moment it terminates, so by the end of a hand-over
window a robot that fell over is already upright again, and an oracle that only reads
live state at the end of the window reports that failure as a success. Phase 2
therefore latches terminations as they happen and evaluates the oracle a fixed number
of steps after *each* env's own hand-over, not once at the end for everybody.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_1 import Arch1
from mjlab.tasks.skills.architectures.arch_1.networks import (
  OBS_GROUPS,
  AIRLDiscriminator,
  DiscBatch,
  DoubleDQN,
  SwitchQNetwork,
)
from mjlab.tasks.skills.architectures.arch_1.windows import (
  ManagerState,
  RingBuffer,
  as_tensor,
  collect_interrupts,
  collect_target_windows,
  restore_interrupts,
  verify_restore,
  window_transitions,
)
from mjlab.tasks.skills.skill import Skill, SkillPool

# A success oracle for the switch-decider: given the env, returns a bool per env saying
# whether the target skill is in a state it took over safely into. It is privileged and
# external, never read off the skill itself (see Skill). It is called every step and
# read at each env's own evaluation moment, so it must be a pure function of the
# current state, with no memory of its own.
SuccessFn = Callable[[ManagerBasedRlEnv], torch.Tensor]


def filter_kwargs(fn, kwargs):
  params = inspect.signature(fn).parameters
  return {k: v for k, v in kwargs.items() if k in params}


def _log_prob(
  actor: MLPModel, obs: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
  """Log-probability of `actions` under the actor's current distribution at `obs`.

  Detached: AIRL treats log pi as a known constant inside the discriminator logit, so
  the discriminator's loss must not push gradients back into the policy.
  """
  with torch.no_grad():
    obs_td = TensorDict({"actor": obs}, batch_size=[obs.shape[0]])
    actor(obs_td, stochastic_output=True)  # Updates the distribution's params
    return actor.get_output_log_prob(actions).squeeze(-1)


def _called_real(discriminator: AIRLDiscriminator, batch: DiscBatch) -> float:
  """Fraction of a batch the discriminator calls "expert" (logit > 0)."""
  with torch.no_grad():
    return float((discriminator.logits(batch) > 0).float().mean())


def train_bridge(
  env: ManagerBasedRlEnv,
  actor: MLPModel,
  target_skill: Skill,
  target_skill_id: int,
  pool: SkillPool,
  entity_name: str,
  *,
  critic_hidden_dims: tuple[int, ...] = (64, 64),
  window_steps: int = 64,
  window_episodes: int = 512,
  interrupt_max_steps: int = 64,
  interrupt_rollouts: int = 16,
  interrupt_capacity: int = 4096,
  disc_hidden_dims: tuple[int, ...] = (100, 100),
  disc_learning_rate: float = 3e-4,
  disc_epochs: int = 4,
  disc_batch_size: int = 512,
  disc_gamma: float = 0.99,
  num_steps_per_env: int = 64,
  num_learning_epochs: int = 5,
  num_mini_batches: int = 4,
  clip_param: float = 0.2,
  bridge_gamma: float = 0.99,
  lam: float = 0.95,
  value_loss_coef: float = 1.0,
  entropy_coef: float = 0.01,
  bridge_learning_rate: float = 1e-3,
  max_grad_norm: float = 1.0,
  bridge_num_iterations: int = 500,
  bridge_log_every: int = 10,
) -> None:
  """Phase 1: train `actor` in place by AIRL + PPO to move like `target_skill`.

  The referee (discriminator) is trained to tell the actor's rollouts from the
  target skill's own initiation-window rollouts; the actor is trained (PPO) to fool
  it, using the referee's confidence as reward. A throwaway critic is built here for
  PPO and discarded (only the actor is kept, in the meta policy).
  """

  device = env.device
  num_envs = env.num_envs
  action_dim = env.action_manager.total_action_dim
  entity = env.scene[entity_name]
  manager_state = ManagerState(env)
  skill_names = tuple(s.name for s in pool.skills)

  obs, _ = env.reset()
  obs_td = TensorDict(obs, batch_size=[num_envs])
  obs_dim = as_tensor(obs_td["actor"]).shape[-1]

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
    gamma=bridge_gamma,
    lam=lam,
    value_loss_coef=value_loss_coef,
    entropy_coef=entropy_coef,
    learning_rate=bridge_learning_rate,
    max_grad_norm=max_grad_norm,
    device=device,
  )

  print(
    f"\n=== bridge -> '{target_skill.name}': AIRL + PPO, "
    f"{bridge_num_iterations} iterations over {num_envs} envs ==="
  )
  print(f"[collect] rolling '{target_skill.name}' for its initiation window...")
  windows = collect_target_windows(
    env, target_skill, target_skill_id, skill_names, window_steps, window_episodes
  )
  expert = window_transitions(windows)
  expert_buffer = RingBuffer(
    int(expert["obs"].shape[0]),
    device,
    {"obs": (obs_dim,), "action": (action_dim,), "next_obs": (obs_dim,), "done": ()},
  )
  expert_buffer.add(**expert)
  print(
    f"[collect] target window: {len(windows)} windows, "
    f"{len(expert_buffer)} usable transitions"
  )

  print(f"[collect] rolling {len(pool) - 1} other skill(s) for interrupt states...")
  interrupts = collect_interrupts(
    env,
    pool,
    entity_name,
    target_skill_id,
    interrupt_max_steps,
    interrupt_rollouts,
    interrupt_capacity,
    window_steps,
  )
  print(f"[collect] interrupt states: {len(interrupts)}")

  max_error, within = verify_restore(env, entity, interrupts, manager_state)
  print(
    f"[verify] interrupt restore round-trip: max obs error {max_error:.3e}, "
    f"{within:.1%} exact"
  )
  if within <= 0.99:
    print(
      "[verify] WARNING: restoring a harvested state does not reproduce the "
      "observation it was captured with. Training would start from states that were "
      "never actually visited; fix this before reading anything below."
    )

  # The discriminator reads raw observations and actions, whose scales are whatever
  # the experiment happens to use (the diffdrive commands wheel velocities in the tens
  # of rad/s). Standardizing against the expert data keeps its first layer in a sane
  # range instead of saturating on the first batch.
  discriminator = AIRLDiscriminator(
    obs_dim, action_dim, disc_hidden_dims, disc_gamma
  ).to(device)
  discriminator.fit_normalization(expert["obs"], expert["action"])
  disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_learning_rate)

  # How far outside the target skill's own states the bridge ends up, in expert
  # standard deviations. This is the one phase-1 number that is not part of the
  # adversarial game, so it is the one neither side can game.
  expert_mean = expert["obs"].mean(dim=0)
  expert_std = expert["obs"].std(dim=0).clamp_min(1e-3)

  for iteration in range(bridge_num_iterations):
    obs = restore_interrupts(env, entity, interrupts, manager_state)
    obs_td = TensorDict(obs, batch_size=[num_envs])
    ppo.train_mode()
    reward_sum = 0.0
    terminations = 0.0
    fake_obs, fake_actions, fake_next_obs, fake_done = [], [], [], []

    for _ in range(num_steps_per_env):
      actions = ppo.act(obs_td)
      next_obs, _, terminated, time_out, extras = env.step(actions)
      done = terminated | time_out
      terminations += float(terminated.float().mean())
      next_obs_td = TensorDict(next_obs, batch_size=[num_envs])

      log_prob = ppo.transition.actions_log_prob
      assert log_prob is not None
      batch = DiscBatch(
        obs=as_tensor(obs_td["actor"]),
        actions=actions,
        next_obs=as_tensor(next_obs_td["actor"]),
        done=done,
        log_prob=log_prob.squeeze(-1),
      )
      # Reward = how convinced the referee is that this looked like the real skill
      with torch.no_grad():
        reward = discriminator.reward(batch)
      reward_sum += float(reward.mean())

      fake_obs.append(as_tensor(obs_td["actor"]))
      fake_actions.append(actions)
      fake_next_obs.append(as_tensor(next_obs_td["actor"]))
      fake_done.append(done)

      step_extras = dict(extras)
      step_extras["time_outs"] = time_out
      ppo.process_env_step(next_obs_td, reward, done, step_extras)
      obs_td = next_obs_td

    ppo.compute_returns(obs_td)

    # Train the referee: this rollout's transitions are the "fake" half, the target
    # skill's own initiation window the "real" half. Uses the actor as it stood
    # during the rollout, so the rewards fed to PPO above are one update behind.
    fake_obs_t = torch.cat(fake_obs)
    fake_actions_t = torch.cat(fake_actions)
    fake_next_obs_t = torch.cat(fake_next_obs)
    fake_done_t = torch.cat(fake_done)
    disc_loss_sum = 0.0
    fooled = 0.0
    caught = 0.0
    for _ in range(disc_epochs):
      idx = torch.randint(0, fake_obs_t.shape[0], (disc_batch_size,), device=device)
      policy_batch = DiscBatch(
        obs=fake_obs_t[idx],
        actions=fake_actions_t[idx],
        next_obs=fake_next_obs_t[idx],
        done=fake_done_t[idx],
        log_prob=_log_prob(actor, fake_obs_t[idx], fake_actions_t[idx]),
      )
      real = expert_buffer.sample(disc_batch_size)
      expert_batch = DiscBatch(
        obs=real["obs"],
        actions=real["action"],
        next_obs=real["next_obs"],
        done=real["done"],
        log_prob=_log_prob(actor, real["obs"], real["action"]),
      )
      disc_loss = discriminator.loss(policy_batch, expert_batch)
      disc_optimizer.zero_grad()
      disc_loss.backward()
      disc_optimizer.step()
      disc_loss_sum += float(disc_loss)
      fooled += _called_real(discriminator, policy_batch)
      caught += _called_real(discriminator, expert_batch)

    losses = ppo.update()

    if bridge_log_every and (
      iteration % bridge_log_every == 0 or iteration == bridge_num_iterations - 1
    ):
      # `airl_reward` is softplus(logit): it falls as the referee gets better and
      # rises as the bridge fools it, so it is not a task score. Read `fooled` (the
      # share of the bridge's own transitions the referee calls real) against `caught`
      # (the share of real ones it calls real). A healthy game keeps caught high while
      # fooled climbs; fooled near 0 with caught near 1 means the referee has won
      # outright and the bridge has no gradient left to climb, and both near 0.5 means
      # it has stopped telling them apart at all. `drift` is the only number here
      # outside the game: mean distance from the target skill's own states, in expert
      # sigmas, and it is what should actually fall if the bridge is learning.
      with torch.no_grad():
        drift = ((fake_obs_t - expert_mean) / expert_std).norm(dim=-1).mean()
      denom = max(disc_epochs, 1)
      print(
        f"[bridge {iteration + 1:4d}/{bridge_num_iterations}] "
        f"airl_reward {reward_sum / num_steps_per_env:7.4f} | "
        f"disc_loss {disc_loss_sum / denom:6.4f} | "
        f"fooled {fooled / denom:5.1%} | "
        f"caught {caught / denom:5.1%} | "
        f"drift {float(drift):6.2f} | "
        f"term {terminations / num_steps_per_env:5.1%} | "
        f"value {losses['value']:8.4f} | "
        f"entropy {losses['entropy']:6.3f}"
      )


def _rate(outcome: torch.Tensor, mask: torch.Tensor) -> float:
  """Share of the envs `mask` selects whose hand-over came out well."""
  return float(outcome[mask].float().mean()) if bool(mask.any()) else float("nan")


def _epsilon(iteration: int, start: float, end: float, decay_iterations: int) -> float:
  """Linear epsilon schedule, measured in *iterations* (one hand-over window each).

  Not in env steps: with a thousand parallel envs a single window is tens of thousands
  of env steps, so a step-counted schedule collapses to its floor partway through the
  first iteration and the decider never explores at all.
  """
  frac = min(1.0, iteration / max(decay_iterations, 1))
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
  switch_learning_rate: float = 1e-4,
  switch_gamma: float = 0.99,
  replay_capacity: int = 200_000,
  batch_size: int = 512,
  updates_per_iteration: int = 8,
  update_target_every: int = 20,
  epsilon_start: float = 1.0,
  epsilon_end: float = 0.05,
  epsilon_decay_iterations: int = 300,
  max_transition_steps: int = 64,
  # How long to let the target skill drive before judging the hand-over. Long enough
  # that a bad hand-off has actually gone wrong by then: on the diffdrive a tip takes
  # 24 to 32 steps to develop, so anything much shorter grades a failure as a success
  # simply because it has not finished falling over yet.
  eval_steps: int = 48,
  window_steps: int = 64,
  interrupt_rollouts: int = 16,
  interrupt_capacity: int = 4096,
  switch_num_iterations: int = 2000,
  warmup_iterations: int = 10,
  switch_log_every: int = 25,
) -> None:
  """Phase 2: train `switch` in place by Double-DQN, against the frozen `actor`.

  The frozen actor drives the robot while the switch-decider chooses, per step,
  whether to hand over. Each env is judged `eval_steps` steps after *its own*
  hand-over: +1 if the oracle is satisfied there and the env never terminated, -1
  otherwise, and -1 for never committing within `max_transition_steps`.
  """

  device = env.device
  num_envs = env.num_envs
  entity = env.scene[entity_name]
  manager_state = ManagerState(env)

  obs, _ = env.reset()
  obs_dim = as_tensor(obs["actor"]).shape[-1]
  dqn = DoubleDQN(switch, switch_gamma, switch_learning_rate, device)
  replay = RingBuffer(
    replay_capacity,
    device,
    {"obs": (obs_dim,), "action": (), "reward": (), "next_obs": (obs_dim,), "done": ()},
    {"action": torch.long},
  )

  print(
    f"\n=== switch -> '{target_skill.name}': Double-DQN, "
    f"{switch_num_iterations} iterations over {num_envs} envs ==="
  )
  print(f"[collect] rolling {len(pool) - 1} other skill(s) for interrupt states...")
  interrupts = collect_interrupts(
    env,
    pool,
    entity_name,
    target_skill_id,
    max_transition_steps,
    interrupt_rollouts,
    interrupt_capacity,
    window_steps,
  )
  print(f"[collect] interrupt states: {len(interrupts)}")

  max_error, within = verify_restore(env, entity, interrupts, manager_state)
  print(
    f"[verify] interrupt restore round-trip: max obs error {max_error:.3e}, "
    f"{within:.1%} exact"
  )

  all_envs = torch.ones(num_envs, dtype=torch.bool, device=device)
  total_steps = max_transition_steps + eval_steps
  last_loss = float("nan")

  # An aggregate success rate is dominated by the easy interrupt states: a source
  # skill only just started has barely built up the momentum the bridge exists to
  # shed, and handing straight over there works anyway. How long the source skill had
  # been running is the one difficulty proxy available without knowing the experiment,
  # so the harder half is reported separately.
  offsets = interrupts.windows.offset
  hard_threshold = float(offsets.float().median())

  for iteration in range(switch_num_iterations):
    indices = interrupts.sample_indices(num_envs, device)
    obs = restore_interrupts(env, entity, interrupts, manager_state, indices)
    hard = offsets[indices].float() >= hard_threshold
    target_skill.reset(all_envs)
    epsilon = _epsilon(iteration, epsilon_start, epsilon_end, epsilon_decay_iterations)

    switched = torch.zeros(num_envs, dtype=torch.bool, device=device)
    switch_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    switch_obs = torch.zeros(num_envs, obs_dim, device=device)
    # A termination anywhere in the window condemns that env: the env auto-resets the
    # moment it fires, so whatever the oracle would read afterwards belongs to a fresh
    # episode and says nothing about how the hand-over went.
    failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    outcome = torch.zeros(num_envs, dtype=torch.bool, device=device)
    outcome_known = torch.zeros(num_envs, dtype=torch.bool, device=device)
    stay_obs, stay_reward, stay_next_obs, stay_done = [], [], [], []

    for t in range(total_steps):
      state = as_tensor(obs["actor"])

      with torch.no_grad():
        bridge_actions = actor(TensorDict({"actor": state}, batch_size=[num_envs]))
      target_actions = target_skill.act(obs, switched)
      actions = torch.where(switched.unsqueeze(-1), target_actions, bridge_actions)

      bridging_before = ~switched
      alive_before = ~failed
      if t < max_transition_steps:
        decisions = dqn.act(state, epsilon)
        new_switch = (decisions == 1) & bridging_before & alive_before
      else:
        new_switch = torch.zeros(num_envs, dtype=torch.bool, device=device)

      next_obs, _, terminated, _, _ = env.step(actions)
      next_state = as_tensor(next_obs["actor"])
      failed = failed | terminated

      if new_switch.any():
        switch_obs[new_switch] = state[new_switch]
        switch_step[new_switch] = t
        switched = switched | new_switch

      # Staying is only a decision while the window is open. It ends the episode (with
      # -1) either when the bridge itself crashed the robot or when the window ran out
      # without ever committing.
      if t < max_transition_steps:
        record_stay = bridging_before & ~new_switch & alive_before
        terminal = terminated | (t == max_transition_steps - 1)
        if record_stay.any():
          stay_obs.append(state[record_stay])
          stay_reward.append(torch.where(terminal, -1.0, 0.0)[record_stay])
          stay_next_obs.append(next_state[record_stay])
          stay_done.append(terminal[record_stay].float())

      # Judge each env `eval_steps` after its own hand-over, not once at the end for
      # everybody: an env that committed on the last step would otherwise be graded on
      # a single step of the target skill.
      due = switched & ~outcome_known & (switch_step + eval_steps == t)
      if due.any():
        step_success = success_fn(env) & ~failed
        outcome = torch.where(due, step_success, outcome)
        outcome_known = outcome_known | due

      obs = next_obs

    if switched.any():
      n = int(switched.sum())
      replay.add(
        obs=switch_obs[switched],
        action=torch.ones(n, device=device),
        reward=torch.where(outcome, 1.0, -1.0)[switched],
        next_obs=switch_obs[switched],  # Unused: done=1 zeroes the bootstrap term
        done=torch.ones(n, device=device),
      )
    if stay_obs:
      obs_rows = torch.cat(stay_obs)
      replay.add(
        obs=obs_rows,
        action=torch.zeros(obs_rows.shape[0], device=device),
        reward=torch.cat(stay_reward),
        next_obs=torch.cat(stay_next_obs),
        done=torch.cat(stay_done),
      )

    warmed_up = len(replay) >= batch_size and iteration >= warmup_iterations
    if warmed_up:
      for _ in range(updates_per_iteration):
        batch = replay.sample(batch_size)
        last_loss = float(
          dqn.update(
            batch["obs"],
            batch["action"].long(),
            batch["reward"],
            batch["next_obs"],
            batch["done"],
          )
        )
      if iteration % update_target_every == 0:
        dqn.sync_target()

    if switch_log_every and (
      iteration % switch_log_every == 0 or iteration == switch_num_iterations - 1
    ):
      # `success` counts only the envs that actually handed over, and only counts a
      # hand-over good if that env survived to its evaluation moment. `hard` is the
      # same over the harder half of the interrupt states, which is where a bridge
      # either earns its keep or does not. `overall` folds in the envs that never
      # committed, so it is the number the decider is really maximizing. `switched`
      # near 1.0 with a poor `success` means it hands over indiscriminately; near 0.0
      # means it never commits and just eats the -1.
      success = _rate(outcome, switched)
      success_hard = _rate(outcome, switched & hard)
      mean_step = (
        float(switch_step[switched].float().mean())
        if bool(switched.any())
        else float("nan")
      )
      print(
        f"[switch {iteration + 1:5d}/{switch_num_iterations}] "
        f"eps {epsilon:4.2f} | "
        f"switched {float(switched.float().mean()):5.1%} @ step {mean_step:5.1f} | "
        f"success {success:5.1%} (hard {success_hard:5.1%}) | "
        f"terminated {float(failed.float().mean()):5.1%} | "
        f"overall {float(torch.where(switched & outcome, 1.0, -1.0).mean()):+5.2f} | "
        f"loss {last_loss:8.4f} | "
        f"replay {len(replay)}" + ("" if warmed_up else " (warmup)")
      )


def train(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  meta: Arch1,
  success_fns: dict[int, SuccessFn],
  **kwargs,
) -> Arch1:
  """Train every bridge and switch-decider `meta` holds, in place, and return it.

  For each target skill: phase 1 (AIRL + PPO) trains its actor, then phase 2
  (Double-DQN) trains its switch-decider against `success_fns[target_id]`.
  """

  bridge_kwargs = filter_kwargs(train_bridge, kwargs)
  switch_kwargs = filter_kwargs(train_switch, kwargs)
  unknown = set(kwargs) - set(bridge_kwargs) - set(switch_kwargs)
  if unknown:
    raise TypeError(f"Unknown training options: {sorted(unknown)}")

  for target_id in range(len(pool)):
    target_skill = pool[target_id]
    actor = meta.actors[target_id]
    train_bridge(
      env, actor, target_skill, target_id, pool, entity_name, **bridge_kwargs
    )

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
      **switch_kwargs,
    )
  return meta
