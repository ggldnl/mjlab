"""arch_1's two-phase training. arch_2 reuses both phases with a different verdict.

For each target skill in the pool, in order:

    collect its opening window (the "real" data) and interrupt states from the others
    phase 1: train that skill's bridge actor by AIRL + PPO      (move like the target)
    phase 2: freeze it and train that skill's decider by Double-DQN   (when to let go)

Phase 1 needs no notion of success at all. Phase 2 does, and that is the only thing the
two architectures differ on, so `train_bridging` takes the success test as a parameter.

Everything the bridge sees, is judged on and is rewarded for goes through the same two
conversions, and *the same* is the point: the experiment's `StateView` picks which
channels anything may work on, and the run's `Units` put them at unit size. AIRL rewards
the actor for producing what the referee cannot separate, so a channel one of them sees
and the other does not is either an unreachable target or a free giveaway.

Two things about resets, both easy to get silently wrong.

Phase 1: the env auto-resets a terminated env to the *arena's* start state, and keeps
producing rollout data from there. That is not the bridge's behavior, and for a target
skill that also starts from rest it is nearly the expert data, so the same states end up
labelled real and fake at once. So a crashed env is put straight back at a fresh
interrupt state and the transition that crossed the reset is dropped.

Phase 2 must not do that, because there the termination *is* the measurement. It latches
terminations as they happen and reads its verdict a fixed number of steps after *each*
env's own hand-over, since by then a robot that fell is upright again in a fresh episode.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_1 import Arch1
from mjlab.tasks.skills.architectures.arch_1.config import (
  BridgePhase,
  BridgeTraining,
  SwitchPhase,
)
from mjlab.tasks.skills.architectures.arch_1.switch import (
  DoubleDQN,
  SuccessFn,
  SwitchQNetwork,
)
from mjlab.tasks.skills.architectures.common.imitation import (
  AIRLDiscriminator,
  DiscBatch,
  RewardScaler,
  called_real,
)
from mjlab.tasks.skills.architectures.common.nets import (
  OBS_GROUPS,
  action_std,
  clamp_action_std,
  obs_td,
  set_action_std,
)
from mjlab.tasks.skills.buffers import RingBuffer
from mjlab.tasks.skills.experiment import Experiment
from mjlab.tasks.skills.spaces import Units, measure_spaces
from mjlab.tasks.skills.windows import (
  InterruptSet,
  ManagerState,
  as_tensor,
  collect_interrupts,
  collect_opening,
  restore_interrupts,
  verify_restore,
)


def _log_prob(
  actor: MLPModel, obs: torch.Tensor, normalized_actions: torch.Tensor
) -> torch.Tensor:
  """Log-probability of `normalized_actions` under the actor at `obs`. Detached.

  Both arguments are already in the run's units: the actor's distribution lives in
  normalized units, so handing it raw ones would be off by orders of magnitude. Detached
  because the referee's loss must not push gradients back into the policy.
  """
  with torch.no_grad():
    actor(obs_td(obs), stochastic_output=True)  # Updates the distribution's params
    return actor.get_output_log_prob(normalized_actions).squeeze(-1)


def _rate(outcome: torch.Tensor, mask: torch.Tensor) -> float:
  """Share of the envs `mask` selects whose hand-over came out well."""
  return float(outcome[mask].float().mean()) if bool(mask.any()) else float("nan")


def _prepare_interrupts(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  target_skill_id: int,
  manager_state: ManagerState,
  num_interrupts: int,
) -> InterruptSet:
  """Harvest the states a bridge gets dropped into, and prove they can be restored."""
  entity = env.scene[exp.entity_name]
  print(f"[collect] rolling {len(exp.pool) - 1} other skill(s) for interrupt states...")
  interrupts = collect_interrupts(
    env, exp.pool, exp.entity_name, target_skill_id, exp.windows, num_interrupts
  )
  print(f"[collect] {len(interrupts)} interrupt states")

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
  return interrupts


def train_bridge(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  actor: MLPModel,
  target_skill_id: int,
  units: Units,
  cfg: BridgePhase,
) -> None:
  """Phase 1: train `actor` in place by AIRL + PPO to move like the target skill.

  The referee is trained to tell the actor's rollouts from the target skill's own opening
  window; the actor is trained to fool it, using the referee's verdict as reward. A
  throwaway critic is built here for PPO and discarded.
  """

  device = env.device
  num_envs = env.num_envs
  action_dim = env.action_manager.total_action_dim
  entity = env.scene[exp.entity_name]
  manager_state = ManagerState(env)
  target_skill = exp.pool[target_skill_id]
  spec = exp.windows[target_skill]
  view = exp.view
  obs_dim = view.dim

  print(
    f"\n=== bridge -> '{target_skill.name}': AIRL + PPO, "
    f"{cfg.num_iterations} iterations over {num_envs} envs ==="
  )
  print(f"[view] matching on {view.label}")

  ##
  # The example to imitate, in the run's own units.
  ##

  print(f"[collect] rolling '{target_skill.name}' for its opening window...")
  opening = collect_opening(
    env, target_skill, target_skill_id, spec, cfg.num_windows, view=view
  )
  raw_expert = opening.transitions()
  expert_buffer = RingBuffer(
    int(raw_expert["obs"].shape[0]),
    device,
    {"obs": (obs_dim,), "action": (action_dim,), "next_obs": (obs_dim,), "done": ()},
  )
  expert_buffer.add(
    obs=units.state.standardize(raw_expert["obs"]),
    action=units.action.normalize(raw_expert["action"]),
    next_obs=units.state.standardize(raw_expert["next_obs"]),
    done=raw_expert["done"],
  )
  print(
    f"[collect] {len(opening)} opening windows of {spec.opening} steps, "
    f"{len(expert_buffer)} usable transitions"
  )

  ##
  # Where the bridge gets dropped in.
  ##

  interrupts = _prepare_interrupts(
    env, exp, target_skill_id, manager_state, cfg.num_interrupts
  )

  ##
  # The two players.
  ##

  obs = restore_interrupts(env, entity, interrupts, manager_state)
  template = obs_td(units.state.standardize(view(as_tensor(obs["actor"]))))

  # The actor was built before any config existed, so its exploration is set here from
  # the one the experiment declared rather than from the constructor's guess.
  set_action_std(actor, cfg.action_init_std)

  critic = MLPModel(
    template,
    OBS_GROUPS,
    "critic",
    1,
    hidden_dims=cfg.critic_hidden_dims,
    activation="tanh",
  ).to(device)
  storage = RolloutStorage(
    "rl", num_envs, cfg.steps_per_env, template, [action_dim], device
  )
  ppo = PPO(
    actor,
    critic,
    storage,
    num_learning_epochs=cfg.num_learning_epochs,
    num_mini_batches=cfg.num_mini_batches,
    clip_param=cfg.clip_param,
    gamma=cfg.gamma,
    lam=cfg.lam,
    value_loss_coef=cfg.value_loss_coef,
    entropy_coef=cfg.entropy_coef,
    learning_rate=cfg.learning_rate,
    max_grad_norm=cfg.max_grad_norm,
    schedule=cfg.ppo_schedule,
    device=device,
  )

  discriminator = AIRLDiscriminator(
    obs_dim, action_dim, cfg.disc_hidden_dims, cfg.disc_gamma
  ).to(device)
  # Sits between the referee and PPO: centers the reward so the bridge is paid neither
  # for surviving nor for crashing, and puts it at unit size so the critic can fit it.
  reward_scaler = RewardScaler(device, clip=cfg.reward_clip)
  disc_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=cfg.disc_learning_rate
  )

  ##
  # The game.
  ##

  for iteration in range(cfg.num_iterations):
    # Every iteration starts every env at a fresh interrupt state, so a rollout is one
    # bridging attempt per env rather than a continuation of the previous one.
    obs = restore_interrupts(env, entity, interrupts, manager_state)
    state = units.state.standardize(view(as_tensor(obs["actor"])))
    template = obs_td(state)
    ppo.train_mode()

    imitation_sum = 0.0
    terminations = 0.0
    restarts = 0.0
    broken = 0.0
    # What the bridge produced this rollout, kept for the referee's "fake" half.
    fake_obs, fake_actions, fake_next_obs, fake_done = [], [], [], []

    for _ in range(cfg.steps_per_env):
      # The actor thinks in normalized units, the simulator wants raw ones. This pair of
      # lines is the only place the two meet.
      normalized_action = ppo.act(template)
      raw_action = units.action.denormalize(normalized_action)

      next_obs, _, terminated, time_out, extras = env.step(raw_action)
      # `done` is exactly "mjlab auto-reset this env inside the step above".
      done = terminated | time_out
      terminations += float(terminated.float().mean())

      next_state = units.state.standardize(view(as_tensor(next_obs["actor"])))

      # A robot driven into itself hard enough overflows the constraint solver, and what
      # comes back is not a number. One such row turns the whole update to NaN, and the
      # first sign is PPO dying inside `Normal` hundreds of iterations later. Treat those
      # envs like the ones that terminated. `broken` is logged, because unlike a
      # termination this is never an acceptable steady state.
      unusable = ~torch.isfinite(next_state).all(dim=-1)
      if bool(unusable.any()):
        broken += float(unusable.float().sum())
        next_state = torch.nan_to_num(next_state)
        done = done | unusable

      next_template = obs_td(next_state)

      log_prob = ppo.transition.actions_log_prob
      assert log_prob is not None
      batch = DiscBatch(
        obs=state,
        actions=normalized_action,
        next_obs=next_state,
        done=done,
        log_prob=log_prob.squeeze(-1),
      )

      # Two clearly separate parts. The imitation part is the referee's verdict, centered
      # and at unit size, so it says both "that looked like the real skill" and "that did
      # not". The crash penalty says the one thing an imitation reward cannot know: that
      # falling over is worse than merely looking wrong. It is the only term with an
      # absolute sign, and the scaler having centered the other is what stops the two
      # from competing.
      with torch.no_grad():
        imitation_reward = reward_scaler(discriminator.reward(batch))
      reward = imitation_reward - cfg.termination_penalty * terminated.float()
      imitation_sum += float(imitation_reward.mean())

      # A transition whose step ended in a reset pairs a state the bridge reached with
      # one it did not: the env is already back at the arena's start state. Those rows
      # would teach the referee that the arena's start state is something the bridge
      # produces, which for a target skill that also starts from rest is the expert data.
      kept = ~done
      if bool(kept.any()):
        fake_obs.append(state[kept])
        fake_actions.append(normalized_action[kept])
        fake_next_obs.append(next_state[kept])
        fake_done.append(done[kept])

      step_extras = dict(extras)
      step_extras["time_outs"] = time_out
      ppo.process_env_step(next_template, reward, done, step_extras)

      # Put the envs the simulator just reset back where a bridge actually lives.
      if bool(done.any()):
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        restarts += float(done_ids.numel())
        next_obs = restore_interrupts(
          env, entity, interrupts, manager_state, env_ids=done_ids
        )
        next_state = units.state.standardize(view(as_tensor(next_obs["actor"])))
        next_template = obs_td(next_state)

      obs, state, template = next_obs, next_state, next_template

    # The rollout ends because the budget ran out, not because anything happened in the
    # world, so bootstrapping from the critic is right even though the next iteration
    # teleports away from it.
    ppo.compute_returns(template)

    ##
    # Train the referee: this rollout is the fake half, the opening window the real one.
    # Uses the actor as it stood during the rollout, so the rewards above are one update
    # behind.
    ##

    fake_obs_t = torch.cat(fake_obs)
    fake_actions_t = torch.cat(fake_actions)
    fake_next_obs_t = torch.cat(fake_next_obs)
    fake_done_t = torch.cat(fake_done)
    disc_loss_sum = 0.0
    fooled = 0.0
    caught = 0.0
    # How far apart the referee holds the two halves, split by which term did it.
    separation_by_reward = 0.0
    separation_by_likelihood = 0.0

    for _ in range(cfg.disc_epochs):
      idx = torch.randint(0, fake_obs_t.shape[0], (cfg.disc_batch_size,), device=device)
      policy_batch = DiscBatch(
        obs=fake_obs_t[idx],
        actions=fake_actions_t[idx],
        next_obs=fake_next_obs_t[idx],
        done=fake_done_t[idx],
        log_prob=_log_prob(actor, fake_obs_t[idx], fake_actions_t[idx]),
      )
      real = expert_buffer.sample(cfg.disc_batch_size)
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
      fooled += called_real(discriminator, policy_batch)
      caught += called_real(discriminator, expert_batch)
      with torch.no_grad():
        expert_reward, expert_likelihood = discriminator.split_logits(expert_batch)
        policy_reward, policy_likelihood = discriminator.split_logits(policy_batch)
        separation_by_reward += float(expert_reward.mean() - policy_reward.mean())
        separation_by_likelihood += float(
          policy_likelihood.mean() - expert_likelihood.mean()
        )

    losses = ppo.update()
    # Every update rather than once, because nothing else bounds it: PPO's entropy bonus
    # raises the std a little on every single one (see clamp_action_std).
    clamp_action_std(actor, cfg.action_max_std)

    if cfg.log_every and (
      iteration % cfg.log_every == 0 or iteration == cfg.num_iterations - 1
    ):
      # How to read this line.
      #
      # `imit` sits near zero by construction (the scaler centers it) and is not a score
      # on its own. What it must not do is sit at +-`reward_clip`, which would mean the
      # bound is binding and the reward has stopped varying.
      #
      # `fooled` (share of the bridge's transitions the referee calls real) against
      # `caught` (share of real ones it calls real) is the health of the game. Healthy
      # keeps caught high while fooled climbs; fooled near 0 with caught near 1 means the
      # referee has won outright, and both near 0.5 means it has stopped telling them
      # apart.
      #
      # `by_f` is the gap the referee opens, and it is now the whole verdict. `by_logp`
      # is the same gap in log-probability; it takes no part in any decision and is here
      # as a tripwire, because it was load-bearing in two earlier versions of this file
      # and broke both (see AIRLDiscriminator.logits).
      #
      # `drift` is the only number outside the game: mean distance from the target
      # skill's own states in standardized units, so it is what should actually fall.
      # Bounded by state clip times sqrt(obs_dim); pinned there means the bridge is
      # nowhere near the target's states, whatever the game reports.
      #
      # `term` is the share of steps that broke the episode and `redo` how many envs had
      # to be put back as a result. Both should fall.
      #
      # `bad` is envs whose observation came back non-finite. It should be exactly zero,
      # and nothing below this line is worth reading until it is.
      #
      # `std` is the actor's exploration. It should settle, not climb to `action_max_std`
      # and park there; if it does, the cause is `entropy_coef` and not the reward.
      with torch.no_grad():
        drift = fake_obs_t.norm(dim=-1).mean()
      denom = max(cfg.disc_epochs, 1)
      print(
        f"[bridge {iteration + 1:4d}/{cfg.num_iterations}] "
        f"imit {imitation_sum / cfg.steps_per_env:7.4f} | "
        f"disc_loss {disc_loss_sum / denom:6.4f} | "
        f"fooled {fooled / denom:5.1%} | "
        f"caught {caught / denom:5.1%} | "
        f"by_f {separation_by_reward / denom:+6.2f} | "
        f"by_logp {separation_by_likelihood / denom:+6.2f} | "
        f"drift {float(drift):6.2f} | "
        f"term {terminations / cfg.steps_per_env:5.1%} | "
        f"redo {restarts / num_envs:5.2f} | "
        f"bad {int(broken):4d} | "
        f"value {losses['value']:8.4f} | "
        f"std {action_std(actor):5.3f}"
      )


def _epsilon(iteration: int, cfg: SwitchPhase) -> float:
  frac = min(1.0, iteration / max(cfg.epsilon_decay_iterations, 1))
  return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def train_switch(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  actor: MLPModel,
  switch: SwitchQNetwork,
  target_skill_id: int,
  units: Units,
  succeeded: SuccessFn,
  label: str,
  cfg: SwitchPhase,
) -> None:
  """Phase 2: train `switch` in place by Double-DQN, against the frozen `actor`.

  The frozen actor drives while the decider chooses, per step, whether to hand over. Each
  env is judged `eval_steps` after *its own* hand-over: +1 if `succeeded` approves and the
  env never terminated, -1 otherwise, and -1 for never committing in time.

  The decider reads the same standardized state the actor does; the verdict does not, and
  must not, since judging a hand-over is privileged and reads the env directly.

  Before reading any number this prints: this sits on top of a frozen bridge and can only
  be as good as phase 1 left it. If the bridge never reaches a state the target skill can
  take over from, every hand-over fails and never committing also fails, and the numbers
  look like a broken decider when the broken thing is upstream. `good` near zero whatever
  `switched` does is that case.
  """

  device = env.device
  num_envs = env.num_envs
  entity = env.scene[exp.entity_name]
  manager_state = ManagerState(env)
  target_skill = exp.pool[target_skill_id]
  view = exp.view
  obs_dim = view.dim

  env.reset()
  dqn = DoubleDQN(switch, cfg.gamma, cfg.learning_rate, device)
  # One buffer per decision rather than one shared: an env produces at most one "switch"
  # row per window but a "stay" row on every step it did not.
  replay_fields: dict[str, tuple[int, ...]] = {
    "obs": (obs_dim,),
    "reward": (),
    "next_obs": (obs_dim,),
    "done": (),
  }
  switch_replay = RingBuffer(cfg.replay_capacity, device, replay_fields)
  stay_replay = RingBuffer(cfg.replay_capacity, device, replay_fields)
  half_batch = cfg.batch_size // 2

  print(
    f"\n=== switch -> '{target_skill.name}': Double-DQN on the '{label}' signal, "
    f"{cfg.num_iterations} iterations over {num_envs} envs ==="
  )

  interrupts = _prepare_interrupts(
    env, exp, target_skill_id, manager_state, cfg.num_interrupts
  )

  total_steps = cfg.max_transition_steps + cfg.eval_steps
  last_loss = float("nan")

  for iteration in range(cfg.num_iterations):
    indices = interrupts.sample_indices(num_envs, device)
    obs = restore_interrupts(env, entity, interrupts, manager_state, indices)
    # An aggregate rate blurs easy and hard cuts together. How far through its own range
    # the source skill was cut is the one difficulty proxy available without knowing the
    # experiment, and the fraction rather than the raw step so it stays comparable.
    hard = interrupts.cut_frac[indices] >= 0.5
    # The target skill is *not* reset here. A skill with memory is told it is starting
    # when control reaches it, which is at the hand-over below (`MetaPolicy.engage` states
    # the same contract for inference). The jump is why it matters: its reset re-anchors
    # its reference clip to the robot and rewinds it, and the clip advances every step
    # regardless of who is driving, so resetting here hands the jump a reference already
    # seconds in and pinned where the robot stood before the bridge moved it.
    epsilon = _epsilon(iteration, cfg)

    switched = torch.zeros(num_envs, dtype=torch.bool, device=device)
    switch_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    switch_obs = torch.zeros(num_envs, obs_dim, device=device)
    # A termination anywhere in the window condemns that env: the env auto-resets the
    # moment it fires, so whatever is read afterwards belongs to a fresh episode. Unlike
    # phase 1, nothing is restored -- the termination is the measurement.
    failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    verdict = torch.zeros(num_envs, dtype=torch.bool, device=device)
    judged = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # `failed` as it stood when each env was judged, so the reported rate is the one the
    # verdict actually saw. Anything that breaks later shows up as `late`.
    broken = torch.zeros(num_envs, dtype=torch.bool, device=device)
    stay_obs, stay_reward, stay_next_obs, stay_done = [], [], [], []

    for t in range(total_steps):
      state = units.state.standardize(view(as_tensor(obs["actor"])))
      driving = switched.clone()  # The target skill is at the controls in these envs

      # Both candidates are produced in raw units before they are mixed: the skill emits
      # raw actions directly, the bridge emits normalized ones.
      with torch.no_grad():
        bridge_actions = units.action.denormalize(actor(obs_td(state)))
      target_actions = target_skill.act(obs, driving)
      actions = torch.where(driving.unsqueeze(-1), target_actions, bridge_actions)

      alive_before = ~failed
      if t < cfg.max_transition_steps:
        decisions = dqn.act(state, epsilon)
        new_switch = (decisions == 1) & ~driving & alive_before
      else:
        new_switch = torch.zeros(num_envs, dtype=torch.bool, device=device)

      next_obs, _, terminated, _, _ = env.step(actions)
      next_state = units.state.standardize(view(as_tensor(next_obs["actor"])))
      failed = failed | terminated

      if new_switch.any():
        switch_obs[new_switch] = state[new_switch]
        switch_step[new_switch] = t
        switched = switched | new_switch
        # Control reaches the target skill on the next step, so this is the moment it is
        # starting: whatever memory it keeps is set up against the state the bridge
        # actually left the robot in.
        target_skill.reset(new_switch)

      # Staying is only a decision while the window is open. It ends the episode (-1)
      # either when the bridge crashed the robot or when the window ran out.
      if t < cfg.max_transition_steps:
        record_stay = ~driving & ~new_switch & alive_before
        terminal = terminated | (t == cfg.max_transition_steps - 1)
        if record_stay.any():
          stay_obs.append(state[record_stay])
          stay_reward.append(torch.where(terminal, -1.0, 0.0)[record_stay])
          stay_next_obs.append(next_state[record_stay])
          stay_done.append(terminal[record_stay].float())

      # Judge each env `eval_steps` after its own hand-over, not once at the end for
      # everybody: an env that committed on the last step would be graded on one step.
      due = switched & ~judged & (switch_step + cfg.eval_steps == t)
      if due.any():
        verdict = torch.where(due, succeeded(env) & ~failed, verdict)
        broken = torch.where(due, failed, broken)
        judged = judged | due

      obs = next_obs

    if switched.any():
      n = int(switched.sum())
      switch_replay.add(
        obs=switch_obs[switched],
        reward=torch.where(verdict, 1.0, -1.0)[switched],
        next_obs=switch_obs[switched],  # Unused: done=1 zeroes the bootstrap term
        done=torch.ones(n, device=device),
      )
    if stay_obs:
      stay_replay.add(
        obs=torch.cat(stay_obs),
        reward=torch.cat(stay_reward),
        next_obs=torch.cat(stay_next_obs),
        done=torch.cat(stay_done),
      )

    warmed_up = (
      len(switch_replay) >= half_batch
      and len(stay_replay) >= half_batch
      and iteration >= cfg.warmup_iterations
    )
    if warmed_up:
      for _ in range(cfg.updates_per_iteration):
        chose_switch = switch_replay.sample(half_batch)
        chose_stay = stay_replay.sample(half_batch)
        # The action taken is implied by which buffer a row came from, so it is not
        # stored: ones for the switch half, zeros for the stay half.
        batch_obs = torch.cat([chose_switch["obs"], chose_stay["obs"]])
        batch_action = torch.cat(
          [
            torch.ones(half_batch, dtype=torch.long, device=device),
            torch.zeros(half_batch, dtype=torch.long, device=device),
          ]
        )
        batch_reward = torch.cat([chose_switch["reward"], chose_stay["reward"]])
        batch_next_obs = torch.cat([chose_switch["next_obs"], chose_stay["next_obs"]])
        batch_done = torch.cat([chose_switch["done"], chose_stay["done"]])
        last_loss = float(
          dqn.update(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
        )
      if iteration % cfg.update_target_every == 0:
        dqn.sync_target()

    if cfg.log_every and (
      iteration % cfg.log_every == 0 or iteration == cfg.num_iterations - 1
    ):
      # `good` counts only the envs that handed over, and only if they survived to
      # judgement. `hard` is the same over the later half of each source's cut range.
      # `overall` folds in the envs that never committed, so it is what the decider is
      # really maximizing. `switched` near 1.0 with a poor `good` means it hands over
      # indiscriminately; near 0.0 means it never commits and just eats the -1.
      #
      # `broke` is the termination rate the verdict saw. `late` is what broke *after*
      # being judged fine, which nothing is penalized for: a `late` far from zero means
      # `eval_steps` ends before the failure has finished developing.
      mean_step = (
        float(switch_step[switched].float().mean())
        if bool(switched.any())
        else float("nan")
      )
      late = float((failed & ~broken & switched).float().mean())
      print(
        f"[switch {iteration + 1:5d}/{cfg.num_iterations}] "
        f"eps {epsilon:4.2f} | "
        f"switched {float(switched.float().mean()):5.1%} @ step {mean_step:5.1f} | "
        f"good {_rate(verdict, switched):5.1%} "
        f"(hard {_rate(verdict, switched & hard):5.1%}) | "
        f"broke {float(broken.float().mean()):5.1%} (late {late:5.1%}) | "
        f"overall {float(torch.where(switched & verdict, 1.0, -1.0).mean()):+5.2f} | "
        f"loss {last_loss:8.4f} | "
        f"replay {len(switch_replay)}/{len(stay_replay)}"
        + ("" if warmed_up else " (warmup)")
      )


def train_bridging(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  meta: Arch1,
  cfg: BridgeTraining,
  success_fns: Mapping[int, SuccessFn],
  label: str,
) -> None:
  """Both phases, for every target skill. Shared by arch_1 and arch_2.

  The two differ only in `success_fns`, which is what phase 2 learns from.
  """
  # Measured once from the whole pool and shared by every target: a transition into one
  # skill starts wherever another left the robot, so the numbers span the pool; and
  # shared so a comparison between two targets is a comparison of behavior, not units.
  # `meta` keeps them because they are as much a part of a trained bridge as its weights.
  units = measure_spaces(
    env, exp.pool, exp.view, exp.windows, clip=cfg.bridge.state_clip
  )
  meta.adopt_units(units)

  for target_id in range(len(exp.pool)):
    actor = meta.actors[target_id]
    train_bridge(env, exp, actor, target_id, units, cfg.bridge)

    # Freeze the actor before training the decider on top of it.
    actor.eval()
    for p in actor.parameters():
      p.requires_grad_(False)

    train_switch(
      env,
      exp,
      actor,
      meta.switches[target_id],
      target_id,
      units,
      success_fns[target_id],
      label,
      cfg.switch,
    )


def train(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  meta: Arch1,
  cfg: BridgeTraining,
  success_fns: Mapping[int, SuccessFn],
) -> Arch1:
  """arch_1: the hand-over decider learns from a hand-written success test."""
  train_bridging(env, exp, meta, cfg, success_fns, label="oracle")
  return meta
