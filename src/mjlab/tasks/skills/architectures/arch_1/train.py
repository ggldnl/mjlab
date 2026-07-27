"""The bridge family's two-phase training, and arch_1's use of it.

For each target skill in the pool, in order:
- collect that skill's opening window (the "real" data to imitate) and interrupt states
  harvested from the *other* skills (where the bridge gets dropped in);
- phase 1: train that skill's bridge actor by AIRL + PPO (move like the target);
- phase 2: freeze the actor and train that skill's switch-decider by Double-DQN
  (when to hand over), against an outcome source.

Phase 1 is the same in every architecture in this family. Phase 2 is the same machinery
too, but the signal it learns from is what the architectures differ on, so
`train_bridging` takes one `OutcomeSource` per target skill and arch_2 and arch_3 reuse
it with a different one. `train` here is arch_1's own entry point: the outcome is a
hand-written oracle.

Each skill's windows come from the experiment's `WindowPlan`, so a skill that needs a
longer look at itself gets one without the others having to change. The training budget
comes from a `BridgeTraining` the experiment declares, so a cart-pole and a humanoid can
want very different numbers without either being hard-coded here.

A note on judging a hand-over, because it is easy to get silently wrong. The env
auto-resets an environment the moment it terminates, so by the end of a hand-over window
a robot that fell over is already upright again, and anything read from live state then
reports that failure as a success. Phase 2 therefore latches terminations as they happen
and reads its outcome source a fixed number of steps after *each* env's own hand-over,
not once at the end for everybody.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_1 import Arch1
from mjlab.tasks.skills.architectures.arch_1.config import (
  BridgePhase,
  BridgeTraining,
  SwitchPhase,
)
from mjlab.tasks.skills.architectures.arch_1.networks import (
  OBS_GROUPS,
  AIRLDiscriminator,
  DiscBatch,
  DoubleDQN,
  SwitchQNetwork,
)
from mjlab.tasks.skills.architectures.arch_1.outcomes import (
  OracleOutcome,
  OutcomeSource,
  SuccessFn,
)
from mjlab.tasks.skills.buffers import RingBuffer
from mjlab.tasks.skills.skill import Skill, SkillPool
from mjlab.tasks.skills.windows import (
  ManagerState,
  WindowPlan,
  as_tensor,
  collect_interrupts,
  collect_opening,
  restore_interrupts,
  verify_restore,
)


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


def _rate(outcome: torch.Tensor, mask: torch.Tensor) -> float:
  """Share of the envs `mask` selects whose hand-over came out well."""
  return float(outcome[mask].float().mean()) if bool(mask.any()) else float("nan")


def train_bridge(
  env: ManagerBasedRlEnv,
  actor: MLPModel,
  target_skill: Skill,
  target_skill_id: int,
  pool: SkillPool,
  entity_name: str,
  plan: WindowPlan,
  cfg: BridgePhase,
) -> None:
  """Phase 1: train `actor` in place by AIRL + PPO to move like `target_skill`.

  The referee (discriminator) is trained to tell the actor's rollouts from the target
  skill's own opening-window rollouts; the actor is trained (PPO) to fool it, using the
  referee's confidence as reward. A throwaway critic is built here for PPO and discarded
  (only the actor is kept, in the meta policy).
  """

  device = env.device
  num_envs = env.num_envs
  action_dim = env.action_manager.total_action_dim
  entity = env.scene[entity_name]
  manager_state = ManagerState(env)
  spec = plan[target_skill]

  obs, _ = env.reset()
  obs_td = TensorDict(obs, batch_size=[num_envs])
  obs_dim = as_tensor(obs_td["actor"]).shape[-1]

  critic = MLPModel(
    obs_td,
    OBS_GROUPS,
    "critic",
    1,
    hidden_dims=cfg.critic_hidden_dims,
    activation="tanh",
  ).to(device)
  storage = RolloutStorage(
    "rl", num_envs, cfg.steps_per_env, obs_td, [action_dim], device
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
    device=device,
  )

  print(
    f"\n=== bridge -> '{target_skill.name}': AIRL + PPO, "
    f"{cfg.num_iterations} iterations over {num_envs} envs ==="
  )
  print(f"[collect] rolling '{target_skill.name}' for its opening window...")
  opening = collect_opening(env, target_skill, target_skill_id, spec, cfg.num_windows)
  expert = opening.transitions()
  expert_buffer = RingBuffer(
    int(expert["obs"].shape[0]),
    device,
    {"obs": (obs_dim,), "action": (action_dim,), "next_obs": (obs_dim,), "done": ()},
  )
  expert_buffer.add(**expert)
  print(
    f"[collect] {len(opening)} opening windows of {spec.opening} steps, "
    f"{len(expert_buffer)} usable transitions"
  )

  print(f"[collect] rolling {len(pool) - 1} other skill(s) for interrupt states...")
  interrupts = collect_interrupts(
    env, pool, entity_name, target_skill_id, plan, cfg.num_interrupts
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

  # The discriminator reads raw observations and actions, whose scales are whatever the
  # experiment happens to use (the diffdrive commands wheel velocities in the tens of
  # rad/s). Standardizing against the expert data keeps its first layer in a sane range
  # instead of saturating on the first batch.
  discriminator = AIRLDiscriminator(
    obs_dim, action_dim, cfg.disc_hidden_dims, cfg.disc_gamma
  ).to(device)
  discriminator.fit_normalization(expert["obs"], expert["action"])
  disc_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=cfg.disc_learning_rate
  )

  # How far outside the target skill's own states the bridge ends up, in expert standard
  # deviations. This is the one phase-1 number that is not part of the adversarial game,
  # so it is the one neither side can game.
  expert_mean = expert["obs"].mean(dim=0)
  expert_std = expert["obs"].std(dim=0).clamp_min(1e-3)

  for iteration in range(cfg.num_iterations):
    obs = restore_interrupts(env, entity, interrupts, manager_state)
    obs_td = TensorDict(obs, batch_size=[num_envs])
    ppo.train_mode()
    reward_sum = 0.0
    terminations = 0.0
    fake_obs, fake_actions, fake_next_obs, fake_done = [], [], [], []

    for _ in range(cfg.steps_per_env):
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
    # skill's own opening window the "real" half. Uses the actor as it stood during the
    # rollout, so the rewards fed to PPO above are one update behind.
    fake_obs_t = torch.cat(fake_obs)
    fake_actions_t = torch.cat(fake_actions)
    fake_next_obs_t = torch.cat(fake_next_obs)
    fake_done_t = torch.cat(fake_done)
    disc_loss_sum = 0.0
    fooled = 0.0
    caught = 0.0
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
      fooled += _called_real(discriminator, policy_batch)
      caught += _called_real(discriminator, expert_batch)

    losses = ppo.update()

    if cfg.log_every and (
      iteration % cfg.log_every == 0 or iteration == cfg.num_iterations - 1
    ):
      # `airl_reward` is softplus(logit): it falls as the referee gets better and rises
      # as the bridge fools it, so it is not a task score. Read `fooled` (the share of
      # the bridge's own transitions the referee calls real) against `caught` (the share
      # of real ones it calls real). A healthy game keeps caught high while fooled
      # climbs; fooled near 0 with caught near 1 means the referee has won outright and
      # the bridge has no gradient left to climb, and both near 0.5 means it has stopped
      # telling them apart at all. `drift` is the only number here outside the game:
      # mean distance from the target skill's own states, in expert sigmas, and it is
      # what should actually fall if the bridge is learning.
      with torch.no_grad():
        drift = ((fake_obs_t - expert_mean) / expert_std).norm(dim=-1).mean()
      denom = max(cfg.disc_epochs, 1)
      print(
        f"[bridge {iteration + 1:4d}/{cfg.num_iterations}] "
        f"airl_reward {reward_sum / cfg.steps_per_env:7.4f} | "
        f"disc_loss {disc_loss_sum / denom:6.4f} | "
        f"fooled {fooled / denom:5.1%} | "
        f"caught {caught / denom:5.1%} | "
        f"drift {float(drift):6.2f} | "
        f"term {terminations / cfg.steps_per_env:5.1%} | "
        f"value {losses['value']:8.4f} | "
        f"entropy {losses['entropy']:6.3f}"
      )


def _epsilon(iteration: int, cfg: SwitchPhase) -> float:
  frac = min(1.0, iteration / max(cfg.epsilon_decay_iterations, 1))
  return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def train_switch(
  env: ManagerBasedRlEnv,
  actor: MLPModel,
  switch: SwitchQNetwork,
  target_skill: Skill,
  target_skill_id: int,
  pool: SkillPool,
  entity_name: str,
  outcome: OutcomeSource,
  plan: WindowPlan,
  cfg: SwitchPhase,
) -> None:
  """Phase 2: train `switch` in place by Double-DQN, against the frozen `actor`.

  The frozen actor drives the robot while the switch-decider chooses, per step, whether
  to hand over. Each env is judged `cfg.eval_steps` after *its own* hand-over: +1 if
  `outcome` approves and the env never terminated, -1 otherwise, and -1 for never
  committing within `cfg.max_transition_steps`.
  """

  device = env.device
  num_envs = env.num_envs
  entity = env.scene[entity_name]
  manager_state = ManagerState(env)

  obs, _ = env.reset()
  obs_dim = as_tensor(obs["actor"]).shape[-1]
  dqn = DoubleDQN(switch, cfg.gamma, cfg.learning_rate, device)
  replay = RingBuffer(
    cfg.replay_capacity,
    device,
    {"obs": (obs_dim,), "action": (), "reward": (), "next_obs": (obs_dim,), "done": ()},
    {"action": torch.long},
  )

  print(
    f"\n=== switch -> '{target_skill.name}': Double-DQN on the '{outcome.label}' "
    f"signal, {cfg.num_iterations} iterations over {num_envs} envs ==="
  )
  outcome.prepare(env, target_skill, cfg.eval_steps)

  print(f"[collect] rolling {len(pool) - 1} other skill(s) for interrupt states...")
  interrupts = collect_interrupts(
    env, pool, entity_name, target_skill_id, plan, cfg.num_interrupts
  )
  print(f"[collect] {len(interrupts)} interrupt states")

  max_error, within = verify_restore(env, entity, interrupts, manager_state)
  print(
    f"[verify] interrupt restore round-trip: max obs error {max_error:.3e}, "
    f"{within:.1%} exact"
  )

  all_envs = torch.ones(num_envs, dtype=torch.bool, device=device)
  total_steps = cfg.max_transition_steps + cfg.eval_steps
  last_loss = float("nan")

  for iteration in range(cfg.num_iterations):
    indices = interrupts.sample_indices(num_envs, device)
    obs = restore_interrupts(env, entity, interrupts, manager_state, indices)
    # An aggregate rate blurs easy and hard cuts together. How far through its own
    # hand-over range a source skill was cut is the one difficulty proxy available
    # without knowing the experiment, and the fraction rather than the raw step so it
    # stays comparable across skills with different ranges.
    hard = interrupts.cut_frac[indices] >= 0.5
    target_skill.reset(all_envs)
    outcome.begin(num_envs, device)
    epsilon = _epsilon(iteration, cfg)

    switched = torch.zeros(num_envs, dtype=torch.bool, device=device)
    switch_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    switch_obs = torch.zeros(num_envs, obs_dim, device=device)
    # A termination anywhere in the window condemns that env: the env auto-resets the
    # moment it fires, so whatever is read afterwards belongs to a fresh episode and
    # says nothing about how the hand-over went.
    failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    verdict = torch.zeros(num_envs, dtype=torch.bool, device=device)
    judged = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # `failed` as it stood when each env was judged, so the reported termination rate is
    # the one the verdict actually saw. Anything that breaks later shows up as `late`.
    broken = torch.zeros(num_envs, dtype=torch.bool, device=device)
    stay_obs, stay_reward, stay_next_obs, stay_done = [], [], [], []

    for t in range(total_steps):
      state = as_tensor(obs["actor"])
      driving = switched.clone()  # The target skill is at the controls in these envs

      with torch.no_grad():
        bridge_actions = actor(TensorDict({"actor": state}, batch_size=[num_envs]))
      target_actions = target_skill.act(obs, driving)
      actions = torch.where(driving.unsqueeze(-1), target_actions, bridge_actions)

      alive_before = ~failed
      if t < cfg.max_transition_steps:
        decisions = dqn.act(state, epsilon)
        new_switch = (decisions == 1) & ~driving & alive_before
      else:
        new_switch = torch.zeros(num_envs, dtype=torch.bool, device=device)

      next_obs, reward, terminated, _, _ = env.step(actions)
      next_state = as_tensor(next_obs["actor"])
      failed = failed | terminated
      # Only what the target skill earned counts, so this accrues from the step after
      # the hand-over onwards.
      outcome.record(reward, driving.float())

      if new_switch.any():
        switch_obs[new_switch] = state[new_switch]
        switch_step[new_switch] = t
        switched = switched | new_switch

      # Staying is only a decision while the window is open. It ends the episode (with
      # -1) either when the bridge itself crashed the robot or when the window ran out
      # without ever committing.
      if t < cfg.max_transition_steps:
        record_stay = ~driving & ~new_switch & alive_before
        terminal = terminated | (t == cfg.max_transition_steps - 1)
        if record_stay.any():
          stay_obs.append(state[record_stay])
          stay_reward.append(torch.where(terminal, -1.0, 0.0)[record_stay])
          stay_next_obs.append(next_state[record_stay])
          stay_done.append(terminal[record_stay].float())

      # Judge each env `eval_steps` after its own hand-over, not once at the end for
      # everybody: an env that committed on the last step would otherwise be graded on a
      # single step of the target skill.
      due = switched & ~judged & (switch_step + cfg.eval_steps == t)
      if due.any():
        verdict = torch.where(due, outcome.verdict(env) & ~failed, verdict)
        broken = torch.where(due, failed, broken)
        judged = judged | due

      obs = next_obs

    if switched.any():
      n = int(switched.sum())
      replay.add(
        obs=switch_obs[switched],
        action=torch.ones(n, device=device),
        reward=torch.where(verdict, 1.0, -1.0)[switched],
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

    warmed_up = len(replay) >= cfg.batch_size and iteration >= cfg.warmup_iterations
    if warmed_up:
      for _ in range(cfg.updates_per_iteration):
        batch = replay.sample(cfg.batch_size)
        last_loss = float(
          dqn.update(
            batch["obs"],
            batch["action"].long(),
            batch["reward"],
            batch["next_obs"],
            batch["done"],
          )
        )
      if iteration % cfg.update_target_every == 0:
        dqn.sync_target()

    if cfg.log_every and (
      iteration % cfg.log_every == 0 or iteration == cfg.num_iterations - 1
    ):
      # `good` counts only the envs that actually handed over, and only counts one good
      # if that env survived to its judgement. `hard` is the same over the later half of
      # each source skill's cut range. `overall` folds in the envs that never committed,
      # so it is the number the decider is really maximizing. `switched` near 1.0 with a
      # poor `good` means it hands over indiscriminately; near 0.0 means it never
      # commits and just eats the -1.
      #
      # `broke` is the termination rate the verdict saw. `late` is what broke *after*
      # being judged fine, which nothing is being penalized for: a `late` that is not
      # near zero means `eval_steps` ends before the failure has finished developing,
      # and the decider is being told those hand-overs were good.
      mean_step = (
        float(switch_step[switched].float().mean())
        if bool(switched.any())
        else float("nan")
      )
      extra = outcome.summary()
      late = float((failed & ~broken & switched).float().mean())
      print(
        f"[switch {iteration + 1:5d}/{cfg.num_iterations}] "
        f"eps {epsilon:4.2f} | "
        f"switched {float(switched.float().mean()):5.1%} @ step {mean_step:5.1f} | "
        f"good {_rate(verdict, switched):5.1%} "
        f"(hard {_rate(verdict, switched & hard):5.1%}) | "
        f"broke {float(broken.float().mean()):5.1%} (late {late:5.1%}) | "
        f"overall {float(torch.where(switched & verdict, 1.0, -1.0).mean()):+5.2f} | "
        + (f"{extra} | " if extra else "")
        + f"loss {last_loss:8.4f} | replay {len(replay)}"
        + ("" if warmed_up else " (warmup)")
      )


def train_bridging(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  meta: Arch1,
  outcomes: Mapping[int, OutcomeSource],
  plan: WindowPlan,
  cfg: BridgeTraining,
) -> None:
  """Train every bridge and switch-decider `meta` holds, in place.

  Shared by every architecture built on a per-target bridge: phase 1 is identical, and
  phase 2 differs only in the `OutcomeSource` handed in per target skill.
  """
  plan.check(pool)

  for target_id in range(len(pool)):
    target_skill = pool[target_id]
    actor = meta.actors[target_id]
    train_bridge(
      env, actor, target_skill, target_id, pool, entity_name, plan, cfg.bridge
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
      outcomes[target_id],
      plan,
      cfg.switch,
    )


def train(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  meta: Arch1,
  success_fns: Mapping[int, SuccessFn],
  windows: WindowPlan,
  training: BridgeTraining,
) -> Arch1:
  """arch_1: the switch-decider learns from a hand-written success oracle."""
  outcomes: Mapping[int, OutcomeSource] = {
    i: OracleOutcome(fn) for i, fn in success_fns.items()
  }
  train_bridging(env, pool, entity_name, meta, outcomes, windows, training)
  return meta
