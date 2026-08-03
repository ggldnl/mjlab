"""The bridge family's two-phase training, and arch_1's use of it.

For each target skill in the pool, in order:
- collect that skill's opening window (the "real" data to imitate) and interrupt states
  harvested from the *other* skills (where the bridge gets dropped in);
- phase 1: train that skill's bridge actor by AIRL + PPO (move like the target);
- phase 2: freeze the actor and train that skill's switch-decider by Double-DQN
  (when to hand over), against an outcome source.

Phase 1 is the same in every architecture in this family. Phase 2 is the same machinery
too, but the signal it learns from is what the architectures differ on, so
`train_bridging` takes one `OutcomeSource` per target skill and arch_2 reuse it
with a different one. `train` here is arch_1's own entry point: the outcome is a
hand-written oracle.

Each skill's windows come from the experiment's `WindowPlan`, so a skill that needs a
longer look at itself gets one without the others having to change. The training budget
comes from a `BridgeTraining` the experiment declares, so a cart-pole and a humanoid can
want very different numbers without either being hard-coded here.

Everything the bridge sees, is judged on and is rewarded for goes through two shared
conversions, and going through *the same* ones is the point:

- the experiment's `StateView` (see view.py) picks which channels of the observation the
  bridging machinery is allowed to work on at all;
- the run's `StateSpace` and `ActionSpace` (see spaces.py) put those channels, and the
  actions, into units of roughly unit size.

The recorded windows, the discriminator, the actor and the switch-decider all read the
output of that pair. AIRL rewards the actor for producing what the discriminator cannot
separate, so a channel one of them sees and the other does not is either an unreachable
target or a free giveaway, and a channel one of them reads in different units than the
other is the same problem in a form that is much harder to see.

Two things about resets, because both are easy to get silently wrong.

The env auto-resets an environment the moment it terminates, and the reset it performs
is the *arena's*, not a bridging one. So an env that tips over during phase 1 reappears
at the arena's start state and keeps producing rollout data from there. That data is not
the bridge's behavior, and worse, for a target skill that also starts from rest it is
nearly indistinguishable from the expert data, so the same states end up labelled real
and fake at once. Phase 1 therefore puts a crashed env straight back at a fresh
interrupt state and drops the transition that crossed the reset.

Phase 2 must not do that, because there a termination *is* the measurement. It latches
terminations as they happen and reads its outcome source a fixed number of steps after
*each* env's own hand-over, not once at the end for everybody, since by then a robot
that fell over is already upright again in a fresh episode.
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
from mjlab.tasks.skills.architectures.arch_1.networks import (
  OBS_GROUPS,
  AIRLDiscriminator,
  DiscBatch,
  DoubleDQN,
  SwitchQNetwork,
  action_std,
  bridge_obs_td,
  clamp_action_std,
  set_action_std,
)
from mjlab.tasks.skills.architectures.arch_1.outcomes import (
  OracleOutcome,
  OutcomeSource,
  SuccessFn,
)
from mjlab.tasks.skills.architectures.arch_1.spaces import (
  ActionSpace,
  StateSpace,
  measure_spaces,
)
from mjlab.tasks.skills.buffers import RingBuffer
from mjlab.tasks.skills.skill import Skill, SkillPool
from mjlab.tasks.skills.view import StateView
from mjlab.tasks.skills.windows import (
  InterruptSet,
  ManagerState,
  WindowPlan,
  as_tensor,
  collect_interrupts,
  collect_opening,
  restore_interrupts,
  verify_restore,
)


def _log_prob(
  actor: MLPModel, obs: torch.Tensor, normalized_actions: torch.Tensor
) -> torch.Tensor:
  """Log-probability of `normalized_actions` under the actor's distribution at `obs`.

  `obs` is already the standardized state the actor was built on, and the actions are
  already normalized: the actor's distribution lives in normalized units, so handing it
  raw ones would produce a number off by orders of magnitude in exactly the place AIRL
  is most sensitive to it (see `AIRLDiscriminator.split_logits`).

  Detached: AIRL treats log pi as a known constant inside the discriminator logit, so
  the discriminator's loss must not push gradients back into the policy.
  """
  with torch.no_grad():
    actor(
      bridge_obs_td(obs), stochastic_output=True
    )  # Updates the distribution's params
    return actor.get_output_log_prob(normalized_actions).squeeze(-1)


def _called_real(discriminator: AIRLDiscriminator, batch: DiscBatch) -> float:
  """Fraction of a batch the discriminator calls "expert" (logit > 0)."""
  with torch.no_grad():
    return float((discriminator.logits(batch) > 0).float().mean())


def _rate(outcome: torch.Tensor, mask: torch.Tensor) -> float:
  """Share of the envs `mask` selects whose hand-over came out well."""
  return float(outcome[mask].float().mean()) if bool(mask.any()) else float("nan")


def _prepare_interrupts(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  target_skill_id: int,
  plan: WindowPlan,
  manager_state: ManagerState,
  num_interrupts: int,
) -> InterruptSet:
  """Harvest the states a bridge gets dropped into, and prove they can be restored."""
  entity = env.scene[entity_name]
  print(f"[collect] rolling {len(pool) - 1} other skill(s) for interrupt states...")
  interrupts = collect_interrupts(
    env, pool, entity_name, target_skill_id, plan, num_interrupts
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
  actor: MLPModel,
  target_skill: Skill,
  target_skill_id: int,
  pool: SkillPool,
  entity_name: str,
  view: StateView,
  action_space: ActionSpace,
  state_space: StateSpace,
  plan: WindowPlan,
  cfg: BridgePhase,
) -> None:
  """Phase 1: train `actor` in place by AIRL + PPO to move like `target_skill`.

  The referee (discriminator) is trained to tell the actor's rollouts from the target
  skill's own opening-window rollouts; the actor is trained (PPO) to fool it, using the
  referee's confidence as reward. A throwaway critic is built here for PPO and discarded
  (only the actor is kept, in the meta policy).

  Both halves of that comparison go through `view` and then through `state_space` /
  `action_space`, so what the referee is asked to separate is the behavior, in units it
  can actually weigh, and not the task context or the choice of units each half happened
  to arrive in. See view.py and spaces.py for why each of those decides whether phase 1
  learns anything at all.
  """

  device = env.device
  num_envs = env.num_envs
  action_dim = env.action_manager.total_action_dim
  entity = env.scene[entity_name]
  manager_state = ManagerState(env)
  spec = plan[target_skill]
  obs_dim = view.dim

  print(
    f"\n=== bridge -> '{target_skill.name}': AIRL + PPO, "
    f"{cfg.num_iterations} iterations over {num_envs} envs ==="
  )
  print(f"[view] matching on {view.label}")

  ##
  # The example to imitate, in the bridge's own units.
  ##

  print(f"[collect] rolling '{target_skill.name}' for its opening window...")
  opening = collect_opening(
    env, target_skill, target_skill_id, spec, cfg.num_windows, view=view
  )
  raw_expert = opening.transitions()
  expert = {
    "obs": state_space.standardize(raw_expert["obs"]),
    "action": action_space.normalize(raw_expert["action"]),
    "next_obs": state_space.standardize(raw_expert["next_obs"]),
    "done": raw_expert["done"],
  }
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

  ##
  # Where the bridge gets dropped in.
  ##

  interrupts = _prepare_interrupts(
    env, pool, entity_name, target_skill_id, plan, manager_state, cfg.num_interrupts
  )

  ##
  # The two players.
  ##

  obs = restore_interrupts(env, entity, interrupts, manager_state)
  obs_td = bridge_obs_td(state_space.standardize(view(as_tensor(obs["actor"]))))

  # The actor was built before any training config existed, so its exploration is set
  # here from the one the experiment declared rather than from the constructor's guess.
  set_action_std(actor, cfg.action_init_std)

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
    schedule=cfg.ppo_schedule,
    device=device,
  )

  discriminator = AIRLDiscriminator(
    obs_dim, action_dim, cfg.disc_hidden_dims, cfg.disc_gamma
  ).to(device)
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
    state = state_space.standardize(view(as_tensor(obs["actor"])))
    obs_td = bridge_obs_td(state)
    ppo.train_mode()

    imitation_sum = 0.0
    terminations = 0.0
    restarts = 0.0
    # What the bridge produced this rollout, kept for the referee's "fake" half.
    fake_obs, fake_actions, fake_next_obs, fake_done = [], [], [], []

    for _ in range(cfg.steps_per_env):
      # The actor thinks in normalized units; the simulator wants raw ones. This pair of
      # lines is the only place the two meet.
      normalized_action = ppo.act(obs_td)
      raw_action = action_space.denormalize(normalized_action)

      next_obs, _, terminated, time_out, extras = env.step(raw_action)
      # `done` is exactly "mjlab auto-reset this env inside the step above", which is
      # what the rest of the loop has to work around.
      done = terminated | time_out
      terminations += float(terminated.float().mean())

      next_state = state_space.standardize(view(as_tensor(next_obs["actor"])))
      next_obs_td = bridge_obs_td(next_state)

      log_prob = ppo.transition.actions_log_prob
      assert log_prob is not None
      batch = DiscBatch(
        obs=state,
        actions=normalized_action,
        next_obs=next_state,
        done=done,
        log_prob=log_prob.squeeze(-1),
      )

      # The reward has two clearly separate parts, and they have opposite signs on
      # purpose. The imitation part is softplus(...) and therefore always positive: it
      # says how convinced the referee is that this looked like the real skill, and it
      # can never say that something was bad. The crash penalty is the only thing in
      # phase 1 that can.
      with torch.no_grad():
        imitation_reward = discriminator.reward(batch)
      crash_penalty = cfg.termination_penalty * terminated.float()
      reward = imitation_reward - crash_penalty
      imitation_sum += float(imitation_reward.mean())

      # A transition whose step ended in a reset pairs a state the bridge reached with
      # one it did not: the env is already back at the arena's start state by now. Those
      # rows would teach the referee that the arena's start state is something the bridge
      # produces, which for a target skill that also starts from rest is the same state
      # it is being shown as expert data.
      kept = ~done
      if bool(kept.any()):
        fake_obs.append(state[kept])
        fake_actions.append(normalized_action[kept])
        fake_next_obs.append(next_state[kept])
        fake_done.append(done[kept])

      step_extras = dict(extras)
      step_extras["time_outs"] = time_out
      ppo.process_env_step(next_obs_td, reward, done, step_extras)

      # Put the envs the simulator just reset back where a bridge actually lives. Without
      # this the rest of the rollout is collected from the arena's start distribution,
      # which is not a state any bridge is ever dropped into.
      if bool(done.any()):
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        restarts += float(done_ids.numel())
        next_obs = restore_interrupts(
          env, entity, interrupts, manager_state, env_ids=done_ids
        )
        next_state = state_space.standardize(view(as_tensor(next_obs["actor"])))
        next_obs_td = bridge_obs_td(next_state)

      obs, state, obs_td = next_obs, next_state, next_obs_td

    # The rollout ends because the budget ran out, not because anything happened in the
    # world, so the critic's estimate of the last state is the right thing to bootstrap
    # from even though the next iteration teleports away from it.
    ppo.compute_returns(obs_td)

    ##
    # Train the referee: this rollout's transitions are the "fake" half, the target
    # skill's own opening window the "real" half. Uses the actor as it stood during the
    # rollout, so the rewards fed to PPO above are one update behind.
    ##

    fake_obs_t = torch.cat(fake_obs)
    fake_actions_t = torch.cat(fake_actions)
    fake_next_obs_t = torch.cat(fake_next_obs)
    fake_done_t = torch.cat(fake_done)
    disc_loss_sum = 0.0
    fooled = 0.0
    caught = 0.0
    # How far apart the referee holds the two halves, split by which term did it. The
    # referee's answer is `f - log_pi`, so the gap between expert and policy is the gap
    # in `f` plus the gap in `-log_pi`, and these accumulate one of each.
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
      fooled += _called_real(discriminator, policy_batch)
      caught += _called_real(discriminator, expert_batch)
      with torch.no_grad():
        expert_reward, expert_likelihood = discriminator.split_logits(expert_batch)
        policy_reward, policy_likelihood = discriminator.split_logits(policy_batch)
        separation_by_reward += float(expert_reward.mean() - policy_reward.mean())
        separation_by_likelihood += float(
          policy_likelihood.mean() - expert_likelihood.mean()
        )

    losses = ppo.update()
    # Applied every update rather than once, because nothing else bounds it: PPO's
    # entropy bonus raises the std a little on every single one (see clamp_action_std).
    clamp_action_std(actor, cfg.action_max_std)

    if cfg.log_every and (
      iteration % cfg.log_every == 0 or iteration == cfg.num_iterations - 1
    ):
      # How to read this line.
      #
      # `imit` is softplus(logit): it falls as the referee gets better and rises as the
      # bridge fools it, so it is not a task score. Read `fooled` (the share of the
      # bridge's own transitions the referee calls real) against `caught` (the share of
      # real ones it calls real). A healthy game keeps caught high while fooled climbs;
      # fooled near 0 with caught near 1 means the referee has won outright and the
      # bridge has no gradient left to climb, and both near 0.5 means it has stopped
      # telling them apart at all.
      #
      # `by_f` and `by_logp` are the single most important pair here. The referee tells
      # the two halves apart by `f - log_pi`, so the gap it opens between them is the gap
      # in `f` plus the gap in `-log_pi`, and these are those two contributions. `by_f`
      # is the referee actually judging behavior, which is the thing being learned.
      # `by_logp` is the referee getting the answer for free from the fact that the
      # expert simply does not take the actions this policy takes.
      #
      # `by_logp` much larger than `by_f` is the failure this whole file was rewritten
      # around: the referee's loss saturates on the free answer, no gradient reaches `f`,
      # the reward collapses to a constant, and PPO then optimizes noise at full speed.
      # With raw wheel velocities `by_logp` reached the hundreds. It should now sit at a
      # few units, comparable to `by_f`. If it starts climbing away, stop and look at
      # the `[spaces]` header rather than at anything else below.
      #
      # `drift` is the only number here outside the game: mean distance from the target
      # skill's own states in standardized units, so it is what should actually fall if
      # the bridge is learning. It is bounded by the state clip times sqrt(obs_dim);
      # pinned at that ceiling means the bridge is nowhere near the target skill's own
      # states, whatever the game is reporting.
      #
      # `term` is the share of steps that broke the episode and `redo` how many envs had
      # to be put back at an interrupt state as a result. Both should fall.
      #
      # `std` is the actor's exploration in normalized action units. It should settle,
      # not climb: a std pinned at `action_max_std` iteration after iteration means the
      # entropy bonus is beating the surrogate and the policy is drifting toward random,
      # which is what the clamp is holding back rather than fixing. Lower `entropy_coef`
      # if that is what it is doing.
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
        f"value {losses['value']:8.4f} | "
        f"std {action_std(actor):5.3f}"
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
  view: StateView,
  action_space: ActionSpace,
  state_space: StateSpace,
  outcome: OutcomeSource,
  plan: WindowPlan,
  cfg: SwitchPhase,
) -> None:
  """Phase 2: train `switch` in place by Double-DQN, against the frozen `actor`.

  The frozen actor drives the robot while the switch-decider chooses, per step, whether
  to hand over. Each env is judged `cfg.eval_steps` after *its own* hand-over: +1 if
  `outcome` approves and the env never terminated, -1 otherwise, and -1 for never
  committing within `cfg.max_transition_steps`.

  The decider reads the same standardized state the actor does, so the two agree on what
  a state is; the outcome it learns from does not, and must not, since judging a
  hand-over is privileged and reads the env directly.

  A note before reading any number this prints: this phase sits on top of a frozen
  bridge, so it can only be as good as phase 1 left it. If the bridge never reaches a
  state the target skill can take over from, every hand-over fails and never committing
  also fails, the decider is genuinely indifferent between its two actions, and the
  numbers below will look like a broken decider when the thing that is broken is
  upstream. `good` near zero whatever `switched` does is that case.
  """

  device = env.device
  num_envs = env.num_envs
  entity = env.scene[entity_name]
  manager_state = ManagerState(env)
  obs_dim = view.dim

  env.reset()
  dqn = DoubleDQN(switch, cfg.gamma, cfg.learning_rate, device)
  # One buffer per decision rather than one shared. An env produces at most one "switch"
  # row per window but a "stay" row on every step it did not switch, so a shared buffer
  # would be around fifty to one against the decision that matters, and the Q-value of
  # switching would be learned from whatever survived the imbalance.
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
    f"\n=== switch -> '{target_skill.name}': Double-DQN on the '{outcome.label}' "
    f"signal, {cfg.num_iterations} iterations over {num_envs} envs ==="
  )
  outcome.prepare(env, target_skill, cfg.eval_steps)

  interrupts = _prepare_interrupts(
    env, pool, entity_name, target_skill_id, plan, manager_state, cfg.num_interrupts
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
    # says nothing about how the hand-over went. Unlike phase 1, nothing is restored
    # here -- the termination is the measurement.
    failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    verdict = torch.zeros(num_envs, dtype=torch.bool, device=device)
    judged = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # `failed` as it stood when each env was judged, so the reported termination rate is
    # the one the verdict actually saw. Anything that breaks later shows up as `late`.
    broken = torch.zeros(num_envs, dtype=torch.bool, device=device)
    stay_obs, stay_reward, stay_next_obs, stay_done = [], [], [], []

    for t in range(total_steps):
      state = state_space.standardize(view(as_tensor(obs["actor"])))
      driving = switched.clone()  # The target skill is at the controls in these envs

      # Both candidates are produced in raw units before they are mixed: the skill emits
      # raw actions directly, the bridge emits normalized ones that have to be converted
      # back first.
      with torch.no_grad():
        bridge_actions = action_space.denormalize(actor(bridge_obs_td(state)))
      target_actions = target_skill.act(obs, driving)
      actions = torch.where(driving.unsqueeze(-1), target_actions, bridge_actions)

      alive_before = ~failed
      if t < cfg.max_transition_steps:
        decisions = dqn.act(state, epsilon)
        new_switch = (decisions == 1) & ~driving & alive_before
      else:
        new_switch = torch.zeros(num_envs, dtype=torch.bool, device=device)

      next_obs, reward, terminated, _, _ = env.step(actions)
      next_state = state_space.standardize(view(as_tensor(next_obs["actor"])))
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
        # stored: it is reconstructed here as ones for the switch half and zeros for
        # the stay half.
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
        + f"loss {last_loss:8.4f} | "
        + f"replay {len(switch_replay)}/{len(stay_replay)}"
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

  # Measured once, from the whole pool, and shared by every target's bridge. Once rather
  # than per target because a bridge into one skill starts wherever another skill left
  # the robot, so the numbers it meets span the pool; and shared so that a comparison
  # between two targets is a comparison of behavior rather than of units. `meta` keeps
  # them because they are as much a part of a trained bridge as its weights are: loaded
  # without them, the actor's output would be interpreted in the wrong units.
  action_space, state_space = measure_spaces(
    env, pool, meta.view, plan, clip=cfg.bridge.state_clip
  )
  meta.adopt_spaces(action_space, state_space)

  for target_id in range(len(pool)):
    target_skill = pool[target_id]
    actor = meta.actors[target_id]
    train_bridge(
      env,
      actor,
      target_skill,
      target_id,
      pool,
      entity_name,
      meta.view,
      action_space,
      state_space,
      plan,
      cfg.bridge,
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
      meta.view,
      action_space,
      state_space,
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
