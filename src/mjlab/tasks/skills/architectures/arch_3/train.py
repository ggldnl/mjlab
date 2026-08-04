"""Training for arch_3: one phase, one residual per target skill.

For each skill in the pool, in order: record its opening window (the behavior to
reproduce), then train the residual that rides on transitions into it. No second phase,
no Q-network, no success test, because there is no hand-over decision to learn.

One iteration is one transition, start to finish:

    pick a skill to leave and how long it has been running
    roll it from its own start state for that long        (warm-up, nothing learned)
    hand the target skill in and run the fade for N steps (alpha 1 -> 0)
    keep rolling with the target skill alone              (alpha 0, the tail)
    PPO over everything after the warm-up

The tail is where the credit assignment comes from. With alpha at zero the residual cannot
touch the robot, so the reward collected there is the target skill's own, and GAE carries
it back into the steps of the fade that produced it. That is the same measurement arch_1's
phase 2 needs a success test, a latched failure flag and a separate Double-DQN loop for.

Two departures from arch_1's phase 1.

The source skill is rolled live rather than teleported to a harvested state. It has to be:
arch_3 uses the source skill's own action during the fade, and a harvested state does not
carry a skill's memory. `ManagerState` snapshots the action and command terms, but the
diffdrive's turn integrates a yaw rate of its own and the parkour jump pins a reference
clip, and neither is in the simulator. Restored into such a state, the source's
contribution to the blend would come from an integral belonging to a different rollout.
The warm-up costs steps and buys a source skill genuinely mid-behavior, drawn from the
same distribution `collect_interrupts` harvests.

The referee runs as a plain classifier: the log-probability term is passed as zero on
both halves. AIRL's logit is `f - log pi(a|s)`, and the action here is not what the policy
emitted -- the policy emits a correction that reaches the env through a deterministic
affine map whose slope is alpha, so the density carries a `-dim * log(alpha)` Jacobian
that is singular exactly where the architecture is designed to end up.
"""

from __future__ import annotations

import random

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_3 import (
  Arch3,
  alpha_at,
  compose,
  residual_input,
)
from mjlab.tasks.skills.architectures.arch_3.config import ResidualTraining
from mjlab.tasks.skills.architectures.common.imitation import (
  AIRLDiscriminator,
  DiscBatch,
  RewardScaler,
  called_real,
  no_log_prob,
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
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.spaces import Units, measure_spaces
from mjlab.tasks.skills.windows import as_tensor, collect_opening, start_skill


def train_residual(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  residual: MLPModel,
  target_skill_id: int,
  units: Units,
  cfg: ResidualTraining,
) -> None:
  """Train `residual` in place: make the fade into the target skill work.

  The referee is trained to tell the composed behavior from the target skill's own opening
  window, and PPO trains the residual against its verdict.
  """
  device = env.device
  num_envs = env.num_envs
  pool = exp.pool
  view = exp.view
  action_dim = env.action_manager.total_action_dim
  obs_dim = view.dim
  everything = torch.ones(num_envs, dtype=torch.bool, device=device)
  target_skill = pool[target_skill_id]
  others = [i for i in range(len(pool)) if i != target_skill_id]
  if not others:
    raise ValueError("A transition needs a skill to come from; the pool has one skill.")

  # A shorter fade simply leaves a longer tail, so the rollout is a fixed length whatever
  # this iteration's transition length turns out to be. PPO's storage needs that.
  horizon = cfg.steps[1] + cfg.tail_steps

  print(
    f"\n=== residual -> '{target_skill.name}': {cfg.num_iterations} iterations over "
    f"{num_envs} envs, fade {cfg.steps[0]}-{cfg.steps[1]} steps + {cfg.tail_steps} tail "
    f"==="
  )
  print(f"[view] matching on {view.label}")

  ##
  # The example to reproduce, in the run's own units.
  ##

  spec = exp.windows[target_skill]
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
  # The two players.
  ##

  obs, _ = env.reset()
  template = obs_td(
    residual_input(
      units.state.standardize(view(as_tensor(obs["actor"]))),
      torch.ones(num_envs, device=device),
      torch.zeros(num_envs, action_dim, device=device),
    )
  )

  set_action_std(residual, cfg.action_init_std)
  critic = MLPModel(
    template,
    OBS_GROUPS,
    "critic",
    1,
    hidden_dims=cfg.critic_hidden_dims,
    activation="tanh",
  ).to(device)
  storage = RolloutStorage("rl", num_envs, horizon, template, [action_dim], device)
  ppo = PPO(
    residual,
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
  # Same treatment as arch_1. This architecture needs the centering for an extra reason:
  # it subtracts a residual penalty from the same total, and two terms can only be traded
  # off against each other if one of them is not drifting.
  reward_scaler = RewardScaler(device, clip=cfg.reward_clip)

  target_rows = pool.driven_by(target_skill_id, num_envs, device)
  target_assignment = torch.full((num_envs,), target_skill_id, device=device)

  ##
  # The game.
  ##

  for iteration in range(cfg.num_iterations):
    source_id = random.choice(others)
    source_skill = pool[source_id]
    source_spec = exp.windows[source_skill]
    warmup = int(source_spec.sample_cuts(1, device))
    fade = random.randint(*cfg.steps)
    source_rows = pool.driven_by(source_id, num_envs, device)
    source_assignment = torch.full((num_envs,), source_id, device=device)

    # Warm-up: the source skill runs from its own start state for as long as the window
    # plan says a hand-over may fall, so the fade begins from a robot genuinely
    # mid-behavior, with the skill's own memory to match.
    obs, _ = env.reset()
    obs = start_skill(env, source_spec, obs)
    pool.reset(everything)
    warmup_deaths = 0.0
    for _ in range(warmup):
      actions = SkillPool.select(pool.act_each(obs, source_rows), source_assignment)
      obs, _, terminated, time_out, _ = env.step(actions)
      warmup_deaths += float((terminated | time_out).float().sum())

    # The target contributes to the blend from the first step of the fade, so it is
    # handed control here rather than at the end of it.
    target_skill.reset(everything)

    ppo.train_mode()
    state = units.state.standardize(view(as_tensor(obs["actor"])))
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    imitation_sum = 0.0
    tail_sum = 0.0
    tail_steps = 0
    correction_sum = 0.0
    terminations = 0.0
    broken = 0.0
    fake_obs, fake_actions, fake_next_obs, fake_done = [], [], [], []
    # The transition PPO is still holding: closed at the top of the next pass, once the
    # observation that follows it has been built.
    pending: tuple[torch.Tensor, torch.Tensor, dict] | None = None

    # One pass past the horizon, which produces no action and takes no env step. The
    # residual's observation carries the blended action it is correcting, and building
    # that needs a pass over the pool at the state in question, so the observation *after*
    # a step cannot be assembled until the following pass. The last transition and the
    # value to bootstrap from both need it, hence the extra lap.
    for t in range(horizon + 1):
      fading = t <= fade
      alpha = alpha_at(torch.full((num_envs,), float(t), device=device), fade)
      # Only the skills whose action is about to be used are advanced, and each exactly
      # once (see skill.py). Past the end of the fade the source is no longer part of the
      # blend and stops driving.
      involved = target_rows | source_rows if fading else target_rows
      skill_actions = pool.act_each(obs, involved)
      source_action = SkillPool.select(skill_actions, source_assignment)
      target_action = SkillPool.select(skill_actions, target_assignment)

      # Composed twice: once with no correction, to give the residual the blended action
      # it is being asked to correct, then again with the correction it returns.
      normalized_blend, _ = compose(
        units.action,
        source_action,
        target_action,
        alpha,
        torch.zeros_like(source_action),
        cfg.residual_scale,
      )
      template = obs_td(residual_input(state, alpha, normalized_blend))

      if pending is not None:
        ppo.process_env_step(template, *pending)
        pending = None
      if t == horizon:
        break

      correction = ppo.act(template)
      _, raw_action = compose(
        units.action,
        source_action,
        target_action,
        alpha,
        correction,
        cfg.residual_scale,
      )

      next_obs, _, terminated, time_out, extras = env.step(raw_action)
      done = terminated | time_out
      terminations += float(terminated.float().mean())
      next_state = units.state.standardize(view(as_tensor(next_obs["actor"])))

      # A robot driven into itself hard enough overflows the constraint solver and the
      # physics stops returning numbers. One such row turns the whole update to NaN, so
      # those envs are marked done and dropped from the referee's half.
      unusable = ~torch.isfinite(next_state).all(dim=-1)
      if bool(unusable.any()):
        broken += float(unusable.float().sum())
        next_state = torch.nan_to_num(next_state)
        done = done | unusable

      # What actually reached the env, minus what the fade alone would have commanded:
      # the correction as applied, alpha included. Zero past the end of the fade by
      # construction, which is the whole point of the schedule.
      normalized_action = units.action.normalize(raw_action)
      applied = (normalized_action - normalized_blend).abs().mean(dim=-1)

      batch = DiscBatch(
        obs=state,
        actions=normalized_action,
        next_obs=next_state,
        done=done,
        log_prob=no_log_prob(num_envs, device),
      )
      with torch.no_grad():
        imitation_reward = reward_scaler(discriminator.reward(batch))
      reward = (
        imitation_reward
        - cfg.termination_penalty * terminated.float()
        - cfg.residual_penalty * applied
      )
      imitation_sum += float(imitation_reward.mean())
      correction_sum += float(applied.mean())
      if not fading:
        tail_sum += float(imitation_reward.mean())
        tail_steps += 1

      # Rows from an env that has already broken belong to a fresh arena episode, not to
      # this transition. Showing them to the referee as behavior the architecture produced
      # is a trap: for a target skill that starts from rest they are the expert data,
      # labelled fake.
      kept = alive & ~done
      if bool(kept.any()):
        fake_obs.append(state[kept])
        fake_actions.append(normalized_action[kept])
        fake_next_obs.append(next_state[kept])
        fake_done.append(done[kept])
      alive = alive & ~done

      step_extras = dict(extras)
      step_extras["time_outs"] = time_out
      pending = (reward, done, step_extras)

      obs, state = next_obs, next_state

    # The rollout ends because the budget ran out, so bootstrapping from the critic is
    # right even though the next iteration starts a fresh transition.
    ppo.compute_returns(template)

    ##
    # Train the referee: this rollout is the fake half, the opening window the real one.
    ##

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
        log_prob=no_log_prob(cfg.disc_batch_size, device),
      )
      real = expert_buffer.sample(cfg.disc_batch_size)
      expert_batch = DiscBatch(
        obs=real["obs"],
        actions=real["action"],
        next_obs=real["next_obs"],
        done=real["done"],
        log_prob=no_log_prob(cfg.disc_batch_size, device),
      )
      disc_loss = discriminator.loss(policy_batch, expert_batch)
      disc_optimizer.zero_grad()
      disc_loss.backward()
      disc_optimizer.step()

      disc_loss_sum += float(disc_loss)
      fooled += called_real(discriminator, policy_batch)
      caught += called_real(discriminator, expert_batch)

    losses = ppo.update()
    clamp_action_std(residual, cfg.action_max_std)

    if cfg.log_every and (
      iteration % cfg.log_every == 0 or iteration == cfg.num_iterations - 1
    ):
      # How to read this line.
      #
      # `tail` is what this architecture is really about: the referee's reward over the
      # steps after the fade ended, when the target skill drives alone and the residual
      # can do nothing. That is hand-over quality, and it is what should rise. `imit` is
      # the same over the whole rollout; `tail` well below `imit` means the transition
      # looks fine while it is being helped and falls apart when the help stops, which is
      # the failure this design exists to make visible.
      #
      # `fooled` against `caught` is the health of the game (see arch_1's train.py).
      #
      # `corr` is the mean size of the correction actually applied, in normalized action
      # units. It should fall as the residual finds a transition the fade can mostly carry
      # on its own. It cannot fall to zero: some correction is the point.
      #
      # `term` is the share of steps that broke the episode, `warm` how many envs broke
      # during the warm-up (before anything being trained had a say), and `bad` how many
      # came back non-finite. `bad` should be exactly zero.
      #
      # `std` is the residual's exploration. It should settle, not park at max.
      denom = max(cfg.disc_epochs, 1)
      print(
        f"[residual {iteration + 1:4d}/{cfg.num_iterations}] "
        f"fade {fade:3d} | "
        f"from '{source_skill.name}' @{warmup:3d} | "
        f"imit {imitation_sum / horizon:7.4f} | "
        f"tail {tail_sum / max(tail_steps, 1):7.4f} | "
        f"disc_loss {disc_loss_sum / denom:6.4f} | "
        f"fooled {fooled / denom:5.1%} | "
        f"caught {caught / denom:5.1%} | "
        f"corr {correction_sum / horizon:6.3f} | "
        f"term {terminations / horizon:5.1%} | "
        f"warm {int(warmup_deaths):4d} | "
        f"bad {int(broken):4d} | "
        f"value {losses['value']:8.4f} | "
        f"std {action_std(residual):5.3f}"
      )


def train(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  meta: Arch3,
  cfg: ResidualTraining,
) -> Arch3:
  """arch_3: one residual per target skill, and nothing else to train."""
  low, high = cfg.steps
  if not low <= cfg.inference_steps <= high:
    raise ValueError(
      f"inference_steps is {cfg.inference_steps}, outside the {cfg.steps} range the "
      f"residual is trained over. It reads alpha, so a fade at that rate is a question "
      f"it was never asked."
    )

  # Measured once over the whole pool and shared by every residual, for the same reason
  # arch_1 does it: a transition starts wherever one skill left the robot and ends where
  # another lives, so the numbers it meets span the pool.
  units = measure_spaces(env, exp.pool, exp.view, exp.windows, clip=cfg.state_clip)
  meta.adopt_units(units)
  meta.transition_steps = cfg.inference_steps
  meta.residual_scale = cfg.residual_scale

  for target_id in range(len(exp.pool)):
    residual = meta.residuals[target_id]
    train_residual(env, exp, residual, target_id, units, cfg)
    residual.eval()
    for parameter in residual.parameters():
      parameter.requires_grad_(False)

  return meta
