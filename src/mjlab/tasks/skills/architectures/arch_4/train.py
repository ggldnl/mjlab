"""Training for arch_4: fill the hole, and be scored on what used to be in it.

One phase, one policy, and no adversary anywhere. An iteration is one masked window per
environment:

    draw a window from the corpus, biased toward where something happens
    put the robot on the frame the hole begins at
    show the policy the frames before the hole and the frames after it
    roll, paying it for reproducing the reference it was never shown
    PPO

The rollout does not stop when the gap does. It runs on into the window the policy *was*
shown, and the reward there is measured against the same clip. That tail is the arrival
test and it is free: the gap and the window after it are one continuous stretch of one
motion, so tracking the whole stretch needs no second reference and no separate terminal
bonus. What separates the two halves in the log is the only thing that separates them at
all -- whether the policy had been shown the answer.

Reference-state initialization is doing a lot of work here, and it is the same argument
ASAP makes and the jump task inherits. A humanoid policy exploring from a standing start
discovers nothing, because a random 29-joint action sequence falls over before it
produces a stride. Starting every environment already inside the motion, at the exact
frame where the hole begins, means the first iteration already sees the hard part. It is
also why a fall part way through is answered by putting that environment back on the
reference rather than by letting the arena's own reset stand: the arena resets to a
stand, and a stand is not a state this policy is ever asked about.

Two things this trainer deliberately does not use, both of which every other architecture
in the family does:

- the window plan's opening and closing windows. arch_4 is not imitating the pool, so
  there is nothing to record from it. The plan is still read, for `measure_spaces`.
- a success test. There is no hand-over decision, so nothing asks whether one went well,
  and this trainer's signature does not mention one.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.architectures.arch_4 import Arch4
from mjlab.tasks.skills.architectures.arch_4.config import MaskedTraining
from mjlab.tasks.skills.architectures.arch_4.dataset import (
  MaskedBatch,
  MotionCorpus,
  PrepareSkill,
  collect_profiles,
  discover_motions,
)
from mjlab.tasks.skills.architectures.arch_4.frames import frame_dim, robot_frame
from mjlab.tasks.skills.architectures.arch_4.networks import (
  TrackingScore,
  input_dim,
  masked_input,
)
from mjlab.tasks.skills.architectures.common.nets import (
  OBS_GROUPS,
  action_std,
  clamp_action_std,
  obs_td,
  set_action_std,
)
from mjlab.tasks.skills.experiment import Experiment
from mjlab.tasks.skills.spaces import StateSpace, Units, measure_spaces
from mjlab.tasks.skills.view import StateView
from mjlab.tasks.skills.windows import as_tensor


def rollout_horizon(cfg: MaskedTraining) -> int:
  """How many steps one training rollout runs for.

  The longest gap, plus the stretch the visible window spans. Fixed rather than per
  environment because PPO's storage is: a shorter gap simply leaves more of the rollout
  on the visible side, which is the half the arrival is measured on anyway.
  """
  stride = max(cfg.context_stride, 1)
  return cfg.gap_range[1] + stride * (cfg.future_steps - 1) + 1


def _score_of(cfg: MaskedTraining, corpus: MotionCorpus) -> TrackingScore:
  return TrackingScore(
    corpus.groups,
    weights={
      "posture": cfg.weight_posture,
      "lin_vel": cfg.weight_lin_vel,
      "ang_vel": cfg.weight_ang_vel,
      "joint_pos": cfg.weight_joint_pos,
      "joint_vel": cfg.weight_joint_vel,
    },
    stds={
      "posture": cfg.std_posture,
      "lin_vel": cfg.std_lin_vel,
      "ang_vel": cfg.std_ang_vel,
      "joint_pos": cfg.std_joint_pos,
      "joint_vel": cfg.std_joint_vel,
    },
  )


class _Context:
  """One iteration's fixed conditioning, and the observation built out of it.

  The two context windows and the gap length are drawn once per iteration and do not
  change while the rollout runs; only the proprioception and the progress scalar do. This
  holds the fixed part so the loop below reads as what it is -- one call per step, with
  the step number the only thing that varies -- and so the policy and the value bootstrap
  are built by the same code rather than by two copies of it.
  """

  def __init__(
    self,
    past: torch.Tensor,
    future: torch.Tensor,
    gap: torch.Tensor,
    max_gap: float,
    view: StateView,
    state_space: StateSpace,
    obs_group: str,
  ) -> None:
    self.past = past
    self.future = future
    self.gap = gap
    self.view = view
    self.state_space = state_space
    self.obs_group = obs_group
    self.length = gap / max(max_gap, 1.0)

  def __call__(self, step: int, obs: VecEnvObs) -> torch.Tensor:
    progress = (torch.full_like(self.gap, float(step)) / self.gap).clamp(0.0, 1.0)
    state = self.state_space.standardize(self.view(as_tensor(obs[self.obs_group])))
    return masked_input(state, self.past, self.future, progress, self.length)


def _fit_frame_space(corpus: MotionCorpus, clip: float) -> StateSpace:
  """Measure the frame conversion over the corpus.

  The context windows go through this before any network sees them, for the same reason
  the observation does (see spaces.py): a frame holds a height of about one, joint
  velocities in the tens, and a gravity direction bounded by one, and a network reading
  the three raw is reading mostly the middle one.

  Measured over the corpus rather than over the pool, which is why it is not part of the
  run's `Units`: the frames the policy reads come from clips, and it is the clips' spread
  that has to be flattened.

  Padding is excluded. Clips are padded out to the longest by repeating their final
  frame, and a short clip in a corpus of long ones would otherwise contribute thousands
  of copies of one standing pose to the statistics.
  """
  valid = (
    torch.arange(corpus.frames.shape[1], device=corpus.device)[None, :]
    < (corpus.lengths[:, None])
  )
  space = StateSpace(corpus.frame_dim, clip).to(corpus.device)
  space.fit(corpus.frames[valid])
  print(f"[frames] {space.describe()}")
  return space


@torch.no_grad()
def teleport(
  env: ManagerBasedRlEnv,
  entity: Entity,
  batch: MaskedBatch,
  step: int,
  env_ids: torch.Tensor | None = None,
) -> VecEnvObs:
  """Put environments on the reference frame at `step` of their own window.

  Mirrors the tail of a reset the way `restore_interrupts` does (write, forward,
  command, sense, observe) so the observation handed back really describes what was
  written. The horizontal position is dropped and replaced by the environment's own
  origin: clips wander tens of metres from where they started and nothing in this
  architecture reads absolute position, so there is no reason to carry it into the arena.

  Joint targets are clamped to the soft limits, which retargeted motion capture does
  cross. A reference outside the joint's range is not reachable, and writing it puts the
  robot in a state the physics has to resolve before anything else happens.
  """
  root_state, joint_pos, joint_vel = batch.at(step)
  root_state = root_state.clone()
  origins = env.scene.env_origins
  if env_ids is None:
    root_state[:, :2] = origins[:, :2]
  else:
    root_state = root_state[env_ids]
    joint_pos = joint_pos[env_ids]
    joint_vel = joint_vel[env_ids]
    root_state[:, :2] = origins[env_ids, :2]

  limits = entity.data.soft_joint_pos_limits
  limits = limits if env_ids is None else limits[env_ids]
  joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

  entity.write_root_state_to_sim(root_state, env_ids)
  entity.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  # The action history the proprioceptive observation carries belongs to the rollout
  # that just ended; left alone the policy's first step reads a `last_action` from a
  # motion that has nothing to do with this one.
  env.action_manager.reset(env_ids)
  if env_ids is None:
    env.episode_length_buf[:] = 0
  else:
    env.episode_length_buf[env_ids] = 0
  env.command_manager.compute(dt=0.0)
  env.sim.sense()
  env.abstraction_manager.compute(dt=0.0)
  return env.observation_manager.compute(update_history=True)


def fill_gaps(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  actor: MLPModel,
  corpus: MotionCorpus,
  units: Units,
  frame_space: StateSpace,
  cfg: MaskedTraining,
  obs_group: str = "actor",
) -> None:
  """Train `actor` in place: produce what the corpus had inside its masked windows."""
  device = env.device
  num_envs = env.num_envs
  entity: Entity = env.scene[exp.entity_name]
  view = exp.view
  action_dim = env.action_manager.total_action_dim
  horizon = rollout_horizon(cfg)
  score = _score_of(cfg, corpus)

  print(
    f"\n=== bridge: {cfg.num_iterations} iterations over {num_envs} envs, "
    f"gap {cfg.gap_range[0]}-{cfg.gap_range[1]} then {horizon - cfg.gap_range[1]} "
    f"visible ==="
  )
  print(f"[view] proprioception is {view.label}")

  width = input_dim(view.dim, corpus.frame_dim, cfg.past_steps, cfg.future_steps)
  template = obs_td(torch.zeros(num_envs, width, device=device))

  set_action_std(actor, cfg.action_init_std)
  critic = MLPModel(
    template,
    OBS_GROUPS,
    "critic",
    1,
    hidden_dims=cfg.critic_hidden_dims,
    activation="elu",
  ).to(device)
  storage = RolloutStorage("rl", num_envs, horizon, template, [action_dim], device)
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
    desired_kl=cfg.desired_kl,
    device=device,
  )

  for iteration in range(cfg.num_iterations):
    batch = corpus.sample(
      num_envs, cfg.gap_range, cfg.past_steps, cfg.future_steps, horizon
    )
    # A gap longer than the rollout would leave the visible window off the end of it.
    assert int(batch.gap.max()) <= horizon
    obs = teleport(env, entity, batch, 0)
    context = _Context(
      past=frame_space.standardize(batch.past),
      future=frame_space.standardize(batch.future),
      gap=batch.gap.float().clamp_min(1.0),
      max_gap=float(cfg.gap_range[1]),
      view=view,
      state_space=units.state,
      obs_group=obs_group,
    )
    ppo.train_mode()

    gap_score = torch.zeros(num_envs, device=device)
    tail_score = torch.zeros(num_envs, device=device)
    gap_steps = torch.zeros(num_envs, device=device)
    tail_steps = torch.zeros(num_envs, device=device)
    group_totals = {name: 0.0 for name, _ in corpus.groups.named()}
    terminations = 0.0
    restarts = 0.0
    broken = 0.0

    for t in range(horizon):
      normalized_action = ppo.act(obs_td(context(t, obs)))
      raw_action = units.action.denormalize(normalized_action)

      next_obs, _, terminated, time_out, extras = env.step(raw_action)
      done = terminated | time_out
      terminations += float(terminated.float().mean())

      # Read the robot before anything is restored below: this is the frame the action
      # actually produced, and it is what the reference is compared against. For an
      # environment that terminated it is already the arena's fresh reset state (see
      # FLOW.md), which scores near zero -- correctly, since that environment fell.
      frame = robot_frame(entity)
      unusable = ~torch.isfinite(frame).all(dim=-1)
      if bool(unusable.any()):
        # A humanoid driven into itself hard enough overflows the constraint solver and
        # the physics stops returning numbers. One such row turns the whole update to
        # NaN, so those environments are treated exactly like the ones that fell.
        broken += float(unusable.float().sum())
        frame = torch.nan_to_num(frame)
        done = done | unusable

      terms = score.terms(frame, batch.reference[:, t])
      step_score = torch.zeros(num_envs, device=device)
      for name, value in terms.items():
        step_score = step_score + value
        group_totals[name] += float(value.mean())
      reward = step_score - cfg.termination_penalty * done.float()

      # Which half of the rollout this step belongs to: inside the hole, or inside the
      # window the policy was shown. Kept apart because the difference between them is
      # the only measurement this architecture really produces.
      inside = t < batch.gap
      gap_score += step_score * inside.float()
      tail_score += step_score * (~inside).float()
      gap_steps += inside.float()
      tail_steps += (~inside).float()

      step_extras = dict(extras)
      step_extras["time_outs"] = time_out
      # Put whatever the arena just reset back on the reference it should have been on.
      # Left alone it would spend the rest of the rollout being scored against a motion
      # it is no longer anywhere near, from a standing start it was never asked about.
      if bool(done.any()):
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        restarts += float(done_ids.numel())
        next_obs = teleport(env, entity, batch, t + 1, env_ids=done_ids)

      ppo.process_env_step(obs_td(context(t + 1, next_obs)), reward, done, step_extras)
      obs = next_obs

    # The rollout ends because the window ran out, not because anything happened in the
    # world, so bootstrapping from the critic is right even though the next iteration
    # teleports away from here.
    ppo.compute_returns(obs_td(context(horizon, obs)))
    losses = ppo.update()
    clamp_action_std(actor, cfg.action_max_std)

    if cfg.log_every and (
      iteration % cfg.log_every == 0 or iteration == cfg.num_iterations - 1
    ):
      # How to read this line.
      #
      # `gap` is the mean per-step tracking score inside the hole, where the policy is
      # producing motion it was never shown. `tail` is the same score over the window it
      # *was* shown, which it should be arriving at. Both are bounded above by the sum of
      # the reward weights (5.0 with the defaults) and both should rise.
      #
      # The pair is the measurement. `tail` well above `gap` means the policy is not so
      # much filling the hole as waiting it out and then catching the visible window,
      # which is a hand-over with extra steps. `gap` above `tail` means it is producing
      # plausible motion that arrives somewhere else, which is the failure the whole
      # architecture exists to avoid; the transition would look fine and then hand the
      # next skill a robot it cannot start from.
      #
      # The per-group columns say which part of the motion is being missed. `joint_pos`
      # is posture, `lin_vel` is pace and direction. On a run-to-jump transition it is
      # `lin_vel` that carries the problem, and a run whose `joint_pos` is high while
      # `lin_vel` stays low is a policy miming the motion without actually changing speed.
      #
      # `term` is the share of steps that ended an episode and `redo` how many
      # environments had to be put back on their reference as a result; both should fall.
      # `bad` is environments whose state came back non-finite and must be exactly zero.
      #
      # `std` is the exploration in normalized action units. It should settle, not climb
      # to `action_max_std` and park there.
      groups = " | ".join(
        f"{name} {group_totals[name] / horizon:5.3f}"
        for name, _ in corpus.groups.named()
      )
      print(
        f"[bridge {iteration + 1:5d}/{cfg.num_iterations}] "
        f"gap {float((gap_score / gap_steps.clamp_min(1)).mean()):6.3f} | "
        f"tail {float((tail_score / tail_steps.clamp_min(1)).mean()):6.3f} | "
        f"{groups} | "
        f"term {terminations / horizon:5.1%} | "
        f"redo {restarts / num_envs:5.2f} | "
        f"bad {int(broken):4d} | "
        f"value {losses['value']:8.4f} | "
        f"std {action_std(actor):5.3f}"
      )


def train(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  meta: Arch4,
  cfg: MaskedTraining,
  prepare: PrepareSkill | None = None,
) -> Arch4:
  """arch_4: one corpus, one policy, and nothing else to train.

  `prepare` is the one thing this architecture needs that no other does: the profiles are
  recorded by letting each skill run, and a goal-conditioned skill left at the arena's
  defaults runs the wrong errand (the corridor's run policy tracking a walk's commanded
  speed produces a profile of a walk). An experiment passes a callback that writes each
  skill's commands; leaving it out records whatever the arena's own defaults produce.
  """
  low, high = cfg.gap_range
  if not low <= cfg.inference_gap <= high:
    raise ValueError(
      f"inference_gap is {cfg.inference_gap}, outside the {cfg.gap_range} range the "
      f"corpus is masked over. The policy reads how far through a gap it is, so a gap "
      f"of that length is a question it was never asked."
    )
  if cfg.past_steps < 1 or cfg.future_steps < 1:
    raise ValueError(
      f"A masked window needs a sequence on each side to stitch: got "
      f"past_steps={cfg.past_steps}, future_steps={cfg.future_steps}."
    )

  entity: Entity = env.scene[exp.entity_name]
  corpus = MotionCorpus(
    discover_motions(cfg.motion_dir, cfg.motion_pattern),
    env.device,
    horizon=rollout_horizon(cfg),
    past_steps=cfg.past_steps,
    stride=cfg.context_stride,
    eventfulness=cfg.eventfulness,
  )
  robot_joints = int(entity.data.joint_pos.shape[-1])
  if corpus.num_joints != robot_joints:
    raise ValueError(
      f"The corpus was recorded on a {corpus.num_joints}-joint robot and "
      f"'{exp.entity_name}' has {robot_joints}. A frame is joint state, so the two have "
      f"to be the same robot in the same joint order; convert the clips against this "
      f"model."
    )
  if corpus.frame_dim != frame_dim(robot_joints):
    raise ValueError("Corpus frame width does not match this robot.")

  # Measured over the whole pool, for the same reason arch_1 does it: the actions this
  # bridge commands have to be comparable with the ones the skills command, and the
  # proprioception it reads has to be in the units every other architecture reads it in.
  units = measure_spaces(env, exp.pool, exp.view, exp.windows, clip=cfg.state_clip)
  frame_space = _fit_frame_space(corpus, cfg.state_clip)
  meta.adopt_units(units)
  meta.adopt_frame_space(frame_space)
  meta.configure(
    cfg.past_steps, cfg.future_steps, cfg.context_stride, cfg.actor_hidden_dims
  )
  meta.gap_steps = cfg.inference_gap
  meta.max_gap = high
  meta.entity_name = exp.entity_name

  # Recorded before training rather than after, so that a run which dies part way through
  # still leaves a checkpoint whose profiles are the ones its policy was going to aim at.
  meta.adopt_profiles(collect_profiles(env, exp, cfg, prepare))

  fill_gaps(
    env, exp, meta.actor, corpus, units, frame_space, cfg, obs_group=meta.obs_group
  )

  meta.actor.eval()
  for parameter in meta.actor.parameters():
    parameter.requires_grad_(False)
  return meta
