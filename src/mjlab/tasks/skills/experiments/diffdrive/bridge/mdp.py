"""MDP terms wiring the bridge into a manager-based RL environment.

The task is defined only by the dataset of couples (one skill1 end trajectory, one skill2
start trajectory per entry). Each episode picks a couple and drops the robot at the
interrupt, where skill1 leaves it. The robot's recent history (the tail of skill1's end
trajectory) and both windows are part of the observation, so the policy knows what it was
doing and what skill2 looks like.

The bridge has two parts that show up here:

* Executor   the PPO policy. Each episode commits to one fixed merge frame on skill2's
             window. The executor first drives to that frame (phase one) and is then
             rewarded for following the rest of skill2 as a reference that advances along
             the recorded states (phase two). It is the actor and critic trained by PPO.
* Selector   a small separate net that picks the merge frame from the two windows. During
             a warmup phase it stays dormant and the merge frame comes from a fixed rule,
             so the executor first learns to reach any commanded frame. Afterwards the
             selector chooses, and learns by its own REINFORCE update on the episodes that
             finish, judged by how well the executor carried out its pick (soonest clean
             join). The update is driven once per training iteration from the runner.

* BridgeCommand  on reset, picks a couple, drops the robot at the interrupt, seeds the
                 history, encodes the two windows, and commits to a merge frame.
* TwistAction    turns the policy's twist into wheel torques, the same servo deployment
                 uses.
* reward and termination functions read the robot and the reference: closeness to the
  advancing reference (tracking), an effort penalty, and success once the robot has reached
  the merge frame and tracked the reference to the end.

There is no reference to walls or any diffdrive-specific geometry. It is loaded only through
bridge_env_cfg (its name keeps it out of mjlab's task auto-import), so importing torch and
the simulator here is fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.skills.experiments.diffdrive.bridge import features
from mjlab.tasks.skills.experiments.diffdrive.bridge.dataset import harvest_dataset
from mjlab.tasks.skills.experiments.diffdrive.experiment import CONFIG, corridor_speeds
from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import (
  BASE_HEIGHT,
  OMEGA,
  THETA,
  TRACK,
  WHEEL_RADIUS,
  V,
  X,
  Y,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

COMMAND_NAME = "bridge"


# Command term: pick a couple, drop at the interrupt, commit to a merge frame, walk a
# reference along skill2's window once the robot reaches it.


@dataclass(kw_only=True)
class BridgeCommandCfg(CommandTermCfg):
  """Configuration for the bridge command term."""

  entity_name: str
  slow: float = 0.5
  fast: float = 1.5
  cell: float = 1.0
  seed: int = 0
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> BridgeCommand:
    return BridgeCommand(self, env)


class MergeSelector(torch.nn.Module):
  """Small MLP that scores every candidate merge frame from the two windows.

  Input is the flattened window encoding (skill1 tail and skill2 start, relative to the
  interrupt); output is one score per frame of skill2's window. A softmax over the scores
  is the selector's distribution over where to merge, and its argmax is the chosen frame.
  """

  def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int) -> None:
    super().__init__()
    layers: list[torch.nn.Module] = []
    last = in_dim
    for h in hidden:
      layers += [torch.nn.Linear(last, h), torch.nn.ELU()]
      last = h
    layers.append(torch.nn.Linear(last, out_dim))
    self.net = torch.nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class BridgeCommand(CommandTerm):
  """Per-episode couple, interrupt, fixed merge frame, and the reference along skill2.

  On reset it samples a couple, places the robot at the interrupt (the last state of the
  end trajectory), seeds the history from the end trajectory's tail, encodes the two
  windows, and commits to one merge frame on skill2's window. During warmup the merge frame
  comes from a fixed rule so the executor learns to reach any commanded frame; afterwards
  the selector picks it. Each step the goal is held at the merge frame until the robot
  reaches it (phase one), then advances along the rest of skill2's window (phase two). The
  goal is in world frame so the goal-relative observation and reward need no per-env
  bookkeeping. The selector is trained by its own REINFORCE update on the episodes that
  finish.
  """

  cfg: BridgeCommandCfg

  def __init__(self, cfg: BridgeCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._env = env
    self.robot: Entity = env.scene[cfg.entity_name]

    world = GridWorld(cell=cfg.cell)
    speeds = corridor_speeds(world, slow=cfg.slow, fast=cfg.fast)
    dataset = harvest_dataset(world, speeds, seed=cfg.seed)
    self._end = torch.as_tensor(
      dataset.end_windows, dtype=torch.float32, device=self.device
    )  # [T, C, L, 5]
    self._start = torch.as_tensor(
      dataset.start_windows, dtype=torch.float32, device=self.device
    )  # [T, C, L, 5]
    # One scale per junction (its tube's own spread), so "close to the reference" is a
    # dimensionless test that does not mix different junctions.
    self._scale = torch.stack(
      [features.tube_scale(tube) for tube in self._start]
    )  # [T, 4]
    self._num_junctions, self._num_couples, self._traj_len = self._start.shape[:3]
    self._history_len = CONFIG.history_len
    self._max_steps = float(env.max_episode_length)

    self._origin_xy = env.scene.env_origins[:, :2].clone()
    left = self.robot.find_joints("left_wheel")[0][0]
    right = self.robot.find_joints("right_wheel")[0][0]
    self._wheel_ids = torch.tensor([left, right], device=self.device)

    # Per-episode state.
    self._t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._c = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._merge = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._r = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._goal = torch.zeros(self.num_envs, 5, device=self.device)
    self._history = torch.zeros(self.num_envs, self._history_len, 2, device=self.device)
    self._env_scale = torch.zeros(
      self.num_envs, self._scale.shape[1], device=self.device
    )

    # Per-episode accumulators feeding the selector reward.
    self._reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._reach_step = torch.zeros(self.num_envs, device=self.device)
    self._reach_mismatch = torch.zeros(self.num_envs, device=self.device)
    self._return = torch.zeros(self.num_envs, device=self.device)
    self._ep_step = torch.zeros(self.num_envs, device=self.device)
    self._by_selector = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # The two windows encoded once per episode (skill1 tail and skill2 start, relative to
    # the interrupt): the selector's input and the executor's context. Its width fixes the
    # observation size.
    self._window_dim = 2 * self._traj_len * 6
    self._window_feat = torch.zeros(self.num_envs, self._window_dim, device=self.device)

    # The selector and its own optimizer, a second loss separate from PPO. Samples of
    # finished episodes pile up here and are consumed by update_selector each iteration.
    self._selector = MergeSelector(
      self._window_dim, CONFIG.selector_hidden, self._traj_len
    ).to(self.device)
    self._sel_opt = torch.optim.Adam(self._selector.parameters(), lr=CONFIG.selector_lr)
    self._sel_log: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    self._step = 0  # control steps seen, for the warmup switch

  @property
  def command(self) -> torch.Tensor:
    return self._goal

  @property
  def history(self) -> torch.Tensor:
    """The recent (v, omega) window, flattened: [num_envs, 2 * history_len]."""
    return self._history.reshape(self.num_envs, -1)

  @property
  def windows(self) -> torch.Tensor:
    """The two windows encoded relative to the interrupt: [num_envs, window_dim]."""
    return self._window_feat

  @property
  def window_dim(self) -> int:
    return self._window_dim

  @property
  def selector(self) -> MergeSelector:
    """The selector net, for export at checkpoint time."""
    return self._selector

  @property
  def reached(self) -> torch.Tensor:
    """Whether the robot has reached its merge frame this episode: [num_envs]."""
    return self._reached

  @property
  def merge_state(self) -> torch.Tensor:
    """The chosen merge frame per env, in world frame: [num_envs, 5] (for the viewer)."""
    state = self._start[self._t, self._c, self._merge].clone()
    state[:, :2] += self._origin_xy
    return state

  def distance_to_goal(self, state: torch.Tensor) -> torch.Tensor:
    """Normalized distance from each robot state to its current reference: [num_envs]."""
    scale = self._env_scale.unsqueeze(1)  # [N, 1, 4], per env's junction
    return features.tube_distance(state, self._goal.unsqueeze(1), scale).squeeze(1)

  def at_end(self) -> torch.Tensor:
    """Whether the reference has been tracked to the last recorded state."""
    return self._r >= self._traj_len - 1

  def _reference_world(self) -> torch.Tensor:
    """The current reference state per env, in world frame: [num_envs, 5]."""
    ref = self._start[self._t, self._c, self._r].clone()
    ref[:, :2] += self._origin_xy
    return ref

  def _update_metrics(self) -> None:
    pass

  def _update_command(self) -> None:
    self._step += 1
    state = _robot_state(self)
    self._ep_step += 1
    dist = self.distance_to_goal(state)

    # Phase one: record the first time the robot reaches the (fixed) merge frame, and how
    # long it took and how cleanly it landed (the selector reward reads these).
    newly = (~self._reached) & (dist < CONFIG.merge_tol)
    self._reach_step = torch.where(newly, self._ep_step, self._reach_step)
    self._reach_mismatch = torch.where(newly, dist, self._reach_mismatch)
    self._reached = self._reached | newly

    # Phase two: once reached, walk the reference along the rest of skill2's window.
    advance = self._reached & (dist < CONFIG.track_tol) & ~self.at_end()
    self._r = torch.where(advance, self._r + 1, self._r)
    self._goal = self._reference_world()

    # An executor-return proxy, used only when the selector reward is in mode A.
    action_sq = torch.sum(self._env.action_manager.action**2, dim=-1)
    self._return += torch.exp(-dist) - CONFIG.effort_weight * action_sq

    recent = state[:, [V, OMEGA]].unsqueeze(1)  # [N, 1, 2]
    self._history = torch.cat([self._history[:, 1:, :], recent], dim=1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    k = len(env_ids)
    if k == 0:
      return
    device = self.device

    # Log the episodes that just finished whose merge frame the selector chose, before
    # their per-episode state is overwritten below.
    self._log_finished(env_ids)

    t = torch.randint(self._num_junctions, (k,), device=device)
    c = torch.randint(self._num_couples, (k,), device=device)
    self._t[env_ids] = t
    self._c[env_ids] = c
    self._env_scale[env_ids] = self._scale[t]

    end_couple = self._end[t, c]  # [k, L, 5], the skill1 end trajectory
    start_couple = self._start[t, c]  # [k, L, 5], the skill2 start trajectory
    feat = features.window_features(end_couple, start_couple)  # [k, window_dim]
    self._window_feat[env_ids] = feat

    merge, by_selector = self._pick_merge(feat)
    self._merge[env_ids] = merge
    self._r[env_ids] = merge
    self._by_selector[env_ids] = by_selector

    # Reset the accumulators for the new episode.
    self._reached[env_ids] = False
    self._reach_step[env_ids] = 0.0
    self._reach_mismatch[env_ids] = 0.0
    self._return[env_ids] = 0.0
    self._ep_step[env_ids] = 0.0

    self._history[env_ids] = end_couple[:, -self._history_len :][:, :, [V, OMEGA]]

    origin = self._origin_xy[env_ids]
    goal = start_couple[
      torch.arange(k, device=device), merge
    ].clone()  # the merge frame
    goal[:, :2] += origin
    self._goal[env_ids] = goal

    interrupt = end_couple[:, -1]  # [k, 5], where skill1 hands off
    self._write_interrupt(env_ids, interrupt, origin)

  def _pick_merge(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose a merge frame per env: a fixed rule during warmup, else the selector."""
    k = feat.shape[0]
    if self._step < CONFIG.warmup_steps:
      no = torch.zeros(k, dtype=torch.bool, device=self.device)
      return self._warmup_merge(k), no
    merge = torch.distributions.Categorical(logits=self._selector(feat)).sample()
    yes = torch.ones(k, dtype=torch.bool, device=self.device)
    return merge, yes

  def _warmup_merge(self, k: int) -> torch.Tensor:
    """The warmup merge frame per CONFIG.warmup_target: random, last, mid, first, or frac."""
    last = self._traj_len - 1
    target = CONFIG.warmup_target
    if target == "random":
      return torch.randint(self._traj_len, (k,), device=self.device)
    if target == "last":
      frame = last
    elif target == "mid":
      frame = last // 2
    elif target == "first":
      frame = 0
    else:
      frame = int(round(float(target) * last))
    return torch.full((k,), frame, dtype=torch.long, device=self.device)

  def _log_finished(self, env_ids: torch.Tensor) -> None:
    """Store (junction, couple, merge, reward) for finished selector-chosen episodes."""
    mask = self._by_selector[env_ids]
    if not bool(mask.any()):
      return
    ids = env_ids[mask]
    reward = self._selector_reward(ids)
    self._sel_log.append(
      (
        self._t[ids].cpu().numpy(),
        self._c[ids].cpu().numpy(),
        self._merge[ids].cpu().numpy(),
        reward.cpu().numpy(),
      )
    )

  def _selector_reward(self, ids: torch.Tensor) -> torch.Tensor:
    """The selector's reward on episodes ids: soonest clean join (B), or the return (A).

    Mode B scores a pick by whether the executor reached the chosen frame (reach), how soon
    (cost), how cleanly it landed (mismatch), and how far it then followed skill2 (track).
    A pick the executor never reached is penalized.
    """
    if CONFIG.selector_mode == "A":
      return self._return[ids]
    reached = self._reached[ids].float()
    cost = self._reach_step[ids] / self._max_steps
    mismatch = self._reach_mismatch[ids]
    merge = self._merge[ids]
    denom = (self._traj_len - 1 - merge).clamp_min(1).float()
    progress = ((self._r[ids] - merge).float() / denom).clamp(0.0, 1.0)
    clean = (
      CONFIG.sel_reach
      - CONFIG.sel_cost * cost
      - CONFIG.sel_mismatch * mismatch
      + CONFIG.sel_track * progress
    )
    return reached * clean - (1.0 - reached) * CONFIG.sel_unreachable

  def update_selector(self) -> dict[str, float]:
    """One REINFORCE step for the selector on the episodes finished since the last call.

    Called once per training iteration, after the PPO update, so it runs with autograd
    enabled (the rollout itself runs under inference mode, which is why the finished
    episodes are logged as plain arrays and the inputs rebuilt here). Returns a small dict
    for logging, empty during warmup when no selector-chosen episodes have been logged.
    """
    if not self._sel_log:
      return {}
    t = torch.as_tensor(
      np.concatenate([s[0] for s in self._sel_log]), device=self.device
    )
    c = torch.as_tensor(
      np.concatenate([s[1] for s in self._sel_log]), device=self.device
    )
    merge = torch.as_tensor(
      np.concatenate([s[2] for s in self._sel_log]), device=self.device
    )
    reward = torch.as_tensor(
      np.concatenate([s[3] for s in self._sel_log]),
      dtype=torch.float32,
      device=self.device,
    )
    self._sel_log.clear()

    # Judge each pick against the typical reward at the same junction, so the selector
    # learns which frame is better there, not which junction happens to be easier.
    advantage = reward.clone()
    for j in torch.unique(t):
      m = t == j
      advantage[m] -= reward[m].mean()

    feat = features.window_features(self._end[t, c], self._start[t, c]).detach()
    logits = self._selector(feat)
    logp = torch.log_softmax(logits, dim=-1)
    chosen = logp.gather(1, merge.unsqueeze(1)).squeeze(1)
    entropy = -(logp.exp() * logp).sum(-1).mean()
    loss = -(advantage * chosen).mean() - CONFIG.selector_entropy * entropy

    self._sel_opt.zero_grad()
    loss.backward()
    self._sel_opt.step()
    return {"selector": float(loss.item()), "selector_reward": float(reward.mean())}

  def _write_interrupt(
    self, env_ids: torch.Tensor, interrupt: torch.Tensor, origin: torch.Tensor
  ) -> None:
    """Place the robot so it realizes the (grid-local) interrupt reduced state."""
    x = interrupt[:, X] + origin[:, 0]
    y = interrupt[:, Y] + origin[:, 1]
    theta = interrupt[:, THETA]
    v = interrupt[:, V]
    omega = interrupt[:, OMEGA]
    z = torch.full_like(x, BASE_HEIGHT)
    zero = torch.zeros_like(x)
    quat = torch.stack([torch.cos(theta / 2), zero, zero, torch.sin(theta / 2)], dim=-1)
    lin_w = torch.stack([v * torch.cos(theta), v * torch.sin(theta), zero], dim=-1)
    ang_w = torch.stack([zero, zero, omega], dim=-1)
    root_state = torch.cat([torch.stack([x, y, z], dim=-1), quat, lin_w, ang_w], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids)

    half = TRACK / 2.0
    left = (v - omega * half) / WHEEL_RADIUS
    right = (v + omega * half) / WHEEL_RADIUS
    self.robot.write_joint_velocity_to_sim(
      torch.stack([left, right], dim=-1), joint_ids=self._wheel_ids, env_ids=env_ids
    )


# Action term: twist to wheel torques.


@dataclass(kw_only=True)
class TwistActionCfg(ActionTermCfg):
  """Configuration for the twist action term."""

  entity_name: str

  def build(self, env: ManagerBasedRlEnv) -> TwistAction:
    return TwistAction(self, env)


class TwistAction(ActionTerm):
  """Drive the wheels from a body-twist action, with the same servo as deployment.

  The policy's 2-vector becomes a clamped twist (forward speed, yaw rate); a velocity
  servo turns that into wheel torques, clamped to the actuator limit. That clamp is
  what gives the robot bounded acceleration, so residual speed cannot vanish at once.
  """

  cfg: TwistActionCfg

  def __init__(self, cfg: TwistActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._entity: Entity = env.scene[cfg.entity_name]
    left = self._entity.find_joints("left_wheel")[0][0]
    right = self._entity.find_joints("right_wheel")[0][0]
    self._wheel_ids = torch.tensor([left, right], device=self.device)
    self._raw = torch.zeros(self.num_envs, CONFIG.action_dim, device=self.device)

  @property
  def action_dim(self) -> int:
    return CONFIG.action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions

  def apply_actions(self) -> None:
    v, omega = features.twist_from_action(self._raw)
    half = TRACK / 2.0
    left_target = (v - omega * half) / WHEEL_RADIUS
    right_target = (v + omega * half) / WHEEL_RADIUS
    target = torch.stack([left_target, right_target], dim=-1)
    wheel_vel = self._entity.data.joint_vel[:, self._wheel_ids]
    torque = torch.clamp(
      CONFIG.kv * (target - wheel_vel), -CONFIG.torque_limit, CONFIG.torque_limit
    )
    self._entity.set_joint_effort_target(torque, joint_ids=self._wheel_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw[env_ids] = 0.0


# Reading the robot and the goal.


def _term(env: ManagerBasedRlEnv) -> BridgeCommand:
  return cast("BridgeCommand", env.command_manager.get_term(COMMAND_NAME))


def _robot_state(term: BridgeCommand) -> torch.Tensor:
  """The robot's reduced state [x, y, theta, v, omega] in world frame."""
  data = term.robot.data
  return torch.stack(
    [
      data.root_link_pos_w[:, 0],
      data.root_link_pos_w[:, 1],
      data.heading_w,
      data.root_link_lin_vel_b[:, 0],
      data.root_link_ang_vel_w[:, 2],
    ],
    dim=-1,
  )


# Observation: where the moving reference is, the recent-motion history, and both windows.


def bridge_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env)
  goal_relative = features.observation(_robot_state(term), term.command)
  return torch.cat([goal_relative, term.history, term.windows], dim=-1)


# Rewards: track the advancing reference, spend little effort. Nothing else.


def tracking(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Closeness to the current reference state, in (0, 1] (a smooth tracking reward)."""
  term = _term(env)
  return torch.exp(-term.distance_to_goal(_robot_state(term)))


def effort(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared action, to discourage spending energy (a penalty)."""
  return torch.sum(env.action_manager.action**2, dim=-1)


# Termination: success once the robot has reached the merge frame and tracked to the end.


def tracked_to_end(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env)
  return term.reached & term.at_end()
