"""Learned termination rollouts with independent handoffs in each environment."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import torch

from mjlab.tasks.bridging.baseline.model import (
  TransitionPolicy,
  advantages,
  proximity_rewards,
  proximity_targets,
)
from mjlab.tasks.bridging.baseline.outcome import SkillOutcome
from mjlab.tasks.bridging.baseline.scene import TransitionScene


@torch.no_grad()
def collect(
  scene: TransitionScene,
  model: TransitionPolicy,
  max_steps: int,
  decay: float,
  stochastic: bool = True,
) -> tuple[dict, dict]:
  if max_steps < 1:
    raise ValueError("max_steps must be positive")
  scene.hold_target()
  scene.run.command.stop()
  count = scene.env.num_envs
  device = scene.env.device
  active = ~scene.failed.clone()
  incoming = torch.zeros(count, dtype=torch.bool, device=device)
  finished = ~active
  age = torch.zeros(count, dtype=torch.long, device=device)
  limits = scene.downstream_steps()
  outcome = SkillOutcome(scene)
  success = torch.zeros(count, dtype=torch.bool, device=device)
  sums = torch.zeros(count, len(scene.channels), device=device)
  jumps = torch.zeros(count, device=device)
  endpoint_errors = torch.zeros(
    count, len(scene.run.command.error_names), device=device
  )
  lengths = torch.zeros(count, dtype=torch.long, device=device)
  observations, actions, values, potentials, terminals, masks, reward_stops = (
    [] for _ in range(7)
  )
  step = 0
  while not bool(finished.all()):
    scene.hold_target(active)
    observation = scene.observe().clone()
    action, value, terminal = model.act(observation, stochastic)
    terminal = terminal.bool() | (step == max_steps - 1)
    if bool(active.any()):
      observations.append(observation)
      actions.append(action)
      values.append(value)
      potentials.append(model.potential(observation))
      terminals.append(terminal.clone())
      reward_stops.append(terminal.clone())
      masks.append(active.clone())
      lengths += active.long()
    ending = active & terminal
    incoming |= ending
    if bool(ending.any()):
      errors, _ = scene.endpoint()
      endpoint_errors[ending] = errors[ending]
    successor_action = scene.skill_action(scene.target_skill)
    if bool(ending.any()):
      delta = scene.joint_targets(successor_action) - scene.joint_targets(
        scene.env.action_manager.action
      )
      jumps[ending] = delta[ending].square().mean(-1).sqrt()
    active &= ~terminal
    if not bool((active | incoming).any()):
      break
    scene.step(torch.where(active[:, None], action, successor_action))
    scene.failed |= scene.target_failures.compute() & incoming
    outcome.update(active, incoming)
    errors = scene.downstream_errors()
    sums[incoming] += errors[incoming].square()
    age += incoming.long()
    crashed = active & scene.failed
    if bool(crashed.any()):
      # Upstream marks the last applied action terminal if the simulator ends
      terminals[-1][crashed] = True
      active &= ~crashed
      finished |= crashed
    completed = incoming & ((age >= limits) | scene.failed)
    success[completed] = outcome.success()[completed] & torch.isfinite(
      sums[completed]
    ).all(-1)
    finished |= completed
    incoming &= ~completed
    step += 1
  traces = {
    name: []
    for name in (
      "observations",
      "actions",
      "values",
      "terminal",
      "advantages",
      "returns",
      "targets",
    )
  }
  returns = []
  stacked = (
    [
      torch.stack(items)
      for items in (
        observations,
        actions,
        values,
        terminals,
        reward_stops,
        potentials,
        masks,
      )
    ]
    if observations
    else []
  )
  for env_id in range(count):
    if not observations or not int(lengths[env_id]):
      continue
    selected = stacked[-1][:, env_id]
    obs, act, val, term, stop, potential = [
      item[:, env_id][selected] for item in stacked[:-1]
    ]
    reward = proximity_rewards(potential, stop)
    adv, ret = advantages(reward, val)
    labels = proximity_targets(success[env_id : env_id + 1], len(obs), decay)[:, 0]
    for name, tensor in zip(
      traces, (obs, act, val, term, adv, ret, labels), strict=True
    ):
      traces[name].append(tensor)
    returns.append(reward.sum())
  if not traces["observations"]:
    raise RuntimeError("Every source execution failed before the transition request")
  batch = {name: torch.cat(tensors) for name, tensors in traces.items()}
  rms = (sums / age.clamp_min(1)[:, None]).sqrt()
  record = {
    "success_rate": float(success.float().mean()),
    "reward_mean": float(torch.stack(returns).mean()),
    "duration_s": float((lengths - 1).clamp_min(0).float().mean() * scene.env.step_dt),
    "action_jump_rad": float(jumps.mean()),
    "transition_steps": float(lengths.float().mean()),
    "target_errors": endpoint_errors.mean(0).tolist(),
    "downstream_rms": rms.mean(0).tolist(),
    "success": success,
  }
  return batch, record
