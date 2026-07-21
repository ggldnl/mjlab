"""Runs a controller, skill pool, and bridge(s) together against an env.

This is a reusable template, not an experiment. A concrete experiment builds real
Skill/Controller/Bridge instances and a real env, then calls `run_episode` with them.
The __main__ block below only raises, since this file has nothing concrete to run
on its own.

`_select_bridge` assumes every env in a batch is on the same (source, target) pair at
a given step, which holds for a single-experiment run where every env follows the
same skill sequence. It will need revisiting once envs can be mid-transition on
different pairs simultaneously.
"""

from collections.abc import Mapping

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.bridge import Bridge
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.skill import NO_SKILL, SkillPool

BridgeSet = Bridge | Mapping[tuple[int, int], Bridge]


def _select_bridge(bridges: BridgeSet, source: int, target: int) -> Bridge:
  if isinstance(bridges, Bridge):
    return bridges
  return bridges[(source, target)]


def run_episode(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  controller: Controller,
  bridges: BridgeSet,
  num_steps: int,
) -> None:

  obs, _ = env.reset()
  num_envs = env.num_envs
  device = env.device

  # The skill the controller is currently committed to (running skill in normal operation,
  # or the skill a bridge is currently driving toward)
  target = controller.decide(env, torch.full((num_envs,), NO_SKILL, device=device))

  # The skill the controller is currently committed to before the switch:
  # NO_SKILL until the first switch ever fires
  source = torch.full_like(target, NO_SKILL)

  # True where a bridge, not a skill, is currently producing actions
  bridging = torch.zeros(num_envs, dtype=torch.bool, device=device)

  for _ in range(num_steps):
    # Ask the controller what should be running right now. If it names a
    # different skill than `target`, that's the switch signal (per env, not
    # globally, so different envs can switch on different steps).
    new_target = controller.decide(env, target)
    switching = new_target != target

    if switching.any():
      # A switch just fired in at least one env: hand those envs to the bridge
      # going from the old `target` (about to become `source`) to `new_target`.

      # Select the bridge: the experiment could be set up in such a way we have
      # a single shared bridge or one bridge per (source, target) pair.
      bridge = _select_bridge(bridges, int(source[0]), int(new_target[0]))

      # `begin` is the bridge's one chance to reset any state it keeps before its
      # first `act` call for this transition.
      bridge.begin(switching, source, new_target)
      source = torch.where(switching, target, source)
      target = new_target
      bridging = bridging | switching

    # Build this step's actions. Envs mid-bridge are excluded from the skill
    # assignment (set to NO_SKILL, so SkillPool.act returns zeros for them) and
    # filled in with the bridge's actions instead.
    assignment = torch.where(bridging, torch.full_like(target, NO_SKILL), target)
    actions = pool.act(obs, assignment)
    if bridging.any():
      bridge = _select_bridge(bridges, int(source[0]), int(target[0]))
      bridge_actions, handover = bridge.act(obs, source, target, bridging)

      # Overwrite the (currently zero) actions of bridging envs with the bridge's.
      actions = torch.where(bridging.unsqueeze(-1), bridge_actions, actions)

      # Wherever the bridge just signaled handover, control passes to the skill
      # named by `target` starting next step.
      bridging = bridging & ~handover

    # Step the sim, then clean up envs that ended (naturally or by timeout): the
    # pool, controller, and our own source/bridging bookkeeping should not carry
    # state across an episode boundary, and a done env needs a fresh target for
    # the episode that's about to start.
    obs, _, terminated, time_out, _ = env.step(actions)
    done = terminated | time_out
    if done.any():
      pool.reset(done)
      controller.reset(done)
      source = torch.where(done, torch.full_like(source, NO_SKILL), source)
      bridging = bridging & ~done
      target = torch.where(done, controller.decide(env, target), target)


if __name__ == "__main__":
  raise SystemExit(
    """
    main.py is a template: import run_episode from an experiment's own entry point
    with concrete Skill/Controller/Bridge instances, rather than running 
    this file directly.
    """
  )
