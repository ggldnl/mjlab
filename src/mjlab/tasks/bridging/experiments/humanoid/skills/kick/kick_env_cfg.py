"""The kick environment: the pass, with a strike that has to be fast to count.

A wrapper around `g1_pass_env_cfg` rather than a copy of it, the way run_env_cfg wraps the
velocity task. Everything the pass decided about standing, latching, curricula and
regularizers is decided once, and what is written here is only what a kick disagrees with.

What to watch, on top of everything the pass lists:

    Episode/rew_shove_cost        should rise from its early trough toward zero. While it
                                  is large and negative the policy is still pushing,
                                  whatever the launch metrics say
    Episode/rew_launch_progress   the new rung. Expected to move before pass_quality does,
                                  because it asks only for speed and not for aim
    Metrics/pass/speed_achieved   what launch_progress is paid on, in m/s, and the number
                                  that says whether this is a kick yet
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.passing.pass_env_cfg import (
  STRIKE_STAGE,
  W_APPROACH,
  g1_pass_env_cfg,
)

##
# Where the ball goes.
##

BALL_FORWARD_RANGE = (0.32, 0.40)
"""Forward offset of the ball's centre from the striking foot's site, in metres.

Exactly the pass's 0.24 to 0.32 shifted out by the eight centimetres between the site it
measures from and the toe this task measures from, so the gap the policy has to close is
the same one the pass already solves. That is the point. The reach problem is not what
makes this a kick, and the version of this task that tried to make it so is what taught the
robot to fall over.

The reasoning that failed is worth writing down, because it is seductive. Put the ball out
at the edge of a standing robot's reach and a shove becomes impossible, so the only contact
left is a swing. What actually happens is that `approach_toe` is a dense position kernel
with no balance qualifier, so its maximum is wherever the ball is, and if that is at the
edge of the reach envelope then the steepest path to it is a forward topple. The policy is
not choosing to fall. It is following the only reaching gradient on offer, and falling is
where that gradient ends.

So the ball is kept comfortably reachable and the shove is ruled out by what it costs
instead. See `shove_cost` in mdp.py."""

BALL_LATERAL_RANGE = (-0.05, 0.05)
"""Unchanged from the pass. Widen it to cover the other foot's line and the policy has to
choose a leg, which is a much harder problem than the one being asked."""

##
# What the new terms are worth.
##

W_SHOVE = -1.0
"""Cost per step of holding the ball in contact with a slow toe.

Large next to the pass's -0.5 early_strike, and it can be, because unlike that one this
fires only on a contact the task is trying to rule out and a clean strike pays it for a
step or two at most."""

W_LAUNCH = 2.0
"""Weight of the speed rung.

Between the pass's ball_touched at 1.0 and its pass_quality at 5.0, which is where it sits
in the ladder: touch the ball, then hit it hard, then hit it hard in the right direction."""

##
# How far the robot may lean.
##

FELL_OVER_ANGLE = math.radians(50.0)
"""Tilt at which the episode is called a failure. The pass allows 70.

Tightened because 70 degrees leaves a wide band of deep leans that are not yet a
termination and still collect `alive`, `stay_put` and a good part of `approach_toe`, which
is a comfortable place for a policy to sit and the last stop before falling. A kick needs
nothing like 50 degrees of torso tilt, and neither does standing, so nothing the task wants
is lost by refusing it."""


def g1_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The pass environment, moved onto the toe.

  Five changes and nothing else:

      the ball is placed further out by the toe's own offset, so the approach is unchanged
      `approach_ball` is measured from the toe instead of the mid-foot site
      `shove_cost` charges for a contact made with a toe that is not moving
      `launch_progress` pays for the ball reaching the commanded speed, aim aside
      the lean allowed before a fall is called is tightened

  The observation grows by seven numbers, which makes this a task that has to be trained
  from scratch. The pass's checkpoint will not load into it.
  """
  cfg = g1_pass_env_cfg(play=play)

  # The ball, further out by the toe's offset. Mutating the event's params rather than
  # rebuilding the term, because the term also carries the placement function that squares
  # the ball up with the striking foot, and restating that here is how the two would drift
  # apart
  ball = cfg.events["reset_ball"]
  ball.params["forward_range"] = BALL_FORWARD_RANGE
  ball.params["lateral_range"] = BALL_LATERAL_RANGE

  # The approach, measured from the toe. Same kernel width: the gap it measures is a gap to
  # the ball's surface either way, so the scale of the problem has not changed
  cfg.rewards["approach_ball"] = RewardTermCfg(
    func=mdp.approach_toe, weight=W_APPROACH, params={"std": 0.15}
  )
  cfg.rewards["launch_progress"] = RewardTermCfg(
    func=mdp.launch_progress, weight=W_LAUNCH, params={"command_name": mdp.COMMAND_NAME}
  )
  cfg.rewards["shove_cost"] = RewardTermCfg(func=mdp.shove_cost, weight=W_SHOVE)

  cfg.terminations["fell_over"].params["limit_angle"] = FELL_OVER_ANGLE

  # The pass's curriculum entry for approach_ball is inherited and still correct: it names
  # the reward by its key and this replaced the term under that same key.
  #
  # Asserted rather than assumed, because the inheritance is invisible from here: nothing
  # in this file mentions the ramp that decides when approach_ball starts paying, and a
  # rename on the pass's side would silently leave this task training at full weight from
  # step zero. Play mode drops the curriculum entirely, hence the guard
  assert play or "approach_ball" in cfg.curriculum

  if not play:
    # Both new terms ramp on the pass's schedule, the penalty included. An earlier version
    # held shove_cost out of the curriculum, on the argument that a penalty arriving late
    # is one the policy has already built a motion around. True, and it produces the
    # opposite failure: with every positive strike term zeroed for the first stage, a
    # penalty at full weight from step zero is the entire reward signal concerning the
    # ball, and what gets learned in three hundred iterations is to stay away from it.
    # Ramping together holds the ratio fixed while the ladder is climbed, which is what the
    # ramp is for
    cfg.curriculum["launch_progress"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "launch_progress",
        "weight_stages": mdp.ramp(W_LAUNCH, STRIKE_STAGE),
      },
    )
    cfg.curriculum["shove_cost"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "shove_cost",
        "weight_stages": mdp.ramp(W_SHOVE, STRIKE_STAGE),
      },
    )

  # The toe in the observation, for the reason the pass gives about its stance window. Its
  # velocity is there because that is what shove_cost gates on, and a gate the policy
  # cannot see is a reward that changes for no visible reason. The critic's terms were
  # copied from the actor's when the pass built them, so the two dicts are separate objects
  # by now and each has to be told
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["toe_pos"] = ObservationTermCfg(func=mdp.toe_pos_b)
    terms["toe_vel"] = ObservationTermCfg(func=mdp.toe_vel_b)
    terms["toe_height"] = ObservationTermCfg(func=mdp.toe_height)

  return cfg
