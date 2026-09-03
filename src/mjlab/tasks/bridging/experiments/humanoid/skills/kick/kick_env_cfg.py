"""The kick environment: the pass, with the strike moved onto the toe and off the floor.

A wrapper around `g1_pass_env_cfg` rather than a copy of it, the way run_env_cfg wraps the
velocity task. Everything the pass decided about standing, latching, curricula and
regularizers is decided once, and what is written here is only what a kick disagrees with.

What to watch, on top of everything the pass lists:

    Episode/rew_sole_strike       should rise from its early trough toward zero. While it
                                  is large and negative the foot is still on the floor when
                                  it meets the ball, which is the shove
    Episode/rew_shove_cost        the same story told by contact speed rather than by the
                                  floor. The two normally move together
    Episode/rew_launch_progress   the new rung. Expected to move before pass_quality does,
                                  because it asks only for speed and not for aim
    Metrics/pass/speed_achieved   what launch_progress is paid on, in m/s, and the number
                                  that says whether this is a kick yet
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.passing.pass_env_cfg import (
  POSTURE_STD,
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
left is a swing. What actually happens is that the approach reward is a dense position
kernel with no balance qualifier, so its maximum is wherever the ball is, and if that is at
the edge of the reach envelope then the steepest path to it is a forward topple. The policy
is not choosing to fall. It is following the only reaching gradient on offer, and falling is
where that gradient ends.

So the ball is kept comfortably reachable and the shove is ruled out by what it costs. See
`sole_strike` and `shove_cost` in mdp.py."""

BALL_LATERAL_RANGE = (-0.05, 0.05)
"""Unchanged from the pass. Widen it to cover the other foot's line and the policy has to
choose a leg, which is a much harder problem than the one being asked."""

##
# What the new terms are worth.
##

W_SOLE = -1.0
"""Cost per step of meeting the ball with the foot on the floor, or with the sole leading.

The term that names the failure. Large next to the pass's -0.5 early_strike, and it can be,
because unlike that one it fires only on a contact the task is trying to rule out, and a
clean strike never fires it at all."""

W_SHOVE = -1.0
"""Cost per step of holding the ball in contact with a slow toe.

Overlaps `sole_strike` on the motion both are aimed at, and that is deliberate: one says
the foot was on the floor, the other says it was not moving, and a shove is both. Together
they come to -2 per step of shoving against a floor worth about four, which is meant to be
uncomfortable rather than fatal. If mean episode length falls monotonically from the first
iteration these are the two numbers to halve."""

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
termination and still collect `alive`, `stay_put` and a good part of the approach, which is
a comfortable place for a policy to sit and the last stop before falling. A kick needs
nothing like 50 degrees of torso tilt, and neither does standing."""

##
# How far the striking leg may be loaded.
##

SWING_POSTURE_STD = {
  **POSTURE_STD,
  r"right_hip_pitch.*": 1.6,
  r"right_knee.*": 2.0,
}
"""The pass's posture tolerances with the striking leg's two big joints let out.

A windup is hip extension and knee flexion held for a moment before the swing, which is a
long way from the stance the posture term measures against. The pass's 1.0 and 1.2 were
chosen for a leg that reaches rather than one that loads, and a term that pays most for
standing still is a term that argues against the backswing every step it is held.

Only these two. The support leg, the waist and the ankles stay where the pass put them,
because those are what the robot is balancing on while the swing happens."""


def g1_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The pass environment, moved onto the toe.

  What differs from the pass:

      the toe and the sole are marked with sites, added to the spec from Python
      the ball is placed further out by the toe's own offset, so the approach is unchanged
      `approach_ball` is measured from the toe, and on the best gap so far, not the live one
      `sole_strike` charges for meeting the ball with the foot planted or the sole leading
      `shove_cost` charges for meeting the ball with a toe that is not moving
      `launch_progress` pays for the ball reaching the commanded speed, aim aside
      the striking leg's posture tolerance is widened so a windup is not charged for
      the lean allowed before a fall is called is tightened

  The observation grows by eight numbers, which makes this a task that has to be trained
  from scratch. The pass's checkpoint will not load into it.
  """
  cfg = g1_pass_env_cfg(play=play)

  # The two markers, added to the robot's spec here rather than to g1.xml. Both are read by
  # the terms below and both draw in the viewer, so what the reward is measuring from can
  # be looked at instead of taken on trust
  entities = cfg.scene.entities
  entities["robot"] = mdp.add_strike_sites(entities["robot"])

  # The striking foot against the floor, on its own sensor. See FOOT_GROUND_SENSOR in
  # mdp.py for why this is not read out of the pass's two footed one
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ContactSensorCfg(
      name=mdp.FOOT_GROUND_SENSOR,
      primary=ContactMatch(mode="subtree", pattern=mdp.STRIKE_BODY, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )

  # The ball, further out by the toe's offset. Mutating the event's params rather than
  # rebuilding the term, because the term also carries the placement function that squares
  # the ball up with the striking foot, and restating that here is how the two would drift
  # apart
  ball = cfg.events["reset_ball"]
  ball.params["forward_range"] = BALL_FORWARD_RANGE
  ball.params["lateral_range"] = BALL_LATERAL_RANGE

  # Registered after the pass's own reset events, which is fine: the latched gap is cleared
  # to a constant and reads nothing off the scene
  cfg.events["reset_kick_phase"] = EventTermCfg(func=mdp.reset_kick_phase, mode="reset")

  # The approach, measured from the toe and paid on the best gap so far. Same kernel width:
  # the gap is a gap to the ball's surface either way, so the scale has not changed
  cfg.rewards["approach_ball"] = RewardTermCfg(
    func=mdp.approach_toe, weight=W_APPROACH, params={"std": 0.15}
  )
  cfg.rewards["launch_progress"] = RewardTermCfg(
    func=mdp.launch_progress, weight=W_LAUNCH, params={"command_name": mdp.COMMAND_NAME}
  )
  cfg.rewards["sole_strike"] = RewardTermCfg(func=mdp.sole_strike, weight=W_SOLE)
  cfg.rewards["shove_cost"] = RewardTermCfg(func=mdp.shove_cost, weight=W_SHOVE)

  cfg.rewards["posture"].params["std"] = SWING_POSTURE_STD
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
    # Every new term ramps on the pass's schedule, the penalties included. An earlier
    # version held the penalty out of the curriculum, on the argument that one arriving
    # late is one the policy has already built a motion around. True, and it produces the
    # opposite failure: with every positive strike term zeroed for the first stage, a
    # penalty at full weight from step zero is the entire reward signal concerning the
    # ball, and what gets learned in three hundred iterations is to stay away from it.
    # Ramping together holds the ratio fixed while the ladder is climbed
    for name, final in (
      ("launch_progress", W_LAUNCH),
      ("sole_strike", W_SOLE),
      ("shove_cost", W_SHOVE),
    ):
      cfg.curriculum[name] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={"reward_name": name, "weight_stages": mdp.ramp(final, STRIKE_STAGE)},
      )

  # The strike point in the observation, for the reason the pass gives about its stance
  # window. Its velocity is there because that is what shove_cost gates on, and the best
  # gap because that is what the approach is paid on: a gate the policy cannot see is a
  # reward that changes for no visible reason. The critic's terms were copied from the
  # actor's when the pass built them, so the two dicts are separate objects by now and each
  # has to be told
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["toe_pos"] = ObservationTermCfg(func=mdp.toe_pos_b)
    terms["toe_vel"] = ObservationTermCfg(func=mdp.toe_vel_b)
    terms["toe_height"] = ObservationTermCfg(func=mdp.toe_height)
    terms["approach_best"] = ObservationTermCfg(func=mdp.approach_best)

  return cfg
