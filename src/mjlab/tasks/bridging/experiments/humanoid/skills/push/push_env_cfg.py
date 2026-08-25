"""The push environment: mjlab's G1 flat velocity task with a crate in the way.

Built by wrapping the locomotion task rather than assembling a new one from parts, which
is the opposite of what the kick does and for the opposite reason. The kick is a standing
task: it has no commanded twist, no gait, and no use for a terrain, so almost nothing in
the velocity config carried over. Pushing is locomotion. The robot still walks, still
tracks a commanded velocity, still has to keep its feet clear and its torso up, and every
one of those terms is already tuned. What changes is that there is now a one metre crate
between the robot and wherever it was told to go.

The reward set is the velocity task's, plus a ladder with three rungs.

  Floor      everything mjlab's flat G1 task already pays: velocity tracking, upright,
             posture, foot clearance and slip, and the regularizers. This is worth about
             six per step to a robot that is simply walking, and it is what the push
             terms have to be measured against.

  Rung one   approach_box. Dense, a kernel on the distance from the robot to the crate's
             surface, reaching one when it is against it. Ungated, so it also pulls the
             robot back if the contact is lost.

  Rung two   box_contact. Paid for every step the contact holds. Contact is the discrete
             step between walking up to a crate and moving it, and the distance kernel
             cannot express it.

  Rung three push_tracking. The fraction of the commanded velocity the crate is actually
             carrying. This is the task, and it is the only term on the curriculum.

Only the top rung is ramped in. The lower two agree with learning to walk from the first
iteration -- they ask the robot to go somewhere and lean on something, which is
locomotion -- while a large weight on the crate's velocity is a reward a robot that
cannot yet walk has no way to earn, and the cheapest way to go looking for it is to
throw itself at the crate.

Run it:

    uv run train Mjlab-Parkour-Push --env.scene.num-envs 4096
    uv run play Mjlab-Parkour-Push

What to watch, in order of how much it tells you:

    Episode_Metrics/push_contact_rate       fraction of the episode spent on the crate
    Episode_Metrics/push_box_displacement   furthest the crate got, in metres
    Episode_Metrics/push_box_speed          how fast it was moving
    Episode/rew_push_tracking               the goal term itself
    Curriculum/push_tracking                whether that weight has been turned on yet

push_contact_rate is the one that separates "learning to push" from "learning to walk
around a crate". If it sits near zero while the velocity tracking reward climbs, the
policy has found the detour, and the answer is a larger approach_box weight rather than a
larger push_tracking one.
"""

from __future__ import annotations

import math

from mjlab.asset_zoo.objects.box import get_box_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.push import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

##
# The crate.
##

# Nominal mass of the one metre cube, in kg.
#
# The size is what makes it a big object; the mass is what makes it a learnable one, and
# the two are set independently on purpose. Neither the terrain nor the G1's collision
# geoms declare a contact priority, so MuJoCo takes the elementwise maximum of the two
# frictions and the crate slides against the ground at the terrain's own 1.0 whatever the
# asset asks for. At that friction shifting the crate costs its full weight in horizontal
# force: 6 kg is about 59 N, roughly a sixth of what a 35 kg G1 weighs, which is a firm
# lean rather than a shove. This is the one number to raise for a heavier task, and
# raising it past about 15 kg asks for more traction than the feet have on a bad friction
# draw.
BOX_MASS = 6.0 * 2

# Where the crate spawns, relative to the robot and in the robot's own heading frame.
# Forward is measured centre to centre, so the near face sits half a metre closer than
# the number says and the robot starts roughly 0.6 to 1.1 m from it. Far enough that
# getting there is walking, near enough that it happens inside the first second or two.
BOX_FORWARD_RANGE = (1.1, 1.6)
BOX_LATERAL_RANGE = (-0.25, 0.25)

# Yaw scatter of the crate itself. Enough that the policy meets a face at an angle as
# often as square on, small enough that it never has to push a corner.
BOX_YAW_RANGE = (math.radians(-20.0), math.radians(20.0))

##
# What the push is asked for.
##

# Environment steps per command curriculum stage. common_step_counter advances once per
# environment step, so at the default 24 steps per environment per iteration this is
# about 2000 iterations a stage.
_SPEED_STAGE = 2000 * 24

# Strictly forward, and never zero. A crate that the robot is asked to reverse away from
# is not a push, and an environment commanded to stand still would collect the top rung
# for free the moment it is standing next to a stationary box, because a box matching a
# zero command is a box doing nothing.
SPEED_STAGES: list[mdp.VelocityStage] = [
  {
    "step": 0,
    "lin_vel_x": (0.3, 0.6),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.3, 0.3),
  },
  {
    "step": _SPEED_STAGE,
    "lin_vel_x": (0.3, 0.9),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.3, 0.3),
  },
  {
    "step": 2 * _SPEED_STAGE,
    "lin_vel_x": (0.3, 1.2),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.3, 0.3),
  },
]

##
# Curriculum.
##

# Environment steps per stage of the push weight ramp: off for the first stage, half
# strength for the second, full after that. About 500 iterations a stage, which is enough
# for a G1 that starts in its own stance keyframe to be walking before the crate's
# velocity starts paying.
PUSH_STAGE = 500 * 24

# Final weights of the ladder. Declared here rather than inline because the curriculum has
# to ramp to exactly the third one, and the two drifting apart would silently leave the
# task training at a weight nobody chose.
W_APPROACH = 1.0
W_CONTACT = 1.0
W_PUSH = 3.0


def g1_push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the push environment.

  Args:
    play: Drop the observation noise and the curriculum, leave the reward weights at
      their final values, and fix the command range at the top of the speed schedule so a
      trained policy is watched at the speed it was trained up to. Inherited from the
      flat velocity task, which also stops the clock and stops shoving the robot.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  ##
  # Scene
  ##

  cfg.scene.entities["box"] = get_box_cfg(
    half_size=mdp.BOX_HALF_SIZE,
    # Overwritten by the reset event on the first step. It matters only to a raw viewer
    # that never resets.
    init_x=BOX_FORWARD_RANGE[0],
    mass=BOX_MASS,
    color=(0.55, 0.4, 0.25, 1.0),
  )

  # Whole robot against the crate, because a G1 walking into one metre of cube meets it
  # with whatever is in front: hands, forearms, chest, hips. Naming the parts would be
  # deciding the technique in advance. The subtree below the pelvis is the whole robot,
  # which is the same trick the self collision sensor uses.
  robot_box_cfg = ContactSensorCfg(
    name=mdp.ROBOT_BOX_SENSOR,
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="box_collision", entity="box"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (robot_box_cfg,)

  # The crate brings its own contacts: four corners on the ground plus however many the
  # robot makes against it, on top of the fourteen foot capsules the G1 already puts down.
  # The failure when this is too small is quiet -- MuJoCo Warp prints "nefc overflow" and
  # then drops the constraints past the limit, so the run keeps going while the physics
  # silently stops being right -- so grep a training log for it before trusting a result.
  cfg.sim.njmax = 400
  cfg.sim.contact_sensor_maxmatch = 128

  ##
  # Observations
  ##

  # The crate, in the robot's own heading frame. Without these the task is not conditioned
  # on anything the policy can act on: it would be asked to track a velocity while
  # something invisible blocked it.
  box_terms = {
    "box_pos": ObservationTermCfg(
      func=mdp.box_pos_b, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "box_vel": ObservationTermCfg(
      func=mdp.box_vel_b, noise=Unoise(n_min=-0.1, n_max=0.1)
    ),
    "box_contact": ObservationTermCfg(func=mdp.box_contact),
  }
  cfg.observations["actor"].terms.update(box_terms)
  cfg.observations["critic"].terms.update(
    {
      **{name: ObservationTermCfg(func=term.func) for name, term in box_terms.items()},
      # How hard the robot is leaning, which stands in for the crate's mass. The reset
      # randomizes that and the actor is never told it, so a critic without this is
      # valuing a state it cannot see.
      "box_contact_force": ObservationTermCfg(func=mdp.box_contact_force),
    }
  )

  ##
  # Commands
  ##

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # Mostly straight ahead, and never standing. Sidestepping and turning are what a policy
  # reaches for to get out from behind the crate, and every standing environment is one
  # not spent pushing.
  twist_cmd.rel_forward_envs = 0.6
  twist_cmd.rel_standing_envs = 0.0
  # No heading environments. A heading target is absolute and sampled over the full
  # circle, while the robot's yaw at reset is too, so a heading environment is asked to
  # turn by up to half a turn -- away from the crate that was just placed in front of it.
  # `heading_command` itself stays on: with no environment claiming a heading the target
  # is sampled and ignored, whereas turning the flag off changes what `ranges.heading`
  # is allowed to be and has bitten this repo before.
  twist_cmd.rel_heading_envs = 0.0
  # Held longer than the stock 3-8 s. A crate takes a second or two to get moving, and a
  # command that resamples on that timescale means the policy spends its life starting
  # pushes and never finishing one.
  twist_cmd.resampling_time_range = (6.0, 12.0)

  # The stage-zero range. The curriculum overwrites this from the first step, but setting
  # it here keeps the config honest about where it begins.
  first = SPEED_STAGES[0]
  assert first["lin_vel_x"] is not None
  assert first["lin_vel_y"] is not None
  assert first["ang_vel_z"] is not None
  twist_cmd.ranges.lin_vel_x = first["lin_vel_x"]
  twist_cmd.ranges.lin_vel_y = first["lin_vel_y"]
  twist_cmd.ranges.ang_vel_z = first["ang_vel_z"]

  ##
  # Events
  ##

  # Appended, so they run after the robot has been placed: the crate is put down relative
  # to where the robot actually ended up, and its spawn is recorded off that.
  cfg.events["reset_box"] = EventTermCfg(
    func=mdp.reset_box_in_front,
    mode="reset",
    params={
      "forward_range": BOX_FORWARD_RANGE,
      "lateral_range": BOX_LATERAL_RANGE,
      "yaw_range": BOX_YAW_RANGE,
    },
  )
  # Mass and inertia together, which is what pseudo_inertia buys over body_mass: the
  # latter leaves the inertia tensor behind and gives a crate that resists translation
  # and rotation by different amounts. alpha is a log density scale, so this range is
  # about 0.6 to 1.6 times BOX_MASS, or roughly 4 to 10 kg.
  cfg.events["box_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("box", body_names=("box",)),
      "alpha_range": (-0.25, 0.25),
    },
  )

  ##
  # Rewards
  ##

  cfg.rewards["approach_box"] = RewardTermCfg(
    func=mdp.approach_box,
    weight=W_APPROACH,
    # The robot spawns about 0.6 to 1.1 m off the crate's face, so this puts the spawn
    # partway up the kernel rather than out on its flat tail.
    params={"std": 0.6},
  )
  cfg.rewards["box_contact"] = RewardTermCfg(
    func=mdp.box_contact_reward, weight=W_CONTACT
  )
  cfg.rewards["push_tracking"] = RewardTermCfg(
    func=mdp.push_tracking,
    weight=W_PUSH,
    params={"command_name": mdp.COMMAND_NAME},
  )

  ##
  # Metrics
  ##

  cfg.metrics["push_contact_rate"] = MetricsTermCfg(func=mdp.contact_rate)
  cfg.metrics["push_box_speed"] = MetricsTermCfg(func=mdp.box_speed)
  # Furthest the crate got, not the average over a trajectory that started at zero.
  cfg.metrics["push_box_displacement"] = MetricsTermCfg(
    func=mdp.box_displacement, reduce="max"
  )

  ##
  # Curriculum
  ##

  if not play:
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=mdp.commands_vel,
      params={"command_name": mdp.COMMAND_NAME, "velocity_stages": SPEED_STAGES},
    )
    cfg.curriculum["push_tracking"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "push_tracking",
        "weight_stages": mdp.ramp(W_PUSH, PUSH_STAGE),
      },
    )
  else:
    # Watch it at the speed it was trained to, not the speed it started at. The flat task
    # already cleared the curriculum and widened the command range for play; both are
    # wrong here, so both are set again.
    cfg.curriculum = {}
    last = SPEED_STAGES[-1]
    assert last["lin_vel_x"] is not None
    twist_cmd.ranges.lin_vel_x = last["lin_vel_x"]
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.2, 0.2)

  return cfg
