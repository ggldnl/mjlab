"""The table tennis scene and the four primitive skills trained in it.

The scene is a KUKA iiwa14 with a racket welded to its wrist, plus a ball, on a ground
plane. There is no table.

The four skills, and what each is actually being asked to do:

- catch   : a ball drops in from above; get the blade under it and take its energy out
            so it comes to rest on the bat instead of bouncing off.
- balance : a nearly-stationary ball sits over the blade; keep it on the sweet spot.
- toss    : a ball resting on the blade; launch it straight up to a commanded apex.
- hit     : a ball drops in with real momentum; strike it so it comes to a standstill
            at the commanded point, arriving there with nothing left.

Three rules shape all four:

**The ball may only touch the racket.** Contact with the arm or the floor ends the
episode as a failure, enforced by the two contact sensors this module adds to the
scene. No task is exempt: `hit` stops at the ball's apex precisely so its ball never
gets the chance to reach the floor.

**Every spawn is anchored to the blade.** The ball always starts at the blade's own
x/y, differing only in how far above it -- on the bat for balance and toss, well above
for catch and hit. An absolute box in world coordinates stops being reachable the
moment the ready stance is retuned, and a ball the arm cannot get to is not a task but
a guaranteed failure. The same reason `ready_face_pos` is computed from the model
rather than written down.

**One observation space, one command.** All four share the same observations and
actions, and the command always means *where the ball should end up*. A bridge is
trained to act in the skills' own space and hand control between them, so those spaces
cannot drift apart.

Two things worth knowing about this scene:

- The KUKA's XML actuators are stiff (kp=2000, kd=200), unstable under the default
  Euler integrator. `MujocoCfg` already defaults to `implicitfast`, so an env is fine;
  only a raw `spec.compile()` for a standalone viewer must set it by hand.
- In the KUKA's own home keyframe the blade ends up vertical, so nothing can rest on
  it. `READY_JOINT_POS` is a face-up stance every skill starts from.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.asset_zoo.objects.ball import get_ball_cfg
from mjlab.asset_zoo.objects.racket import racket_constants
from mjlab.asset_zoo.robots.kuka_iiwa_14 import kuka_constants
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  generated_commands,
  joint_pos_rel,
  joint_torques_l2,
  joint_vel_l2,
  joint_vel_rel,
  last_action,
  reset_joints_by_offset,
  sample_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.skills.experiments.table_tennis import mdp
from mjlab.tasks.skills.experiments.table_tennis.mdp import (
  _ROBOT,
  ANY_CONTACT_SENSOR,
  BALL_RADIUS,
  COMMAND_NAME,
  RACKET_CONTACT_SENSOR,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

##
# Ball. Regulation table tennis ball: 40 mm diameter, 2.7 g.
##

BALL_MASS = 0.0027

# Low friction: a celluloid ball on a rubber blade slides far more than the size-5
# football the ball asset defaults to.
BALL_FRICTION = 0.3
BALL_COLOR = (0.95, 0.55, 0.1, 1.0)

# Where the ball sits in the compiled model before any reset event runs. Every task
# overrides this from its own reset, so it only matters to a raw viewer.
BALL_SPAWN_POS = (1.08, 0.0, 0.5)


##
# Arm + racket.
##

# Face-up stance: blade normal within ~2 deg of world +Z, blade centre out in front of
# the base at roughly (1.08, 0.0, 0.65). Found by sweeping the arm's joints for a pose
# a ball actually rests on; the KUKA's own home keyframe holds the blade vertical,
# which nothing can balance on.
READY_JOINT_POS = {
  "joint1": 0.0,
  "joint2": 0.1,
  "joint3": 0.0,
  "joint4": -2.0,
  "joint5": 0.0,
  "joint6": -0.7,
  "joint7": -1.55,
}

READY_KEYFRAME = EntityCfg.InitialStateCfg(
  joint_pos=dict(READY_JOINT_POS),
  joint_vel={".*": 0.0},
)


def get_arm_with_racket_spec() -> mujoco.MjSpec:
  """Build the KUKA iiwa14 spec with the racket welded to ``attachment_site``."""
  arm_spec = kuka_constants.get_spec()
  racket_spec = racket_constants.get_spec()

  # racket.xml ships its own demo-only ground plane and light directly under its
  # worldbody (added for racket_constants.py's standalone viewer). attach() would
  # carry both along as children of the attachment site, and a <plane> geom can only
  # compile as a static (world) geom, so drop them here first.
  for geom in list(racket_spec.worldbody.geoms):
    racket_spec.delete(geom)
  for light in list(racket_spec.worldbody.lights):
    racket_spec.delete(light)

  # Attaching at a site welds the child's root frame to that site's pose. The racket's
  # own attachment site is what lands there; its "face" site, ~0.2 m further out, is
  # the blade centre everything here actually cares about.
  attachment_site = arm_spec.site("attachment_site")
  arm_spec.attach(child=racket_spec, prefix="racket_", site=attachment_site)
  return arm_spec


def get_arm_with_racket_cfg() -> EntityCfg:
  """Get a fresh arm+racket configuration instance, starting in the ready stance."""
  return EntityCfg(
    spec_fn=get_arm_with_racket_spec,
    articulation=kuka_constants.KUKA_ARTICULATION,
    init_state=READY_KEYFRAME,
  )


@functools.lru_cache(maxsize=1)
def ready_face_pos() -> tuple[float, float, float]:
  """Where the blade centre sits in the ready stance, computed from the model.

  Derived rather than written down. The stance is a tuning knob, and a hardcoded
  position silently goes stale the moment it is turned -- which puts every spawn and
  every target somewhere the arm cannot reach, with nothing to indicate why the skills
  stopped being learnable. Computing it means the two can never disagree.
  """
  model = get_arm_with_racket_spec().compile()
  data = mujoco.MjData(model)
  for name, angle in READY_JOINT_POS.items():
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[joint_id]] = angle
  mujoco.mj_forward(model, data)
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, mdp.FACE_SITE)
  x, y, z = data.site_xpos[site_id]
  return float(x), float(y), float(z)


def get_table_tennis_ball_cfg() -> EntityCfg:
  """Get a fresh ball configuration instance, scaled to a table tennis ball."""
  return get_ball_cfg(
    radius=BALL_RADIUS,
    mass=BALL_MASS,
    friction=BALL_FRICTION,
    color=BALL_COLOR,
    pos=BALL_SPAWN_POS,
  )


##
# Contact sensors. These are what enforce "the ball only touches the racket".
##

# The racket is matched as a subtree rather than by geom pattern: a contact sensor's
# secondary must resolve to a single element, and the blade is three geoms (core plus
# two rubber sheets). The subtree rooted at racket_middle covers all three.
_RACKET_CONTACT = ContactSensorCfg(
  name=RACKET_CONTACT_SENSOR,
  primary=ContactMatch(mode="subtree", pattern="racket_middle", entity="robot"),
  secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="ball"),
  fields=("found",),
  reduce="none",
  num_slots=1,
)

# No secondary: any contact the ball makes at all. Subtracting the racket sensor from
# this is what identifies a contact with the floor or the arm.
_ANY_CONTACT = ContactSensorCfg(
  name=ANY_CONTACT_SENSOR,
  primary=ContactMatch(mode="geom", pattern="ball_collision", entity="ball"),
  secondary=None,
  fields=("found",),
  reduce="none",
  num_slots=1,
)


def table_tennis_scene_cfg(num_envs: int = 1, env_spacing: float = 3.0) -> SceneCfg:
  """The arm+racket and the ball on a ground plane, with the two contact sensors.

  The entity names are the handles everything downstream uses: "robot" for the
  arm+racket (matching the other experiments) and "ball" for the ball.
  """
  return SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={
      "robot": get_arm_with_racket_cfg(),
      "ball": get_table_tennis_ball_cfg(),
    },
    sensors=(_RACKET_CONTACT, _ANY_CONTACT),
    num_envs=num_envs,
    env_spacing=env_spacing,
  )


##
# Ball spawns.
##


@dataclass(frozen=True)
class BallSpawn:
  """Where a task's ball starts, expressed relative to the blade centre.

  Relative, never absolute. An absolute box in world coordinates stops being reachable
  the moment the ready stance is retuned, and a ball the arm cannot get to is not a
  task, just a guaranteed failure. Anchoring every spawn to the live face pose couples
  the ball to the arm's configuration, so whatever stance the robot starts in, the
  ball starts somewhere it can actually be played.

  `height` is metres straight up from where the ball would rest on the blade, so zero
  means "on the bat" (balance, toss) and a large value means "dropped in from above"
  (catch, hit). `lateral` is the jitter around the blade's own x/y, deliberately small
  so the ball always falls more or less onto the face.

  Declared as data rather than baked into an event term so the same description drives
  both the reset and the initial-state view in demo.py -- what you look at is then
  guaranteed to be what training samples.
  """

  height: tuple[float, float] = (0.0, 0.0)
  lateral: float = 0.0
  vel: dict[str, tuple[float, float]] = field(default_factory=dict)

  def event(self) -> EventTermCfg:
    return EventTermCfg(
      func=mdp.reset_ball_above_face,
      mode="reset",
      params={
        "height_range": self.height,
        "lateral_range": self.lateral,
        "vel_range": dict(self.vel),
      },
    )


# Dropped in from well above the bat, slowly, so there is time to get under it and
# absorb it. The x/y jitter is small: the ball comes down onto the face, not beside it.
CATCH_SPAWN = BallSpawn(height=(0.60, 0.90), lateral=0.04, vel={"z": (-0.5, 0.0)})

# Already on the bat and nearly still, but scattered off-centre and drifting slightly,
# so holding the arm rigid is not a solution.
BALANCE_SPAWN = BallSpawn(
  height=(0.0, 0.0),
  lateral=0.025,
  vel={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
)

# Resting on the bat, essentially at rest: the toss supplies all the energy.
TOSS_SPAWN = BallSpawn(height=(0.0, 0.0), lateral=0.012)

# Higher and falling faster than catch: hit has to reverse real momentum rather than
# receive a gentle drop, which is what separates the two skills.
HIT_SPAWN = BallSpawn(height=(0.70, 1.00), lateral=0.04, vel={"z": (-1.2, -0.4)})

TASK_SPAWNS = {
  "catch": CATCH_SPAWN,
  "balance": BALANCE_SPAWN,
  "toss": TOSS_SPAWN,
  "hit": HIT_SPAWN,
}


##
# Command: where the ball should end up.
##


class BallTargetCommand(CommandTerm):
  """A single 3D point, resampled once per episode: where the ball should end up.

  One term serves all four skills because they all answer the same question about a
  different moment: catch settles the ball there, balance holds it there, toss sends
  its apex there, hit lands it there. Keeping it one term with different ranges is
  what keeps the four observation spaces identical.
  """

  cfg: BallTargetCommandCfg

  def __init__(self, cfg: BallTargetCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._command = torch.zeros(self.num_envs, 3, device=self.device)
    self.metrics["error_target"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    ranges = torch.tensor(
      [self.cfg.ranges.x, self.cfg.ranges.y, self.cfg.ranges.z], device=self.device
    )
    self._command[env_ids] = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=self.device
    )

  def _update_command(self) -> None:
    # Fixed for the whole episode; nothing to recompute per step.
    pass

  def _update_metrics(self) -> None:
    self.metrics["error_target"] = torch.norm(
      mdp.ball_pos(self._env) - self._command, dim=-1
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw the goal and the ball's current velocity.

    The velocity arrow is the point of the initial-state view in demo.py: a spawn is
    a position *and* a velocity, and only one of those is visible in a screenshot.
    """
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    goal = self._command.cpu().numpy()
    ball = mdp.ball_pos(self._env).cpu().numpy()
    vel = mdp.ball_vel(self._env).cpu().numpy()
    for i in env_indices:
      visualizer.add_sphere(
        goal[i], self.cfg.marker_radius, (0.1, 0.8, 0.3, 0.45), label="goal"
      )
      # Scaled so a typical spawn speed reads as a hand-sized arrow rather than a
      # spike across the whole scene.
      tip = ball[i] + vel[i] * self.cfg.velocity_arrow_scale
      visualizer.add_arrow(ball[i], tip, (0.2, 0.4, 0.95, 0.9), width=0.008)


@dataclass(kw_only=True)
class BallTargetCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

  ranges: Ranges
  # Drawn size of the goal marker. For `hit` this is the real goal radius the shot is
  # scored against; for the others it is only a visual cue.
  marker_radius: float = 0.05
  velocity_arrow_scale: float = 0.25

  def build(self, env: ManagerBasedRlEnv) -> BallTargetCommand:
    return BallTargetCommand(self, env)


##
# Target distributions.
##

_FX, _FY, _FZ = ready_face_pos()

# Somewhere on the bat, at about the height the arm naturally holds it: where catch
# should bring the ball to rest and where balance should hold it.
_ON_BAT_TARGET = BallTargetCommandCfg.Ranges(
  x=(_FX - 0.06, _FX + 0.06), y=(_FY - 0.06, _FY + 0.06), z=(_FZ, _FZ + 0.12)
)

# Straight up from the bat. x/y stay tight around the blade so the toss is vertical.
_APEX_TARGET = BallTargetCommandCfg.Ranges(
  x=(_FX - 0.04, _FX + 0.04), y=(_FY - 0.04, _FY + 0.04), z=(_FZ + 0.40, _FZ + 0.90)
)

# Where a struck ball should come to a standstill: straight up from the bat, higher
# than a toss because the ball arrives with downward momentum to reverse first.
_HIT_TARGET = BallTargetCommandCfg.Ranges(
  x=(_FX - 0.05, _FX + 0.05), y=(_FY - 0.05, _FY + 0.05), z=(_FZ + 0.50, _FZ + 1.00)
)

# How near the goal the apex must be, and how slow the ball must still be there, for
# `hit` to count as a success.
HIT_GOAL_RADIUS = 0.15
HIT_APEX_SPEED = 0.5


##
# Shared pieces.
##

EPISODE_LENGTH_S = 3.0
BALANCE_EPISODE_LENGTH_S = 5.0


def _observations() -> dict[str, ObservationGroupCfg]:
  """The one observation space all four skills share.

  Actor and critic see exactly the same thing. No privileged terms: a trained critic
  should stay usable as a value estimate on states a bridge produces, and one needing
  privileged inputs could not be queried there. Ball velocity is included for both --
  a real system would have to estimate it, but catching and hitting are not solvable
  from position alone at this control rate.
  """
  terms = {
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": _ROBOT},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": _ROBOT},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "ball_pos": ObservationTermCfg(
      func=mdp.ball_pos_rel_face, noise=Unoise(n_min=-0.005, n_max=0.005)
    ),
    "ball_vel": ObservationTermCfg(
      func=mdp.ball_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "actions": ObservationTermCfg(func=last_action),
    "command": ObservationTermCfg(
      func=generated_commands, params={"command_name": COMMAND_NAME}
    ),
  }
  return {
    "actor": ObservationGroupCfg(terms=terms, enable_corruption=True),
    "critic": ObservationGroupCfg(terms={**terms}, enable_corruption=False),
  }


def _actions() -> dict[str, ActionTermCfg]:
  """The KUKA ships tuned position servos, so the action is a joint position target,
  offset from the ready stance: a zero action holds the stance and the policy only has
  to learn the departure from it."""
  return {
    "arm_position": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=kuka_constants.KUKA_JOINT_NAMES,
      scale=0.5,
      use_default_offset=True,
    ),
  }


def _effort_rewards() -> dict[str, RewardTermCfg]:
  """The regularizers every task carries.

  `joint_torques` is deliberately tiny. The KUKA's servos are stiff (kp=2000), so
  `sum(torque^2)` runs into the thousands the moment the arm moves. At a larger weight
  this term alone outweighs every task term, which makes *moving* net-negative and the
  policy learns to end the episode as fast as it can instead of doing the task.
  """
  return {
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
    "joint_vel": RewardTermCfg(
      func=joint_vel_l2, weight=-1e-4, params={"asset_cfg": _ROBOT}
    ),
    "joint_torques": RewardTermCfg(
      func=joint_torques_l2, weight=-1e-7, params={"asset_cfg": _ROBOT}
    ),
  }


def _events(spawn: BallSpawn) -> dict:
  """Reset events, in the order they must run.

  The arm goes back to the ready stance first, so a ball reset that reads the blade
  pose sees the stance and not the previous episode's final pose. The phase tracker is
  cleared last, once the ball is where the new episode starts it.
  """
  return {
    "reset_arm": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": _ROBOT,
      },
    ),
    "reset_ball": spawn.event(),
    "reset_phase": EventTermCfg(func=mdp.reset_phase, mode="reset", params={}),
  }


def _make_env_cfg(
  target_ranges: BallTargetCommandCfg.Ranges,
  spawn: BallSpawn,
  rewards: dict[str, RewardTermCfg],
  terminations: dict[str, TerminationTermCfg],
  episode_length_s: float = EPISODE_LENGTH_S,
  marker_radius: float = 0.05,
) -> ManagerBasedRlEnvCfg:
  """Assemble one table tennis environment from the shared pieces."""
  return ManagerBasedRlEnvCfg(
    scene=table_tennis_scene_cfg(num_envs=1),
    observations=_observations(),
    actions=_actions(),
    commands={
      COMMAND_NAME: BallTargetCommandCfg(
        resampling_time_range=(episode_length_s, episode_length_s),
        ranges=target_ranges,
        marker_radius=marker_radius,
        debug_vis=True,
      )
    },
    events=_events(spawn),
    rewards={**rewards, **_effort_rewards()},
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="link0",
      distance=3.0,
      elevation=-20.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      # The ball against the blade and the floor; the arm touches nothing else.
      nconmax=16,
      njmax=48,
      mujoco=MujocoCfg(timestep=0.002),
    ),
    decimation=5,  # 100 Hz control: a stroke is over in a few hundred ms.
    episode_length_s=episode_length_s,
  )


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.episode_length_s = 1e10
  cfg.observations["actor"].enable_corruption = False
  return cfg


##
# The four skills.
##


def catch_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Receive a falling ball on the bat without letting it bounce off.

  The hard part is not interception but dissipation: a ball meeting a rigid blade
  rebounds, so the arm has to withdraw as it arrives and take the energy out. Success
  is the ball resting quietly on the bat, which is why the reward is dominated by
  "slow *while in contact*" rather than by proximity.
  """
  cfg = _make_env_cfg(
    target_ranges=_ON_BAT_TARGET,
    spawn=CATCH_SPAWN,
    rewards={
      # Dense: get the bat to the ball at all.
      "reach": RewardTermCfg(func=mdp.reach_ball, weight=1.0, params={"std": 0.35}),
      "over_face": RewardTermCfg(
        func=mdp.ball_over_face, weight=1.0, params={"std": 0.06}
      ),
      "contact": RewardTermCfg(func=mdp.in_contact, weight=1.0),
      # The objective proper: energy out of the ball, and only while on the bat.
      "damping": RewardTermCfg(
        func=mdp.ball_slow_on_racket, weight=4.0, params={"std": 0.4}
      ),
      "settle_at_target": RewardTermCfg(
        func=mdp.ball_at_command, weight=1.0, params={"std": 0.25}
      ),
      "success": RewardTermCfg(func=mdp.caught_bonus, weight=50.0),
      "illegal": RewardTermCfg(func=mdp.illegal_contact_penalty, weight=-5.0),
    },
    terminations={
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      "caught": TerminationTermCfg(func=mdp.caught),
      "illegal_contact": TerminationTermCfg(func=mdp.illegal_contact),
      "out_of_reach": TerminationTermCfg(func=mdp.ball_out_of_reach),
    },
  )
  return _apply_play_overrides(cfg) if play else cfg


def balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Keep a nearly-stationary ball resting on the middle of the bat.

  The ball starts slightly off-centre and drifting, so standing still eventually drops
  it; the arm has to keep the blade under it and level.

  Every term that means "doing the task" is gated on contact, and that gate is the
  whole design. Ungated, the obvious kernels are all maximised by a bat held under a
  *falling* ball it never touches -- lateral-offset and target-distance rewards cannot
  tell the difference, and tracking a ball down is far easier to discover than
  balancing one. A policy that found it would look exactly like an arm retracting at
  the start of the episode and following the ball to the floor. Gating means that
  behaviour pays essentially nothing, and `ball_left_bat` ends the episode as soon as
  the ball is genuinely off the blade, so there is no window to collect it in either.
  """
  cfg = _make_env_cfg(
    target_ranges=_ON_BAT_TARGET,
    spawn=BALANCE_SPAWN,
    rewards={
      # Touching the ball at all is the precondition for everything else.
      "contact": RewardTermCfg(func=mdp.in_contact, weight=3.0),
      # The objective proper: on the bat, as near the face site as possible.
      "on_face": RewardTermCfg(func=mdp.ball_on_face, weight=6.0, params={"std": 0.05}),
      # Tight on purpose. At the lenient width this started with (0.40) a ball being
      # tapped along at 0.1 m/s still scored 0.81 against a resting ball's 1.00, so
      # the term could not tell juggling from holding -- which is most of why juggling
      # looked like a good idea. At 0.08 the same ball scores 0.24.
      "steady": RewardTermCfg(
        func=mdp.ball_slow_on_racket, weight=2.0, params={"std": 0.08}
      ),
      "hold_at_target": RewardTermCfg(
        func=mdp.ball_at_command_on_racket, weight=1.0, params={"std": 0.20}
      ),
      # The only ungated term, and deliberately small: enough to guide the bat back
      # under a ball that has bounced, far too little to be worth farming.
      "reach": RewardTermCfg(func=mdp.reach_ball, weight=0.5, params={"std": 0.10}),
      # Unnecessary movement is the thing to suppress here, so it is priced directly.
      "still_bat": RewardTermCfg(func=mdp.racket_speed_l2, weight=-30.0),
      "illegal": RewardTermCfg(func=mdp.illegal_contact_penalty, weight=-5.0),
    },
    terminations={
      # Reaching the time limit *is* success here: the ball never left the bat.
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      "left_bat": TerminationTermCfg(
        func=mdp.ball_left_bat, params={"grace_steps": mdp.BALANCE_GRACE_STEPS}
      ),
      # Contact alone is satisfiable by juggling; this demands the ball actually rest.
      "not_resting": TerminationTermCfg(
        func=mdp.ball_not_resting,
        params={"grace_steps": mdp.BALANCE_SETTLE_GRACE},
      ),
      "illegal_contact": TerminationTermCfg(func=mdp.illegal_contact),
      "out_of_reach": TerminationTermCfg(func=mdp.ball_out_of_reach),
    },
    episode_length_s=BALANCE_EPISODE_LENGTH_S,
  )
  return _apply_play_overrides(cfg) if play else cfg


def toss_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Launch a resting ball straight up to a commanded apex.

  Two things are being asked at once: reach the commanded height, and go *straight*
  up. The verticality term is what makes it a toss rather than a flick -- a ball that
  drifts sideways is one nothing can catch again, so the command's x/y sit tight
  around the blade and the ball is scored on staying under them.
  """
  cfg = _make_env_cfg(
    target_ranges=_APEX_TARGET,
    spawn=TOSS_SPAWN,
    rewards={
      "apex": RewardTermCfg(
        func=mdp.apex_matches_command, weight=8.0, params={"std": 0.25}
      ),
      "vertical": RewardTermCfg(
        func=mdp.toss_is_vertical, weight=3.0, params={"std": 0.10}
      ),
      # Keeps the ball on the bat while the arm winds up, so it is launched rather
      # than merely abandoned.
      "contact": RewardTermCfg(func=mdp.in_contact, weight=0.5),
      "over_face": RewardTermCfg(
        func=mdp.ball_over_face, weight=1.0, params={"std": 0.06}
      ),
      "illegal": RewardTermCfg(func=mdp.illegal_contact_penalty, weight=-5.0),
    },
    terminations={
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      # The toss is over once the ball is back down at bat height; catching it again
      # is another skill's job.
      "returned": TerminationTermCfg(func=mdp.toss_returned),
      "illegal_contact": TerminationTermCfg(func=mdp.illegal_contact),
      "out_of_reach": TerminationTermCfg(func=mdp.ball_out_of_reach),
    },
  )
  return _apply_play_overrides(cfg) if play else cfg


def hit_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Strike a falling ball so it comes to a standstill at the commanded point.

  The ball arrives with real downward momentum and has to leave going straight up,
  arriving at the goal with nothing left: the apex is where it is scored, on both
  where it got to and how much speed it still carried. A projectile's vertical speed
  is zero at its apex for free, so what `apex_is_still` actually buys is killing the
  *horizontal* component -- the difference between placing the ball and lobbing it
  past the target.

  This is what separates hit from toss. Toss starts from a ball at rest and supplies
  all the energy; hit has to reverse momentum that is already there, and get the
  reversal exactly right.

  Ending the episode at the apex is deliberate: the outcome is settled there, and
  stopping keeps the ball off the floor, so hit obeys the same "only ever touch the
  racket" rule as the other three rather than carving out an exception for a landing.
  """
  cfg = _make_env_cfg(
    target_ranges=_HIT_TARGET,
    spawn=HIT_SPAWN,
    rewards={
      "reach": RewardTermCfg(func=mdp.reach_ball, weight=1.5, params={"std": 0.30}),
      "contact": RewardTermCfg(func=mdp.in_contact, weight=4.0),
      "over_face": RewardTermCfg(
        func=mdp.ball_over_face, weight=1.0, params={"std": 0.06}
      ),
      # Where the ball ended up, and how still it was when it got there.
      "apex_position": RewardTermCfg(
        func=mdp.apex_at_command, weight=10.0, params={"std": 0.30}
      ),
      "apex_still": RewardTermCfg(
        func=mdp.apex_is_still, weight=6.0, params={"std": 0.60}
      ),
      "success": RewardTermCfg(
        func=mdp.apex_in_goal,
        weight=30.0,
        params={
          "goal_radius": HIT_GOAL_RADIUS,
          "speed_threshold": HIT_APEX_SPEED,
        },
      ),
      "illegal": RewardTermCfg(func=mdp.illegal_contact_penalty, weight=-5.0),
    },
    terminations={
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      # The shot is settled once the ball stops rising.
      "apex_reached": TerminationTermCfg(func=mdp.apex_reached),
      "illegal_contact": TerminationTermCfg(func=mdp.illegal_contact),
      "out_of_reach": TerminationTermCfg(func=mdp.ball_out_of_reach),
    },
    marker_radius=HIT_GOAL_RADIUS,
  )
  return _apply_play_overrides(cfg) if play else cfg


TASK_ENV_CFGS = {
  "catch": catch_env_cfg,
  "balance": balance_env_cfg,
  "toss": toss_env_cfg,
  "hit": hit_env_cfg,
}


##
# RL config.
##


def table_tennis_ppo_runner_cfg(
  experiment_name: str, init_std: float = 1.0
) -> RslRlOnPolicyRunnerCfg:
  """PPO config shared by every table tennis task.

  obs_normalization is off on purpose, as in diffdrive: it keeps the critic a plain
  function of the raw observation, so a trained skill's value function stays usable on
  states a bridge produces without carrying a normalizer whose statistics came from a
  different distribution.

  `init_std` is per-task because the contact skills are far more sensitive to
  exploration than the striking ones. Gaussian exploration is white noise: an
  independent perturbation every control step, which at the default std means the
  blade is commanded +-0.5 rad of jitter at 100 Hz. Measured on `balance`, that flings
  the 2.7 g ball off the bat within ~15 steps, whereas *smooth* motion of the same
  amplitude keeps it on for ~350. The task is not the problem; the character of the
  noise is. Lowering the initial std for the contact tasks is what gives the policy
  episodes long enough to learn from. It costs nothing at inference, where the mean
  action is used, so a bridge is unaffected.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": init_std,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64), activation="elu", obs_normalization=False
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )
