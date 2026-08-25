"""Velocity tracking, pushed past the speed the stock task is tuned for.

This is mjlab's G1 flat velocity environment with one thing changed in earnest: how
fast it is asked to go. The stock task's own curriculum tops out at 3.0 m/s forward
and spends most of its command budget on turning and sidestepping, because it is
meant to be a general locomotion policy. The parkour corridor wants the opposite:
one direction, as fast as the robot can hold it.

So the changes are all about where the training budget goes.

- The forward range is widened in stages, further and for longer than the stock
  schedule, ending well past a walk.
- Sideways and turning commands are narrowed. Speed and agility compete for the same
  actuators, and a policy asked for both at once gets neither at the top of the range.
- More environments get forward-only commands, for the same reason.
- Episodes are longer, because reaching a high speed from a standing start takes a
  meaningful fraction of a short episode, and an episode that ends before the robot
  is up to speed teaches mostly acceleration.

Everything else, including the carefully tuned per-joint pose tolerances, is left as
mjlab has it. The stock reward set already distinguishes running from walking: its
`pose` term switches to a looser per-joint tolerance above 1.5 m/s, which is what
lets the gait open up rather than being held near the nominal stance.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

# Top forward speed the curriculum works up to [m/s]. The G1's stock task reaches
# 3.0; this asks for half again as much. Treat it as a target to be approached, not
# a promise: the curriculum only advances the command range, and whether the policy
# can actually hold the top of it is what the training run answers.
TOP_SPEED = 4.5

# Environment steps per curriculum stage. `common_step_counter` advances once per
# environment step, and the default 24 steps per environment per iteration makes
# this roughly 1500 iterations a stage.
_STAGE = 1500 * 24

SPEED_STAGES: list[mdp.VelocityStage] = [
  # Start where the stock task starts. The first stage is ordinary walking, and it
  # has to be solved before anything faster is worth asking for.
  {
    "step": 0,
    "lin_vel_x": (-1.0, 1.5),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_z": (-0.5, 0.5),
  },
  {
    "step": _STAGE,
    "lin_vel_x": (-1.0, 2.5),
    "lin_vel_y": (-0.4, 0.4),
    "ang_vel_z": (-0.4, 0.4),
  },
  {
    "step": 2 * _STAGE,
    "lin_vel_x": (-1.0, 3.5),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.4, 0.4),
  },
  {
    "step": 3 * _STAGE,
    "lin_vel_x": (-1.0, 4.0),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.3, 0.3),
  },
  {
    "step": 4 * _STAGE,
    "lin_vel_x": (-1.0, TOP_SPEED),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.3, 0.3),
  },
]


def g1_run_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the G1 running configuration.

  Args:
    play: Play-mode overrides, inherited from the stock flat task, plus a command
      range fixed at the top of the curriculum so a trained policy is watched at
      the speed it was trained up to rather than the speed it started from.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)

  # Most environments get a straight-line command. The stock task uses 0.2 because
  # it wants a policy that can do everything; here the corridor only ever asks for
  # forward, and the top of the speed range is hard enough to deserve the budget.
  twist_cmd.rel_forward_envs = 0.6
  # Standing still is not what this skill is for, and every standing environment is
  # one not spent learning to run. Kept above zero so the policy does not forget how.
  twist_cmd.rel_standing_envs = 0.05

  # The stage-zero range. The curriculum overwrites this from the first step, but
  # setting it here keeps the config honest about where it begins.
  first = SPEED_STAGES[0]
  assert first["lin_vel_x"] is not None
  assert first["lin_vel_y"] is not None
  assert first["ang_vel_z"] is not None
  twist_cmd.ranges.lin_vel_x = first["lin_vel_x"]
  twist_cmd.ranges.lin_vel_y = first["lin_vel_y"]
  twist_cmd.ranges.ang_vel_z = first["ang_vel_z"]

  # A run needs room. Accelerating to 4 m/s and holding it is most of a short episode,
  # and an episode that ends before the robot is up to speed teaches acceleration and
  # not much else. Play mode keeps the effectively infinite length it inherited.
  if not play:
    cfg.episode_length_s = 20.0

  # Commands are held longer than the stock 3-8 s: at these speeds a resample every
  # few seconds means the robot spends its life accelerating between targets and
  # never actually runs at one.
  twist_cmd.resampling_time_range = (6.0, 12.0)

  if not play:
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=mdp.commands_vel,
      params={"command_name": "twist", "velocity_stages": SPEED_STAGES},
    )
  else:
    # Watch it at the speed it was trained to, not the speed it started at.
    cfg.curriculum.pop("command_vel", None)
    last = SPEED_STAGES[-1]
    assert last["lin_vel_x"] is not None
    twist_cmd.ranges.lin_vel_x = (1.0, TOP_SPEED)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  return cfg
