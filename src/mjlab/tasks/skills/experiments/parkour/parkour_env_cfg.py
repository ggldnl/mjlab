"""One env cfg factory, four goal-conditioned skills.

Each skill is the same environment pointed at a different reference folder and given
a different command box. That is the entire difference between walk, run, sprint and
jump: `SKILL_COMMANDS` says what velocities the skill is asked for, and
`data/lafan1_g1/clips/<skill>/` says what those velocities should look like.

Built from the G1 flat *velocity* env, with two edits:

- the uniform twist command is swapped for `mdp.TwistCommand`, which adds a vertical
  channel (how a jump is asked for) and drops the heading controller;
- the hand-written gait reward stack -- foot clearance, air time, swing height,
  slip, posture, soft landing -- is deleted. Every one of those encoded an idea of
  what walking should look like; the discriminator reads that off the reference clips
  instead. What is left besides goal and style is only what keeps the robot from
  hurting itself.

Since the skill pool runs every frozen skill against one shared env, all four must
agree on their observation and action spaces. They do here by construction: the only
per-skill values are the reference folder and the sampling box, neither of which is
observed. Keep it that way when tuning.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.skills.experiments.parkour import mdp as parkour_mdp
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

# Where dataset.py filed the reference clips: one folder per skill.
CLIP_ROOT = Path("data/lafan1_g1/clips")

Ranges = parkour_mdp.TwistCommandCfg.Ranges

# The goal box per skill, and the main tuning knob of the whole experiment. A box the
# reference clips do not cover puts the goal reward and the style reward in direct
# conflict, so these are read off the "suggested box" lines `dataset.py` prints, and
# they need re-checking whenever the dataset is rebuilt with different source clips.
# The values below come from the subset of LAFAN1 clips that was on hand, so treat
# them as a starting point rather than as measured truth.
#
# v_up is pinned to zero everywhere but jump: walking, running and sprinting are not
# asked to leave the ground. Run and sprint overlap in v_fwd on purpose -- they are
# separated by which clips their discriminator saw, not by a speed boundary.
SKILL_COMMANDS: dict[str, Ranges] = {
  "walk": Ranges(
    v_fwd=(0.3, 1.0),
    v_lat=(-0.4, 0.4),
    v_up=(0.0, 0.0),
    yaw_rate=(-0.5, 0.5),
  ),
  "run": Ranges(
    v_fwd=(1.5, 3.0),
    v_lat=(-0.5, 0.5),
    v_up=(0.0, 0.0),
    yaw_rate=(-0.6, 0.6),
  ),
  "sprint": Ranges(
    v_fwd=(3.0, 3.6),
    v_lat=(-0.3, 0.3),
    v_up=(0.0, 0.0),
    yaw_rate=(-0.6, 0.6),
  ),
  # v_up is the jump dial: held at a positive value it asks for repeated hops, since
  # the robot cannot keep rising (see mdp.track_vertical_velocity). The reference
  # takeoffs peak around 1 m/s, so a box much above that is asking for motion the
  # discriminator has never seen.
  "jump": Ranges(
    v_fwd=(-0.5, 1.5),
    v_lat=(-0.4, 0.4),
    v_up=(0.4, 1.0),
    yaw_rate=(-0.6, 0.6),
  ),
}

SKILL_NAMES = tuple(SKILL_COMMANDS)


def parkour_skill_env_cfg(
  skill: str, clip_root: str | Path = CLIP_ROOT, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """The env for one skill: its command box, its reference clips."""
  if skill not in SKILL_COMMANDS:
    raise ValueError(f"Unknown skill '{skill}'; known: {SKILL_NAMES}.")

  cfg = unitree_g1_flat_env_cfg(play=play)
  clip_dir = str(Path(clip_root) / skill)

  cfg.commands = {
    "twist": parkour_mdp.TwistCommandCfg(
      entity_name="robot",
      ranges=SKILL_COMMANDS[skill],
      resampling_time_range=(3.0, 6.0),
    )
  }
  # The velocity curriculum widens a uniform sampling box this env no longer has.
  cfg.curriculum.pop("command_vel", None)

  cfg.rewards = {
    "track_planar_velocity": RewardTermCfg(
      func=parkour_mdp.track_planar_velocity,
      weight=1.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_vertical_velocity": RewardTermCfg(
      func=parkour_mdp.track_vertical_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "track_yaw_rate": RewardTermCfg(
      func=parkour_mdp.track_yaw_rate,
      weight=0.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "amp_style": RewardTermCfg(
      func=parkour_mdp.amp_style_reward,
      weight=2.0,
      params={
        "clip_dir": clip_dir,
        "entity_name": "robot",
        # One discriminator update per PPO rollout, on what that rollout produced.
        "update_every": 24,
        "num_updates": 2,
        "batch_size": 4096,
      },
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  return cfg
