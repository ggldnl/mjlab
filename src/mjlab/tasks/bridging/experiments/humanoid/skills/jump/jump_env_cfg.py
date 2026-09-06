"""The single clip jump environment: track one ASAP jump, end to end.

The continuous jump's environment has two stages:

    stage one        a set of jump clips is used. A goal is associated to each clip based on
                     the landing distance. They are stretched horizontally to cover more cases
                     (a jump could ask for a distance to cover that no reference clip gives).

                     When we request a goal, we find the clip with the most similar landing
                     distance and perform horizontal stretching to make it equal to what we
                     ask. We train a goal-conditioned policy this way. For this reason, the
                     policy will always need a reference to track
    stage two        we use the above as a teacher and we distill a student that doesn't need
                     reference clips. It learns to jump to match the goal regardless.

This is the continuous jump's environment with the goal taken out and one clip left in.
Everything that makes the motion physical is unchanged.

What is removed, and why:

    the clip set     one clip instead of five, so there is nothing to interpolate between
    scale_range      pinned at 1.0. Stretching existed to fill the gaps between clips, and
                     with one clip we don't need it
    goal terms       the goal observation and the two goal rewards. With one clip at one
                     scale the goal is a constant, so it tells the policy nothing and pays
                     the same every episode
    the teacher      one observation group instead of two. There is no distillation phase,
                     so the policy that trains is the policy that runs and it reads the
                     reference

Subtracted from the continuous config rather than restated, so the two cannot drift apart in
their stds, their sensors or their curriculum. Everything below is the diff.

Run

1. Convert the clips, if they are not converted already. Writes to data/asap/motions.

    uv run --with joblib python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.dataset

2. Train.

    uv run train Mjlab-G1-Jump --env.scene.num-envs 4096

3. Watch.

    uv run play Mjlab-G1-Jump --checkpoint-file <path>
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.dataset import (
  motion_file,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.jump_continuous_env_cfg import (
  g1_jump_continuous_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp import (
  JumpCommandCfg,
)


def g1_jump_env_cfg(
  clip: str | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the single clip jump environment.

  Args:
    clip: The converted npz to track. Defaults to what dataset.py writes.
    play: Start every episode at the entry landmark rather than sampling the clip, drop the
      observation noise and the pushes, and leave the reference unperturbed.
  """
  cfg = g1_jump_continuous_env_cfg(
    motion_files=(clip or str(motion_file()),), play=play
  )

  # One observation group, and it is the one carrying the reference. The continuous task
  # keeps two because it trains a tracker and then distils it; there is nothing to distil
  # onto here, so the teacher's group is promoted to "actor" and the student's is dropped.
  # Promoted rather than rebuilt, so the noise and the term order stay identical
  reference = cfg.observations.pop("teacher")
  reference.terms.pop("goal")
  cfg.observations["actor"] = reference
  cfg.observations["critic"].terms.pop("goal")

  # "landed" stays. It is the flight phase, not the goal, and the critic still needs to know
  # whether the robot is in the air to value a state

  cfg.rewards.pop("jump_goal_pos")
  cfg.rewards.pop("jump_goal_success")

  motion = cfg.commands["motion"]
  assert isinstance(motion, JumpCommandCfg)
  motion.scale_range = (1.0, 1.0)
  # The viewer dial asks for a distance. There is one distance
  motion.gui = False

  return cfg
