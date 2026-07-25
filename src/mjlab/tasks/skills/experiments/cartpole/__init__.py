"""A cart-pole with two skills whose regimes momentum genuinely couples.

The differential-drive testbed has a weakness: a rolling diff-drive's heading is
kinematically decoupled from its forward momentum, so momentum can only displace
a turn, not corrupt it. The cart-pole does not have that escape hatch. The pole's
angle is the controlled variable, and the energy the swing-up pumps into it is
exactly the momentum the balancer then has to catch, there is nowhere for it to
hide.

Two skills, each competent only inside its own regime:
- spin_up: pumps energy into the pole to bring it from hanging to upright, by
  driving the cart back and forth in phase with the swing. It only ever shapes
  the pole's energy toward the upright value; it has no notion of the unstable
  equilibrium, so left to its own devices it swings the pole up, sails through
  the top, and lets it fall again. It cannot balance.
- balance: an LQR that holds the pole upright, valid only in the linear regime
  near the top. Handed a pole that is still swinging (or far from vertical) its
  linear law reads a huge angle error and slams the cart the wrong way, so it
  cannot catch a pole that has not already been brought up gently.

That gap is the whole point: a naive hand-off from spin_up to balance at the
wrong instant (pole not yet up, or up but carrying too much angular velocity)
fails, and the bridge's job is to hand over exactly when the pole is up and
slow enough for the balancer's basin of attraction. Unlike the diff-drive, here
the failure is unavoidable and graded: too much residual energy and the balancer
loses the pole.

The analytical experts live in dynamics.py; they are the hand-written
counterparts of RL skills trained from the existing Mjlab-Cartpole-Swingup
(pole starts hanging) and Mjlab-Cartpole-Balance (pole starts upright) tasks,
which share the observation and action spaces the analytical experts read.

For this experiment, I will use the analytical skills as they are more
reliable: we can trust them working individually but not together.

Watch an analytical expert on its own:

    uv run python -m mjlab.tasks.skills.skill \\
        --task-id Mjlab-Cartpole-Swingup \\
        --factory mjlab.tasks.skills.experiments.cartpole.dynamics:analytical_spin_up
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.skills.skill import SkillPool

# Names shared by this experiment's train and demo entry points. `EXPERIMENT_NAME`
# is the folder architecture checkpoints are saved under; `ENTITY_NAME` is the
# scene entity the bridges harvest interrupt states from and read state off.
EXPERIMENT_NAME = "cartpole"
ENTITY_NAME = "cartpole"

# The two skill tasks. They share one observation/action space; the arena is built
# from the swingup task so the pole starts hanging and spin_up has a job.
SPINUP_TASK_ID = "Mjlab-Cartpole-Swingup"
BALANCE_TASK_ID = "Mjlab-Cartpole-Balance"


def build_pool(
  env: ManagerBasedRlEnv,
  device: str,
  *,
  analytical: bool = True,
  spinup_task_id: str = SPINUP_TASK_ID,
  balance_task_id: str = BALANCE_TASK_ID,
  spinup_checkpoint: str | None = None,
  balance_checkpoint: str | None = None,
) -> SkillPool:
  """The two-skill pool for this experiment: spin_up (id 0), then balance (id 1).

  Analytical experts by default (recommended: their competence is not in question).
  With `analytical=False`, the frozen RL policies are loaded instead; a `None`
  checkpoint falls back to the latest trained one for that task.
  """
  # Imported lazily so merely importing this package stays cheap (the demo imports
  # it just for the constants above).
  from mjlab.tasks.skills.experiments.cartpole.dynamics import (
    analytical_balance,
    analytical_spin_up,
  )
  from mjlab.tasks.skills.skill import PolicySkill, SkillPool
  from mjlab.tasks.skills.utils import retrieve_latest_checkpoint

  if analytical:
    return SkillPool([analytical_spin_up(), analytical_balance()])

  spinup_ckpt = spinup_checkpoint or retrieve_latest_checkpoint(spinup_task_id)
  balance_ckpt = balance_checkpoint or retrieve_latest_checkpoint(balance_task_id)
  return SkillPool(
    [
      PolicySkill("spin_up", spinup_task_id, spinup_ckpt, env, device),
      PolicySkill("balance", balance_task_id, balance_ckpt, env, device),
    ]
  )
