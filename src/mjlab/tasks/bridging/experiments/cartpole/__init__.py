"""A cart-pole with two skills whose regimes momentum genuinely couples.

The differential-drive testbed has a weakness: a rolling diff-drive's heading is
kinematically decoupled from its forward momentum, so momentum can only displace
a turn, not corrupt it. The cart-pole does not have that escape hatch. The pole's
angle is the controlled variable, and the energy the swing-up pumps into it is
exactly the momentum the balancer then has to catch, there is nowhere for it to
hide.

Two skills, each competent only inside its own regime:
- `spin_up`: pumps energy into the pole to bring it from hanging to upright, by
  driving the cart back and forth in phase with the swing. It only ever shapes
  the pole's energy toward the upright value; it has no notion of the unstable
  equilibrium, so left to its own devices it swings the pole up, sails through
  the top, and lets it fall again. It cannot balance.
- `balance`: an LQR that holds the pole upright, valid only in the linear regime
  near the top. Handed a pole that is still swinging (or far from vertical) its
  linear law reads a huge angle error and slams the cart the wrong way, so it
  cannot catch a pole that has not already been brought up gently.

That gap is the whole point: a naive hand-off from `spin_up` to `balance` at the
wrong instant (pole not yet up, or up but carrying too much angular velocity)
fails, and the bridge's job is to hand over exactly when the pole is up and
slow enough for the balancer's basin of attraction. The failure is unavoidable
and graded: too much residual energy and the balancer loses the pole.

That gap only exists because this experiment gives the pole a lossy hinge, and it is
worth knowing why. `spin_up` shapes the pole's energy to exactly the energy of standing
upright, and "exactly enough energy to reach the top" is the same statement as "arrives
at the top with no speed left" -- which is precisely what `balance` needs. On the shared
asset's near-frictionless pole that energy is never lost, so a mistimed hand-over costs
nothing at all: `balance` sits at zero force outside its basin and the pole comes back
around to offer itself again, indefinitely. Measured over 256 envs, handing over at the
bottom of the swing succeeded 100% of the time, so there was nothing for a bridge to do.
With the damping in cartpole_env_cfg.py it succeeds 0% of the time: the energy bleeds
away before the pole reaches the top, it falls short, and since `balance` cannot swing
up, the failure is permanent. `spin_up` is unaffected because it has an energy source
and can pay the loss; a coasting pole cannot.

The analytical experts live in dynamics.py; they are the handwritten
counterparts of RL skills trained from the existing Mjlab-Cartpole-Swingup
(pole starts hanging) and Mjlab-Cartpole-Balance (pole starts upright) tasks,
which share the observation and action spaces the analytical experts read.

For this experiment, I will use the analytical skills as they are more
reliable: we can trust them working individually but not together.

Watch an analytical expert on its own:

    uv run python -m mjlab.tasks.bridging.skill \\
        --task-id Mjlab-Cartpole-Swingup \\
        --factory mjlab.tasks.bridging.experiments.cartpole.dynamics:analytical_spin_up
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.tasks.bridging.architectures import Budgets
from mjlab.tasks.bridging.architectures.arch_1.config import (
  BridgePhase,
  BridgeTraining,
  SwitchPhase,
)
from mjlab.tasks.bridging.architectures.arch_3.config import ResidualTraining
from mjlab.tasks.bridging.experiments.cartpole.cartpole_env_cfg import (
  damped_cartpole_env_cfg,
)
from mjlab.tasks.bridging.view import StateViewCfg
from mjlab.tasks.bridging.windows import SkillInit, SkillWindowSpec, WindowPlan
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_ppo_runner_cfg
from mjlab.tasks.registry import register_mjlab_task

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.bridging.experiment import Experiment
  from mjlab.tasks.bridging.skill import SkillPool

# Names shared by this experiment's train and demo entry points. EXPERIMENT_NAME
# is the folder architecture checkpoints are saved under; ENTITY_NAME is the
# scene entity the bridges harvest interrupt states from and read state off.
EXPERIMENT_NAME = "cartpole"
ENTITY_NAME = "cartpole"

# The two skill tasks. They share one observation/action space; the arena is built
# from the swingup task so the pole starts hanging and spin_up has a job.
#
# These are this experiment's own registrations, not the shared `Mjlab-Cartpole-*`
# tasks, and they differ from them in exactly one number: the hinge damping (see
# cartpole_env_cfg.py). Everything else -- observations, action, rewards, resets -- is
# the shared config, replaced entity and all. A frictionless pole makes the hand-over
# impossible to fail, because the energy spin_up built is never lost and the pole keeps
# re-offering a catchable moment forever; a lossy one makes a mistimed hand-over
# permanent, since balance coasts at zero force and cannot swing the pole back up.
SPINUP_TASK_ID = "Mjlab-Skills-Cartpole-Swingup"
BALANCE_TASK_ID = "Mjlab-Skills-Cartpole-Balance"


# How much of each skill is recorded around a hand-over. Shared by every architecture's
# training and by inspect.py, so what you watch is what is trained on.
#
# Control runs at 20 Hz, so 20 steps is a second: enough to see a whole up-and-over
# rather than a snapshot. spin_up gets the wide cut range, from one second to six
# seconds in, which covers the pole hanging and being pumped, swinging wide, and
# arriving near the top. That spread is the point of this experiment: the balancer can
# catch the pole from some of those moments and not others, and it is the same pole
# angle at very different speeds.
#
# balance is almost never the skill handed away from here, and when it is there is
# nothing to wait for: it holds the pole still, so every moment looks the same. Its
# range is short and its closing window shorter, since there is no build-up to record.
#
# `overrun` is zero for both: no architecture here asks to see what a skill would have
# gone on to do had control not been taken away. Raise it for one that does.
# Each skill also declares where it starts. The arena is one env, built from the swing-up
# task, so `env.reset()` hangs the pole for *both* skills -- and `balance` is an LQR valid
# only in a narrow band around upright. Recording its opening window from a hanging pole
# recorded the one state it cannot work in, and since the opening window is exactly what a
# bridge is trained to reproduce, the whole bridge into `balance` was aimed at the wrong
# place. So `balance` re-states its own start (upright, near still) on top of the reset;
# `spin_up` needs none, because the arena's own reset is already its regime.
#
# This is about the shared arena, not about analytical versus learned skills: the
# `Mjlab-Cartpole-Balance` task starts upright when trained on its own, so a checkpoint
# loaded into this pool would have been mis-recorded in exactly the same way.
#
# The ranges mirror what the balance task's own reset does (hinge within ~0.034 rad of
# upright, slider within 0.1 m of center, both nearly at rest); they are absolute here
# rather than offsets, since the entity default they would offset from is the swing-up
# one that is being corrected.
_BALANCE_INIT = SkillInit(
  entity_name=ENTITY_NAME,
  joint_pos={"hinge_1": (-0.034, 0.034), "slider": (-0.1, 0.1)},
  joint_vel={"hinge_1": (-0.01, 0.01), "slider": (-0.01, 0.01)},
)

# What the bridge is allowed to see, and therefore what it is compared against.
#
# The actor observation is four terms, five channels in all:
#   cart_pos    (1)  slider position, relative to the rail's center
#   pole_angle  (2)  cos and sin of the hinge angle, +1 cos upright
#   cart_vel    (1)  slider velocity
#   pole_vel    (1)  hinge angular velocity
#
# Everything is kept, there is nothing here to drop: the cart-pole env has no command
# term and no actions term, and none of the four describes a task or a world frame.
# They are five numbers about the machine, and the state this experiment turns on
# ("is the pole up, and how fast is it going") is spread across three of them.
#
# Listed by name rather than left empty so the layout is written down and a term can be
# taken out by deleting a line.
BRIDGE_VIEW = StateViewCfg(
  keep=(
    "cart_pos",
    "pole_angle",
    "cart_vel",
    "pole_vel",
  )
)

WINDOWS = WindowPlan(
  {
    "spin_up": SkillWindowSpec(
      opening=128, closing=128, overrun=0, interrupt_range=(128, 512)
    ),
    "balance": SkillWindowSpec(
      opening=128,
      closing=128,
      overrun=0,
      interrupt_range=(128, 512),
      init=_BALANCE_INIT,
    ),
  }
)

# How long to train, per architecture. Small, because this experiment is small.
# eval_steps is two seconds of control, long enough for a pole the balancer did not
# really catch to be visibly falling by the time the hand-over is judged.
#
# Note this task never terminates, so architecture 2 (which judges a hand-over purely by
# survival) has nothing to learn from here. Architecture 3 is the one to compare against
# architecture 1 on the cart-pole; its fade range and tail mirror arch_1's
# max_transition_steps and eval_steps.
_BRIDGE = BridgeTraining(
  bridge=BridgePhase(num_iterations=50, num_windows=512, num_interrupts=4096),
  switch=SwitchPhase(
    num_iterations=100,
    num_interrupts=4096,
    max_transition_steps=40,
    eval_steps=40,
    epsilon_decay_iterations=150,
  ),
)

BUDGETS = Budgets(
  arch_1=_BRIDGE,
  arch_2=_BRIDGE,
  arch_3=ResidualTraining(
    num_iterations=200,
    num_windows=512,
    steps=(10, 40),
    tail_steps=40,
    inference_steps=24,
  ),
  # arch_4_backup has never been run on this experiment; its defaults stand until it is.
)


register_mjlab_task(
  task_id=SPINUP_TASK_ID,
  env_cfg=damped_cartpole_env_cfg(swing_up=True),
  play_env_cfg=damped_cartpole_env_cfg(swing_up=True, play=True),
  rl_cfg=cartpole_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id=BALANCE_TASK_ID,
  env_cfg=damped_cartpole_env_cfg(swing_up=False),
  play_env_cfg=damped_cartpole_env_cfg(swing_up=False, play=True),
  rl_cfg=cartpole_ppo_runner_cfg(),
)


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
  With analytical=False, the frozen RL policies are loaded instead; a None
  checkpoint falls back to the latest trained one for that task.
  """
  # Imported lazily so merely importing this package stays cheap (the demo imports
  # it just for the constants above).
  from mjlab.tasks.bridging.experiments.cartpole.dynamics import (
    analytical_balance,
    analytical_spin_up,
  )
  from mjlab.tasks.bridging.skill import PolicySkill, SkillPool
  from mjlab.tasks.bridging.utils import retrieve_latest_checkpoint

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


def build_experiment(env: ManagerBasedRlEnv, device: str, **pool_kwargs) -> Experiment:
  """Everything an architecture needs from this experiment, in one object."""
  from mjlab.tasks.bridging.experiment import Experiment

  return Experiment(
    name=EXPERIMENT_NAME,
    entity_name=ENTITY_NAME,
    pool=build_pool(env, device, **pool_kwargs),
    view=BRIDGE_VIEW.resolve(env),
    windows=WINDOWS,
  )
