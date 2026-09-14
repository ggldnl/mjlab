"""Drive one skill, fire a switch, let the bridge hand over to the next.

A transition script names two actors and what is on the floor. Everything else is here.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2jump
    uv run python -m ...transitions.walk2pass
    uv run python -m ...transitions.walk2push

    # headless, firing the switch on step 120 instead of waiting for a button
    uv run python -m ...transitions.walk2jump --viewer none --auto 120

    # which architecture crosses. Any of the three, on any couple
    uv run python -m ...transitions.walk2kick --bridge distillation

    # only this one, instead of loading every architecture into the dropdown
    uv run python -m ...transitions.walk2kick --bridges "('imitation',)"

    # which policies drive. One flag per skill of the couple, plus the bridge's
    uv run python -m ...transitions.walk2kick --kick-checkpoint logs/rsl_rl/g1_kick/x/model_1.pt

Three flags make two runs comparable, and a formal test sets all three. `--bridge` fixes
the architecture, `--entry` fixes the state being crossed to, and the third fixes the state
being crossed from, because pressed by hand the button lands on a different stride every
time and the bridge starts from a different velocity and phase.

What that third flag is depends on whether the entering skill has an object, and the
difference is not cosmetic. A skill with one is entered at a *place*: `Actor.arrive` says
where the robot has to stand for the swing to meet the ball, the ball does not move, and so
the whole of choosing when is walking onto that spot. `--fire-at` says how far before it to
fire and `--duration-s` says how long the bridge then gets, and both are read off the same
line the viewer counts down.

    uv run python -m ...transitions.walk2kick --bridge imitation --entry 3 \
        --fire-at 0.5 --duration-s 0.6

A step is the wrong handle for that skill and was being used as one. `--auto 130` lands
wherever two and a half seconds of walking happened to put the robot, which moves with the
commanded speed, with where the ball spawned and with how much turning the approach needed,
so two runs differing only in approach speed were crossing from different places. Firing
half a metre out means the same thing in both.

For a skill with nothing on the floor there is no spot, so `--auto` is still the handle and
still fixes the control step.

    uv run python -m ...transitions.walk2jump --bridge imitation --entry 3 --auto 130

Under a viewer all of it is on the panel: the `bridge` dropdown, the `entry` slider, and
then either `fire at, m` with `window, s` or `switch step`, whichever the couple takes. The
switch stays on the button until the checkbox beside them is turned on, which for a skill
with an object it is by default.

Every architecture that has a checkpoint is loaded, not only the one that drives, so the
dropdown swaps which one crosses without rebuilding anything. They can share one arena
because they differ in their window term and their observation and in nothing else: one
robot, one floor, one action convention, one corpus. The swap lands at the next hand-over
rather than immediately, since one architecture's opening and another's follow through are
not a crossing anybody performed, and the Active line names whoever is driving. An
architecture with no code or no checkpoint is dropped with a line saying which.

Three phases:

    leaving    the first skill drives, steered from the panel
    bridge     the bridge drives for `steps` steps, toward one target state
    entering   the second skill drives, from wherever the bridge left the robot

The viewer's Active line names whoever is driving, a skill by its own name or "bridge".
Each selected entry requests the profile in tests/entry_tolerances.py. Override one channel
with --tolerances.arm-joint-pos 0.1; other channels keep that entry's values. The profile
is frozen when the crossing starts and used by the bridge's arrival check at handoff.
From the moment a target is chosen a translucent robot stands in it and stays there after
the hand-over, so the gap to the real robot is the arrival error. Scene > Debug Viz >
Bridge turns it off.

The target is a single state: a pose, a height, a tilt, joint angles and every velocity at
one instant, which is what the bridge trains on. It comes off the entering skill's table in
the selector package, and the `entry` slider says which one, over the whole of that skill's
window, in the order selector.view draws it.

Choosing is the point of the arena, so nothing chooses for you. The selector used to rank
the entries by reachability and hand back the easiest; that answer is gone, and what is
left is the effort of whichever entry you asked for, printed beside it.

The rest of that skill's window is drawn with it, paler, and that is the difference from a
target standing on its own. An entry is one moment of a sequence, so the whole sequence is
placed the way the skill passes through it, spaced by EntryTable.trail and hung off the
target so the aimed state sits in its own place in the line. What that shows is how much of
the skill's run-up the hand-over is skipping, and, for a skill built around an object, where
that object puts the whole approach: the kick's line of ghosts runs back from the ball.

##
# Seeing where to be, before deciding when
##

For a skill with an object that line is drawn from the first step, not at the switch, and
`show the entry states` on the panel turns it off. That is the one thing here that helps
with the decision the operator is actually making.

A hand-over into an object skill has to be fired at a moment, and firing it late is not a
worse score, it is a miss: the robot walks through its own ball and the swing closes on
nothing. Nothing in the arena knows when that moment is, because the button is what a
controller would be replacing and this harness exists to be watched. So the question is how
to make the moment visible, and the answer is that it is a place rather than a time.

`Actor.arrive` already says where the robot has to stand for the skill to meet its object,
and an object does not move. Evaluated at each entry's own frame it gives a line of poses
fixed to the floor, drawn before anything is aimed, and pressing the button at the right
moment reduces to walking onto them. The viewer's line says the same number, the metres
still to go, so the switch can be fired on a figure closing on zero rather than on a guess.

Which needs the demand and not the trail. `EntryTable.trail` reconstructs the spacing from
the entries' own velocities and is a few centimetres out over half a second, and half a
second is most of the box a ball skill was trained in. `arrive` reads the clip. So the line
drawn before the switch is the exact one, and the one drawn after it is the trail hung off
whatever target the crossing was actually given. See `Run.demanded`.

A skill with nothing on the floor gets neither the line nor the checkbox. Its target is a
guess about where its own momentum will carry it, so it moves with the robot every step and
there is nothing to walk towards.

Either way the entering skill resumes at the step the entry was recorded at, not at its own
first frame. A tracking policy has no privileged beginning: any frame of its clip is a state
it continues from, provided the robot is in that state, which is what the bridge delivers.
Entering at frame zero instead would spend the opening replaying the run-up to a moment the
bridge has already crossed to.

The entry carries that step, so the state and the phase come from the same row and cannot
disagree. `aim` winds the clip to it when the switch fires and `resume` winds it there again
at the hand-over, because the clip keeps playing while the bridge crosses.

Which makes the hand-over measurable rather than plausible. After the bridge has crossed,
the entering skill drives for `entering_steps` and is scored on its own reward terms under
its own discount, so a hand-over that lands out of phase reads as a low number rather than
as a good number with a bad episode after it.

##
# What the panel drives
##

Each skill's folder is built from what that skill declares it can be told, so walking gets a
forward speed, a sideways speed and a heading, the strike skills get a ball speed and an
aim, and the clip trackers get no folder at all, because a jump or a punch combination is
one clip with nothing to aim. A skill's conditioning is applied only while that skill owns
the world, which is what lets the walk and the run share one velocity term without writing
over each other.

Read on two clocks, and a skill says which by what it declares. A `condition` is written
every step the skill drives, for a number the policy re-reads: a twist, a launch velocity.
An `enter` is called once, when the bridge is aimed, and it now sees the same values, for a
number that decides what the reference is rather than what to do with it. The jump's
distance is the second kind. It picks which of five clips plays and how far that clip is
stretched, and the anchor turns that clip to face the way the robot is going, so it has to
be settled before the reference is placed and cannot change afterwards.

The window has its own control and a way to let go of it. Released, it comes from the entry
the selector hands back, or is solved so the crossing lands where the entering skill needs
the robot. That is new, and it is the difference the bridge's duration argument makes: a
switch that could only pick *when* had to wait for the world to drift into agreement with a
fixed window, and one that can pick *how long* as well fires as soon as the demand is
reachable by some window the bridge was trained on. A ball skill fires early with a long
window when the ball is far and late with a short one when it is close.

Entry states come from the measured table in the selector package. Look at them, and
measure what a hand-over is worth, before spending a bridge on it:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.handoff

##
# The seam, which no score in here reports
##

A hand-over can be correct and still look wrong, and until recently every one staged here
did. A joint position target is default + scale * action, so the action a policy emits is
the PD setpoint; when the driver changes, the setpoint jumps by whatever the two policies
disagree about, in a single control step, and the actuators answer with a torque spike.
Measured on walk2kick at entry 3 that jump was three times the median change of the steps
around it, and up to thirteen times on the arrangement the end2end testbench started from.

Nothing else in this file sees it. The arrival score measures a state, the entering skill's
discounted return measures the seconds after, and both are blind to one bad step: across a
full ablation on walk2kick every crossing struck the ball while the seam varied by a factor
of seventeen. `--seam` is the only view of it. It prints the steps either side of the switch
and the switch step against its own neighbours.

Three settings, and they cure two different things:

    blend_steps   ramp out of the parting policy's last action over this many steps. The
                  action seam. Two policies fitted separately can agree about the state and
                  still disagree about what to command in it, and nothing handed across the
                  switch makes them agree, so the disagreement is spent over several steps
                  instead of one. Applied at both switches, not only the hand-over
    count_in      walk the entering tracker's clip up to its entry frame during the crossing
                  so it arrives there as control changes. The observation seam. Left off, the
                  clip plays on for the length of the window and is rewound at the hand-over,
                  which steps 29 reference angles, 29 reference rates, the phase and the
                  anchor error at once
    seam          measure it

What the action manager is left holding is the third piece and it needs no setting, because
this file already does the right thing by doing nothing: `resume` never writes the action
manager, so the entering skill reads the bridge's last action, which is the setpoint the
joints are actually tracking. Writing the entry's own recorded action there instead, which
looks like the tidier choice, is worse: it is consistent with the recording and inconsistent
with the robot that was delivered, and it measured twice the seam.

Both defaults are on. Turn them off to see what they fix:

    uv run python -m ...transitions.walk2kick --viewer none --auto 130 --entry 3 \
        --seam True --blend-steps 0 --count-in False

##
# Two things that are easy to get wrong
##

Where the target is put. The bridge trained on targets placed where a body carrying its
current momentum would arrive, at the centre of the reachable disc:

    (v_now + v_target) * T / 2   ahead of the robot

Dropping the target on top of the robot instead asks it to stop dead in every test, which
it was never asked to do in training, and reads as the bridge being worse than it is.

When the entering skill's reference is pinned. A clip tracker has to be told where its clip
is. Pinning it at hand-over, to wherever the robot actually ended up, silently erases the
arrival error: the clip moves to meet the robot and every hand-over looks perfect. So
`Actor.enter` is called when the target is chosen, not when control changes, and the clip
is pinned to where the robot is supposed to be. Whatever gap is left is the real one.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, make_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, NamedTuple, cast, get_type_hints

import mujoco
import numpy as np
import torch
import tyro

from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges import (
  BRIDGE_KINDS,
  DEFAULT_BRIDGE,
  BridgeKind,
  BridgeSpec,
  resolve,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  CHANNELS,
  ROOT_STATE_DIM,
  BridgeCommand,
  BridgeCommandCfg,
  Tolerances,
  arrival_score,
  channel_errors,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import (
  Entry,
  EntryTable,
  Reach,
  reaches,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import resume as entry_resume
from mjlab.tasks.bridging.experiments.humanoid.selector.reach import (
  lines as entry_lines,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  TABLE_PATH,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  Entry as RecordedEntry,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.entry_tolerances import (
  ToleranceOverrides,
  entry_tolerances,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul, yaw_quat
from mjlab.viewer.debug_visualizer import DebugVisualizer

ROBOT = "robot"
BRIDGE_GROUP = "bridge"

ARRIVE_SLACK = 0.10
"""How far apart, in metres, the predicted crossing and the spot an entering skill demands
may be when the switch fires. Only meaningful for a skill that demands one.

Sized to the error being corrected rather than to what the bridge tolerates, and the first
attempt got that backwards. `_draw_target` scatters its training targets around the
ballistic midpoint by up to `travel_speed * horizon`, so firing the moment the demanded spot
was anywhere inside that scatter looked defensible. Over a 0.6 s window it is 0.72 m, and
the bridge was told to cover most of a metre while also arriving nearly stopped:

    slack 0.72 m    0.40 m off the target, travelled 0.59x, score 0.146
    slack 0.10 m    0.08 m off the target, travelled 0.83x, score 0.554

Ten centimetres leaves the ballistic model deciding when, within a stride, and the object
deciding only where."""

SCORING_WINDOW_S = 3.0
"""How long the entering skill is scored for, in seconds.

The horizon a hand-over is judged over. Nothing in the selector fixes it, because nothing in
the selector scores a hand-over: it says where a skill can be entered and stops there."""

DEFAULT_DURATION_S = 0.6
"""How far ahead a target is placed when nothing more specific says, in seconds.

Near the middle of how far apart the corpus cuts a pair, which is where the entry states a
crossing has to reach were measured. A couple that sheds or builds more momentum than most
says so itself, and `Config.duration_s` overrides both."""

PROBE_S = 1.0
"""The window entries are ranked over, in seconds.

An effort is a required rate over an achievable one, so it scales with 1/seconds and the
probe decides the numbers rather than the order: the ranking is the same at 0.3 s and at
1.2 s. One second, because that makes an effort read as the seconds the entry needs."""

DEFAULT_SWITCH_STEP = 130
"""Where the panel's switch step starts when nothing asked for one, in control steps.

The timing transitions/walk2kick measures, which is a walk-up of about two and a half
seconds. It is a starting point for the slider and nothing reads it otherwise: the switch
stays on the button until the checkbox beside it is turned on."""

SWITCH_STEP_MAX = 400
"""How far out the panel's switch step reaches, in control steps. Eight seconds at 50 Hz,
which is longer than any approach here and short enough that the slider still resolves a
single stride. A larger --auto widens it rather than being clamped to it."""

SMOOTH_S = 0.30
"""Time constant of the root velocity the steering and the switch are decided on, in
seconds. About half a stride: long enough that the swing and stance halves of a step average
out, short enough that it still follows the walk turning onto the spot."""

MAX_ACCEL = 3.0
"""What a humanoid root sustains, in m/s^2, roughly.

Not a training parameter. The bridge learns from recorded stretches of motion, feasible
because a body performed them, so nothing in training decides whether a pair can be joined.
Here it does: the slider will happily ask for a velocity change in a tenth of a second, and
a score of zero on a window no body could cross says nothing about the bridge. This is a
sanity line printed next to the request, and nothing more."""


class Knob(NamedTuple):
  """One number a skill can be told, and the slider that tells it.

  A skill's conditioning is part of the skill, so it is declared beside it in actors.py
  rather than restated in every couple that uses it. The panel builds itself out of these,
  which means a skill gains a control by declaring one and nothing else has to be touched.
  """

  name: str
  low: float
  high: float
  step: float
  initial: float
  hint: str = ""


class Actor(NamedTuple):
  """One frozen skill, and how to tell it where it is taking over."""

  name: str
  task: str

  controls: tuple[Knob, ...] = ()
  """What this skill can be told while it drives. Empty for a skill that takes nothing,
  which is most of the clip trackers: a punch combination is one clip and there is no goal
  to aim it at."""

  condition: Callable[[ManagerBasedRlEnv, dict[str, float]], None] | None = None
  """Write the control values into whatever this skill reads. Called every step that this
  skill owns, so it must be idempotent.

  Applied to the skill that is driving and to nobody else. Two skills can share a command
  term, which walk and run do, and a term written by both at once holds whichever wrote
  last. That used to need a special case for exactly one couple; owning the term instead is
  the general version of it.
  """

  enter: (
    Callable[
      [ManagerBasedRlEnv, torch.Tensor, torch.Tensor, int, dict[str, float]], None
    ]
    | None
  ) = None
  """Put this skill's reference, and whatever else it keeps per episode, where it belongs.

  Called as `enter(env, pos, heading, frame, values)`: the world position the robot will be
  at, the direction it should head, the step of its own trajectory it is resuming from, and
  what its own controls are currently set to. A skill with no reference and no episode state
  of its own does not need one.

  `values` is the same dict `condition` reads, and the two are the same idea on different
  clocks. A number the policy re-reads every step is a `condition`, and it can be written a
  step after the bridge is aimed. A number that decides what the reference *is* has to be
  known before the reference is placed, because placement reads it: a goal that picks the
  clip or its stretch picks what `anchor_to_robot` then turns and winds. Written a step late
  it anchors the old clip and winds the new one to a frame that belongs to neither. So a
  skill with a goal of that kind declares a control and reads it here rather than declaring
  a `condition`. Nothing here has one now: every entering skill with a reference carries one
  clip at one scale.

  Placement only. What state the robot has to arrive in comes off the table: a state this
  skill was measured starting from. A reference is not one, and a retargeted clip is not one
  by several centimetres of floor.

  `frame` is the entry's own, so a tracker resumes its clip where the target state was
  recorded rather than at frame zero. Zero is right wherever the target is not an entry: a
  fresh episode, and the leaving skill going back to work after a transition.

  It is a frame of the clip, not a step count. These policies reset into a sampled frame of
  their reference, so the two are unrelated, and the selector records the clip frame for
  exactly this.

  A skill with no trajectory ignores it, which the walk, the run, the pass and the push all
  do. Every tracker uses it: the jump, the front kick and the punch combo each wind their
  clip to it. The line `cross` prints says the entry was recorded at that frame rather than
  that the skill resumes there, because only the trackers do.

  `pos` is where the robot is meant to be, not where it is, because the bridge has not
  crossed yet. See this module's header for why that is the difference between measuring a
  hand-over and faking one.
  """

  ready: Callable[[ManagerBasedRlEnv, torch.Tensor], torch.Tensor] | None = None
  """Whether this skill's precondition is met, so the switch can fire on the world.

  Called as `ready(env, at)` every step while the first skill drives, where `at` is where
  the robot is predicted to be when control would change. Returns one bool per environment.
  None means no precondition to watch and the switch waits for the button, which is what the
  jump does: a robot can jump anywhere.

  The prediction is the point. A trigger reading the present fires a stride late every time,
  because by the time the bridge has crossed the robot has walked on.

  Only called for a skill that declares no `arrive`. A demand is the stronger form of the
  same statement, since solving the window to land the crossing on the demanded pose puts
  the object in its box by construction, so for the pass and the push this is read for
  whether it is None and never evaluated. Which is the honest state of it rather than the
  intended one: what those two still need to say is that they fire themselves rather than
  waiting for a button, and a precondition is a roundabout way of saying so."""

  place: EventTermCfg | None = None
  """Where this skill's object starts, as a reset event. None for a skill without one."""

  robot: Callable[[EntityCfg], EntityCfg] | None = None
  """Patch the robot this skill needs, if it needs one patched.

  A skill that reads its observation off sites the bridge's robot does not carry cannot be
  served by that robot: the arena used to keep the bridge's plain robot, and the skill's own
  observation raised on a site it could not find. Declared here rather than detected,
  because detecting it means comparing two spec builders for equality and getting that wrong
  silently produces a robot with the wrong body on it.

  Patches from both skills of a couple compose, in the order leaving then entering.
  """

  arrive: Callable[[ManagerBasedRlEnv, int], torch.Tensor] | None = None
  """Where the robot has to end up, in world coordinates, for this skill's object to be in
  the box it was trained with. `(N, 3)`, and only the ground plane is read.

  Called as `arrive(env, frame)`, where `frame` is the entry the bridge is aiming at. Most
  skills ignore it: the pass wants the ball a quarter of a metre in front of its foot and
  that is the same demand whichever entry it opens from. A skill whose entries are a run-up
  does not. The kick's window is fifty frames of walking at a ball, so where the robot has
  to be depends on how much of that walk is left, and the answer moves by most of a metre
  across the window.

  Declaring this is what decides where the target goes. A skill that declares nothing is
  aimed at the ballistic midpoint, where a body carrying this momentum would arrive; a skill
  that declares one is aimed here, at the spot it says. There is no flag any more.

  There was, and taking it out is the point. It was off by default because commanding the
  arrival measured slightly worse on one number:

      walk2pass, ballistic placement    0.083 m from where the pass wants the robot
      walk2pass, commanded placement    0.104 m

  Both errors are three to ten centimetres against a box eight deep, so that comparison was
  never deciding anything, and it was measuring the wrong thing. A target is not judged on
  its own. The entry states drawn for the operator stand where this says, because that is
  where the skill works, and a bridge aimed anywhere else is aimed at a state whose object
  is out of reach however well it arrives. Two answers to one question, one of them shown on
  screen and the other one used, is not a trade to tune. So the demand places the target and
  the diagnostic below measures what that cost.

  The diagnostic is unchanged. Every hand-over into a skill that declares this prints how
  far the robot ended up from the spot the skill wants, which is the question an object
  skill is asking and the one the arrival score cannot answer: a hand-over can score well
  against a target half a metre from where the ball needs it."""


@dataclass(frozen=True)
class Couple:
  """Two skills and the hand-over between them. Everything else each skill brings itself."""

  leaving: Actor
  entering: Actor

  duration_s: float | None = None
  """How far ahead this couple places its target, in physical seconds, or None to solve it.

  None is the right default now that an entry carries its own figure: a crouch and a stand
  are not equally far from wherever a walk leaves the robot. Set it per couple to override,
  which is a real decision worth making by hand when one pair sheds more momentum than the
  entry's own figure assumed. See `Config.duration_s` for why this is not a window the
  bridge is given."""

  overshoot: float = 0.0
  """This couple's measured overshoot. See `Config.overshoot` for what it means and how to
  read it off a run. Here because it is a property of one pair of skills at one window: how
  far the bridge carries a robot leaving *this* skill past a target of *that* one."""


##
# The arena.
##


TARGET_COLOR = (1.0, 0.72, 0.2, 0.45)
"""The target ghost. Warm, so it reads as a goal rather than a second robot."""

TRAIL_COLOR = (0.45, 0.62, 1.0, 0.18)
"""The rest of the entering skill's window. Cool and faint, because these are context and
only one of them is being aimed at: the target has to stay the thing the eye lands on."""


class Aimed(BridgeCommand):
  """The bridge command with window drawing switched off, and its target drawn.

  In training a window is drawn on reset and the robot is teleported onto its start frame.
  Here the robot is wherever the leaving skill left it, which is the whole point, so the
  target arrives from outside through `aim` and nothing is ever teleported.
  """

  def __init__(self, cfg: BridgeCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.aimed = False
    self.trail: torch.Tensor | None = None
    """Every entry of the entering skill, placed in the world. `(M, 13 + 2J)` or None.

    Written from outside every step, and drawn whenever it is not None, which is what lets
    it appear before anything has been aimed. `Run` puts two different lines here: while the
    leaving skill drives it is where the entering skill's object demands the robot be, and
    from the switch on it is the line the crossing is actually aiming into. Never read by
    anything."""
    self._opened = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    """The environment step each window was opened on. See `step`."""
    # An identity rotation, because the zeros the base class opens target with are not one.
    # In training that never shows, since a window is drawn on the first reset and the zeros
    # are gone before anything reads them. Here nothing is ever drawn, so they stand until
    # the first aim, and the 6D rotation the policy reads divides by the quaternion's own
    # norm: what it reads for the whole leaving phase is six NaNs
    self.target[:, 3] = 1.0
    self._ghosts: dict[tuple[float, float, float, float], mujoco.MjModel] = {}

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids

  @property
  def step(self) -> torch.Tensor:
    """How far into the current window, in control steps, counted from `open_window`.

    The base class counts from the environment's own episode counter, which is right where
    it is written: in training a window is an episode, so the two start together.

    Here they do not. Nothing in this arena terminates, since a fall is the result being
    measured rather than an error to recover from, so that counter is the whole run's clock
    and is already in the hundreds by the time a hand-over fires. Left alone, every window
    would open past its own deadline. The policy cannot shrug that off: two numbers in its
    observation are the time left and the fraction of the window left.
    """
    return (self._env.episode_length_buf - self._opened).clamp(min=0)

  def open_window(
    self,
    env_ids: torch.Tensor,
    duration_s: torch.Tensor,
    previous_action: torch.Tensor | None = None,
    *,
    tolerances: Tolerances | torch.Tensor | None = None,
  ) -> None:
    """Start the clock on a target that was placed from outside.

    The base class does everything except note when the window opened, which it has no need
    of: in training a window is an episode and the environment's own counter is the clock.
    See `step` for why that is not true here.

    previous_action is what the entering skill last did. place writes it from the dataset
    when it teleports; a live hand-over teleports nobody and still has to, or the last
    action the policy reads is the leaving skill's. None leaves it alone, and so does a
    probe env with no action manager, which is what the unit tests drive this with.
    """
    super().open_window(env_ids, duration_s, tolerances=tolerances)
    self._opened[env_ids] = self._env.episode_length_buf[env_ids]
    actions = getattr(self._env, "action_manager", None)
    if previous_action is not None and actions is not None:
      actions.action[env_ids] = previous_action

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """The state the bridge is crossing to.

    A pose and nothing else, because a pose is all that can be drawn: half of a target is
    velocity and a still body says nothing about that. It shows where `aim` decided a body
    carrying this momentum could be after `steps`, and it stays put after the hand-over so
    the gap to the robot is the arrival error left standing.

    The rest of that skill's window goes with it, faint, standing where the skill would have
    passed through it on the way to the target. One state is a place to arrive; the line
    says what the arrival is part of, and for a skill with an object on the floor it says
    where that object put the whole approach.

    The two are gated separately, on purpose. No target is drawn before the first `aim`,
    because `target` opens as an identity quaternion at the world origin and a robot lying
    in the floor at the corner of the scene is not a target. The line has no such problem:
    an object standing on the floor says where a skill needs the robot long before anything
    is aimed, so `Run` fills it in from the first step and it is drawn from then on.
    """
    if self.trail is not None:
      self._draw_trail(visualizer, self.trail, TRAIL_COLOR, "entry")
    if self.aimed:
      self._draw(visualizer, self.target, TARGET_COLOR, "target")

  def _draw(
    self,
    visualizer: DebugVisualizer,
    states: torch.Tensor,
    color: tuple[float, float, float, float],
    tag: str,
  ) -> None:
    """One translucent robot per visualized env, standing in `states`, `(N, 13 + 2J)`."""
    for batch in visualizer.get_env_indices(self.num_envs):
      self._stand(visualizer, states[batch], color, f"{tag}_{batch}")

  def _draw_trail(
    self,
    visualizer: DebugVisualizer,
    states: torch.Tensor,
    color: tuple[float, float, float, float],
    tag: str,
  ) -> None:
    """One translucent robot per row of `states`, `(M, 13 + 2J)`.

    Rows are entries and not environments, which is why this cannot be `_draw`. A window is
    a sequence one robot passes through, so all of it is drawn for whichever env is being
    watched rather than one state per env.
    """
    if not list(visualizer.get_env_indices(self.num_envs)):
      return
    for row in range(states.shape[0]):
      self._stand(visualizer, states[row], color, f"{tag}_{row}")

  def _stand(
    self,
    visualizer: DebugVisualizer,
    row: torch.Tensor,
    color: tuple[float, float, float, float],
    label: str,
  ) -> None:
    """One translucent robot standing in one state, `(13 + 2J,)`."""
    ghost = self._ghosts.get(color)
    if ghost is None:
      ghost = self._ghosts[color] = self._tinted(color)

    indexing = self.robot.indexing
    free = indexing.free_joint_q_adr.cpu().numpy()
    joints = indexing.joint_q_adr.cpu().numpy()
    # From qpos0 rather than zeros: everything in this arena that is not the robot keeps
    # its own default, and a zero quaternion is not a rotation
    qpos = np.array(self._env.sim.mj_model.qpos0, dtype=np.float64)
    values = row.cpu().numpy()
    qpos[free[0:3]] = values[0:3]
    qpos[free[3:7]] = values[3:7]
    qpos[joints] = values[ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
    # alpha and not the colour's own: the viewer takes an rgba geom's opacity from this
    # argument and only its hue from the model
    visualizer.add_ghost_mesh(qpos, model=ghost, alpha=color[3], label=label)

  def _tinted(self, color: tuple[float, float, float, float]) -> mujoco.MjModel:
    """This arena's model with the robot painted `color` and everything else hidden."""
    ghost = copy.deepcopy(self._env.sim.mj_model)
    mine = set(self.robot.indexing.geom_ids.tolist())
    for geom in range(ghost.ngeom):
      solid = ghost.geom_contype[geom] or ghost.geom_conaffinity[geom]
      if geom in mine and not solid:
        ghost.geom_rgba[geom] = color
      else:
        # Everything else out of the way. Collision geoms are the crude convex stand-ins
        # the solver uses and draw a robot made of boxes. The rest of this arena is whatever
        # the two skills put on the floor, which at the target would be a second ball or a
        # second crate hanging in the air beside it
        ghost.geom_rgba[geom, 3] = 0.0
    return ghost


@dataclass(kw_only=True)
class AimedCfg(BridgeCommandCfg):
  """The imitation window, aimed from outside. Every other architecture derives its own."""

  def build(self, env: ManagerBasedRlEnv) -> Aimed:
    return Aimed(self, env)


@lru_cache(maxsize=None)
def aimed_type(cfg_cls: type[BridgeCommandCfg]) -> type[AimedCfg]:
  """The aimed config class for one architecture's window, derived from that window.

  An architecture is free to subclass the window, and the distillation bridge does: its
  config carries the keyframe mask and its command answers the observation term that reads
  the interior of the crossing. Copying only the base fields into AimedCfg dropped the
  extra ones, and building a plain BridgeCommand from them left the student's observation
  term with no command to read, which is why --bridge distillation used to stop here.

  `Aimed` comes first in both base lists, so its overrides win over the architecture's and
  `isinstance(command, Aimed)` still answers yes whichever architecture is loaded. The
  command class is read off `build`'s return annotation rather than kept in a table, so an
  architecture declares it in the one place it already had to.

  Cached because a dataclass built twice is two types, and a config compared across them
  is not equal to itself.
  """
  if issubclass(cfg_cls, AimedCfg):
    return cfg_cls
  if cfg_cls is BridgeCommandCfg:
    return AimedCfg
  command_cls = get_type_hints(cfg_cls.build)["return"]
  aimed_cls = type(f"Aimed{command_cls.__name__}", (Aimed, command_cls), {})
  return dataclass(kw_only=True)(
    type(
      f"Aimed{cfg_cls.__name__}",
      (AimedCfg, cfg_cls),
      {"build": lambda self, env, cls=aimed_cls: cls(self, env)},
    )
  )


def aimed_cfg(trained: BridgeCommandCfg) -> AimedCfg:
  """The architecture's own window config, with its target supplied from outside.

  Same fields, so the policy reads what it trained on, and the two callers that stage a
  crossing outside training, this module and the parkour demo, build it the same way.
  """
  aimed = {f.name: getattr(trained, f.name) for f in fields(trained)}
  # Draws the target ghost, and gets the viewer to offer a Bridge checkbox under
  # Scene > Debug Viz that turns it off again
  aimed["debug_vis"] = True
  # No corpus. No window is ever drawn in this arena, since `Aimed` takes its target from
  # outside, so the bridge's training dataset was being read for a frame rate the
  # environment already knows. It also means a transition can be staged before the corpus
  # has been built, which is the order the work actually happens in
  aimed["dataset_path"] = None
  return aimed_type(type(trained))(**aimed)


def bridge_group(kind: str) -> str:
  """What one architecture's observation group is called in an arena hosting several.

  The one that drives at startup keeps the plain name, so an arena built for a single
  architecture is what it always was and every other caller of `arena` is untouched.
  """
  return f"{BRIDGE_GROUP}_{kind}"


def widest_window(cfgs: dict[str, ManagerBasedRlEnvCfg]) -> BridgeCommandCfg:
  """The one window config that serves every architecture here, or a refusal.

  Several architectures in one arena share one command term, and they do not all want the
  same one: the distillation bridge's observation reads the interior of the crossing and
  only a MaskedBridgeCommand has one. So the term is the most specific of them, which
  serves the others because the specific one is a subclass of the general one.

  Two architectures subclassing the window in ways neither covers is not a thing to guess
  at, so it is refused by name rather than served by whichever came first.
  """
  windows: dict[str, BridgeCommandCfg] = {}
  for kind, cfg in cfgs.items():
    window = cfg.commands["bridge"]
    assert isinstance(window, BridgeCommandCfg)
    windows[kind] = window
  for window in windows.values():
    if all(isinstance(window, type(other)) for other in windows.values()):
      return window
  raise SystemExit(
    "These architectures do not share one window: "
    + ", ".join(f"{k} wants {type(w).__name__}" for k, w in windows.items())
    + ". Run them one at a time with --bridge."
  )


def arena(
  couple: Couple, bridge: BridgeSpec, also: tuple[BridgeSpec, ...] = ()
) -> ManagerBasedRlEnvCfg:
  """The bridge's own environment, with both skills' machinery merged into it.

  Built on the chosen bridge's play config rather than a skill's, so the robot, the terrain
  and the bridge's observation are exactly what that architecture trained against. Each skill
  then contributes what it needs and nothing else: its entities, its commands, the sensors
  its observation reads, and the observation itself.

  The observation is copied from the skill's own task verbatim. A checkpoint is tied to its
  term list in order, and feeding it the same numbers shuffled does not fail, it acts on
  nonsense. So the order comes from the one place that cannot disagree with the checkpoint,
  rather than being restated here where it would go stale.

  `also` names architectures that get an observation group here without driving, so a
  viewer can swap between them on one crossing. They can share this arena because the three
  of them differ in their window term and their observation and in nothing else: the robot,
  the floor, the action convention and the corpus are the bridge problem's, not any one
  architecture's, which is why bridges/dataset sits above all three. What that buys is a
  comparison where the only thing that changed is the policy.

  Empty is one architecture and the arena it always built.
  """
  assert bridge.env_cfg is not None  # resolve refuses a stub before it gets here
  cfg = bridge.env_cfg(play=True)
  built = {bridge.kind: cfg}
  for spec in also:
    assert spec.env_cfg is not None
    built[spec.kind] = spec.env_cfg(play=True)

  cfg.commands["bridge"] = aimed_cfg(widest_window(built))
  cfg.observations = {BRIDGE_GROUP: cfg.observations["actor"]} | {
    bridge_group(spec.kind): built[spec.kind].observations["actor"] for spec in also
  }

  # Nothing should reset: a fall is the result of the test, not an error to recover from,
  # and a reset mid-run would move the robot out from under the phase machine
  cfg.terminations = {}

  # The entering skill's own reward, carried over so a hand-over is judged by what the skill
  # itself is paid for rather than by a second opinion. Nothing is trained here and nothing reads this to act; it is
  # computed every step and the run reads the last value off env.reward_buf. Its terms are
  # the ones that decide whether a jump was a jump, and restating them here would be a
  # second opinion that could drift from the first
  cfg.rewards = copy.deepcopy(
    load_env_cfg(couple.entering.task, play=True).rewards or {}
  )

  for actor in (couple.leaving, couple.entering):
    task = load_env_cfg(actor.task, play=True)
    cfg.observations[actor.name] = replace(
      task.observations["actor"], enable_corruption=False
    )
    for name, entity in (task.scene.entities or {}).items():
      cfg.scene.entities.setdefault(name, entity)
    # The robot is the one entity every skill already has, so `setdefault` never reaches it
    # and a skill that needs it modified has to say so
    if actor.robot is not None:
      cfg.scene.entities[ROBOT] = actor.robot(cfg.scene.entities[ROBOT])
    for name, command in task.commands.items():
      # Frozen, because in this arena the harness owns when a skill's goal changes and
      # Actor.enter is how it says so. A guard, not a fix for anything observed: both skills
      # carrying a goal already resample on a timer long enough never to fire, and the
      # twist's is overwritten every step below. What it rules out is a skill's goal moving
      # under a transition halfway through being measured.
      #
      # gui and debug_vis go off with it. Both are for watching a skill train on its own:
      # the sliders duplicate the panel below and fight it for the same fields, and the
      # ghost draws the jump's reference clip, which in a transition is a second robot
      # standing in the scene that is not what is being tested
      cfg.commands.setdefault(
        name,
        replace(
          command,
          resampling_time_range=(1.0e9, 1.0e9),
          gui=False,
          debug_vis=False,
        ),
      )
    # And the sensors those observations read. The pass watches a foot-ball contact, the
    # push a robot-crate one, and neither exists in an arena that never saw the object
    have = {sensor.name for sensor in (cfg.scene.sensors or ())}
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
      sensor for sensor in (task.scene.sensors or ()) if sensor.name not in have
    )

  # The bridge's budgets were sized for a bare plane and one robot. This arena carries
  # whatever the two skills brought, and a constraint dropped on overflow is a contact that
  # silently did not happen, which looks like a robot sinking into the floor rather than an
  # error. 500 because the broadphase asked for 469 on the jump, the couple that needs the
  # most. njmax is left alone because nothing has reported overflowing it
  cfg.sim.nconmax = 500
  cfg.sim.njmax = 800
  cfg.sim.contact_sensor_maxmatch = 500

  # Each skill puts its own object out, with its own placement function. Deep copied
  # because building an environment resolves names into indices inside the config it is
  # handed, and a term shared between two environments carries the first one's indices into
  # the second
  for actor in (couple.leaving, couple.entering):
    if actor.place is not None:
      cfg.events[f"place_{actor.name}"] = copy.deepcopy(actor.place)
  return cfg


##
# Loading a frozen policy.
##


def find_checkpoint(experiment: str, explicit: Path | None = None) -> Path:
  """The newest checkpoint of an experiment, printed.

  Picked by modification time when not given, which is convenient and has gone wrong here
  before: a stale run left in logs/ outranks the one you meant. The path is printed rather
  than assumed, so loading the wrong policy is at least a visible mistake.
  """
  if explicit is not None:
    if not explicit.exists():
      raise SystemExit(f"No checkpoint at {explicit}.")
    return explicit
  root = Path("logs") / "rsl_rl" / experiment
  found = sorted(root.rglob("model_*.pt"), key=lambda p: p.stat().st_mtime)
  if not found:
    raise SystemExit(f"No checkpoint under {root}. Train '{experiment}' first.")
  return found[-1]


def offered_bridges(
  wanted: tuple[str, ...], driving: str
) -> tuple[tuple[BridgeSpec, Path], ...]:
  """Every architecture the dropdown can offer, with the checkpoint each would load.

  Resolved before the simulator is built, which is the whole reason this is separate from
  loading them: the arena has to know which architectures it is hosting in order to carry
  an observation group for each, and finding out by trying to load a policy needs the
  environment that decision goes into.

  The one that drives is first and is required. The rest are dropped with a line when the
  package is a stub or its experiment holds no checkpoint, because an architecture nobody
  has trained yet is the normal state of one of them and a run that refused on that would
  be a run nobody could start.
  """
  order = (driving, *(kind for kind in wanted if kind != driving))
  found: list[tuple[BridgeSpec, Path]] = []
  for kind in order:
    try:
      spec = resolve(kind)
      found.append((spec, find_checkpoint(spec.experiment)))
    except SystemExit as missing:
      if kind == driving:
        raise
      print(f"[bridge] {kind}: not offered, {missing}")
  return tuple(found)


BASE_VARIANTS = ("base", "baseline")
"""Variant names meaning the skill's own experiment, with no suffix."""


def variant_checkpoint(
  experiment: str, variant: str, explicit: Path | None = None
) -> Path:
  """The newest checkpoint of one named version of a skill.

  A variant is a suffix on the skill's experiment name, so for the kick "base" is g1_kick as
  it was trained, "robust" is what skills/finetune.py wrote to g1_kick_robust and "recover"
  is what skills/recover.py wrote to g1_kick_recover. One vocabulary for every script that
  compares versions, so a version is named once instead of pasted as a path.
  """
  if explicit is not None:
    return find_checkpoint(experiment, explicit)
  suffix = "" if variant in BASE_VARIANTS else f"_{variant}"
  return find_checkpoint(f"{experiment}{suffix}")


class Policy:
  """A frozen actor, reading one named observation group of this arena."""

  def __init__(
    self, task: str, checkpoint: Path, env: ManagerBasedRlEnv, group: str, device: str
  ) -> None:
    agent = load_rl_cfg(task)
    # Point both roles at this arena's group instead of the task's own. load_rl_cfg hands
    # back a deep copy, so the registry is not disturbed
    agent.obs_groups = {"actor": (group,), "critic": (group,)}
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = runner_cls(wrapped, asdict(agent), device=device)
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    self._policy = runner.get_inference_policy(device=device)
    self._group = group

  @torch.no_grad()
  def __call__(self, obs) -> torch.Tensor:
    from tensordict import TensorDict

    state = obs[self._group]
    assert isinstance(state, torch.Tensor)
    return self._policy(TensorDict(obs, batch_size=[state.shape[0]]))


##
# States, and how to aim at one.
##


def state(robot: Entity) -> torch.Tensor:
  """(N, 13 + 2J), the layout the bridge's target uses."""
  data = robot.data
  return torch.cat(
    [
      data.root_link_pos_w,
      data.root_link_quat_w,
      data.root_link_lin_vel_w,
      data.root_link_ang_vel_w,
      data.joint_pos,
      data.joint_vel,
    ],
    dim=-1,
  )


def turn_state(state: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
  """A state facing a different way. Position is the caller's to place."""
  out = state.clone()
  out[:, 3:7] = quat_mul(turn, state[:, 3:7])
  out[:, 7:10] = quat_apply(turn, state[:, 7:10])
  out[:, 10:ROOT_STATE_DIM] = quat_apply(turn, state[:, 10:ROOT_STATE_DIM])
  return out


def facing(entry: torch.Tensor, here: torch.Tensor) -> torch.Tensor:
  """One entry state, turned to face the way the robot faces now.

  Every skill here is egocentric, so a state it can start from facing one way is one it can
  start from facing another. The turn has to reach the orientation and both velocity vectors,
  or the pose and the momentum disagree about which way the body is going.

  Ground position is not touched, because an entry state carries no ground position at all:
  the selector canonicalizes it away. The caller places it.
  """
  turn = quat_mul(yaw_quat(here[:, 3:7]), quat_conjugate(yaw_quat(entry[:, 3:7])))
  return turn_state(entry, turn)


def trail_states(
  table: EntryTable,
  skill: str,
  aimed: str,
  here: torch.Tensor,
  target: torch.Tensor,
) -> torch.Tensor:
  """One skill's whole window, turned and placed so the aimed entry lands on the target.
  `(M, 13 + 2J)`.

  What the target is one of. The bridge aims at a single state, and a single state says
  nothing about how far into the skill it is or what the skill was doing on the way to it,
  which is most of what somebody watching a hand-over wants to know.

  Two placements, and the second is the whole idea. Every entry is turned to face the way
  the robot faces, the same rotation `aim` gives the target, so the line points where the
  robot is going. Then the line is shifted bodily until the aimed entry sits exactly on the
  target, which means the other entries land wherever they stand relative to it: earlier
  ones behind, later ones ahead, at the distances EntryTable.trail measured.

  So the line follows the target rather than being drawn beside it, and for a skill whose
  target is fixed by an object, the object decides where the whole approach goes. Aim the
  kick at a ball and the run-up appears behind the ball.

  Approximate in exactly one way worth naming. The trail integrates the entries' own
  velocities, so it is the same ballistic model `crossing` uses and not a recorded path.
  Over half a second of a run-up that is a few centimetres.
  """
  entries = table.of(skill)
  index = next(i for i, entry in enumerate(entries) if entry.name == aimed)
  rows = torch.as_tensor(
    np.stack([entry.state for entry in entries]), dtype=here.dtype, device=here.device
  )
  offsets = torch.as_tensor(table.trail(skill), dtype=here.dtype, device=here.device)
  placed = facing(rows, here[0:1].expand(rows.shape[0], -1))
  heading = yaw_quat(here[0:1, 3:7]).expand(rows.shape[0], -1)
  shift = quat_apply(heading, offsets - offsets[index])
  placed[:, 0:2] = target[0, 0:2] + shift[:, 0:2]
  return placed


def crossing(
  here: torch.Tensor, target: torch.Tensor, horizon: float, overshoot: float = 0.0
) -> torch.Tensor:
  """Where the robot ends up on the ground once the bridge has crossed. (N, 2).

  A body changing velocity steadily from the one it has to the one it is asked for covers
  the mean of the two, which is where training put every target and therefore where `aim`
  has to put this one.

  One function with two callers, and that is the point. `aim` calls it to place the target
  and `Run.triggered` calls it to decide when to fire, and those two answers have to be the
  same distance or the switch is measuring a crossing that is not the one about to happen.
  They used to differ: the trigger predicted `v_now * horizon / 2`, which is this formula
  with the target standing still. Every target that moves is placed further out than that,
  so the robot walked past its object by however much momentum the entering skill wanted.
  Invisible on a skill that can locomote out of the error, fatal on one that kicks in place.

  `overshoot` is the caller's, and only the trigger passes it. See `Config.overshoot`.
  """
  reach = (here[:, 7:9] + target[:, 7:9]) * horizon / 2.0
  return here[:, 0:2] + reach * (1.0 + overshoot)


def crossing_time(
  here: torch.Tensor, target: torch.Tensor, want: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """How long a crossing needs to land on `want`, and how far off the line it would still be.

  `crossing` read backwards. It says where the robot ends up given a duration; this says
  which duration ends up somewhere, which is the question worth asking now that duration is
  something the bridge takes rather than something baked into it.

  The difference is not cosmetic. With a fixed window the switch could only wait for the
  world to drift into agreement with it, so a skill that needed the robot half a metre
  further on had to be walked towards until the arithmetic happened to work out, and if the
  robot was already too close the moment never came at all. Solving for the duration turns
  that around: the demand is met by choosing the window, and the switch fires as soon as the
  window it would need is one the bridge was trained on.

  The crossing travels along the mean of the two velocities, so the reachable set is a line
  and not the plane. `t` is the projection onto it, and `residual` is what is left over: how
  far off that line the demand sits, which no duration can close and which the leaving skill
  has to steer out instead. Both are needed. A `t` inside the trained range with a metre of
  residual is a crossing that arrives on time somewhere else.
  """
  along = (here[:, 7:9] + target[:, 7:9]) / 2.0
  gap = want - here[:, 0:2]
  speed_sq = (along * along).sum(dim=-1).clamp(min=1.0e-6)
  seconds = (gap * along).sum(dim=-1) / speed_sq
  residual = (gap - along * seconds.unsqueeze(-1)).norm(dim=-1)
  return seconds, residual


def defaults(actor: Actor) -> dict[str, float]:
  """What a skill is being told when nobody is telling it anything.

  One dict built from the actor's own declarations, so a caller with no panel, and `aim`
  when it is handed none, mean the same thing by "unset" as the panel does at startup.
  """
  return {knob.name: knob.initial for knob in actor.controls}


def entry_target(
  env: ManagerBasedRlEnv,
  entering: Actor,
  entry: torch.Tensor,
  here: torch.Tensor,
  duration_s: float,
  frame: int,
  arrive=None,
  values=None,
  recorded: RecordedEntry | None = None,
) -> torch.Tensor:
  """Place one entry and its reference together, also used by previews and triggers."""
  if recorded is not None:
    entry_resume.prepare(env, recorded)
  heading = yaw_quat(here[:, 3:7])
  target = facing(entry, here)
  target[:, :2] = (
    arrive(env, frame)[:, :2]
    if arrive is not None
    else crossing(here, target, duration_s)
  )
  if entering.enter is not None:
    entering.enter(env, target[:, :3], heading, frame, values or defaults(entering))
  if "motion" in env.command_manager.active_terms and (
    recorded is None or recorded.motion_file
  ):
    if isinstance(env.command_manager.get_term("motion"), JumpCommand):
      if recorded is None:
        raise ValueError("Tracking handoffs require the complete recorded entry")
      target = entry_resume.target(env, recorded)
  return target


def aim(
  env: ManagerBasedRlEnv,
  command: Aimed,
  entering: Actor,
  entry: torch.Tensor,
  here: torch.Tensor,
  duration_s: float,
  frame: int = 0,
  arrive: Callable[[ManagerBasedRlEnv, int], torch.Tensor] | None = None,
  values: dict[str, float] | None = None,
  recorded: RecordedEntry | None = None,
  *,
  tolerances: Tolerances | torch.Tensor | None = None,
) -> torch.Tensor:
  """Place the reference first, then transform the recorded robot with it."""
  if tolerances is None:
    tolerances = entry_tolerances(
      entering.name, recorded.frame if recorded is not None else frame
    )
  target = entry_target(
    env, entering, entry, here, duration_s, frame, arrive, values, recorded
  )
  env_ids = torch.arange(command.num_envs, device=command.device)
  # open_window crosses to the target the command already holds, so it is written first
  command.target[:] = target
  command.aimed = True
  # Seconds, not ticks. The command converts, and it is the only thing that should
  command.open_window(
    env_ids,
    torch.full((command.num_envs,), duration_s, device=command.device),
    tolerances=tolerances,
  )
  return target


##
# The run.
##


@dataclass(frozen=True)
class Config:
  tolerances: ToleranceOverrides = field(default_factory=ToleranceOverrides)
  """Override individual physical limits from tests/entry_tolerances.py."""

  duration_s: float | None = None
  """How far ahead to place a target nothing else fixes, in seconds, or None to solve it.

  Not how long the bridge gets: it is given no duration at all. This places the target for a
  skill with no object, and it buys the crossing patience before an unsuccessful one is
  abandoned. A skill that says where it needs the robot ignores the first half of that.
  """

  entry: int = 0
  """Which state to aim at, in the order selector.view draws them. The slider moves it.

  Table order, so an index means one particular posture for the whole run: a list that
  reordered as the robot moved would be a different state every step."""

  table: Path | None = None
  """Which file the states come from. None is what selector.build writes."""

  entering_steps: int = 220
  """Control steps the entering skill drives for before the verdict is printed. Long enough
  that a skill which is going to fail from a bad entry has done so."""

  speed: float | None = None
  """Forward command for the leaving skill, in m/s, or None for whatever its own control
  declares. Only meaningful for a skill that has a `forward` control."""

  tell: dict[str, float] = field(default_factory=dict)
  """What the entering skill is asked for, by control name, overriding its declared default.

  `speed` for the other end of the couple, and general because the entering skills are. The
  panel is the usual way to set these and it only exists under a viewer, so a headless run,
  which is what `--auto` is for, had no way to ask for anything but the default:

      uv run python -m ...transitions.walk2jump --auto 120 --tell "{'distance': 2.0}"

  A name the skill does not declare is refused rather than ignored, because a typo that
  silently leaves the default in place looks exactly like the skill not responding to the
  control."""

  preview: bool = True
  """Whether to draw the entering skill's entry states while the leaving skill is still
  driving, for a skill that has an object.

  On, because it is the only thing in this harness that helps with the decision the operator
  actually has to make. See `Run.demanded`. The panel has a checkbox for it, and a skill
  with no object gets neither: nothing stands still for it to be drawn against."""

  baseline: float | None = None
  """Discounted return the entering skill earns from a perfect arrival, for a verdict.

  Read off the `perfect` column of tests/handoff.py, which measures exactly this: the same
  skill, over the same window, teleported into the entry state instead of bridged to it.
  None prints the raw number and refuses the verdict, because a share of an invented
  baseline is not a measurement."""

  bar: float = 0.8
  """Share of that baseline a hand-over has to reach to pass."""

  overshoot: float = 0.0
  """How much further the robot really travels than the target it was given, as a fraction.

  Applied to the switch and to nothing else, which is why it works. The trigger asks whether
  the object will be in reach when control changes; `aim` asks where the target should go.
  Those look like one quantity and are not: the target is where the robot is told to be, the
  arrival is where it ends up, and the gap is the bridge's own tracking error. Correct both
  and it cancels exactly, since the trigger fires later and the target moves the same
  distance further, leaving the object in the same wrong place. Correct only the trigger and
  the object lands where the skill wants it.

  Zero until measured. Every hand-over prints `travelled`, the distance actually covered
  over the distance the placement predicted. Set this to that number minus one."""

  blend_steps: int = 8
  """Control steps to ramp out of the parting policy's last action at each switch.

  Zero is a hard switch, which is what this harness did and what makes a hand-over visible.
  A joint position target is default + scale * action, so the action a policy emits is the
  PD setpoint, and at a hard switch it jumps by whatever the two policies disagree about in
  one control step. Measured on walk2kick at entry 3 that was eight to thirteen times the
  median change of an ordinary step of the entering skill, and the body was thrown a step
  later at nearly three times the ordinary joint velocity.

  No amount of delivering the robot accurately removes it. The two policies were fitted
  separately, so they can agree about the state and still disagree about what to command in
  it. What a ramp does is spend that disagreement over several control steps instead of one,
  which makes the joint targets continuous across the switch.

  Eight is 0.16 s at 50 Hz. The seam is gone by four; sixteen starts to be a stretch of the
  motion during which neither policy is in control. Set it to zero to see what it fixes.
  """

  count_in: bool = True
  """Whether the entering tracker's clip counts up to its entry frame during the crossing.

  Off, which is what this harness did, the clip plays on while the bridge crosses and is
  rewound at the hand-over. Over a one second window that is fifty frames, and the rewind is
  a step change in every observation term the entering policy reads off its motion: 29
  reference angles, 29 reference rates, the phase and the anchor error.

  On, the reference is held `left` frames short of the entry and marches forward one frame
  per step, arriving exactly as control changes. The entering skill then reads a reference
  that has been approaching it smoothly and there is no rewind to make. It is the cure for
  the observation half of a seam and does nothing for the action half; the two are separate
  and `blend_steps` is the other one.

  Nothing for a skill with no reference, which the pass and the push are."""

  seam: bool = False
  """Measure every step and print the hand-over as a waveform when the run ends.

  What this costs is a tensor per step, and what it buys is the only view of a transition
  that shows whether the switch was visible. A hand-over can score perfectly and still snap:
  across the full ablation on walk2kick every crossing struck the ball and the seam ranged
  over a factor of seventeen, so nothing about the outcome reports it."""

  slack: float = ARRIVE_SLACK
  """How far off the crossing line the spot may sit when the switch fires, in metres.

  The alignment half of the distance trigger. `steer` is what drives this down and this is
  what waits for it, so a run where the walk cannot line up does not hand over at all, which
  is a result and prints as one, rather than handing over sideways and blaming the bridge."""

  steer: bool = True
  """Whether the leaving skill is aimed at the spot the entering skill needs.

  On, and only for a couple whose entering skill declares `Actor.arrive`, because only those
  have a spot. It writes the leaving skill's heading control every step, so the heading
  slider does nothing while it is on and the panel has a checkbox to hand it back.

  This is the fix for the thing that made hand-overs into the kick miss the ball, and it is
  not where anybody looks for it. The arrival was never far off: measured over eight
  placements the robot landed within 12 cm of the spot, inside the arrival tolerance on most
  channels, and the swing still passed 20 cm wide of the ball. What was wrong was sideways,
  and sideways is the one error nothing downstream can absorb. Every frame of the anchored
  clip lies on the same line, so no choice of resume frame moves the swing across it, and
  the kick can only watch the ball go by.

  What produces it is that a crossing travels along the mean of the robot's velocity and the
  target's, and a kick entry is the middle of a run-up whose pelvis carries several degrees
  of sideways velocity. So the crossing line leaves the body at an angle to the heading, and
  a walk aimed at the spot puts the line past it. Aiming the line instead:

      steered      8 of 8 placements score, sole 0.04 m across the ball at the strike
      not          4 to 7 of 8, sole 0.09 to 0.13 m across it

  See `off_line` for the correction and why it is fed back rather than solved.
  """

  window_s: float = 1.0
  """Seconds the bridge gets when the switch fires on distance.

  A second, measured, and it is not the 0.6 s that places a target for a skill with nothing
  on the floor. Those are different jobs: that one is a guess about where momentum carries a
  body, this one is how long the bridge is given to arrive somewhere it has been told about.

  The difference is most of whether the kick works. At 0.6 s the crossing was bimodal over
  five resets of walk2kick at entry 3, three landing 0.14 m from the spot and two at 0.33 m,
  and the bad ones arrived travelling 0.05 m/s at a target that wants 0.93 and had sailed
  half again as far as they were asked to. At 1.0 s the same five land 0.01 to 0.19 m, the
  bimodality is gone, and the best of them arrives at 0.95 m/s against 0.93.

  `duration_s` still overrides this, and did the measuring.
  """

  fire_at: float = 0.5
  """How far from where the entering skill needs the robot to fire the switch, in metres.

  What replaces the switch step for a skill with an object. An entry of a skill built around
  one is a place, not a moment: `Actor.arrive` says where the robot has to stand for the
  swing to meet the ball, the ball does not move, so the whole of choosing when is walking
  onto that spot. Firing half a metre before it is a statement anybody can repeat and that
  means the same thing at any approach speed.

  The step is not. A hand-over pinned to control step 130 lands wherever two and a half
  seconds of walking happened to put the robot, which moves with the walk's commanded speed,
  with the ball's spawn and with how much turning the approach needed. Two runs that differ
  only in approach speed were crossing from different places and being compared as though
  they were not.

  Read against the metres the viewer counts down beside the active skill, which is the same
  number this fires on.
  """

  auto: int | None = None
  """Fire the switch on control step N of every episode, regardless of the world.

  Needed headless for a skill with no precondition of its own, and an override for one that
  has. Under a viewer it is what the panel's switch step starts at, and setting it turns
  that checkbox on, so a run started with --auto 130 repeats the same crossing every
  episode until somebody turns it off. Which is what a formal comparison needs: the entry
  fixes the state being aimed at, this fixes the state being aimed from."""

  patience: int = 900
  """Steps a headless run gets before it gives up waiting for the switch to fire."""

  bridge: BridgeKind = DEFAULT_BRIDGE
  """Which bridge architecture drives the hand-over to begin with. See bridges/__init__.py.

  The arena is built on the chosen one's own play config, so switching this switches the
  observation the bridge policy reads as well as the checkpoint it loads. The two skills are
  untouched by it, which is what makes two architectures comparable on one transition.

  Under a viewer this is the dropdown's starting position rather than the whole choice. See
  `bridges`."""

  bridges: tuple[BridgeKind, ...] = BRIDGE_KINDS
  """Which architectures to load into the panel's bridge dropdown.

  All of them, because comparing two on one crossing is the reason there is more than one,
  and a dropdown that has to be asked for is a comparison nobody makes. One that has no
  code yet, or no checkpoint under its experiment, is dropped with a line saying so rather
  than failing the run: an untrained architecture is the normal state of one of them.
  `bridge` is the exception and is required, since that is the one that was asked for.

  Narrow it to spend less time loading, or to be sure a headless run is the architecture it
  says it is:

      uv run python -m ...transitions.walk2kick --bridges "('imitation',)"
  """

  bridge_checkpoint: Path | None = None
  """An explicit checkpoint for the bridge, instead of the newest under its experiment.

  Each skill has one of these too, named after the skill rather than after its role, so
  walk2kick takes --walk-checkpoint and --kick-checkpoint. They are added per couple by
  `config_for`, since which two skills exist is the couple's to say."""

  viewer: Literal["viser", "none"] = "viser"
  device: str | None = None
  seed: int = 0


def checkpoint_flag(actor: Actor) -> str:
  """What one skill's checkpoint flag is called. One place, so `main` and `config_for`
  cannot spell it differently."""
  return f"{actor.name}_checkpoint"


def config_for(couple: Couple) -> type[Config]:
  """Config, plus a checkpoint flag for each of this couple's two skills.

  Named after the skill and not after its role, because that is how anybody running one of
  these thinks about it: walk2kick takes --walk-checkpoint and --kick-checkpoint, and
  --bridge-checkpoint is the third. Each defaults to the newest under that skill's own
  experiment, which is what `find_checkpoint` picks and what has gone wrong before, so
  naming one by hand is the way to pin a comparison to a particular pair of policies.

  Built per couple rather than declared once as two role-shaped fields, so the flag a
  script takes says which policy it swaps without anyone having to remember which skill is
  leaving. Frozen like Config, since a non-frozen dataclass cannot inherit from one.
  """
  actors = (couple.leaving, couple.entering)
  assert len({actor.name for actor in actors}) == len(actors), (
    "A couple's two skills need different names, or their checkpoint flags collide"
  )
  built = make_dataclass(
    f"{couple.leaving.name}2{couple.entering.name}",
    [
      (
        checkpoint_flag(actor),
        Annotated[
          Path | None,
          tyro.conf.arg(
            help=f"Checkpoint for the {actor.name} policy. None is the newest one."
          ),
        ],
        field(default=None),
      )
      for actor in actors
    ],
    bases=(Config,),
    frozen=True,
  )
  return cast("type[Config]", built)


class Seam:
  """Every step of every env, kept so the hand-over can be read as a waveform.

  The question this answers is not whether the kick recovered, which the sweep already
  answers. It is whether anything happened at the instant control changed that would not
  have happened if one policy had been driving throughout, because that is what somebody
  watching the transition sees.

  Four channels, and they are four different questions:

      action   what was asked of the actuators. A joint position target is
               default + scale * action, so a step change here is a step change in the PD
               setpoint and in the torque that follows. This is the one that shows
      joint    where the joints actually went. Lags the action by the actuator, so a spike
               here one step after a spike there is the body being thrown rather than moved
      vel      joint velocity, where a torque spike lands first
      obs      what the entering policy read. A jump here would explain a jump in the
               action without being one, and the two have different cures: an observation
               seam is fixed by handing the skill a world consistent with the one it was
               reading a step earlier, an action seam by what drives the actuators across
               the switch

  Only the worst channel of each is kept, per env per step. A hand-over is abrupt if any
  joint is thrown, and a mean over 29 of them hides the one that was.
  """

  KEYS = ("action", "joint", "vel", "obs")

  NEIGHBOURHOOD = 8
  """How many steps either side of the switch the switch is judged against. See `ordinary`."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    num_envs: int,
    names: tuple[str, str, str] = ("leaving", "bridge", "entering"),
  ) -> None:
    self.env = env
    self.names = names
    """What to call each phase in the table. The couple's own, so a transition into the
    pass does not print a column headed kick."""
    self.steps: list[dict[str, torch.Tensor]] = []
    self.switch = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
    """Index into steps of the first step the entering skill drove, per env."""
    self._last: dict[str, torch.Tensor] = {}

  def record(
    self,
    action: torch.Tensor,
    obs,
    group: str,
    clip: torch.Tensor,
    phase: torch.Tensor,
  ) -> None:
    """One step, as it is about to be applied. `phase` is who is about to drive."""
    robot = self.env.scene[ROBOT]
    now = {
      "action": action.detach(),
      "joint": robot.data.joint_pos,
      "vel": robot.data.joint_vel,
      "obs": obs[group].detach(),
    }
    row = {"label": phase.clone(), "clip": clip.clone()}
    for key, value in now.items():
      before = self._last.get(key)
      row["d_" + key] = (
        torch.zeros(value.shape[0], device=value.device)
        if before is None
        else (value - before).abs().amax(dim=-1)
      )
    self._last = {key: value.clone() for key, value in now.items()}
    first = (phase == 2) & (self.switch < 0)
    self.switch = torch.where(first, len(self.steps), self.switch)
    self.steps.append(row)

  def _window(self, env_id: int, key: str, low: int, high: int) -> list[float]:
    return [float(step[key][env_id]) for step in self.steps[low:high]]

  def ordinary(self, env_id: int, key: str) -> float:
    """The median size of this change over the steps either side of the switch.

    The yardstick, and it is local on purpose. Abrupt means out of line with its neighbours,
    so the switch step is judged against the steps around it rather than against a stretch
    of motion half a second away.

    Measuring against that stretch instead is what the first version did and it misreports
    two cases badly. A skill that enters something quiet, which the pass does, has a settled
    step a tenth the size of anything near the hand-over, so a switch that is smooth in
    context reads as many times an ordinary step. A skill that falls out of the hand-over is
    worse: it ends up motionless on the floor, the yardstick collapses towards zero, and the
    worse the failure the larger the number. Neither says anything about the seam.

    The switch step itself is left out of its own baseline, and both sides are included: the
    steps before it are the bridge finishing and the steps after are the entering skill
    starting, which are exactly the two things the switch has to be continuous between.
    """
    switch = int(self.switch[env_id])
    if switch < 0:
      return float("nan")
    span = self.NEIGHBOURHOOD
    values = sorted(
      self._window(env_id, key, max(switch - span, 0), switch)
      + self._window(env_id, key, switch + 1, switch + span + 1)
    )
    return values[len(values) // 2] if values else float("nan")

  def settled(self, env_id: int, key: str) -> float:
    """The same median over half a second of the entering skill, well clear of the switch.

    Context rather than a yardstick. Read beside `ordinary`: a settled figure far below the
    local one is a skill that has gone quiet, and far above is one that has run away.
    """
    switch = int(self.switch[env_id])
    if switch < 0:
      return float("nan")
    values = sorted(self._window(env_id, key, switch + 25, switch + 75))
    return values[len(values) // 2] if values else float("nan")

  def ratio(self, env_id: int, key: str = "d_action") -> float:
    """The switch step in units of a neighbouring step. One or less is invisible.

    The number every intervention here is judged on. An absolute joint target jump means
    nothing on its own: a kick throws a leg, and the same jump that is a snap during a
    run-up is unremarkable during the swing. Always read it with the absolute beside it,
    which is what `report` prints.
    """
    switch = int(self.switch[env_id])
    if switch < 0:
      return float("nan")
    normal = self.ordinary(env_id, key)
    return float(self.steps[switch][key][env_id]) / max(normal, 1e-6)

  def jump(self, env_id: int, key: str = "d_action") -> float:
    """The switch step's own change, in natural units. The absolute behind `ratio`.

    Worth reading on its own wherever the neighbourhood is not a fair baseline, which the
    observation channel is not: counting the reference in quiets the steps either side of
    the switch as well as the switch itself, so the ratio can rise while the jump it is
    made of falls by a factor of ten. For the action channel the neighbours are the two
    policies commanding ordinary motion and the ratio is the better number.
    """
    switch = int(self.switch[env_id])
    if switch < 0:
      return float("nan")
    return float(self.steps[switch][key][env_id])

  def report(self, env_id: int, span: int = 8) -> None:
    """The steps either side of the switch, against an ordinary step of the same run."""
    switch = int(self.switch[env_id])
    if switch < 0:
      print()
      print("no hand-over on the watched env, so there is no seam to read.")
      return
    names = self.names
    print()
    print(
      f"the seam on env {env_id}, one step per line. d_ columns are the change from the "
      "step before, worst channel of that step."
    )
    print()
    print(
      f"  {'step':>6}{'driver':>8}{'clip':>6}"
      f"{'d_action':>10}{'d_joint':>9}{'d_vel':>9}{'d_obs':>9}"
    )
    low = max(switch - span, 1)
    high = min(switch + span, len(self.steps))
    for index in range(low, high):
      step = self.steps[index]
      mark = "   <- control changes here" if index == switch else ""
      print(
        f"  {index:>6}{names[int(step['label'][env_id])]:>8}"
        f"{int(step['clip'][env_id]):>6}"
        f"{float(step['d_action'][env_id]):>10.3f}"
        f"{float(step['d_joint'][env_id]):>9.4f}"
        f"{float(step['d_vel'][env_id]):>9.3f}"
        f"{float(step['d_obs'][env_id]):>9.3f}{mark}"
      )
    print()
    print(
      f"  the switch step against the median of the {self.NEIGHBOURHOOD} steps either "
      f"side of it, and against half a second of settled {names[2]}:"
    )
    print(f"    {'':>7} {'switch':>8}   {'nearby':>7}          {'settled':>8}")
    for key in self.KEYS:
      normal = self.ordinary(env_id, "d_" + key)
      jump = float(self.steps[switch]["d_" + key][env_id])
      print(
        f"    {key:>7} {jump:8.3f} / {normal:7.3f} = {jump / max(normal, 1e-6):5.1f}x"
        f"   {self.settled(env_id, 'd_' + key):8.3f}"
      )


def crossfade(
  action: torch.Tensor,
  parting: torch.Tensor,
  steps: torch.Tensor,
  since: torch.Tensor,
) -> torch.Tensor:
  """Ramp out of the bridge's parting action over the first `steps` of the kick.

  The cure for an action seam that survives handing the entering skill a consistent world.
  Two policies that agree about the state can still disagree about what to do in it, and
  nothing about the hand-over can make them agree: they were fitted separately.

  What a crossfade does is spend the disagreement over several control steps instead of one.
  The joint targets are a continuous function of time across the switch, so the actuators
  see a slew rather than a step, and the size of what is left is the disagreement divided by
  the length of the ramp.

  What it costs is that for those steps neither policy is in control, so the ramp has to be
  short against the motion. Ten steps is a fifth of a second, which is most of the kick's
  backswing; four is where this stops being visible and the kick has not yet moved.

  Zero steps is a hard switch and the default this replaced.
  """
  weight = ((since.float() + 1.0) / steps.clamp(min=1.0)).clamp(0.0, 1.0)
  fading = (steps > 0) & (since >= 0) & (since < steps)
  mixed = weight.unsqueeze(-1) * action + (1.0 - weight.unsqueeze(-1)) * parting
  return torch.where(fading.unsqueeze(-1), mixed, action)


class Run:
  """Leaving skill, bridge, entering skill, then back to the leaving skill."""

  PHASES = ("leaving", "bridge", "entering")

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    couple: Couple,
    policies: dict[str, Policy],
    table: EntryTable,
    cfg: Config,
    bridges: dict[str, Policy] | None = None,
  ) -> None:
    self.env, self.couple, self.table, self.cfg = env, couple, table, cfg
    self.acts = (
      policies[couple.leaving.name],
      policies[BRIDGE_GROUP],
      policies[couple.entering.name],
    )
    self.names = (couple.leaving.name, BRIDGE_GROUP, couple.entering.name)
    self.bridges: dict[str, Policy] = {
      cfg.bridge: policies[BRIDGE_GROUP],
      **(bridges or {}),
    }
    """Every architecture loaded, by name. The dropdown picks one of these."""
    self.driving = cfg.bridge
    """Which of them crosses next. Read by `cross` and not before, so a swap made while a
    crossing is running shows one architecture's opening and another's follow through as
    if they were one hand-over, which compares nothing."""
    self.robot: Entity = env.scene[ROBOT]
    command = env.command_manager.get_term("bridge")
    assert isinstance(command, Aimed)
    self.command = command

    self.knobs: dict[str, dict[str, float]] = {
      actor.name: defaults(actor) for actor in (couple.leaving, couple.entering)
    }
    """What each skill is currently being told, by name. The panel writes here and
    `condition` reads it, so a control is one declaration in actors.py and nothing else."""
    if cfg.speed is not None and "forward" in self.knobs[couple.leaving.name]:
      self.knobs[couple.leaving.name]["forward"] = cfg.speed
    entering = self.knobs[couple.entering.name]
    for name, value in cfg.tell.items():
      if name not in entering:
        raise SystemExit(
          f"'{couple.entering.name}' has no control called '{name}'. It takes "
          f"{sorted(entering) or 'nothing'}."
        )
      entering[name] = value
    self.arrive = couple.entering.arrive
    """Where the entering skill needs the robot, or None for a skill with no object, which
    falls back on the ballistic placement. Resolved once so `triggered`, `cross` and the
    drawing cannot disagree about it."""
    self.entry = cfg.entry
    self.preview = cfg.preview and couple.entering.arrive is not None
    """Whether the entering skill's entry states are drawn while the walk is still driving.
    Off for good for a skill with no object, which has nowhere fixed to draw them."""
    self.order: tuple[Entry, ...] = table.of(couple.entering.name)
    """The states the slider walks, in the order the table holds and selector.view draws
    them. Fixed for the run, unlike `ranked`, so slider position N is the Nth robot in the
    viewer and stays that state while the robot moves."""
    self._entry_at, self._entry_state = "", torch.zeros(0)
    self._ranked: tuple[Reach, ...] = ()
    self._ranked_at = -1
    self.fire = self.done = False
    self.steer_walk = cfg.steer and couple.entering.arrive is not None
    """Whether to aim the leaving skill at the spot. Off for a couple with no spot to aim
    at, whatever the flag says."""
    self.smoothed = state(env.scene[ROBOT])[:, 7:9].clone()
    """Root ground velocity, low passed. The steering and the switch are both built out of
    it, and a walking root's instantaneous velocity swings through a gait cycle hard enough
    to make both of them oscillate at step frequency."""
    self.decay = float(torch.exp(torch.tensor(-env.step_dt / SMOOTH_S)))
    self.blend_steps = cfg.blend_steps
    """Steps of crossfade at each switch. Mutable where Config is not, so the panel can put
    it on a slider and the seam can be watched appearing and disappearing."""
    self.by_distance = couple.entering.arrive is not None and cfg.auto is None
    """Whether the switch fires on the metres left to the demanded spot rather than on a
    step count. On for a skill with an object, which is the only kind that has a spot; off
    when --auto named a step, since that is somebody asking for the old behaviour."""
    self.fire_at = cfg.fire_at
    self.window_s = (
      cfg.duration_s
      if cfg.duration_s is not None
      else (couple.duration_s if couple.duration_s is not None else cfg.window_s)
    )
    """Seconds the bridge is given when the switch fires on distance. Pinned rather than
    solved: the point of firing at a fixed place is that both ends of the crossing are
    stated rather than inferred."""
    self._gap = float("inf")
    """Metres to the demanded spot on the previous step, so the switch can tell a robot
    walking onto the spot from one that has already walked through it."""
    self.switch_at: int | None = cfg.auto
    """Control step the switch fires itself on, or None to wait for the button.

    Mutable where Config is not, so a viewer can put the hand-over instant on a slider.
    Comparing two skills on the same crossing needs it fixed, and a button cannot be
    pressed twice on the same step."""
    self.phase = self.tick = self.until = 0
    self.earned, self.scored = 0.0, 0
    """Discounted reward the entering skill has collected since it took over, and over how
    many steps. The hand-over verdict."""
    self.discount = float(
      getattr(load_rl_cfg(couple.entering.task).algorithm, "gamma", 0.99)  # ty: ignore[unresolved-attribute]
    )
    """The entering skill's own PPO discount, so the verdict and the skill agree about
    what a second of reward is worth."""
    self.scoring_steps = max(1, round(SCORING_WINDOW_S * self.command.fps))
    self.mass = (1.0 - self.discount**self.scoring_steps) / max(
      1.0 - self.discount, 1e-9
    )
    """Total discount weight in the scoring window. Divides `earned`, so a hand-over the skill
    falls out of forfeits what it did not collect rather than being rescaled back up."""
    self.fell = False
    self.parting = torch.zeros_like(env.action_manager.action)
    """The last action the actuators were given. What a crossfade ramps out of."""
    self.fading = 0
    """Blended steps still owed after the most recent switch. See `Config.blend_steps`."""
    self.seam = (
      Seam(env, env.num_envs, (couple.leaving.name, BRIDGE_GROUP, couple.entering.name))
      if cfg.seam
      else None
    )
    # The entering skill first, so whatever it keeps per episode exists before anything
    # runs: a reference has to be anchored somewhere or its observation reads a clip that
    # was never placed. `arena` freezes every skill's resampling, so what its reset drew
    # stands for the whole run. Its placement and its phase are both wrong until `aim` and
    # `resume` set them from where the robot will actually be
    self.enter(couple.entering)
    self.enter(couple.leaving)

    self.written_duration_s = (
      cfg.duration_s if cfg.duration_s is not None else couple.duration_s
    )
    """A duration named by hand, or None to let the entry and the geometry decide."""
    self.solved_duration_s: float | None = None
    """The window `triggered` last worked out would land the crossing where the entering
    skill wants it. Held so `cross` uses the same number the switch fired on."""

  def enter(self, actor: Actor, frame: int = 0) -> None:
    """Hand the world to a skill that is about to take over where the robot stands.

    Frame zero by default, because every caller here starts a skill rather than resumes one:
    the leaving skill at a reset, and the entering skill's goal before the run begins. `aim`
    is the one that resumes, and it passes the entry's own frame.

    The panel's current settings, not the declared defaults, so the entering skill's
    reference is placed for the jump that is actually going to be asked for. The two agree
    until somebody moves a slider.
    """
    if actor.enter:
      here = state(self.robot)
      actor.enter(
        self.env,
        here[:, 0:3],
        here[:, 3:7],
        frame,
        self.knobs.get(actor.name) or defaults(actor),
      )

  def ranked(self) -> tuple[Reach, ...]:
    """Every entry of the entering skill, in table order, with what each costs from here.

    Order is fixed, effort is not: the same entry is a different distance from a body
    mid-stride than from one standing. Cached against the control step, because `triggered`
    and `cross` both ask on the same one and must not get different answers.
    """
    if self._ranked_at != self.tick or not self._ranked:
      here = state(self.robot)[0].double().cpu().numpy()
      self._ranked = reaches(self.table, self.couple.entering.name, here, PROBE_S)
      self._ranked_at = self.tick
    return self._ranked

  def chosen(self) -> Reach:
    """The entry the next crossing aims at, and what reaching it would demand.

    Whichever row the slider is on. Nothing picks for you: which phase of a skill to enter
    is the thing this arena exists to let somebody look at.
    """
    return self.ranked()[min(max(self.entry, 0), len(self.order) - 1)]

  @property
  def duration_s(self) -> float:
    """How long the bridge gets, in seconds. Four sources, most specific first.

    Firing on distance wins, because that mode states both ends of the crossing by hand and
    a window solved from the geometry would be the other half deciding itself.

    A number typed on the command line or written into the couple wins, because someone
    asked for it. Otherwise the solved one, which is the window that puts the robot where
    the entering skill needs it. Otherwise the one the ranking implies, which is the only
    answer available when the skill has no opinion about where the robot should stand.
    """
    if self.by_distance:
      return self.window_s
    if self.written_duration_s is not None:
      return self.written_duration_s
    if self.solved_duration_s is not None:
      return self.solved_duration_s
    return self.reach_duration_s

  @property
  def reach_duration_s(self) -> float:
    """The default window, stretched if the chosen entry cannot be reached inside it.

    An effort of 1 is a change as fast as any recorded skill performed, and effort scales
    with 1/seconds, so PROBE_S * effort is the window at which this entry costs exactly
    that. Which makes it a floor and not a set point: crossing at the fastest rate the
    corpus ever produced is not a thing to ask for by default, and the corpus rates are
    per channel and far above the MAX_ACCEL a root sustains. Handing that number straight
    to the bridge asked a walk to shed 1.6 m/s in 0.31 s.

    So the longer of the floor and the default, clamped to what the bridge trained on. An
    entry whose floor is past the top of the range is one the slider should be walked past.
    """
    low, high = self.command.cfg.duration_s_range
    floor = PROBE_S * self.chosen().effort
    return float(min(max(floor, DEFAULT_DURATION_S, low), high))

  @property
  def entry_state(self) -> torch.Tensor:
    """The state the next crossing aims at. (1, 13 + 2J).

    Cached against the entry's name, because `triggered` reads this every control step and a
    table row is a numpy array on the wrong side of the device.
    """
    entry = self.chosen().entry
    if self._entry_at != entry.name:
      self._entry_at = entry.name
      self._entry_state = torch.as_tensor(
        entry.state[None], dtype=torch.float32, device=self.env.device
      )
    return self._entry_state

  def demanded(self) -> torch.Tensor | None:
    """Every entry of the entering skill, standing where its object demands. `(M, 13 + 2J)`,
    or None for a skill with no object.

    What the switch is being timed against, and the reason it can be drawn before the switch
    at all. A target placed ballistically is a guess about where this robot's momentum will
    carry it, so it moves with the robot every step and there is nothing to walk towards. An
    object does not move. `Actor.arrive` says where the robot has to stand for the skill to
    meet it, so evaluating that at each entry's own frame gives a line fixed to the floor,
    and walking the robot onto it is the whole of pressing the button at the right moment.

    Asked of `Actor.arrive` per entry rather than reconstructed from `EntryTable.trail`, and
    on a skill whose window is an approach the two are not the same line. The trail
    integrates the entries' own velocities and is a few centimetres out over half a second;
    `arrive` reads the clip, which is where the strike really is. Half a second is most of
    the box a ball skill was trained in, so the exact one is the one to walk towards.

    A frame it does not depend on is a line that collapses to a point, and that is correct
    too. The pass shoves from a stand and keeps a single early entry.

    The same call `aim` places the target with, so the target is one of these poses rather
    than something near them. That agreement is the point and it is why this is not a
    drawing helper: a line the operator times the switch against, and a target the switch
    then aims somewhere else, is worse than drawing nothing.
    """
    arrive = self.arrive
    if arrive is None:
      return None
    here = state(self.robot)
    return torch.cat(
      [
        entry_target(
          self.env,
          self.couple.entering,
          torch.as_tensor(entry.state[None], dtype=here.dtype, device=here.device),
          here,
          self.duration_s,
          entry.frame,
          arrive,
          self.knobs[self.couple.entering.name],
          entry,
        )
        for entry in self.order
      ],
      dim=0,
    )

  def triggered(self) -> bool:
    """Whether to start crossing on this step.

    Three ways in, and the entering skill's own precondition is the one that means something.
    A skill with an object was trained with it in a particular place, and the switch fires
    when walking has brought that place within reach. That is what a controller would be
    deciding, reduced to the one rule that makes the scenario run. The button and --auto are
    for skills with no such rule, and an override for those with one.

    What changed is which quantity gives way. The bridge now takes a duration, so a demand
    that used to be waited for can be solved for instead: `crossing_time` says how long a
    window would have to be to land the robot where the skill wants it, and the switch fires
    as soon as that is a window the bridge was trained on. So a ball skill fires early with
    a long window when the ball is far and late with a short one when it is close, rather
    than firing at the single distance a fixed window happened to match.

    The residual is what keeps that honest. A crossing travels along one line, so a duration
    can only fix how far, never which way, and a demand off that line stays off it. Firing on
    the duration alone would hand over on time to the wrong place.
    """
    entering = self.couple.entering
    here = state(self.robot)
    target = facing(self.entry_state, here)
    # A button press is still a hand-over that has to be given a window, so the solve below
    # runs first and this is read afterwards. Returning here on sight of the press would
    # hand `cross` whatever the previous step worked out, for a different chosen entry
    pressed = self.fire or self.tick == self.switch_at

    if self.arrive is not None:
      chosen = self.chosen().entry
      target = entry_target(
        self.env,
        entering,
        self.entry_state,
        here,
        self.duration_s,
        chosen.frame,
        self.arrive,
        self.knobs[entering.name],
        chosen,
      )
      want = target[:, :2]
      seconds, residual = crossing_time(here, target, want)
      low, high = self.command.cfg.duration_s_range
      if self.written_duration_s is None:
        # Clamped, because a skill fired by hand is fired whenever the button is pressed and
        # the window that would land the crossing exactly may be one the bridge never saw.
        # `verdict` says when the demand and the window have parted company
        # Clamped in floats and not on the tensor. float32's nearest value to the top of
        # the range widens to something above it, and `verdict` then reports the window it
        # was just handed as outside the range it came from
        self.solved_duration_s = min(max(float(seconds.min()), low), high)
      if pressed:
        return True
      if self.by_distance:
        return self.arrived_at(here, want, residual)
      # `ready` is read and not called, and for a skill with a demand that is all it can be.
      # A precondition asks whether the object will be in its box when control changes, and
      # `want` is by construction the pose that puts it there, so calling it here answers
      # its own question. What is left of the declaration is whether this skill fires itself
      # at all, and a skill that declares none has said the moment is the operator's
      if entering.ready is None:
        return False
      fits = (seconds >= low) & (seconds <= high)
      return bool((fits & (residual <= ARRIVE_SLACK)).all())

    if pressed:
      return True
    ready = entering.ready
    if ready is None:
      return False
    # Where the robot will be when control changes: the crossing aim is about to ask for,
    # plus whatever the bridge is known to overshoot it by. Asked of the same frame cross
    # will pick, because a target moving at 1 m/s is placed half a metre further out over a
    # one-second window than a standing one, and that half metre is the whole box the pass
    # was trained in.
    #
    # Over every window the bridge was trained on, not just one. A precondition is a box the
    # object has to be in when control changes, and the window decides how far the robot
    # travels before that happens, so a longer window reaches a box that a shorter one stops
    # short of. Asking at a single duration turns a range of moments when the hand-over would
    # work into the one moment that particular number happens to land in, and on the ball
    # skills that is the difference between striking and walking past.
    windows = self.candidate_windows()
    fits = [w for w in windows if bool(ready(self.env, self.at(here, target, w)).all())]
    if not fits:
      return False
    # The middle of what works, not the first. The edges of the range are where the
    # prediction is about to stop being true, and a hand-over aimed at one is a hand-over
    # that fails on the next step's rounding
    if self.written_duration_s is None:
      self.solved_duration_s = fits[len(fits) // 2]
    return True

  def arrived_at(
    self, here: torch.Tensor, want: torch.Tensor, residual: torch.Tensor
  ) -> bool:
    """Whether the robot has walked to within `fire_at` of the spot, on the line to it.

    Three conditions and each rules out a hand-over that happens somewhere else.

    Near enough, which is the one the operator sets. Closing, because the gap bottoms out
    and grows again once the robot walks through its own ball, and a threshold on distance
    alone fires a second time on the way out at a spot now behind the robot.

    And on the line, which is the one that was missing and the one that made the swing miss.
    A crossing travels along the mean of the two velocities, so a duration fixes how far it
    goes and never which way; `residual` is how far off that line the spot sits, and no
    window closes it. Firing anyway hands the bridge a target it has to reach sideways,
    which it does badly. Measured over five resets of walk2kick at entry 3, distance alone
    was bimodal, 0.14 m of standing error on three and 0.33 m on two, and the bad ones are
    the ones that fired while still off the line. `steer` is what brings the residual down;
    this is what waits for it.

    The metres `status` prints are the first of the three, measured to the same spot `aim`
    will place the target on, so what the viewer counts down and what fires agree.
    """
    gap = float((here[0, 0:2] - want[0]).norm())
    closing, self._gap = gap < self._gap, gap
    return closing and gap <= self.fire_at and float(residual[0]) <= self.cfg.slack

  def candidate_windows(self, count: int = 9) -> list[float]:
    """Durations to consider, across what the bridge was trained on.

    Only what it trained on. A window outside `duration_s_range` is a question the policy was
    never asked, and a trigger free to invent one would hand over on a prediction made by a
    formula rather than by anything the bridge has demonstrated.
    """
    if self.written_duration_s is not None:
      return [self.written_duration_s]
    low, high = self.command.cfg.duration_s_range
    return [low + (high - low) * i / (count - 1) for i in range(count)]

  def at(
    self, here: torch.Tensor, target: torch.Tensor, duration_s: float
  ) -> torch.Tensor:
    """Where the robot is predicted to be when a window of this length closes. (N, 3)."""
    out = here[:, 0:3].clone()
    out[:, 0:2] = crossing(here, target, duration_s, self.cfg.overshoot)
    return out

  def aim_walk(self) -> None:
    """Point the leaving skill so its crossing line runs through the spot.

    Written into the leaving skill's own `heading` control rather than into the command
    term, so the one place a skill's conditioning is applied stays `Actor.condition` and
    this is just another thing setting the same dial.

    Two terms. The first aims the body at the spot, which is the obvious half. The second is
    the angle between the line the crossing will actually travel along and the direction of
    the spot, and it is the half that matters: without it the residual sits between a tenth
    and three tenths of a metre for the whole approach and the swing misses by most of that.

    Fed back rather than solved, because solving it is a fixed point: turning the robot
    turns the clip that is anchored along its heading, which moves the target's velocity,
    which moves the line. A proportional term converges while the robot is still a second
    away from needing it.
    """
    if not self.steer_walk or self.arrive is None:
      return
    knobs = self.knobs[self.couple.leaving.name]
    if "heading" not in knobs:
      return
    here = state(self.robot)
    spot = self.arrive(self.env, self.chosen().entry.frame)
    target = facing(self.entry_state, here)
    gap = spot[:, 0:2] - here[:, 0:2]
    along = (self.smoothed + target[:, 7:9]) / 2.0
    cross = along[:, 0] * gap[:, 1] - along[:, 1] * gap[:, 0]
    lead = torch.atan2(cross, (along * gap).sum(dim=-1))
    knobs["heading"] = float(torch.atan2(gap[0, 1], gap[0, 0]) + lead[0])

  def condition(self) -> None:
    """Tell whichever skill owns the world right now what it is being asked for.

    Only the one that owns it. Walk and run read the same twist term, and a term written by
    both every step holds whoever wrote last, which is how a run bridged out of a walk used
    to be handed the walking speed it had just been bridged out of. Ownership passes at the
    moment the bridge is aimed rather than when the skill starts driving, because a skill
    with a goal has to have been told it before it takes over: the pass needs its launch
    velocity while the bridge is still crossing, not a step afterwards.
    """
    owner = self.couple.leaving if self.phase == 0 else self.couple.entering
    if owner.condition is not None:
      owner.condition(self.env, self.knobs[owner.name])

  @property
  def label(self) -> str:
    return self.PHASES[self.phase]

  @property
  def active(self) -> str:
    """Whoever is driving right now, by name: a skill's own, or the bridge's.

    What the viewer shows, and the phase name is not it. "entering" says a hand-over has
    happened and leaves you to remember which skill that was, which is the one thing
    somebody watching a transition is trying to read off the screen.

    The bridge says which architecture too, when more than one is loaded. Two of them
    crossing the same gap look alike enough that the dropdown's position is not something
    to be trusted to memory.
    """
    if self.phase == 1 and len(self.bridges) > 1:
      return f"{BRIDGE_GROUP} ({self.driving})"
    return self.names[self.phase]

  @property
  def status(self) -> str:
    """The viewer's line: who is driving, and how far off the mark the robot still is.

    The second half only while the leaving skill drives at an object, because that is the
    only moment anybody is deciding anything. The ghosts say where to be and this says how
    far away it is, which together are what the button is pressed on: a number closing on
    zero is a switch about to be worth firing, and one that has gone past it and started
    growing again is a robot that has walked through its own ball.
    """
    arrive = self.couple.entering.arrive
    if self.phase != 0 or not self.preview or arrive is None:
      return self.active
    entry = self.chosen().entry
    gap = float(
      (state(self.robot)[0, 0:2] - arrive(self.env, entry.frame)[0, 0:2]).norm()
    )
    if not self.by_distance:
      return f"{self.active}  {gap:.2f} m from '{entry.name}'"
    return (
      f"{self.active}  {gap:.2f} m from '{entry.name}', firing at {self.fire_at:.2f} m "
      f"into a {self.window_s:.2f} s window"
    )

  def reset(self) -> None:
    """Start over, because the world just did.

    The viewer's reset button resets the environment and then calls this. Without it the two
    disagree: the robot is back at its opening pose while whichever skill was driving when
    the button was pressed carries on, out of a hand-over that never happened. The clock
    goes back with the phase, the leaving skill gets a world it is standing at the start of,
    and the target stops being drawn because its crossing no longer exists.

    Deliberately not reset: the sliders. `forward`, `steps` and `frame` are what the person
    watching set them to, and a reset means run the same transition again, not undo their
    settings.
    """
    self.fire = self.done = False
    self.phase = self.tick = self.until = self.fading = 0
    self.earned, self.scored, self.fell = 0.0, 0, False
    self.parting.zero_()
    self._gap = float("inf")
    self.command.aimed = False
    self.command.trail = None
    self.enter(self.couple.leaving)

  @torch.no_grad()
  def __call__(self, obs):
    self.smoothed = (
      self.decay * self.smoothed
      + (1.0 - self.decay) * self.robot.data.root_link_lin_vel_w[:, 0:2]
    )
    if self.phase == 0:
      self.aim_walk()
    self.condition()

    if self.phase == 0 and self.triggered():
      obs = self.cross()
    elif self.phase == 1 and self.crossed:
      obs = self.hand_over()
    elif self.phase == 2:
      self.watch()
      if self.tick >= self.until:
        self.settle()

    # Refreshed every step the leaving skill drives, because the line is expressed in the
    # robot's heading frame and the robot is turning. From the switch on it is `cross`'s to
    # write: what the bridge is going to do stops being a question about the object
    if self.phase == 0:
      self.command.trail = self.demanded() if self.preview else None

    # The reference walks up to the entry frame while the bridge crosses, so that the
    # entering skill is handed one that has been approaching it rather than one that has to
    # be rewound underneath it
    if self.phase == 1 and self.cfg.count_in:
      self.count_in()

    self.tick += 1
    action = self.acts[self.phase](obs)
    action = self.faded(action)
    self.parting = action.clone()
    if self.seam is not None:
      clip = torch.zeros(self.env.num_envs, device=action.device)
      if "motion" in self.env.command_manager.active_terms:
        motion = self.env.command_manager.get_term("motion")
        if isinstance(motion, JumpCommand):
          clip = motion.time_steps
      phase = torch.full_like(clip, float(self.phase), dtype=torch.long)
      self.seam.record(action, obs, self.couple.entering.name, clip, phase)
    return action

  def faded(self, action: torch.Tensor) -> torch.Tensor:
    """Ramp out of the parting policy's last action over the steps a switch still owes.

    Applied at both switches, not only the hand-over. A transition has two of them and both
    are a change of driver, so both put a step into the joint targets; the one out of the
    leaving skill is the less visible of the pair only because the bridge is the more
    compliant of the two policies.

    See `Config.blend_steps` and `crossfade`, which this is the single env form of.
    """
    steps = self.blend_steps
    if self.fading <= 0 or steps <= 0:
      return action
    weight = float(steps - self.fading + 1) / float(steps)
    self.fading -= 1
    return weight * action + (1.0 - weight) * self.parting

  def count_in(self) -> None:
    """Hold the entering tracker's clip `left` frames short of its entry and march it in.

    `left` comes off the window the crossing was given rather than off a clock kept here,
    so a bridge that arrives early leaves the reference a few frames short and `resume`
    closes that instead of fifty. Clamped at zero, so a crossing running into its patience
    overrun finds the reference waiting at the entry frame rather than marching past it.

    Nothing to do for an entering skill with no reference.
    """
    if "motion" not in self.env.command_manager.active_terms:
      return
    motion = self.env.command_manager.get_term("motion")
    if not isinstance(motion, JumpCommand):
      return
    left = int((self.command.window_steps - self.command.step).clamp(min=0)[0])
    entry_resume.rewind(self.env, max(self.aimed_frame - left, 0))

  @property
  def crossed(self) -> bool:
    """Whether the bridge is finished, either by arriving or by running out of patience.

    The switch out of the bridge phase. It used to be the clock, which handed over at the
    same tick whether the crossing had worked or not; the bridge now says which, and
    `hand_over` prints it.
    """
    return bool(self.command.arrived_now.all() or self.command.out_of_patience.all())

  def watch(self) -> None:
    """Score the entering skill while it drives, on its own terms.

    The arena has no reward manager, since nothing here is trained, so the skill's own reward
    is computed on demand from the config it was trained with. Same terms, same weights, same
    discount, and the baseline it is divided by is measured the same way, by tests/handoff.py
    teleporting into the same entry. So the number here and the number there are comparable.

    Discounted rather than averaged. A plain mean over the window forgives an entry the skill
    stumbles out of and then recovers from, which is the one failure a hand-over is
    responsible for: the opening is the bridge's work and the recovery is the skill's.

    A fall stops the accumulation rather than ending the run. Nothing in this arena
    terminates, so the robot lies on the floor collecting whatever a prone robot collects,
    and counting that would flatter the transition.

    Read off projected gravity, not root height: a deep crouch and a fall reach the same
    height and only one is a failure, and a jump is full of deep crouches.

    Only the first `scoring_steps`, which is the window a trial was scored over. The skill
    keeps driving after that so it can be watched, but a ratio against the trials' baseline
    only means something over the trials' own horizon.
    """
    if self.fell or self.scored >= self.scoring_steps:
      return
    if float(self.robot.data.projected_gravity_b[0, 2]) > -0.7:
      self.fell = True
      return
    self.earned += self.discount**self.scored * float(self.env.reward_buf[0])
    self.scored += 1

  def settle(self) -> None:
    """The transition is over. Say whether it worked.

    The only number in this file that answers the question the whole pipeline is for. Score
    and arrival error say how close the bridge got; this says whether the skill it handed to
    could do its job from there.
    """
    self.phase, self.done = 0, True
    # Divided by the whole window's discount weight, not by the steps that were scored, so a
    # hand-over the skill falls out of forfeits everything after the fall the way an entry
    # did. Dividing by `scored` would rescale a short bad run back up to look like a full one
    earned = self.earned / self.mass
    baseline = self.cfg.baseline
    if baseline is None:
      # Nothing to be a share of, so say the number and refuse the verdict rather than
      # dividing by an invented one. tests/handoff.py is what measures a baseline
      print(
        f"  {self.couple.entering.name} {'fell' if self.fell else 'ran'}, discounted return "
        f"{earned:.3f}. No baseline given, so there is no pass or fail."
      )
      self.enter(self.couple.leaving)
      return
    performance = earned / max(baseline, 1e-6)
    passed = not self.fell and performance >= self.cfg.bar
    print(
      f"  {'PASS' if passed else 'FAIL'}: {self.couple.entering.name} "
      f"{'fell' if self.fell else 'ran'}, performance {performance:.2f} "
      f"against a bar of {self.cfg.bar:.2f}"
    )
    self.enter(self.couple.leaving)

  def cross(self):
    """Aim the bridge at one state off the entering skill's window, draw the rest of that
    window behind it, place the skill and start the clock."""
    self.fire, self.phase = False, 1
    self.fading = self.blend_steps
    # Whichever architecture the dropdown is on, read here and nowhere else. Index two is
    # left as it was found, because walk2kick_robust_test swaps the entering policy the
    # same way and the two must not write over each other
    self.acts = (self.acts[0], self.bridges[self.driving], self.acts[2])
    here = state(self.robot)
    chosen = self.chosen()
    self.left_from = here[:, 0:3].clone()
    # Kept for `resume`, which repeats this placement at the hand-over. The heading is the
    # one `aim` uses, the robot's at the switch, not the target's: a target carries the
    # entry's own pelvis twist and a clip is anchored along its direction of travel
    self.aimed_from = yaw_quat(here[:, 3:7]).clone()
    self.aimed_frame = chosen.entry.frame
    self.target = aim(
      self.env,
      self.command,
      self.couple.entering,
      self.entry_state,
      here,
      self.duration_s,
      chosen.entry.frame,
      self.arrive,
      self.knobs[self.couple.entering.name],
      recorded=chosen.entry,
      tolerances=self.cfg.tolerances.for_entry(chosen.entry.skill, chosen.entry.frame),
    )
    # Frozen where it already was, for a skill with an object. `aim` placed the target with
    # the same call at the same tick, so the chosen entry's ghost is the target rather than
    # near it, and the line the switch was timed against is the line it fired on. Without an
    # object there is no demand to freeze and the reconstructed trail is hung off whatever
    # the ballistic placement decided
    self.command.trail = (
      self.demanded()
      if self.arrive is not None
      else trail_states(
        self.table,
        self.couple.entering.name,
        chosen.entry.name,
        here,
        self.target,
      )
    )
    print(
      f"\ncross: to '{chosen.entry.name}', recorded at frame {chosen.entry.frame}, "
      f"placed {self.duration_s:.2f} s ahead, effort {chosen.effort:.2f} on "
      f"{chosen.binding}{self.verdict(here)}"
    )
    profile = self.command.window_tolerances[0].tolist()
    print(
      "  tolerances: "
      + ", ".join(
        f"{name}={value:g}" for name, value in zip(CHANNELS, profile, strict=True)
      )
    )
    return fresh_obs(self.env)

  def verdict(self, here: torch.Tensor) -> str:
    """Whether a body could cross this at all, in the time the placement assumed.

    One question now, not two. There used to be a second, whether the window was one the
    bridge had seen, and it no longer means anything: the bridge is given no duration, so
    there is no band of them it was trained inside. `duration_s_range` describes how far
    apart the corpus cuts a training pair and says nothing about what may be asked here.

    The acceleration is physics and still holds. The placement puts the target where a body
    changing velocity steadily would be after this long, and nothing stops the state the
    selector picked from being further from the robot's current one than a body can travel
    in that time.
    """
    change = float((self.target[0, 7:10] - here[0, 7:10]).norm())
    seconds = self.duration_s
    accel = change / seconds
    # How far there is to go, which with a distance fired switch is the other half of the
    # request and the half nothing else prints. `fire_at` and the window are independent
    # once the switch stops solving for one of them, so a crossing can be asked for half a
    # metre in a window that covers a fifth of it, and the only sign is the standing error
    # after the fact
    gap = float((self.target[0, 0:2] - here[0, 0:2]).norm())

    said = (
      f", {gap:.2f} m to cover, sheds {change:.1f} m/s in {seconds:.2f} s "
      f"({accel:.1f} m/s^2)"
    )
    if accel > MAX_ACCEL:
      return f"{said}: past the {MAX_ACCEL:.0f} m/s^2 a humanoid sustains"
    return said

  def resume(self) -> None:
    """Rewind the reference at its original placement and clear skill entry latches."""
    entering = self.couple.entering
    if entering.enter is None:
      return
    entry_resume.rewind(self.env, self.aimed_frame)
    position = self.target[:, :3]
    if "motion" in self.env.command_manager.active_terms:
      motion = self.env.command_manager.get_term("motion")
      assert isinstance(motion, JumpCommand)
      position = motion.body_pos_w[:, 0]
    entering.enter(
      self.env,
      position,
      self.aimed_from,
      self.aimed_frame,
      self.knobs[entering.name],
    )

  def hand_over(self):
    """Report the arrival, then let the entering skill drive.

    Report the live state against the requested profile and the fixed baseline.
    The command's score describes its best earlier state, not necessarily the handoff.
    """
    self.phase = 2
    self.fading = self.blend_steps
    self.resume()
    self.until = self.tick + self.cfg.entering_steps
    self.earned, self.scored, self.fell = 0.0, 0, False
    verdict = "arrived" if bool(self.command.arrived_now.all()) else "gave up"
    now, want = state(self.robot)[0], self.target[0]
    joints = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.robot.num_joints)
    errors = channel_errors(now.unsqueeze(0), want.unsqueeze(0), self.command.arms)
    profile = self.command.window_tolerances[:1]
    score = float(arrival_score(errors, profile)[0])
    fixed_score = float(arrival_score(errors, self.command.tolerances)[0])
    worst_ratio = float((errors / profile).amax())
    # How far the body really went, over how far it was asked to go. Config.overshoot is
    # this minus one, and this is the only way to measure it: the placement is a model of a
    # walking body decelerating, and the body is the truth
    went = float((now[:2] - self.left_from[0, :2]).norm())
    asked = float((want[:2] - self.left_from[0, :2]).norm())
    travelled = f"  travelled {went / asked:.2f}x" if asked > 1e-3 else ""
    print(
      f"  {verdict}: requested score {score:.3f}  fixed score {fixed_score:.3f}  "
      f"worst error/tolerance {worst_ratio:.2f}  "
      f"{float((now[:3] - want[:3]).norm()):.2f} m off  "
      f"speed {float(now[7:10].norm()):.2f} vs {float(want[7:10].norm()):.2f} m/s  "
      f"joints {float((now[joints] - want[joints]).abs().max()):.2f} rad" + travelled
    )

    # The number the arrival score cannot give, for a skill that says where it needs the
    # robot. `score` measures the gap to the target, and the whole question about an object
    # is whether the target was in the right place: a hand-over can score well against a
    # target put half a metre from where the ball needs it and be useless. Evaluated now,
    # so this is the real standing error the entering skill inherits
    arrive = self.couple.entering.arrive
    if arrive is not None:
      spot = arrive(self.env, self.aimed_frame)[0, 0:2]
      print(
        f"  standing {float((now[0:2] - spot).norm()):.3f} m from where "
        f"{self.couple.entering.name} wants the robot"
      )
    return fresh_obs(self.env)


def fresh_obs(env: ManagerBasedRlEnv):
  """The observation as of right now, not as of the last step.

  `ObservationManager.compute()` hands back a cached `_obs_buffer` whenever one exists and
  `update_history` is false, which is what keeps a second call inside one control step from
  double-pushing the delay buffers. It also means a caller that has just changed something
  the observation reads gets the value from before the change.

  Both callers here have. `cross` has just pointed the bridge at a target and opened its
  window, and `hand_over` has just handed the world to a skill that reads its own command
  terms. Without dropping the cache the first action after each is computed from the
  previous step's observation: the bridge's opening step aims at the target it had before
  `aim` moved it.

  Not `update_history=True`, which would recompute but also push another frame into every
  history and delay buffer, so the policy would see one control step counted twice.
  """
  env.observation_manager._obs_buffer = None
  return env.observation_manager.compute()


def panel(server, run: Run) -> None:
  """Every flag, as a slider. Showing a value and letting you move it are the same job.

  The skill folders build themselves out of `Actor.controls`, so a skill gains a control by
  declaring one next to itself in actors.py and nothing here changes. A skill that takes
  nothing gets no folder rather than an empty one: the punch combination is a single clip
  with no goal to aim, and a slider that silently does nothing is worse than no slider.
  """
  for actor, when in (
    (run.couple.leaving, "while it drives"),
    (run.couple.entering, "once it takes over"),
  ):
    if not actor.controls:
      continue
    with server.gui.add_folder(f"{actor.name}, {when}"):
      for knob in actor.controls:
        slider = server.gui.add_slider(
          knob.name,
          min=knob.low,
          max=knob.high,
          step=knob.step,
          initial_value=run.knobs[actor.name][knob.name],
          hint=knob.hint,
        )
        slider.on_update(
          lambda _, a=actor.name, k=knob.name, sl=slider: run.knobs[a].__setitem__(
            k, float(sl.value)
          )
        )

  with server.gui.add_folder(f"Hand over to {run.couple.entering.name}"):
    button = server.gui.add_button(run.couple.entering.name)
    button.on_click(lambda _: setattr(run, "fire", True))

    # Only when there is a choice to make. One architecture loaded is a dropdown with one
    # entry, which says nothing and implies the others are missing rather than untrained
    if len(run.bridges) > 1:
      which = server.gui.add_dropdown(
        "bridge",
        tuple(run.bridges),
        initial_value=run.driving,
        hint="Which architecture crosses. Applied at the next hand-over, since swapping "
        "one mid-crossing compares nothing. The Active line names whoever is driving.",
      )
      which.on_update(lambda _: setattr(run, "driving", str(which.value)))

    # Only for a skill with an object. Everything else has its entry states drawn at the
    # switch and nowhere to put them before it, so a checkbox there would be one that does
    # nothing until the moment it stops mattering
    if run.couple.entering.arrive is not None:
      entries = server.gui.add_checkbox(
        "show the entry states",
        initial_value=run.preview,
        hint="Draw where the entering skill needs the robot for each of its entry states, "
        "before the switch. They stand still, because the object does: walk onto them and "
        "press the button.",
      )
      entries.on_update(lambda _: setattr(run, "preview", bool(entries.value)))

    entry = server.gui.add_slider(
      "entry",
      min=0,
      max=max(len(run.order) - 1, 1),
      step=1,
      initial_value=run.entry,
      hint="Which state to aim at, in the order selector.view draws them.",
    )
    entry.on_update(lambda _: setattr(run, "entry", int(entry.value)))

    # How much of the disagreement between the two policies reaches the actuators in one
    # step. Live, because the seam it removes is a single control step and the only way to
    # believe a number about it is to put it back and watch
    fade = server.gui.add_slider(
      "crossfade steps",
      min=0,
      max=24,
      step=1,
      initial_value=run.blend_steps,
      hint="Control steps to ramp out of the parting policy's last action at each switch. "
      "Zero is a hard switch and is what makes the hand-over visible. Applied at the next "
      "switch, not the one running.",
    )
    fade.on_update(lambda _: setattr(run, "blend_steps", int(fade.value)))

    if run.couple.entering.arrive is not None:
      # A skill with an object is entered at a place, so the switch is a distance and not a
      # step. See Config.fire_at. The step controls are not offered beside these: two ways
      # to say when, one of which silently wins, is worse than either
      distance = server.gui.add_slider(
        "fire at, m",
        min=0.0,
        max=2.0,
        step=0.05,
        initial_value=run.fire_at,
        hint="Metres from where the entering skill needs the robot, at which the switch "
        "fires. The same number the line above counts down.",
      )
      distance.on_update(lambda _: setattr(run, "fire_at", float(distance.value)))

      low, high = run.command.cfg.duration_s_range
      window = server.gui.add_slider(
        "window, s",
        min=low,
        max=high,
        step=0.05,
        initial_value=min(max(run.window_s, low), high),
        hint="Seconds the bridge is given. Pinned rather than solved, because the point of "
        "firing at a fixed place is that both ends of the crossing are stated.",
      )
      window.on_update(lambda _: setattr(run, "window_s", float(window.value)))

      walked = server.gui.add_checkbox(
        "fire on the distance above",
        initial_value=run.by_distance,
        hint="On, the same crossing repeats every episode: same entry, same metres out, "
        "same window, whatever the approach did. Off hands the moment back to the button.",
      )
      walked.on_update(lambda _: setattr(run, "by_distance", bool(walked.value)))

      aimed = server.gui.add_checkbox(
        "steer the walk at the spot",
        initial_value=run.steer_walk,
        hint="On, the walk's heading is written every step so the crossing line runs "
        "through where the entering skill needs the robot, and the heading slider does "
        "nothing. Off hands the heading back. This is what decides whether the swing "
        "meets the ball.",
      )
      aimed.on_update(lambda _: setattr(run, "steer_walk", bool(aimed.value)))
    else:
      # The other half of repeating a crossing, for a skill with nothing on the floor. The
      # entry fixes the state being aimed at and this fixes the state being aimed from,
      # which is what a hand pressed button cannot do: it lands on a different stride every
      # time. There is no spot to walk onto here, so a step is the only thing left to pin
      step = server.gui.add_slider(
        "switch step",
        min=0,
        max=max(SWITCH_STEP_MAX, run.switch_at or 0),
        step=1,
        initial_value=run.switch_at
        if run.switch_at is not None
        else DEFAULT_SWITCH_STEP,
        hint="Control step the hand-over fires on, counted from the reset. Only read while "
        "the checkbox below is on.",
      )
      pinned = server.gui.add_checkbox(
        "fire on the step above",
        initial_value=run.switch_at is not None,
        hint="On repeats the same crossing every episode, so two runs differ in nothing "
        "but what is being compared. Off hands the moment back to the button.",
      )

      def instant(_) -> None:
        run.switch_at = int(step.value) if pinned.value else None

      step.on_update(instant)
      pinned.on_update(instant)

    # No duration slider. The bridge is not given one: it reads the gap to its target and
    # the best it has managed so far, and it is paid for the best moment of the window
    # whenever that happens, so there is no window length to set for it. What a duration
    # still decides is where a target with nothing to fix it gets placed, and how long an
    # unsuccessful crossing is left running. Neither is a dial worth putting in front of
    # somebody watching a hand-over: one is geometry and the other is a give-up timer.
    # `Config.duration_s` is still there for a run that wants to pin it.


def main(couple: Couple, extra: Callable[[Any, "Run"], None] | None = None) -> None:
  """Run one transition. `extra` adds controls to the panel, for a variant of this arena."""
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  config_cls = config_for(couple)
  cfg = tyro.cli(
    config_cls,
    default=replace(
      config_cls(),
      duration_s=couple.duration_s,
      overshoot=couple.overshoot,
    ),
    config=mjlab.TYRO_FLAGS,
  )
  if (
    cfg.viewer == "none"
    and cfg.auto is None
    and couple.entering.ready is None
    and couple.entering.arrive is None
  ):
    raise SystemExit(
      f"--viewer none has nothing to press the button, and {couple.entering.name} has "
      f"neither a precondition nor a spot to walk onto: pass --auto N."
    )

  if cfg.bridge not in cfg.bridges:
    raise SystemExit(
      f"--bridge {cfg.bridge} is not one of --bridges {cfg.bridges}. The one that drives "
      f"has to be among the ones offered."
    )

  # First, because an architecture that has no code behind it is the cheapest thing to be
  # wrong about and the message says so plainly, and because the arena needs to know which
  # architectures it is hosting before it is built
  offered = offered_bridges(cfg.bridges, cfg.bridge)
  spec = offered[0][0]

  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Before the simulation and the three checkpoints, because this is the one thing a couple
  # can be missing and building the arena first means waiting a minute to be told a file is
  # not there
  table = EntryTable.load(cfg.table or TABLE_PATH)
  for line in table.lines(couple.entering.name):
    print(line)

  others = tuple(spec for spec, _ in offered[1:])
  env = ManagerBasedRlEnv(cfg=arena(couple, spec, others), device=device)
  bridge = Actor(BRIDGE_GROUP, spec.task_id)
  policies: dict[str, Policy] = {}
  for actor in (couple.leaving, couple.entering, bridge):
    explicit = (
      cfg.bridge_checkpoint if actor is bridge else getattr(cfg, checkpoint_flag(actor))
    )
    checkpoint = find_checkpoint(load_rl_cfg(actor.task).experiment_name, explicit)
    print(f"{actor.name:8s} {checkpoint}")
    policies[actor.name] = Policy(actor.task, checkpoint, env, actor.name, device)

  # The rest of the dropdown. Each reads its own observation group, so the one thing that
  # changes when the dropdown moves is the policy. A checkpoint that will not load is
  # dropped here rather than at the moment somebody picks it, since an architecture whose
  # network changed under its logs is the same normal state as one nobody has trained
  alternatives: dict[str, Policy] = {}
  for other, checkpoint in offered[1:]:
    print(f"{other.kind:8s} {checkpoint}")
    try:
      alternatives[other.kind] = Policy(
        other.task_id, checkpoint, env, bridge_group(other.kind), device
      )
    except Exception as broken:  # noqa: BLE001
      print(f"[bridge] {other.kind}: not offered, {type(broken).__name__}: {broken}")

  env.reset()
  run = Run(env, couple, policies, table, cfg, alternatives)
  print(f"{len(run.order)} states on the slider:")
  for index, entry in enumerate(run.order):
    print(f"  {index:2d}. {entry.name:<8} {entry.why}")
  for line in entry_lines(run.ranked()):
    print(line)
  print(
    f"{couple.entering.name}: aiming at '{run.chosen().entry.name}', "
    f"target placed {run.duration_s:.2f} s ahead"
  )

  if cfg.viewer == "none":
    obs = env.get_observations()
    for _ in range(cfg.patience):
      if run.done:
        break
      obs, _, _, _, _ = env.step(run(obs))
    else:
      # A trigger that never fires is a result, not a hang. The robot walked past its
      # object or never lined up with it, and sitting here forever hides that
      print(f"\ngave up after {cfg.patience} steps in the '{run.label}' phase")
    if run.seam is not None:
      run.seam.report(0)
    env.close()
    return

  import viser

  from mjlab.viewer import ViserPlayViewer

  server = viser.ViserServer(label=f"{couple.leaving.name}2{couple.entering.name}")
  panel(server, run)
  if extra is not None:
    extra(server, run)
  wrapped = RslRlVecEnvWrapper(
    env, clip_actions=load_rl_cfg(couple.entering.task).clip_actions
  )
  ViserPlayViewer(
    wrapped,
    run,
    viser_server=server,
    info_provider=lambda _: run.status,
    record_name=f"{couple.leaving.name}2{couple.entering.name}",
  ).run()
  if run.seam is not None:
    run.seam.report(0)
  wrapped.close()
