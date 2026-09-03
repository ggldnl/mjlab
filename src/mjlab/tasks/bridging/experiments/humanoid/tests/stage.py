"""Drive one skill, fire a switch, let the bridge hand over to the next.

A transition script names two actors and what is on the floor. Everything else is here.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2jump
    uv run python -m ...transitions.walk2pass
    uv run python -m ...transitions.walk2push

    # headless, firing the switch on step 120 instead of waiting for a button
    uv run python -m ...transitions.walk2jump --viewer none --auto 120

Three phases:

    leaving    the first skill drives, steered from the panel
    bridge     the bridge drives for `steps` steps, toward one target state
    entering   the second skill drives, from wherever the bridge left the robot

The viewer's Active line names whoever is driving, a skill by its own name or "bridge".
From the moment a target is chosen a translucent robot stands in it and stays there after
the hand-over, so the gap to the real robot is the arrival error. Scene > Debug Viz >
Bridge turns it off.

The target is a single state: a pose, a height, a tilt, joint angles and every velocity at
one instant, which is what the bridge trains on. It comes off the entering skill's table in
the selector package, and the `entry` slider walks it.

Which makes the hand-over measurable rather than plausible. After the bridge has crossed,
the entering skill drives for `entering_steps` and is scored on its own reward terms under
its own discount, so a hand-over that lands out of phase reads as a low number rather than
as a good number with a bad episode after it.

##
# What the panel drives
##

Each skill's folder is built from what that skill declares it can be told, so walking gets a
forward speed, a sideways speed and a heading, the strike skills get a ball speed and an
aim, and the punch combination gets no folder at all because it is one clip with nothing to
aim. A skill's conditioning is applied only while that skill owns the world, which is what
lets the walk and the run share one velocity term without writing over each other.

The window has its own control and a way to let go of it. Released, it comes from the entry
the selector hands back, or is solved so the crossing lands where the entering skill needs
the robot. That is new, and it is the difference the bridge's duration argument makes: a
switch that could only pick *when* had to wait for the world to drift into agreement with a
fixed window, and one that can pick *how long* as well fires as soon as the demand is
reachable by some window the bridge was trained on. The kick fires early with a long window
when the ball is far and late with a short one when it is close.

Entry states come from the measured table in the selector package. Look at them, and
measure what a hand-over is worth, before spending a bridge on it:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.handoff

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
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable, Literal, NamedTuple

import mujoco
import numpy as np
import torch
import tyro

from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridge import BRIDGE_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.bridge.env_cfg import (
  bridge_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp import (
  ROOT_STATE_DIM,
  BridgeCommand,
  BridgeCommandCfg,
  arrival_score,
  channel_errors,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import Selector, Shortlist
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul, yaw_quat
from mjlab.viewer.debug_visualizer import DebugVisualizer

ROBOT = "robot"
BRIDGE_GROUP = "bridge"

ARRIVE_SLACK = 0.10
"""How far apart, in metres, the predicted crossing and the spot an entering skill demands
may be when the switch fires. Only used when `Config.commanded` is on.

Sized to the error being corrected rather than to what the bridge tolerates, and the first
attempt got that backwards. `_draw_target` scatters its training targets around the
ballistic midpoint by up to `travel_speed * horizon`, so firing the moment the demanded spot
was anywhere inside that scatter looked defensible. Over a 0.6 s window it is 0.72 m, and
the bridge was told to cover most of a metre while also arriving nearly stopped:

    slack 0.72 m    0.40 m off the target, travelled 0.59x, score 0.146
    slack 0.10 m    0.08 m off the target, travelled 0.83x, score 0.554

Ten centimetres leaves the ballistic model deciding when, within a stride, and the object
deciding only where."""

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

  enter: Callable[[ManagerBasedRlEnv, torch.Tensor, torch.Tensor], None] | None = None
  """Put this skill's reference, and whatever else it keeps per episode, where it belongs.

  Called as `enter(env, pos, heading)`: the world position the robot will be at, and the
  direction it should head. A skill with no reference and no episode state of its own does
  not need one.

  Placement only. What state the robot has to arrive in comes off the shortlist: a state
  this skill was measured starting from, and measured opening well from. A reference is not
  one, and a retargeted clip is not one by several centimetres of floor.

  Whatever the skill keeps per episode is set to what a fresh episode would hold, because
  that is the condition the shortlist was measured under. A clip starts at its first frame,
  not part way in.

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
  because by the time the bridge has crossed the robot has walked on."""

  place: EventTermCfg | None = None
  """Where this skill's object starts, as a reset event. None for a skill without one."""

  robot: Callable[[EntityCfg], EntityCfg] | None = None
  """Patch the robot this skill needs, if it needs one patched.

  The kick adds two sites to the striking foot and reads its observation off them, so a robot
  without them cannot serve that policy: the arena used to keep the bridge's plain robot and
  the kick's own observation raised on the site it could not find. Declared here rather than
  detected, because detecting it means comparing two spec builders for equality and getting
  that wrong silently produces a robot with the wrong body on it.

  Patches from both skills of a couple compose, in the order leaving then entering.
  """

  arrive: Callable[[ManagerBasedRlEnv], torch.Tensor] | None = None
  """Where the robot has to end up, in world coordinates, for this skill's object to be in
  the box it was trained with. `(N, 3)`, and only the ground plane is read.

  Two jobs, and only one of them worked.

  The one that did is the diagnostic. Every hand-over into a skill that declares this prints
  how far the robot ended up from the spot the skill actually wants, which is the question
  an object skill is asking and the one the arrival score cannot answer: a hand-over can
  score well against a target placed half a metre from where the ball needs it. That line
  now prints whichever placement is in use.

  The one that did not is placing the target. The idea was to stop predicting where the
  robot would drift to and command the spot outright, on the reasoning that the bridge is
  the accurate part of this pipeline. It is, measurably: aimed at the ballistic midpoint it
  lands 0.03 m from its target and travels 0.96x the distance it was given. But commanding
  the arrival measured worse on the only number that matters:

      walk2pass, ballistic placement    0.083 m from where the pass wants the robot
      walk2pass, commanded placement    0.104 m

  Because the two errors trade off exactly. Aimed at its natural stopping point the bridge
  tracks to 0.03 m and the target itself is the thing 0.08 m out of place. Sent up to
  ARRIVE_SLACK off that point it hits the right target, and its own tracking error grows to
  0.08 m. Nothing is gained, because both errors are three to eight centimetres and the box
  the pass was trained in is eight centimetres deep. The bottleneck is the box, not the
  placement, and no way of aiming the bridge fixes a tolerance tighter than the accuracy of
  everything upstream of it.

  So `Config.commanded` defaults off and this stays declared, for the diagnostic and because
  the trade is not fixed. Widen the skill's box, lengthen the window, or improve the
  bridge's tracking and it tips the other way: the commanded placement has no systematic
  error to speak of, only the bridge's, while the ballistic one is a model of a walking gait
  that will always be a model."""


@dataclass(frozen=True)
class Couple:
  """Two skills and the hand-over between them. Everything else each skill brings itself."""

  leaving: Actor
  entering: Actor

  duration_s: float | None = None
  """How long the bridge gets, in physical seconds, or None to take it from the entry itself.

  None is the right default now that an entry carries its own duration: a crouch and a stand
  are not equally far from wherever a walk leaves the robot, and a selector that hands back a
  state without a time has only answered half the question. Set it per couple to override,
  which is a real decision worth making by hand when one pair sheds more momentum than the
  entry's own figure assumed."""

  overshoot: float = 0.0
  """This couple's measured overshoot. See `Config.overshoot` for what it means and how to
  read it off a run. Here because it is a property of one pair of skills at one window: how
  far the bridge carries a robot leaving *this* skill past a target of *that* one."""


##
# The arena.
##


TARGET_COLOR = (1.0, 0.72, 0.2, 0.45)
"""The target ghost. Warm, so it reads as a goal rather than a second robot."""


class Aimed(BridgeCommand):
  """The bridge command with window drawing switched off, and its target drawn.

  In training a window is drawn on reset and the robot is teleported onto its start frame.
  Here the robot is wherever the leaving skill left it, which is the whole point, so the
  target arrives from outside through `aim` and nothing is ever teleported.
  """

  def __init__(self, cfg: BridgeCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.aimed = False
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

  def open_window(self, env_ids: torch.Tensor, duration_s: torch.Tensor) -> None:
    """Start the clock on a target that was placed from outside.

    The base class does everything except note when the window opened, which it has no need
    of: in training a window is an episode and the environment's own counter is the clock.
    See `step` for why that is not true here.
    """
    super().open_window(env_ids, duration_s)
    self._opened[env_ids] = self._env.episode_length_buf[env_ids]

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """The state the bridge is crossing to.

    A pose and nothing else, because a pose is all that can be drawn: half of a target is
    velocity and a still body says nothing about that. It shows where `aim` decided a body
    carrying this momentum could be after `steps`, and it stays put after the hand-over so
    the gap to the robot is the arrival error left standing.

    Nothing is drawn before the first `aim`. `target` opens as an identity quaternion at the
    world origin, and a robot lying in the floor at the corner of the scene is not a target.
    """
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
    ghost = self._ghosts.get(color)
    if ghost is None:
      ghost = self._ghosts[color] = self._tinted(color)

    indexing = self.robot.indexing
    free = indexing.free_joint_q_adr.cpu().numpy()
    joints = indexing.joint_q_adr.cpu().numpy()
    for batch in visualizer.get_env_indices(self.num_envs):
      # From qpos0 rather than zeros: everything in this arena that is not the robot keeps
      # its own default, and a zero quaternion is not a rotation
      qpos = np.array(self._env.sim.mj_model.qpos0, dtype=np.float64)
      row = states[batch].cpu().numpy()
      qpos[free[0:3]] = row[0:3]
      qpos[free[3:7]] = row[3:7]
      qpos[joints] = row[ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
      # alpha and not the colour's own: the viewer takes an rgba geom's opacity from this
      # argument and only its hue from the model
      visualizer.add_ghost_mesh(
        qpos, model=ghost, alpha=color[3], label=f"{tag}_{batch}"
      )

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
  def build(self, env: ManagerBasedRlEnv) -> Aimed:
    return Aimed(self, env)


def arena(couple: Couple) -> ManagerBasedRlEnvCfg:
  """The bridge's own environment, with both skills' machinery merged into it.

  Built on the bridge's play config rather than a skill's, so the robot, the terrain and the
  bridge's observation are exactly what it trained against. Each skill then contributes what
  it needs and nothing else: its entities, its commands, the sensors its observation reads,
  and the observation itself.

  The observation is copied from the skill's own task verbatim. A checkpoint is tied to its
  term list in order, and feeding it the same numbers shuffled does not fail, it acts on
  nonsense. So the order comes from the one place that cannot disagree with the checkpoint,
  rather than being restated here where it would go stale.
  """
  cfg = bridge_env_cfg(play=True)

  trained = cfg.commands["bridge"]
  assert isinstance(trained, BridgeCommandCfg)
  aimed = {f.name: getattr(trained, f.name) for f in fields(trained)}
  # Draws the target ghost, and gets the viewer to offer a Bridge checkbox under
  # Scene > Debug Viz that turns it off again
  aimed["debug_vis"] = True
  # No corpus. No window is ever drawn in this arena, since `Aimed` takes its target from
  # outside, so the bridge's training dataset was being read for a frame rate the
  # environment already knows. It also means a transition can be staged before the corpus
  # has been built, which is the order the work actually happens in
  aimed["dataset_path"] = None
  cfg.commands["bridge"] = AimedCfg(**aimed)
  cfg.observations = {BRIDGE_GROUP: cfg.observations["actor"]}

  # Nothing should reset: a fall is the result of the test, not an error to recover from,
  # and a reset mid-run would move the robot out from under the phase machine
  cfg.terminations = {}

  # The entering skill's own reward, carried over so the hand-over can be scored the way the
  # selector scored an entry. Nothing is trained here and nothing reads this to act; it is
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
    # And the sensors those observations read. The kick watches a foot-ball contact, the
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
  """One state off a shortlist, turned to face the way the robot faces now.

  Every skill here is egocentric, so a state it can start from facing one way is one it can
  start from facing another. The turn has to reach the orientation and both velocity vectors,
  or the pose and the momentum disagree about which way the body is going.

  Ground position is not touched, because a state off the shortlist carries whichever tile
  it was recorded on. The caller places it.
  """
  turn = quat_mul(yaw_quat(here[:, 3:7]), quat_conjugate(yaw_quat(entry[:, 3:7])))
  return turn_state(entry, turn)


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


def aim(
  env: ManagerBasedRlEnv,
  command: Aimed,
  entering: Actor,
  entry: torch.Tensor,
  here: torch.Tensor,
  duration_s: float,
  arrive: Callable[[ManagerBasedRlEnv], torch.Tensor] | None = None,
) -> torch.Tensor:
  """Point the bridge at one state off the entering skill's shortlist, moved to meet the
  robot.

  The target is the recorded state and nothing else. Every skill here is egocentric: it
  reads its own body frame, and a tracker reads poses relative to an anchor it carries. So a
  state produced facing one way is a state the skill can be in facing another, and where in
  the world that happens does not enter into it. The move has to preserve everything the
  pose says relative to the direction of travel: pelvis twist, roll and pitch, joint angles,
  velocities. One yaw rotation, by the difference between the heading the rollout was
  recorded in and the heading the robot has now, does exactly that. The height is recorded,
  not chosen.

  The state comes from the selector, not from the entering skill's reference. For a clip
  tracker that reference is a retargeted human motion, which no robot is ever in. At frame 90
  of the jump, clip against policy:

      foot height     3.8 cm into the floor
      root height     1.4 cm below where the policy holds it
      joints          up to 0.42 rad apart
      descent rate    0.28 m/s against the policy's 0.15

  So the bridge was scored on arriving in a state its own entering skill never occupies.

  `Actor.enter` is still called, still with where the robot is meant to be rather than where
  it is, because a clip tracker has to be told where its clip goes. It just no longer says
  what the target is. The clip lands under the arrival, so the skill takes over from a robot
  sitting on its reference rather than the few centimetres off it the rollout was.

  One placement, not the two this used to need. The rotation alone settles the velocity to
  arrive with, so the centre of the disc, where training put every target, is known before
  anything is placed.
  """
  heading = yaw_quat(here[:, 3:7])
  horizon = duration_s

  target = facing(entry, here)
  # Commanded if the caller says where the skill needs the robot, predicted otherwise.
  # See Actor.arrive
  if arrive is not None:
    target[:, 0:2] = arrive(env)[:, 0:2]
  else:
    target[:, 0:2] = crossing(here, target, horizon)

  if entering.enter is not None:
    entering.enter(env, target[:, 0:3], heading)

  env_ids = torch.arange(command.num_envs, device=command.device)
  command.target[:] = target
  command.aimed = True
  # Seconds, not ticks. The command converts, and it is the only thing that should
  command.open_window(
    env_ids, torch.full((command.num_envs,), duration_s, device=command.device)
  )
  return target


##
# The run.
##


@dataclass(frozen=True)
class Config:
  duration_s: float | None = None
  """How long the bridge gets, in seconds, or None to take it from the entry. It trained on
  the configured duration range, so an override outside that range is being asked for
  something it never saw, which `verdict` says out loud."""

  entry: int = 0
  """Which state off the entering skill's shortlist to aim at. 0 is the first in the table.

  The list is ordered by a property of the entering skill alone, so walking down it is a
  search for a state the bridge can reach from where the leaving skill left the robot, not a
  search for one the skill opens well from. Only the first question is open here: every entry
  in the table is claimed to be a good opening, and the transition is what tests the claim."""

  entering_steps: int = 220
  """Control steps the entering skill drives for before the verdict is printed. Long enough
  that a skill which is going to fail from a bad entry has done so."""

  speed: float | None = None
  """Forward command for the leaving skill, in m/s, or None for whatever its own control
  declares. Only meaningful for a skill that has a `forward` control."""

  commanded: bool = False
  """Whether an entering skill's `Actor.arrive` places the target, instead of the ballistic
  model. Off because it measures worse: see `Actor.arrive` for the numbers and why.

  Kept because the comparison is worth being able to rerun, and because which of the two
  wins is a property of the current tolerances rather than of the idea."""

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

  auto: int | None = None
  """Fire the switch after N steps regardless of the world. Needed headless for a skill with
  no precondition of its own, and an override for one that has."""

  patience: int = 900
  """Steps a headless run gets before it gives up waiting for the switch to fire."""

  checkpoint: Path | None = None
  """An explicit bridge checkpoint. The skills always come from their own logs."""

  viewer: Literal["viser", "none"] = "viser"
  device: str | None = None
  seed: int = 0


class Run:
  """Leaving skill, bridge, entering skill, then back to the leaving skill."""

  PHASES = ("leaving", "bridge", "entering")

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    couple: Couple,
    policies: dict[str, Policy],
    selector: Selector,
    cfg: Config,
  ) -> None:
    self.env, self.couple, self.selector, self.cfg = env, couple, selector, cfg
    self.acts = (
      policies[couple.leaving.name],
      policies[BRIDGE_GROUP],
      policies[couple.entering.name],
    )
    self.names = (couple.leaving.name, BRIDGE_GROUP, couple.entering.name)
    self.robot: Entity = env.scene[ROBOT]
    command = env.command_manager.get_term("bridge")
    assert isinstance(command, Aimed)
    self.command = command

    self.knobs: dict[str, dict[str, float]] = {
      actor.name: {knob.name: knob.initial for knob in actor.controls}
      for actor in (couple.leaving, couple.entering)
    }
    """What each skill is currently being told, by name. The panel writes here and
    `condition` reads it, so a control is one declaration in actors.py and nothing else."""
    if cfg.speed is not None and "forward" in self.knobs[couple.leaving.name]:
      self.knobs[couple.leaving.name]["forward"] = cfg.speed
    self.arrive = couple.entering.arrive if cfg.commanded else None
    """Where the entering skill needs the robot, or None to fall back on the ballistic
    placement. Resolved once so `triggered` and `cross` cannot disagree about it."""
    self.entry = cfg.entry
    self._entry_at, self._entry_state = -1, torch.zeros(0)
    self.fire = self.done = False
    self.phase = self.tick = self.until = 0
    self.earned, self.scored = 0.0, 0
    """Discounted reward the entering skill has collected since it took over, and over how
    many steps. The hand-over verdict."""
    self.scoring_steps = selector.scoring_steps(self.command.fps)
    self.mass = (1.0 - selector.discount**self.scoring_steps) / max(
      1.0 - selector.discount, 1e-9
    )
    """Total discount weight in the scoring window. Divides `earned`, so a hand-over the skill
    falls out of forfeits what it did not collect rather than being rescaled back up."""
    self.fell = False
    # The entering skill first, and only for its goal. `arena` freezes every skill's
    # resampling, so whatever its reset drew would otherwise stand for the whole run, and
    # the states worth aiming at depend on which goal it is: the openings for a 1.55 m jump
    # are not the openings for a short one. Its placement is wrong until `aim` calls this
    # again with where the robot will actually be, and that is the only part that moves
    self.enter(couple.entering)
    self.enter(couple.leaving)

    self.shortlist: Shortlist = selector.shortlist(self.env)
    """The entering skill's entry states, in table order. Built here rather than at load time
    because turning a written posture into a state needs the robot: its joint names, its
    defaults, and forward kinematics to stand the posture on the floor."""
    self.written_duration_s = (
      cfg.duration_s if cfg.duration_s is not None else couple.duration_s
    )
    """A duration named by hand, or None to let the entry and the geometry decide."""
    self.solved_duration_s: float | None = None
    """The window `triggered` last worked out would land the crossing where the entering
    skill wants it. Held so `cross` uses the same number the switch fired on."""

  def enter(self, actor: Actor) -> None:
    """Hand the world to a skill that is about to take over where the robot stands."""
    if actor.enter:
      here = state(self.robot)
      actor.enter(self.env, here[:, 0:3], here[:, 3:7])

  @property
  def duration_s(self) -> float:
    """How long the bridge gets, in seconds. Three sources, most specific first.

    A number typed on the command line or written into the couple wins, because someone
    asked for it. Otherwise the solved one, which is the window that puts the robot where
    the entering skill needs it. Otherwise the entry's own, which is what the selector says
    that posture takes to reach and is the only answer available when the skill has no
    opinion about where the robot should stand.
    """
    if self.written_duration_s is not None:
      return self.written_duration_s
    if self.solved_duration_s is not None:
      return self.solved_duration_s
    return float(self.shortlist.duration_s[self.entry])

  @property
  def entry_state(self) -> torch.Tensor:
    """The state off the shortlist the next crossing aims at. (1, 13 + 2J).

    `entry` indexes it, so 0 is the best opening the entering skill was measured to have and
    the slider walks down from there. That is the whole selection policy for now: aim at the
    best state, and when the bridge cannot reach it from where the leaving skill left the
    robot, at the next one.

    Cached against the slider, because `triggered` reads this every control step and the
    shortlist is a numpy array on the wrong side of the device.
    """
    if self._entry_at != self.entry:
      states, _, _ = self.shortlist[self.entry]
      self._entry_at = self.entry
      self._entry_state = torch.as_tensor(
        states, dtype=torch.float32, device=self.env.device
      )
    return self._entry_state

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
    as soon as that is a window the bridge was trained on. So the kick fires early with a
    long window when the ball is far and late with a short one when it is close, rather than
    firing at the single distance a fixed window happened to match.

    The residual is what keeps that honest. A crossing travels along one line, so a duration
    can only fix how far, never which way, and a demand off that line stays off it. Firing on
    the duration alone would hand over on time to the wrong place.
    """
    if self.fire or self.tick == self.cfg.auto:
      return True
    entering = self.couple.entering
    here = state(self.robot)
    target = facing(self.entry_state, here)

    if self.arrive is not None:
      want = self.arrive(self.env)[:, 0:2]
      seconds, residual = crossing_time(here, target, want)
      low, high = self.command.cfg.duration_s_range
      if self.written_duration_s is None:
        self.solved_duration_s = float(seconds.min())
      fits = (seconds >= low) & (seconds <= high)
      return bool((fits & (residual <= ARRIVE_SLACK)).all())

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

  def condition(self) -> None:
    """Tell whichever skill owns the world right now what it is being asked for.

    Only the one that owns it. Walk and run read the same twist term, and a term written by
    both every step holds whoever wrote last, which is how a run bridged out of a walk used
    to be handed the walking speed it had just been bridged out of. Ownership passes at the
    moment the bridge is aimed rather than when the skill starts driving, because a skill
    with a goal has to have been told it before it takes over: the kick needs its launch
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
    """
    return self.names[self.phase]

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
    self.phase = self.tick = self.until = 0
    self.earned, self.scored, self.fell = 0.0, 0, False
    self.command.aimed = False
    self.enter(self.couple.leaving)

  @torch.no_grad()
  def __call__(self, obs):
    self.condition()

    if self.phase == 0 and self.triggered():
      obs = self.cross()
    elif self.phase == 1 and bool((self.command.step >= self.command.deadline).all()):
      obs = self.hand_over()
    elif self.phase == 2:
      self.watch()
      if self.tick >= self.until:
        self.settle()

    self.tick += 1
    return self.acts[self.phase](obs)

  def watch(self) -> None:
    """Score the entering skill while it drives, the way the selector scored it.

    The arena has no reward manager, since nothing here is trained, so the skill's own reward
    is computed on demand from the config it was trained with. Same terms, same weights, same
    discount, same normalization by the same baseline: a hand-over that passes here scores
    what an entry off the top of the shortlist scored.

    Discounted, and that is the change. A plain mean over the window forgives an entry the
    skill stumbles out of and then recovers from, which is exactly the state the shortlist is
    built to avoid aiming at, so a verdict computed that way would keep passing the
    hand-overs the selector was reworked to stop choosing.

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
    self.earned += self.selector.discount**self.scored * float(self.env.reward_buf[0])
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
    baseline = self.selector.baseline
    if baseline is None:
      # No measured baseline for this skill, so there is nothing to be a share of. Say the
      # number and refuse the verdict rather than dividing by an invented one
      print(
        f"  {self.couple.entering.name} {'fell' if self.fell else 'ran'}, discounted return "
        f"{earned:.3f}. No baseline measured for this skill, so there is no pass or fail."
      )
      self.enter(self.couple.leaving)
      return
    performance = earned / max(baseline, 1e-6)
    passed = not self.fell and performance >= self.selector.bar
    print(
      f"  {'PASS' if passed else 'FAIL'}: {self.couple.entering.name} "
      f"{'fell' if self.fell else 'ran'}, performance {performance:.2f} "
      f"against a bar of {self.selector.bar:.2f}"
    )
    self.enter(self.couple.leaving)

  def cross(self):
    """Aim the bridge at one state off the entering skill's shortlist, place that skill,
    start the clock."""
    self.fire, self.phase = False, 1
    here = state(self.robot)
    self.left_from = here[:, 0:3].clone()
    self.target = aim(
      self.env,
      self.command,
      self.couple.entering,
      self.entry_state,
      here,
      self.duration_s,
      self.arrive,
    )
    print(f"\ncross: {self.duration_s:.2f} s to entry {self.entry}{self.verdict(here)}")
    return fresh_obs(self.env)

  def verdict(self, here: torch.Tensor) -> str:
    """Whether the window just asked for is one the bridge has seen, and one a body could
    cross at all. Two questions, two answers.

    The band is what training drew: durations uniform over `duration_s_range` and nothing
    else, because every window it saw was a stretch of one rollout that long. Outside that
    band the bridge is being asked something it was never shown, so a bad score there is
    evidence about the slider.

    The acceleration is physics, and the one thing the recordings cannot vouch for. At
    inference the pair is not a recording: it is wherever the outgoing skill left the robot
    and whichever frame of the entering skill was picked, and nothing stops those two from
    being further apart than a body can travel in the time given.
    """
    change = float((self.target[0, 7:10] - here[0, 7:10]).norm())
    cfg = self.command.cfg
    seconds = self.duration_s
    accel = change / seconds

    said = f", sheds {change:.1f} m/s in {seconds:.2f} s ({accel:.1f} m/s^2)"
    low, high = cfg.duration_s_range
    if not low <= seconds <= high:
      return (
        f"{said}: {seconds:.2f} s is outside the {low:.2f}-{high:.2f} s it trained on"
      )
    if accel > MAX_ACCEL:
      return f"{said}: past the {MAX_ACCEL:.0f} m/s^2 a humanoid sustains"
    return said

  def hand_over(self):
    """Report the arrival, then let the entering skill drive.

    The score is the command's own metric against its own calibrated tolerances, so a
    hand-over here and a line of evaluate.py read on one scale. Computed here rather than
    read off the command because nothing in this arena is scored: the reward term that
    latches an arrival in training never runs, so the command's `score` stays at zero.
    """
    self.phase = 2
    self.until = self.tick + self.cfg.entering_steps
    self.earned, self.scored, self.fell = 0.0, 0, False
    now, want = state(self.robot)[0], self.target[0]
    joints = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.robot.num_joints)
    errors = channel_errors(now.unsqueeze(0), want.unsqueeze(0), self.command.arms)
    score = float(arrival_score(errors, self.command.tolerances)[0])
    # How far the body really went, over how far it was asked to go. Config.overshoot is
    # this minus one, and this is the only way to measure it: the placement is a model of a
    # walking body decelerating, and the body is the truth
    went = float((now[:2] - self.left_from[0, :2]).norm())
    asked = float((want[:2] - self.left_from[0, :2]).norm())
    travelled = f"  travelled {went / asked:.2f}x" if asked > 1e-3 else ""
    print(
      f"  arrived: score {score:.3f}  "
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
      spot = arrive(self.env)[0, 0:2]
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

    entry = server.gui.add_slider(
      "entry",
      min=0,
      max=max(len(run.shortlist) - 1, 1),
      step=1,
      initial_value=run.entry,
      hint="Which state off the entering skill's table to aim at.",
    )
    entry.on_update(lambda _: setattr(run, "entry", int(entry.value)))

    low, high = run.command.cfg.duration_s_range
    duration = server.gui.add_slider(
      "duration_s",
      min=low,
      max=high,
      step=0.05,
      initial_value=run.duration_s,
      hint="How long the bridge gets. Released, it comes from the entry, or is solved so "
      "the crossing lands where the entering skill needs the robot.",
    )
    duration.on_update(
      lambda _: setattr(run, "written_duration_s", float(duration.value))
    )

    # A duration is now something to release rather than only something to set: the entry
    # carries one and the geometry can imply one, and both are better answers than a number
    # left on a slider from the last run
    auto = server.gui.add_checkbox(
      "duration from the entry",
      initial_value=run.written_duration_s is None,
      hint="Let the entry state and the entering skill's demand decide the window.",
    )
    auto.on_update(
      lambda _: setattr(
        run, "written_duration_s", None if auto.value else float(duration.value)
      )
    )


def main(couple: Couple) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  cfg = tyro.cli(
    Config,
    default=replace(
      Config(),
      duration_s=couple.duration_s,
      overshoot=couple.overshoot,
    ),
    config=mjlab.TYRO_FLAGS,
  )
  if cfg.viewer == "none" and cfg.auto is None and couple.entering.ready is None:
    raise SystemExit(
      f"--viewer none has nothing to press the button, and {couple.entering.name} has no "
      f"precondition to fire on: pass --auto N."
    )

  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Before the simulation and the three checkpoints, because this is the one thing a couple
  # can be missing and building the arena first means waiting a minute to be told a file is
  # not there
  selector = Selector.load(couple.entering.name)
  for line in selector.lines():
    print(f"{couple.entering.name}: {line}")

  env = ManagerBasedRlEnv(cfg=arena(couple), device=device)
  bridge = Actor(BRIDGE_GROUP, BRIDGE_TASK_ID)
  policies: dict[str, Policy] = {}
  for actor in (couple.leaving, couple.entering, bridge):
    explicit = cfg.checkpoint if actor is bridge else None
    checkpoint = find_checkpoint(load_rl_cfg(actor.task).experiment_name, explicit)
    print(f"{actor.name:8s} {checkpoint}")
    policies[actor.name] = Policy(actor.task, checkpoint, env, actor.name, device)

  env.reset()
  run = Run(env, couple, policies, selector, cfg)
  print(
    f"{couple.entering.name}: aiming at '{run.shortlist.postures[cfg.entry].name}' "
    f"over {run.duration_s:.2f} s, {len(run.shortlist)} entries in the table"
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
    env.close()
    return

  import viser

  from mjlab.viewer import ViserPlayViewer

  server = viser.ViserServer(label=f"{couple.leaving.name}2{couple.entering.name}")
  panel(server, run)
  wrapped = RslRlVecEnvWrapper(
    env, clip_actions=load_rl_cfg(couple.entering.task).clip_actions
  )
  ViserPlayViewer(
    wrapped, run, viser_server=server, info_provider=lambda _: run.active
  ).run()
  wrapped.close()
