"""Drive one skill, press a button, let the bridge hand over to the next.

A transition script names two actors and what is on the floor. Everything else is here.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2jump
    uv run python -m ...transitions.walk2jump --viewer none --auto 120

##
# The three phases
##

    leaving   the first skill drives, steered from the panel
    bridge    the bridge drives, for `steps` steps, toward one target state
    entering  the second skill drives, from wherever the bridge left the robot

The viewer's `Active` line names whoever is driving, a skill by its own name or `bridge`,
and from the moment a target is chosen a translucent robot stands in it. It stays there
after the hand-over, so the gap between it and the real robot is the arrival error. Scene >
Debug Viz > Bridge turns it off.

The target is a single state, because that is what the bridge is trained on: a pose, a
height, a tilt, joint angles and every velocity, at one instant. Where does such a state
come from? From the entering skill itself. `profile` rolls it out once from its own reset
and records what it does, and any frame of that rollout is a state it is known to work
from. Picking one is the whole of what a real controller would have to decide; here it is
a slider.

From the rollout, and pointedly not from whatever reference the skill was tracking while
it ran. A retargeted clip is a description of a motion, not a state a robot is ever in:
the jump's, at frame 90, stands nearly four centimetres into the floor.

A second slider, `preview`, stands a ghost in whichever frame it is on, cool where the
target's is warm and where the rollout was recorded rather than where the crossing is
going, so a frame worth aiming at can be picked by looking at it. Since the target is that
same frame moved rigidly, what the preview shows is what the bridge will be sent to.

##
# Two things that are easy to get wrong and are the reason this file exists
##

**Where the target is put.** The bridge trained on targets placed where a body carrying
the momentum it has would naturally arrive: the centre of the reachable disc, `(v_now +
v_target) * T / 2` ahead. Dropping the target on top of the robot instead asks it to stop
dead in every test, which it was never asked to do in training, and reads as the bridge
being worse than it is.

**When the entering skill's reference is pinned.** A clip tracker has to be told where its
clip is. Pinning it at hand-over, to wherever the robot actually ended up, silently erases
the arrival error -- the clip moves to meet the robot and every hand-over looks perfect. So
`Actor.enter` is called when the target is chosen, not when control changes, and the clip
is pinned to where the robot is *supposed* to be. Whatever gap is left is the real one.
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

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.tests.bridge_simple import BRIDGE_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.tests.bridge_simple.env_cfg import (
  bridge_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.bridge_simple.mdp import (
  BridgeCommand,
  BridgeCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.bridge_simple.motions import (
  ROOT_STATE_DIM,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul, yaw_quat
from mjlab.viewer.debug_visualizer import DebugVisualizer

ROBOT = "robot"
BRIDGE_GROUP = "bridge"

MAX_ACCEL = 3.0
"""What a humanoid root sustains, in m/s^2, roughly.

Not a training parameter any more. The bridge learns from recorded stretches of motion,
which are feasible because a body performed them, so nothing in training has to decide
whether a pair can be joined. Here it does: the slider will happily ask for a velocity
change in a tenth of a second, and a score of zero on a window no body could cross says
nothing about the bridge. It is a sanity line printed next to the request, and that is
all."""


class Actor(NamedTuple):
  """One frozen skill, and how to tell it where it is taking over."""

  name: str
  task: str

  enter: Callable[[ManagerBasedRlEnv, int, torch.Tensor, torch.Tensor], None] | None = (
    None
  )
  """Put this skill's reference, and whatever else it keeps per episode, where it belongs.

  Called as `enter(env, frame, pos, heading)`: the profile frame it will take over at, the
  world position the robot will be at, and the direction it should head. A skill with no
  reference and no episode state of its own does not need one.

  Placement only. What state the robot has to arrive in is `aim`'s to say, and it says it
  from the profiled rollout: a state this skill was measured holding. A reference is not
  one, and a retargeted clip is not one by several centimetres of floor.

  `pos` is where the robot is *meant* to be, not where it is: the bridge has not crossed
  yet. See this module's header for why that distinction is the difference between measuring
  a hand-over and faking one.
  """

  ready: Callable[[ManagerBasedRlEnv, torch.Tensor], torch.Tensor] | None = None
  """Whether this skill's precondition is met, so the switch can fire on the world.

  Called as `ready(env, at)` every step while the first skill is driving, where `at` is
  where the robot is predicted to be when control would change. Returns one bool per
  environment. None means this skill has no precondition to watch and the switch waits for
  the button, which is what the jump does: a robot can jump anywhere.

  The prediction is the point. A trigger reading the present fires a stride late every
  time, because by the time the bridge has crossed the robot has walked on."""

  place: EventTermCfg | None = None
  """Where this skill's object starts, as a reset event. None for a skill without one."""


@dataclass(frozen=True)
class Couple:
  """Two skills and the hand-over between them. Everything else each skill brings itself."""

  leaving: Actor
  entering: Actor

  steps: int = 30
  frame: int = 50
  """This couple's window: how long the bridge gets, and which frame of the entering
  skill's profiled rollout to aim at. Defaults for the sliders, and overridable on the
  command line.

  Hardcoded per couple because choosing them is a real decision and not one this harness
  makes. Which state of the entering skill is worth arriving in depends on how much
  momentum there is to shed and on what that skill is doing at that frame, and getting it
  wrong does not look like a bad choice, it looks like a broken bridge. The push is the
  case that shows it: at frame 50 its profile is already shoving a crate at 1.1 m/s, so
  aiming there asks a walking robot to shed 2 m/s in 0.6 s, which is 3.3 m/s^2 and past
  what a body does -- and it scores 0.30. Aiming at the stance the push actually starts
  from, with half a second more to get there, scores 0.81 on the same bridge.

  A learned component is meant to make this choice eventually, from the state the robot is
  in, the frames on offer and the time each would need. Until then it is two numbers per
  script."""


##
# The arena.
##


TARGET_COLOR = (1.0, 0.72, 0.2, 0.45)
"""The target ghost. Warm, so it reads as a goal rather than as a second robot, and not
the blue `evaluate.py` gives the recorded body it draws beside the same policy."""

PREVIEW_COLOR = (0.35, 0.85, 0.45, 0.35)
"""The preview ghost. Cool and fainter than the target, because the two are on screen
together and they are not the same kind of thing: one is where the bridge is being sent,
the other is a frame of a rollout being looked through."""


class Aimed(BridgeCommand):
  """The bridge command with its own window drawing switched off, and its target drawn.

  In training a window is drawn on reset and the robot is teleported onto its start frame.
  Here the robot is wherever the leaving skill left it, which is the entire point, so the
  target arrives from outside through `aim` and nothing is ever teleported.
  """

  def __init__(self, cfg: BridgeCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.aimed = False
    self.preview: torch.Tensor | None = None
    """A state to stand a second ghost in, or None. Set from the panel and read nowhere
    else: it is drawn and that is all it does."""
    self._ghosts: dict[tuple[float, float, float, float], mujoco.MjModel] = {}

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """The state the bridge is crossing to, and whichever frame the preview is parked on.

    A pose and nothing else, because a pose is all that can be drawn: half of what the
    target says is velocity, and a still body says nothing about that. What it does show is
    the two things the sliders move -- where `aim` decided a body carrying this momentum
    could be after `steps`, and which of the entering skill's frames it is being asked to
    hold there -- and it stays where it was put after the hand-over, so the gap between it
    and the robot is the arrival error, left standing to be looked at.

    Nothing before the first `aim`. `target` opens as an identity quaternion at the world
    origin, and a robot lying in the floor at the corner of the scene is not a target.

    The preview is the same rollout `frame` indexes into, drawn in its own colour and left
    where it was recorded rather than moved to where the crossing is going. Scrubbing it
    walks the entering skill through its own run, so which frame to aim at can be decided
    by looking at the skill instead of by pressing the button and finding out.
    """
    if self.aimed:
      self._draw(visualizer, self.target, TARGET_COLOR, "target")
    if self.preview is not None:
      self._draw(visualizer, self.preview, PREVIEW_COLOR, "preview")

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
      # From `qpos0` rather than from zeros: everything this arena holds that is not the
      # robot keeps its own default, and a zero quaternion is not a rotation.
      qpos = np.array(self._env.sim.mj_model.qpos0, dtype=np.float64)
      row = states[batch].cpu().numpy()
      qpos[free[0:3]] = row[0:3]
      qpos[free[3:7]] = row[3:7]
      qpos[joints] = row[ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
      # `alpha` and not the colour's own: the viewer takes an rgba geom's opacity from
      # this argument and only its hue from the model.
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
        # the solver uses and draw a robot made of boxes; the rest of this arena is
        # whatever the two skills put on the floor, which at the target would be a second
        # ball or a second crate hanging in the air beside it.
        ghost.geom_rgba[geom, 3] = 0.0
    return ghost


@dataclass(kw_only=True)
class AimedCfg(BridgeCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> Aimed:
    return Aimed(self, env)


def arena(couple: Couple) -> ManagerBasedRlEnvCfg:
  """The bridge's own environment, with both skills' machinery merged into it.

  Built on the bridge's play config rather than on a skill's, so the robot, the terrain and
  the bridge's observation are exactly what it trained against. Each skill then contributes
  what it needs and nothing else: its entities, its commands, the sensors its observation
  reads, and the observation itself.

  That observation is copied from the skill's own task verbatim. A checkpoint is tied to
  its term list *in order*, and feeding it the same numbers shuffled does not fail, it acts
  on nonsense -- so the order is taken from the one place that cannot disagree with the
  checkpoint, instead of being restated here where it would go stale.
  """
  cfg = bridge_env_cfg(play=True)

  trained = cfg.commands["bridge"]
  assert isinstance(trained, BridgeCommandCfg)
  aimed = {f.name: getattr(trained, f.name) for f in fields(trained)}
  # Draws the target ghost, and gets the viewer to offer a 'Bridge' checkbox under
  # Scene > Debug Viz that turns it off again.
  aimed["debug_vis"] = True
  # No window is ever drawn here -- `Aimed` takes its target from outside -- so the corpus
  # is loaded for its frame rate and nothing else. Unmirrored, which halves the couple of
  # hundred megabytes a viewer would otherwise spend on frames it will not read.
  aimed["corpus"] = replace(trained.corpus, mirror=False)
  cfg.commands["bridge"] = AimedCfg(**aimed)
  cfg.observations = {BRIDGE_GROUP: cfg.observations["actor"]}

  # Nothing is being trained and nothing should reset: a fall is the result of the test,
  # not an error to recover from, and a reset mid-run would move the robot out from under
  # the phase machine.
  cfg.rewards = {}
  cfg.terminations = {}

  for actor in (couple.leaving, couple.entering):
    task = load_env_cfg(actor.task, play=True)
    cfg.observations[actor.name] = replace(
      task.observations["actor"], enable_corruption=False
    )
    for name, entity in (task.scene.entities or {}).items():
      cfg.scene.entities.setdefault(name, entity)
    for name, command in task.commands.items():
      # Frozen, because in this arena the harness owns when a skill's goal changes and
      # `Actor.enter` is how it says so. A guard rather than a fix for anything observed:
      # both skills that carry a goal already resample on a timer long enough never to
      # fire, and the twist's is overwritten every step below. What it rules out is a
      # skill's goal moving under a transition that is halfway through being measured.
      # `gui` and `debug_vis` off with it. Both are for watching a skill train on its own:
      # the sliders duplicate the panel below and fight it for the same fields, and the
      # ghost draws the jump's reference clip, which in a transition is a second robot
      # standing in the scene that nobody asked about and that is not what is being tested.
      cfg.commands.setdefault(
        name,
        replace(
          command,
          resampling_time_range=(1.0e9, 1.0e9),
          gui=False,
          debug_vis=False,
        ),
      )
    # And the sensors those observations read: the kick watches a foot-ball contact, the
    # push a robot-crate one, and neither exists in an arena that has never seen the object.
    have = {sensor.name for sensor in (cfg.scene.sensors or ())}
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
      sensor for sensor in (task.scene.sensors or ()) if sensor.name not in have
    )

  # The bridge's budgets were sized for a bare plane and one robot. This arena carries
  # whatever the two skills brought with them, and a constraint dropped on overflow is a
  # contact that silently did not happen, which looks like a robot sinking into the floor
  # rather than like an error.
  cfg.sim.nconmax = 140
  cfg.sim.njmax = 800
  cfg.sim.contact_sensor_maxmatch = 500

  # Each skill puts its own object out, with its own placement function. Deep copied
  # because building an environment resolves names into indices inside the config it is
  # handed, and a term shared between two environments carries the first one's indices into
  # the second.
  for actor in (couple.leaving, couple.entering):
    if actor.place is not None:
      cfg.events[f"place_{actor.name}"] = copy.deepcopy(actor.place)
  return cfg


##
# Loading a frozen policy.
##


def find_checkpoint(experiment: str, explicit: Path | None = None) -> Path:
  """The newest checkpoint of an experiment, said out loud.

  Picked by modification time when not given, which is convenient and has bitten this
  project before: a stale run left in `logs/` outranks the one you meant. The path is
  printed rather than assumed, so loading the wrong policy is at least a visible mistake.
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
    # Point both roles at this arena's group instead of the task's own. `load_rl_cfg`
    # hands back a deep copy, so the registry is not disturbed.
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


def aim(
  env: ManagerBasedRlEnv,
  command: Aimed,
  entering: Actor,
  ref: torch.Tensor,
  frame: int,
  here: torch.Tensor,
  steps: int,
) -> torch.Tensor:
  """Point the bridge at a frame of the entering skill's own rollout, moved to meet the robot.

  The target is the recorded state and nothing else. Every skill here is egocentric -- it
  reads its own body frame, and a tracker reads poses relative to an anchor it carries with
  it -- so a state it produced facing one way is a state it can be in facing another, and
  where in the world that happens does not enter into it. What the move has to preserve is
  everything the pose says *relative to the direction of travel*: the pelvis twist, the roll
  and pitch, the joint angles, the velocities. One yaw rotation, by the difference between
  the heading the rollout was recorded in and the heading the robot has now, does exactly
  that. The height is recorded, not chosen.

  What this replaced was reading the target off the entering skill's reference, through a
  return value from `Actor.enter`. For a clip tracker that hands the bridge a *retargeted
  human motion*, and no robot is ever in one: at frame 90 of the jump the clip stands 3.8 cm
  into the floor, 1.4 cm below the height the policy holds there, up to 0.42 rad from it in
  the joints, and descending at 0.28 m/s where the policy descends at 0.15. The bridge was
  being scored on arriving in a state its own entering skill never occupies.

  `Actor.enter` is still called, and still with where the robot is meant to be rather than
  where it is -- a clip tracker has to be told where its clip goes, and this module's header
  says what pinning it at hand-over instead would hide. It just no longer says what the
  target is. The clip lands under the arrival, so the skill takes over from a robot sitting
  on its reference rather than the few centimetres off it the rollout was.

  One placement, not the two this used to need. The velocity to arrive with is settled by
  the rotation alone, so where a body carrying this momentum can be after `steps` -- the
  centre of the disc, which is where training put every target -- is known before anything
  is placed.
  """
  heading = yaw_quat(here[:, 3:7])
  horizon = steps / command.corpus.fps

  # Frame 0 is the heading the rollout was recorded in, because that is the one `profile`
  # handed the skill. Everything the skill did afterwards is relative to it.
  target = turn_state(
    ref[:, frame], quat_mul(heading, quat_conjugate(yaw_quat(ref[:, 0, 3:7])))
  )
  target[:, 0:2] = here[:, 0:2] + (here[:, 7:9] + target[:, 7:9]) * horizon / 2.0

  if entering.enter is not None:
    entering.enter(env, frame, target[:, 0:3], heading)

  env_ids = torch.arange(command.num_envs, device=command.device)
  command.target[:] = target
  command.aimed = True
  command.open_window(env_ids, torch.full_like(command.deadline, steps))
  return target


def profile(
  env: ManagerBasedRlEnv, actor: Actor, policy: Policy, steps: int
) -> torch.Tensor:
  """Roll a skill once from its own reset, recording every state. (1, steps + 1, 13 + 2J).

  This is the skill's initiation set, sampled: the states it is known to work from, because
  they are the states it puts itself in when nothing has gone wrong. One of them becomes the
  bridge's target.
  """
  env.reset()
  robot: Entity = env.scene[ROBOT]
  here = state(robot)
  if actor.enter:
    actor.enter(env, 0, here[:, 0:3], here[:, 3:7])

  env.scene.write_data_to_sim()
  env.sim.forward()
  env.command_manager.compute(dt=0.0)
  env.sim.sense()
  env.abstraction_manager.compute(dt=0.0)
  obs = env.observation_manager.compute(update_history=True)

  rows = [state(robot)]
  for _ in range(steps):
    # Only the policy call goes in inference mode: stepping the env in there turns every
    # buffer it writes into an inference tensor and the next reset cannot write them.
    with torch.inference_mode():
      action = policy(obs)
    obs, _, _, _, _ = env.step(action)
    rows.append(state(robot))
  return torch.stack(rows, dim=1)


##
# The run.
##


@dataclass(frozen=True)
class Config:
  steps: int = 30
  """How long the bridge gets, in control steps. It trained on 10 to 50."""

  frame: int = 50
  """Which frame of the profiled rollout to aim at. Zero is the skill's own opening, which
  is what a naive hand-over gets; further in is a state the skill reaches at speed, and
  finding one that works better is most of what this harness is for."""

  entering_steps: int = 220
  profile_steps: int = 340
  speed: float = 1.0

  overshoot: float = 0.0
  """How much further the robot really travels than the target it was given, as a fraction.

  Applied to the switch and to nothing else, and that is the whole of why it works. The
  trigger asks "will the object be in reach when control changes"; `aim` asks "where shall
  the target go". Those read like the same quantity and are not: the target is where the
  robot is *told* to be, and the arrival is where it *ends up*, and the gap between them is
  the bridge's own tracking error. Put a correction into both and it cancels exactly --
  the trigger fires later, the target moves the same distance further, and the object ends
  up in the same wrong place. Put it into the trigger alone and the object lands where the
  skill wants it.

  Zero until it is measured. Every hand-over prints `travelled`, which is the actual
  distance covered over the ratio the placement predicted; set this to that number minus
  one and the switch is as precise as the prediction allows."""

  auto: int | None = None
  """Press the button by itself after N steps, whatever the world is doing. Needed headless
  for a skill with no precondition of its own, and an override for one that has."""

  patience: int = 900
  """Steps a headless run gets before it gives up waiting for the switch to fire."""

  checkpoint: Path | None = None
  """An explicit bridge checkpoint. The skills are always taken from their own logs."""

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
    ref: torch.Tensor,
    cfg: Config,
  ) -> None:
    self.env, self.couple, self.ref, self.cfg = env, couple, ref, cfg
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

    self.forward, self.lateral, self.heading = cfg.speed, 0.0, 0.0
    self.steps, self.frame = cfg.steps, cfg.frame
    self.preview = -1
    self.fire = self.done = False
    self.phase = self.tick = self.until = 0
    self.enter(couple.leaving)

  @property
  def preview(self) -> int:
    """Which frame of the entering skill's profiled rollout to stand a ghost in, -1 for
    none.

    A viewer for `ref` and nothing more: the run never reads it, and moving it changes no
    part of the transition. It is here because `frame` picks a state out of that rollout
    blind, and the only way to see what was picked used to be to press the button.
    """
    return self._preview

  @preview.setter
  def preview(self, frame: int) -> None:
    self._preview = frame
    self.command.preview = None if frame < 0 else self.ref[:, frame]

  def enter(self, actor: Actor, frame: int = 0) -> None:
    """Hand the world to a skill that is about to take over where the robot stands."""
    if actor.enter:
      here = state(self.robot)
      actor.enter(self.env, frame, here[:, 0:3], here[:, 3:7])

  def triggered(self) -> bool:
    """Whether to start crossing on this step.

    Three ways in, and the entering skill's own precondition is the one that means
    something. A skill with an object was trained with it in a particular place, and the
    switch fires when walking has brought the object into that place -- which is what a
    controller would be deciding, reduced to the one rule that makes the scenario run.
    The button and `--auto` remain for the skills that have no such rule, and as an
    override for the ones that do.
    """
    if self.fire or self.tick == self.cfg.auto:
      return True
    ready = self.couple.entering.ready
    if ready is None:
      return False
    here = state(self.robot)
    # Where the robot will *be* when control changes, which is not where `aim` will put the
    # target: half the ballistic lead, which is what a target with a standing object's own
    # velocity gets, and then whatever the bridge is known to overshoot it by.
    at = here[:, 0:3].clone()
    lead = (self.steps / self.command.corpus.fps) / 2.0
    at[:, 0:2] += here[:, 7:9] * lead * (1.0 + self.cfg.overshoot)
    return bool(ready(self.env, at).all())

  @property
  def label(self) -> str:
    return self.PHASES[self.phase]

  @property
  def active(self) -> str:
    """Whoever is driving right now, by name: a skill's own, or the bridge's.

    What the viewer shows, and the phase name is not it. 'entering' says a hand-over has
    happened and leaves you to remember which of the two skills that was, which is the one
    thing somebody watching a transition is actually trying to read off the screen.
    """
    return self.names[self.phase]

  def reset(self) -> None:
    """Start over, because the world just did.

    The viewer's reset button resets the environment and then calls this. Without it the
    two disagree: the robot is back at its opening pose and whichever skill was driving
    when the button was pressed carries on driving it, out of a hand-over that never
    happened. The clock goes back with the phase, the leaving skill is handed a world it
    is standing at the start of, and the target stops being drawn because it is the target
    of a crossing that no longer exists.

    Deliberately not reset: the sliders. `forward`, `steps` and `frame` are what the person
    watching set them to, and a reset is a request to run the same transition again, not to
    undo their settings.
    """
    self.fire = self.done = False
    self.phase = self.tick = self.until = 0
    self.command.aimed = False
    self.enter(self.couple.leaving)

  @torch.no_grad()
  def __call__(self, obs):
    twist = self.env.command_manager.get_term("twist")
    assert isinstance(twist, UniformVelocityCommand)
    twist.vel_command_b[:, 0] = self.forward
    twist.vel_command_b[:, 1] = self.lateral
    twist.heading_target[:] = self.heading
    twist.is_heading_env[:] = True

    if self.phase == 0 and self.triggered():
      obs = self.cross()
    elif self.phase == 1 and bool(
      (self.command.elapsed >= self.command.deadline).all()
    ):
      obs = self.hand_over()
    elif self.phase == 2 and self.tick >= self.until:
      self.phase, self.done = 0, True
      self.enter(self.couple.leaving)

    self.tick += 1
    return self.acts[self.phase](obs)

  def cross(self):
    """Aim the bridge at a frame of the entering skill, place that skill, start the clock."""
    self.fire, self.phase = False, 1
    frame = min(self.frame, self.ref.shape[1] - 1)
    here = state(self.robot)
    self.left_from = here[:, 0:3].clone()
    self.target = aim(
      self.env,
      self.command,
      self.couple.entering,
      self.ref,
      frame,
      here,
      self.steps,
    )
    print(f"\ncross: {self.steps} steps to frame {frame}{self.verdict(here)}")
    return self.env.observation_manager.compute()

  def verdict(self, here: torch.Tensor) -> str:
    """Whether the window just asked for is one the bridge has ever seen, and one a body
    could cross at all.

    Two different questions and they now have two different answers. The band is what
    training drew: deadlines uniform between `min_steps` and `max_steps`, and nothing else,
    because every window it saw was a real recording and a recording needs no feasibility
    rule to be feasible. Outside that band the bridge is being asked something it was never
    shown, and a bad score there is evidence about the slider.

    The acceleration is physics, and it is the one thing the recordings cannot vouch for.
    At inference the pair is not a recording: it is wherever the outgoing skill left the
    robot and whichever frame of the entering skill was picked, and nothing stops those two
    from being further apart than a body can travel in the time given.
    """
    change = float((self.target[0, 7:10] - here[0, 7:10]).norm())
    cfg = self.command.cfg
    seconds = self.steps / self.command.corpus.fps
    accel = change / seconds

    said = f", sheds {change:.1f} m/s in {seconds:.2f} s ({accel:.1f} m/s^2)"
    if not cfg.min_steps <= self.steps <= cfg.max_steps:
      return (
        f"{said}: {self.steps} steps is outside the "
        f"{cfg.min_steps}-{cfg.max_steps} it trained on"
      )
    if accel > MAX_ACCEL:
      return f"{said}: past the {MAX_ACCEL:.0f} m/s^2 a humanoid sustains"
    return said

  def hand_over(self):
    """Report the arrival, then let the entering skill drive.

    The score is the command's own, the same number training optimizes, so a hand-over here
    and a training episode are being read on one scale.
    """
    self.phase = 2
    self.until = self.tick + self.cfg.entering_steps
    now, want = state(self.robot)[0], self.target[0]
    joints = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.robot.num_joints)
    # How far the body really went, over how far it was asked to go. One is what `Config.
    # overshoot` has to be set to, minus one, and it is the only way to measure it: the
    # placement is a model of a walking body decelerating and the body is the truth.
    went = float((now[:2] - self.left_from[0, :2]).norm())
    asked = float((want[:2] - self.left_from[0, :2]).norm())
    travelled = f"  travelled {went / asked:.2f}x" if asked > 1e-3 else ""
    print(
      f"  arrived: score {float(self.command.closeness().max()):.3f}  "
      f"{float((now[:3] - want[:3]).norm()):.2f} m off  "
      f"speed {float(now[7:10].norm()):.2f} vs {float(want[7:10].norm()):.2f} m/s  "
      f"joints {float((now[joints] - want[joints]).abs().max()):.2f} rad" + travelled
    )
    return self.env.observation_manager.compute()


def panel(server, run: Run) -> None:
  """Every flag, as a slider. Showing the value and letting you move it are the same job."""
  with server.gui.add_folder("Drive"):
    for name, lo, hi, step in (
      ("forward", -0.5, 2.0, 0.05),
      ("lateral", -0.5, 0.5, 0.05),
      ("heading", -3.14, 3.14, 0.05),
    ):
      slider = server.gui.add_slider(
        name, min=lo, max=hi, step=step, initial_value=getattr(run, name)
      )
      slider.on_update(lambda _, n=name, s=slider: setattr(run, n, float(s.value)))

  with server.gui.add_folder(f"Hand over to {run.couple.entering.name}"):
    button = server.gui.add_button(run.couple.entering.name)
    button.on_click(lambda _: setattr(run, "fire", True))
    for name, lo, hi in (
      ("steps", run.command.cfg.min_steps, run.command.cfg.max_steps),
      ("frame", 0, run.ref.shape[1] - 1),
    ):
      slider = server.gui.add_slider(
        name, min=lo, max=hi, step=1, initial_value=getattr(run, name)
      )
      slider.on_update(lambda _, n=name, s=slider: setattr(run, n, int(s.value)))

    # Its own slider rather than another row in that loop: the two above are the hand-over,
    # this one only looks at the rollout they choose from.
    preview = server.gui.add_slider(
      "preview",
      min=-1,
      max=run.ref.shape[1] - 1,
      step=1,
      initial_value=run.preview,
      hint=f"Stand a ghost in this frame of the profiled {run.couple.entering.name}, "
      "where it was recorded. -1 hides it.",
    )
    preview.on_update(lambda _: setattr(run, "preview", int(preview.value)))


def main(couple: Couple) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  cfg = tyro.cli(
    Config,
    default=replace(Config(), steps=couple.steps, frame=couple.frame),
    config=mjlab.TYRO_FLAGS,
  )
  if cfg.viewer == "none" and cfg.auto is None and couple.entering.ready is None:
    raise SystemExit(
      f"--viewer none has nothing to press the button, and {couple.entering.name} has no "
      f"precondition to fire on: pass --auto N."
    )

  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env = ManagerBasedRlEnv(cfg=arena(couple), device=device)

  bridge = Actor(BRIDGE_GROUP, BRIDGE_TASK_ID)
  policies: dict[str, Policy] = {}
  for actor in (couple.leaving, couple.entering, bridge):
    explicit = cfg.checkpoint if actor is bridge else None
    checkpoint = find_checkpoint(load_rl_cfg(actor.task).experiment_name, explicit)
    print(f"{actor.name:8s} {checkpoint}")
    policies[actor.name] = Policy(actor.task, checkpoint, env, actor.name, device)

  ref = profile(env, couple.entering, policies[couple.entering.name], cfg.profile_steps)
  print(f"profiled {couple.entering.name}: {ref.shape[1]} frames")

  env.reset()
  run = Run(env, couple, policies, ref, cfg)

  if cfg.viewer == "none":
    obs = env.get_observations()
    for _ in range(cfg.patience):
      if run.done:
        break
      obs, _, _, _, _ = env.step(run(obs))
    else:
      # A trigger that never fires is a result, not a hang: the robot walked past its
      # object, or never lined up with it, and either way sitting here forever hides that.
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
