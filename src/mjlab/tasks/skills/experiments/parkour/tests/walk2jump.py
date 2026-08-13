"""The bridge scored on the hand-over it was built for: walking into a jump.

`bridge/evaluate.py` scores the bridge on held-out corpus windows, which answers whether
it learned to in-between. It does not answer what the architecture claims, because in a
corpus window both sides of the hole are one body doing one continuous thing, and at
inference they are two policies that have never met. This builds that window out of the
two policies themselves.

The whole script is four steps.

1. Walk. The walk policy is rolled down a bare plane and cut somewhere in its stride,
   drawn per environment from the experiment's own interrupt range. The frame at the cut
   is the hand-off: where the robot is, which way it faces, what its joints are doing and
   how fast it is going at the instant control is taken away. The `past` frames leading
   into it are the first unmasked window.

   The walk is then left running for `gap` frames past its own cut. Those frames are
   handed to nothing and are no part of the hand-over; they are what this policy would
   have gone on to do had the signal never come, which is what step 2 needs and what
   `SkillWindowSpec.overrun` names. Collected here directly rather than through the
   window plan, which asks for it nowhere in this experiment.

2. Jump, from where the walk was going. The jump policy is rolled from its nominal
   conditions, the clip's first frame, exactly the state its own training reset produces,
   but standing on the spot the walk would have reached `gap` frames after the cut and
   facing the heading it would have had there. Only that ground position and that heading
   come from the walk; the pose and the momentum are the jump's own.

   The look-ahead is the arrangement's whole point. Anchoring the jump on the cut asks
   the question after the fact: the target sits on ground the robot has already covered,
   and a body carrying a metre a second into the hole can only satisfy it by throwing the
   momentum away and stepping back into its own past. Anchored ahead, the same hole reads
   as an instruction rather than a correction. In `gap` frames you were going to be there;
   be in this state instead. What the bridge has to change is the state it arrives in, not
   the ground it covers.

   It is also the shape the bridge was trained on. Every corpus window travels across its
   hole, by construction (`WindowCfg.min_travel`), so a target sitting on top of the
   hand-off is a displacement the training set does not contain.

   The second unmasked window is the stretch of that rollout starting at the entry.
   `--entry-lead` says where that is: zero puts it on the crouch, the frame where the torso
   is lowest before takeoff, and a positive lead moves it that many frames earlier. What
   lies before it is the policy standing up and settling, which is the second and a half a
   robot arriving at speed should not have to spend, and how much of it to leave the jump
   policy is the thing the lead trades.

3. The hole is what lies between the hand-off and the entry, and it is empty. No
   recording covers it because no body performed this pair, and producing something a body
   could perform there is the bridge's whole job. The bridge is handed the hole the way
   training hands it a spliced window: the arrival frame held across it and nothing to
   track, because there is nothing there to track.

   How long it is, is now an input. The look-ahead cannot be taken without committing to
   a duration first, and the jump that follows is rolled at whatever that duration picked
   out, so the entry is downstream of the hole rather than the thing that sizes it.
   `--gap` is that duration, and it is both quantities at once: how far ahead the target
   is placed, and how long the bridge has to arrive in it.

4. Resume. The jump policy is handed the robot wherever the bridge left it, with its clip
   pinned exactly where the nominal rollout had it and wound to the entry. The clip is
   never re-pinned onto the robot's actual arrival; doing that would slide the target onto
   whatever the bridge produced and erase the error this script exists to measure.

Everything lives in one world, the arena's, and never moves between coordinate systems.
The one exception is the bridge, which runs in its own environment and slides the hand-off
onto its origin; `bridge_across` shifts back on the way out, so nothing above it ever sees
the bridge's coordinates.

The same hand-over is run again with no bridge at all, the walk's final state passed
straight to the jump, which is architecture 0. That column is the thing to beat, and the
look-ahead is what makes it a real column: standing still through the hole now costs it
the whole delta t of ground the target was placed across, on top of arriving in the wrong
state. Under the old placement it was already about where it needed to be.

    uv run python -m mjlab.tasks.skills.experiments.parkour.tests.walk2jump
    uv run python -m mjlab.tasks.skills.experiments.parkour.tests.walk2jump --view True
    uv run python -m mjlab.tasks.skills.experiments.parkour.tests.walk2jump \\
        --num-envs 32 --bridge False

The viewer colours it the way the corpus viewer does. Solid blue is what actually
happened. The ghost is the nominal jump, wound so its entry falls on the frame the solid
robot arrives at: green once the two are supposed to agree, red across the hole, where the
ghost is standing up and settling and the bridge is being asked to skip all of it. The two
meeting at the entry is the thing to watch for, and the gap between them when they get
there is the arrival error.

The ghost now stands a look-ahead ahead of the robot when the hole opens, rather than on
top of it, so the two closing over the hole is the shape a good crossing has. A bridge
that brakes leaves the ghost behind and reaches the entry late and short.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.skills.architectures.arch_4.bridge import BRIDGE_TASK_ID
from mjlab.tasks.skills.architectures.arch_4.bridge.dataset import (
  G1,
  ROOT_STATE_DIM,
  WindowCfg,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.env_cfg import bridge_env_cfg
from mjlab.tasks.skills.architectures.arch_4.bridge.evaluate import (
  find_checkpoint,
  load_policy,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.commands import (
  BridgeCommand,
  BridgeCommandCfg,
  arrival_error,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.view import (
  CONTEXT_COLOR,
  MASKED_COLOR,
  MODEL_COLOR,
  Ghost,
  visual_meshes,
)
from mjlab.tasks.skills.experiments.parkour import (
  ENTITY_NAME,
  WINDOWS,
  build_pool,
)
from mjlab.tasks.skills.experiments.parkour.arena import (
  JUMP_COMMAND,
  TWIST_COMMAND,
  WALK_SPEED,
  parkour_arena_env_cfg,
)
from mjlab.tasks.skills.experiments.parkour.controller import JUMP, WALK
from mjlab.tasks.skills.experiments.parkour.jump.jump_env_cfg import ANCHOR_BODY
from mjlab.tasks.skills.experiments.parkour.jump.mdp.commands import JumpCommand
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.velocity.mdp import UniformVelocityCommand

CONTROL_HZ = 50.0


@dataclass(frozen=True)
class Config:
  num_envs: int = 1
  """How many hand-overs to build. Each draws its own cut into the walk and its own jump
  distance, so they are different questions rather than one question repeated."""

  bridge: bool = True
  """Run the bridge across the hole. `--bridge False` measures only the no-bridge
  baseline, which needs no checkpoint and is the right thing to run before one exists."""

  checkpoint: Path | None = None
  """The bridge to score. Defaults to the newest under logs/; the path is printed."""

  gap: int = 30  # WindowCfg().gap_range[0]
  """Delta t, in frames. How far past the cut the walk is looked ahead to place the
  target, and how long the bridge then has to arrive in it. One number rather than two,
  because the hole says "in this many frames you were going to be there, be in this state
  instead", and the horizon and the deadline are the same instant.

  It can no longer be derived from the jump the way it once was. The nominal jump is
  rolled at the look-ahead, so the frame it crouches on is downstream of this number and
  cannot also set it. Keep it inside `WindowCfg.gap_range`, which is the span of hole
  durations the bridge was trained across and also where the old derivation capped out on
  nearly every hand-over."""

  entry_lead: int = 10
  """Frames before the crouch to hand the robot over: which state of the nominal jump the
  bridge is aimed at. Zero is the crouch itself, the bottom of the dip.

  The crouch is the hardest frame in the rollout to be delivered into. It is the most
  loaded pose the jump makes before it leaves the ground, and a bridge that arrives in it
  slightly wrong has no frames left to be wrong in. Entering earlier hands over a shallower
  state and leaves the jump policy the last stretch of its own descent to absorb the
  residual with, which is what a tracking policy is good at. What it costs is that the
  entry walks back into the stand-and-settle this arrangement exists to skip, so there is a
  lead beyond which the bridge is being asked to reproduce the standing up rather than skip
  it. Somewhere between those is what this parameter exists to sweep.

  Anchored on the crouch rather than counted from the start of the rollout, on purpose. The
  crouch lands anywhere between step 73 and 111 depending on which clip was drawn and how
  far it was stretched, so a fixed index from the start is a different moment of the motion
  in every environment and the hand-overs stop being the same question. A lead is the same
  moment in all of them.

  Negative enters after the crouch, during the push off the ground. Allowed, and unlikely
  to be a good idea: the bridge would have to deliver a body already on its way up."""

  jump_distance: tuple[float, float] = (1.5, 1.6)
  """Range the per-environment jump goal is drawn from [m]."""

  walk_settle: int = 64
  """Steps of walking before any cut may be taken, so the gait is a gait."""

  jump_steps: int = 200
  """Steps of the nominal jump rollout. Long enough to cover the longest clip's crouch,
  its flight and its landing."""

  window: int = WindowCfg().future
  """Frames of the nominal jump kept from the entry onward: the second unmasked window.

  The bridge is aimed at the first `WindowCfg().future` of it, which is the width its
  target observation was trained on, so making this longer changes what the viewer and the
  resume are shown against rather than what the bridge is pointed at."""

  resume_steps: int = 160
  """Steps the jump policy is given after the hand-over. The experiment's own number: the
  clips touch down between steps 90 and 141, so this falls after every landing and before
  any clip runs out."""

  view: bool = False
  """Open the viewer instead of printing a table."""

  port: int = 8080
  device: str | None = None
  seed: int = 0


##
# State, in one layout.
##


def read_state(robot: Entity) -> torch.Tensor:
  """The robot as one corpus-layout row per environment. (N, 13 + 2J).

  The same thirteen root numbers and the same joint order the corpus uses, so a recorded
  rollout and a recorded motion are the same kind of thing and go through the same
  machinery. `root_link_*` rather than `root_com_*` because that is what the bridge's
  reward compares against, and the two are half a hand apart on this robot.
  """
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


def write_state(robot: Entity, states: torch.Tensor, num_joints: int) -> None:
  """Put the robot into a corpus-layout state, joint limits respected."""
  limits = robot.data.soft_joint_pos_limits
  joint_pos = states[:, ROOT_STATE_DIM : ROOT_STATE_DIM + num_joints].clamp(
    limits[:, :, 0], limits[:, :, 1]
  )
  robot.write_joint_state_to_sim(joint_pos, states[:, ROOT_STATE_DIM + num_joints :])
  robot.write_root_state_to_sim(states[:, 0:ROOT_STATE_DIM])


def reference_state(jump: JumpCommand) -> torch.Tensor:
  """The clip's current frame, in the same layout. (N, 13 + 2J).

  Wherever the clip is currently pinned, so writing this onto the robot puts it exactly on
  the reference rather than near it.
  """
  return torch.cat(
    [
      jump.body_pos_w[:, 0],
      jump.body_quat_w[:, 0],
      jump.body_lin_vel_w[:, 0],
      jump.body_ang_vel_w[:, 0],
      jump.joint_pos,
      jump.joint_vel,
    ],
    dim=-1,
  )


##
# The two rollouts.
##


@dataclass
class Rollout:
  """One policy driven for a while, recorded frame by frame.

  `states` is (N, T + 1, S): the state before any action, then one per action. `torso_z`
  is kept alongside because the crouch is defined by it and a root height cannot tell a
  crouch from a short robot.

  `alive` is (N, T + 1) and says, frame by frame, whether that environment had terminated
  yet. Frames after a termination are held copies of the last real one, so this is what
  separates a recording from a freeze. It is per frame rather than a single flag because
  the walk is deliberately rolled past the point each environment cares about: asking
  whether it was still up at the end would judge every hand-over on a stretch of walking
  that happens after the signal it is built around.
  """

  states: torch.Tensor
  torso_z: torch.Tensor
  alive: torch.Tensor

  @property
  def survived(self) -> torch.Tensor:
    """(N,). Environments that were still up at the last recorded frame."""
    return self.alive[:, -1]


def jump_command(env: ManagerBasedRlEnv) -> JumpCommand:
  term = env.command_manager.get_term(JUMP_COMMAND)
  assert isinstance(term, JumpCommand)
  return term


def refresh(env: ManagerBasedRlEnv):
  """Recompute everything downstream of a state written by hand.

  The tail of `ManagerBasedRlEnv.reset`, minus the reset itself. `command_manager.compute`
  is included rather than skipped, and it does advance the jump's frame pointer by one,
  which is exactly what a real reset does: the policy is handed the observation it was
  trained to act on rather than a tidier one.
  """
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.command_manager.compute(dt=0.0)
  env.sim.sense()
  env.abstraction_manager.compute(dt=0.0)
  return env.observation_manager.compute(update_history=True)


def record(
  env: ManagerBasedRlEnv, pool: SkillPool, skill_id: int, obs, steps: int, torso: int
) -> Rollout:
  """Drive one skill for `steps` and keep every frame of it.

  An environment that terminates is latched dead and its last kept frame is held from
  then on. This has to happen before the frame is read: `env.step` auto-resets whatever
  terminated inside it, so by the time control comes back the robot is already standing in
  a fresh episode somewhere else, and recording that would put a teleport in the middle of
  a rollout.
  """
  robot: Entity = env.scene[ENTITY_NAME]
  device = env.device
  assignment = torch.full((env.num_envs,), skill_id, dtype=torch.long, device=device)
  alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)

  frames = [read_state(robot)]
  heights = [robot.data.body_link_pos_w[:, torso, 2]]
  standing = [alive.clone()]
  for _ in range(steps):
    with torch.inference_mode():
      action = pool.act(obs, assignment)
    obs, _, terminated, time_out, _ = env.step(action)
    alive = alive & ~(terminated | time_out)
    frames.append(torch.where(alive.unsqueeze(-1), read_state(robot), frames[-1]))
    heights.append(
      torch.where(alive, robot.data.body_link_pos_w[:, torso, 2], heights[-1])
    )
    standing.append(alive.clone())

  return Rollout(
    states=torch.stack(frames, dim=1),
    torso_z=torch.stack(heights, dim=1),
    alive=torch.stack(standing, dim=1),
  )


def roll_walk(
  env: ManagerBasedRlEnv, pool: SkillPool, cfg: Config, torso: int
) -> tuple[Rollout, torch.Tensor]:
  """Walk down the plane, and pick where each environment gets interrupted.

  The cut is drawn from the experiment's own interrupt range for the walk, which is what
  every bridging architecture here samples hand-overs from. A cut is a moment in a gait,
  not a moment in time: half of them catch a foot planted and half catch one in the air,
  and a bridge that only ever leaves from one of those has not been tested.

  The recording runs `gap` frames past the latest cut, so every environment has its own
  cut plus a delta t of walking on the end of it. That tail is counterfactual and is
  handed to nothing: it is only ever read for where it says the robot was going, which is
  the ex ante half of the question. The command is safe to leave alone across it, since
  the arena pushes the twist resampling clock out of reach (`arena._constrain_twist`), so
  the walk keeps the speed and the heading it was given the whole way.
  """
  low, high = WINDOWS["walk"].interrupt_range
  low = max(low, cfg.walk_settle, WindowCfg().past)
  cut = torch.randint(low, high + 1, (env.num_envs,), device=env.device)

  obs, _ = env.reset()
  twist = env.command_manager.get_term(TWIST_COMMAND)
  assert isinstance(twist, UniformVelocityCommand)
  twist.vel_command_b[:, 0] = WALK_SPEED
  twist.vel_command_b[:, 1] = 0.0

  return record(env, pool, WALK, obs, int(cut.max()) + cfg.gap, torso), cut


def cut_walk(walk: Rollout, cut: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """The `past` frames running into the cut, and the cut frame itself.

  The second return is the hand-off: the state control is taken away in, and the state
  everything downstream is placed against.
  """
  past = WindowCfg().past
  index = torch.arange(walk.states.shape[0], device=walk.states.device)
  history = torch.stack(
    [walk.states[index, cut - past + 1 + k] for k in range(past)], dim=1
  )
  return history, history[:, -1]


def look_ahead(
  walk: Rollout, cut: torch.Tensor, gap: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """Where the walk would have been `gap` frames after the cut, and whether it got there.

  This is the whole ex ante mechanism, and it is one index. The walk kept running past
  every cut, so the frame is already recorded; nothing here predicts anything, it reads
  off what the outgoing policy actually went on to do. That is an oracle, and it is the
  right one for this script: the question is whether a target placed ahead of the robot
  is a better thing to ask a bridge for, not whether the placement can be guessed. A
  composition running for real has to estimate this frame instead, from the skill it is
  leaving, and that estimate is a separate piece of work.

  The second return says the walk was still up at that frame rather than at the end of
  the shared recording. An environment whose counterfactual tail falls over has no target
  and is not a question; one that falls over later, long after its own cut, is untouched
  by it.
  """
  index = torch.arange(walk.states.shape[0], device=walk.states.device)
  ahead = cut + gap
  return walk.states[index, ahead], walk.alive[index, ahead]


@dataclass
class Pinning:
  """Where the nominal rollout had its clip, as it was pinned before the rollout ran.

  Captured before rather than read after, because a rollout that terminates is auto-reset
  inside `env.step` and a reset resamples the clip and its scale. Read afterwards, an
  environment that fell would report the placement of a clip that never ran, and the resume
  would be restored onto it.
  """

  motion_ids: torch.Tensor
  scales: torch.Tensor
  anchor_pos: torch.Tensor
  anchor_yaw: torch.Tensor


def roll_jump(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  cfg: Config,
  torso: int,
  at: torch.Tensor,
  num_joints: int,
) -> tuple[Rollout, torch.Tensor, Pinning]:
  """Roll the jump from its own initiation set, standing where the walk was going.

  Nominal means nominal in everything except placement: the robot is written onto the
  clip's first frame, which is the state this policy's training reset produces and the
  only one it can be assumed to be good from. What is not nominal is where that happens.
  The clip is pinned to `at` first, so the whole jump plays out from there rather than
  from the origin.

  `at` is the look-ahead frame, where the walk would have been a delta t after the cut,
  not the cut itself. Pinning on the cut asks what a jump started at this instant would
  have done, which is a target on ground the robot has already left; a body still carrying
  its walking speed can only reach it by braking. Pinning ahead asks what a jump started
  from here would have done by the time the robot got here anyway, and leaves the momentum
  something to do.

  Only `at`'s xy and yaw are used. Its joint angles, its height and its momentum are the
  walk's, and reproducing them is the bridge's job, not the reference's.

  What is recorded is the robot, not the clip. A reference is what the policy was aiming
  at; the window has to be a stretch of motion that actually happened.

  The crouch is the lowest the torso gets before the clip leaves the ground. Searching
  only up to takeoff is what keeps it a crouch: a body in flight passes through lower
  torso heights on the way down, and the bottom of a landing is not somewhere to hand a
  robot over to.

  It is returned as a landmark, not as the hand-over. Where the hand-over actually goes is
  `--entry-lead` frames ahead of it and `assemble` decides that, because this function has
  no business knowing how much of its own run-up the jump policy is being left.
  """
  robot: Entity = env.scene[ENTITY_NAME]
  jump = jump_command(env)
  env_ids = torch.arange(env.num_envs, device=env.device)

  env.reset()
  distances = torch.empty(env.num_envs, device=env.device).uniform_(*cfg.jump_distance)
  # Which clip each environment jumps, and how far it is stretched.
  jump.apply_goals(env_ids, distances)
  # Then move the clip to the look-ahead, and the robot onto the clip.
  jump.anchor_to_robot(env_ids, start_frame=0, at_pos=at[:, 0:3], at_quat=at[:, 3:7])
  write_state(robot, reference_state(jump), num_joints)
  obs = refresh(env)

  # Everything about where this clip is, before a step can auto-reset it out from under us.
  pinning = Pinning(
    motion_ids=jump.motion_ids.clone(),
    scales=jump.scales.clone(),
    anchor_pos=jump.anchor_pos.clone(),
    anchor_yaw=jump.anchor_yaw.clone(),
  )

  rollout = record(env, pool, JUMP, obs, cfg.jump_steps, torso)

  takeoff = jump.takeoff_steps_all[pinning.motion_ids]
  span = rollout.torso_z.shape[1]
  before_takeoff = torch.arange(span, device=env.device).unsqueeze(
    0
  ) < takeoff.unsqueeze(1)
  crouch = rollout.torso_z.masked_fill(~before_takeoff, float("inf")).argmin(dim=1)
  return rollout, crouch, pinning


##
# The window the two rollouts make between them.
##


@dataclass
class Handover:
  """One walk-to-jump question. Every tensor is in the arena's coordinates."""

  past: torch.Tensor
  """(N, past, S). The walk running into the hand-off."""

  hand_off: torch.Tensor
  """(N, S). The frame the walk was cut on: what the bridge is handed."""

  ahead: torch.Tensor
  """(N, S). Where the walk would have been `gap` frames later, had nothing interrupted
  it. Never handed to anything and never scored against; the jump was pinned on its xy and
  yaw, and it is kept so the report and the viewer can say how far ahead that put the
  target."""

  future: torch.Tensor
  """(N, window, S). The nominal jump from the entry onward: the second unmasked window,
  and what the bridge is aimed at."""

  nominal: torch.Tensor
  """(N, jump_steps + 1, S). The whole nominal jump rollout, crouch and all.

  Kept because it is what the viewer draws in the hole. The stand-and-settle before the
  entry is the stretch the bridge is being asked to replace, and showing it against the
  bridge replacing it is the only way to see whether the replacement was worth making."""

  gap: int
  """Delta t: frames from the hand-off to the entry, and the horizon the target was placed
  at. One number for every environment, so one viewer timeline serves all of them."""

  entry_frame: torch.Tensor
  """(N,). Which step of its own nominal rollout each environment is entered at: the
  crouch, less `--entry-lead`. Everything downstream reads this rather than the crouch,
  because it is the frame the bridge is aimed at, the frame the resume winds the clip to
  and the frame the ghost is wound around."""

  crouch_frame: torch.Tensor
  """(N,). Where the crouch itself was, kept so the report can say how far ahead of it the
  entry sits. Nothing else uses it."""

  motion_ids: torch.Tensor
  scales: torch.Tensor
  anchor_pos: torch.Tensor
  anchor_yaw: torch.Tensor
  """Where the nominal rollout had the clip pinned, kept so the resume can put it back
  exactly there. Env-local, as `JumpCommand` stores them."""

  valid: torch.Tensor
  """(N,). Environments whose two rollouts both survived long enough to be a question."""


def assemble(
  history: torch.Tensor,
  hand_off: torch.Tensor,
  ahead: torch.Tensor,
  still_walking: torch.Tensor,
  jump: Rollout,
  crouch: torch.Tensor,
  pinning: Pinning,
  cfg: Config,
) -> Handover:
  """Cut the window out of the two rollouts.

  There is no coordinate work here, and that is the point: the jump was rolled at the
  look-ahead, so its frames are already in the right place and the window is a slice.
  Where the slice starts is the one decision: the crouch is a landmark in the rollout, the
  entry is where `--entry-lead` puts the hand-over relative to it, and it is the entry the
  window opens on.

  The clamp at zero is for a lead longer than the run-up an environment had. That leaves it
  entering on the rollout's first frame, the clip's own opening stand, with a shorter
  effective lead than the ones around it. Reported rather than rejected, since a hand-over
  into the stand is still a hand-over, just not the one that was asked for.
  """
  device = history.device
  count = history.shape[0]
  index = torch.arange(count, device=device)

  entry = (crouch - cfg.entry_lead).clamp(min=0)

  last = jump.states.shape[1] - 1
  window = torch.stack(
    [jump.states[index, (entry + k).clamp(max=last)] for k in range(cfg.window)], dim=1
  )

  # A rollout that fell over is not a hand-over, and neither is one whose future window
  # ran off the end of what was recorded. The walk's side of that is `still_walking`,
  # which is aliveness at the look-ahead rather than at the end of its recording.
  room = (entry + cfg.window) < jump.states.shape[1]
  valid = still_walking & jump.survived & room

  return Handover(
    past=history,
    hand_off=hand_off,
    ahead=ahead,
    future=window,
    nominal=jump.states,
    gap=cfg.gap,
    entry_frame=entry,
    crouch_frame=crouch,
    motion_ids=pinning.motion_ids,
    scales=pinning.scales,
    anchor_pos=pinning.anchor_pos,
    anchor_yaw=pinning.anchor_yaw,
    valid=valid,
  )


##
# Crossing the hole.
##


def bridge_across(
  env: ManagerBasedRlEnv, policy, wrapped, window: Handover
) -> tuple[torch.Tensor, torch.Tensor]:
  """Drive the bridge from the hand-off to where the entry is supposed to be.

  Returns every frame it produced, back in the arena's coordinates, and the environments
  still standing when it got there. The window is injected rather than drawn from the
  corpus (`BridgeCommand.place_window`), which is the only reason this is possible at all:
  no corpus contains a walk that turns into a jump.

  The hole is handed over as unscored, which is what it is: no body performed this pair, so
  there is no in-between to be off. That is the same thing a spliced training window says
  about itself, so the reference is laid out the same way -- the arrival frame held across
  the hole, then the window -- and the environment reads it the same way. It is also why
  `lost_tracking` can be left switched on. Against a held arrival frame that termination
  becomes the stray bound, which ends an episode for a robot walking away from the entry
  and not for one taking a run-up at it.

  It runs for `gap + 1` steps, not `gap`. The reference frame the bridge is aimed at is the
  one after the last hole frame, so reaching it takes one action more than there are frames
  in the hole, and that last step is also the one on which the command latches
  `hand_off_score`.
  """
  command = env.command_manager.get_term("bridge")
  assert isinstance(command, BridgeCommand)
  robot: Entity = env.scene[ENTITY_NAME]
  device = env.device
  count = env.num_envs
  env_ids = torch.arange(count, device=device)
  gap = window.gap

  env.reset()

  # The reference, trimmed or padded to the fixed width the command expects.
  span = command.reference.shape[1]
  tail = torch.cat([window.future[:, :1].expand(-1, gap, -1), window.future], dim=1)
  if tail.shape[1] < span:
    tail = torch.cat([tail, tail[:, -1:].expand(-1, span - tail.shape[1], -1)], dim=1)
  command.place_window(
    env_ids,
    hand_off=window.hand_off,
    reference=tail[:, :span].contiguous(),
    gap=torch.full((count,), gap, dtype=torch.long, device=device),
    scored_hole=torch.zeros(count, dtype=torch.bool, device=device),
  )
  refresh(env)

  alive = torch.ones(count, dtype=torch.bool, device=device)
  frames = [read_state(robot)]
  for _ in range(gap + 1):
    with torch.inference_mode():
      action = policy(wrapped.get_observations())
    _, _, terminated, time_out, _ = env.step(action)
    alive = alive & ~(terminated | time_out)
    frames.append(torch.where(alive.unsqueeze(-1), read_state(robot), frames[-1]))

  # Out of the bridge's world and back into the arena's, here rather than at the call site.
  # `place_window` slid the window so the hand-off sat on this environment's origin, so
  # undoing it is the same shift backwards, and it has to reach every frame: the caller
  # wants the crossing to draw, not only the state it ended in.
  states = torch.stack(frames, dim=1)
  shift = window.hand_off[:, 0:2] - env.scene.env_origins[:, :2]
  states[:, :, 0:2] += shift.unsqueeze(1)
  return states, alive


##
# Handing the robot to the jump.
##


@dataclass
class Resume:
  """What the jump policy did with the state it was given."""

  states: torch.Tensor
  """(N, resume + 1, S), in the arena's coordinates."""

  fell: torch.Tensor
  goal_error: torch.Tensor
  landed: torch.Tensor


def resume_jump(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  window: Handover,
  arrival: torch.Tensor,
  cfg: Config,
  num_joints: int,
  keep_falling: bool = False,
) -> Resume:
  """Start the jump from `arrival`, with its clip back where the nominal rollout had it.

  `arrival` is a corpus-layout state per environment, in the arena's coordinates: the
  bridge's last frame, or the walk's hand-off when there is no bridge. The robot is put
  there and the clip is restored to the exact placement it had during `roll_jump`, wound
  to the entry. It is not re-pinned onto the robot, and that distinction is the whole
  measurement: re-pinning would slide the jump onto the arrival error and the composition
  would look perfect no matter what the bridge did.

  `keep_falling` is about what is recorded and changes nothing about what is scored. By
  default an environment that terminates is latched and its last good frame is held from
  then on, because `env.step` auto-resets it and reading the robot afterwards would put a
  teleport into a fresh episode in the middle of the states. A termination fires the
  moment a body is judged to be going over, though, well before it is on the floor, so
  what a recording gets out of that is a fall frozen halfway down. Set this and the
  auto-reset is switched off for the rollout, every frame is the robot as it actually
  was, and a body that goes over carries on to the ground and stays there.

  The numbers do not move either way. `fell`, `landed` and `goal_error` only ever
  accumulate while an environment is alive, and aliveness is latched at the first
  termination whether or not the world it happened in was reset out from under it.
  """
  robot: Entity = env.scene[ENTITY_NAME]
  device = env.device
  count = env.num_envs
  jump = jump_command(env)

  auto_reset = env.cfg.auto_reset
  env.cfg.auto_reset = auto_reset and not keep_falling
  env.reset()
  write_state(robot, arrival, num_joints)
  env.scene.write_data_to_sim()
  env.sim.forward()

  jump.motion_ids[:] = window.motion_ids
  jump.scales[:] = window.scales
  jump.anchor_pos[:] = window.anchor_pos
  jump.anchor_yaw[:] = window.anchor_yaw
  jump.time_steps[:] = window.entry_frame
  jump.motion_done[:] = False
  jump.update_relative_body_poses()
  obs = refresh(env)

  assignment = torch.full((count,), JUMP, dtype=torch.long, device=device)
  alive = torch.ones(count, dtype=torch.bool, device=device)
  fell = torch.zeros(count, dtype=torch.bool, device=device)
  landed = torch.zeros(count, dtype=torch.bool, device=device)
  error = torch.full((count,), float("nan"), device=device)
  frames = [read_state(robot)]

  for _ in range(cfg.resume_steps):
    with torch.inference_mode():
      action = pool.act(obs, assignment)
    obs, _, terminated, time_out, _ = env.step(action)
    if keep_falling:
      # With the auto-reset off the env wants a reset before the next step; the whole
      # point here is not to give it one, so the request is cleared and the fall runs on.
      env._manual_reset_pending.zero_()  # noqa: SLF001
    fell = fell | (terminated & alive)
    alive = alive & ~(terminated | time_out)
    state = read_state(robot)
    frames.append(
      state if keep_falling else torch.where(alive.unsqueeze(-1), state, frames[-1])
    )
    # The goal error is only meaningful once the reference has touched down, and only for
    # environments still standing when it did.
    settled = jump.has_landed & alive
    error = torch.where(settled, jump.goal_pos_error, error)
    landed = landed | settled

  env.cfg.auto_reset = auto_reset
  return Resume(
    states=torch.stack(frames, dim=1), fell=fell, goal_error=error, landed=landed
  )


##
# Reporting.
##


@dataclass
class Arm:
  """One way of getting from the walk to the jump, and how it went."""

  arrival: torch.Tensor
  """(N, S). The state the jump was started from."""

  resume: Resume
  crossed: torch.Tensor
  """(N,). Environments where getting to the hand-off worked at all. Always true for the
  baseline, which has nothing to get wrong; false for a bridge that fell in the hole."""


def report(window: Handover, arms: dict[str, Arm], num_joints: int) -> None:
  """One row per hand-over, one summary per arm.

  `arrival` is the root distance from the entry and `score` is the honest one, the same
  across-every-channel measure training gates the resume on (`arrival_error`), where the
  momentum and the joint pose show up.

  Both are read differently now that the target is placed ahead. Under the old ex post
  placement `arrival` flattered a direct hand-over, because the entry sat about where a
  walking robot already was and only the state was wrong. Placed a delta t downrange it
  no longer does: the baseline stands still through the hole, so the metre it did not
  travel is in its arrival distance, and the number now separates the two arms instead of
  compressing them. That is not the baseline being handicapped. It is the deficit
  architecture 0 always had, finally being asked about.
  """
  valid = window.valid
  count = valid.shape[0]
  target = window.future[:, 0]
  tolerances = BridgeCommandCfg(resampling_time_range=(1.0e9, 1.0e9))
  travel = (window.ahead[:, :2] - window.hand_off[:, :2]).norm(dim=-1)
  reach = (target[:, :2] - window.hand_off[:, :2]).norm(dim=-1)
  lead = window.crouch_frame - window.entry_frame
  seen = valid if bool(valid.any()) else torch.ones_like(valid)

  print(
    f"\nhole {window.gap} frames ({window.gap / CONTROL_HZ:.2f} s), "
    f"look-ahead {float(travel[seen].mean()):.2f} m, "
    f"target {float(reach[seen].mean()):.2f} m from the hand-off"
  )
  print(
    f"entry at rollout step {int(window.entry_frame.min())}"
    f"-{int(window.entry_frame.max())}, "
    f"{int(lead.min())}"
    f"{'' if int(lead.min()) == int(lead.max()) else f'-{int(lead.max())}'} "
    f"frames before the crouch, "
    f"{int(valid.sum())} of {count} hand-overs built"
  )

  for name, arm in arms.items():
    distance, error = arrival_error(arm.arrival, target, num_joints, tolerances)
    score = torch.exp(-error)
    resume = arm.resume
    print(f"\n{name}")
    print(
      f"{'env':>4} {'arrival':>9} {'speed':>9} {'score':>7} {'fell':>6} {'goal err':>10}"
    )
    for i in range(count):
      if not bool(valid[i]):
        print(f"{i:>4}   --  the two rollouts never made a usable window")
        continue
      if not bool(arm.crossed[i]):
        print(f"{i:>4}   --  fell before reaching the hand-off")
        continue
      goal = float(resume.goal_error[i])
      print(
        f"{i:>4} {float(distance[i]):>8.3f}m "
        f"{float(arm.arrival[i, 7:10].norm()):>7.2f}m/s "
        f"{float(score[i]):>7.3f} "
        f"{'yes' if bool(resume.fell[i]) else 'no':>6} "
        f"{'-' if goal != goal else f'{goal:.3f}m':>10}"
      )

    kept = valid & arm.crossed
    if not bool(kept.any()):
      print("     nothing reached the hand-off")
      continue
    goals = resume.goal_error[kept & ~resume.fell & resume.landed]
    print(
      f"     reached the hand-off {float(kept[valid].float().mean()):.0%}   "
      f"arrival {float(distance[kept].mean()):.3f} m   "
      f"score {float(score[kept].mean()):.3f}   "
      f"then fell {float(resume.fell[kept].float().mean()):.0%}   "
      f"goal error over survivors "
      f"{float(goals.mean()) if goals.numel() else float('nan'):.3f} m"
    )


##
# The viewer.
##


def build_tracks(
  window: Handover, bridge: torch.Tensor | None, resume: Resume, frames: int
) -> tuple[np.ndarray, np.ndarray]:
  """The two timelines the viewer plays: what happened, and what the jump alone would do.

  Both are (N, past + gap + frames, S), so the same frame index is the same moment in each.
  Where there is no bridge run, the solid track holds the hand-off for the length of the
  hole, which reads as the robot waiting, which is exactly what architecture 0 does with
  the time.

  The ghost is the nominal rollout, wound backwards from its own entry so that the entry
  lands on the frame the solid robot arrives at. Every frame of it is motion a policy
  actually produced. What it shows during the hole is the tail of the stand-and-settle the
  bridge is being asked to skip, and the entry is where the two tracks are supposed to
  meet: the distance between them at that instant is the arrival error, and it is visible
  rather than tabulated.

  The ghost steps forward by the look-ahead distance at the hand-off frame, because that
  is where the jump was pinned. It reads as the target standing a metre down the lane
  rather than on top of the robot, which is the ex ante arrangement drawn: the two are
  supposed to converge over the hole, not start together and be pulled apart.
  """
  device = window.future.device
  gap = window.gap
  crossing = (
    bridge[:, 1 : gap + 1]
    if bridge is not None
    else window.hand_off.unsqueeze(1).expand(-1, gap, -1)
  )
  solid = torch.cat([window.past, crossing, resume.states[:, :frames]], dim=1)

  last = window.nominal.shape[1] - 1
  reach = torch.arange(-gap, frames, device=device).unsqueeze(0)
  index = (window.entry_frame.unsqueeze(1) + reach).clamp(0, last)
  wound = torch.gather(
    window.nominal, 1, index.unsqueeze(-1).expand(-1, -1, window.nominal.shape[-1])
  )
  ghost = torch.cat([window.past, wound], dim=1)

  span = min(solid.shape[1], ghost.shape[1])
  return solid[:, :span].cpu().numpy(), ghost[:, :span].cpu().numpy()


class HandoverViewer:
  """Plays one assembled hand-over: green policies, red hole, solid composition."""

  def __init__(
    self,
    server,
    g1: G1,
    window: Handover,
    track: np.ndarray,
    ghost: np.ndarray,
    num_joints: int,
  ) -> None:
    self.server = server
    self.g1 = g1
    self.window = window
    self.track = track
    self.ghost = ghost
    self.num_joints = num_joints
    self.past = window.past.shape[1]
    self.gap = window.gap

    meshes = visual_meshes(g1.model)
    self.robot = Ghost(server, meshes, "robot", MODEL_COLOR, opacity=1.0)
    self.policy = Ghost(server, meshes, "policy", CONTEXT_COLOR, opacity=0.45)
    self.hole = Ghost(server, meshes, "hole", MASKED_COLOR, visible=False, opacity=0.45)
    server.scene.add_grid("/ground", width=16.0, height=16.0, cell_size=0.5)

    self.index = 0
    self.frame = 0
    self.origin = np.zeros(3, dtype=np.float32)
    self._syncing = False
    self._build_gui()
    self.load(0)

  def _build_gui(self) -> None:
    gui = self.server.gui
    with gui.add_folder("Hand-over"):
      self.sl_index = gui.add_slider(
        "Entry", min=0, max=max(self.track.shape[0] - 1, 1), step=1, initial_value=0
      )
      self.html = gui.add_html("")

      @self.sl_index.on_update
      def _(_) -> None:
        if not self._syncing:
          self.load(int(self.sl_index.value))

    with gui.add_folder("Playback"):
      self.cb_play = gui.add_checkbox("Play", initial_value=True)
      self.sl_speed = gui.add_slider(
        "Speed", min=0.1, max=2.0, step=0.1, initial_value=0.6
      )
      self.sl_frame = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
      self.cb_ghost = gui.add_checkbox("Show policies and hole", initial_value=True)

      @self.sl_frame.on_update
      def _(_) -> None:
        if not self._syncing:
          self.frame = int(self.sl_frame.value)

  def load(self, index: int) -> None:
    self.index = index % self.track.shape[0]
    self.origin = self.track[self.index, 0, :3].copy()
    self.origin[2] = 0.0
    self.frame = 0

    self._syncing = True
    self.sl_index.value = self.index
    self.sl_frame.max = self.track.shape[1] - 1
    self.sl_frame.value = 0
    self._syncing = False
    self._draw_path()
    self.render()

  def _draw_path(self) -> None:
    """The composition's own root path, green where a policy drove and red where the
    bridge did. The line is what makes an arrival error legible: a hand-over that lands
    short shows as a red stretch that stops before the green one starts."""
    root = self.track[self.index, :, :3] - self.origin
    segments = np.stack([root[:-1], root[1:]], axis=1)
    inside = np.array(
      [self.past <= t < self.past + self.gap for t in range(root.shape[0] - 1)]
    )
    per_segment = np.where(
      inside[:, None],
      np.array(MASKED_COLOR, dtype=np.uint8),
      np.array(CONTEXT_COLOR, dtype=np.uint8),
    )
    colors = np.repeat(per_segment[:, None, :], 2, axis=1).astype(np.uint8)
    self.server.scene.add_line_segments(
      "/path", points=segments.astype(np.float32), colors=colors, line_width=3.0
    )

  def _pose(self, ghost: Ghost, state: np.ndarray) -> None:
    data = self.g1.data
    data.qpos[0:3] = state[0:3] - self.origin
    data.qpos[3:7] = state[3:7]
    data.qpos[7:] = state[ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
    mujoco.mj_kinematics(self.g1.model, data)
    ghost.pose(data)

  def render(self) -> None:
    in_hole = self.past <= self.frame < self.past + self.gap
    with self.server.atomic():
      self._pose(self.robot, self.track[self.index, self.frame])
      if self.cb_ghost.value:
        self._pose(
          self.hole if in_hole else self.policy, self.ghost[self.index, self.frame]
        )
      self.policy.visible = self.cb_ghost.value and not in_hole
      self.hole.visible = self.cb_ghost.value and in_hole

    window = self.window
    target = window.future[self.index, 0, 0:3].cpu().numpy()
    speed = window.past[self.index, -1, 7:10].norm().item()
    travel = (
      (window.ahead[self.index, :2] - window.hand_off[self.index, :2]).norm().item()
    )
    stage = (
      "hole (bridge)" if in_hole else ("walk" if self.frame < self.past else "jump")
    )
    colour = "#e13a2d" if in_hole else "#3cc85a"
    self.html.content = (
      '<div style="font-size:0.85em;line-height:1.4;padding:0 1em 0.5em 1em;">'
      f"<b>Hand-over:</b> {self.index + 1} / {self.track.shape[0]}"
      f"<br/><b>Layout:</b> {self.past} + <span style='color:#e13a2d'>{self.gap}</span>"
      f" + {window.future.shape[1]} frames"
      f"<br/><b>Handed over at:</b> {speed:.2f} m/s"
      f"<br/><b>Look-ahead:</b> {travel:.2f} m in "
      f"{self.gap / CONTROL_HZ:.2f} s"
      f"<br/><b>Must reach:</b> "
      f"({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
      f"<br/><b>Now:</b> <span style='color:{colour}'>{stage}</span> "
      f"({self.frame + 1} / {self.track.shape[1]})"
      f"<br/><b>Usable:</b> {'yes' if bool(window.valid[self.index]) else 'no'}"
      "</div>"
    )

  def step(self, dt: float) -> None:
    if self.cb_play.value:
      advance = dt * CONTROL_HZ * self.sl_speed.value
      self.frame = int(round(self.frame + advance)) % self.track.shape[1]
      self._syncing = True
      self.sl_frame.value = self.frame
      self._syncing = False
    self.render()


##
# Putting it together.
##


def build_arena(num_envs: int, device: str) -> ManagerBasedRlEnv:
  """The world everything above is rolled in.

  The arena on a bare plane: the two skills are being rolled, not run down a corridor,
  and an obstacle in the way would be a second thing going wrong at once.

  Startup randomization is off. It is right for training and wrong here: the bridge runs
  in its own environment, which has none, and a robot whose centre of mass moved between
  the two is a state transfer that silently is not one.
  """
  arena_cfg = parkour_arena_env_cfg(obstacles=None)
  arena_cfg.scene.num_envs = num_envs
  for name in ("base_com", "encoder_bias", "foot_friction"):
    arena_cfg.events.pop(name, None)
  return ManagerBasedRlEnv(cfg=arena_cfg, device=device)


def build_handover(
  arena: ManagerBasedRlEnv, pool: SkillPool, cfg: Config, torso: int, num_joints: int
) -> Handover:
  """Steps 1 to 3: roll the walk, cut it, roll the jump from the look-ahead, assemble.

  The whole question, up to but not including anyone's attempt to cross it. Both the
  report below and the side-by-side recording in `skills/compare.py` start here, so
  neither can be comparing a different hand-over from the other.
  """
  print(
    f"\nrolling the walk, cut somewhere in {WINDOWS['walk'].interrupt_range}, "
    f"then {cfg.gap} frames past each cut for the look-ahead"
  )
  walk, cut = roll_walk(arena, pool, cfg, torso)
  history, hand_off = cut_walk(walk, cut)
  ahead, still_walking = look_ahead(walk, cut, cfg.gap)

  print(f"rolling the jump from the look-ahead, {cfg.jump_steps} steps")
  jump, crouch, pinning = roll_jump(arena, pool, cfg, torso, ahead, num_joints)
  print(
    f"crouch found at rollout step {int(crouch.min())}-{int(crouch.max())} "
    f"({float(crouch.float().mean()) / CONTROL_HZ:.2f} s in on average)"
  )
  window = assemble(history, hand_off, ahead, still_walking, jump, crouch, pinning, cfg)

  low, high = WindowCfg().gap_range
  if not low <= cfg.gap <= high:
    print(
      f"\n[note] --gap {cfg.gap} is outside the {low} to {high} the bridge was trained "
      f"on, so it is being asked for a hole of a duration it has never seen. Bring it "
      f"back inside, or widen --windows.gap-range and retrain."
    )
  natural = int(window.entry_frame.min()) - 1
  if natural > cfg.gap:
    print(
      f"\n[note] the nominal jump takes {natural} frames to reach the entry and the "
      f"bridge is given {cfg.gap}, so it is being asked to skip the stand-and-settle "
      f"rather than reproduce it. That is the intent: a robot arriving with momentum has "
      f"no reason to spend a second and a half standing up first."
    )
  return window


def cross_with_bridge(
  cfg: Config, window: Handover, device: str
) -> tuple[torch.Tensor, torch.Tensor, BridgeCommand]:
  """Step 4: run the bridge across the hole, in the environment it was trained in.

  Its own world, built and torn down here, since nothing outside the crossing has any
  use for it. What comes back is already in the arena's coordinates (`bridge_across`),
  plus the command term, which is read afterwards for the score it latched.
  """
  bridge_cfg = bridge_env_cfg(play=True)
  bridge_cfg.scene.num_envs = cfg.num_envs
  bridge_env = ManagerBasedRlEnv(cfg=bridge_cfg, device=device)

  checkpoint = find_checkpoint(
    cfg.checkpoint, load_rl_cfg(BRIDGE_TASK_ID).experiment_name
  )
  print(f"bridge: {checkpoint}")
  try:
    policy, wrapped = load_policy(bridge_env, checkpoint, device)
  except (RuntimeError, ValueError) as exc:
    raise SystemExit(
      f"That checkpoint does not fit this environment: {exc}\n"
      f"The bridge observation gained a third phase channel when the second window was "
      f"added to training (see bridge/env_cfg.py), so checkpoints from before that do "
      f"not load. Retrain, or rerun this with --no-bridge for the baseline alone."
    ) from exc

  states, standing = bridge_across(bridge_env, policy, wrapped, window)
  term = bridge_env.command_manager.get_term("bridge")
  assert isinstance(term, BridgeCommand)
  bridge_env.close()
  return states, standing, term


def main(cfg: Config) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  if cfg.gap < 1:
    raise SystemExit(
      "--gap is a delta t in frames and has to be at least 1: the target is placed where "
      "the walk would have been that many frames on, and a hole of zero frames has "
      "nowhere to place it."
    )

  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  arena = build_arena(cfg.num_envs, device)
  robot: Entity = arena.scene[ENTITY_NAME]
  num_joints = robot.data.joint_pos.shape[-1]
  torso = robot.body_names.index(ANCHOR_BODY)
  pool = build_pool(arena, device)

  window = build_handover(arena, pool, cfg, torso, num_joints)

  arms: dict[str, Arm] = {}
  bridge_states: torch.Tensor | None = None
  command: BridgeCommand | None = None

  if cfg.bridge:
    bridge_states, standing, command = cross_with_bridge(cfg, window, device)
    arrival = bridge_states[:, -1].clone()

    # A bridge that fell in the hole is judged on that, not on what the jump then made of
    # a state it should never have been given. The baseline is unaffected: it has no hole
    # to fall in, and folding this into the shared window would quietly disqualify the
    # comparison it exists to be measured against.
    arms["with bridge"] = Arm(
      arrival=arrival,
      resume=resume_jump(arena, pool, window, arrival, cfg, num_joints),
      crossed=standing,
    )

  # The baseline: no bridge at all, the walk's last state handed straight over. This is
  # architecture 0, and it is the thing a bridge has to be better than.
  arms["no bridge"] = Arm(
    arrival=window.hand_off,
    resume=resume_jump(arena, pool, window, window.hand_off, cfg, num_joints),
    crossed=torch.ones_like(window.valid),
  )

  report(window, arms, num_joints)
  if command is not None:
    score = command.hand_off_score[window.valid & arms["with bridge"].crossed]
    if score.numel():
      print(
        f"\nthe bridge's own latched hand-off score: {float(score.mean()):.3f} mean, "
        f"{float(score.min()):.3f} worst"
      )

  if cfg.view:
    import viser

    name = "with bridge" if "with bridge" in arms else "no bridge"
    track, ghost = build_tracks(window, bridge_states, arms[name].resume, cfg.window)
    server = viser.ViserServer(port=cfg.port, label="Walk to jump")
    viewer = HandoverViewer(server, G1(), window, track, ghost, num_joints)
    print(f"\nViewer at http://localhost:{cfg.port} -- Ctrl-C to quit.")
    last = time.time()
    try:
      while True:
        now = time.time()
        viewer.step(now - last)
        last = now
        time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
      print("\nShutting down.")

  arena.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
