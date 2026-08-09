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

2. Jump, from the hand-off. The jump policy is rolled from its nominal conditions, the
   clip's first frame, exactly the state its own training reset produces, but standing on
   the hand-off's spot on the floor and facing the hand-off's heading. That is what "the
   jump happens where the robot was when the signal came" means: only the ground position
   and the heading come from the walk, the pose and the momentum are the jump's own.

   The second unmasked window is the stretch of that rollout starting at the crouch, the
   frame where the torso is lowest before takeoff. Everything before the crouch is the
   policy standing up and settling, which is the second and a half a robot arriving at
   speed should not have to spend.

3. The hole is what lies between the hand-off and the crouch, and it is empty. No
   recording covers it because no body performed this pair, and producing something a body
   could perform there is the bridge's whole job. What the viewer draws in it is a straight
   interpolation: the question, not an answer.

4. Resume. The jump policy is handed the robot wherever the bridge left it, with its clip
   pinned exactly where the nominal rollout had it and wound to the crouch. The clip is
   never re-pinned onto the robot's actual arrival; doing that would slide the target onto
   whatever the bridge produced and erase the error this script exists to measure.

Everything lives in one world, the arena's, and never moves between coordinate systems.
The one exception is the bridge, which runs in its own environment and slides the hand-off
onto its origin; `bridge_across` shifts back on the way out.

The same hand-over is run again with no bridge at all, the walk's final state passed
straight to the jump, which is architecture 0. That column is the thing to beat.

    uv run python -m mjlab.tasks.skills.experiments.parkour.tests.walk2jump
    uv run python -m mjlab.tasks.skills.experiments.parkour.tests.walk2jump --view True
    uv run python -m mjlab.tasks.skills.experiments.parkour.tests.walk2jump \\
        --num-envs 32 --bridge False

The viewer colours it the way the corpus viewer does. Green is a policy: the walk running
into the hand-off, and the nominal jump from the crouch onward. Red is the hole. Solid
blue is what actually happened.
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
  slerp,
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

  gap: int | None = None
  """Frames the bridge gets, hand-off to crouch. Defaults to however long the nominal jump
  took to reach its own crouch, capped at the widest hole the bridge was trained on. The
  cap is not a workaround: the stand-and-settle it cuts off is the part a robot arriving
  with momentum has no reason to reproduce."""

  jump_distance: tuple[float, float] = (0.8, 1.6)
  """Range the per-environment jump goal is drawn from [m]."""

  walk_settle: int = 64
  """Steps of walking before any cut may be taken, so the gait is a gait."""

  jump_steps: int = 200
  """Steps of the nominal jump rollout. Long enough to cover the longest clip's crouch,
  its flight and its landing."""

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


def interpolate(start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
  """The straight line between two states, exclusive of both. (S,), (S,) -> (steps, S).

  Not a guess at what the bridge should do, and never scored against. It exists so the
  hole has something in it for the viewer to draw and for the environment's metrics to
  subtract from, and it is drawn in red for the same reason the corpus viewer draws a
  masked stretch in red: it is the question, not an answer.
  """
  if steps <= 0:
    return start.new_zeros((0, start.shape[-1]))
  alpha = torch.arange(1, steps + 1, device=start.device, dtype=start.dtype) / (
    steps + 1
  )
  out = start.unsqueeze(0) + (end - start).unsqueeze(0) * alpha.unsqueeze(-1)
  out[:, 3:7] = slerp(
    start[3:7].expand(steps, 4), end[3:7].expand(steps, 4), alpha.unsqueeze(-1)
  )
  return out


##
# The two rollouts.
##


@dataclass
class Rollout:
  """One policy driven for a while, recorded frame by frame.

  `states` is (N, T + 1, S): the state before any action, then one per action. `alive`
  marks the environments that never terminated, and `torso_z` is kept alongside because
  the crouch is defined by it and a root height cannot tell a crouch from a short robot.
  """

  states: torch.Tensor
  torso_z: torch.Tensor
  alive: torch.Tensor


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
  for _ in range(steps):
    with torch.inference_mode():
      action = pool.act(obs, assignment)
    obs, _, terminated, time_out, _ = env.step(action)
    alive = alive & ~(terminated | time_out)
    frames.append(torch.where(alive.unsqueeze(-1), read_state(robot), frames[-1]))
    heights.append(
      torch.where(alive, robot.data.body_link_pos_w[:, torso, 2], heights[-1])
    )

  return Rollout(
    states=torch.stack(frames, dim=1),
    torso_z=torch.stack(heights, dim=1),
    alive=alive,
  )


def roll_walk(
  env: ManagerBasedRlEnv, pool: SkillPool, cfg: Config, torso: int
) -> tuple[Rollout, torch.Tensor]:
  """Walk down the plane, and pick where each environment gets interrupted.

  The cut is drawn from the experiment's own interrupt range for the walk, which is what
  every bridging architecture here samples hand-overs from. A cut is a moment in a gait,
  not a moment in time: half of them catch a foot planted and half catch one in the air,
  and a bridge that only ever leaves from one of those has not been tested.
  """
  low, high = WINDOWS["walk"].interrupt_range
  low = max(low, cfg.walk_settle, WindowCfg().past)
  cut = torch.randint(low, high + 1, (env.num_envs,), device=env.device)

  obs, _ = env.reset()
  twist = env.command_manager.get_term(TWIST_COMMAND)
  assert isinstance(twist, UniformVelocityCommand)
  twist.vel_command_b[:, 0] = WALK_SPEED
  twist.vel_command_b[:, 1] = 0.0

  return record(env, pool, WALK, obs, int(cut.max()), torso), cut


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


def roll_jump(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  cfg: Config,
  torso: int,
  hand_off: torch.Tensor,
  num_joints: int,
) -> tuple[Rollout, torch.Tensor]:
  """Roll the jump from its own initiation set, standing where the walk handed over.

  Nominal means nominal in everything except placement: the robot is written onto the
  clip's first frame, which is the state this policy's training reset produces and the
  only one it can be assumed to be good from. What is not nominal is where that happens.
  The clip is pinned to the hand-off's horizontal position and heading first, so the whole
  jump plays out from the spot the walk was interrupted on rather than from the origin.
  That is the question being asked: what a jump started at this instant would have done.

  Note that only the hand-off's xy and yaw are used. Its joint angles, its height and its
  momentum are the walk's, and reproducing them is the bridge's job, not the reference's.

  What is recorded is the robot, not the clip. A reference is what the policy was aiming
  at; the window has to be a stretch of motion that actually happened.

  The crouch is the lowest the torso gets before the clip leaves the ground. Searching
  only up to takeoff is what keeps it a crouch: a body in flight passes through lower
  torso heights on the way down, and the bottom of a landing is not somewhere to hand a
  robot over to.
  """
  robot: Entity = env.scene[ENTITY_NAME]
  jump = jump_command(env)
  env_ids = torch.arange(env.num_envs, device=env.device)

  env.reset()
  distances = torch.empty(env.num_envs, device=env.device).uniform_(*cfg.jump_distance)
  # Which clip each environment jumps, and how far it is stretched.
  jump.apply_goals(env_ids, distances)
  # Then move the clip to the hand-off, and the robot onto the clip.
  jump.anchor_to_robot(
    env_ids, start_frame=0, at_pos=hand_off[:, 0:3], at_quat=hand_off[:, 3:7]
  )
  write_state(robot, reference_state(jump), num_joints)
  obs = refresh(env)

  rollout = record(env, pool, JUMP, obs, cfg.jump_steps, torso)

  takeoff = jump.takeoff_steps_all[jump.motion_ids]
  span = rollout.torso_z.shape[1]
  before_takeoff = torch.arange(span, device=env.device).unsqueeze(
    0
  ) < takeoff.unsqueeze(1)
  crouch = rollout.torso_z.masked_fill(~before_takeoff, float("inf")).argmin(dim=1)
  return rollout, crouch


##
# The window the two rollouts make between them.
##


@dataclass
class Handover:
  """One walk-to-jump question. Every tensor is in the arena's coordinates."""

  past: torch.Tensor
  """(N, past, S). The walk running into the hand-off."""

  hand_off: torch.Tensor
  """(N, S). The last frame the walk produced: what the bridge is handed."""

  hole: torch.Tensor
  """(N, gap, S). Straight interpolation across the hole. Not motion; see `interpolate`."""

  future: torch.Tensor
  """(N, future, S). The nominal jump from the crouch onward."""

  gap: int
  """Frames from the hand-off to the crouch, shared by every environment so one viewer
  timeline serves all of them."""

  crouch_frame: torch.Tensor
  """(N,). Which frame of its own clip each environment's crouch is."""

  motion_ids: torch.Tensor
  scales: torch.Tensor
  anchor_pos: torch.Tensor
  anchor_yaw: torch.Tensor
  """Where the nominal rollout had the clip pinned, kept so the resume can put it back
  exactly there. Env-local, as `JumpCommand` stores them."""

  valid: torch.Tensor
  """(N,). Environments whose two rollouts both survived long enough to be a question."""


def assemble(
  walk: Rollout,
  history: torch.Tensor,
  hand_off: torch.Tensor,
  jump: Rollout,
  crouch: torch.Tensor,
  command: JumpCommand,
  cfg: Config,
  max_gap: int,
) -> Handover:
  """Cut the window out of the two rollouts.

  There is no coordinate work here, and that is the point: the jump was rolled at the
  hand-off, so its frames are already in the right place and the window is a slice.
  """
  device = walk.states.device
  count = walk.states.shape[0]
  future = WindowCfg().future
  index = torch.arange(count, device=device)

  last = jump.states.shape[1] - 1
  window = torch.stack(
    [jump.states[index, (crouch + k).clamp(max=last)] for k in range(future)], dim=1
  )

  gap = cfg.gap if cfg.gap is not None else min(int(crouch.min()) - 1, max_gap)
  gap = max(int(gap), 1)
  hole = torch.stack(
    [interpolate(hand_off[i], window[i, 0], gap) for i in range(count)], dim=0
  )

  # A rollout that fell over is not a hand-over, and neither is one whose future window
  # ran off the end of what was recorded.
  room = (crouch + future) < jump.states.shape[1]
  valid = walk.alive & jump.alive & room

  return Handover(
    past=history,
    hand_off=hand_off,
    hole=hole,
    future=window,
    gap=gap,
    crouch_frame=crouch,
    motion_ids=command.motion_ids.clone(),
    scales=command.scales.clone(),
    anchor_pos=command.anchor_pos.clone(),
    anchor_yaw=command.anchor_yaw.clone(),
    valid=valid,
  )


##
# Crossing the hole.
##


def bridge_across(
  env: ManagerBasedRlEnv, policy, wrapped, window: Handover
) -> tuple[torch.Tensor, torch.Tensor]:
  """Drive the bridge from the hand-off to where the crouch is supposed to be.

  Returns every frame it produced, in the bridge environment's own coordinates, and the
  environments still standing when it got there. The window is injected rather than drawn
  from the corpus (`BridgeCommand.place_window`), which is the only reason this is possible
  at all: no corpus contains a walk that turns into a jump.

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

  env.reset()

  # The reference this run needs, padded out to the fixed width the command expects.
  span = command.reference.shape[1]
  tail = torch.cat([window.hole, window.future], dim=1)
  if tail.shape[1] < span:
    pad = tail[:, -1:].expand(-1, span - tail.shape[1], -1)
    tail = torch.cat([tail, pad], dim=1)
  command.place_window(
    env_ids,
    hand_off=window.hand_off,
    reference=tail[:, :span].contiguous(),
    gap=torch.full((count,), window.gap, dtype=torch.long, device=device),
  )
  refresh(env)

  alive = torch.ones(count, dtype=torch.bool, device=device)
  frames = [read_state(robot)]
  for _ in range(window.gap + 1):
    with torch.inference_mode():
      action = policy(wrapped.get_observations())
    _, _, terminated, time_out, _ = env.step(action)
    alive = alive & ~(terminated | time_out)
    frames.append(torch.where(alive.unsqueeze(-1), read_state(robot), frames[-1]))

  return torch.stack(frames, dim=1), alive


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
) -> Resume:
  """Start the jump from `arrival`, with its clip back where the nominal rollout had it.

  `arrival` is a corpus-layout state per environment, in the arena's coordinates: the
  bridge's last frame, or the walk's hand-off when there is no bridge. The robot is put
  there and the clip is restored to the exact placement it had during `roll_jump`, wound
  to the crouch. It is not re-pinned onto the robot, and that distinction is the whole
  measurement: re-pinning would slide the jump onto the arrival error and the composition
  would look perfect no matter what the bridge did.
  """
  robot: Entity = env.scene[ENTITY_NAME]
  device = env.device
  count = env.num_envs
  jump = jump_command(env)

  env.reset()
  write_state(robot, arrival, num_joints)
  env.scene.write_data_to_sim()
  env.sim.forward()

  jump.motion_ids[:] = window.motion_ids
  jump.scales[:] = window.scales
  jump.anchor_pos[:] = window.anchor_pos
  jump.anchor_yaw[:] = window.anchor_yaw
  jump.time_steps[:] = window.crouch_frame
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
    fell = fell | (terminated & alive)
    alive = alive & ~(terminated | time_out)
    frames.append(torch.where(alive.unsqueeze(-1), read_state(robot), frames[-1]))
    # The goal error is only meaningful once the reference has touched down, and only for
    # environments still standing when it did.
    settled = jump.has_landed & alive
    error = torch.where(settled, jump.goal_pos_error, error)
    landed = landed | settled

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

  `arrival` is the root distance from the crouch, and it is the number that flatters a
  direct hand-over most: a walking robot is already about where the crouch is, it is just
  not in it. `score` is the honest one, the same across-every-channel measure training
  gates the resume on (`arrival_error`), and it is where the momentum and the joint pose
  show up.
  """
  valid = window.valid
  count = valid.shape[0]
  target = window.future[:, 0]
  tolerances = BridgeCommandCfg(resampling_time_range=(1.0e9, 1.0e9))

  print(
    f"\nhole {window.gap} frames ({window.gap / CONTROL_HZ:.2f} s), "
    f"crouch at clip frame {int(window.crouch_frame.min())}"
    f"-{int(window.crouch_frame.max())}, "
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
  """The two timelines the viewer plays: what happened, and what the policies alone did.

  Both are (N, past + gap + frames, S), so the same frame index is the same moment in each.
  Where there is no bridge run, the solid track holds the hand-off for the length of the
  hole, which reads as the robot waiting, which is exactly what architecture 0 does with
  the time.
  """
  crossing = (
    bridge[:, 1 : window.gap + 1]
    if bridge is not None
    else window.hand_off.unsqueeze(1).expand(-1, window.gap, -1)
  )
  # The resume's first frame is the arrival, which is the same moment the ghost shows the
  # crouch. Lining those up is what makes the arrival error visible as a gap between the
  # solid robot and the green one rather than as a lag.
  solid = torch.cat([window.past, crossing, resume.states[:, :frames]], dim=1)
  ghost = torch.cat([window.past, window.hole, window.future[:, :frames]], dim=1)
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


def main(cfg: Config) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # The arena on a bare plane: the two skills are being rolled, not run down a corridor,
  # and an obstacle in the way would be a second thing going wrong at once.
  arena_cfg = parkour_arena_env_cfg(obstacles=None)
  arena_cfg.scene.num_envs = cfg.num_envs
  # Startup randomization off. It is right for training and wrong here: the bridge runs in
  # its own environment, which has none, and a robot whose centre of mass moved between
  # the two is a state transfer that silently is not one.
  for name in ("base_com", "encoder_bias", "foot_friction"):
    arena_cfg.events.pop(name, None)
  arena = ManagerBasedRlEnv(cfg=arena_cfg, device=device)

  robot: Entity = arena.scene[ENTITY_NAME]
  num_joints = robot.data.joint_pos.shape[-1]
  torso = robot.body_names.index(ANCHOR_BODY)
  pool = build_pool(arena, device)

  print(f"\nrolling the walk, cut somewhere in {WINDOWS['walk'].interrupt_range}")
  walk, cut = roll_walk(arena, pool, cfg, torso)
  history, hand_off = cut_walk(walk, cut)

  print(f"rolling the jump from the hand-off, {cfg.jump_steps} steps")
  jump, crouch = roll_jump(arena, pool, cfg, torso, hand_off, num_joints)
  print(
    f"crouch found at clip frame {int(crouch.min())}-{int(crouch.max())} "
    f"({float(crouch.float().mean()) / CONTROL_HZ:.2f} s in on average)"
  )
  max_gap = WindowCfg().gap_range[1]
  window = assemble(
    walk, history, hand_off, jump, crouch, jump_command(arena), cfg, max_gap
  )
  natural = int(crouch.min()) - 1
  if cfg.gap is None and natural > max_gap:
    print(
      f"\n[note] the nominal jump takes {natural} frames to reach its crouch, and the "
      f"bridge was trained on holes of at most {max_gap}. The hole is capped at "
      f"{window.gap}, which asks the bridge to skip the stand-and-settle rather than "
      f"reproduce it. Pass --gap to override, or widen --windows.gap-range and retrain "
      f"to ask for the whole thing."
    )

  arms: dict[str, Arm] = {}
  bridge_states: torch.Tensor | None = None
  command: BridgeCommand | None = None

  if cfg.bridge:
    bridge_cfg = bridge_env_cfg(play=True)
    bridge_cfg.scene.num_envs = cfg.num_envs
    # The hole holds an interpolation rather than a recording (see `interpolate`), so
    # ending an episode for straying from it would be ending it for disagreeing with a
    # straight line. Falling over is still a failure and is still caught.
    bridge_cfg.terminations.pop("lost_tracking", None)
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

    bridge_states, standing = bridge_across(bridge_env, policy, wrapped, window)
    term = bridge_env.command_manager.get_term("bridge")
    assert isinstance(term, BridgeCommand)
    command = term

    # Out of the bridge's world and back into the arena's: the bridge slid the window so
    # the hand-off sat on its own origin, so undoing that is the same shift backwards.
    shift = window.hand_off[:, 0:2] - bridge_env.scene.env_origins[:, :2]
    arrival = bridge_states[:, -1].clone()
    arrival[:, 0:2] += shift

    # A bridge that fell in the hole is judged on that, not on what the jump then made of
    # a state it should never have been given. The baseline is unaffected: it has no hole
    # to fall in, and folding this into the shared window would quietly disqualify the
    # comparison it exists to be measured against.
    arms["with bridge"] = Arm(
      arrival=arrival,
      resume=resume_jump(arena, pool, window, arrival, cfg, num_joints),
      crossed=standing,
    )
    bridge_env.close()

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
    track, ghost = build_tracks(
      window, bridge_states, arms[name].resume, WindowCfg().future
    )
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
