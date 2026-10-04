"""Walk, bridge, jump, bridge, walk, bridge, then kick without teleporting."""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor
from mjlab.tasks.bridging.bridges.imitation.command import (
  CHANNELS,
  channel_errors,
  upper_body_mask,
)
from mjlab.tasks.bridging.bridges.interface import Bridge
from mjlab.tasks.bridging.config.g1 import selector as resume
from mjlab.tasks.bridging.config.g1.demos.soccer.arena import JUMP, KICK
from mjlab.tasks.bridging.config.g1.demos.soccer.config import Handoff, Settings
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.bridging.config.g1.skills.kick import mdp as kick_mdp
from mjlab.tasks.bridging.config.g1.skills.kick.mdp import KickCommand
from mjlab.tasks.bridging.config.t1.demos.sokoban.controller import place_entry
from mjlab.tasks.bridging.selector.table import Entry, EntryTable
from mjlab.tasks.bridging.tests.stage import fresh_obs, state
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply
from mjlab.viewer.debug_visualizer import DebugVisualizer

Policy = Callable[[TensorDict], torch.Tensor]


def rotate_xy(vector: torch.Tensor, yaw: float) -> torch.Tensor:
  x, y = vector.unbind(-1)
  return torch.stack(
    (math.cos(yaw) * x - math.sin(yaw) * y, math.sin(yaw) * x + math.cos(yaw) * y), -1
  )


def gate(
  position: torch.Tensor,
  entry: torch.Tensor,
  yaw: float,
  distance: float,
  lateral_tolerance: float,
) -> bool:
  """Trigger when the robot crosses the approach plane inside the entry lane."""
  delta = rotate_xy(entry - position, -yaw)
  return float(delta[0]) <= distance and abs(float(delta[1])) <= lateral_tolerance


def crossed_goal(
  previous: torch.Tensor, current: torch.Tensor, settings: Settings
) -> bool:
  scene = settings.scene
  center = current.new_tensor([*scene.goal_position, 0.0])
  yaw = math.radians(scene.goal_heading_degrees)
  before = rotate_xy((previous - center)[:2], -yaw)
  after = rotate_xy((current - center)[:2], -yaw)
  if not float(before[0]) < 0 <= float(after[0]):
    return False
  fraction = -before[0] / (after[0] - before[0])
  lateral = before[1] + fraction * (after[1] - before[1])
  height = previous[2] + fraction * (current[2] - previous[2])
  return (
    abs(float(lateral)) < scene.goal_width / 2 and 0 < float(height) < scene.goal_height
  )


def select_entry(table: EntryTable, skill: str, index: int) -> Entry:
  entries = table.of(skill)
  if not 0 <= index < len(entries):
    raise ValueError(
      f"{skill} entry index {index} is outside the {len(entries)} recorded entries"
    )
  return entries[index]


def center_jump(
  env: ManagerBasedRlEnv, command: JumpCommand, entry_frame: int, center: torch.Tensor
) -> torch.Tensor:
  """Center the airborne span and return takeoff, landing and exit XY."""
  motion_id = int(command.motion_ids[0])
  takeoff = int(command.takeoff_steps_all[motion_id])
  landing = int(command.landing_step[0])
  length = int(command.motion.time_step_total_per_motion[motion_id])
  if not 0 <= entry_frame < takeoff < landing < length:
    raise ValueError("Jump entry needs valid takeoff and landing landmarks after it")
  roots = []
  try:
    for frame in (takeoff, landing, length - 1):
      resume.rewind(env, frame, JUMP)
      roots.append(command.body_pos_w[0, 0, :2].clone())
  finally:
    resume.rewind(env, entry_frame, JUMP)
  points = torch.stack(roots)
  if (
    not bool(torch.isfinite(points).all())
    or float((points[1] - points[0]).norm()) < 1e-6
  ):
    raise ValueError("Jump reference has no finite horizontal flight span")
  shift = center - points[:2].mean(0)
  command.anchor_pos += shift
  command.update_relative_body_poses()
  return points + shift


def recorded_target(
  env: ManagerBasedRlEnv, entry: Entry, command: JumpCommand, future: int
) -> torch.Tensor:
  sequence = (
    np.concatenate((entry.state[None], entry.future_states), axis=0)
    if entry.future_states is not None
    else entry.state[None]
  )
  if len(sequence) < future or (
    future > 1
    and (entry.future_mask is None or not entry.future_mask[: future - 1].all())
  ):
    raise ValueError(
      f"{entry.skill} frame {entry.frame} needs {future} consecutive selector states; rebuild the selector"
    )
  states = torch.as_tensor(sequence[:future], device=env.device, dtype=torch.float32)
  reference = torch.as_tensor(entry.reference, device=env.device).expand(
    len(states), -1
  )
  placed = torch.cat((command.body_pos_w[:, 0], command.body_quat_w[:, 0]), -1).expand(
    len(states), -1
  )
  result = resume.place(states, reference, placed)[None]
  if not bool(torch.isfinite(result).all()) or result.shape[-1] != state(env).shape[-1]:
    raise ValueError(f"Invalid G1 {entry.skill} selector state window")
  return result


@dataclass(frozen=True)
class EntryPoint:
  target: torch.Tensor
  yaw: float

  @property
  def xy(self) -> torch.Tensor:
    return self.target[0, 0, :2]


class Controller:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    policies: Mapping[str, Policy],
    table: EntryTable,
    runtime: Bridge,
    settings: Settings,
    history_length: int = 30,
    future: int = 1,
    tolerances: torch.Tensor | None = None,
  ) -> None:
    if env.num_envs != 1:
      raise ValueError("Soccer controller requires one environment")
    if not math.isclose(table.fps, 1.0 / env.step_dt):
      raise ValueError("Selector and environment control frequencies differ")
    self.env, self.policies, self.runtime, self.settings = (
      env,
      policies,
      runtime,
      settings,
    )
    self.future = future
    self.history: deque[torch.Tensor] = deque(maxlen=history_length)
    twist = env.command_manager.get_term("twist")
    jump, kick = env.command_manager.get_term(JUMP), env.command_manager.get_term(KICK)
    if (
      not isinstance(twist, UniformVelocityCommand)
      or not isinstance(jump, JumpCommand)
      or not isinstance(kick, KickCommand)
    ):
      raise TypeError(
        "Soccer requires velocity, jump tracking and kick tracking commands"
      )
    self.twist, self.jump, self.kick = twist, jump, kick
    self.entries = {
      "jump": select_entry(table, "jump", settings.walk_to_jump.entry_index),
      "kick": select_entry(table, "kick", settings.walk_to_kick.entry_index),
    }
    self.tolerances = tolerances
    self.entry_ghost: mujoco.MjModel | None = None
    self.feet, _ = env.scene["robot"].find_bodies(
      ["left_ankle_roll_link", "right_ankle_roll_link"]
    )
    self.upper_body = upper_body_mask(tuple(env.scene["robot"].joint_names), env.device)
    self.reset()

  @property
  def done(self) -> bool:
    return self.phase in {"done", "failed"}

  @property
  def status(self) -> str:
    speed = float(self.env.scene["robot"].data.root_link_lin_vel_w[0, :2].norm())
    active = f"bridge ({self.transition})" if self.phase == "bridge" else self.skill
    return f"Active skill: {active} | Phase: {self.phase} | Speed: {speed:.2f} m/s\n{self.reason} | Goal: {'scored' if self.scored else 'pending'}"

  def _xy(self, position: tuple[float, float]) -> torch.Tensor:
    return (
      torch.tensor(position, device=self.env.device, dtype=torch.float32)
      + self.env.scene.env_origins[0, :2]
    )

  def _motion_point(self, skill: str, handoff: Handoff) -> EntryPoint:
    command = self.jump if skill == "jump" else self.kick
    entry = self.entries[skill]
    resume.prepare(self.env, entry, JUMP if skill == "jump" else KICK)
    ids = torch.zeros(1, device=self.env.device, dtype=torch.long)
    scene = self.settings.scene
    if handoff.heading_degrees is None:
      delta = (
        self._xy(scene.goal_position)
        - self.env.scene["ball"].data.root_link_pos_w[0, :2]
      )
      yaw = math.atan2(float(delta[1]), float(delta[0]))
    else:
      yaw = math.radians(handoff.heading_degrees)
    quaternion = torch.tensor(
      [[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], device=self.env.device
    )
    command.anchor_to_robot(
      ids, entry.frame, at_pos=state(self.env)[:, :3], at_quat=quaternion
    )
    if skill == "jump":
      self.jump_takeoff_xy, self.jump_landing_xy, self.jump_end_xy = center_jump(
        self.env, command, entry.frame, self.fallen_xy
      )
    else:
      assert isinstance(command, KickCommand)
      if entry.frame > int(command.motion_reset_limit[command.motion_ids[0]]):
        raise ValueError("Kick entry starts too late, after the safe pre-strike frame")
      # Align the measured strike direction, then place its ball target on the real ball
      direction = command.kick_direction[0]
      command.anchor_yaw[:] = yaw - math.atan2(float(direction[1]), float(direction[0]))
      ball_local = quat_apply(command.anchor_yaw_quat, command.ball_target)[0, :2]
      wanted = self.env.scene["ball"].data.root_link_pos_w[0, :2]
      command.anchor_pos[:] = wanted - self.env.scene.env_origins[0, :2] - ball_local
    command.update_relative_body_poses()
    return EntryPoint(recorded_target(self.env, entry, command, self.future), yaw)

  def _walk_point(self) -> EntryPoint:
    handoff = self.settings.jump_to_walk
    path = Path(self.settings.policies.locomotion_entry_path)
    if not path.exists():
      raise FileNotFoundError(
        f"No locomotion entry at {path}; run soccer.record with this config first"
      )
    with np.load(path, allow_pickle=False) as data:
      if tuple(data["joint_names"]) != tuple(
        self.env.scene["robot"].joint_names
      ) or not np.isclose(float(data["fps"]), 1.0 / self.env.step_dt):
        raise ValueError(
          "Locomotion recording uses a different joint layout or control frequency"
        )
      if not np.isclose(float(data["speed"]), self.settings.policies.speed_after_jump):
        raise ValueError(
          "Locomotion entry speed differs from speed_after_jump; record it again"
        )
      clip = torch.as_tensor(
        data["states"], device=self.env.device, dtype=torch.float32
      )
    clip = clip[handoff.entry_index : handoff.entry_index + self.future]
    if clip.shape != (self.future, state(self.env).shape[-1]) or not bool(
      torch.isfinite(clip).all()
    ):
      raise ValueError(
        "Locomotion entry must cover the requested frame and planner future window"
      )
    yaw = (
      math.radians(handoff.heading_degrees)
      if handoff.heading_degrees is not None
      else self.points["walk_to_jump"].yaw
    )
    target = place_entry(clip, self.jump_end_xy, yaw)
    target[:, :, 2] += self.env.scene.env_origins[0, 2]
    return EntryPoint(target, yaw)

  def reset(self) -> None:
    self.fallen_xy = self.env.scene["fallen_robot"].data.root_link_pos_w[0, :2].clone()
    self.history.clear()
    self.runtime.reset()
    self.phase, self.skill, self.reason = (
      "approach_jump",
      "walk",
      "Approach fallen robot",
    )
    self.transition = "walk_to_jump"
    self.elapsed = self.bridge_elapsed = 0.0
    self.airborne, self.scored = False, False
    self.kick_succeeded = False
    self.handoffs: list[dict] = []
    self.points = {
      "walk_to_jump": self._motion_point("jump", self.settings.walk_to_jump),
    }
    self.points["jump_to_walk"] = self._walk_point()
    self.points["walk_to_kick"] = self._motion_point("kick", self.settings.walk_to_kick)
    self.previous_ball = (
      self.env.scene["ball"].data.root_link_pos_w[0].clone()
      - self.env.scene.env_origins[0]
    )
    kick_mdp.reset_kick_phase(self.env)
    self.twist.vel_command_b.zero_()
    for policy in self.policies.values():
      reset = getattr(policy, "reset", None)
      if callable(reset):
        reset()

  def _fail(self, reason: str) -> torch.Tensor:
    self.phase, self.reason = "failed", reason
    self.twist.vel_command_b.zero_()
    return torch.zeros_like(self.env.action_manager.action)

  def _set_phase(self, phase: str, skill: str, reason: str) -> None:
    self.phase, self.skill, self.reason, self.elapsed = phase, skill, reason, 0.0

  def _start_bridge(self, transition: str) -> None:
    self.transition = transition
    self.runtime.reset()
    self.bridge_elapsed = 0.0
    self._set_phase("bridge", self.skill, transition)
    self.twist.vel_command_b.zero_()

  def _bridge(self) -> torch.Tensor:
    handoff = self.settings.handoffs[self.transition]
    point = self.points[self.transition]
    output = self.runtime(
      torch.stack(tuple(self.history), 1),
      point.target,
      point.target.new_tensor([handoff.duration]),
    )
    self.bridge_elapsed += self.env.step_dt
    if bool(output.handoff[0]):
      errors = channel_errors(state(self.env), point.target[:, 0], self.upper_body)[0]
      within = bool(output.within_endpoint_box[0])
      if self.tolerances is not None:
        within = bool((errors <= self.tolerances * handoff.tolerance_scale).all())
      self.handoffs.append(
        {
          "transition": self.transition,
          "distance_at_start": self.distance_at_start,
          "duration": handoff.duration,
          "within_endpoint": within,
          "errors": dict(zip(CHANNELS, errors.tolist(), strict=True)),
          "speed": float(state(self.env)[0, 7:9].norm()),
        }
      )
      if type(self.runtime) is not Bridge and handoff.require_endpoint and not within:
        return self._fail(f"{self.transition} missed entry tolerances")
      if self.transition == "walk_to_jump":
        resume.rewind(self.env, self.entries["jump"].frame, JUMP)
        self._set_phase("jump", "jump", "Clear fallen robot")
      elif self.transition == "jump_to_walk":
        self._set_phase("approach_kick", "walk", "Approach stationary ball")
      else:
        resume.rewind(self.env, self.entries["kick"].frame, KICK)
        self._set_phase("kick", "kick", "Kick towards goal")
      return self._action()
    if self.bridge_elapsed > handoff.duration + 2 * self.env.step_dt:
      return self._fail(f"{self.transition} bridge did not finish on time")
    return output.action

  def _approach(self, transition: str, speed: float) -> torch.Tensor:
    point, cfg = self.points[transition], self.settings.controller
    robot = self.env.scene["robot"]
    position, heading = (
      robot.data.root_link_pos_w[0, :2],
      float(robot.data.heading_w[0]),
    )
    delta = point.xy - position
    along = rotate_xy(delta, -point.yaw)
    error = math.atan2(math.sin(point.yaw - heading), math.cos(point.yaw - heading))
    handoff = self.settings.handoffs[transition]
    if float(along[0]) < -cfg.lateral_tolerance:
      return self._fail(f"Passed {transition} entry before alignment")
    if (
      gate(position, point.xy, point.yaw, handoff.start_distance, cfg.lateral_tolerance)
      and abs(error) <= cfg.heading_tolerance
    ):
      self.distance_at_start = float(delta.norm())
      self._start_bridge(transition)
      return self._bridge()
    desired = math.atan2(float(delta[1]), float(delta[0]))
    turn = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
    self.twist.vel_command_b[0] = self.twist.vel_command_b.new_tensor(
      [
        speed * max(0.0, math.cos(turn)),
        0.0,
        max(-cfg.turn_speed, min(cfg.turn_speed, 2 * turn)),
      ]
    )
    return self.policies["walk"](fresh_obs(self.env))

  def _action(self) -> torch.Tensor:
    if self.phase == "approach_jump":
      return self._approach("walk_to_jump", self.settings.policies.speed_before_jump)
    if self.phase == "approach_kick":
      return self._approach("walk_to_kick", self.settings.policies.speed_after_jump)
    return self.policies[self.skill](fresh_obs(self.env))

  @torch.no_grad()
  def __call__(self, obs) -> torch.Tensor:
    del obs
    actual = state(self.env).clone()
    if not bool(torch.isfinite(actual).all()):
      return self._fail("Robot state is not finite")
    if self.phase == "done":
      return self._action()
    if self.phase == "failed":
      return self.env.action_manager.action.clone()
    if not self.history:
      self.history.extend(actual for _ in range(self.history.maxlen or 1))
    else:
      self.history.append(actual)
    self.elapsed += self.env.step_dt
    cfg = self.settings.controller
    if float(actual[0, 2] - self.env.scene.env_origins[0, 2]) < cfg.fall_height:
      return self._fail("Robot fell")
    timeout = cfg.kick_timeout if self.phase == "kick" else cfg.phase_timeout
    if self.elapsed > timeout:
      return self._fail(f"{self.phase} timed out")
    # Keep inactive references at their entries while physics advances
    if self.phase != "jump":
      resume.rewind(self.env, self.entries["jump"].frame, JUMP)
    if self.phase != "kick":
      resume.rewind(self.env, self.entries["kick"].frame, KICK)
    ball = self.env.scene["ball"]
    if (
      self.phase != "kick"
      and float(
        (
          ball.data.root_link_pos_w[0, :2] - self._xy(self.settings.scene.ball_position)
        ).norm()
      )
      > cfg.ball_position_tolerance
    ):
      return self._fail("Ball moved before the kick")
    if self.phase == "bridge":
      return self._bridge()
    if self.phase == "jump":
      sensor = self.env.scene["feet_ground_contact"]
      assert isinstance(sensor, ContactSensor) and sensor.data.found is not None
      grounded = bool((sensor.data.found > 0).any())
      foot_height = (
        self.env.scene["robot"].data.body_link_pos_w[0, self.feet, 2]
        - self.env.scene.env_origins[0, 2]
      )
      self.airborne |= not grounded and float(foot_height.min()) > cfg.lift_height
      point = self.points["jump_to_walk"]
      position = actual[0, :2]
      clearance = rotate_xy(
        position - self.fallen_xy,
        self.points["walk_to_jump"].yaw * -1,
      )[0]
      ready = (
        self.airborne
        and float(clearance) >= cfg.clearance_distance
        and (grounded or not cfg.jump_exit_requires_landing)
      )
      if float(rotate_xy(point.xy - position, -point.yaw)[0]) < -cfg.lateral_tolerance:
        return self._fail("Passed jump_to_walk entry before a valid jump exit")
      if ready and gate(
        position,
        point.xy,
        point.yaw,
        self.settings.jump_to_walk.start_distance,
        cfg.lateral_tolerance,
      ):
        self.distance_at_start = float((position - point.xy).norm())
        self._start_bridge("jump_to_walk")
        return self._bridge()
    elif self.phase == "kick":
      current = ball.data.root_link_pos_w[0] - self.env.scene.env_origins[0]
      touched = bool(kick_mdp.phase(self.env, KICK).touched[0])
      speed = float(ball.data.root_link_lin_vel_w[0, :2].norm())
      self.scored |= touched and crossed_goal(
        self.previous_ball, current, self.settings
      )
      self.previous_ball = current.clone()
      self.kick_succeeded |= self.scored or (
        not cfg.require_goal and touched and speed >= cfg.launch_speed
      )
      if self.kick_succeeded:
        result = "Goal scored" if self.scored else "Ball launched"
        self.reason = f"{result}; finishing kick recovery"
        if bool(self.kick.motion_done[0]):
          self._set_phase("done", "kick", result)
    return self._action()

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    robot = self.env.scene["robot"]
    if self.entry_ghost is None:
      self.entry_ghost = copy.deepcopy(self.env.sim.mj_model)
      ghost = self.entry_ghost
      visible = np.zeros(ghost.ngeom, dtype=bool)
      visible[robot.indexing.geom_ids.cpu().numpy()] = True
      visible &= (ghost.geom_contype == 0) & (ghost.geom_conaffinity == 0)
      ghost.geom_rgba[:, 3] = 0
      ghost.geom_rgba[visible] = (0.2, 0.9, 0.8, 0.35)
    transition = {
      "jump": "jump_to_walk",
      "approach_kick": "walk_to_kick",
      "kick": "walk_to_kick",
      "done": "walk_to_kick",
    }.get(self.phase, self.transition)
    target = self.points[transition].target[0, 0].cpu().numpy()
    qpos = self.entry_ghost.qpos0.copy()
    qpos[robot.indexing.free_joint_q_adr.cpu().numpy()] = target[:7]
    joints = robot.indexing.joint_q_adr.cpu().numpy()
    qpos[joints] = target[13 : 13 + len(joints)]
    visualizer.add_ghost_mesh(
      qpos, self.entry_ghost, alpha=0.35, label=f"{transition} target"
    )
    for name, point in self.points.items():
      entry = point.target[0, 0, :3].cpu().numpy().copy()
      entry[2] = 0.04 + float(self.env.scene.env_origins[0, 2])
      start = entry.copy()
      start[:2] -= self.settings.handoffs[name].start_distance * np.array(
        [math.cos(point.yaw), math.sin(point.yaw)]
      )
      visualizer.add_sphere(entry, 0.06, (0.2, 0.9, 0.3, 0.8), label=f"{name} entry")
      visualizer.add_sphere(start, 0.045, (1.0, 0.6, 0.1, 0.8), label=f"{name} start")
      visualizer.add_arrow(start, entry, (1.0, 0.6, 0.1, 0.8), label=name)
