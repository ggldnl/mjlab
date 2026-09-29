"""Readable rule controller for the parkour demo."""

# pyright: reportPrivateImportUsage=false, reportArgumentType=false

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.bridges.diffusion.execution.runtime import DiffusionRuntime
from mjlab.tasks.bridging.bridges.imitation.command import score
from mjlab.tasks.bridging.bridges.interface import BridgeCommand
from mjlab.tasks.bridging.config.g1 import selector as resume
from mjlab.tasks.bridging.config.g1.demos.parkour.arena import (
  CLIMB_MOTION,
  JUMP_MOTION,
  Focus,
  obstacle_names,
)
from mjlab.tasks.bridging.config.g1.demos.parkour.course import (
  SHORT,
  TALL,
  ControllerCfg,
  Course,
  Obstacle,
  Settings,
  climb_box,
)
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.bridging.selector.table import Entry, EntryTable
from mjlab.tasks.bridging.tests.stage import BRIDGE, Policy, fresh_obs
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_from_euler_xyz

RULES = {TALL: "climb", SHORT: "jump"}
MOTIONS = {"jump": JUMP_MOTION, "climb": CLIMB_MOTION}
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def plan_lines(course: Course) -> list[str]:
  rows = [
    "| # | obstacle | skill |",
    "|---|---|---|",
    *(
      f"| {index} | {obstacle.kind} | {RULES[obstacle.kind]} |"
      for index, obstacle in enumerate(course)
    ),
  ]
  return [
    "rules: tall -> climb, short -> jump",
    "",
    *course.lines(),
    "",
    "plan:",
    *rows,
    "go_to_goal",
  ]


def _wrap(angle: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def _yaw(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat(yaw: torch.Tensor) -> torch.Tensor:
  zero = torch.zeros_like(yaw)
  return quat_from_euler_xyz(zero, zero, yaw)


def _rotate(vector: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  cos, sin = torch.cos(yaw), torch.sin(yaw)
  return torch.stack(
    (
      vector[:, 0] * cos - vector[:, 1] * sin,
      vector[:, 0] * sin + vector[:, 1] * cos,
    ),
    dim=-1,
  )


def _entry(entries: tuple[Entry, ...], frame: int) -> Entry:
  if not entries:
    raise ValueError("The selector has no entry for this skill")
  try:
    return next(entry for entry in entries if entry.frame == frame)
  except StopIteration as error:
    available = ", ".join(str(entry.frame) for entry in entries)
    raise ValueError(
      f"Frame {frame} is not a selector entry; available frames: {available}"
    ) from error


@dataclass(frozen=True)
class Approach:
  xy: torch.Tensor
  yaw: torch.Tensor
  anchor_yaw: torch.Tensor
  entry: Entry
  target: torch.Tensor


def _motion(env: ManagerBasedRlEnv, skill: str) -> JumpCommand:
  command = env.command_manager.get_term(MOTIONS[skill])
  if not isinstance(command, JumpCommand):
    raise TypeError(f"{skill} does not have a clip tracker")
  return command


def _place_reference(
  env: ManagerBasedRlEnv,
  command: JumpCommand,
  frame: int,
  wanted_xy: torch.Tensor,
  wanted_yaw: torch.Tensor,
  obstacle_pose: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  ids = torch.arange(env.num_envs, device=env.device)
  at_xy, at_yaw = wanted_xy.clone(), wanted_yaw.clone()
  box = climb_box()

  for _ in range(3 if obstacle_pose is not None else 1):
    at_pos = torch.cat((at_xy, env.scene.env_origins[:, 2:3]), dim=-1)
    command.anchor_to_robot(
      ids, start_frame=frame, at_pos=at_pos, at_quat=_quat(at_yaw)
    )
    if obstacle_pose is None:
      break
    obstacle_pos, obstacle_yaw = obstacle_pose
    clip_offset = torch.tensor(
      box.pos[:2], device=env.device, dtype=at_xy.dtype
    ).expand(env.num_envs, 2)
    clip_xy = (
      _rotate(clip_offset, command.anchor_yaw)
      + command.anchor_pos
      + env.scene.env_origins[:, :2]
    )
    at_yaw += _wrap(obstacle_yaw - (box.yaw + command.anchor_yaw))
    at_xy += obstacle_pos[:, :2] - clip_xy

  return command.body_pos_w[:, 0, :2], _yaw(command.body_quat_w[:, 0]), at_yaw


def solve_approach(
  env: ManagerBasedRlEnv,
  obstacle: Obstacle,
  obstacle_pos: torch.Tensor,
  obstacle_yaw: torch.Tensor,
  skill: str,
  entry: Entry,
  cfg: ControllerCfg,
) -> Approach:
  """Place the skill entry against one physical obstacle."""
  command_name = MOTIONS[skill]
  resume.prepare(env, entry, command_name)
  command = _motion(env, skill)

  if obstacle.kind == TALL:
    box = climb_box()
    offset = torch.tensor(
      box.pos[:2], device=env.device, dtype=obstacle_pos.dtype
    ).expand(env.num_envs, 2)
    wanted_yaw = _wrap(obstacle_yaw - box.yaw)
    wanted_xy = obstacle_pos[:, :2] - _rotate(offset, wanted_yaw)
    physical = obstacle_pos, obstacle_yaw
  else:
    offset = obstacle.length / 2.0 + cfg.hurdle_takeoff
    back = torch.stack((torch.cos(obstacle_yaw), torch.sin(obstacle_yaw)), dim=-1)
    wanted_xy = obstacle_pos[:, :2] - offset * back
    wanted_yaw = obstacle_yaw
    physical = None

  xy, yaw, anchor_yaw = _place_reference(
    env, command, entry.frame, wanted_xy, wanted_yaw, physical
  )
  return Approach(
    xy=xy,
    yaw=yaw,
    anchor_yaw=anchor_yaw,
    entry=entry,
    target=resume.target(env, entry, command_name).clone(),
  )


class Controller:
  """Run go_to, bridge and traverse for every obstacle, then reach the goal."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    policies: dict[str, Policy],
    table: EntryTable,
    runtime: DiffusionRuntime,
    history_length: int,
    course: Course,
    settings: Settings,
    focus: Focus,
  ) -> None:
    self.env = env
    self.policies = policies
    self.runtime = runtime
    self.history_length = history_length
    self.course = course
    self.settings = settings
    self.cfg = settings.controller
    self.automatic = True
    self.fire = False
    self.bridge_distance = dict(self.cfg.bridge_distance)
    self.duration_s = dict(self.cfg.bridge_duration_s)
    self.walk_speed = self.cfg.walk_speed
    self.focus = focus
    self.robot: Entity = env.scene["robot"]
    twist = env.command_manager.get_term("twist")
    if not isinstance(twist, UniformVelocityCommand):
      raise TypeError("Parkour requires the walk velocity command")
    self.twist = twist
    command = env.command_manager.get_term(BRIDGE)
    if not isinstance(command, BridgeCommand):
      raise TypeError("Parkour requires a bridge command")
    self.command = command
    self.command.tolerances *= self.cfg.capture_tolerance_scale
    self.entries = {
      "jump": _entry(table.of("jump"), settings.skills.jump_entry_frame),
      "climb": _entry(table.of("climb"), settings.skills.climb_entry_frame),
    }
    self.feet, _ = self.robot.find_bodies(list(FEET))
    self.reset()

  @property
  def done(self) -> bool:
    return self.phase in {"done", "failed"}

  @property
  def status(self) -> str:
    if self.phase == "goal":
      return "walk to goal"
    if self.index >= len(self.course):
      return self.phase
    if self.phase == "go_to":
      approach = self.approaches[self.index]
      distance = torch.linalg.vector_norm(
        self._hold_xy(approach) - self.robot.data.root_link_pos_w[:, :2], dim=-1
      ).max()
      yaw_error = _wrap(self.robot.data.heading_w - approach.yaw).abs().max()
      return (
        f"go_to: obstacle {self.index} ({self.course[self.index].kind}), "
        f"distance={float(distance):.2f} m, yaw_error={float(yaw_error):.2f} rad"
      )
    return f"{self.phase}: obstacle {self.index} ({self.course[self.index].kind})"

  def reset(self) -> None:
    self.tick = 0
    self.index = 0
    self.phase = "go_to"
    self.phase_steps = 0
    self.history: torch.Tensor | None = None
    self.target = torch.empty(0, device=self.env.device)
    self.traverse_armed = False
    self.landed_steps = 0
    self.motion_done_steps = 0
    self.fire = False
    self.command.stop()
    self.runtime.reset()
    self.env.action_manager.action.zero_()
    self.focus.index = 0
    self.approaches = self._approaches()
    preview = getattr(self.command, "set_preview_targets", None)
    if callable(preview):
      preview(torch.cat(tuple(approach.target for approach in self.approaches)))

  def _approaches(self) -> tuple[Approach, ...]:
    names = obstacle_names(self.course)
    out: list[Approach] = []
    for index, obstacle in enumerate(self.course):
      entity: Entity = self.env.scene[names[index]]
      skill = RULES[obstacle.kind]
      out.append(
        solve_approach(
          self.env,
          obstacle,
          entity.data.root_link_pos_w,
          _yaw(entity.data.root_link_quat_w),
          skill,
          self.entries[skill],
          self.cfg,
        )
      )
    return tuple(out)

  def _history_step(self) -> torch.Tensor:
    actual = self.command.state_now()
    if self.history is None:
      self.history = actual[:, None].repeat(1, self.history_length, 1)
    else:
      self.history = torch.cat((self.history[:, 1:], actual[:, None]), dim=1)
    return actual

  def _hold_xy(self, approach: Approach) -> torch.Tensor:
    skill = RULES[self.course[self.index].kind]
    travel = torch.stack(
      (torch.cos(approach.anchor_yaw), torch.sin(approach.anchor_yaw)), dim=-1
    )
    return approach.xy - self.bridge_distance[skill] * travel

  def _at(
    self,
    xy: torch.Tensor,
    yaw: torch.Tensor | None,
    radius: float | None = None,
  ) -> bool:
    position = self.robot.data.root_link_pos_w[:, :2]
    tolerance = self.cfg.position_tolerance if radius is None else radius
    if bool((torch.linalg.vector_norm(xy - position, dim=-1) > tolerance).any()):
      return False
    if yaw is None:
      return True
    return bool(
      (_wrap(self.robot.data.heading_w - yaw).abs() <= self.cfg.yaw_tolerance).all()
    )

  def _walk(
    self, target: torch.Tensor, target_yaw: torch.Tensor | None
  ) -> torch.Tensor:
    delta_w = target - self.robot.data.root_link_pos_w[:, :2]
    distance = torch.linalg.vector_norm(delta_w, dim=-1)
    heading = self.robot.data.heading_w
    delta_b = _rotate(delta_w, -heading) * self.cfg.position_gain
    norm = torch.linalg.vector_norm(delta_b, dim=-1).clamp(min=1.0e-6)
    scale = torch.minimum(
      torch.ones_like(norm), torch.full_like(norm, self.walk_speed) / norm
    )
    velocity = delta_b * scale[:, None]
    velocity[distance <= self.cfg.position_tolerance] = 0.0

    self.twist.vel_command_b[:, :2] = velocity
    self.twist.is_world_env[:] = False
    self.twist.is_forward_env[:] = False
    self.twist.is_standing_env[:] = False
    self.twist.is_heading_env[:] = True
    if target_yaw is None:
      wanted = torch.atan2(delta_w[:, 1], delta_w[:, 0])
      wanted = torch.where(distance > self.cfg.position_tolerance, wanted, heading)
    else:
      wanted = target_yaw
    self.twist.heading_target[:] = wanted
    yaw_rate = self.twist.cfg.heading_control_stiffness * _wrap(wanted - heading)
    self.twist.vel_command_b[:, 2] = yaw_rate.clamp(*self.twist.cfg.ranges.ang_vel_z)
    return self.policies["walk"](fresh_obs(self.env))

  def _hold_motion(self, skill: str) -> None:
    entry = self.entries[skill]
    command = _motion(self.env, skill)
    lengths = command.motion.time_step_total_per_motion[command.motion_ids]
    command.time_steps[:] = torch.minimum(
      torch.full_like(command.time_steps, entry.frame), lengths - 1
    )
    command.motion_done[:] = False
    command.update_relative_body_poses()

  def _start_bridge(self) -> None:
    skill = RULES[self.course[self.index].kind]
    approach = self.approaches[self.index]
    entry = approach.entry
    command_name = MOTIONS[skill]
    resume.prepare(self.env, entry, command_name)
    motion = _motion(self.env, skill)
    ids = torch.arange(self.env.num_envs, device=self.env.device)
    motion.anchor_to_robot(
      ids,
      start_frame=entry.frame,
      at_pos=torch.cat((approach.xy, self.env.scene.env_origins[:, 2:3]), dim=-1),
      at_quat=_quat(approach.anchor_yaw),
    )
    self.target = approach.target
    low, high = self.command.cfg.duration_s_range
    duration_s = self.duration_s[skill]
    if not low <= duration_s <= high:
      raise ValueError(f"Bridge duration must be between {low:g} and {high:g} seconds")
    duration = torch.full((self.env.num_envs,), duration_s, device=self.env.device)
    self.command.open_window(ids, self.target, duration)
    start_errors = self.command.target_errors()[0]
    self.runtime.reset()
    self.phase = "bridge"
    self.phase_steps = 0
    self.fire = False
    print(
      f"bridge to {skill} f{entry.frame:03d} at obstacle {self.index} "
      f"for {duration_s:.2f} s"
      " (start actual/target: "
      + ", ".join(
        f"{name}={float(value):.3f}"
        for name, value in zip(self.command.error_names, start_errors, strict=True)
      )
      + ")"
    )

  def _finish_bridge(self) -> torch.Tensor:
    errors = self.command.target_errors()[0]
    endpoint_score = float(score(errors[None], self.command.tolerances)[0])
    within = bool((errors <= self.command.tolerances).all())
    print(
      "handoff (actual/allowed): "
      + ", ".join(
        f"{name}={float(value):.3f}/{float(limit):.3f}"
        for name, value, limit in zip(
          self.command.error_names, errors, self.command.tolerances, strict=True
        )
      )
      + f", endpoint_score={endpoint_score:.3f}, within_endpoint_box={within}"
    )
    self.command.stop()
    self.phase = "traverse"
    self.phase_steps = 0
    self.traverse_armed = False
    self.landed_steps = 0
    self.motion_done_steps = 0
    skill = RULES[self.course[self.index].kind]
    bridge_action = self.env.action_manager.action.clone()
    expected = torch.as_tensor(
      self.approaches[self.index].entry.previous_action,
      device=self.env.device,
    ).expand_as(bridge_action)
    skill_action = self.policies[skill](fresh_obs(self.env))
    action_error = (bridge_action - expected).square().mean(dim=-1).sqrt()
    action_jump = (skill_action - bridge_action).square().mean(dim=-1).sqrt()
    print(
      f"handoff actions: bridge/entry={float(action_error[0]):.3f}, "
      f"first {skill}/bridge={float(action_jump[0]):.3f}"
    )
    return skill_action

  def _bridge(self) -> torch.Tensor:
    skill = RULES[self.course[self.index].kind]
    self._hold_motion(skill)
    assert self.history is not None
    remaining = (
      self.command.window_steps - self.command.step
    ).float() / self.command.fps
    output = self.runtime(self.history, self.command.target[:, None], remaining)
    if bool(output.handoff.all()):
      return self._finish_bridge()
    grace = round(self.cfg.capture_grace_s / self.env.step_dt)
    overdue = self.command.step - self.command.window_steps
    if bool(self.command.deadline.all()) and bool((overdue >= grace).all()):
      self.phase = "failed"
      print("bridge failed: generated path did not finish before capture timeout")
      return torch.zeros_like(self.env.action_manager.action)
    return output.action

  def _foot_height(self) -> float:
    feet = self.robot.data.body_link_pos_w[:, self.feet, 2]
    floor = self.env.scene.env_origins[:, 2:3]
    return float((feet - floor).min(dim=-1).values.max())

  def _lean(self) -> float:
    gravity = self.robot.data.projected_gravity_b[:, 2]
    return float(torch.acos((-gravity).clamp(-1.0, 1.0)).max())

  def _traverse(self) -> torch.Tensor:
    skill = RULES[self.course[self.index].kind]
    height = self._foot_height()
    self.traverse_armed |= height >= self.cfg.lift_height
    landed = (
      self.traverse_armed
      and height <= self.cfg.land_height
      and self._lean() <= self.cfg.stand_angle
    )
    self.landed_steps = self.landed_steps + 1 if landed else 0
    motion = _motion(self.env, skill)
    if self.landed_steps >= self.cfg.settle_steps:
      print(f"cleared obstacle {self.index} with {skill}")
      self.index += 1
      self.focus.index = min(self.index, len(self.course) - 1)
      self.phase = "goal" if self.index >= len(self.course) else "go_to"
      self.phase_steps = 0
      return self._walk_target()
    self.motion_done_steps = (
      self.motion_done_steps + 1 if bool(motion.motion_done.all()) else 0
    )
    if self.motion_done_steps > self.cfg.traverse_patience:
      self.phase = "failed"
      print(f"{skill} failed to land upright at obstacle {self.index}")
      return torch.zeros_like(self.env.action_manager.action)
    return self.policies[skill](fresh_obs(self.env))

  def _walk_target(self) -> torch.Tensor:
    if self.phase == "goal":
      target = torch.tensor(
        self.course.goal.position,
        device=self.env.device,
        dtype=torch.float32,
      )[None].expand(self.env.num_envs, 2)
      return self._walk(target, None)
    approach = self.approaches[self.index]
    return self._walk(self._hold_xy(approach), approach.yaw)

  def _fallen(self) -> bool:
    height = self.robot.data.root_link_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
    return bool((height < self.cfg.fall_height).any())

  @torch.no_grad()
  def __call__(self, obs) -> torch.Tensor:
    del obs
    self._history_step()
    if self.done:
      return torch.zeros_like(self.env.action_manager.action)
    if self._fallen():
      where = self.status
      self.phase = "failed"
      print(f"robot fell during {where}")
      return torch.zeros_like(self.env.action_manager.action)

    if self.phase == "go_to":
      approach = self.approaches[self.index]
      ready = self._at(self._hold_xy(approach), approach.yaw)
      if self.fire or (self.automatic and ready):
        self._start_bridge()
        action = self._bridge()
      else:
        action = self._walk_target()
    elif self.phase == "bridge":
      action = self._bridge()
    elif self.phase == "traverse":
      action = self._traverse()
    else:
      goal = torch.tensor(
        self.course.goal.position, device=self.env.device, dtype=torch.float32
      )[None].expand(self.env.num_envs, 2)
      if self._at(goal, None, self.course.goal.radius):
        self.phase = "done"
        print(f"goal reached after {self.tick} steps")
        action = torch.zeros_like(self.env.action_manager.action)
      else:
        action = self._walk_target()

    self.tick += 1
    self.phase_steps += 1
    if self.phase_steps > self.cfg.stuck_steps and self.phase in {"go_to", "goal"}:
      where = self.status
      self.phase = "failed"
      print(f"controller stuck during {where}")
    return action
