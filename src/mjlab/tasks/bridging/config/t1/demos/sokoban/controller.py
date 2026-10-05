"""Execute a validated solver plan with guarded skill handoffs."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.bridges.interface import Bridge
from mjlab.tasks.bridging.config.t1.demos.sokoban.arena import box_names
from mjlab.tasks.bridging.config.t1.demos.sokoban.board import (
  DIRECTIONS,
  Board,
  Instruction,
)
from mjlab.tasks.bridging.config.t1.skills.push.command import PushCommand
from mjlab.tasks.bridging.tests.stage import fresh_obs, state
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply, quat_mul, wrap_to_pi

Policy = Callable[[TensorDict], torch.Tensor]


@dataclass(frozen=True)
class Settings:
  position_tolerance: float = 0.08
  heading_tolerance: float = 0.10
  box_tolerance: float = 0.04
  box_yaw_tolerance: float = 0.04
  settled_speed: float = 0.06
  settle_seconds: float = 0.4
  bridge_seconds: float = 1.0
  action_timeout: float = 60.0
  walk_speed: float = 0.5
  turn_speed: float = 0.6


def place_entry(clip: torch.Tensor, xy: torch.Tensor, yaw: float) -> torch.Tensor:
  """Rotate an actual skill rollout into the requested entry pose"""
  if clip.ndim != 2 or clip.shape[0] < 1 or clip.shape[1] < 15:
    raise ValueError("Entry must be a nonempty (time, robot state) tensor")
  w, x, y, z = clip[0, 3:7]
  current_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  angle = torch.as_tensor(yaw, device=clip.device) - current_yaw
  rotation = clip.new_zeros((len(clip), 4))
  rotation[:, 0] = torch.cos(angle / 2)
  rotation[:, 3] = torch.sin(angle / 2)
  result = clip.clone()
  origin = clip[0, :3].clone()
  origin[2] = 0.0
  result[:, :3] = quat_apply(rotation, clip[:, :3] - origin)
  result[:, :2] += xy
  result[:, 3:7] = quat_mul(rotation, clip[:, 3:7])
  result[:, 7:10] = quat_apply(rotation, clip[:, 7:10])
  result[:, 10:13] = quat_apply(rotation, clip[:, 10:13])
  return result[None]


class Controller:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    board: Board,
    plan: tuple[Instruction, ...],
    policies: Mapping[str, Policy],
    bridge: Bridge,
    entries: dict[str, torch.Tensor] | None = None,
    history_length: int = 30,
    settings: Settings | None = None,
  ) -> None:
    if env.num_envs != 1 or history_length < 1:
      raise ValueError(
        "Sokoban controller requires one environment and positive history length"
      )
    self.env, self.board, self.plan = env, board, plan
    self.policies, self.bridge = policies, bridge
    self.entries, self.settings = entries or {}, settings or Settings()
    self.history: deque[torch.Tensor] = deque(maxlen=history_length)
    self.reset()

  def reset(self) -> None:
    self.bridge.reset()
    self.history.clear()
    self.index = 0
    self.skill = "walk"
    self.phase = "execute"
    self.reason = "Follow solver plan"
    self.elapsed = self.stable = self.bridge_elapsed = 0.0
    self.target: torch.Tensor | None = None
    self.entering = "walk"
    self.prepared_index = -1
    self.expected_boxes = list(self.board.boxes)
    self.walk_waypoint = 1
    self._twist().vel_command_b.zero_()
    for policy in self.policies.values():
      reset = getattr(policy, "reset", None)
      if callable(reset):
        reset()

  @property
  def done(self) -> bool:
    return self.phase in ("done", "failed")

  @property
  def status(self) -> str:
    active = (
      f"bridge ({self.skill} -> {self.entering})"
      if self.phase == "bridge"
      else self.skill
    )
    return (
      f"Active skill: {active} | Phase: {self.phase}\n"
      f"Plan action: {min(self.index + 1, len(self.plan))}/{len(self.plan)} | {self.reason}"
    )

  def _fail(self, reason: str) -> None:
    self.phase, self.reason = "failed", reason
    self.skill = "walk"
    self._twist().vel_command_b.zero_()

  def _twist(self) -> UniformVelocityCommand:
    command = self.env.command_manager.get_term("twist")
    assert isinstance(command, UniformVelocityCommand)
    return command

  def _push(self) -> PushCommand:
    command = self.env.command_manager.get_term("push")
    assert isinstance(command, PushCommand)
    return command

  def _xy(self, cell: tuple[int, int]) -> torch.Tensor:
    return (
      torch.tensor(self.board.center(cell), device=self.env.device)
      + self.env.scene.env_origins[0, :2]
    )

  def _robot_pose(self) -> tuple[torch.Tensor, float]:
    robot = self.env.scene["robot"]
    return robot.data.root_link_pos_w[0, :2], float(robot.data.heading_w[0])

  def _walk_to(self, target: torch.Tensor, yaw: float) -> bool:
    position, heading = self._robot_pose()
    delta = target - position
    yaw_error = float(wrap_to_pi(torch.tensor(yaw - heading)))
    twist = self._twist()
    twist.vel_command_b.zero_()
    distance = float(delta.norm())
    desired_yaw = (
      math.atan2(float(delta[1]), float(delta[0]))
      if distance > self.settings.position_tolerance
      else yaw
    )
    error = math.atan2(math.sin(desired_yaw - heading), math.cos(desired_yaw - heading))
    twist.vel_command_b[0, 2] = max(
      -self.settings.turn_speed, min(self.settings.turn_speed, 2.0 * error)
    )
    if distance > self.settings.position_tolerance and abs(error) < 0.2:
      twist.vel_command_b[0, 0] = min(self.settings.walk_speed, distance)
    return (
      distance <= self.settings.position_tolerance
      and abs(yaw_error) <= self.settings.heading_tolerance
    )

  def _switch(self, skill: str, xy: torch.Tensor, yaw: float) -> None:
    if skill == self.skill:
      return
    self._twist().vel_command_b.zero_()
    self.bridge.reset()
    self.entering = skill
    self.bridge_elapsed = 0.0
    self.phase = "bridge"
    self.reason = f"{self.skill} -> {skill}"
    if type(self.bridge) is Bridge:
      self.target = state(self.env)[:, None].clone()
    else:
      if skill not in self.entries:
        raise ValueError(f"A recorded {skill} entry is required for the learned bridge")
      self.target = place_entry(self.entries[skill], xy, yaw)
      if self.target.shape[-1] != state(self.env).shape[-1]:
        raise ValueError("Entry joint layout does not match the robot")
      self.target[:, :, 2] += self.env.scene.env_origins[0, 2]

  def _box_ready(self, index: int, destination: tuple[int, int]) -> bool:
    box = self.env.scene[box_names(self.board)[index]]
    position = box.data.root_link_pos_w[0]
    yaw = float(box.data.heading_w[0])
    yaw_error = abs(math.atan2(math.sin(4 * yaw), math.cos(4 * yaw))) / 4
    quat = box.data.root_link_quat_w
    up = quat_apply(quat, quat.new_tensor([[0.0, 0.0, 1.0]]))[0, 2]
    return (
      float((position[:2] - self._xy(destination)).norm())
      <= self.settings.box_tolerance
      and yaw_error <= self.settings.box_yaw_tolerance
      and float(up) > math.cos(self._push().cfg.tilt_tolerance)
      and abs(float(position[2] - self.env.scene.env_origins[0, 2]) - 0.5) < 0.04
      and float(box.data.root_link_lin_vel_w.norm()) < self.settings.settled_speed
      and float(box.data.root_link_ang_vel_w.norm()) < self.settings.settled_speed
    )

  def _complete(self) -> None:
    instruction = self.plan[self.index]
    if instruction.box_index is not None:
      assert instruction.box_finish is not None
      self.expected_boxes[instruction.box_index] = instruction.box_finish
    self.index += 1
    self.elapsed = self.stable = 0.0
    self.walk_waypoint = 1
    self.reason = "Action complete"

  @torch.no_grad()
  def __call__(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
    current = state(self.env).clone()
    if not self.history:
      for _ in range(self.history.maxlen or 1):
        self.history.append(current)
    else:
      self.history.append(current)
    if self.done:
      return self.policies[self.skill](fresh_obs(self.env))
    self.elapsed += self.env.step_dt
    if self.elapsed > self.settings.action_timeout:
      self._fail("Action timed out; inspect placement or provide a new plan")
      return self.policies[self.skill](fresh_obs(self.env))
    if self.phase == "bridge":
      assert self.target is not None
      output = self.bridge(
        torch.stack(tuple(self.history), dim=1),
        self.target,
        current.new_tensor(
          [max(0.0, self.settings.bridge_seconds - self.bridge_elapsed)]
        ),
      )
      self.bridge_elapsed += self.env.step_dt
      if bool(output.handoff[0]):
        if type(self.bridge) is not Bridge and not bool(output.within_endpoint_box[0]):
          self._fail("Bridge finished outside the skill entry tolerances")
          return self.policies[self.skill](fresh_obs(self.env))
        self.skill, self.phase = self.entering, "execute"
        if self.skill == "push":
          instruction = self.plan[self.index]
          direction = DIRECTIONS[instruction.action.direction]
          position, yaw = self._robot_pose()
          wanted_yaw = math.atan2(direction[1], direction[0])
          if (
            float((position - self._xy(instruction.start)).norm())
            > self.settings.position_tolerance
            or abs(math.atan2(math.sin(yaw - wanted_yaw), math.cos(yaw - wanted_yaw)))
            > self.settings.heading_tolerance
          ):
            self._fail("Push entry position or heading is unsuitable")
        return self.policies[self.skill](fresh_obs(self.env))
      if self.bridge_elapsed > self.settings.bridge_seconds + 1.0:
        self._fail("Bridge did not hand off before its deadline")
        return self.policies[self.skill](fresh_obs(self.env))
      return output.action
    if self.index == len(self.plan):
      if self.skill != "walk":
        position, yaw = self._robot_pose()
        self._switch("walk", position, yaw)
      elif all(self._box_ready(i, cell) for i, cell in enumerate(self.expected_boxes)):
        self.phase, self.reason = "done", "All boxes settled on goals"
        self._twist().vel_command_b.zero_()
      else:
        self._fail("Final box configuration differs from the plan")
      return self.policies[self.skill](fresh_obs(self.env))
    instruction = self.plan[self.index]
    direction = DIRECTIONS[instruction.action.direction]
    yaw = math.atan2(direction[1], direction[0])
    if instruction.action.skill == "walk":
      if self.skill != "walk":
        position, heading = self._robot_pose()
        self._switch("walk", position, heading)
      else:
        waypoint = (
          instruction.start[0] + direction[0] * self.walk_waypoint,
          instruction.start[1] + direction[1] * self.walk_waypoint,
        )
        if self._walk_to(self._xy(waypoint), yaw):
          self.walk_waypoint += 1
          if self.walk_waypoint > instruction.action.cells:
            self._complete()
    elif self.skill == "walk":
      self.reason = "Align with pushing cell and box face"
      if self._walk_to(self._xy(instruction.start), yaw):
        assert instruction.box_index is not None and instruction.box_start is not None
        if not all(
          self._box_ready(i, cell) for i, cell in enumerate(self.expected_boxes)
        ):
          self._fail("Boxes are outside their expected cells or not settled")
        else:
          vector = current.new_tensor([direction])
          # Lane and destination on the grid, never on the physical box
          self._push().set_goal(
            box_names(self.board)[instruction.box_index],
            instruction.action.cells,
            vector,
            self._xy(instruction.box_start),
          )
          self.prepared_index = self.index
          self._switch("push", self._xy(instruction.start), yaw)
    else:
      if self.prepared_index != self.index:
        position, heading = self._robot_pose()
        self._switch("walk", position, heading)
      else:
        assert instruction.box_index is not None and instruction.box_finish is not None
        self.reason = f"Push {instruction.action.cells} cells {instruction.action.direction}; wait for box to settle"
        self.stable = (
          self.stable + self.env.step_dt
          if self._box_ready(instruction.box_index, instruction.box_finish)
          and self._push().at_goal.item()
          else 0.0
        )
        if self.stable >= self.settings.settle_seconds:
          self._complete()
    return self.policies[self.skill](fresh_obs(self.env))
