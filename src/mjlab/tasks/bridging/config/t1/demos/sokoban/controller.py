"""Execute a validated solver plan with timed bridge handoffs."""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np
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
from mjlab.tasks.bridging.config.t1.demos.sokoban.config import Handoff, Handoffs
from mjlab.tasks.bridging.config.t1.skills.push.command import PushCommand
from mjlab.tasks.bridging.config.t1.skills.turn.command import TurnCommand
from mjlab.tasks.bridging.config.t1.skills.walk.distance import WalkDistanceCommand
from mjlab.tasks.bridging.tests.stage import fresh_obs, state
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply, quat_mul, wrap_to_pi
from mjlab.viewer.debug_visualizer import DebugVisualizer

Policy = Callable[[TensorDict], torch.Tensor]


@dataclass(frozen=True)
class Settings:
  position_tolerance: float = 0.25
  heading_tolerance: float = 0.02
  walk_position_tolerance: float = 0.05
  push_standoff: float = 1.0
  box_tolerance: float = 0.04
  box_yaw_tolerance: float = 0.04
  settled_speed: float = 0.15
  settle_seconds: float = 0.2
  action_timeout: float = 60.0
  walk_speed: float = 0.5
  position_gain: float = 3.0


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
    handoffs: Handoffs | None = None,
  ) -> None:
    if env.num_envs != 1 or history_length < 1:
      raise ValueError(
        "Sokoban controller requires one environment and positive history length"
      )
    self.env, self.board, self.plan = env, board, plan
    self.policies, self.bridge = policies, bridge
    self.entries, self.settings = entries or {}, settings or Settings()
    self.handoffs = handoffs or Handoffs()
    self.history: deque[torch.Tensor] = deque(maxlen=history_length)
    self.entry_ghost: mujoco.MjModel | None = None
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
    self.walk_aligned = False
    self.walk_commanded = False
    self.walk_target: torch.Tensor | None = None
    self.walk_heading = 0.0
    self.starting = True
    self.feedback = ""
    self.handoff = Handoff()
    self.bridge_duration = self.handoff.duration
    self._hold_walk()
    if "turn" in self.policies:
      self._turn().set_angle(torch.zeros(1, device=self.env.device))
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
      f"Plan action: {min(self.index + 1, len(self.plan))}/{len(self.plan)} | {self.reason}\n"
      f"{self.feedback}"
    )

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Show the bridge's full target pose and keep it visible on failure"""
    if self.target is None or self.phase == "done":
      return
    pose = self.target[0, 0].detach().cpu().numpy()
    robot = self.env.scene["robot"]
    if self.entry_ghost is None:
      self.entry_ghost = copy.deepcopy(self.env.sim.mj_model)
      visible = np.zeros(self.entry_ghost.ngeom, dtype=bool)
      visible[robot.indexing.geom_ids.cpu().numpy()] = True
      visible &= (self.entry_ghost.geom_contype == 0) & (
        self.entry_ghost.geom_conaffinity == 0
      )
      self.entry_ghost.geom_rgba[:, 3] = 0.0
      self.entry_ghost.geom_rgba[visible] = (0.2, 0.9, 0.8, 0.35)
    qpos = self.entry_ghost.qpos0.copy()
    qpos[robot.indexing.free_joint_q_adr.cpu().numpy()] = pose[:7]
    joints = robot.indexing.joint_q_adr.cpu().numpy()
    qpos[joints] = pose[13 : 13 + len(joints)]
    visualizer.add_ghost_mesh(
      qpos, self.entry_ghost, alpha=0.35, label=f"{self.entering} entry target"
    )
    entry = pose[:3].copy()
    entry[2] = float(self.env.scene.env_origins[0, 2]) + 0.08
    w, x, y, z = pose[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    direction = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    color = (1.0, 0.6, 0.1, 0.9)
    visualizer.add_sphere(entry, 0.08, color, label=f"{self.entering} entry")
    visualizer.add_arrow(
      entry, entry + 0.5 * direction, color, label="Bridge target heading"
    )

  def _fail(self, reason: str) -> None:
    self.phase, self.reason = "failed", f"{reason}; {self.feedback}"
    self.skill = "walk"
    self._hold_walk()

  def _hold_walk(self) -> None:
    command = self._twist()
    if isinstance(command, WalkDistanceCommand):
      command.set_distance(command.heading_target.new_zeros(self.env.num_envs))
      self.walk_commanded = False
    else:
      command.vel_command_b.zero_()

  def _start_distance_walk(self) -> None:
    command = self._twist()
    if (
      isinstance(command, WalkDistanceCommand)
      and not self.walk_commanded
      and self.walk_target is not None
    ):
      command.set_goal(
        self.walk_target[None], self.walk_target.new_tensor([self.walk_heading])
      )
      self.walk_commanded = True

  def _twist(self) -> UniformVelocityCommand:
    command = self.env.command_manager.get_term("twist")
    assert isinstance(command, UniformVelocityCommand)
    return command

  def _push(self) -> PushCommand:
    command = self.env.command_manager.get_term("push")
    assert isinstance(command, PushCommand)
    return command

  def _turn(self) -> TurnCommand:
    command = self.env.command_manager.get_term("turn")
    assert isinstance(command, TurnCommand)
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
    twist = self._twist()
    if isinstance(twist, WalkDistanceCommand) and self.walk_commanded:
      target, yaw = twist.target_xy[0], float(twist.heading_target[0])
    else:
      self.walk_target, self.walk_heading = target.clone(), yaw
    if not isinstance(twist, WalkDistanceCommand):
      twist.vel_command_b.zero_()
    delta = target - position
    axis = delta.new_tensor([math.cos(yaw), math.sin(yaw)])
    remaining = float(delta @ axis)
    lateral = float(delta[0] * axis[1] - delta[1] * axis[0])
    yaw_error = float(wrap_to_pi(delta.new_tensor(yaw - heading)))
    robot = self.env.scene["robot"].data
    speed = float(robot.root_link_lin_vel_w[0, :2].norm())
    turn_rate = abs(float(robot.root_link_ang_vel_w[0, 2]))
    if self.skill == "turn":
      # Finish the one turn requested for this straight segment
      yaw_error = float(self._turn().remaining[0])
    self.feedback = (
      f"Robot ({float(position[0]):.2f}, {float(position[1]):.2f}) | "
      f"Remaining {remaining:.3f} m, lateral error {lateral:.3f} m, heading {yaw_error:.3f} rad | "
      f"Speed {speed:.3f} m/s, turn {turn_rate:.3f} rad/s"
    )
    if self.skill == "turn":
      ready = (
        abs(yaw_error) <= self.settings.heading_tolerance
        and speed <= self.settings.settled_speed
        and turn_rate <= self.settings.settled_speed
      )
      self.stable = self.stable + self.env.step_dt if ready else 0.0
      self.reason = f"Turn in place; heading error {yaw_error:.3f} rad"
      if self.stable >= self.settings.settle_seconds:
        self.walk_aligned = True
        self._switch("walk", self._xy(self.plan[self.index].start), yaw)
      return False
    if not self.walk_aligned and abs(yaw_error) > self.settings.heading_tolerance:
      self.stable = 0.0
      if "turn" not in self.policies:
        raise ValueError("Direction changes require a trained turn skill")
      if (
        speed <= self.settings.settled_speed
        and turn_rate <= self.settings.settled_speed
      ):
        self._turn().set_heading(delta.new_tensor([yaw]))
        if self.starting:
          self.skill = self.entering = "turn"
          self.reason = "Start turning from standing"
        else:
          self._switch("turn", position, heading)
      return False
    self.walk_aligned = True
    self.reason = "Walk straight to segment end"
    if isinstance(twist, WalkDistanceCommand):
      self._start_distance_walk()
      return bool(twist.arrived[0])
    if remaining > self.settings.walk_position_tolerance:
      twist.vel_command_b[0, 0] = min(
        self.settings.walk_speed, self.settings.position_gain * remaining
      )
    ready = (
      remaining <= self.settings.walk_position_tolerance
      and speed <= self.settings.settled_speed
      and turn_rate <= self.settings.settled_speed
    )
    self.stable = self.stable + self.env.step_dt if ready else 0.0
    if self.stable < self.settings.settle_seconds:
      return False
    if float(delta.norm()) > self.settings.walk_position_tolerance:
      self._fail(
        "Straight segment missed its destination; inspect turn and walk accuracy"
      )
      return False
    return True

  def _switch(self, skill: str, xy: torch.Tensor, yaw: float) -> None:
    if skill == self.skill:
      return
    self.handoff = self.handoffs.switches[f"{self.skill}_to_{skill}"]
    self.bridge_duration = (
      round(self.handoff.duration / self.env.step_dt) * self.env.step_dt
    )
    if skill != "push":
      xy = (
        xy + xy.new_tensor([math.cos(yaw), math.sin(yaw)]) * self.handoff.start_distance
      )
    self._hold_walk()
    self.bridge.reset()
    self.entering = skill
    self.bridge_elapsed = 0.0
    self.stable = 0.0
    self.phase = "bridge"
    self.reason = f"{self.skill} -> {skill}"
    if type(self.bridge) is Bridge:
      self.target = state(self.env)[:, None].clone()
      self.target[0, 0, :2] = xy
      self.target[0, 0, 3:7] = self.target.new_tensor(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
      )
    else:
      name = skill
      if skill == "turn":
        # The heading is latched before the switch, so its sign picks the direction
        left = float(self._turn().remaining[0]) >= 0
        name = "turn_left" if left else "turn_right"
      if name not in self.entries:
        raise ValueError(f"A recorded {name} entry is required for the learned bridge")
      clips = self.entries[name]
      if not 0 <= self.handoff.entry_index < len(clips):
        raise ValueError(
          f"{self.reason}: entry_index {self.handoff.entry_index} is outside the {len(clips)} {name} entries"
        )
      self.target = place_entry(clips[self.handoff.entry_index], xy, yaw)
      if self.target.shape[-1] != state(self.env).shape[-1]:
        raise ValueError("Entry joint layout does not match the robot")
      self.target[:, :, 2] += self.env.scene.env_origins[0, 2]

  def _push_entry(self, instruction: Instruction) -> tuple[torch.Tensor, float]:
    assert instruction.box_index is not None
    direction = DIRECTIONS[instruction.action.direction]
    axis_yaw = math.atan2(direction[1], direction[0])
    box = self.env.scene[box_names(self.board)[instruction.box_index]].data
    box_yaw = float(box.heading_w[0])
    yaw = (
      axis_yaw
      + math.atan2(
        math.sin(4 * (box_yaw - axis_yaw)), math.cos(4 * (box_yaw - axis_yaw))
      )
      / 4
    )
    axis = box.root_link_pos_w.new_tensor([math.cos(yaw), math.sin(yaw)])
    return box.root_link_pos_w[0, :2] - self.settings.push_standoff * axis, yaw

  def _box_ready(
    self, index: int, destination: tuple[int, int], approach: bool = False
  ) -> bool:
    box = self.env.scene[box_names(self.board)[index]]
    position = box.data.root_link_pos_w[0]
    yaw = float(box.data.heading_w[0])
    yaw_error = abs(math.atan2(math.sin(4 * yaw), math.cos(4 * yaw))) / 4
    quat = box.data.root_link_quat_w
    up = quat_apply(quat, quat.new_tensor([[0.0, 0.0, 1.0]]))[0, 2]
    return (
      float((position[:2] - self._xy(destination)).norm())
      <= (self.settings.position_tolerance if approach else self.settings.box_tolerance)
      and yaw_error
      <= (
        self.settings.heading_tolerance if approach else self.settings.box_yaw_tolerance
      )
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
    self.walk_aligned = False
    self.walk_commanded = False
    self.walk_target = None
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
    if self.phase != "bridge" and self.elapsed > self.settings.action_timeout:
      self._fail("Action timed out; inspect placement or provide a new plan")
      return self.policies[self.skill](fresh_obs(self.env))
    if self.phase == "bridge":
      assert self.target is not None
      output = self.bridge(
        torch.stack(tuple(self.history), dim=1),
        self.target,
        current.new_tensor([max(0.0, self.bridge_duration - self.bridge_elapsed)]),
      )
      self.feedback = (
        f"Robot ({float(current[0, 0]):.2f}, {float(current[0, 1]):.2f}) | "
        f"Entry position error {float((current[0, :2] - self.target[0, 0, :2]).norm()):.3f} m | "
        f"Within endpoint: {bool(output.within_endpoint_box[0])} | "
        f"Elapsed {self.bridge_elapsed:.2f}/{self.bridge_duration:.2f} s"
      )
      if (
        type(self.bridge) is Bridge
        or self.bridge_elapsed + 1e-9 >= self.bridge_duration
      ):
        self.skill, self.phase = self.entering, "execute"
        self.elapsed = 0.0
        if self.skill == "walk":
          self._start_distance_walk()
        return self.policies[self.skill](fresh_obs(self.env))
      self.bridge_elapsed += self.env.step_dt
      return output.action
    if self.index == len(self.plan):
      if self.skill != "walk":
        position, yaw = self._robot_pose()
        self._switch("walk", position, yaw)
      elif all(self._box_ready(i, cell) for i, cell in enumerate(self.expected_boxes)):
        self.phase, self.reason = "done", "All boxes settled on goals"
        self._hold_walk()
      else:
        self._fail("Final box configuration differs from the plan")
      return self.policies[self.skill](fresh_obs(self.env))
    instruction = self.plan[self.index]
    direction = DIRECTIONS[instruction.action.direction]
    yaw = math.atan2(direction[1], direction[0])
    if instruction.action.skill == "walk":
      if self.skill not in ("walk", "turn"):
        position, heading = self._robot_pose()
        self._switch("walk", position, heading)
      else:
        if self._walk_to(self._xy(instruction.finish), yaw):
          self._complete()
    elif self.skill in ("walk", "turn"):
      self.reason = "Align with pushing cell and box face"
      entry_xy, entry_yaw = self._push_entry(instruction)
      approach = (
        entry_xy
        - entry_xy.new_tensor([math.cos(entry_yaw), math.sin(entry_yaw)])
        * self.handoffs.walk_to_push.start_distance
      )
      if self._walk_to(approach, entry_yaw):
        assert instruction.box_index is not None and instruction.box_start is not None
        vector = current.new_tensor([direction])
        # Lane and destination on the grid, never on the physical box
        self._push().set_goal(
          box_names(self.board)[instruction.box_index],
          instruction.action.cells,
          vector,
          self._xy(instruction.box_start),
        )
        self.prepared_index = self.index
        self._switch("push", entry_xy, entry_yaw)
    else:
      if self.prepared_index != self.index:
        position, heading = self._robot_pose()
        self._switch("walk", position, heading)
      else:
        assert instruction.box_index is not None and instruction.box_finish is not None
        self.reason = f"Push {instruction.action.cells} cells {instruction.action.direction}; wait for box to settle"
        position, heading = self._robot_pose()
        self.feedback = (
          f"Robot ({float(position[0]):.2f}, {float(position[1]):.2f}), "
          f"heading {heading:.3f} rad | Box error "
          f"{self._push().position_error.item():.3f} m | "
          f"Settled {self.stable:.2f}/{self.settings.settle_seconds:.2f} s"
        )
        self.stable = (
          self.stable + self.env.step_dt
          if self._box_ready(instruction.box_index, instruction.box_finish)
          and self._push().at_goal.item()
          else 0.0
        )
        if self.stable >= self.settings.settle_seconds:
          self._complete()
    self.starting = False
    return self.policies[self.skill](fresh_obs(self.env))
