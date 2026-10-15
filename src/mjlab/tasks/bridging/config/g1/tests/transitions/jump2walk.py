"""Run a G1 jump-to-walk handoff after landing.

Run

    uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.jump2walk

    uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.jump2walk \
      --viewer none --automatic True

    uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.jump2walk \
      --kinematic-only True

The jump starts at the cropped clip's first frame. After airborne feet touch
the ground, bridge to a recorded walk entry, then keep walking forward.
--entry selects the walk state. --walk-distance places it beyond the end of the
jump's recovery, so the bridge has room after touchdown. --trigger-distance
starts the automatic bridge within the chosen horizontal distance of the entry,
after landing. Walk speed defaults to its recorded command.
--automatic False waits for Start bridge, which also requires landing.
--kinematic-only shows the plan with no tracker, then starts walking.
"""

# pyright: reportPrivateImportUsage=false, reportArgumentType=false

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.bridging.bridges import BRIDGES, BridgeSpec
from mjlab.tasks.bridging.bridges.cvae.goal.command import (
  GoalCommand,
  post_goal_features,
  rotation_6d,
)
from mjlab.tasks.bridging.bridges.diffusion.execution.learned_tracker import (
  LearnedTrackerExecutor,
)
from mjlab.tasks.bridging.bridges.diffusion.execution.runtime import (
  DiffusionRuntime,
  latest_planner_checkpoint,
)
from mjlab.tasks.bridging.bridges.imitation.command import PLAN_COLOR, score
from mjlab.tasks.bridging.bridges.interface import BridgeCommand
from mjlab.tasks.bridging.bridges.mixed.runtime import MixedRuntime
from mjlab.tasks.bridging.config import get_robot
from mjlab.tasks.bridging.config.g1 import selector as resume
from mjlab.tasks.bridging.config.g1.demos.soccer.controller import entry_sequence
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.bridging.config.t1.demos.sokoban.controller import place_entry
from mjlab.tasks.bridging.selector import paths as selector_paths
from mjlab.tasks.bridging.selector.table import (
  Entry,
  EntryTable,
)
from mjlab.tasks.bridging.tests.stage import (
  BRIDGE,
  Policy,
  arena,
  fresh_obs,
  load_policy,
  state,
)
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply
from mjlab.viewer.debug_visualizer import DebugVisualizer

G1 = get_robot("g1")
WALK_TASK_ID = G1.skills["walk"]
JUMP_TASK_ID = G1.skills["jump"]
SELECTOR_PATH = selector_paths("g1")[1]


@dataclass(frozen=True)
class Config:
  bridge: str = "diffusion"
  bridge_duration_s: float = 0.8
  capture_grace_s: float = 0.5
  capture_tolerance_scale: float = 2.0
  trigger_distance: float = 1.0
  automatic: bool = True
  walk_speed: float | None = None
  walk_distance: float = 0.5
  bridge_checkpoint: Path | None = None
  tracker_checkpoint: Path | None = None
  imitation_checkpoint: Path | None = None
  walk_checkpoint: Path | None = None
  jump_checkpoint: Path | None = None
  selector_path: Path = SELECTOR_PATH
  entry: int = 0
  viewer: Literal["viser", "none"] = "viser"
  steps: int = 600
  device: str | None = None
  seed: int = 0
  exact_entry_baseline: bool = False
  diagnostic_path: Path | None = None
  diffusion_sample_steps: int | None = None
  kinematic_only: bool = False


def _diffusion_checkpoints(
  cfg: Config, runtime: DiffusionRuntime | None = None
) -> tuple[Path, Path | None]:
  """Planner and tracker checkpoints, the newest of each unless given.

  A planner improvement checkpoint carries its own tracker, which is used when
  no tracker is given.
  """
  latest = runtime.latest_checkpoint if runtime is not None else None
  planner = cfg.bridge_checkpoint or (latest or latest_planner_checkpoint)()
  tracker = cfg.tracker_checkpoint
  if tracker is None:
    saved = torch.load(planner, map_location="cpu", weights_only=True)
    if "planner" in saved and "actor_state_dict" in saved:
      tracker = planner
  return planner, tracker


def _motion(env: ManagerBasedRlEnv) -> JumpCommand:
  command = env.command_manager.get_term("motion")
  if not isinstance(command, JumpCommand):
    raise TypeError("The jump skill does not have a clip tracker")
  return command


def _robot_model(
  env: ManagerBasedRlEnv,
  robot: Entity,
  color: tuple[float, float, float, float] | None = None,
) -> mujoco.MjModel:
  """Copy of the sim model that draws only the robot, in its own colors or in one."""
  model = copy.deepcopy(env.sim.mj_model)
  robot_geoms = set(robot.indexing.geom_ids.tolist())
  for geom in range(model.ngeom):
    collision = model.geom_contype[geom] or model.geom_conaffinity[geom]
    if geom not in robot_geoms or collision:
      model.geom_rgba[geom, 3] = 0.0
    elif color is not None:
      model.geom_rgba[geom] = color
    elif model.geom_matid[geom] >= 0:
      model.geom_rgba[geom] = model.mat_rgba[model.geom_matid[geom]]
  return model


class Run:
  """Drive jump, bridge, then walk in one environment."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    policies: dict[str, Policy],
    entries: tuple[Entry, ...],
    cfg: Config,
    bridges: dict[str, BridgeSpec],
  ) -> None:
    self.env = env
    self.policies = policies
    self.entries = entries
    self.entry_index = cfg.entry
    self.cfg = cfg
    command = env.command_manager.get_term(BRIDGE)
    if not isinstance(command, BridgeCommand):
      raise TypeError("The jump2walk stage does not implement BridgeCommand")
    self.command = command
    self.command.tolerances *= cfg.capture_tolerance_scale
    self.motion = _motion(env)
    self.robot: Entity = env.scene["robot"]
    self.phase = "jump"
    self.within_endpoint_box: bool | None = None
    self.fire = False
    self.automatic = cfg.automatic
    self.trigger_distance = cfg.trigger_distance
    self.duration_s = cfg.bridge_duration_s
    self.walk_speed = 0.0
    self.walk_heading = 0.0
    self.airborne = False
    self.landed = False
    self.feet, _ = self.robot.find_bodies(
      ["left_ankle_roll_link", "right_ankle_roll_link"]
    )
    self.bridges = bridges
    self.bridge = cfg.bridge
    self.active_bridge = cfg.bridge
    self.target = torch.empty(0, device=env.device)
    self.target_window = torch.empty(0, device=env.device)
    self.history: torch.Tensor | None = None
    self.diagnostic_actual: list[np.ndarray] = []
    self.diagnostic_planned: list[np.ndarray] = []
    self.diagnostic_phase: list[int] = []
    self.diagnostic_action: list[np.ndarray] = []
    self.diagnostic_jump_errors: list[np.ndarray] = []
    action_dim = entries[0].previous_action.size
    self.jump_tracking_count = 0
    self.jump_fell = False
    self.jump_tracking_sum = {
      name: 0.0
      for name in (
        "error_anchor_pos",
        "error_body_pos",
        "error_joint_pos",
        "error_joint_vel",
      )
    }
    self.runtime_bridges = {
      name: spec.runtime(action_dim).to(env.device)
      for name, spec in bridges.items()
      if spec.runtime is not None
    }
    chosen = self.runtime_bridges.get(cfg.bridge)
    self.history_length = 4
    self.future = 1
    if isinstance(chosen, DiffusionRuntime) and cfg.kinematic_only:
      chosen.checkpoint = cfg.bridge_checkpoint or chosen.latest_checkpoint()
      chosen.sample_steps = cfg.diffusion_sample_steps
      # No tracker, the plan is only drawn, see debug_vis
      chosen.set_executor(lambda history, _: env.action_manager.action.clone())
      self.history_length = chosen.load(env.device).history
      self.future = chosen.load(env.device).future
    elif isinstance(chosen, DiffusionRuntime):
      chosen.checkpoint, tracker_checkpoint = _diffusion_checkpoints(cfg, chosen)
      chosen.sample_steps = cfg.diffusion_sample_steps
      tracker = LearnedTrackerExecutor.load(env, tracker_checkpoint)
      tracker.tolerances = self.command.tolerances
      chosen.set_executor(tracker, tracker.within_endpoint_box)
      self.history_length = max(
        chosen.load(env.device).history, getattr(tracker, "history", 1)
      )
      self.future = chosen.load(env.device).future
    self.frozen: torch.Tensor | None = None
    self.robot_model = _robot_model(env, self.robot)
    self.plan_model = _robot_model(env, self.robot, PLAN_COLOR)
    self.reset()

  @property
  def distance(self) -> float:
    delta = self.robot.data.root_link_pos_w[0, :2] - self.command.target[0, :2]
    return float(torch.linalg.vector_norm(delta))

  @property
  def status(self) -> str:
    entry = self.entries[self.entry_index].name
    if self.phase == "jump":
      mode = "auto" if self.automatic else "manual"
      flight = "landed" if self.landed else "airborne" if self.airborne else "takeoff"
      return f"jump  {flight}  {self.distance:.2f} m from walk entry {entry}  {mode}"
    if self.phase == "bridge":
      left = max(self.duration_s - float(self.command.step[0]) * self.env.step_dt, 0.0)
      kind = " plan" if self.cfg.kinematic_only else ""
      return f"bridge ({self.active_bridge}{kind})  {left:.2f} s left"
    if self.phase == "failed":
      return f"failed ({self.active_bridge})"
    return "walk"

  def reset(self) -> None:
    self.env.reset()
    self.phase = "jump"
    self.airborne = self.landed = False
    self.within_endpoint_box = None
    self.fire = False
    self.jump_tracking_count = 0
    self.jump_fell = False
    self.jump_tracking_sum = dict.fromkeys(self.jump_tracking_sum, 0.0)
    self.history = None
    self.frozen = None
    self.diagnostic_actual.clear()
    self.diagnostic_planned.clear()
    self.diagnostic_phase.clear()
    self.diagnostic_action.clear()
    self.diagnostic_jump_errors.clear()
    self.command.stop()
    self.env.action_manager.action.zero_()
    for policy in self.policies.values():
      policy.reset()

    entry = self.entries[self.entry_index]
    if entry.command is None or entry.command.shape != (3,):
      raise ValueError(
        "Walk entries need a recorded velocity command; run the selector again"
      )
    recorded_speed = float(entry.command[0])
    self.walk_speed = (
      recorded_speed if self.cfg.walk_speed is None else self.cfg.walk_speed
    )
    if not np.allclose(entry.command, (self.walk_speed, 0.0, 0.0)):
      raise ValueError(
        f"Walk entry was not recorded at ({self.walk_speed:g}, 0, 0) m/s; "
        "use its recorded speed or record matching walk entries"
      )
    ids = torch.arange(self.env.num_envs, device=self.env.device)
    heading = torch.zeros(self.env.num_envs, 4, device=self.env.device)
    heading[:, 0] = 1.0
    self.motion.anchor_to_robot(
      ids, start_frame=0, at_pos=self.env.scene.env_origins, at_quat=heading
    )
    self._teleport(self._reference())
    resume.restore_action(self.env, torch.zeros_like(self.env.action_manager.action))
    self.env.sim.forward()

    # Recovery travels past touchdown, so leave the bridge room beyond its end
    recovery_end = self.motion.goal_pos_w[0, :2].clone()
    ahead = torch.tensor([[self.cfg.walk_distance, 0.0, 0.0]], device=self.env.device)
    xy = recovery_end + quat_apply(heading, ahead)[0, :2]
    clip = torch.as_tensor(entry_sequence(entry, self.future), device=self.env.device)
    self.target_window = place_entry(clip, xy, self.walk_heading)
    self.target_window[:, :, 2] += self.env.scene.env_origins[0, 2]
    self.target = self.target_window[:, 0]
    if isinstance(self.command, GoalCommand):
      foot_fields = (
        entry.foot_pos_b,
        entry.foot_quat_b,
        entry.foot_lin_vel_b,
        entry.foot_ang_vel_b,
        entry.foot_contact,
      )
      if any(value is None for value in foot_fields):
        raise ValueError(
          "Goal CVAE needs target foot poses, velocities and contacts. "
          "Re-run selector.record and selector.build"
        )
      assert all(value is not None for value in foot_fields)
      foot_pos = torch.as_tensor(entry.foot_pos_b, device=self.env.device)
      foot_quat = torch.as_tensor(entry.foot_quat_b, device=self.env.device)
      foot_lin = torch.as_tensor(entry.foot_lin_vel_b, device=self.env.device)
      foot_ang = torch.as_tensor(entry.foot_ang_vel_b, device=self.env.device)
      foot = (
        torch.cat((foot_pos, rotation_6d(foot_quat), foot_lin, foot_ang), dim=-1)
        .flatten()[None]
        .expand(self.env.num_envs, -1)
      )
      contact = torch.as_tensor(entry.foot_contact, device=self.env.device)
      future = None
      if (
        entry.future_states is not None
        and entry.future_contact is not None
        and entry.future_mask is not None
      ):
        future = post_goal_features(
          torch.as_tensor(entry.state, device=self.env.device)[None],
          torch.as_tensor(entry.future_states, device=self.env.device)[None],
          torch.as_tensor(entry.future_contact, device=self.env.device)[None],
          torch.as_tensor(entry.future_mask, device=self.env.device)[None].bool(),
        ).expand(self.env.num_envs, -1)
      self.command.aim(
        self.target,
        target_foot=foot,
        target_foot_quat=foot_quat[None].expand(self.env.num_envs, -1, -1),
        target_contact=contact[None].expand(self.env.num_envs, -1),
        target_future=future,
      )
    else:
      self.command.aim(self.target)
    self._set_walk()

  def select_entry(self, index: int) -> None:
    """Aim at another selected walk state and restart the handoff."""
    if not 0 <= index < len(self.entries):
      raise IndexError(f"Walk has {len(self.entries)} selector states")
    self.entry_index = index
    self.reset()

  def _reference(self) -> torch.Tensor:
    return torch.cat(
      (
        self.motion.body_pos_w[:, 0],
        self.motion.body_quat_w[:, 0],
        self.motion.body_lin_vel_w[:, 0],
        self.motion.body_ang_vel_w[:, 0],
        self.motion.joint_pos,
        self.motion.joint_vel,
      ),
      dim=-1,
    )

  def _landed(self) -> bool:
    sensor = self.env.scene["feet_ground_contact"]
    assert isinstance(sensor, ContactSensor) and sensor.data.found is not None
    grounded = bool((sensor.data.found > 0).any())
    height = (
      self.robot.data.body_link_pos_w[0, self.feet, 2]
      - self.env.scene.env_origins[0, 2]
    )
    self.airborne |= not grounded and float(height.min()) > 0.08
    self.landed |= self.airborne and grounded and bool(self.motion.has_landed[0])
    return self.landed

  def _set_walk(self) -> None:
    twist = self.env.command_manager.get_term("twist")
    assert isinstance(twist, UniformVelocityCommand)
    twist.vel_command_b[:, 0] = self.walk_speed
    twist.vel_command_b[:, 1:] = 0.0
    twist.heading_target[:] = self.walk_heading
    twist.is_heading_env[:] = True
    twist.is_standing_env[:] = False
    twist.is_world_env[:] = False

  def _start_bridge(self) -> None:
    low, high = self.command.cfg.duration_s_range
    if not low <= self.duration_s <= high:
      raise ValueError(f"Bridge duration must be between {low:g} and {high:g} seconds")
    ids = torch.arange(self.env.num_envs, device=self.env.device)
    duration = torch.full((self.env.num_envs,), self.duration_s, device=self.env.device)
    self.command.open_window(ids, self.target, duration)
    self.active_bridge = self.bridge
    if self.active_bridge in self.policies:
      self.policies[self.active_bridge].reset()
    runtime = self.runtime_bridges.get(self.active_bridge)
    if runtime is not None:
      runtime.reset()
    if self.cfg.kinematic_only:
      self.frozen = state(self.env).clone()
    self.phase = "bridge"
    self.fire = False
    print(
      f"bridge started {self.distance:.2f} m from target for {self.duration_s:.2f} s"
    )

  def _teleport(self, goal: torch.Tensor) -> None:
    """Write a state in the bridge layout straight into the sim."""
    joints = self.robot.data.joint_pos.shape[-1]
    self.robot.write_root_state_to_sim(goal[:, :13])
    self.robot.write_joint_state_to_sim(
      goal[:, 13 : 13 + joints], goal[:, 13 + joints :]
    )
    self.env.sim.forward()

  def _entry_action(self) -> torch.Tensor:
    """Action recorded with the entry state."""
    return torch.as_tensor(
      self.entries[self.entry_index].previous_action, device=self.env.device
    ).expand(self.env.num_envs, -1)

  def _qpos(self, state: torch.Tensor) -> np.ndarray:
    """Sim qpos with the robot set to a state in the bridge layout."""
    qpos = np.array(self.env.sim.mj_model.qpos0, dtype=np.float64)
    values = state.cpu().numpy()
    free = self.robot.indexing.free_joint_q_adr.cpu().numpy()
    joints = self.robot.indexing.joint_q_adr.cpu().numpy()
    qpos[free] = values[:7]
    qpos[joints] = values[13 : 13 + len(joints)]
    return qpos

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Kinematic only: the landed jump, the waiting walk and the plan ghost.

    The sim has one robot. It jumps, then sits on the entry state from the
    bridge start, so the robot that is not simulated is drawn here.
    """
    if self.frozen is None:
      entry = self._qpos(self.target[0])
      visualizer.add_ghost_mesh(entry, self.robot_model, alpha=1.0, label="entry")
      return
    jump = self._qpos(self.frozen[0])
    visualizer.add_ghost_mesh(jump, self.robot_model, alpha=1.0, label="jump")
    runtime = self.runtime_bridges.get(self.active_bridge)
    if (
      self.phase == "bridge"
      and isinstance(runtime, DiffusionRuntime)
      and runtime.path is not None
      and runtime._index is not None
    ):
      planned = self._qpos(runtime.path.states[0, runtime._index[0]])
      visualizer.add_ghost_mesh(
        planned, self.plan_model, alpha=PLAN_COLOR[3], label="plan"
      )

  def _finish_bridge(self, reason: str | None = None, handoff: bool = True):
    if self.cfg.exact_entry_baseline or self.cfg.kinematic_only:
      self._teleport(self.target)
      # Teleporting skips the actions that lead to the entry, restore the last one
      resume.restore_action(self.env, self._entry_action())
      self.env.sim.forward()
    errors = self.command.target_errors()[0]
    within_endpoint_box = bool((errors <= self.command.tolerances).all())
    endpoint_score = float(score(errors.unsqueeze(0), self.command.tolerances)[0])
    goal = self.command if isinstance(self.command, GoalCommand) else None
    foot_errors = goal.foot_errors()[0] if goal is not None else None
    if foot_errors is not None:
      assert goal is not None
      within_endpoint_box &= bool(
        (
          foot_errors <= torch.tensor(goal.cfg.foot_tolerances, device=self.env.device)
        ).all()
      )
    print(
      (reason or ("captured" if not bool(self.command.deadline[0]) else "deadline"))
      + " (actual/allowed): "
      + ", ".join(
        f"{name}={float(value):.3f}/{float(limit):.3f}"
        for name, value, limit in zip(
          self.command.error_names, errors, self.command.tolerances, strict=True
        )
      )
      + f", endpoint_score={endpoint_score:.3f}"
      + f", within_endpoint_box={within_endpoint_box}"
      + (
        ", "
        + ", ".join(
          f"{name}={float(value):.3f}"
          for name, value in zip(
            ("foot_pos", "foot_ori", "foot_lin_vel", "foot_ang_vel"),
            foot_errors,
            strict=True,
          )
        )
        if foot_errors is not None
        else ""
      )
    )
    self.within_endpoint_box = within_endpoint_box
    self.command.stop()
    if not handoff:
      self.phase = "failed"
      return torch.zeros_like(self.env.action_manager.action)
    self.phase = "walk"
    self._set_walk()
    bridge_action = self.env.action_manager.action.clone()
    walk_action = self.policies["walk"](fresh_obs(self.env))
    action_jump = (walk_action - bridge_action).square().mean(dim=-1).sqrt()
    print(f"first walk action jump: {float(action_jump[0]):.3f}")
    return walk_action

  def _record_jump_tracking(self) -> None:
    self.jump_tracking_count += 1
    self.jump_fell |= bool(self.robot.data.projected_gravity_b[0, 2] > -0.2)
    for name in self.jump_tracking_sum:
      self.jump_tracking_sum[name] += float(self.motion.metrics[name][0])
    interval = round(0.5 / self.env.step_dt)
    if self.jump_tracking_count in (interval, 2 * interval):
      print(
        f"jump tracking after {self.jump_tracking_count * self.env.step_dt:.1f}s "
        f"(fell={self.jump_fell}): "
        + ", ".join(
          f"{name}={total / self.jump_tracking_count:.3f}"
          for name, total in self.jump_tracking_sum.items()
        )
      )

  def save_diagnostic(self) -> None:
    if self.cfg.diagnostic_path is None:
      return
    path = self.cfg.diagnostic_path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      path,
      actual=np.asarray(self.diagnostic_actual),
      planned=np.asarray(self.diagnostic_planned),
      phase=np.asarray(self.diagnostic_phase),
      action=np.asarray(self.diagnostic_action),
      jump_errors=np.asarray(self.diagnostic_jump_errors),
      target=self.target[0].cpu().numpy(),
      fps=1.0 / self.env.step_dt,
      entry=self.entry_index,
      airborne=self.airborne,
      landed=self.landed,
      exact_entry_baseline=self.cfg.exact_entry_baseline,
    )
    print(f"diagnostic saved: {path}")

  @torch.no_grad()
  def __call__(self, obs):
    del obs
    if self.phase == "failed":
      return torch.zeros_like(self.env.action_manager.action)
    actual = self.command.state_now()
    self.history = (
      actual[:, None].repeat(1, self.history_length, 1)
      if self.history is None
      else torch.cat((self.history[:, 1:], actual[:, None]), dim=1)
    )
    if self.phase == "jump":
      self._record_jump_tracking()
      if self.jump_fell:
        self.phase = "failed"
        print("jump failed: robot fell before the walk handoff")
        return torch.zeros_like(self.env.action_manager.action)
      landed = self._landed()
      if landed and (
        self.fire or (self.automatic and self.distance <= self.trigger_distance)
      ):
        self._start_bridge()
        if self.cfg.exact_entry_baseline:
          return self._finish_bridge("exact entry baseline")
      else:
        if bool(self.motion.motion_done[0]) and not landed:
          self.phase = "failed"
          print("jump failed: clip ended without a confirmed landing")
          return torch.zeros_like(self.env.action_manager.action)
        action = self.policies["jump"](fresh_obs(self.env))
        self.diagnostic_actual.append(actual[0].cpu().numpy().copy())
        self.diagnostic_planned.append(self._reference()[0].cpu().numpy().copy())
        self.diagnostic_phase.append(0)
        self.diagnostic_action.append(action[0].cpu().numpy().copy())
        self.diagnostic_jump_errors.append(
          np.array(
            [float(self.motion.metrics[name][0]) for name in self.jump_tracking_sum]
          )
        )
        return action

    if self.phase == "bridge":
      runtime = self.runtime_bridges.get(self.active_bridge)
      planned = None
      if runtime is not None:
        duration = self.command.window_steps.float() / self.command.fps
        output = runtime(self.history, self.target_window, duration)
        if self.cfg.kinematic_only and isinstance(runtime, DiffusionRuntime):
          # The entering robot waits on the entry while the ghost plays the plan
          self._teleport(self.target)
          output = output._replace(action=self._entry_action())
        if (
          isinstance(runtime, (DiffusionRuntime, MixedRuntime))
          and runtime.path is not None
        ):
          assert runtime._index is not None
          index = min(int(runtime._index[0]) - 1, int(runtime.path.duration[0]))
          self.diagnostic_actual.append(actual[0].cpu().numpy().copy())
          self.diagnostic_planned.append(
            runtime.path.states[0, index].cpu().numpy().copy()
          )
          self.diagnostic_phase.append(1)
          self.diagnostic_action.append(output.action[0].cpu().numpy().copy())
          self.diagnostic_jump_errors.append(np.full(4, np.nan))
        if bool(output.handoff[0]):
          return self._finish_bridge(self.active_bridge)
        grace = round(self.cfg.capture_grace_s / self.env.step_dt)
        overdue = self.command.step[0] - self.command.window_steps[0]
        if bool(self.command.deadline[0]) and int(overdue) >= grace:
          return self._finish_bridge("capture timeout", handoff=False)
        planned = output.action
      if not isinstance(runtime, (DiffusionRuntime, MixedRuntime)) and bool(
        self.command.handoff[0]
      ):
        return self._finish_bridge()
      if planned is not None and self.active_bridge not in self.policies:
        return planned
      spec = self.bridges[self.active_bridge]
      observations = fresh_obs(self.env)
      action = self.policies[self.active_bridge](observations)
      if spec.base is not None and spec.mix is not None:
        base = self.policies[spec.base](observations)
        action = spec.mix(base, action, self.command)
      return action

    self._set_walk()
    action = self.policies["walk"](fresh_obs(self.env))
    self.diagnostic_actual.append(actual[0].cpu().numpy().copy())
    self.diagnostic_planned.append(np.full(actual.shape[-1], np.nan))
    self.diagnostic_phase.append(2)
    self.diagnostic_action.append(action[0].cpu().numpy().copy())
    self.diagnostic_jump_errors.append(np.full(4, np.nan))
    return action


def panel(server, run: Run) -> None:
  with server.gui.add_folder("Handoff"):
    button = server.gui.add_button("Start bridge")
    button.on_click(lambda _: setattr(run, "fire", True))

    bridge = server.gui.add_dropdown(
      "Bridge", tuple(run.bridges), initial_value=run.bridge
    )
    bridge.on_update(lambda _: setattr(run, "bridge", str(bridge.value)))

    automatic = server.gui.add_checkbox("Automatic", initial_value=run.automatic)
    automatic.on_update(lambda _: setattr(run, "automatic", bool(automatic.value)))

    placement = server.gui.add_slider(
      "Walk distance beyond recovery, m",
      min=0.0,
      max=max(3.0, run.cfg.walk_distance),
      step=0.05,
      initial_value=run.cfg.walk_distance,
    )

    @placement.on_update
    def move_entry(_):
      run.cfg = replace(run.cfg, walk_distance=float(placement.value))
      run.reset()

    distance = server.gui.add_slider(
      "Trigger distance, m",
      min=0.0,
      max=max(2.0, run.trigger_distance),
      step=0.05,
      initial_value=run.trigger_distance,
    )
    distance.on_update(
      lambda _: setattr(run, "trigger_distance", float(distance.value))
    )

    low, high = run.command.cfg.duration_s_range
    duration = server.gui.add_slider(
      "Bridge duration, s",
      min=low,
      max=high,
      step=0.05,
      initial_value=run.duration_s,
    )
    duration.on_update(lambda _: setattr(run, "duration_s", float(duration.value)))

    entry = server.gui.add_slider(
      "Walk state",
      min=0,
      max=len(run.entries) - 1,
      step=1,
      initial_value=run.entry_index,
    )
    entry.on_update(lambda _: run.select_entry(int(entry.value)))

  with server.gui.add_folder("Walk"):
    speed = server.gui.add_slider(
      "Speed, m/s",
      min=0.0,
      max=2.0,
      step=0.05,
      initial_value=run.walk_speed,
    )
    speed.on_update(lambda _: setattr(run, "walk_speed", float(speed.value)))


def main() -> None:
  from mjlab import tasks as _tasks

  del _tasks

  cfg = tyro.cli(Config, config=mjlab.TYRO_FLAGS)
  if cfg.bridge not in BRIDGES:
    raise SystemExit(f"Unknown bridge '{cfg.bridge}'. Known: {', '.join(BRIDGES)}")
  if cfg.kinematic_only and cfg.bridge != "diffusion":
    raise SystemExit("--kinematic-only needs --bridge diffusion")
  if cfg.capture_grace_s < 0:
    raise SystemExit("--capture-grace-s cannot be negative")
  if not math.isfinite(cfg.capture_tolerance_scale) or cfg.capture_tolerance_scale <= 0:
    raise SystemExit("--capture-tolerance-scale must be finite and positive")
  if not math.isfinite(cfg.walk_distance) or cfg.walk_distance < 0:
    raise SystemExit("--walk-distance must be finite and nonnegative")
  if not math.isfinite(cfg.trigger_distance) or cfg.trigger_distance < 0:
    raise SystemExit("--trigger-distance must be finite and nonnegative")
  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  selected = BRIDGES[cfg.bridge]
  stage = BRIDGES[selected.base] if selected.base is not None else selected
  env_cfg = arena(stage.task, JUMP_TASK_ID, WALK_TASK_ID)
  if cfg.kinematic_only:
    # Run.debug_vis draws the robots, the bridge target ghosts would clutter it
    env_cfg.commands[BRIDGE] = replace(env_cfg.commands[BRIDGE], debug_vis=False)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(stage.task).clip_actions)
  policies = {
    "jump": load_policy(JUMP_TASK_ID, wrapped, "leaving", device, cfg.jump_checkpoint),
    "walk": load_policy(WALK_TASK_ID, wrapped, "entering", device, cfg.walk_checkpoint),
  }
  available = {name: spec for name, spec in BRIDGES.items() if not spec.learned}
  for name, spec in BRIDGES.items():
    if not spec.learned:
      continue
    if spec.task != stage.task and not (
      spec.base is not None and BRIDGES[spec.base].task == stage.task
    ):
      continue
    checkpoint = cfg.bridge_checkpoint if name == cfg.bridge else None
    if name == selected.base and cfg.imitation_checkpoint is not None:
      checkpoint = cfg.imitation_checkpoint
    try:
      policies[name] = load_policy(
        spec.task,
        wrapped,
        BRIDGE,
        device,
        checkpoint,
        task_runner=False,
      )
      available[name] = spec
    except SystemExit:
      if name == cfg.bridge:
        raise
      print(f"{name:8s} skipped: no checkpoint")

  available = {
    name: spec
    for name, spec in available.items()
    if spec.base is None or spec.base in policies
  }
  if cfg.bridge not in available:
    raise SystemExit(f"Bridge '{cfg.bridge}' is missing its base policy")

  command = env.command_manager.get_term(BRIDGE)
  if not isinstance(command, BridgeCommand):
    raise TypeError(f"{selected.task} does not implement BridgeCommand")
  table = EntryTable.load(cfg.selector_path)
  if not math.isclose(table.fps, 1.0 / env.step_dt):
    raise ValueError("Selector and environment control frequencies differ")
  entries = table.of("walk")
  if not 0 <= cfg.entry < len(entries):
    raise ValueError(f"Walk has {len(entries)} selector states, not entry {cfg.entry}")
  env.reset()
  run = Run(env, policies, entries, cfg, available)
  if cfg.kinematic_only:
    env.manager_visualizers["handoff"] = run

  if cfg.viewer == "none":
    obs = wrapped.get_observations()
    for _ in range(cfg.steps):
      action = run(obs)
      if run.phase == "failed":
        break
      obs, _, _, _ = wrapped.step(action)
    print(
      f"result: phase={run.phase}, landed={run.landed}, "
      f"within_endpoint_box={bool(run.within_endpoint_box)}"
    )
    run.save_diagnostic()
    wrapped.close()
    return

  import viser

  from mjlab.viewer import ViserPlayViewer

  server = viser.ViserServer(label="jump2walk")
  panel(server, run)
  ViserPlayViewer(
    wrapped,
    run,
    viser_server=server,
    info_provider=lambda _: run.status,
    record_name="jump2walk",
  ).run()
  wrapped.close()


if __name__ == "__main__":
  main()
