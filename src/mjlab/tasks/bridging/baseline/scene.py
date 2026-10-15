"""Reuse the experiment scenes and frozen skills for complete transition trials."""

# pyright: reportPrivateUsage=false, reportPrivateImportUsage=false

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
from tensordict import TensorDict

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.termination_manager import TerminationManager
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.baseline.metrics import (
  TRACKING_CHANNELS,
  TRACKING_FLOORS,
  WALK_CHANNELS,
  WALK_FLOORS,
)
from mjlab.tasks.bridging.bridges import BRIDGES
from mjlab.tasks.bridging.bridges.diffusion.config import tracker_experiment
from mjlab.tasks.bridging.bridges.imitation.command import (
  channel_errors,
  upper_body_mask,
)
from mjlab.tasks.bridging.config import get_robot
from mjlab.tasks.bridging.config.g1 import selector as resume
from mjlab.tasks.bridging.config.g1.demos.soccer.controller import entry_sequence
from mjlab.tasks.bridging.config.g1.tests.transitions.jump2walk import (
  Run as JumpWalkRun,
)
from mjlab.tasks.bridging.config.t1.demos.sokoban.controller import place_entry
from mjlab.tasks.bridging.selector import STATES_PATH
from mjlab.tasks.bridging.selector.table import EntryTable
from mjlab.tasks.bridging.tests.stage import (
  Policy,
  arena,
  find_checkpoint,
  fresh_obs,
  load_policy,
  state,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

Pair = Literal["walk2jump", "walk2kick", "jump2walk"]


class _BatchedJumpRun(JumpWalkRun):
  """The jump2walk viewer's reset with one placed walk target per environment."""

  def reset(self) -> None:
    self.env.reset()
    self.airborne = self.landed = False
    self.command.stop()
    self.env.action_manager.action.zero_()
    entry = self.entries[self.entry_index]
    if entry.command is None or entry.command.shape != (3,):
      raise ValueError("Walk entries need their recorded velocity command")
    self.walk_speed = (
      float(entry.command[0]) if self.cfg.walk_speed is None else self.cfg.walk_speed
    )
    if not torch.allclose(
      torch.as_tensor(entry.command), torch.tensor([self.walk_speed, 0.0, 0.0])
    ):
      raise ValueError("walk_speed must match the recorded entry command")
    ids = torch.arange(self.env.num_envs, device=self.env.device)
    heading = torch.zeros(self.env.num_envs, 4, device=self.env.device)
    heading[:, 0] = 1.0
    self.motion.anchor_to_robot(
      ids, start_frame=0, at_pos=self.env.scene.env_origins, at_quat=heading
    )
    self._teleport(self._reference())
    resume.restore_action(self.env, torch.zeros_like(self.env.action_manager.action))
    self.env.sim.forward()
    xy = self.motion.goal_pos_w[:, :2].clone()
    xy[:, 0] += self.cfg.walk_distance
    clip = torch.as_tensor(entry_sequence(entry, self.future), device=self.env.device)
    self.target_window = torch.cat(
      [place_entry(clip, row, self.walk_heading) for row in xy]
    )
    self.target_window[:, :, 2] += self.env.scene.env_origins[:, None, 2]
    self.target = self.target_window[:, 0]
    self.command.aim(self.target)
    self._set_walk()


@dataclass(frozen=True)
class SceneCfg:
  pair: Pair = "walk2jump"
  entry: int = 0
  selector_path: Path = STATES_PATH
  source_checkpoint: Path | None = None
  target_checkpoint: Path | None = None
  walk_speed: float = 0.8
  placement_distance: float = 3.0
  walk_distance: float = 0.1
  initial_joint_noise: float = 0.01
  initial_velocity_noise: float = 0.02
  source_timeout_s: float = 15.0
  post_s: float = 2.0
  device: str = "cuda:0"

  def __post_init__(self) -> None:
    if self.entry < 0:
      raise ValueError("entry must be nonnegative")
    for name in ("walk_speed", "placement_distance", "source_timeout_s", "post_s"):
      if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    for name in ("walk_distance", "initial_joint_noise", "initial_velocity_noise"):
      if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def scene_env_cfg(pair: Pair, num_envs: int = 64):
  source, target = pair.split("2")
  skills = get_robot("g1").skills
  cfg = arena(BRIDGES["no-op"].task, skills[source], skills[target])
  cfg.scene.num_envs = num_envs
  cfg.auto_reset = False
  for command in cfg.commands.values():
    command.debug_vis = False
  return cfg


class TransitionScene:
  """Batched training, single environment matched evaluation.

  Training batches interrupt the source at the first robot's trigger time
  Evaluation uses one environment so every distance trigger is measured exactly
  """

  def __init__(
    self,
    cfg: SceneCfg,
    num_envs: int = 1,
    bridge: str = "no-op",
    bridge_checkpoint: Path | None = None,
    tracker_checkpoint: Path | None = None,
    wrapped: RslRlVecEnvWrapper | None = None,
  ) -> None:
    if num_envs < 1:
      raise ValueError("num_envs must be positive")
    self.cfg = cfg
    self.source, self.target_skill = cfg.pair.split("2")
    skills = get_robot("g1").skills
    self.source_task, self.target_task = skills[self.source], skills[self.target_skill]
    module = importlib.import_module(
      f"mjlab.tasks.bridging.config.g1.tests.transitions.{cfg.pair}"
    )
    spec = BRIDGES[bridge]
    if spec.base is not None:
      raise ValueError("Residual bridge comparison is not supported")
    if wrapped is None:
      env_cfg = arena(spec.task, self.source_task, self.target_task)
      env_cfg.scene.num_envs = num_envs
      env_cfg.auto_reset = False
      for command in env_cfg.commands.values():
        command.debug_vis = False
      self.env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
      wrapped = RslRlVecEnvWrapper(
        self.env, clip_actions=load_rl_cfg(spec.task).clip_actions
      )
    else:
      self.env = wrapped.unwrapped
    self.robot: Entity = self.env.scene["robot"]
    self.wrapped = wrapped
    self.checkpoints = {
      name: find_checkpoint(load_rl_cfg(task).experiment_name, checkpoint).resolve()
      for name, task, checkpoint in (
        (self.source, self.source_task, cfg.source_checkpoint),
        (self.target_skill, self.target_task, cfg.target_checkpoint),
      )
    }
    policies: dict[str, Policy] = {
      self.source: load_policy(
        self.source_task,
        self.wrapped,
        "leaving",
        cfg.device,
        self.checkpoints[self.source],
      ),
      self.target_skill: load_policy(
        self.target_task,
        self.wrapped,
        "entering",
        cfg.device,
        self.checkpoints[self.target_skill],
      ),
    }
    if spec.learned:
      bridge_checkpoint = find_checkpoint(
        load_rl_cfg(spec.task).experiment_name, bridge_checkpoint
      ).resolve()
      policies[bridge] = load_policy(
        spec.task,
        self.wrapped,
        "bridge",
        cfg.device,
        bridge_checkpoint,
        task_runner=False,
      )
    self.policies = policies
    table = EntryTable.load(cfg.selector_path)
    if not math.isclose(table.fps, 1 / self.env.step_dt):
      raise ValueError("Selector and scene control rates differ")
    entries = table.of(self.target_skill)
    if not 0 <= cfg.entry < len(entries):
      raise ValueError(f"{self.target_skill} has {len(entries)} entries")
    run_args: dict[str, Any] = {
      "bridge": bridge,
      "entry": cfg.entry,
      "automatic": False,
      "walk_speed": cfg.walk_speed,
      "bridge_checkpoint": bridge_checkpoint,
      "tracker_checkpoint": tracker_checkpoint,
    }
    if cfg.pair == "walk2jump":
      run_args["jump_distance"] = cfg.placement_distance
    elif cfg.pair == "walk2kick":
      run_args["ball_distance"] = cfg.placement_distance
    else:
      run_args["walk_distance"] = cfg.walk_distance
    self.comparison_checkpoints: dict[str, str] = {}
    if bridge in ("diffusion", "mixed"):
      assert spec.runtime is not None
      planner, tracker = module._diffusion_checkpoints(
        module.Config(**run_args), spec.runtime(self.wrapped.num_actions)
      )
      tracker = find_checkpoint(tracker_experiment("g1"), tracker).resolve()
      run_args["bridge_checkpoint"] = planner.resolve()
      run_args["tracker_checkpoint"] = tracker
      self.comparison_checkpoints = {
        "bridge": str(planner.resolve()),
        "tracker": str(tracker),
      }
    elif spec.learned:
      self.comparison_checkpoints = {"bridge": str(bridge_checkpoint)}
    run_class = (
      _BatchedJumpRun if cfg.pair == "jump2walk" and num_envs > 1 else module.Run
    )
    self.run: Any = run_class(
      self.env, policies, entries, module.Config(**run_args), {bridge: spec}
    )
    self.bridge_name = bridge
    self.spec = spec
    self.upper = upper_body_mask(tuple(self.robot.joint_names), cfg.device)
    # Resolve the skill's own failure conditions without enabling automatic resets
    failure_cfg = {
      name: term
      for name, term in load_env_cfg(self.target_task, play=True).terminations.items()
      if term is not None and not term.time_out
    }
    self.target_failures = TerminationManager(failure_cfg, self.env)
    self.channels = WALK_CHANNELS if self.target_skill == "walk" else TRACKING_CHANNELS
    self.floors = torch.tensor(
      WALK_FLOORS if self.target_skill == "walk" else TRACKING_FLOORS, device=cfg.device
    )
    self.failed = torch.zeros(num_envs, dtype=torch.bool, device=cfg.device)
    self.history: list[torch.Tensor] = []
    self.last_request: dict = {}

  def observe(self) -> torch.Tensor:
    current = state(self.env).clone()
    current[:, :3] -= self.env.scene.env_origins
    return torch.cat((current, self.env.action_manager.action), -1)

  def capture_request(self) -> dict:
    """Capture one source interruption for identical independent method trials."""
    return {
      "state": state(self.env).clone(),
      "action": self.env.action_manager.action.clone(),
      "history": [item.clone() for item in self.history],
      "target": self.run.target.clone(),
      "failed": self.failed.clone(),
      "previous_action": self.env.action_manager.prev_action.clone(),
      "previous_previous_action": self.env.action_manager.prev_prev_action.clone(),
      "motion_step": self.run.motion.time_steps.clone(),
      "motion_done": self.run.motion.motion_done.clone(),
      "objects": {
        name: torch.cat(
          (
            entity.data.root_link_pos_w,
            entity.data.root_link_quat_w,
            entity.data.root_link_lin_vel_w,
            entity.data.root_link_ang_vel_w,
          ),
          -1,
        ).clone()
        for name, entity in self.env.scene.entities.items()
        if name != "robot" and not entity.is_fixed_base
      },
    }

  def restore_request(self, request: dict) -> None:
    if not torch.allclose(self.run.target, request["target"], rtol=1e-5, atol=1e-5):
      raise ValueError("Captured request and method have different target placement")
    self.run._teleport(request["state"])
    resume.restore_action(self.env, request["action"])
    self.env.action_manager.prev_action.copy_(request["previous_action"])
    self.env.action_manager.prev_prev_action.copy_(request["previous_previous_action"])
    self.run.target = request["target"].clone()
    self.run.command.target.copy_(self.run.target)
    self.run.motion.time_steps.copy_(request["motion_step"])
    self.run.motion.motion_done.copy_(request["motion_done"])
    self.run.motion.update_relative_body_poses()
    for name, root in request["objects"].items():
      self.env.scene[name].write_root_state_to_sim(root)
    self.failed.copy_(request["failed"])
    self.history = [item.clone() for item in request["history"]]
    self.env.sim.forward()

  def skill_action(self, name: str) -> torch.Tensor:
    obs = TensorDict(fresh_obs(self.env), batch_size=[self.env.num_envs])
    return self.policies[name](obs)

  def select_entry(self, index: int) -> None:
    if not 0 <= index < len(self.run.entries):
      raise ValueError(f"Entry {index} does not exist")
    self.cfg = replace(self.cfg, entry=index)
    self.run.entry_index = index

  def fallen(self) -> torch.Tensor:
    current = state(self.env)
    return (
      ~torch.isfinite(current).all(-1)
      | (self.robot.data.projected_gravity_b[:, 2] > -0.2)
      | (current[:, 2] - self.env.scene.env_origins[:, 2] < 0.25)
    )

  def hold_target(self) -> None:
    if self.target_skill != "walk":
      self.run._hold_motion(0)
    self.run._set_walk()

  @torch.no_grad()
  def reset(self, seed: int) -> None:
    torch.manual_seed(seed)
    self.env.reset(seed=seed)
    self.run.reset()
    for policy in self.policies.values():
      policy.reset()
    q = self.robot.data.joint_pos.clone()
    q += self.cfg.initial_joint_noise * torch.randn_like(q)
    v = self.robot.data.joint_vel.clone()
    v += self.cfg.initial_velocity_noise * torch.randn_like(v)
    self.robot.write_joint_state_to_sim(q, v)
    self.env.sim.forward()
    self.failed.zero_()
    self.history.clear()
    self.target_failures.reset()

  def step(self, action: torch.Tensor) -> None:
    if not bool(torch.isfinite(action).all()):
      raise RuntimeError("Nonfinite action during transition trial")
    self.wrapped.step(action)
    if not bool(torch.isfinite(state(self.env)).all()):
      raise RuntimeError("Simulation became nonfinite during transition trial")
    self.failed |= self.fallen()
    self.history.append(state(self.env).clone())
    self.history = self.history[-256:]

  @torch.no_grad()
  def interrupt(self, distance: float) -> None:
    if not math.isfinite(distance) or distance <= 0:
      raise ValueError("distance must be finite and positive")
    for _ in range(math.ceil(self.cfg.source_timeout_s / self.env.step_dt)):
      if self.source == "walk":
        self.hold_target()
        ready = self.run.distance <= distance
      else:
        self.run._set_walk()
        ready = self.run._landed() and self.run.distance <= distance
      if ready:
        return
      if bool(self.failed.all()):
        return
      self.step(self.skill_action(self.source))
    self.failed[:] = True

  @torch.no_grad()
  def nominal(self, perturb: float = 0.0) -> None:
    target = self.run.target.clone()
    joints = self.robot.data.joint_pos.shape[-1]
    target[:, 13 : 13 + joints] += perturb * torch.randn_like(
      target[:, 13 : 13 + joints]
    )
    self.run._teleport(target)
    action = torch.as_tensor(
      self.run.entries[self.cfg.entry].previous_action, device=self.cfg.device
    )
    resume.restore_action(self.env, action.expand(self.env.num_envs, -1))
    self.env.sim.forward()
    self.failed.zero_()
    self.history = [state(self.env).clone()]

  @torch.no_grad()
  def endpoint(self) -> tuple[torch.Tensor, torch.Tensor]:
    errors = channel_errors(state(self.env), self.run.target, self.upper)
    return errors, (errors <= self.run.command.tolerances).all(-1)

  def joint_targets(self, action: torch.Tensor) -> torch.Tensor:
    """Joint position commands after action scaling and clipping, in radians."""
    term = self.env.action_manager.get_term("joint_pos")
    if not isinstance(term, JointPositionAction):
      raise TypeError("Handoff measurement requires joint position actions")
    if self.wrapped.clip_actions is not None:
      action = action.clamp(-self.wrapped.clip_actions, self.wrapped.clip_actions)
    targets = action * term.scale + term.offset
    if term.cfg.clip is not None:
      targets = targets.clamp(term._clip[:, :, 0], term._clip[:, :, 1])
    return targets

  @torch.no_grad()
  def successor(
    self, collect: bool = False
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    self.hold_target()
    self.run.command.stop()
    last_action = self.env.action_manager.action.clone()
    first_action = self.skill_action(self.target_skill)
    term = self.env.action_manager.get_term("joint_pos")
    assert isinstance(term, JointPositionAction)
    scale = torch.as_tensor(term.scale, device=self.cfg.device).expand_as(last_action)
    active = scale[0].abs() > 0
    if not bool(active.any()):
      raise ValueError("No active joint position commands")
    self.handoff_joint_names = [
      name
      for name, used in zip(term.target_names, active.tolist(), strict=True)
      if used
    ]
    self.handoff_delta = (
      self.joint_targets(first_action) - self.joint_targets(last_action)
    )[:, active]
    self.handoff_jump_raw = (first_action - last_action).square().mean(-1).sqrt()
    jump = self.handoff_delta.square().mean(-1).sqrt()
    self.post_steps = 0
    self.successor_observations: list[torch.Tensor] = []
    self.successor_actions: list[torch.Tensor] = []
    sums = torch.zeros(self.env.num_envs, len(self.channels), device=self.cfg.device)
    steps = max(1, round(self.cfg.post_s / self.env.step_dt))
    if self.target_skill != "walk":
      remaining = (
        self.run.motion.motion.time_step_total_per_motion[self.run.motion.motion_ids]
        - self.run.motion.time_steps
      )
      steps = min(steps, int(remaining.min()))
    if steps < 1:
      raise ValueError("Entry has no remaining reference frames")
    for index in range(steps):
      action = first_action if index == 0 else self.skill_action(self.target_skill)
      if collect:
        self.successor_observations.append(self.observe().clone())
        self.successor_actions.append(action.clone())
      self.step(action)
      self.post_steps += 1
      self.failed |= self.target_failures.compute()
      if self.target_skill == "walk":
        data = self.robot.data
        errors = torch.stack(
          (
            data.root_link_lin_vel_b[:, 0] - self.run.walk_speed,
            data.root_link_lin_vel_b[:, 1],
            data.root_link_ang_vel_b[:, 2],
          ),
          -1,
        )
      else:
        errors = torch.stack(
          [self.run.motion.metrics[name] for name in self.channels], -1
        )
      sums += errors.square()
    rms = (sums / steps).sqrt()
    self.failed |= ~torch.isfinite(rms).all(-1)
    return rms, ~self.failed, jump

  @torch.no_grad()
  def bridge_action(self, duration: float) -> tuple[torch.Tensor, bool]:
    self.hold_target()
    runtime = self.run.runtime_bridges.get(self.bridge_name)
    if runtime is None:
      handoff = bool(self.run.command.handoff.all())
      return self.env.action_manager.action.clone() if handoff else self.skill_action(
        self.bridge_name
      ), handoff
    history = torch.stack(self.history[-self.run.history_length :], 1)
    if history.shape[1] < self.run.history_length:
      history = torch.cat(
        (
          history[:, :1].expand(-1, self.run.history_length - history.shape[1], -1),
          history,
        ),
        1,
      )
    output = runtime(
      history,
      self.run.target_window,
      torch.full((self.env.num_envs,), duration, device=self.cfg.device),
    )
    return output.action, bool(output.handoff.all())

  def open_bridge(self, duration: float) -> None:
    self.run.duration_s = duration
    self.run._start_bridge()
    if not self.history:
      self.history.append(state(self.env).clone())

  def close(self) -> None:
    self.wrapped.close()
