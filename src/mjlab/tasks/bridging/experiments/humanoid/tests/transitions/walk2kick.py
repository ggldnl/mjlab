"""Run a walk to kick handoff.

Run

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick \
      --viewer none --automatic True
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.asset_zoo.objects.ball import BALL_RADIUS
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges import BRIDGES, BridgeSpec
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae import CVAE_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.command import CvaeCommand
from mjlab.tasks.bridging.experiments.humanoid.bridges.interface import BridgeCommand
from mjlab.tasks.bridging.experiments.humanoid.selector import STATES_PATH, resume
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  Entry,
  EntryTable,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import mdp as kick_mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.kick.mdp import KickCommand
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  BRIDGE,
  Policy,
  arena,
  fresh_obs,
  load_policy,
  state,
)
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply, yaw_quat

BALL = "ball"


@dataclass(frozen=True)
class Config:
  bridge: str = "cvae"
  bridge_duration_s: float = 0.8
  trigger_distance: float = 0.5
  automatic: bool = True
  walk_speed: float = 1.0
  ball_distance: float = 3.0
  bridge_checkpoint: Path | None = None
  imitation_checkpoint: Path | None = None
  walk_checkpoint: Path | None = None
  kick_checkpoint: Path | None = None
  selector_path: Path = STATES_PATH
  entry: int = 0
  viewer: Literal["viser", "none"] = "viser"
  steps: int = 600
  device: str | None = None
  seed: int = 0


def _put_ball(env: ManagerBasedRlEnv, position: torch.Tensor) -> None:
  ball: Entity = env.scene[BALL]
  root = torch.zeros(env.num_envs, 13, device=env.device)
  root[:, 0:3] = position
  root[:, 2] = BALL_RADIUS
  root[:, 3] = 1.0
  ball.write_root_state_to_sim(root)
  env.sim.forward()
  kick_mdp.reset_kick_phase(env)


def _reference_ball(env: ManagerBasedRlEnv, command: KickCommand) -> torch.Tensor:
  position = quat_apply(command.anchor_yaw_quat, command.ball_target)
  position[:, :2] += command.anchor_pos + env.scene.env_origins[:, :2]
  position[:, 2] = BALL_RADIUS
  return position


class Run:
  """Drive walk, bridge, then kick in one environment."""

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
      raise TypeError("The walk2kick stage does not implement BridgeCommand")
    self.command = command
    self.motion = kick_mdp.command(env)
    self.robot: Entity = env.scene["robot"]
    self.phase = "walk"
    self.fire = False
    self.automatic = cfg.automatic
    self.trigger_distance = cfg.trigger_distance
    self.duration_s = cfg.bridge_duration_s
    self.walk_speed = cfg.walk_speed
    self.bridges = bridges
    self.bridge = cfg.bridge
    self.active_bridge = cfg.bridge
    self.target = torch.empty(0, device=env.device)
    action_dim = entries[0].previous_action.size
    self.kick_tracking_count = 0
    self.kick_fell = False
    self.kick_tracking_sum = {
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
    self.reset()

  @property
  def distance(self) -> float:
    delta = self.robot.data.root_link_pos_w[0, :2] - self.command.target[0, :2]
    return float(torch.linalg.vector_norm(delta))

  @property
  def status(self) -> str:
    entry = self.entries[self.entry_index].name
    if self.phase == "walk":
      mode = "auto" if self.automatic else "manual"
      return f"walk  {self.distance:.2f} m from {entry}  {mode}"
    if self.phase == "bridge":
      left = max(self.duration_s - float(self.command.step[0]) * self.env.step_dt, 0.0)
      return f"bridge ({self.active_bridge})  {left:.2f} s left"
    return "kick"

  def reset(self) -> None:
    self.phase = "walk"
    self.fire = False
    self.kick_tracking_count = 0
    self.kick_fell = False
    self.kick_tracking_sum = dict.fromkeys(self.kick_tracking_sum, 0.0)
    self.command.stop()
    self.env.action_manager.action.zero_()

    # Ball is always in front of the robot, at distance=self.cfg.ball_distance
    entry = self.entries[self.entry_index]
    resume.prepare(self.env, entry)
    here = state(self.env)
    heading = yaw_quat(here[:, 3:7])
    ahead = torch.zeros(self.env.num_envs, 3, device=self.env.device)
    ahead[:, 0] = self.cfg.ball_distance
    ball = here[:, 0:3] + quat_apply(heading, ahead)
    ball[:, 2] = BALL_RADIUS

    ids = torch.arange(self.env.num_envs, device=self.env.device)
    self.motion.anchor_to_robot(
      ids, start_frame=entry.frame, at_pos=here[:, :3], at_quat=heading
    )
    self.motion.anchor_pos += (ball - _reference_ball(self.env, self.motion))[:, :2]
    self.motion.update_relative_body_poses()
    self.target = resume.target(self.env, entry)
    self._hold_motion(0)
    _put_ball(self.env, ball)
    action = torch.as_tensor(entry.previous_action, device=self.env.device)
    if isinstance(self.command, CvaeCommand):
      if entry.foot_contact is None:
        raise ValueError(
          "Selector entries need recorded foot_contact. Re-run selector.record "
          "and selector.build before using the CVAE bridge"
        )
      contact = torch.as_tensor(entry.foot_contact, device=self.env.device)
      self.command.aim(
        self.target,
        target_contact=contact.expand(self.env.num_envs, -1),
        target_previous_action=action.expand(self.env.num_envs, -1),
      )
    else:
      self.command.aim(self.target)
    self._set_walk()

  def select_entry(self, index: int) -> None:
    """Aim at another selected kick state and restart the handoff."""
    if not 0 <= index < len(self.entries):
      raise IndexError(f"Kick has {len(self.entries)} selector states")
    self.entry_index = index
    self.reset()

  def _hold_motion(self, offset: int) -> None:
    step = self.entries[self.entry_index].frame + offset
    lengths = self.motion.motion.time_step_total_per_motion[self.motion.motion_ids]
    self.motion.time_steps[:] = torch.minimum(
      torch.full_like(self.motion.time_steps, step), lengths - 1
    )
    self.motion.motion_done[:] = False
    self.motion.update_relative_body_poses()

  def _set_walk(self) -> None:
    twist = self.env.command_manager.get_term("twist")
    assert isinstance(twist, UniformVelocityCommand)
    delta = self.command.target[:, :2] - self.robot.data.root_link_pos_w[:, :2]
    twist.vel_command_b[:, 0] = self.walk_speed
    twist.vel_command_b[:, 1:] = 0.0
    twist.heading_target[:] = torch.atan2(delta[:, 1], delta[:, 0])
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
    runtime = self.runtime_bridges.get(self.active_bridge)
    if runtime is not None:
      runtime.reset()
    self.phase = "bridge"
    self.fire = False
    print(
      f"bridge started {self.distance:.2f} m from target for {self.duration_s:.2f} s"
    )

  def _finish_bridge(self, reason: str | None = None):
    errors = self.command.target_errors()[0]
    success = bool((errors <= self.command.tolerances).all())
    cvae = self.command if isinstance(self.command, CvaeCommand) else None
    action_error = float(cvae.action_error()[0]) if cvae is not None else None
    if action_error is not None:
      assert cvae is not None
      success &= action_error <= cvae.cvae_cfg.action_tolerance
    print(
      (reason or ("captured" if not bool(self.command.deadline[0]) else "deadline"))
      + ": "
      + ", ".join(
        f"{name}={float(value):.3f}"
        for name, value in zip(self.command.error_names, errors, strict=True)
      )
      + f", strict_success={success}"
      + (
        f", target_action_error={action_error:.3f}" if action_error is not None else ""
      )
    )
    self.phase = "kick"
    self.command.stop()
    bridge_action = self.env.action_manager.action.clone()
    kick_action = self.policies["kick"](fresh_obs(self.env))
    action_jump = (kick_action - bridge_action).square().mean(dim=-1).sqrt()
    print(f"first kick action jump: {float(action_jump[0]):.3f}")
    return kick_action

  def _record_kick_tracking(self) -> None:
    self.kick_tracking_count += 1
    self.kick_fell |= bool(self.robot.data.projected_gravity_b[0, 2] > -0.2)
    for name in self.kick_tracking_sum:
      self.kick_tracking_sum[name] += float(self.motion.metrics[name][0])
    interval = round(0.5 / self.env.step_dt)
    if self.kick_tracking_count in (interval, 2 * interval):
      print(
        f"kick tracking after {self.kick_tracking_count * self.env.step_dt:.1f}s "
        f"(fell={self.kick_fell}): "
        + ", ".join(
          f"{name}={total / self.kick_tracking_count:.3f}"
          for name, total in self.kick_tracking_sum.items()
        )
      )

  @torch.no_grad()
  def __call__(self, obs):
    del obs
    if self.phase == "walk":
      self._hold_motion(0)
      self._set_walk()
      if self.fire or (self.automatic and self.distance <= self.trigger_distance):
        self._start_bridge()
      else:
        return self.policies["walk"](fresh_obs(self.env))

    if self.phase == "bridge":
      self._hold_motion(0)
      runtime = self.runtime_bridges.get(self.active_bridge)
      if runtime is not None:
        remaining = (
          self.command.window_steps - self.command.step
        ).float() / self.command.fps
        output = runtime(
          self.command.state_now()[:, None], self.command.target[:, None], remaining
        )
        if bool(output.handoff[0]):
          return self._finish_bridge(self.active_bridge)
      if bool(self.command.handoff[0]):
        return self._finish_bridge()
      spec = self.bridges[self.active_bridge]
      observations = fresh_obs(self.env)
      action = self.policies[self.active_bridge](observations)
      if spec.base is not None and spec.mix is not None:
        base = self.policies[spec.base](observations)
        action = spec.mix(base, action, self.command)
      return action

    self._record_kick_tracking()
    return self.policies["kick"](fresh_obs(self.env))


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

    distance = server.gui.add_slider(
      "Trigger distance, m",
      min=0.0,
      max=2.0,
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
      "Kick state",
      min=0,
      max=len(run.entries) - 1,
      step=1,
      initial_value=run.entry_index,
    )
    entry.on_update(lambda _: run.select_entry(int(entry.value)))

  with server.gui.add_folder("Walk"):
    speed = server.gui.add_slider(
      "Speed, m/s",
      min=0.1,
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
  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  selected = BRIDGES[cfg.bridge]
  stage = BRIDGES[selected.base] if selected.base is not None else selected
  env = ManagerBasedRlEnv(
    cfg=arena(stage.task, WALK_TASK_ID, KICK_TASK_ID),
    device=device,
  )
  wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(stage.task).clip_actions)
  policies = {
    "walk": load_policy(WALK_TASK_ID, wrapped, "leaving", device, cfg.walk_checkpoint),
    "kick": load_policy(KICK_TASK_ID, wrapped, "entering", device, cfg.kick_checkpoint),
  }
  available = {name: spec for name, spec in BRIDGES.items() if not spec.learned}
  for name, spec in BRIDGES.items():
    if not spec.learned:
      continue
    if (spec.task == CVAE_TASK_ID) != (stage.task == CVAE_TASK_ID):
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
  entries = EntryTable.load(cfg.selector_path).of("kick")
  if not 0 <= cfg.entry < len(entries):
    raise ValueError(f"Kick has {len(entries)} selector states, not entry {cfg.entry}")
  env.reset()
  run = Run(env, policies, entries, cfg, available)

  if cfg.viewer == "none":
    obs = wrapped.get_observations()
    for _ in range(cfg.steps):
      obs, _, _, _ = wrapped.step(run(obs))
    wrapped.close()
    return

  import viser

  from mjlab.viewer import ViserPlayViewer

  server = viser.ViserServer(label="walk2kick")
  panel(server, run)
  ViserPlayViewer(
    wrapped,
    run,
    viser_server=server,
    info_provider=lambda _: run.status,
    record_name="walk2kick",
  ).run()
  wrapped.close()


if __name__ == "__main__":
  main()
