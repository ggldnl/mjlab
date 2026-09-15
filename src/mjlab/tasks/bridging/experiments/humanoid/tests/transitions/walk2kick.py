"""Run a walk to docking bridge to kick handoff.

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
from mjlab.tasks.bridging.experiments.humanoid.bridges import BRIDGES
from mjlab.tasks.bridging.experiments.humanoid.bridges.docking.bridge import (
  CHANNELS,
  channel_errors,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.interface import Bridge
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import mdp as kick_mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.kick.mdp import KickCommand
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  BRIDGE,
  Policy,
  TransitionDockingCommand,
  arena,
  fresh_obs,
  load_policy,
  state,
)
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

BALL = "ball"
NO_OP = "no-op"


@dataclass(frozen=True)
class Config:
  bridge: str = "docking"
  bridge_duration_s: float = 0.8
  trigger_distance: float = 0.5
  automatic: bool = True
  walk_speed: float = 1.0
  ball_distance: float = 3.0
  bridge_checkpoint: Path | None = None
  walk_checkpoint: Path | None = None
  kick_checkpoint: Path | None = None
  viewer: Literal["viser", "none"] = "viser"
  steps: int = 600
  device: str | None = None
  seed: int = 0


@dataclass(frozen=True)
class KickSample:
  states: torch.Tensor
  actions: torch.Tensor
  trajectory: torch.Tensor
  phase: int
  ball: torch.Tensor
  reference_root: torch.Tensor
  heading: torch.Tensor


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


@torch.no_grad()
def sample_kick(
  env: ManagerBasedRlEnv,
  policy: Policy,
  bridge: TransitionDockingCommand,
) -> KickSample:
  """Record the opening of the actual kick policy under physics."""
  ids = torch.arange(env.num_envs, device=env.device)
  robot: Entity = env.scene["robot"]
  motion = kick_mdp.command(env)
  heading = yaw_quat(robot.data.root_link_quat_w).clone()
  motion.anchor_to_robot(ids, start_frame=0, at_quat=heading)
  motion.place_on_reference(ids)
  env.sim.forward()
  ball = _reference_ball(env, motion)
  _put_ball(env, ball)

  offsets = bridge.target_offsets
  center = -int(offsets[0])
  required = center + int(offsets[-1]) + 1
  count = max(
    required,
    int(motion.motion.time_step_total_per_motion[motion.motion_ids].max()),
  )
  states: list[torch.Tensor] = []
  actions: list[torch.Tensor] = []
  phases: list[int] = []
  references: list[torch.Tensor] = []
  for _ in range(count):
    states.append(state(env).clone())
    phases.append(int(motion.time_steps[0]))
    references.append(motion.body_pos_w[:, 0].clone())
    action = policy(fresh_obs(env))
    actions.append(action.clone())
    env.step(action)

  rows = center + offsets
  return KickSample(
    states=torch.stack(states, dim=1)[:, rows],
    actions=torch.stack(actions, dim=1)[:, rows],
    trajectory=torch.stack(states, dim=1)[:, center:],
    phase=phases[center],
    ball=ball,
    reference_root=references[center],
    heading=heading,
  )


def _transform(
  states: torch.Tensor,
  turn: torch.Tensor,
  source: torch.Tensor,
  destination: torch.Tensor,
) -> torch.Tensor:
  out = states.clone()
  batch, time, _ = out.shape
  rotation = turn[:, None].expand(batch, time, 4)
  old_origin = source[:, None].expand(batch, time, 3)
  new_origin = destination[:, None].expand(batch, time, 3)
  out[..., 0:3] = new_origin + quat_apply(rotation, states[..., 0:3] - old_origin)
  out[..., 3:7] = quat_mul(rotation, states[..., 3:7])
  out[..., 7:10] = quat_apply(rotation, states[..., 7:10])
  out[..., 10:13] = quat_apply(rotation, states[..., 10:13])
  return out


class Run:
  """Drive walk, bridge, then kick in one environment."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    policies: dict[str, Policy],
    sample: KickSample,
    cfg: Config,
  ) -> None:
    self.env = env
    self.policies = policies
    self.sample = sample
    self.cfg = cfg
    command = env.command_manager.get_term(BRIDGE)
    assert isinstance(command, TransitionDockingCommand)
    self.command = command
    self.motion = kick_mdp.command(env)
    self.robot: Entity = env.scene["robot"]
    self.phase = "walk"
    self.fire = False
    self.automatic = cfg.automatic
    self.trigger_distance = cfg.trigger_distance
    self.duration_s = cfg.bridge_duration_s
    self.walk_speed = cfg.walk_speed
    self.bridge = cfg.bridge
    self.targets = sample.states
    self.target_actions = sample.actions
    self.playback = sample.trajectory
    self.last_kick_action = sample.actions[:, self.command.target_index].clone()
    self.no_op = Bridge(self.last_kick_action.shape[1]).to(env.device)
    self.reset()

  @property
  def distance(self) -> float:
    delta = self.robot.data.root_link_pos_w[0, :2] - self.command.target[0, :2]
    return float(torch.linalg.vector_norm(delta))

  @property
  def status(self) -> str:
    if self.phase == "walk":
      mode = "auto" if self.automatic else "manual"
      return f"walk  {self.distance:.2f} m from target  {mode}"
    if self.phase == "bridge":
      left = max(self.duration_s - float(self.command.step[0]) * self.env.step_dt, 0.0)
      return f"bridge ({self.bridge})  {left:.2f} s left"
    return "kick"

  def reset(self) -> None:
    self.phase = "walk"
    self.fire = False
    self.command.aimed = False
    self.command.stop()
    self.env.action_manager.action.zero_()

    here = state(self.env)
    heading = yaw_quat(here[:, 3:7])
    ahead = torch.zeros(self.env.num_envs, 3, device=self.env.device)
    ahead[:, 0] = self.cfg.ball_distance
    ball = here[:, 0:3] + quat_apply(heading, ahead)
    ball[:, 2] = BALL_RADIUS

    turn = quat_mul(heading, quat_conjugate(self.sample.heading))
    self.targets = _transform(self.sample.states, turn, self.sample.ball, ball)
    self.playback = _transform(self.sample.trajectory, turn, self.sample.ball, ball)
    reference = ball + quat_apply(turn, self.sample.reference_root - self.sample.ball)
    ids = torch.arange(self.env.num_envs, device=self.env.device)
    self.motion.anchor_to_robot(
      ids, start_frame=self.sample.phase, at_pos=reference, at_quat=heading
    )
    self._hold_motion(0)
    _put_ball(self.env, ball)
    self.target_actions = self.sample.actions.clone()
    self.last_kick_action = self.target_actions[:, self.command.target_index].clone()
    self.command.target_sequence[:] = self.targets
    self.command.aimed = True
    self._set_walk()

  def _hold_motion(self, offset: int) -> None:
    step = self.sample.phase + offset
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
    self.command.open_window(ids, self.targets, duration, self.target_actions)
    self.phase = "bridge"
    self.fire = False
    print(
      f"bridge started {self.distance:.2f} m from target for {self.duration_s:.2f} s"
    )

  def _kick_action(self):
    previous = self.env.action_manager.action.clone()
    self.env.action_manager.action[:] = self.last_kick_action
    action = self.policies["kick"](fresh_obs(self.env))
    self.env.action_manager.action[:] = previous
    self.last_kick_action = action.clone()
    return action

  def _finish_bridge(self, reason: str | None = None):
    errors = channel_errors(state(self.env), self.command.target)[0]
    print(
      (reason or ("captured" if bool(self.command.handoff[0]) else "deadline"))
      + ": "
      + ", ".join(
        f"{name}={float(value):.3f}"
        for name, value in zip(CHANNELS, errors, strict=True)
      )
    )
    self.phase = "kick"
    playback_start = max(int(self.motion.time_steps[0]) - self.sample.phase, 0)
    playback_start = min(playback_start, self.playback.shape[1] - 1)
    self.command.start_playback(self.playback[:, playback_start:])
    self.command.stop()
    self.env.action_manager.action[:] = self.last_kick_action
    return self.policies["kick"](fresh_obs(self.env))

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
      self.command.advance()
      age = (
        int((self.command.step - self.command.capture_step)[0])
        if bool(self.command.captured[0])
        else 0
      )
      self._hold_motion(age)
      if self.bridge == NO_OP:
        remaining = (
          self.command.window_steps - self.command.step
        ).float() / self.command.fps
        output = self.no_op(
          self.command.history, self.command.target_sequence, remaining
        )
        if bool(output.handoff[0]):
          return self._finish_bridge(NO_OP)
      if bool(self.command.handoff[0] or self.command.deadline[0]):
        return self._finish_bridge()
      target_action = self._kick_action()
      self.command.target_actions[:] = target_action[:, None]
      return self.policies[self.bridge](fresh_obs(self.env))

    return self.policies["kick"](fresh_obs(self.env))


def panel(server, run: Run) -> None:
  with server.gui.add_folder("Handoff"):
    button = server.gui.add_button("Start bridge")
    button.on_click(lambda _: setattr(run, "fire", True))

    bridge = server.gui.add_dropdown(
      "Bridge",
      (NO_OP, *(name for name in run.policies if name not in ("walk", "kick"))),
      initial_value=run.bridge,
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

    recording = server.gui.add_checkbox(
      "Show kick recording", initial_value=run.command.show_playback
    )
    recording.on_update(
      lambda _: setattr(run.command, "show_playback", bool(recording.value))
    )

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
  if cfg.bridge != NO_OP and cfg.bridge not in BRIDGES:
    raise SystemExit(
      f"Unknown bridge '{cfg.bridge}'. Known: {NO_OP}, {', '.join(BRIDGES)}"
    )
  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  selected = BRIDGES.get(cfg.bridge, BRIDGES["docking"])
  env = ManagerBasedRlEnv(
    cfg=arena(selected.task, WALK_TASK_ID, KICK_TASK_ID), device=device
  )
  wrapped = RslRlVecEnvWrapper(
    env, clip_actions=load_rl_cfg(selected.task).clip_actions
  )
  policies = {
    "walk": load_policy(WALK_TASK_ID, wrapped, "leaving", device, cfg.walk_checkpoint),
    "kick": load_policy(KICK_TASK_ID, wrapped, "entering", device, cfg.kick_checkpoint),
  }
  for name, spec in BRIDGES.items():
    explicit = cfg.bridge_checkpoint if name == cfg.bridge else None
    try:
      policies[name] = load_policy(spec.task, wrapped, BRIDGE, device, explicit)
    except SystemExit:
      if name == cfg.bridge:
        raise

  command = env.command_manager.get_term(BRIDGE)
  assert isinstance(command, TransitionDockingCommand)
  sample = sample_kick(env, policies["kick"], command)
  env.reset()
  run = Run(env, policies, sample, cfg)

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
