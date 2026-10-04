"""Measure handoff distance and duration against nominal incoming tracking.

Run

    uv run python -m mjlab.tasks.bridging.calibrate --leaving walk --entering climb
    uv run python -m mjlab.tasks.bridging.calibrate --leaving walk --entering kick

Results are saved incrementally as YAML, one row per entry and parameter pair.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
import yaml

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.bridges.diffusion.execution.learned_tracker import (
  LearnedTrackerExecutor,
)
from mjlab.tasks.bridging.bridges.diffusion.execution.runtime import (
  latest_planner_checkpoint,
)
from mjlab.tasks.bridging.bridges.interface import Bridge
from mjlab.tasks.bridging.bridges.mixed.runtime import MixedRuntime, runtime_kind
from mjlab.tasks.bridging.config import get_robot
from mjlab.tasks.bridging.config.g1 import selector as resume
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.bridging.selector import paths
from mjlab.tasks.bridging.selector.table import Entry, EntryTable
from mjlab.tasks.bridging.tests.stage import (
  find_checkpoint,
  fresh_obs,
  load_policy,
  state,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply, yaw_quat

CHANNELS = (
  "error_anchor_pos",
  "error_anchor_rot",
  "error_body_pos",
  "error_body_rot",
  "error_joint_pos",
  "error_joint_vel",
)
ERROR_FLOORS = np.array([0.01, 0.02, 0.01, 0.02, 0.05, 0.1])


@dataclass
class Config:
  robot: str = "g1"
  leaving: str = "walk"
  entering: str = "climb"
  leaving_task: str | None = None
  entering_task: str | None = None
  bridge: Literal["diffusion", "mixed", "no-op"] = "diffusion"
  distances: tuple[float, ...] = (0.2, 0.4, 0.6)
  durations: tuple[float, ...] = (0.4, 0.8, 1.2)
  seeds: tuple[int, ...] = (0, 1, 2)
  entries: tuple[int, ...] = ()
  leaving_entry: int | None = None
  selector_path: Path | None = None
  leaving_checkpoint: Path | None = None
  entering_checkpoint: Path | None = None
  bridge_checkpoint: Path | None = None
  tracker_checkpoint: Path | None = None
  walk_speed: float = 0.8
  warmup_seconds: float = 1.0
  post_seconds: float = 2.0
  sample_steps: int | None = None
  device: str | None = None
  out: Path | None = None
  summarize: Path | None = None
  viser: bool = False
  port: int = 8080
  viewer_speed: float = 1.0
  wait_for_viewer: bool = True


@dataclass
class Rollout:
  completed: bool
  steps: int
  rmse: list[float] | None
  failure: str | None = None


class CalibrationViewer:
  """Display the evaluator's states without taking over simulation stepping"""

  def __init__(self, cfg: Config):
    import viser

    from mjlab.viewer.viser.scene import MjlabViserScene

    self.scene_type = MjlabViserScene
    self.server = viser.ViserServer(port=cfg.port, label="Bridge calibration")
    self.readout = self.server.gui.add_markdown("Waiting for calibration")
    self.pause = self.server.gui.add_checkbox("Pause", initial_value=False)
    self.speed = self.server.gui.add_slider(
      "Playback speed", min=0.1, max=4.0, step=0.1, initial_value=cfg.viewer_speed
    )
    self.stopped = False
    self.started = not cfg.wait_for_viewer
    self.env: ManagerBasedRlEnv | None = None
    self.scene: MjlabViserScene | None = None
    self.entry_frame: viser.FrameHandle | None = None
    self.entry_label: viser.LabelHandle | None = None
    self.last_frame = time.perf_counter()
    stop = self.server.gui.add_button("Stop calibration")

    @stop.on_click
    def _(_) -> None:
      self.stopped = True

    @self.server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
      position = (
        self.env.scene["robot"].data.root_link_pos_w[0].cpu().numpy()
        if self.env is not None
        else np.zeros(3)
      )
      client.camera.position = position + np.array([2.5, -3.0, 1.5])
      client.camera.look_at = position

    print(f"Calibration viewer: http://localhost:{self.server.get_port()}", flush=True)
    if cfg.wait_for_viewer:
      print("Open the viewer to start calibration", flush=True)

  def show(
    self,
    env: ManagerBasedRlEnv,
    message: str,
    *,
    target: torch.Tensor | None = None,
    advance: bool = False,
  ) -> None:
    if self.env is not env:
      self.server.scene.reset()
      self.env = env
      self.entry_frame = None
      self.entry_label = None
      self.scene = self.scene_type(
        self.server,
        env.sim.mj_model,
        num_envs=1,
        sim_model=env.sim.model,
        expanded_fields=env.sim.expanded_fields,
      )
      self.scene.camera_tracking_enabled = False
    assert self.scene is not None
    if bool(torch.isfinite(env.sim.data.xpos).all()) and bool(
      torch.isfinite(env.sim.data.xmat).all()
    ):
      self.scene.update(env.sim.data, env_idx=0)
    if target is not None:
      pose = target[0].cpu().numpy()
      if self.entry_frame is None:
        self.entry_frame = self.server.scene.add_frame(
          "/calibration/entry",
          position=pose[:3],
          wxyz=pose[3:7],
          axes_length=0.25,
          axes_radius=0.01,
        )
        self.entry_label = self.server.scene.add_label(
          "/calibration/entry_label",
          text="Skill entry",
          position=pose[:3] + np.array([0.0, 0.0, 0.3]),
        )
      else:
        self.entry_frame.position = pose[:3]
        self.entry_frame.wxyz = pose[3:7]
        assert self.entry_label is not None
        self.entry_label.position = pose[:3] + np.array([0.0, 0.0, 0.3])
    self.readout.content = message
    while not self.stopped and (
      self.pause.value or (not self.started and not self.server.get_clients())
    ):
      time.sleep(0.05)
    if self.stopped:
      raise KeyboardInterrupt("Calibration stopped in Viser")
    self.started = True
    if advance:
      delay = self.last_frame + env.step_dt / self.speed.value - time.perf_counter()
      if delay > 0:
        time.sleep(delay)
    self.last_frame = time.perf_counter()

  def close(self) -> None:
    self.server.stop()


def tracking_quality(
  trial: Rollout, nominal: Rollout
) -> tuple[float | None, list[float] | None]:
  """One is nominal tracking, larger is better, failed trials score zero"""
  if not nominal.completed or nominal.rmse is None:
    return None, None
  if not trial.completed or trial.rmse is None:
    return 0.0, None
  normal = np.maximum(np.asarray(nominal.rmse), ERROR_FLOORS)
  ratio = np.maximum(np.asarray(trial.rmse), ERROR_FLOORS) / normal
  return float(1.0 / ratio.mean()), ratio.tolist()


def translate_history(
  history: torch.Tensor, target: torch.Tensor, distance: float
) -> torch.Tensor:
  """Place the outgoing state exactly distance metres before a fixed entry"""
  forward = quat_apply(yaw_quat(target[:, 3:7]), target.new_tensor([[1.0, 0.0, 0.0]]))
  reference = history[:, -1, :7]
  placed = torch.cat((reference[:, :3].clone(), yaw_quat(target[:, 3:7])), dim=-1)
  placed[:, :2] = target[:, :2] - distance * forward[:, :2]
  return torch.stack(
    [resume.place(frame, reference, placed) for frame in history.unbind(1)], dim=1
  )


def target_window(env: ManagerBasedRlEnv, entry: Entry, future: int) -> torch.Tensor:
  motion = env.command_manager.get_term("motion")
  assert isinstance(motion, JumpCommand)
  states = [entry.state]
  if future > 1:
    if (
      entry.future_states is None
      or entry.future_mask is None
      or len(entry.future_states) < future - 1
      or not entry.future_mask[: future - 1].all()
    ):
      raise ValueError(
        f"{entry.skill}/{entry.name} lacks {future - 1} consecutive future states; rebuild the selector"
      )
    states.extend(entry.future_states[: future - 1])
  recorded = torch.as_tensor(np.stack(states), device=env.device)
  reference = torch.as_tensor(entry.reference, device=env.device)[None]
  placed = torch.cat((motion.body_pos_w[:, 0], motion.body_quat_w[:, 0]), dim=-1)
  return torch.stack(
    [resume.place(frame[None], reference, placed) for frame in recorded], dim=1
  )


def write_report(path: Path, report: dict) -> None:
  report["best_by_entry"] = entry_recommendations(report)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
  temporary.replace(path)


def recommend(rows: list[dict]) -> dict | None:
  valid = [
    row for row in rows if row["mean_quality"] is not None and row["success_rate"] > 0
  ]
  if not valid:
    return None
  best = max(
    valid,
    key=lambda row: (row["mean_quality"], row["success_rate"], -row["std_quality"]),
  )
  return {
    key: best[key]
    for key in (
      "distance_m",
      "duration_s",
      "mean_quality",
      "std_quality",
      "success_rate",
    )
  }


def entry_recommendations(report: dict) -> list[dict]:
  settings = report.get("settings", {})
  expected = len(settings.get("distances_m", [])) * len(settings.get("durations_s", []))
  recommendations = []
  for entry in report.get("entries", []):
    trials = entry["trials"]
    recommendations.append(
      {
        "index": entry["index"],
        "frame": entry["frame"],
        "motion_file": entry["motion_file"],
        "tested_pairs": len(trials),
        "expected_pairs": expected,
        "grid_complete": len(trials) == expected and expected > 0,
        "best": recommend(trials),
      }
    )
  return recommendations


def print_recommendations(report: dict) -> None:
  for entry in entry_recommendations(report):
    best = entry["best"]
    result = (
      f"x={best['distance_m']:g} m, y={best['duration_s']:g} s, quality={best['mean_quality']:.3f}, success={best['success_rate']:.0%}"
      if best is not None
      else "no successful measured handoff"
    )
    coverage = "complete" if entry["grid_complete"] else "partial"
    print(
      f"Entry {entry['index']} (frame {entry['frame']}): {result} [{coverage}]",
      flush=True,
    )


def pair_grid(report: dict) -> list[dict]:
  combined = []
  for distance in report["settings"]["distances_m"]:
    for duration in report["settings"]["durations_s"]:
      rows = [
        row
        for entry in report["entries"]
        for row in entry["trials"]
        if row["distance_m"] == distance and row["duration_s"] == duration
      ]
      qualities = [
        run["quality"]
        for row in rows
        for run in row["runs"]
        if run["quality"] is not None
      ]
      combined.append(
        {
          "distance_m": distance,
          "duration_s": duration,
          "mean_quality": float(np.mean(qualities)) if qualities else None,
          "std_quality": float(np.std(qualities)) if qualities else None,
          "success_rate": float(np.mean([row["success_rate"] for row in rows]))
          if rows
          else 0.0,
        }
      )
  return combined


def termination_reason(env: ManagerBasedRlEnv) -> str:
  manager = env.termination_manager
  return (
    ", ".join(name for name in manager.active_terms if bool(manager.get_term(name)[0]))
    or "environment terminated"
  )


def evaluation_cfg(task: str, seed: int):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.auto_reset = False
  cfg.seed = seed
  cfg.rewards, cfg.metrics, cfg.curriculum = {}, {}, {}
  cfg.events.pop("push_robot", None)
  for command in cfg.commands.values():
    command.resampling_time_range = (1e9, 1e9)
    command.gui = command.debug_vis = False
  return cfg


class Experiment:
  def __init__(self, cfg: Config, table: EntryTable):
    self.cfg, self.table = cfg, table
    skills = get_robot(cfg.robot).skills
    self.leaving_task = cfg.leaving_task or skills[cfg.leaving]
    self.entering_task = cfg.entering_task or skills[cfg.entering]
    self.device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    self.source = self.incoming = None
    self.viewer: CalibrationViewer | None = None
    self.context = f"{cfg.robot}: {cfg.leaving} → {cfg.entering}"
    try:
      self.source = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(
          evaluation_cfg(self.leaving_task, cfg.seeds[0]), device=self.device
        )
      )
      self.incoming = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(
          evaluation_cfg(self.entering_task, cfg.seeds[0]), device=self.device
        )
      )
      self.env = self.incoming.unwrapped
      motion = (
        self.env.command_manager.get_term("motion")
        if "motion" in self.env.cfg.commands
        else None
      )
      if not isinstance(motion, JumpCommand):
        raise ValueError(
          "Incoming skill must use the clip tracker; reward shaped skills need a task quality metric"
        )
      self.motion = motion
      if not math.isclose(
        self.source.unwrapped.step_dt, self.env.step_dt
      ) or not math.isclose(table.fps, 1 / self.env.step_dt):
        raise ValueError("Skills and selector must use the same control frequency")
      if tuple(self.source.unwrapped.scene["robot"].joint_names) != tuple(
        self.env.scene["robot"].joint_names
      ):
        raise ValueError("Skills use different robot joint orders")
      self.source_action = self.source.unwrapped.action_manager.get_term("joint_pos")
      self.incoming_action = self.env.action_manager.get_term("joint_pos")
      if (
        not isinstance(self.source_action, BaseAction)
        or not isinstance(self.incoming_action, BaseAction)
        or self.source_action.target_names != self.incoming_action.target_names
      ):
        raise ValueError("Calibration requires matching joint position action targets")
      self.leaving_checkpoint = find_checkpoint(
        load_rl_cfg(self.leaving_task).experiment_name, cfg.leaving_checkpoint
      )
      self.entering_checkpoint = find_checkpoint(
        load_rl_cfg(self.entering_task).experiment_name, cfg.entering_checkpoint
      )
      self.source_policy = load_policy(
        self.leaving_task, self.source, "actor", self.device, self.leaving_checkpoint
      )
      self.incoming_policy = load_policy(
        self.entering_task,
        self.incoming,
        "actor",
        self.device,
        self.entering_checkpoint,
      )
      self.runtime: Bridge = Bridge(self.incoming.num_actions)
      self.history_length, self.future = 4, 1
      self.sample_steps = None
      self.bridge_checkpoint = self.tracker_checkpoint = None
      if cfg.bridge != "no-op":
        kind = runtime_kind(cfg.bridge, cfg.bridge_checkpoint)
        runtime = kind(
          self.incoming.num_actions,
          checkpoint=cfg.bridge_checkpoint,
          sample_steps=cfg.sample_steps,
        )
        if isinstance(runtime, MixedRuntime):
          runtime.robot = cfg.robot
        runtime.checkpoint = cfg.bridge_checkpoint or (
          runtime.latest_checkpoint()
          if isinstance(runtime, MixedRuntime)
          else latest_planner_checkpoint(cfg.robot)
        )
        self.bridge_checkpoint = runtime.checkpoint
        planner = runtime.load(self.device)
        self.sample_steps = planner.process.cfg.sample_steps
        if planner.robot != cfg.robot or not math.isclose(planner.fps, table.fps):
          raise ValueError(
            "Bridge checkpoint robot or control frequency does not match the pair"
          )
        for duration in cfg.durations:
          ticks = round(duration / self.env.step_dt)
          if not planner.min_steps <= ticks <= planner.max_steps:
            raise ValueError(
              f"Duration {duration:g}s is outside the planner's trained range"
            )
          if not math.isclose(ticks * self.env.step_dt, duration, abs_tol=1e-6):
            raise ValueError("Durations must be whole control ticks")
        tracker_path = cfg.tracker_checkpoint
        if tracker_path is None:
          saved = torch.load(runtime.checkpoint, map_location="cpu", weights_only=True)
          if "planner" in saved and "actor_state_dict" in saved:
            tracker_path = runtime.checkpoint
        if tracker_path is None:
          from mjlab.tasks.bridging.bridges.diffusion.config import tracker_experiment

          tracker_path = find_checkpoint(tracker_experiment(cfg.robot))
        self.tracker_checkpoint = tracker_path
        tracker = LearnedTrackerExecutor.load(self.env, tracker_path, robot=cfg.robot)
        runtime.set_executor(tracker, tracker.within_endpoint_box)
        self.history_length, self.future = (
          max(planner.history, tracker.history),
          planner.future,
        )
        self.runtime = runtime
      if cfg.viser:
        self.viewer = CalibrationViewer(cfg)
    except BaseException:
      self.close()
      raise

  def close(self) -> None:
    if self.viewer is not None:
      self.viewer.close()
    for env in (self.source, self.incoming):
      if env is not None:
        env.close()

  def draw(
    self,
    message: str,
    env: ManagerBasedRlEnv | None = None,
    *,
    target: torch.Tensor | None = None,
    advance: bool = False,
  ) -> None:
    if self.viewer is not None:
      self.viewer.show(
        self.env if env is None else env,
        f"{self.context}\n\n{message}",
        target=target,
        advance=advance,
      )

  def source_snapshot(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert self.source is not None
    assert isinstance(self.source_action, BaseAction)
    assert isinstance(self.incoming_action, BaseAction)
    torch.manual_seed(seed)
    self.source_policy.reset()
    obs, _ = self.source.reset()
    env = self.source.unwrapped
    self.context = f"{self.cfg.robot}: {self.cfg.leaving} warmup · seed {seed}"
    if self.cfg.leaving_entry is not None:
      entry = self.table.of(self.cfg.leaving)[self.cfg.leaving_entry]
      resume.prepare(env, entry)
      self.put_state(
        env,
        resume.target(env, entry),
        torch.as_tensor(entry.previous_action, device=self.device)[None],
      )
    twist = (
      env.command_manager.get_term("twist") if "twist" in env.cfg.commands else None
    )
    if isinstance(twist, UniformVelocityCommand):
      twist.is_heading_env[:] = twist.is_standing_env[:] = twist.is_world_env[
        :
      ] = twist.is_forward_env[:] = False
      twist.vel_command_b.zero_()
      twist.vel_command_b[:, 0] = self.cfg.walk_speed
    history: deque[torch.Tensor] = deque(maxlen=self.history_length)
    steps = max(self.history_length - 1, round(self.cfg.warmup_seconds / env.step_dt))
    history.append(state(env).clone())
    self.draw("Outgoing skill warmup", env)
    for tick in range(steps):
      obs, _, done, _ = self.source.step(self.source_policy(fresh_obs(env)))
      if bool(done.any()) or not bool(torch.isfinite(state(env)).all()):
        raise RuntimeError(f"Outgoing skill failed during warmup for seed {seed}")
      history.append(state(env).clone())
      self.draw(
        f"Outgoing skill warmup · {(tick + 1) * env.step_dt:.2f} s", env, advance=True
      )
    desired = (
      self.source_action.offset + self.source_action.scale * env.action_manager.action
    )
    action = (desired - self.incoming_action.offset) / self.incoming_action.scale
    return torch.stack(tuple(history), dim=1), action.clone()

  @staticmethod
  def put_state(
    env: ManagerBasedRlEnv, value: torch.Tensor, previous_action: torch.Tensor
  ) -> None:
    robot = env.scene["robot"]
    count = robot.data.joint_pos.shape[1]
    robot.write_root_state_to_sim(value[:, :13])
    robot.write_joint_state_to_sim(value[:, 13 : 13 + count], value[:, 13 + count :])
    robot.reset()
    resume.restore_action(env, previous_action)
    env.sim.forward()

  def prepare(self, entry: Entry, seed: int) -> torch.Tensor:
    assert self.incoming is not None
    torch.manual_seed(seed)
    self.incoming.reset()
    self.incoming_policy.reset()
    self.runtime.reset()
    resume.prepare(self.env, entry)
    # References remain in their trained object frame, independent of distance
    if "ball" in self.env.cfg.scene.entities:
      from mjlab.asset_zoo.objects.ball import BALL_RADIUS
      from mjlab.tasks.bridging.config.g1.skills.kick.mdp import (
        KickCommand,
        reset_kick_phase,
      )

      if not isinstance(self.motion, KickCommand):
        raise ValueError("Ball placement needs the kick clip's contact geometry")
      ball = self.env.scene["ball"]
      position = quat_apply(self.motion.anchor_yaw_quat, self.motion.ball_target)
      position[:, :2] += self.motion.anchor_pos
      position += self.env.scene.env_origins
      root = position.new_zeros((1, 13))
      root[:, :3], root[:, 3] = position, 1.0
      root[:, 2] = self.env.scene.env_origins[:, 2] + BALL_RADIUS
      ball.write_root_state_to_sim(root)
      self.env.sim.forward()
      reset_kick_phase(self.env)
    return target_window(self.env, entry, self.future)

  def tracking(
    self, entry: Entry, steps: int, phase: str = "Nominal tracking"
  ) -> Rollout:
    assert self.incoming is not None
    resume.rewind(self.env, entry.frame)
    values = []
    for tick in range(steps + 1):
      self.motion.update_relative_body_poses()
      self.motion._update_metrics()
      errors = [float(self.motion.metrics[name][0]) for name in CHANNELS]
      self.draw(
        f"{phase} · {tick * self.env.step_dt:.2f} / {steps * self.env.step_dt:.2f} s\n\n"
        + "\n\n".join(
          f"{name}: {value:.4f}" for name, value in zip(CHANNELS, errors, strict=True)
        ),
        advance=tick > 0,
      )
      self.env.termination_manager.compute()
      if not np.isfinite(errors).all() or bool(
        self.env.termination_manager.terminated[0]
      ):
        failure = (
          termination_reason(self.env)
          if np.isfinite(errors).all()
          else "nonfinite tracking"
        )
        self.draw(f"{phase} failed: {failure}")
        return Rollout(False, tick, None, failure)
      values.append(errors)
      if tick == steps:
        break
      _, _, done, _ = self.incoming.step(self.incoming_policy(fresh_obs(self.env)))
      if bool(done.any()):
        failure = termination_reason(self.env)
        self.draw(f"{phase} failed: {failure}")
        return Rollout(False, tick + 1, None, failure)
    return Rollout(True, steps, np.sqrt(np.square(values).mean(axis=0)).tolist())

  def trial(
    self,
    entry: Entry,
    seed: int,
    distance: float,
    duration: float,
    snapshot: tuple[torch.Tensor, torch.Tensor],
    post_steps: int,
  ) -> dict:
    assert self.incoming is not None
    self.context = (
      f"{self.cfg.robot}: {self.cfg.leaving} → {self.cfg.entering} · {entry.motion_file}/{entry.name} · seed {seed}"
      f"\n\nDistance: {distance:g} m · Duration: {duration:g} s"
    )
    target = self.prepare(entry, seed)
    history = translate_history(snapshot[0], target[:, 0], distance)
    self.put_state(self.env, history[:, -1], snapshot[1])
    measured_distance = float((state(self.env)[:, :2] - target[:, 0, :2]).norm())
    endpoint = None
    executed_ticks = 0
    self.draw("Bridge start", target=target[:, 0])
    for _ in range(round(duration / self.env.step_dt) + 2):
      resume.rewind(self.env, entry.frame)
      output = self.runtime(history, target, target.new_tensor([duration]))
      if bool(output.handoff[0]):
        endpoint = (
          bool(output.within_endpoint_box[0])
          if type(self.runtime) is not Bridge
          else None
        )
        break
      _, _, done, _ = self.incoming.step(output.action)
      executed_ticks += 1
      self.draw(
        f"Bridge · {executed_ticks * self.env.step_dt:.2f} / {duration:g} s",
        advance=True,
      )
      actual = state(self.env).clone()
      if (
        bool(done.any())
        or not bool(torch.isfinite(actual).all())
        or float(self.env.scene["robot"].data.projected_gravity_b[0, 2]) > -0.2
      ):
        failure = (
          termination_reason(self.env)
          if bool(done.any())
          else "bridge fell or became nonfinite"
        )
        result = Rollout(False, 0, None, failure)
        break
      history = torch.cat((history[:, 1:], actual[:, None]), dim=1)
    else:
      result = Rollout(False, 0, None, "bridge missed deadline")
    handoff_distance = float((state(self.env)[:, :2] - target[:, 0, :2]).norm())
    if endpoint is not None or type(self.runtime) is Bridge:
      result = self.tracking(entry, post_steps, "Incoming skill after handoff")
    return {
      "seed": seed,
      "measured_distance_m": measured_distance,
      "executed_bridge_seconds": executed_ticks * self.env.step_dt,
      "within_endpoint_box": endpoint,
      "handoff_root_distance_m": handoff_distance,
      "tracking": asdict(result),
    }


@torch.inference_mode()
def main(cfg: Config) -> Path:
  if cfg.summarize is not None:
    report = yaml.safe_load(cfg.summarize.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or "entries" not in report:
      raise ValueError("Expected a calibration report with entries")
    summary = {
      "source": str(cfg.summarize),
      "status": report["status"],
      "robot": report["robot"],
      "pair": report["pair"],
      "best_by_entry": entry_recommendations(report),
    }
    out = cfg.out or cfg.summarize.with_name(f"{cfg.summarize.stem}_entries.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print_recommendations(report)
    print(f"Saved entry recommendations to {out}")
    return out
  from mjlab import tasks as _tasks

  del _tasks
  if (
    not cfg.seeds
    or not cfg.distances
    or not cfg.durations
    or cfg.post_seconds <= 0
    or cfg.warmup_seconds < 0
    or not math.isfinite(cfg.post_seconds)
    or not math.isfinite(cfg.warmup_seconds)
  ):
    raise ValueError("Need seeds, positive post window and a nonempty parameter grid")
  if any(
    not math.isfinite(value) or value <= 0 for value in (*cfg.distances, *cfg.durations)
  ):
    raise ValueError("Distances and durations must be finite and positive")
  if cfg.viser and (not 0.1 <= cfg.viewer_speed <= 4.0 or not 1 <= cfg.port <= 65535):
    raise ValueError("Viewer speed must be in [0.1, 4.0] and port in [1, 65535]")
  table_path = cfg.selector_path or paths(cfg.robot)[1]
  table = EntryTable.load(table_path)
  entries = table.of(cfg.entering)
  indices = cfg.entries or tuple(range(len(entries)))
  if any(index < 0 or index >= len(entries) for index in indices):
    raise ValueError("Incoming entry index is out of range")
  out = (
    cfg.out
    or Path("data/bridging/calibration")
    / f"{cfg.robot}_{cfg.leaving}_to_{cfg.entering}_{cfg.bridge}.yaml"
  )
  experiment = Experiment(cfg, table)
  try:
    report = {
      "schema_version": 1,
      "status": "running",
      "robot": cfg.robot,
      "pair": {"leaving": cfg.leaving, "entering": cfg.entering},
      "bridge": cfg.bridge,
      "selector": str(table_path.resolve()),
      "checkpoints": {
        "leaving": str(experiment.leaving_checkpoint.resolve()),
        "entering": str(experiment.entering_checkpoint.resolve()),
        "bridge": str(experiment.bridge_checkpoint.resolve())
        if experiment.bridge_checkpoint
        else None,
        "tracker": str(experiment.tracker_checkpoint.resolve())
        if experiment.tracker_checkpoint
        else None,
      },
      "metric": {
        "name": "nominal_tracking_retention",
        "larger_is_better": True,
        "nominal_value": 1.0,
        "failed_trial_value": 0.0,
        "channels": list(CHANNELS),
        "channel_units": ["m", "rad", "m", "rad", "joint L2 rad", "joint L2 rad/s"],
        "error_floors": ERROR_FLOORS.tolist(),
        "formula": "1 / mean(max(trial_rmse, floor) / max(nominal_rmse, floor))",
      },
      "settings": {
        "sample_steps": experiment.sample_steps,
        "control_dt": experiment.env.step_dt,
        "device": experiment.device,
        "distances_m": list(cfg.distances),
        "durations_s": list(cfg.durations),
        "seeds": list(cfg.seeds),
        "warmup_seconds": cfg.warmup_seconds,
        "walk_speed": cfg.walk_speed,
        "post_seconds": cfg.post_seconds,
        "leaving_entry": cfg.leaving_entry,
        "source_initialization": "translated actual outgoing rollout and previous action",
      },
      "entries": [],
      "best": None,
    }
    write_report(out, report)
    snapshots = {seed: experiment.source_snapshot(seed) for seed in cfg.seeds}
    for index in indices:
      entry = entries[index]
      entry_target = experiment.prepare(entry, cfg.seeds[0])
      length = int(
        experiment.motion.motion.time_step_total_per_motion[
          experiment.motion.motion_ids
        ][0]
      )
      post_steps = min(
        round(cfg.post_seconds / experiment.env.step_dt), length - entry.frame - 1
      )
      if post_steps < 1:
        raise ValueError(f"Entry {entry.name} has no post handoff reference remaining")
      entry_result = {
        "index": index,
        "frame": entry.frame,
        "motion_file": entry.motion_file,
        "motion_scale": entry.motion_scale,
        "target_root_pose": entry_target[0, 0, :7].tolist(),
        "objects": {
          name: torch.cat(
            (entity.data.root_link_pos_w, entity.data.root_link_quat_w), dim=-1
          )[0].tolist()
          for name, entity in experiment.env.scene.entities.items()
          if name != "robot"
        },
        "post_seconds": post_steps * experiment.env.step_dt,
        "nominal": [],
        "trials": [],
        "best": None,
      }
      report["entries"].append(entry_result)
      nominal = {}
      for seed in cfg.seeds:
        experiment.context = f"{cfg.robot}: {cfg.entering} · {entry.motion_file}/{entry.name} · seed {seed} · nominal"
        target = experiment.prepare(entry, seed)
        experiment.put_state(
          experiment.env,
          target[:, 0],
          torch.as_tensor(entry.previous_action, device=experiment.device)[None],
        )
        experiment.draw("Nominal entry", target=target[:, 0])
        nominal[seed] = experiment.tracking(entry, post_steps)
        entry_result["nominal"].append({"seed": seed, **asdict(nominal[seed])})
      for distance in cfg.distances:
        for duration in cfg.durations:
          trials = []
          for seed in cfg.seeds:
            if not nominal[seed].completed:
              trials.append(
                {"seed": seed, "quality": None, "failure": "nominal tracking failed"}
              )
              continue
            trial = experiment.trial(
              entry, seed, distance, duration, snapshots[seed], post_steps
            )
            quality, ratios = tracking_quality(
              Rollout(**trial["tracking"]), nominal[seed]
            )
            trial.update(quality=quality, error_ratios=ratios)
            trials.append(trial)
          qualities = [
            trial["quality"] for trial in trials if trial["quality"] is not None
          ]
          row = {
            "distance_m": distance,
            "duration_s": duration,
            "mean_quality": float(np.mean(qualities)) if qualities else None,
            "std_quality": float(np.std(qualities)) if qualities else None,
            "success_rate": sum(
              trial.get("tracking", {}).get("completed", False) for trial in trials
            )
            / len(trials),
            "runs": trials,
          }
          entry_result["trials"].append(row)
          entry_result["best"] = recommend(entry_result["trials"])
          print(
            f"{cfg.leaving}->{cfg.entering} {entry.name} x={distance:g} y={duration:g}: quality={row['mean_quality']}, success={row['success_rate']:.2f}",
            flush=True,
          )
          write_report(out, report)
    report["status"] = "complete"
    combined = pair_grid(report)
    report["best"] = recommend(combined)
    report["grid"] = combined
    write_report(out, report)
    print_recommendations(report)
    print(f"Saved calibration to {out}")
    return out
  except BaseException as error:
    if "report" in locals():
      report["status"] = "failed"
      report["error"] = f"{type(error).__name__}: {error}"
      write_report(out, report)
      print_recommendations(report)
    raise
  finally:
    experiment.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
