"""Run the configurable G1 fallen robot and stationary ball demo."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import tyro
import yaml

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.bridges.diffusion.execution.learned_tracker import (
  LearnedTrackerExecutor,
)
from mjlab.tasks.bridging.bridges.imitation.command import Tolerances
from mjlab.tasks.bridging.bridges.interface import Bridge
from mjlab.tasks.bridging.bridges.mixed.runtime import runtime_kind
from mjlab.tasks.bridging.config.g1.demos.soccer.arena import soccer_env_cfg
from mjlab.tasks.bridging.config.g1.demos.soccer.config import CONFIG_PATH, Settings
from mjlab.tasks.bridging.config.g1.demos.soccer.controller import Controller
from mjlab.tasks.bridging.config.g1.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.selector.table import EntryTable
from mjlab.tasks.bridging.tests.stage import load_policy
from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(frozen=True)
class Config:
  config: Path = CONFIG_PATH
  viewer: Literal["viser", "native", "none"] = "viser"
  bridge: Literal["diffusion", "mixed", "no-op"] | None = None
  device: str | None = None
  steps: int = 5000
  seed: int = 0
  dry: bool = False
  diagnostics: Path | None = None


def main(config: Config) -> None:
  from mjlab import tasks as _tasks

  del _tasks
  settings = Settings.load(config.config)
  if config.bridge is not None:
    settings = replace(
      settings,
      bridge=replace(
        settings.bridge,
        kind=config.bridge,
        checkpoint=settings.bridge.checkpoint
        if config.bridge == settings.bridge.kind
        else None,
      ),
    )
  print("walk/run -> jump -> walk/run -> kick")
  print(
    f"Fallen robot: {settings.scene.fallen_position}; ball: {settings.scene.ball_position}; goal: {settings.scene.goal_position}"
  )
  for name, handoff in settings.handoffs.items():
    print(
      f"{name}: entry index={handoff.entry_index}, trigger={handoff.start_distance:g} m, duration={handoff.duration:g} s"
    )
  if config.dry:
    return
  torch.manual_seed(config.seed)
  device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = soccer_env_cfg(settings)
  env_cfg.seed = config.seed
  env = ManagerBasedRlEnv(env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  controller = None
  try:
    policies = {
      skill: load_policy(
        task,
        wrapped,
        skill,
        device,
        Path(checkpoint) if checkpoint else None,
        task_runner=False,
      )
      for skill, task, checkpoint in (
        (
          "walk",
          settings.policies.locomotion_task,
          settings.policies.locomotion_checkpoint,
        ),
        ("jump", JUMP_TASK_ID, settings.policies.jump_checkpoint),
        ("kick", KICK_TASK_ID, settings.policies.kick_checkpoint),
      )
    }
    runtime = Bridge(wrapped.num_actions)
    history, future = 30, 1
    tolerances = Tolerances().tensor(device)
    if settings.bridge.kind != "no-op":
      checkpoint = (
        Path(settings.bridge.checkpoint) if settings.bridge.checkpoint else None
      )
      kind = runtime_kind(settings.bridge.kind, checkpoint)
      diffusion = kind(
        wrapped.num_actions,
        checkpoint=checkpoint,
        sample_steps=settings.bridge.sample_steps,
      )
      checkpoint = checkpoint or diffusion.latest_checkpoint()
      diffusion.checkpoint = checkpoint
      planner = diffusion.load(device)
      if planner.robot != "g1" or not abs(planner.fps - 1.0 / env.step_dt) < 1e-6:
        raise ValueError(
          "Bridge must use the G1 joint layout and environment frequency"
        )
      for name, handoff in settings.handoffs.items():
        ticks = round(handoff.duration * planner.fps)
        if not planner.min_steps <= ticks <= planner.max_steps:
          raise ValueError(
            f"{name}.duration is outside the planner range {planner.min_steps / planner.fps:g} to {planner.max_steps / planner.fps:g} seconds"
          )
      tracker_path = (
        Path(settings.bridge.tracker_checkpoint)
        if settings.bridge.tracker_checkpoint
        else None
      )
      if tracker_path is None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "planner" in saved and "actor_state_dict" in saved:
          tracker_path = checkpoint
      tracker = LearnedTrackerExecutor.load(env, tracker_path)
      diffusion.set_executor(tracker, tracker.within_endpoint_box)
      runtime = diffusion
      history, future = max(planner.history, tracker.history), planner.future
      tolerances = tracker.tolerances
    controller = Controller(
      env,
      policies,
      EntryTable.load(Path(settings.policies.selector_path)),
      runtime,
      settings,
      history,
      future,
      tolerances,
    )
    environment_visualizers = env.update_visualizers

    def draw_scene(visualizer: DebugVisualizer) -> None:
      environment_visualizers(visualizer)
      controller.debug_vis(visualizer)

    env.update_visualizers: Callable[[DebugVisualizer], None] = draw_scene
    if config.viewer == "none":
      obs = wrapped.get_observations()
      for _ in range(config.steps):
        if controller.done:
          break
        obs, _, _, _ = wrapped.step(controller(obs))
      print(controller.status)
      if controller.phase != "done":
        raise RuntimeError(f"Soccer demo did not complete: {controller.status}")
    elif config.viewer == "native":
      from mjlab.viewer import NativeMujocoViewer

      NativeMujocoViewer(wrapped, controller).run()
    else:
      import viser

      from mjlab.viewer import ViserPlayViewer

      server = viser.ViserServer(label="G1 soccer: jump and shoot")
      try:
        ViserPlayViewer(
          wrapped,
          controller,
          viser_server=server,
          info_provider=lambda _: controller.status,
          record_name="g1-soccer",
        ).run()
      finally:
        server.stop()
  finally:
    if config.diagnostics is not None and controller is not None:
      config.diagnostics.parent.mkdir(parents=True, exist_ok=True)
      config.diagnostics.write_text(
        yaml.safe_dump(
          {
            "phase": controller.phase,
            "reason": controller.reason,
            "goal_scored": controller.scored,
            "handoffs": controller.handoffs,
          }
        ),
        encoding="utf-8",
      )
    wrapped.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
