"""Run the G1 parkour demo.

Run:
  uv run python -m mjlab.tasks.bridging.config.g1.demos.parkour.run
  uv run python -m mjlab.tasks.bridging.config.g1.demos.parkour.run --viewer none
  uv run python -m mjlab.tasks.bridging.config.g1.demos.parkour.run --dry True
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.bridges.dataset.dataset import find_checkpoint
from mjlab.tasks.bridging.bridges.diffusion import TRAIN_EXPERIMENT
from mjlab.tasks.bridging.bridges.diffusion.execution.learned_tracker import (
  LearnedTrackerExecutor,
)
from mjlab.tasks.bridging.bridges.diffusion.execution.runtime import DiffusionRuntime
from mjlab.tasks.bridging.config.g1.demos.parkour.arena import (
  Focus,
  course_env_cfg,
  obstacle_names,
)
from mjlab.tasks.bridging.config.g1.demos.parkour.controller import (
  Controller,
  plan_lines,
)
from mjlab.tasks.bridging.config.g1.demos.parkour.course import Settings, generate
from mjlab.tasks.bridging.config.g1.skills.climb import CLIMB_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.walk import WALK_TASK_ID
from mjlab.tasks.bridging.selector import paths as selector_paths
from mjlab.tasks.bridging.selector.table import EntryTable
from mjlab.tasks.bridging.tests.stage import load_policy
from mjlab.tasks.registry import load_rl_cfg

SELECTOR_PATH = selector_paths("g1")[1]


@dataclass(frozen=True)
class Config:
  config: Path | None = None
  seed: int | None = None
  count: int | None = None
  dry: bool = False
  viewer: Literal["viser", "native", "none"] = "viser"
  steps: int = 12_000
  device: str | None = None
  torch_seed: int = 0
  selector_path: Path = SELECTOR_PATH
  bridge_checkpoint: Path | None = None
  tracker_checkpoint: Path | None = None
  walk_checkpoint: Path | None = None
  jump_checkpoint: Path | None = None
  climb_checkpoint: Path | None = None
  diffusion_sample_steps: int | None = None


def _diffusion_checkpoints(cfg: Config) -> tuple[Path, Path | None]:
  planner = cfg.bridge_checkpoint or find_checkpoint(
    (TRAIN_EXPERIMENT,),
    hint=" Train one with `uv run train Mjlab-G1-Diffusion-Planner-Improvement`.",
  )
  tracker = cfg.tracker_checkpoint
  if tracker is None:
    saved = torch.load(planner, map_location="cpu", weights_only=True)
    if "planner" in saved and "actor_state_dict" in saved:
      tracker = planner
  return planner, tracker


def panel(server, controller: Controller) -> None:
  """The walk2kick handoff controls, applied to every parkour handoff."""
  with server.gui.add_folder("Handoff"):
    button = server.gui.add_button("Start bridge")
    button.on_click(lambda _: setattr(controller, "fire", True))

    automatic = server.gui.add_checkbox("Automatic", initial_value=controller.automatic)
    automatic.on_update(
      lambda _: setattr(controller, "automatic", bool(automatic.value))
    )

    low, high = controller.command.cfg.duration_s_range
    for skill in ("climb", "jump"):
      distance = server.gui.add_slider(
        f"{skill.capitalize()} distance, m",
        min=0.0,
        max=2.0,
        step=0.05,
        initial_value=controller.bridge_distance[skill],
      )
      distance.on_update(
        lambda _, skill=skill, control=distance: controller.bridge_distance.__setitem__(
          skill, float(control.value)
        )
      )
      duration = server.gui.add_slider(
        f"{skill.capitalize()} duration, s",
        min=low,
        max=high,
        step=0.05,
        initial_value=controller.duration_s[skill],
      )
      duration.on_update(
        lambda _, skill=skill, control=duration: controller.duration_s.__setitem__(
          skill, float(control.value)
        )
      )

  with server.gui.add_folder("Walk"):
    speed = server.gui.add_slider(
      "Speed, m/s",
      min=0.1,
      max=2.0,
      step=0.05,
      initial_value=controller.walk_speed,
    )
    speed.on_update(lambda _: setattr(controller, "walk_speed", float(speed.value)))


def main(cfg: Config) -> None:
  from mjlab import tasks as _tasks

  del _tasks
  settings = Settings.load(cfg.config)
  course = generate(settings, seed=cfg.seed, count=cfg.count)
  print("\n".join(plan_lines(course)))
  if cfg.dry:
    return

  torch.manual_seed(cfg.torch_seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  focus = Focus(course, obstacle_names(course))
  env = ManagerBasedRlEnv(course_env_cfg(course, focus), device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(WALK_TASK_ID).clip_actions)
  policies = {
    "walk": load_policy(WALK_TASK_ID, wrapped, "walk", device, cfg.walk_checkpoint),
    "jump": load_policy(JUMP_TASK_ID, wrapped, "jump", device, cfg.jump_checkpoint),
    "climb": load_policy(CLIMB_TASK_ID, wrapped, "climb", device, cfg.climb_checkpoint),
  }
  table = EntryTable.load(cfg.selector_path)
  planner, tracker_checkpoint = _diffusion_checkpoints(cfg)
  tracker = LearnedTrackerExecutor.load(env, tracker_checkpoint)
  runtime = DiffusionRuntime(
    wrapped.num_actions,
    checkpoint=planner,
    sample_steps=cfg.diffusion_sample_steps,
  ).to(device)
  runtime.set_executor(tracker, tracker.within_endpoint_box)
  history_length = max(runtime.load(device).history, tracker.history)
  controller = Controller(
    env, policies, table, runtime, history_length, course, settings, focus
  )
  tracker.tolerances = controller.command.tolerances

  if cfg.viewer == "none":
    obs = wrapped.get_observations()
    for _ in range(cfg.steps):
      if controller.done:
        break
      obs, _, _, _ = wrapped.step(controller(obs))
    else:
      print(f"gave up after {cfg.steps} steps during {controller.status}")
    print(f"result: {controller.phase}, cleared={controller.index}/{len(course)}")
    wrapped.close()
    return

  if cfg.viewer == "native":
    from mjlab.viewer import NativeMujocoViewer

    NativeMujocoViewer(wrapped, controller).run()
  else:
    import viser

    from mjlab.viewer import ViserPlayViewer

    server = viser.ViserServer(label=f"parkour seed {course.seed}")
    panel(server, controller)
    ViserPlayViewer(
      wrapped,
      controller,
      viser_server=server,
      info_provider=lambda _: controller.status,
      record_name=f"parkour-seed-{course.seed}",
    ).run()
  wrapped.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
