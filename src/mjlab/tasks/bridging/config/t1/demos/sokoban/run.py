"""Run a solver or a supplied Sokoban plan in the shared T1 scene."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.bridges.interface import Bridge
from mjlab.tasks.bridging.config.t1.demos.sokoban.arena import (
  WALK_TASK,
  PushPolicy,
  push_rest_pose,
  sokoban_env_cfg,
)
from mjlab.tasks.bridging.config.t1.demos.sokoban.board import (
  Action,
  Board,
  compile_plan,
)
from mjlab.tasks.bridging.config.t1.demos.sokoban.controller import (
  Controller,
  Policy,
  Settings,
)
from mjlab.tasks.bridging.config.t1.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.tests.stage import load_policy

LEVELS = Path(__file__).with_name("levels")
PLANS = Path(__file__).with_name("plans")


@dataclass
class Config:
  robot: Literal["t1", "g1"] = "t1"
  level: str | None = None
  map: Path | None = None
  plan: Path | None = None
  solver: str | None = None
  walk_task: str = WALK_TASK
  walk_checkpoint: Path | None = None
  push_checkpoint: Path | None = None
  bridge: Literal["diffusion", "no-op"] = "diffusion"
  bridge_checkpoint: Path | None = None
  tracker_checkpoint: Path | None = None
  entries: Path | None = None
  viewer: Literal["viser", "native", "none"] = "viser"
  steps: int = 15000
  device: str | None = None
  dry: bool = False
  controller: Settings = field(default_factory=Settings)


def level_path(cfg: Config) -> Path:
  if cfg.map is not None:
    if cfg.level is not None:
      raise ValueError("Provide --level or --map, not both")
    return cfg.map
  name = cfg.level or "default"
  if name in (".", "..") or any(char in name for char in "/\\:"):
    raise ValueError("Level must be a name without a path or extension")
  if Path(name).suffix:
    raise ValueError("Level must be a name without a path or extension")
  return LEVELS / f"{name}.txt"


def read_plan(cfg: Config, board: Board) -> list[Action]:
  if cfg.plan is not None and cfg.solver is not None:
    raise ValueError("Provide a plan file or solver, not both")
  if cfg.solver is not None:
    module, separator, name = cfg.solver.partition(":")
    if not separator:
      raise ValueError("Solver must be module:function")
    actions = getattr(importlib.import_module(module), name)(board)
  elif cfg.plan is not None:
    text = cfg.plan.read_text(encoding="utf-8")
    if cfg.plan.suffix.lower() == ".json":
      actions = json.loads(text)
    else:
      from mjlab.tasks.bridging.config.t1.demos.sokoban.solver import translate_plan

      actions = translate_plan(board, text)
  else:
    path = PLANS / f"{level_path(cfg).stem}.json"
    if not path.is_file():
      raise ValueError(f"No saved plan at {path}; provide --solver or --plan")
    actions = json.loads(path.read_text(encoding="utf-8"))
  result = [
    action if isinstance(action, Action) else Action(**action) for action in actions
  ]
  if cfg.solver is not None:
    compile_plan(board, result)
    PLANS.mkdir(parents=True, exist_ok=True)
    path = PLANS / f"{level_path(cfg).stem}.json"
    path.write_text(
      json.dumps([asdict(action) for action in result], indent=2) + "\n",
      encoding="utf-8",
    )
  return result


def main(cfg: Config) -> None:
  from mjlab import tasks as _tasks

  del _tasks
  board = Board.parse(level_path(cfg).read_text(encoding="utf-8"))
  plan = compile_plan(board, read_plan(cfg, board))
  for index, instruction in enumerate(plan, 1):
    print(
      f"{index}: {instruction.action.skill} {instruction.action.direction} {instruction.action.cells} cells"
    )
  if cfg.dry:
    return
  if cfg.bridge == "diffusion" and cfg.entries is None:
    raise ValueError(
      f"Diffusion needs --entries with recorded {cfg.robot.upper()} walk and push entry states"
    )
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  push_task = PUSH_TASK_ID
  if cfg.robot == "g1":
    from mjlab.tasks.bridging.config.g1.skills.push import PUSH_TASK_ID as G1_PUSH

    push_task = G1_PUSH
  env = ManagerBasedRlEnv(
    sokoban_env_cfg(board, cfg.walk_task, push_task), device=device
  )
  wrapped = RslRlVecEnvWrapper(env)
  try:
    policies: dict[str, Policy] = {
      "walk": load_policy(
        cfg.walk_task, wrapped, "walk", device, cfg.walk_checkpoint, task_runner=False
      ),
      "push": load_policy(
        push_task, wrapped, "push", device, cfg.push_checkpoint, task_runner=False
      ),
    }
    policies["push"] = PushPolicy(env, policies["push"], push_rest_pose(push_task))
    runtime: Bridge = Bridge(wrapped.num_actions)
    history_length = 30
    if cfg.bridge == "diffusion":
      from mjlab.tasks.bridging.bridges.diffusion.execution.learned_tracker import (
        LearnedTrackerExecutor,
      )
      from mjlab.tasks.bridging.bridges.diffusion.execution.runtime import (
        DiffusionRuntime,
        latest_planner_checkpoint,
      )

      diffusion = DiffusionRuntime(
        wrapped.num_actions,
        checkpoint=cfg.bridge_checkpoint or latest_planner_checkpoint(cfg.robot),
      )
      planner = diffusion.load(device)
      if planner.robot != cfg.robot:
        raise ValueError(f"Sokoban requires a {cfg.robot.upper()} bridge checkpoint")
      tracker = LearnedTrackerExecutor.load(
        env, cfg.tracker_checkpoint, robot=cfg.robot
      )
      diffusion.set_executor(tracker, tracker.within_endpoint_box)
      history_length = max(planner.history, tracker.history)
      runtime = diffusion
    entries = {}
    if cfg.entries is not None:
      with np.load(cfg.entries, allow_pickle=False) as data:
        if "joint_names" in data and tuple(data["joint_names"]) != tuple(
          env.scene["robot"].joint_names
        ):
          raise ValueError("Entry recordings use a different joint order")
        if "fps" in data and not np.isclose(float(data["fps"]), 1.0 / env.step_dt):
          raise ValueError("Entry recordings use a different control frequency")
        entries = {
          name: torch.as_tensor(data[name], device=device, dtype=torch.float32)
          for name in ("walk", "push")
        }
      if any(
        entry.ndim != 2
        or entry.shape[0] == 0
        or entry.shape[1] != 13 + 2 * len(env.scene["robot"].joint_names)
        or not bool(torch.isfinite(entry).all())
        for entry in entries.values()
      ):
        raise ValueError(
          f"Entry recordings must contain finite {cfg.robot.upper()} state windows"
        )
      if cfg.bridge == "diffusion" and any(
        len(entry) < planner.future for entry in entries.values()
      ):
        raise ValueError(
          f"Each entry needs at least {planner.future} consecutive states"
        )
    controller = Controller(
      env, board, plan, policies, runtime, entries, history_length, cfg.controller
    )
    if cfg.viewer == "none":
      obs = wrapped.get_observations()
      for _ in range(cfg.steps):
        if controller.done:
          break
        obs, _, _, _ = wrapped.step(controller(obs))
      print(controller.status)
      if controller.phase != "done":
        raise RuntimeError(f"Sokoban did not complete: {controller.status}")
    elif cfg.viewer == "native":
      from mjlab.viewer import NativeMujocoViewer

      NativeMujocoViewer(wrapped, controller).run()
    else:
      import viser

      from mjlab.viewer import ViserPlayViewer

      server = viser.ViserServer(label=f"{cfg.robot.upper()} Sokoban")
      ViserPlayViewer(
        wrapped,
        controller,
        viser_server=server,
        info_provider=lambda _: controller.status,
        record_name=f"{cfg.robot}-sokoban",
      ).run()
  finally:
    wrapped.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
