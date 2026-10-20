"""Run a solver or a supplied Sokoban plan in the shared T1 scene."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
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
from mjlab.tasks.bridging.config.t1.demos.sokoban.config import CONFIG_PATH, Handoffs
from mjlab.tasks.bridging.config.t1.demos.sokoban.controller import (
  Controller,
  Policy,
  Settings,
)
from mjlab.tasks.bridging.config.t1.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.config.t1.skills.turn import TURN_TASK_ID
from mjlab.tasks.bridging.config.t1.skills.turn.command import TurnCommandCfg
from mjlab.tasks.bridging.selector import paths
from mjlab.tasks.bridging.selector.table import EntryTable
from mjlab.tasks.bridging.tests.stage import load_policy
from mjlab.viewer.debug_visualizer import DebugVisualizer

LEVELS = Path(__file__).with_name("levels")
PLANS = Path(__file__).with_name("plans")


@dataclass
class Config:
  config: Path = CONFIG_PATH
  robot: Literal["t1", "g1"] = "t1"
  level: str | None = None
  map: Path | None = None
  plan: Path | None = None
  solver: str | None = None
  walk_task: str | None = None
  walk_checkpoint: Path | None = None
  push_checkpoint: Path | None = None
  turn_task: str | None = None
  turn_checkpoint: Path | None = None
  bridge: Literal["diffusion", "no-op"] = "diffusion"
  bridge_checkpoint: Path | None = None
  tracker_checkpoint: Path | None = None
  selector: Path | None = None
  """Entry points from python -m mjlab.tasks.bridging.selector, the robot's by default."""
  viewer: Literal["viser", "native", "none"] = "viser"
  steps: int = 15000
  device: str | None = None
  seed: int = 0
  result_path: Path | None = None
  result_seeds: tuple[int, ...] = ()
  dry: bool = False
  controller: Settings = field(default_factory=Settings)

  def __post_init__(self) -> None:
    if self.walk_task is None:
      self.walk_task = "Mjlab-G1-Walk-Straight" if self.robot == "g1" else WALK_TASK
    if self.turn_task is None:
      self.turn_task = "Mjlab-G1-Turn" if self.robot == "g1" else TURN_TASK_ID


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


def entry_sets(skills: set[str]) -> dict[str, str]:
  """Selector set behind each entry the controller asks for, keyed by that entry.

  A turn has one set per direction.
  """
  sets = {"walk": "walk", "push": "push"}
  if "turn" in skills:
    sets |= {"turn_left": "turn_left", "turn_right": "turn_right"}
  return sets


def check_entry_sets(
  table: EntryTable,
  robot: str,
  tasks: Mapping[str, str | None],
  pins: Mapping[str, Path | None],
  skills: set[str],
) -> dict[str, str]:
  """The sets the demo reads, refused unless recorded with the tasks and pins it runs.

  Cheap, so it runs before the simulation is built.
  """
  sets = entry_sets(skills)
  for key, name in sets.items():
    skill = key.split("_")[0]
    if table.tasks.get(name) != tasks[skill]:
      raise ValueError(
        f"The selector's {name} set was recorded with {table.tasks.get(name)}, the demo "
        f"runs {tasks[skill]}; run python -m mjlab.tasks.bridging.selector --robot {robot}"
      )
    pinned, recorded = pins[skill], table.checkpoints.get(name)
    if pinned is not None and (
      recorded is None or Path(recorded).resolve() != pinned.resolve()
    ):
      raise ValueError(
        f"The selector's {name} set was recorded with {recorded}, not {pinned}; "
        f"pin it with --checkpoints {name} {pinned} and run the selector again"
      )
  return sets


def selector_entries(
  table: EntryTable, sets: dict[str, str], env: ManagerBasedRlEnv, future: int
) -> dict[str, torch.Tensor]:
  """Each entry point followed by its future, keyed by the entry the controller asks for."""
  if not np.isclose(table.fps, 1.0 / env.step_dt):
    raise ValueError("Selector entry points use a different control frequency")
  width = 13 + 2 * len(env.scene["robot"].joint_names)
  entries = {}
  for key, name in sets.items():
    clips = []
    for entry in table.of(name):
      if entry.future_states is None or entry.future_mask is None:
        raise ValueError(f"The selector's {name} set has no future states")
      clip = np.concatenate((entry.state[None], entry.future_states), axis=0)[:future]
      if (
        len(clip) < future
        or not entry.future_mask[: future - 1].all()
        or clip.shape[1] != width
        or not np.isfinite(clip).all()
      ):
        raise ValueError(f"The selector's {name} set needs {future} finite states")
      clips.append(clip)
    entries[key] = torch.as_tensor(
      np.stack(clips), device=env.device, dtype=torch.float32
    )
  return entries


def main(cfg: Config) -> None:
  from mjlab import tasks as _tasks

  del _tasks
  handoffs = Handoffs.load(cfg.config)
  board = Board.parse(level_path(cfg).read_text(encoding="utf-8"))
  plan = compile_plan(board, read_plan(cfg, board))
  for index, instruction in enumerate(plan, 1):
    print(
      f"{index}: {instruction.action.skill} {instruction.action.direction} {instruction.action.cells} cells"
    )
  for name, handoff in handoffs.switches.items():
    print(
      f"{name}: entry index={handoff.entry_index}, distance={handoff.start_distance:g} m, duration={handoff.duration:g} s"
    )
  if cfg.dry:
    return
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(cfg.seed)
  assert cfg.walk_task is not None
  push_task = PUSH_TASK_ID
  if cfg.robot == "g1":
    from mjlab.tasks.bridging.config.g1.skills.push import PUSH_TASK_ID as G1_PUSH

    push_task = G1_PUSH
  table, sets = None, {}
  if cfg.bridge == "diffusion":
    table = EntryTable.load(cfg.selector or paths(cfg.robot)[1])
    skills = {"walk", "push"} | ({"turn"} if cfg.turn_task is not None else set())
    sets = check_entry_sets(
      table,
      cfg.robot,
      {"walk": cfg.walk_task, "push": push_task, "turn": cfg.turn_task},
      {
        "walk": cfg.walk_checkpoint,
        "push": cfg.push_checkpoint,
        "turn": cfg.turn_checkpoint,
      },
      skills,
    )
    for name, handoff in handoffs.switches.items():
      incoming = name.split("_to_")[1]
      names = ("turn_left", "turn_right") if incoming == "turn" else (incoming,)
      for entry_set in names:
        if entry_set not in sets:
          continue
        count = len(table.of(sets[entry_set]))
        if handoff.entry_index >= count:
          raise ValueError(
            f"{name}.entry_index {handoff.entry_index} is outside the {count} {entry_set} entries"
          )
  env_cfg = sokoban_env_cfg(
    board,
    cfg.walk_task,
    push_task,
    cfg.turn_task,
    wall_collisions=cfg.robot == "g1",
  )
  env_cfg.seed = cfg.seed
  if cfg.robot == "t1" and cfg.turn_task is not None:
    turn_cfg = env_cfg.commands["turn"]
    assert isinstance(turn_cfg, TurnCommandCfg)
    turn_cfg.heading_tolerance = handoffs.turn.command_deadzone
    turn_cfg.heading_control_stiffness = handoffs.turn.heading_gain
  env = ManagerBasedRlEnv(env_cfg, device=device)
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
    if cfg.turn_task is not None:
      policies["turn"] = load_policy(
        cfg.turn_task, wrapped, "turn", device, cfg.turn_checkpoint, task_runner=False
      )
    runtime: Bridge = Bridge(wrapped.num_actions)
    history_length = 30
    future = 1
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
      if planner.robot != cfg.robot or not np.isclose(planner.fps, 1.0 / env.step_dt):
        raise ValueError(
          f"Sokoban requires a {cfg.robot.upper()} bridge checkpoint at {1.0 / env.step_dt:g} Hz"
        )
      for name, handoff in handoffs.switches.items():
        ticks = round(handoff.duration * planner.fps)
        if not planner.min_steps <= ticks <= planner.max_steps:
          raise ValueError(
            f"{name}.duration is outside the planner range {planner.min_steps / planner.fps:g} to {planner.max_steps / planner.fps:g} seconds"
          )
      tracker = LearnedTrackerExecutor.load(
        env, cfg.tracker_checkpoint, robot=cfg.robot
      )
      diffusion.set_executor(tracker, tracker.within_endpoint_box)
      history_length = max(planner.history, tracker.history)
      future = planner.future
      runtime = diffusion
    entries = {}
    if table is not None:
      entries = selector_entries(table, sets, env, future)
    controller = Controller(
      env,
      board,
      plan,
      policies,
      runtime,
      entries,
      history_length,
      cfg.controller,
      handoffs,
    )
    environment_visualizers = env.update_visualizers

    def draw_scene(visualizer: DebugVisualizer) -> None:
      environment_visualizers(visualizer)
      controller.debug_vis(visualizer)

    env.update_visualizers: Callable[[DebugVisualizer], None] = draw_scene
    if cfg.viewer == "none":
      if cfg.result_path is not None:
        from mjlab.tasks.bridging.demo_metrics import run, trials
        from mjlab.tasks.bridging.evaluate import save

        if cfg.result_seeds:
          trials(
            wrapped,
            controller,
            cfg.steps,
            "sokoban",
            cfg.result_path,
            cfg.bridge,
            cfg.result_seeds,
          )
          return
        result = run(wrapped, controller, cfg.steps, "sokoban")
        result.update(seed=cfg.seed, method=cfg.bridge, level=cfg.level or "default")
        save(cfg.result_path, result)
        print(result)
        return
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
