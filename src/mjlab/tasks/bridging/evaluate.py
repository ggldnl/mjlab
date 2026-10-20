"""Run the chapter 7 comparisons and export their measured tables.

Run:
  uv run python -m mjlab.tasks.bridging.evaluate
  uv run python -m mjlab.tasks.bridging.evaluate --suite transitions
  uv run python -m mjlab.tasks.bridging.evaluate --suite tables
"""

# pyright: reportPrivateUsage=false, reportPrivateImportUsage=false

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
import yaml

import mjlab
from mjlab.tasks.bridging.baseline.model import fingerprint
from mjlab.tasks.bridging.baseline.scene import SceneCfg, TransitionScene
from mjlab.tasks.bridging.calibrate import translate_history
from mjlab.tasks.bridging.config import get_robot
from mjlab.tasks.bridging.config.g1.demos.soccer.config import Settings
from mjlab.tasks.bridging.selector.table import EntryTable

ROOT = Path("data/bridging/results")
TABLES = Path("src/mjlab/tasks/bridging/Masters-Thesis/thesis/tables")
PAIRS = ("walk2kick", "walk2jump", "jump2walk")
LABELS = ("Walk to kick", "Walk to jump", "Jump to walk")


@dataclass
class Config:
  suite: Literal["all", "transitions", "demos", "tables"] = "all"
  output: Path = ROOT
  seeds: tuple[int, ...] = tuple(range(50))
  distances: tuple[float, ...] = (0.2, 0.4, 0.6)
  durations: tuple[float, ...] = (0.4, 0.8, 1.2)
  device: str = "cuda:0"


def score(measure: float, nominal: float, larger: bool) -> float | None:
  """Return the chapter's ratio, leaving a zero denominator undefined."""
  if any(not math.isfinite(value) or value < 0 for value in (measure, nominal)):
    raise ValueError("Task measures must be finite and nonnegative")
  numerator, denominator = (measure, nominal) if larger else (nominal, measure)
  return numerator / denominator if denominator > 0 else None


def summary(rows: list[dict]) -> tuple[float, float | None]:
  if not rows:
    raise ValueError("Cannot summarize an empty experiment")
  successful = [row for row in rows if row["success"]]
  values = [row["score"] for row in successful if row["score"] is not None]
  mean = float(np.mean(values)) if values and len(values) == len(successful) else None
  return len(successful) / len(rows), mean


def best_settings(rows: list[dict], entry: int | None = None) -> dict | None:
  groups = {
    (row["entry"], row["distance"], row["budget_s"])
    for row in rows
    if row["method"] == "diffusion" and (entry is None or row["entry"] == entry)
  }
  candidates = []
  for index, distance, duration in sorted(groups):
    rate, quality = summary(
      [
        row
        for row in rows
        if row["method"] == "diffusion"
        and row["entry"] == index
        and row["distance"] == distance
        and row["budget_s"] == duration
      ]
    )
    if quality is not None:
      candidates.append(
        {
          "entry": index,
          "distance": distance,
          "duration": duration,
          "score": quality,
          "success_rate": rate,
        }
      )
  return (
    max(
      candidates,
      key=lambda item: (item["score"], item["success_rate"], -item["duration"]),
    )
    if candidates
    else None
  )


def best_entry(
  rows: list[dict], distance: float, duration: float
) -> tuple[float, float | None]:
  """Success rate and score of the bridge's best entry point for one grid cell.

  Same ranking as best_settings: score first, then success rate. Entries without a
  score only win when no entry has one.
  """
  bridge = [
    row
    for row in rows
    if row["method"] == "diffusion"
    and row["distance"] == distance
    and row["budget_s"] == duration
  ]
  if not bridge:
    raise ValueError("Cannot summarize an empty grid cell")
  cells = [
    summary([row for row in bridge if row["entry"] == entry])
    for entry in sorted({row["entry"] for row in bridge})
  ]
  return max(cells, key=lambda cell: (cell[1] is not None, cell[1] or 0.0, cell[0]))


def save(path: Path, report: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(".tmp")
  temporary.write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
  )
  temporary.replace(path)


def source_request(scene: TransitionScene, cfg: Config) -> dict:
  """Record the source once per seed, then place it at each test distance."""
  scene.reset(0)
  robot = scene.robot
  q, v = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
  for index, seed in enumerate(cfg.seeds):
    generator = torch.Generator(device=cfg.device).manual_seed(seed)
    dq = 0.01 * torch.randn(q.shape[-1], device=cfg.device, generator=generator)
    dv = 0.02 * torch.randn(v.shape[-1], device=cfg.device, generator=generator)
    q[index :: len(cfg.seeds)] += dq
    v[index :: len(cfg.seeds)] += dv
  robot.write_joint_state_to_sim(q, v)
  scene.env.sim.forward()
  airborne = torch.zeros(scene.env.num_envs, dtype=torch.bool, device=cfg.device)
  warmup = max(scene.run.history_length, round(1.0 / scene.env.step_dt))
  for tick in range(round(scene.cfg.source_timeout_s / scene.env.step_dt)):
    if scene.source == "walk":
      if tick >= warmup:
        break
      scene.hold_target()
    else:
      grounded = (
        scene.env.scene["feet_ground_contact"]
        .data.found.reshape(scene.env.num_envs, -1)
        .gt(0)
        .any(-1)
      )
      height = (
        robot.data.body_link_pos_w[:, scene.run.feet, 2]
        - scene.env.scene.env_origins[:, 2, None]
      )
      airborne |= ~grounded & (height.min(-1).values > 0.08)
      if bool((airborne & grounded & scene.run.motion.has_landed).all()):
        break
      scene.run._set_walk()
    scene.step(scene.skill_action(scene.source))
    if bool(scene.failed.any()):
      raise RuntimeError(
        "Source skill failed before the request; no switch can be scored"
      )
  else:
    raise RuntimeError("Source skill did not reach the request")
  request = scene.capture_request()
  distances = torch.tensor(cfg.distances, device=cfg.device).repeat_interleave(
    len(cfg.seeds)
  )
  placed = translate_history(torch.stack(scene.history, 1), scene.run.target, distances)
  request["source_state"] = request["state"].clone()
  request["state"] = placed[:, -1].clone()
  request["history"] = list(placed.unbind(1))
  return request


def batch_trial(
  scene: TransitionScene, request: dict, method: str, duration: float, cfg: Config
) -> list[dict]:
  if method == "direct" and (cfg.distances != (0.0,) or duration != 0.0):
    raise ValueError("Direct switches have no distance or duration grid")
  scene.reset(0)
  scene.restore_request(request)
  handoff = method != "diffusion"
  ticks = 0
  if method == "nominal":
    scene.nominal()
  elif method == "diffusion":
    torch.manual_seed(0)
    scene.open_bridge(duration)
    budget = round(duration / scene.env.step_dt)
    for tick in range(budget + 1):
      action, handoff = scene.bridge_action(duration)
      if handoff or tick == budget:
        break
      scene.step(action)
      ticks += 1
  if handoff:
    bridge_failed = scene.failed.clone()
    _, successes, _ = scene.successor()
  else:
    bridge_failed = scene.failed.clone()
    successes = torch.zeros_like(scene.failed)
  rows = []
  for index in range(scene.env.num_envs):
    rows.append(
      {
        "method": method,
        "seed": cfg.seeds[index % len(cfg.seeds)],
        "entry": scene.cfg.entry,
        "distance": None
        if method == "direct"
        else cfg.distances[index // len(cfg.seeds)],
        "budget_s": None if method == "direct" else duration,
        "duration_s": ticks * scene.env.step_dt,
        "handoff": handoff,
        "success": bool(successes[index]),
        "bridge_failed": bool(bridge_failed[index]),
        "failure_terms": [
          name for name, mask in scene.failure_masks.items() if bool(mask[index])
        ]
        if handoff
        else [],
        "task_measure": float(scene.task_measure[index]) if handoff else None,
        "post_steps": scene.post_steps if handoff else 0,
        "request_state": request["state"][index].cpu().tolist(),
        "source_state": request["source_state"][index].cpu().tolist(),
        "target": request["target"][index].cpu().tolist(),
        "batch_index": index,
        "planner_seed": 0,
        "frame": scene.run.entries[scene.cfg.entry].frame,
      }
    )
  return rows


@torch.inference_mode()
def transitions(cfg: Config) -> None:
  from mjlab import tasks as _tasks

  del _tasks
  settings = Settings.load(
    Path("src/mjlab/tasks/bridging/config/g1/demos/soccer/config.yml")
  )
  table = EntryTable.load(Path(settings.policies.selector_path))
  for pair in PAIRS:
    path = cfg.output / f"{pair}.json"
    if path.exists():
      previous = json.loads(path.read_text(encoding="utf-8"))
      if any(
        row["method"] == "direct" and row["distance"] is not None
        for row in previous["trials"]
      ):
        raise ValueError(
          "Old Direct results used the bridge distance grid; choose a new output directory"
        )
      if previous["status"] == "complete" and all(
        previous["settings"][name] == list(getattr(cfg, name))
        for name in ("seeds", "distances", "durations")
      ):
        for name, pin in previous["artifacts"].items():
          if fingerprint(Path(pin)) != previous["sha256"][name]:
            raise ValueError(f"Checkpoint changed since {path}")
        if (
          fingerprint(Path(settings.policies.selector_path))
          != previous["sha256"]["selector"]
        ):
          raise ValueError("Selector changed since the completed comparison")
        print(f"Reusing completed {path}", flush=True)
        continue
      raise FileExistsError(
        f"Refusing to replace {path}; choose a new --output directory"
      )
    source, target = pair.split("2")
    pins = {skill: table.checkpoints[skill] for skill in ("walk", "jump", "kick")}
    if settings.bridge.checkpoint is None or settings.bridge.tracker_checkpoint is None:
      raise ValueError("Pin the planner and tracker in the soccer configuration")
    scene_cfg = SceneCfg(
      pair=pair,  # type: ignore[arg-type]
      selector_path=Path(settings.policies.selector_path),
      source_checkpoint=Path(pins[source]),
      target_checkpoint=Path(pins[target]),
      walk_distance=0.6,
      initial_joint_noise=0.0,
      initial_velocity_noise=0.0,
      device=cfg.device,
    )
    for skill in (source, target):
      if table.tasks.get(skill) != get_robot("g1").skills[skill]:
        raise ValueError(f"Selector task mismatch for {skill}")
    report = {
      "status": "running",
      "pair": pair,
      "settings": asdict(cfg),
      "checkpoints": pins,
      "nominal": [],
      "trials": [],
      "metric": {
        "walk": "mean L2 error of body forward/lateral velocity and yaw rate",
        "jump": "mean tracked body position distance in metres",
        "kick": "peak planar ball speed after recorded contact in m/s",
        "score": "trial / nominal for kick, nominal / trial otherwise; successful trials only",
      }[target],
    }
    report["settings"] = {**asdict(cfg), "output": str(cfg.output)}
    scene = TransitionScene(
      scene_cfg,
      num_envs=len(cfg.seeds) * len(cfg.distances),
      bridge="diffusion",
      bridge_checkpoint=Path(settings.bridge.checkpoint),
      tracker_checkpoint=Path(settings.bridge.tracker_checkpoint),
    )
    direct_scene = TransitionScene(scene_cfg, num_envs=len(cfg.seeds))
    direct_cfg = replace(cfg, distances=(0.0,))
    report["artifacts"] = {
      **{k: str(v) for k, v in scene.checkpoints.items()},
      **scene.comparison_checkpoints,
    }
    report["sha256"] = {
      name: fingerprint(Path(path)) for name, path in report["artifacts"].items()
    }
    report["sha256"]["selector"] = fingerprint(scene_cfg.selector_path)
    try:
      save(path, report)
      for index, _entry in enumerate(table.of(target)):
        scene.select_entry(index)
        if target != "walk":
          lengths = scene.run.motion.motion.time_step_total_per_motion
          scene.cfg = replace(
            scene.cfg, post_s=float(lengths.max()) * scene.env.step_dt
          )
        request = source_request(scene, cfg)
        recording = cfg.output / f"{pair}_entry{index}_request.pt"
        torch.save(request, recording)
        nominal = batch_trial(scene, request, "nominal", 0.0, cfg)
        report["nominal"].extend(nominal)
        direct_scene.select_entry(index)
        direct_scene.cfg = replace(direct_scene.cfg, post_s=scene.cfg.post_s)
        direct_scene.reset(0)
        direct_request = direct_scene.capture_request()
        count = len(cfg.seeds)
        history = torch.stack(request["history"], 1)[:count].clone()
        origins = (
          direct_scene.env.scene.env_origins - scene.env.scene.env_origins[:count]
        )
        history[:, :, :3] += origins[:, None, :]
        placed = translate_history(history, direct_request["target"], 0.0)
        direct_request["source_state"] = request["source_state"][:count].clone()
        direct_request["source_state"][:, :3] += origins
        direct_request["state"] = placed[:, -1].clone()
        direct_request["history"] = list(placed.unbind(1))
        for name in ("action", "previous_action", "previous_previous_action"):
          direct_request[name] = request[name][:count].clone()
        direct_nominal = batch_trial(
          direct_scene, direct_request, "nominal", 0.0, direct_cfg
        )
        direct = batch_trial(direct_scene, direct_request, "direct", 0.0, direct_cfg)
        for method, duration in [("direct", 0.0)] + [
          ("diffusion", value) for value in cfg.durations
        ]:
          rows = (
            direct
            if method == "direct"
            else batch_trial(scene, request, method, duration, cfg)
          )
          references = direct_nominal if method == "direct" else nominal
          for row, reference in zip(rows, references, strict=True):
            row["score"] = (
              score(row["task_measure"], reference["task_measure"], target == "kick")
              if row["success"] and reference["success"]
              else None
            )
            row["nominal_success"] = reference["success"]
            row["recording"] = str(recording)
          report["trials"].extend(rows)
          save(path, report)
          print(
            f"{pair} entry={index} method={method} t={duration}: success={summary(rows)[0]:.2f} score={summary(rows)[1]}",
            flush=True,
          )
      report["status"] = "complete"
      save(path, report)
    finally:
      scene.close()
      direct_scene.close()


def demos(cfg: Config) -> None:
  from concurrent.futures import ProcessPoolExecutor, as_completed
  from multiprocessing import get_context

  from mjlab.tasks.bridging.bridges.diffusion.config import tracker_experiment
  from mjlab.tasks.bridging.bridges.diffusion.execution.runtime import (
    latest_planner_checkpoint,
  )
  from mjlab.tasks.bridging.config.g1.demos.soccer import run as soccer
  from mjlab.tasks.bridging.config.g1.demos.soccer.config import CONFIG_PATH
  from mjlab.tasks.bridging.config.t1.demos.sokoban import run as sokoban
  from mjlab.tasks.bridging.tests.stage import find_checkpoint

  settings = Settings.load()
  g1_table = EntryTable.load(Path(settings.policies.selector_path))
  settings = replace(
    settings,
    policies=replace(
      settings.policies,
      locomotion_checkpoint=g1_table.checkpoints["walk"],
      jump_checkpoint=g1_table.checkpoints["jump"],
      kick_checkpoint=g1_table.checkpoints["kick"],
    ),
  )
  choices = {}
  for pair, transition in zip(
    PAIRS, ("walk_to_kick", "walk_to_jump", "jump_to_walk"), strict=True
  ):
    report = json.loads((cfg.output / f"{pair}.json").read_text(encoding="utf-8"))
    if report["status"] != "complete":
      raise ValueError("Complete the transitions before selecting soccer settings")
    handoff = settings.handoffs[transition]
    best = best_settings(report["trials"])
    choices[transition] = best
    if best is not None:
      settings = replace(
        settings,
        **{
          transition: replace(
            handoff,
            entry_index=best["entry"],
            start_distance=best["distance"],
            duration=best["duration"],
          )
        },
      )
  soccer_config = cfg.output / "soccer.yml"
  soccer_config.write_text(
    yaml.safe_dump(asdict(settings), sort_keys=False), encoding="utf-8"
  )

  table = EntryTable.for_robot("t1")
  pins = {
    skill: Path(table.checkpoints[entry])
    for skill, entry in (("walk", "walk"), ("turn", "turn_left"), ("push", "push"))
  }
  planner = latest_planner_checkpoint("t1")
  tracker = find_checkpoint(tracker_experiment("t1"))

  methods: tuple[Literal["no-op", "diffusion"], ...] = ("no-op", "diffusion")
  for demo in ("soccer", "sokoban"):
    for method in methods:
      for seed in cfg.seeds:
        path = cfg.output / f"{demo}_{method}_{seed}.json"
        if path.exists():
          raise FileExistsError(f"Refusing to replace {path}")
  with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
    jobs = []
    for method in methods:
      jobs.append(
        pool.submit(
          soccer.main,
          soccer.Config(
            config=soccer_config,
            bridge=method,
            viewer="none",
            device=cfg.device,
            result_path=cfg.output / f"soccer_{method}.json",
            result_seeds=cfg.seeds,
          ),
        )
      )
      jobs.append(
        pool.submit(
          sokoban.main,
          sokoban.Config(
            bridge=method,
            viewer="none",
            device=cfg.device,
            result_path=cfg.output / f"sokoban_{method}.json",
            result_seeds=cfg.seeds,
            level="default",
            walk_checkpoint=pins["walk"],
            turn_checkpoint=pins["turn"],
            push_checkpoint=pins["push"],
            bridge_checkpoint=planner,
            tracker_checkpoint=tracker,
          ),
        )
      )
    for job in as_completed(jobs):
      job.result()
  for demo in ("soccer", "sokoban"):
    for method in methods:
      for seed in cfg.seeds:
        path = cfg.output / f"{demo}_{method}_{seed}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if demo == "sokoban":
          result["level"] = "default"
          result["artifacts"] = {
            **{name: str(pin.resolve()) for name, pin in pins.items()},
            "planner": str(planner.resolve()),
            "tracker": str(tracker.resolve()),
          }
          result["sha256"] = {
            name: fingerprint(Path(pin)) for name, pin in result["artifacts"].items()
          }
        else:
          result["config_sha256"] = fingerprint(soccer_config)
          result["base_config_sha256"] = fingerprint(CONFIG_PATH)
          result["handoff_selection"] = choices
          result["configuration"] = asdict(settings)
        save(path, result)


def number(value: float | None) -> str:
  return "--" if value is None else f"{value:.2f}"


def table_rows(output: Path, name: str, lines: list[str]) -> None:
  formats = {
    "transitions": (
      "lcccccc",
      r"& \multicolumn{3}{c}{Success rate} & \multicolumn{3}{c}{Score} \\"
      + "\n"
      + r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}"
      + "\n"
      + r"Transition & Direct & Transition & Bridge & Direct & Transition & Bridge \\",
    ),
    "grid": (
      "llcccccc",
      r"& & \multicolumn{3}{c}{Success rate} & \multicolumn{3}{c}{Score} \\"
      + "\n"
      + r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}"
      + "\n"
      + r"Transition & Distance & 0.4~s & 0.8~s & 1.2~s & 0.4~s & 0.8~s & 1.2~s \\",
    ),
    "soccer": (
      "lccccc",
      r"& & \multicolumn{3}{c}{Failed at the switch into} & \\"
      + "\n"
      + r"\cmidrule(lr){3-5}"
      + "\n"
      + r"Switches & Completed & Jump & Walk & Kick & Failed elsewhere \\",
    ),
    "sokoban": (
      "lcccc",
      r"Switches & Solved & Stopped at a switch & Stopped at an action & Time (s) \\",
    ),
  }
  columns, header = formats[name]
  content = (
    "% Generated by mjlab.tasks.bridging.evaluate\n"
    + r"\begin{tabular}{@{}"
    + columns
    + "@{}}\n"
    + "\\toprule\n"
    + header
    + "\n\\midrule\n"
    + "\n".join(lines)
    + "\n\\bottomrule\n\\end{tabular}\n"
  )
  TABLES.mkdir(parents=True, exist_ok=True)
  (output / f"{name}.tex").write_text(content, encoding="utf-8")
  (TABLES / f"ch7_{name}.tex").write_text(content, encoding="utf-8")


def tables(cfg: Config) -> None:
  demo_rows = {}
  for demo in ("soccer", "sokoban"):
    for method in ("no-op", "diffusion"):
      rows = []
      for seed in cfg.seeds:
        path = cfg.output / f"{demo}_{method}_{seed}.json"
        if not path.is_file():
          raise FileNotFoundError(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        allowed = (
          ("completed", "jump", "walk", "kick", "elsewhere")
          if demo == "soccer"
          else ("completed", "switch", "action")
        )
        if (
          row["seed"] != seed
          or row["method"] != method
          or row["outcome"] not in allowed
        ):
          raise ValueError(f"Invalid demo result at {path}")
        rows.append(row)
      demo_rows[demo, method] = rows
    for direct, bridge in zip(
      demo_rows[demo, "no-op"], demo_rows[demo, "diffusion"], strict=True
    ):
      if direct["initial_scene"] != bridge["initial_scene"]:
        raise ValueError(f"{demo} methods started in different scenes")
      if direct.get("initial_robot_state") != bridge.get("initial_robot_state"):
        raise ValueError(f"{demo} methods started with different robot states")
  reports = [json.loads((cfg.output / f"{pair}.json").read_text()) for pair in PAIRS]
  if any(report["status"] != "complete" for report in reports):
    raise ValueError("Tables require complete transition reports")
  if any(
    report["settings"]["distances"] != [0.2, 0.4, 0.6]
    or report["settings"]["durations"] != [0.4, 0.8, 1.2]
    or report["settings"]["seeds"] != list(cfg.seeds)
    for report in reports
  ):
    raise ValueError("Thesis tables require the full grid and matching seeds")
  lines, grid = [], []
  for label, report in zip(LABELS, reports, strict=True):
    rows = report["trials"]
    direct_rows = [row for row in rows if row["method"] == "direct"]
    if any(
      row["distance"] is not None or row["budget_s"] is not None for row in direct_rows
    ):
      raise ValueError(
        "Old Direct results used the bridge distance grid and must be rerun"
      )
    direct = summary(direct_rows)
    bridge = summary([row for row in rows if row["method"] == "diffusion"])
    lines.append(
      f"    {label} & {number(direct[0])} & -- & {number(bridge[0])} & {number(direct[1])} & -- & {number(bridge[1])} \\\\"
    )
    for i, distance in enumerate(report["settings"]["distances"]):
      cells = [
        best_entry(rows, distance, duration)
        for duration in report["settings"]["durations"]
      ]
      values = [number(cell[j]) for j in (0, 1) for cell in cells]
      grid.append(
        f"    {label if i == 0 else ''} & {distance:.1f}~m & "
        + " & ".join(values)
        + r" \\"
      )
    grid.append(r"    \midrule")
  for name, content in (("transitions", lines), ("grid", grid[:-1])):
    table_rows(cfg.output, name, content)
  for demo, categories in (
    ("soccer", ("completed", "jump", "walk", "kick", "elsewhere")),
    ("sokoban", ("completed", "switch", "action")),
  ):
    lines = []
    for method, label in (("no-op", "Direct"), ("diffusion", "Bridge")):
      rows = demo_rows[demo, method]
      values = [
        number(sum(row["outcome"] == category for row in rows) / len(rows))
        for category in categories
      ]
      if demo == "sokoban":
        times = [row["seconds"] for row in rows if row["outcome"] == "completed"]
        values.append(number(float(np.mean(times)) if times else None))
      lines.append(f"    {label} & " + " & ".join(values) + r" \\")
    table_rows(cfg.output, demo, lines)


def main(cfg: Config) -> None:
  if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds) or min(cfg.seeds) < 0:
    raise ValueError("Seeds must be nonempty, nonnegative and unique")
  for values in (cfg.distances, cfg.durations):
    if (
      not values
      or len(set(values)) != len(values)
      or any(not math.isfinite(v) or v <= 0 for v in values)
    ):
      raise ValueError("Grid values must be unique, finite and positive")
  if cfg.suite in ("all", "transitions"):
    transitions(cfg)
  if cfg.suite in ("all", "demos"):
    demos(cfg)
  if cfg.suite in ("all", "tables"):
    tables(cfg)


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
