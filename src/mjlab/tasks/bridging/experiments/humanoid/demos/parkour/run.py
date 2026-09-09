"""Run the parkour demo, or any part of it short of running it.

Four things this can do, cheapest first, and the first two need no checkpoints at all:

    --scene True   draw the course and stop. No robot, no policies, no simulation
    --dry True     print the rules, the course and the plan, and stop
    (default)      run it, in viser or in MuJoCo's own window
    --viewer none  run it headless, for a number rather than a picture

Everything else lives beside this file:

    config.yml     every number the course is drawn from
    course.py      what is on the course and where
    arena.py       the environment, and the bare model the scene viewer serves
    pool.py        the skills, each wrapping one frozen policy
    bridge.py      the bridge, aimed at a pose
    controller.py  the rules, the alignment, and the phase machine

Run

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.run
    uv run python -m ...demos.parkour.run --scene True
    uv run python -m ...demos.parkour.run --dry True --seed 7 --count 8
    uv run python -m ...demos.parkour.run --viewer native
    uv run python -m ...demos.parkour.run --viewer none
    uv run python -m ...demos.parkour.run --config my_course.yml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.arena import (
  Focus,
  course_env_cfg,
  obstacle_names,
  show,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.bridge import Bridge
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.controller import (
  CRUISE_SKILL,
  DECISION_HEADER,
  ROSTER,
  Controller,
  plan,
  plan_lines,
  unsolved,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course import Settings
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course import (
  generate as generate_course,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.pool import SkillPool


@dataclass(frozen=True)
class Config:
  config: Path | None = None
  """Where the course parameters come from. None for the package's own config.yml."""
  seed: int | None = None
  """Which course to draw. None to take the seed from the config."""
  count: int | None = None
  """How many obstacles. None to take the count from the config."""

  scene: bool = False
  """Draw the course and stop. No robot, no policies, no simulation."""
  dry: bool = False
  """Print the plan and stop. No simulation and no checkpoints."""

  viewer: Literal["viser", "native", "none"] = "viser"
  """viser serves in a browser, native opens MuJoCo's own window, none runs headless."""
  port: int = 8080
  """Viser only."""
  patience: int = 8000
  """Control steps before a headless run gives up."""

  device: str | None = None
  torch_seed: int = 0
  scored: str | None = None
  """A skill whose reward terms the arena carries, so a hand-over into it can be scored."""


def needed(steps) -> None:
  """Check the selector has profiled every skill this course will hand over to.

  Before the arena, because the alternative is what it replaced: three checkpoints loaded, a
  robot walking, and the run dying at the first switch on a table lookup. A skill can be
  driven without entry rows, so the pool only warns; the controller cannot aim a bridge at
  one, so this refuses.

  A skill is absent for one of two reasons and the message covers both: it was never
  recorded, or it has no window in selector/__init__.py, which is how a skill is declared
  to have nowhere a bridge may aim.

  Recording is all or nothing, which is why the message says so. `selector.record` writes one
  file holding exactly the skills it was given, so recording the missing one on its own
  replaces the rollouts of every other skill and takes the entry table down with it.
  """
  from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.controller import (
    CRUISE_SKILL,
  )
  from mjlab.tasks.bridging.experiments.humanoid.selector import EntryTable

  table = EntryTable.load()
  wanted = {CRUISE_SKILL.name} | {s.rule.skill for s in steps if s.rule is not None}
  absent = sorted(wanted - set(table.skills))
  if not absent:
    return
  raise SystemExit(
    f"\nRefusing to start: no entry points for {', '.join(absent)}, so the bridge has "
    f"nowhere to aim. The table holds {', '.join(table.skills)}.\n\n"
    f"Profiling one skill rewrites the rollouts of all of them, so record them together:\n"
    f"  1. add {', '.join(absent)} to SKILLS in skills/__init__.py, and give each one a "
    f"window in selector/__init__.py\n"
    f"  2. uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record\n"
    f"  3. uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build"
  )


def main(cfg: Config) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  course = generate_course(Settings.load(cfg.config), seed=cfg.seed, count=cfg.count)
  steps = plan(course)
  for line in plan_lines(course, steps):
    print(line)

  # The course alone, before anything that needs a checkpoint. After the plan is printed
  # rather than instead of it, because the two answer different questions
  if cfg.scene:
    show(course, "native" if cfg.viewer == "native" else "viser", cfg.port)
    return

  if unsolved(steps):
    raise SystemExit(
      "\nRefusing to start: the table does not cover every obstacle on this course."
    )
  if cfg.dry:
    return

  needed(steps)

  torch.manual_seed(cfg.torch_seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Before the simulation, because a missing checkpoint is the one thing this demo cannot
  # work around and finding out after the arena is up costs a minute per attempt
  for name, path in SkillPool.resolve(ROSTER).items():
    print(f"{name:<8} {path}")

  focus = Focus(names=obstacle_names(course))
  env = ManagerBasedRlEnv(
    cfg=course_env_cfg(ROSTER, course, focus, scored=cfg.scored), device=device
  )
  pool = SkillPool.load(ROSTER, env, device)
  for line in pool.lines():
    print(line)

  env.reset()
  controller = Controller(env, pool, Bridge(env, pool), course, steps, focus)
  print("")
  print("\n".join(DECISION_HEADER))

  if cfg.viewer == "none":
    obs = env.get_observations()
    for _ in range(cfg.patience):
      if controller.done:
        break
      obs, _, _, _, _ = env.step(controller(obs))
    else:
      print(f"\ngave up after {cfg.patience} steps, '{controller.driving}' driving")
    for line in controller.report():
      print(line)
    env.close()
    return

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  # Clipped the way the cruise skill was trained. Every policy here shares one action space,
  # so one wrapper serves all of them
  wrapped = RslRlVecEnvWrapper(
    env, clip_actions=load_rl_cfg(CRUISE_SKILL.task).clip_actions
  )
  if cfg.viewer == "native":
    from mjlab.viewer import NativeMujocoViewer

    NativeMujocoViewer(wrapped, controller).run()
  else:
    import viser

    from mjlab.viewer import ViserPlayViewer

    server = viser.ViserServer(label=f"parkour seed {course.seed}", port=cfg.port)
    ViserPlayViewer(
      wrapped,
      controller,
      viser_server=server,
      info_provider=lambda _: controller.driving,
    ).run()
  for line in controller.report():
    print(line)
  wrapped.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
