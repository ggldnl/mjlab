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
    approach.py    where the robot has to stand before a traversal will work
    controller.py  the plan, and the loop that runs it one action at a time

Run

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.run
    uv run python -m ...demos.parkour.run --scene True
    uv run python -m ...demos.parkour.run --dry True --seed 7 --count 8
    uv run python -m ...demos.parkour.run --viewer native
    uv run python -m ...demos.parkour.run --viewer none
    uv run python -m ...demos.parkour.run --config my_course.yml
    uv run python -m ...demos.parkour.run --bridge imitation
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges import (
  DEFAULT_BRIDGE,
  BridgeKind,
  resolve,
)
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
  ENTRIES,
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
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.pool import (
  SkillPool,
  with_bridge,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.entry_tolerances import (
  ToleranceOverrides,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import BRIDGE_GROUP


@dataclass(frozen=True)
class Config:
  config: Path | None = None
  """Where the course parameters come from. None for the package's own config.yml."""
  seed: int | None = None
  """Which course to draw. None to take the seed from the config."""
  count: int | None = None
  """How many obstacles. None to take the count from the config."""

  bridge: BridgeKind = DEFAULT_BRIDGE
  """Which bridge architecture switches the skills. See bridges/__init__.py.

  The course is built on the chosen one's play config, so this picks the observation the
  bridge policy reads along with the checkpoint it loads. The skills are untouched by it."""

  ##
  # The hand-over. Every one of these overrides config.yml, and each is the same quantity
  # the transition scripts carry under the same name: see tests/stage.py Config.
  ##

  hold_back: float | None = None
  """Metres short of the pose a skill needs that the walk stops, leaving the rest to the
  bridge. tests/stage.py calls this fire_at. None keeps config.yml."""
  window: float | None = None
  """Seconds the bridge gets to cross. Has to sit inside the range it trained on, which is
  BridgeCommandCfg.duration_s_range. None keeps config.yml."""
  blend_steps: int | None = None
  """Control steps to ramp out of the parting policy's last action at every switch. Zero is
  a hard switch. None keeps config.yml."""
  count_in: bool | None = None
  """Whether the entering tracker's clip marches up to its entry frame during the crossing
  instead of being rewound under it at the hand-over. None keeps config.yml."""
  entries: dict[str, int] = field(default_factory=dict)
  """Which clip frame to enter a skill at, by skill name, overriding controller.ENTRIES.
  The nearest recorded entry wins:

      --entries "{'jump': 78, 'climb': 49}"
  """
  tolerances: ToleranceOverrides = field(default_factory=ToleranceOverrides)
  """Override individual channels of the arrival tolerance the bridge is held to, which
  otherwise comes from tests/entry_tolerances.py by skill and frame."""

  ##
  # Which policies. Named after the skill rather than after its role, as the transition
  # scripts are, because the newest run under a log directory has been the wrong one before.
  ##

  bridge_checkpoint: Path | None = None
  walk_checkpoint: Path | None = None
  jump_checkpoint: Path | None = None
  climb_checkpoint: Path | None = None

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


def tuned(settings: Settings, cfg: Config) -> Settings:
  """config.yml with whatever the command line overrode.

  The flags and the file hold the same quantities, so the file stays the place the numbers
  are written down and explained, and a flag is how one of them is moved for a single run.
  """
  changed = {
    name: value
    for name, value in (
      ("hold_back", cfg.hold_back),
      ("window_s", cfg.window),
      ("blend_steps", cfg.blend_steps),
      ("count_in", cfg.count_in),
    )
    if value is not None
  }
  if not changed:
    return settings
  for name, value in changed.items():
    print(f"approach.{name} = {value} (overridden)")
  return dataclasses.replace(
    settings, approach=dataclasses.replace(settings.approach, **changed)
  )


def aimed_at(entries: dict[str, int]) -> None:
  """Move which clip frame a skill is entered at, for this run.

  Refused rather than ignored for a skill the demo does not drive, because a typo here
  looks exactly like the flag doing nothing.
  """
  for skill, frame in entries.items():
    if skill not in ENTRIES:
      raise SystemExit(
        f"'{skill}' is not one of the skills this demo hands over to. It drives "
        f"{', '.join(sorted(ENTRIES))}."
      )
    print(f"{skill} entered at frame {frame} rather than {ENTRIES[skill]} (overridden)")
    ENTRIES[skill] = frame


def chosen(cfg: Config) -> dict[str, Path]:
  """Checkpoints named on the command line, by skill. Empty means search for each."""
  named = (
    (CRUISE_SKILL.name, cfg.walk_checkpoint),
    ("jump", cfg.jump_checkpoint),
    ("climb", cfg.climb_checkpoint),
    (BRIDGE_GROUP, cfg.bridge_checkpoint),
  )
  return {name: path for name, path in named if path is not None}


def main(cfg: Config) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  settings = tuned(Settings.load(cfg.config), cfg)
  aimed_at(cfg.entries)
  course = generate_course(settings, seed=cfg.seed, count=cfg.count)
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

  # The architecture first, because it is the cheapest thing to be wrong about and a stub
  # says so plainly, then the entry table the controller needs to aim at
  spec = resolve(cfg.bridge)
  needed(steps)

  torch.manual_seed(cfg.torch_seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Before the simulation, because a missing checkpoint is the one thing this demo cannot
  # work around and finding out after the arena is up costs a minute per attempt. The bridge
  # is in this list rather than found later, since an untrained architecture is exactly what
  # --bridge makes easy to ask for
  explicit = chosen(cfg)
  for name, path in SkillPool.resolve(with_bridge(ROSTER, spec), explicit).items():
    print(f"{name:<8} {path}")

  focus = Focus(names=obstacle_names(course))
  env = ManagerBasedRlEnv(
    cfg=course_env_cfg(ROSTER, course, focus, spec, scored=cfg.scored), device=device
  )
  pool = SkillPool.load(ROSTER, env, device, spec, checkpoints=explicit)
  for line in pool.lines():
    print(line)

  env.reset()
  controller = Controller(
    env, pool, Bridge(env, pool), course, steps, focus, tolerances=cfg.tolerances
  )
  # The compiled plan, with the coordinates the approach solve produced. plan_lines above
  # printed which skill each obstacle asks for; this is where the robot will actually go
  for line in controller.lines():
    print(line)
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
      record_name=f"parkour-seed-{course.seed}",
    ).run()
  for line in controller.report():
    print(line)
  wrapped.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
