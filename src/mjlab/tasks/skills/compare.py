"""Watch two architectures run the same episode side by side, and record it.

One script for every experiment. It stands the experiment up once, gives each
architecture its own world inside that one environment, and renders every world under
identical camera settings into one wide frame: architecture 0 on the left, architecture
1 on the right, same corridor, same clock, same view. The frames are streamed straight
to an mp4, for as long as --seconds asks for.

Compare the baseline against a trained bridge:

    uv run python -m mjlab.tasks.skills.compare --experiment parkour --archs 0 1

The video runs for --seconds of simulated time, and is paced so that a second of it is
a second of simulation:

    uv run python -m mjlab.tasks.skills.compare --archs 0 1 --seconds 120

Any number of architectures, in any order, and the panels follow the flag:

    uv run python -m mjlab.tasks.skills.compare --experiment diffdrive --archs 0 3 1

Point an architecture at a specific run directory instead of its latest one:

    uv run python -m mjlab.tasks.skills.compare --archs 0 1 \\
        --checkpoints 1 logs/skills/parkour/arch_1/<run>

Why one environment with several worlds rather than one environment per architecture:
the worlds share a model, a terrain and a step, so nothing about the arena, and nothing
about the clock, can differ between panels. What each architecture does not share is
its skill pool: a pool carries per-env memory and is ticked once per step, so two
architectures reading the same pool would advance it twice a step and corrupt each
other's skills. Each panel therefore loads its own pool, which is also why this costs
one set of skill checkpoints per architecture. Each panel also gets its own camera,
for a reason worth reading before changing it (see `_build_renderers`).

Each world runs its own episode. Auto-reset is off, so a world that fails is left where
it fell for a moment (see --linger-seconds) rather than being reset away at the instant
it is detected, and then that world alone restarts. Nothing waits for it: a panel that
is still going is not cut short because another one crashed, which is the whole picture
on the diffdrive, where the baseline tips over and over while a working bridge drives on.

The opening episode is the one they share. It re-seeds the same generator before each
world's first reset, so all panels start from the same state and an early difference on
screen is the architecture rather than the draw.

Only an early one. The starts match, the simulator does not keep them matching:
mujoco-warp is not bit-deterministic across worlds, and on a robot with contacts and
29 joints that is enough to pull two worlds apart. Measured here by running the
baseline against itself: on the cart-pole `--archs 0 0` stays identical for as long as
you care to watch, while on parkour the two panels are visibly apart within about a
second. So read a hand-over, where the architectures are doing different things to the
same body, and not the long tail after it, where they would have drifted apart anyway.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import mediapy as media
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, VecEnvObs
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.skills.architectures import build
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.experiment import Experiment
from mjlab.tasks.skills.meta import ComposedPolicy
from mjlab.tasks.skills.utils import retrieve_latest_architecture_checkpoint
from mjlab.utils.random import seed_rng
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

# The strip above the panels, and the gap between them. Dark enough that a light
# scene stops at the panel edge and the labels stay readable over it
_CHROME_COLOR = (24, 24, 28)
_LABEL_COLOR = (235, 235, 240)
_HEADER_HEIGHT = 34
_GUTTER = 6


@dataclass(frozen=True)
class Camera:
  """Where the one camera every panel is rendered from sits.

  `track` follows the experiment's robot, which is what keeps a panel useful once the
  robot has left the origin; with it off the camera stays put and looks at `lookat`.
  """

  track: bool = True
  lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
  distance: float = 5.0
  azimuth: float = 90.0
  elevation: float = -15.0


@dataclass(frozen=True)
class Scenario:
  """How one experiment is stood up, as this script needs it.

  The same four things every demo builds (an arena, a pool, a controller, and the
  action clipping the skills were trained under), named once so the loop below never
  has to know which experiment it is running.
  """

  entity_name: str
  """The scene entity that is the robot: what the camera follows."""

  clip_task_id: str
  """Whose rl cfg supplies the action clipping, as the experiment's demo picks it."""

  camera: Camera
  env_cfg: Callable[[], ManagerBasedRlEnvCfg]
  experiment: Callable[[ManagerBasedRlEnv, str], Experiment]
  controller: Callable[[Experiment], Controller]


def _cartpole() -> Scenario:
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.skills.experiments.cartpole import (
    ENTITY_NAME,
    SPINUP_TASK_ID,
    build_experiment,
  )
  from mjlab.tasks.skills.experiments.cartpole.controller import CartpoleController

  return Scenario(
    entity_name=ENTITY_NAME,
    clip_task_id=SPINUP_TASK_ID,
    # The cart-pole is planar, so a straight side view sees everything there is
    camera=Camera(distance=3.5, azimuth=90.0, elevation=-5.0),
    env_cfg=lambda: load_env_cfg(SPINUP_TASK_ID, play=True),
    experiment=build_experiment,
    controller=lambda exp: CartpoleController(exp.pool),
  )


def _diffdrive() -> Scenario:
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.skills.experiments.diffdrive import (
    DRIVE_TASK_ID,
    ENTITY_NAME,
    build_experiment,
  )
  from mjlab.tasks.skills.experiments.diffdrive.controller import DiffdriveController

  return Scenario(
    entity_name=ENTITY_NAME,
    clip_task_id=DRIVE_TASK_ID,
    # Three quarters and low: the failure here is a chassis rolling onto its side,
    # which a top-down view flattens away
    camera=Camera(distance=2.0, azimuth=135.0, elevation=-20.0),
    env_cfg=lambda: load_env_cfg(DRIVE_TASK_ID, play=True),
    experiment=build_experiment,
    # The demo's phase lengths, not the controller's defaults: they are what make the
    # hand-off happen often enough to watch
    controller=lambda exp: DiffdriveController(
      exp.pool, straight_steps=300, turn_steps=150
    ),
  )


def _parkour() -> Scenario:
  from mjlab.tasks.skills.experiments.parkour import (
    ENTITY_NAME,
    JUMP_TASK_ID,
    build_experiment,
  )
  from mjlab.tasks.skills.experiments.parkour.arena import parkour_arena_env_cfg
  from mjlab.tasks.skills.experiments.parkour.controller import ParkourController

  return Scenario(
    entity_name=ENTITY_NAME,
    clip_task_id=JUMP_TASK_ID,
    # Side on: a jump is judged on height and on where it lands, and both are read
    # off a profile
    camera=Camera(distance=5.0, azimuth=90.0, elevation=-10.0),
    env_cfg=parkour_arena_env_cfg,
    experiment=build_experiment,
    controller=lambda exp: ParkourController(exp.pool, entity_name=exp.entity_name),
  )


# The experiments this script can run. Adding one is a builder above and a line here.
# Soccer and table tennis are missing on purpose: neither declares an `Experiment` yet
# (no view, no window plan, no checkpoint folder), so there is nothing for `build` to
# put an architecture together from
SCENARIOS: dict[str, Callable[[], Scenario]] = {
  "cartpole": _cartpole,
  "diffdrive": _diffdrive,
  "parkour": _parkour,
}


@dataclass(frozen=True)
class CompareConfig:
  # Which experiment to run. See SCENARIOS
  experiment: Literal["cartpole", "diffdrive", "parkour"] = "parkour"

  # The architectures to put side by side, left to right. One world and one panel
  # each, so `--archs 0 1` is the baseline against the first bridge
  archs: tuple[int, ...] = (0, 1)

  # Run directories for the trained architectures, as `--checkpoints 1 <path>`.
  # Anything omitted falls back to that architecture's latest run. Architecture 0
  # trains nothing and needs none
  checkpoints: dict[int, str] | None = None

  # Where to write the video. None puts it under logs/skills/<experiment>/compare
  output: str | None = None

  # Width of one panel [px]. The video is this wide per architecture, plus the gaps
  width: int = 640

  # Height of one panel [px]. The video is this tall plus the label strip
  height: int = 480

  # Video frame rate. The simulation is rendered every few steps so that a second of
  # video is a second of simulated time; the closest achievable rate is used
  fps: float = 30.0

  # Camera overrides. None keeps the experiment's own choice (see the scenarios above)
  distance: float | None = None
  azimuth: float | None = None
  elevation: float | None = None

  # Draw what the managers and sensors visualize (command arrows, goal markers, and
  # so on), as an interactive viewer does. Off keeps the panels to the scene itself
  debug_vis: bool = False

  # How long a finished episode is left on screen before every panel restarts [s], so
  # a fall is visible instead of being reset away the instant it is detected
  linger_seconds: float = 1.5

  # Open every panel's episode from the same state, by re-seeding before each world's
  # first reset. Turn it off to let each panel draw its own start
  identical_starts: bool = True

  # Seed the first episode; each later one is seeded from this plus its index
  seed: int = 0

  # How long the video lasts [s], in simulated time. The recording is paced so that
  # this is also its wall-clock length when played back
  seconds: float = 30.0

  device: str | None = None


def _load_font(size: int) -> Any | None:
  """A font for the panel labels, or None if Pillow is not around to draw them.

  Pillow arrives with mediapy rather than on its own, so the labels are a nicety this
  refuses to fail over: without it the header strip is simply blank.
  """
  try:
    from PIL import ImageFont
  except ImportError:
    print("[WARN] Pillow is not installed; the panels will not be labelled")
    return None
  try:
    return ImageFont.load_default(size=size)
  except TypeError:
    # Pillow older than 10.1 has no sized default font
    return ImageFont.load_default()


def _canvas_shape(panels: int, width: int, height: int) -> tuple[int, int]:
  """The composed frame's (height, width), rounded up to even for h264."""
  total_width = panels * width + (panels - 1) * _GUTTER
  total_height = height + _HEADER_HEIGHT
  return total_height + total_height % 2, total_width + total_width % 2


def _compose(
  frames: list[np.ndarray],
  labels: list[str],
  font: Any | None,
  shape: tuple[int, int],
) -> np.ndarray:
  """One video frame: the labelled header strip over the panels, side by side."""
  height, width = frames[0].shape[:2]
  gutter = np.full((height, _GUTTER, 3), _CHROME_COLOR, dtype=np.uint8)
  row: list[np.ndarray] = []
  for frame in frames:
    if row:
      row.append(gutter)
    row.append(frame)
  panels = np.concatenate(row, axis=1)

  header = np.full((_HEADER_HEIGHT, panels.shape[1], 3), _CHROME_COLOR, dtype=np.uint8)
  if font is not None:
    from PIL import Image, ImageDraw

    image = Image.fromarray(header)
    draw = ImageDraw.Draw(image)
    for i, label in enumerate(labels):
      draw.text(
        (i * (width + _GUTTER) + 10, _HEADER_HEIGHT // 2),
        label,
        fill=_LABEL_COLOR,
        font=font,
        anchor="lm",
      )
    header = np.asarray(image)

  canvas = np.concatenate([header, panels], axis=0)
  pad_h = shape[0] - canvas.shape[0]
  pad_w = shape[1] - canvas.shape[1]
  if pad_h or pad_w:
    canvas = np.pad(canvas, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
  return canvas


def _render_panels(
  env: ManagerBasedRlEnv,
  renderers: list[OffscreenRenderer],
  debug_vis: bool,
) -> list[np.ndarray]:
  """One image per world, each from its own camera."""
  callback = env.update_visualizers if debug_vis else None
  frames = []
  for renderer in renderers:
    renderer.update(env.sim.data, debug_vis_callback=callback)
    # Copied because the next render is free to hand back the same buffer
    frames.append(np.array(renderer.render(), dtype=np.uint8))
  return frames


def _open_episode(
  env: ManagerBasedRlEnv, panels: int, seed: int, identical: bool
) -> VecEnvObs:
  """Start every world, optionally from the same state.

  The one reset the panels share. A reset draws per-env: joint offsets, a spawn, a
  command. Seeding once and resetting all the worlds together would hand each of them a
  different draw, so instead each world is reset on its own from the same generator
  state, and they come out matching.

  Everything after this is per-world and unaligned by design, so the restarts the loop
  issues are left to draw for themselves. This fixes what the opening reset draws and
  nothing beyond it; see the module docstring for how long two worlds stay together.
  """
  if not identical:
    obs, _ = env.reset()
    return obs

  obs: VecEnvObs = {}
  for i in range(panels):
    seed_rng(seed)
    obs, _ = env.reset(env_ids=torch.tensor([i], dtype=torch.int64, device=env.device))
  return obs


def _output_path(cfg: CompareConfig) -> Path:
  if cfg.output is not None:
    return Path(cfg.output)
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = "-vs-".join(f"arch{a}" for a in cfg.archs)
  return Path("logs") / "skills" / cfg.experiment / "compare" / f"{name}_{stamp}.mp4"


def _build_renderers(
  env: ManagerBasedRlEnv, scenario: Scenario, cfg: CompareConfig, panels: int
) -> list[OffscreenRenderer]:
  """One renderer per world, all set up from the same camera settings.

  Deliberately not one renderer moved from world to world. A tracking camera does not
  jump to the body it follows, it eases towards it, so a single camera alternating
  between the panels would keep half of each other panel's lookat: with one robot down
  and the other still running, both views would drift towards the midpoint between
  them. A camera per panel follows only its own robot, and since they are configured
  identically the panels stay comparable.
  """
  camera = scenario.camera
  renderers = []
  for i in range(panels):
    viewer = ViewerConfig(
      lookat=camera.lookat,
      distance=cfg.distance if cfg.distance is not None else camera.distance,
      azimuth=cfg.azimuth if cfg.azimuth is not None else camera.azimuth,
      elevation=cfg.elevation if cfg.elevation is not None else camera.elevation,
      env_idx=i,
      # A panel shows one architecture and nothing else: the neighbouring worlds the
      # renderer would otherwise draw for context are the other architectures
      max_extra_envs=0,
      width=cfg.width,
      height=cfg.height,
    )
    if camera.track:
      viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
      viewer.entity_name = scenario.entity_name
    else:
      viewer.origin_type = ViewerConfig.OriginType.WORLD
    renderer = OffscreenRenderer(
      model=env.sim.mj_model,
      cfg=viewer,
      scene=env.scene,
      sim_model=env.sim.model,
      expanded_fields=env.sim.expanded_fields,
    )
    renderer.initialize()
    renderers.append(renderer)
  return renderers


def run_comparison(cfg: CompareConfig) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  if not cfg.archs:
    raise ValueError("--archs needs at least one architecture, e.g. --archs 0 1.")
  if cfg.experiment not in SCENARIOS:
    raise ValueError(
      f"Unknown experiment '{cfg.experiment}'; registered: {sorted(SCENARIOS)}."
    )

  scenario = SCENARIOS[cfg.experiment]()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  panels = len(cfg.archs)

  env_cfg = scenario.env_cfg()
  env_cfg.scene.num_envs = panels
  # The panels restart together, which is this script's decision to make rather than
  # the env's; see the linger countdown in the loop below
  env_cfg.auto_reset = False
  # No render_mode: the env's own renderer draws one world, and this needs one per
  # architecture (see _build_renderers)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  renderers = _build_renderers(env, scenario, cfg, panels)

  # One composition per architecture, each with its own pool: a pool is ticked once
  # per step and remembers what it was doing, so sharing one would have two
  # architectures advancing each other's skills
  checkpoints = cfg.checkpoints or {}
  policies: list[ComposedPolicy] = []
  for architecture in cfg.archs:
    exp = scenario.experiment(env, device)
    meta = build(env, architecture, exp)
    if architecture != 0:
      run_dir = checkpoints.get(
        architecture
      ) or retrieve_latest_architecture_checkpoint(exp.name, architecture)
      print(f"[INFO] arch {architecture} checkpoint: {run_dir}")
      meta.load(Path(run_dir))
    policies.append(ComposedPolicy(env, scenario.controller(exp), meta))

  clip_actions = load_rl_cfg(scenario.clip_task_id).clip_actions
  control_hz = 1.0 / env.step_dt
  render_every = max(1, round(control_hz / cfg.fps))
  fps = control_hz / render_every
  linger_steps = max(0, round(cfg.linger_seconds / env.step_dt))
  total_steps = max(1, round(cfg.seconds / env.step_dt))
  shape = _canvas_shape(panels, cfg.width, cfg.height)
  font = _load_font(18)

  path = _output_path(cfg)
  path.parent.mkdir(parents=True, exist_ok=True)
  print(
    f"[INFO] recording {panels} panel(s) at {fps:.1f} fps to {path}\n"
    f"[INFO] {cfg.seconds:.0f}s of simulated time, {total_steps} steps"
  )

  obs = _open_episode(env, panels, cfg.seed, cfg.identical_starts)
  # Latched per world so a panel stays marked as failed while it lingers, rather than
  # only on the single step the termination fired
  failed = torch.zeros(panels, dtype=torch.bool, device=env.device)
  # How long each world has left on screen before it restarts; -1 is "still running"
  countdown = torch.full((panels,), -1, dtype=torch.long, device=env.device)
  linger = torch.full_like(countdown, linger_steps)
  running = torch.full_like(countdown, -1)
  frames = 0

  try:
    # crf rather than mediapy's default, which drops to qp 28 above 640x480: two
    # panels are always above it, and what it blurs is the poses being compared
    with media.VideoWriter(path, shape=shape, fps=fps, crf=20) as writer:
      for step in range(total_steps):
        # Each composition acts on every world; only the one it owns is kept
        actions = torch.stack([policy(obs)[i] for i, policy in enumerate(policies)])
        if clip_actions is not None:
          actions = actions.clamp(-clip_actions, clip_actions)
        obs, _, terminated, time_out, _ = env.step(actions)

        # With auto-reset off the env demands a reset before the next step; the
        # decision is this loop's, so the request is cleared and answered below
        env._manual_reset_pending.zero_()  # noqa: SLF001
        failed |= terminated

        if step % render_every == 0:
          labels = [
            f"arch {architecture}: {policy.active_label(i)}"
            + (" [terminated]" if bool(failed[i]) else "")
            for i, (architecture, policy) in enumerate(
              zip(cfg.archs, policies, strict=True)
            )
          ]
          panel_frames = _render_panels(env, renderers, cfg.debug_vis)
          writer.add_image(_compose(panel_frames, labels, font, shape))
          frames += 1
          if frames % 300 == 0:
            print(f"[INFO] {frames / fps:.0f}s of {cfg.seconds:.0f}s")

        # A world that finished is held for a moment and then restarts on its own.
        # Nothing waits for it: a panel that is still going is not cut short because
        # another one crashed, which is the whole picture on the diffdrive, where the
        # baseline tips over and over while a working bridge drives on
        starting = (terminated | time_out) & (countdown < 0)
        countdown = torch.where(starting, linger, countdown)
        countdown = torch.where((countdown > 0) & ~starting, countdown - 1, countdown)
        expired = countdown == 0
        if expired.any():
          obs, _ = env.reset(env_ids=expired.nonzero(as_tuple=False).squeeze(-1))
          failed &= ~expired
          countdown = torch.where(expired, running, countdown)
  finally:
    for renderer in renderers:
      renderer.close()
    env.close()

  print(f"[INFO] wrote {frames} frames ({frames / fps:.1f}s) to {path}")


if __name__ == "__main__":
  run_comparison(tyro.cli(CompareConfig))
