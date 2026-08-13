"""Watch two ways of doing the same thing side by side, and record it.

One script for every experiment. Each architecture gets its own world inside one
environment and its own panel, every world is rendered under identical camera settings
into one wide frame, and the frames are streamed to an mp4 that runs for --seconds.
There are two things to watch, and which one you get depends on the experiment.

The cart-pole and the diffdrive run the composition live. A controller names a skill, an
architecture carries out the switch, and the panels are the same episode driven several
ways:

    uv run python -m mjlab.tasks.skills.compare --experiment diffdrive --archs 0 3
    uv run python -m mjlab.tasks.skills.compare --experiment cartpole --archs 0 1 \\
        --checkpoints 1 logs/skills/cartpole/arch_1/<run>

Parkour plays one hand-over instead. Its corridor composition does not work yet, so what
is recorded is the question `tests/walk2jump.py` asks -- a walk cut mid-stride and handed
to a jump -- crossed both ways at once:

    uv run python -m mjlab.tasks.skills.compare --archs 0 4 --seconds 60
    uv run python -m mjlab.tasks.skills.compare --archs 0 4 --handover.gap 40

Arm 4 is that script's, imported rather than rebuilt, rollouts and bridge and all: the
bridge crosses the hole and the jump is entered at the frame it was aimed at. Arm 0 is
the naive hand-off, and it is not that script's baseline. At the cut the jump skill
simply starts, which pins its clip to wherever the robot is and winds it to its own
first frame, a stand. Nothing about the body changes at that instant -- there is no
teleport and no pause, it keeps the pose and the momentum the walk left it with -- so
what the panel shows is a robot at a metre a second being told to reproduce a stand. It
does not survive it, which is the whole point of building a bridge.

The bridged panel carries a faded ghost with it: the same jump policy from its own
reset, in nominal conditions, wound so that the frame it hands over on falls where its
robot arrives. The two are supposed to meet there, and how far apart they are when they
do is the arrival error, seen rather than tabulated.

The baseline panel has no ghost and nothing to wait through. Its clip is pinned to the
robot rather than to a place the robot was supposed to reach, so there is no nominal
rollout to hold it against, and control passes at the cut itself. It walks, it is handed
the jump, it goes down. It is left lying there for as long as the bridged arm is still
going, so the panels start and end together.

Nothing else is drawn, and nothing is coloured by who is driving: the phase is in the
label instead.

Why one environment with several worlds rather than one environment per architecture:
the worlds share a model, a terrain and a step, so nothing about the arena, and nothing
about the clock, can differ between panels. What the live mode's architectures do not
share is a skill pool: a pool carries per-env memory and is ticked once per step, so two
architectures reading the same one would advance it twice a step and corrupt each other's
skills. Each panel therefore loads its own, which is why this costs one set of skill
checkpoints per architecture. Each panel also gets its own camera, for a reason worth
reading before changing it (see `_build_renderers`).

In the live mode each world runs its own episode. Auto-reset is off, so a world that
fails is left where it fell for a moment (see --linger-seconds) rather than being reset
away at the instant it is detected, and then that world alone restarts. Nothing waits for
it: a panel that is still going is not cut short because another one crashed, which is
the whole picture on the diffdrive, where the baseline tips over and over while a working
bridge drives on. The opening episode is the one they share, re-seeded per world so all
panels start from the same state and an early difference is the architecture rather than
the draw.

Only an early one. The starts match, the simulator does not keep them matching:
mujoco-warp is not bit-deterministic across worlds, and on a robot with contacts and 29
joints that is enough to pull two worlds apart. Measured by running the baseline against
itself: on the cart-pole `--archs 0 0` stays identical for as long as you care to watch.
The parkour mode is not exposed to any of this, since both its arms were rolled before
the recording began and playback only writes recorded frames onto the robot.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import mediapy as media
import mujoco
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


# Parkour is not a scenario: it plays one walk-to-jump hand-over rather than a
# composition running down a corridor (see _run_handover), and its two arms are the two
# ways across that hand-over rather than two architectures out of architectures/.
_HANDOVER_ARMS = {0: "no bridge", 4: "bridge"}

# Side on, and close: a jump is judged on height and on where it lands, and both are
# read off a profile. Far enough back to hold the robot and a ghost standing a
# look-ahead in front of it
_HANDOVER_CAMERA = Camera(distance=5.0, azimuth=90.0, elevation=-8.0)


def _phase(frame: int, past: int, gap: int) -> str:
  """Which stretch of the hand-over a track frame falls in.

  The stage the walk2jump viewer says with colour. Said in words here instead, since
  colouring the robot is the thing this recording deliberately does not do.

  `gap` is the arm's, not the window's: without a bridge there is no hole to be in, so
  it is zero there and the label goes straight from walk to jump.
  """
  if frame < past:
    return "walk"
  return "hole" if frame < past + gap else "jump"


def _hold_to(track: torch.Tensor, span: int) -> torch.Tensor:
  """Pad a track to `span` frames by holding its last one.

  The arms do not take the same time. Without a bridge the jump starts at the cut rather
  than a hole later, so that panel reaches the end of its rollout first, and stopping it
  there would leave the other panel playing to a dead half of the frame. What is held is
  whatever the arm ended on, and for a hand-over that failed that is the robot on the
  ground: a rollout latches the frame it fell on and holds it (see `record` in
  walk2jump), so the fall stays in view for as long as the other arm is still going.
  """
  if track.shape[1] >= span:
    return track[:, :span]
  tail = track[:, -1:].expand(-1, span - track.shape[1], -1)
  return torch.cat([track, tail], dim=1)


def _cycle(handovers: list[int], timeline: list[int]):
  """Every hand-over in turn, frame by frame, over and over."""
  while True:
    for order, index in enumerate(handovers):
      for frame in timeline:
        yield order, index, frame


# The experiments this script can run. Adding one is a builder above and a line here.
# Soccer and table tennis are missing on purpose: neither declares an `Experiment` yet
# (no view, no window plan, no checkpoint folder), so there is nothing for `build` to
# put an architecture together from
SCENARIOS: dict[str, Callable[[], Scenario]] = {
  "cartpole": _cartpole,
  "diffdrive": _diffdrive,
}


@dataclass(frozen=True)
class HandoverOptions:
  """The parkour hand-over, as `tests/walk2jump.py` builds it. Ignored elsewhere."""

  # How many hand-overs to build and then play in turn. Each draws its own cut into the
  # walk and its own jump distance, so they are different questions rather than one
  # repeated. Raised to two worlds per panel if it is smaller than that
  handovers: int = 4

  # Delta t, in frames: how far past the cut the target is placed, and how long the
  # bridge has to arrive in it. Keep it inside the range the bridge was trained across
  gap: int = 30

  # Frames before the crouch to hand the robot over. Zero is the crouch itself
  entry_lead: int = 10

  # The bridge to run. None takes the newest under logs/rsl_rl; the path is printed
  checkpoint: str | None = None

  # How see-through the nominal jump's ghost is drawn, 0 to 1
  ghost_alpha: float = 0.35

  # A beat held on the last frame of each hand-over before the next one starts [s]
  hold_seconds: float = 0.6


@dataclass(frozen=True)
class CompareConfig:
  # Which experiment to run. cartpole and diffdrive run their composition live (see
  # SCENARIOS); parkour plays one walk-to-jump hand-over (see _run_handover)
  experiment: Literal["cartpole", "diffdrive", "parkour"] = "parkour"

  # The architectures to put side by side, left to right. One panel each, so
  # `--archs 0 1` is the baseline against the first bridge. On parkour the arms are
  # 0 (no bridge) and 4 (arch_4's bridge), so `--archs 0 4`
  archs: tuple[int, ...] = (0, 1)

  # Run directories for the trained architectures, as `--checkpoints 1 <path>`.
  # Anything omitted falls back to that architecture's latest run. Architecture 0
  # trains nothing and needs none, and parkour takes its bridge from
  # --handover.checkpoint instead
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

  # Seed the first episode; each later one is seeded from this plus its index. On
  # parkour it seeds the whole set of hand-overs
  seed: int = 0

  # The parkour hand-over's own settings, as `--handover.gap 40`
  handover: HandoverOptions = field(default_factory=HandoverOptions)

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
  title: str | None = None,
) -> np.ndarray:
  """One video frame: the labelled header strip over the panels, side by side.

  One label per panel, drawn over that panel, and `title` for anything that is true of
  the whole frame rather than of one panel, right-aligned at the end of the strip.
  """
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
    if title is not None:
      draw.text(
        (panels.shape[1] - 10, _HEADER_HEIGHT // 2),
        title,
        fill=_LABEL_COLOR,
        font=font,
        anchor="rm",
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
  env: ManagerBasedRlEnv,
  camera: Camera,
  entity_name: str,
  cfg: CompareConfig,
  worlds: Sequence[int],
) -> list[OffscreenRenderer]:
  """One renderer per panel, all set up from the same camera settings.

  `worlds` says which world each panel draws, which is not always the panel's own index:
  the hand-over mode keeps a ghost in the world next to each panel's robot.

  Deliberately not one renderer moved from world to world. A tracking camera does not
  jump to the body it follows, it eases towards it, so a single camera alternating
  between the panels would keep half of each other panel's lookat: with one robot down
  and the other still running, both views would drift towards the midpoint between
  them. A camera per panel follows only its own robot, and since they are configured
  identically the panels stay comparable.
  """
  renderers = []
  for i in worlds:
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
      viewer.entity_name = entity_name
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


class _GhostOverlay:
  """Draws a second world's robot into a panel's image, faded.

  The renderer already knows how to put another world's geoms into a frame: that is how
  it draws neighbouring environments for context. What it does not do is let the caller
  say which world, and it picks by origin distance, which decides nothing on a plane
  arena where every origin is the same point. So the world is named here instead, and
  the appended geoms are faded on the way in, which is what makes the second robot read
  as a reference rather than as a competitor.

  Only dynamic geoms are added, so the ghost brings its own body and not a second copy
  of the ground.
  """

  def __init__(self, model: mujoco.MjModel, alpha: float) -> None:
    self._model = model
    self._data = mujoco.MjData(model)
    self._opt = mujoco.MjvOption()
    self._pert = mujoco.MjvPerturb()
    self._alpha = alpha

  def add(self, renderer: OffscreenRenderer, data: Any, world: int) -> None:
    if self._model.nq > 0:
      self._data.qpos[:] = data.qpos[world].cpu().numpy()
      self._data.qvel[:] = data.qvel[world].cpu().numpy()
    mujoco.mj_forward(self._model, self._data)
    scene = renderer.renderer.scene
    first = scene.ngeom
    mujoco.mjv_addGeoms(
      self._model,
      self._data,
      self._opt,
      self._pert,
      mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
      scene,
    )
    for i in range(first, scene.ngeom):
      scene.geoms[i].rgba[3] = self._alpha


def _roll_naive_jump(arena, pool, window, case, num_joints) -> tuple[Any, Any]:
  """Hand the jump the walk's last state and let it start the way it always starts.

  Architecture 0 with nothing in it. At the cut the jump skill's own `reset` fires,
  which pins its clip to wherever the robot is and winds it to frame zero, the clip's
  opening stand (see `JumpSkill`, and `JumpCommand.anchor_to_robot`, which moves the
  reference and never the robot). So nothing about the body changes across the instant:
  it keeps the pose and the momentum the walk left it with, and what changes is that the
  thing it is now tracking begins standing still at its feet. A robot at a metre a second
  being told to reproduce a stand is the failure this whole experiment is about, and it
  is what the panel has to show.

  Deliberately not walk2jump's "no bridge" arm, which asks a different question. That one
  winds the clip to the entry frame and leaves it pinned where the nominal rollout had
  it, because it is scored against a bridge that was supposed to deliver the robot into
  that state and did not. Started there, the jump is continuing a motion from the middle,
  which is a reference a moving robot can look plausible against. Here there is no bridge
  and no promise: the jump just starts.

  The clip and its stretch are the ones the other arm jumps, so the two panels differ in
  how the jump is entered and in nothing else. Auto-reset is off and every frame is kept,
  so the fall this produces runs all the way to the floor.
  """
  from mjlab.tasks.skills.experiments.parkour import ENTITY_NAME
  from mjlab.tasks.skills.experiments.parkour.controller import JUMP
  from mjlab.tasks.skills.experiments.parkour.tests.walk2jump import (
    jump_command,
    read_state,
    refresh,
    write_state,
  )

  robot = arena.scene[ENTITY_NAME]
  count = arena.num_envs
  jump = jump_command(arena)

  auto_reset = arena.cfg.auto_reset
  arena.cfg.auto_reset = False
  arena.reset()
  write_state(robot, window.hand_off, num_joints)
  arena.scene.write_data_to_sim()
  arena.sim.forward()

  jump.motion_ids[:] = window.motion_ids
  jump.scales[:] = window.scales
  everything = torch.ones(count, dtype=torch.bool, device=arena.device)
  pool[JUMP].reset(everything)
  obs = refresh(arena)

  assignment = torch.full((count,), JUMP, dtype=torch.long, device=arena.device)
  alive = everything.clone()
  fell = torch.zeros(count, dtype=torch.bool, device=arena.device)
  frames = [read_state(robot)]
  for _ in range(case.resume_steps):
    with torch.inference_mode():
      action = pool.act(obs, assignment)
    obs, _, terminated, time_out, _ = arena.step(action)
    arena._manual_reset_pending.zero_()  # noqa: SLF001
    fell = fell | (terminated & alive)
    alive = alive & ~(terminated | time_out)
    frames.append(read_state(robot))

  arena.cfg.auto_reset = auto_reset
  return torch.stack(frames, dim=1), fell


def _run_handover(cfg: CompareConfig, device: str) -> None:
  """Play the walk-to-jump hand-over, one arm per panel.

  The question and the walk and jump rollouts it is cut out of are `tests/walk2jump.py`'s,
  imported rather than rebuilt, and so is the bridged arm, which is what that script
  scores. What this adds is the side-by-side, the video, and a baseline that is the naive
  hand-off rather than that script's own (see `_roll_naive_jump`).

  The bridged panel carries the nominal jump as a faded ghost, wound so its entry falls
  on the frame its robot arrives at. The two meeting there is the thing to watch, and the
  distance between them when they get there is the arrival error the table reports.

  Nothing is stepped here. Both arms were rolled before the recording began, so playback
  is writing recorded frames onto the robot and rendering, which is also why the panels
  cannot drift apart the way a live comparison does.
  """
  from mjlab.tasks.skills.experiments.parkour import ENTITY_NAME, build_pool
  from mjlab.tasks.skills.experiments.parkour.jump.jump_env_cfg import ANCHOR_BODY
  from mjlab.tasks.skills.experiments.parkour.tests.walk2jump import (
    Arm,
    build_arena,
    build_handover,
    build_tracks,
    cross_with_bridge,
    report,
    resume_jump,
    write_state,
  )
  from mjlab.tasks.skills.experiments.parkour.tests.walk2jump import (
    Config as HandoverCase,
  )

  panels = len(cfg.archs)
  unknown = [a for a in cfg.archs if a not in _HANDOVER_ARMS]
  if unknown:
    raise ValueError(
      f"The parkour comparison plays one walk-to-jump hand-over, and the only arms it "
      f"has are {sorted(_HANDOVER_ARMS)}: 0 hands the jump the walk's last state "
      f"directly, 4 crosses the hole with arch_4's bridge. Got {unknown}. "
      f"Try --archs 0 4."
    )

  # One world per panel for the robot and one for its ghost, and one hand-over per
  # world, so the arena has to hold whichever is the larger of the two.
  case = HandoverCase(
    num_envs=max(cfg.handover.handovers, 2 * panels),
    gap=cfg.handover.gap,
    entry_lead=cfg.handover.entry_lead,
    checkpoint=Path(cfg.handover.checkpoint) if cfg.handover.checkpoint else None,
    seed=cfg.seed,
    device=device,
  )
  torch.manual_seed(case.seed)

  arena = build_arena(case.num_envs, device)
  robot = arena.scene[ENTITY_NAME]
  num_joints = robot.data.joint_pos.shape[-1]
  torso = robot.body_names.index(ANCHOR_BODY)
  pool = build_pool(arena, device)

  window = build_handover(arena, pool, case, torso, num_joints)
  usable = window.valid.nonzero(as_tuple=False).squeeze(-1).tolist()
  if not usable:
    raise SystemExit(
      "None of the hand-overs came out usable: every walk or jump rollout fell over "
      "before the window could be cut. Try again with a different --seed."
    )

  # The crossing, once, however many panels asked for it.
  crossing: torch.Tensor | None = None
  standing: torch.Tensor | None = None
  if 4 in cfg.archs:
    crossing, standing, _ = cross_with_bridge(case, window, device)

  # The full resume rather than the bridge's target window: what a hand-over did is only
  # legible once the jump it was handed to has landed.
  frames = case.resume_steps
  past = window.past.shape[1]
  tracks: list[torch.Tensor] = []
  ghosts: list[torch.Tensor | None] = []
  gaps: list[int] = []
  arms: dict[str, Arm] = {}
  naive_fell: Any | None = None
  for architecture in cfg.archs:
    if architecture == 4:
      assert crossing is not None and standing is not None
      # A bridge that fell in the hole is judged on that; the state it left the robot
      # in is still what the jump is handed, which is what the panel shows
      arrival = crossing[:, -1].clone()
      resume = resume_jump(
        arena, pool, window, arrival, case, num_joints, keep_falling=True
      )
      track, ghost = build_tracks(window, crossing, resume, frames)
      tracks.append(torch.as_tensor(track, device=device))
      ghosts.append(torch.as_tensor(ghost, device=device))
      gaps.append(case.gap)
      arms[_HANDOVER_ARMS[architecture]] = Arm(
        arrival=arrival, resume=resume, crossed=standing
      )
    else:
      # Walk, then the jump starting the way it always starts, with nothing in between
      # and nothing to wait through. No ghost: there is no nominal rollout on this side
      # to hold it against, since the clip is pinned to the robot rather than to a place
      # the robot was supposed to reach
      states, naive_fell = _roll_naive_jump(arena, pool, window, case, num_joints)
      tracks.append(torch.cat([window.past, states[:, :frames]], dim=1))
      ghosts.append(None)
      gaps.append(0)

  # One clock for every panel, set by the arm that takes longest
  span = max(track.shape[1] for track in tracks)
  tracks = [_hold_to(track, span) for track in tracks]
  ghosts = [None if g is None else _hold_to(g, span) for g in ghosts]

  # walk2jump's own table, on the same rollouts the video is about to show, so the
  # numbers and the pictures cannot be about different runs. Only the bridged arm goes
  # through it: that table scores an arm against the entry the bridge was aimed at, and
  # the naive arm is not aimed anywhere -- its clip is pinned to the robot, so there is
  # no target to be near and nothing for an arrival distance to mean. What is worth
  # saying about it is whether it stayed up.
  if arms:
    report(window, arms, num_joints)
  if naive_fell is not None:
    down = int((naive_fell & window.valid).sum())
    print(
      f"\nno bridge\n     the jump was started from its own first frame, a stand, on a "
      f"robot still walking: fell in {down} of {int(window.valid.sum())} hand-overs"
    )

  renderers = _build_renderers(
    arena, _HANDOVER_CAMERA, ENTITY_NAME, cfg, [2 * i for i in range(panels)]
  )
  ghost_overlay = _GhostOverlay(arena.sim.mj_model, cfg.handover.ghost_alpha)

  control_hz = 1.0 / arena.step_dt
  render_every = max(1, round(control_hz / cfg.fps))
  fps = control_hz / render_every
  budget = max(1, round(cfg.seconds * fps))
  # A beat on the last frame, so the eye lands on where the jump ended before the next
  # hand-over starts.
  timeline = list(range(0, span, render_every))
  timeline += [span - 1] * max(0, round(cfg.handover.hold_seconds * fps))
  shape = _canvas_shape(panels, cfg.width, cfg.height)
  font = _load_font(18)

  path = _output_path(cfg)
  path.parent.mkdir(parents=True, exist_ok=True)
  print(
    f"\n[INFO] recording {panels} panel(s) at {fps:.1f} fps to {path}\n"
    f"[INFO] {len(usable)} usable hand-over(s), "
    f"{len(timeline) / fps:.1f}s each, {cfg.seconds:.0f}s of video"
  )

  written = 0
  try:
    with media.VideoWriter(path, shape=shape, fps=fps, crf=20) as writer:
      for order, k, t in _cycle(usable, timeline):
        if written >= budget:
          break
        # Every world is posed at once: each panel's robot on an even world, its ghost
        # on the odd one beside it, and whatever is left over parked on a copy. A panel
        # with no ghost leaves its odd world on that copy, which nothing draws.
        states = tracks[0][k, t].unsqueeze(0).repeat(case.num_envs, 1)
        for i in range(panels):
          states[2 * i] = tracks[i][k, t]
          ghost = ghosts[i]
          if ghost is not None:
            states[2 * i + 1] = ghost[k, t]
        write_state(robot, states, num_joints)
        arena.scene.write_data_to_sim()
        arena.sim.forward()

        images = []
        for i, renderer in enumerate(renderers):
          renderer.update(arena.sim.data)
          if ghosts[i] is not None:
            ghost_overlay.add(renderer, arena.sim.data, 2 * i + 1)
          images.append(np.array(renderer.render(), dtype=np.uint8))
        labels = [
          f"{_HANDOVER_ARMS[a]}: {_phase(t, past, g)}"
          for a, g in zip(cfg.archs, gaps, strict=True)
        ]
        writer.add_image(
          _compose(
            images, labels, font, shape, title=f"hand-over {order + 1}/{len(usable)}"
          )
        )

        written += 1
        if written % 300 == 0:
          print(f"[INFO] {written / fps:.0f}s of {cfg.seconds:.0f}s")
  finally:
    for renderer in renderers:
      renderer.close()
    arena.close()

  print(f"[INFO] wrote {written} frames ({written / fps:.1f}s) to {path}")


def _run_live(cfg: CompareConfig, device: str) -> None:
  """Run the composition itself, one architecture per world, and record it.

  What every experiment but parkour does here: a controller names a skill, an
  architecture carries out the switch, and the panels show the same episode being
  driven several ways.
  """
  scenario = SCENARIOS[cfg.experiment]()
  panels = len(cfg.archs)

  env_cfg = scenario.env_cfg()
  env_cfg.scene.num_envs = panels
  # The panels restart together, which is this script's decision to make rather than
  # the env's; see the linger countdown in the loop below
  env_cfg.auto_reset = False
  # No render_mode: the env's own renderer draws one world, and this needs one per
  # architecture (see _build_renderers)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  renderers = _build_renderers(
    env, scenario.camera, scenario.entity_name, cfg, range(panels)
  )

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


def run_comparison(cfg: CompareConfig) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  if not cfg.archs:
    raise ValueError("--archs needs at least one architecture, e.g. --archs 0 1.")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.experiment == "parkour":
    _run_handover(cfg, device)
  elif cfg.experiment in SCENARIOS:
    _run_live(cfg, device)
  else:
    raise ValueError(
      f"Unknown experiment '{cfg.experiment}'; registered: "
      f"{sorted([*SCENARIOS, 'parkour'])}."
    )


if __name__ == "__main__":
  run_comparison(tyro.cli(CompareConfig))
