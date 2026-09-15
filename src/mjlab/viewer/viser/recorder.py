"""Write the episode to an mp4 from the viewer's Record box."""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import mediapy as media
import numpy as np
import viser

from mjlab.viewer.offscreen_renderer import OffscreenRenderer

SIZES: dict[str, tuple[int, int]] = {
  "480p": (854, 480),
  "720p": (1280, 720),
  "1080p": (1920, 1080),
}

MAX_REPEATS = 4
"""Frames a single capture may be repeated for, when the sim crossed several
frame intervals at once."""


class ViserRecorder:
  """The viewer's Record box: an mp4 of the episode, written while it plays.

  Frames come from an offscreen MuJoCo camera rather than from the browser, so a
  frame costs a few ms instead of a round trip and carries the debug visualizers
  (target ghosts, arrows) with it. The camera is the environment's own
  ViewerConfig, which for these tasks tracks the robot.

  Frames are taken on sim time, not on wall clock, so the file plays back at real
  speed however fast the viewer manages to run, and the speed buttons record as
  slow motion or fast forward.

  Run:
      the Record folder in the viewer's Controls tab. Writes
      <folder>/<name>-<timestamp>.mp4, and Folder is editable in the panel.
  """

  def __init__(
    self,
    server: viser.ViserServer,
    env: Any,
    request: Callable[[], None],
    folder: Path | str = "videos",
    name: str = "episode",
    fps: float = 30.0,
    env_idx: Callable[[], int] = lambda: 0,
  ) -> None:
    self._server = server
    self._env = env
    # The button runs on a viser thread, the capture on the viewer's loop. Toggling
    # from the click would tear down the renderer under a render in progress, so the
    # click only asks the viewer to call `toggle` on its own thread
    self._request = request
    self._folder = Path(folder)
    self._name = name
    self._fps = fps
    self._env_idx = env_idx

    self._renderer: OffscreenRenderer | None = None
    self._writer: Any | None = None
    self._path: Path | None = None
    self._frames = 0
    self._next_time = 0.0
    self._last_time = 0.0

  @property
  def recording(self) -> bool:
    return self._writer is not None

  def create_gui(self) -> None:
    """Build the Record folder. Call inside the tab the box belongs in."""
    with self._server.gui.add_folder("Record"):
      self._folder_input = self._server.gui.add_text(
        "Folder",
        initial_value=str(self._folder),
        hint="Where the mp4 goes. Created if missing.",
      )
      self._size_input = self._server.gui.add_dropdown(
        "Size",
        options=tuple(SIZES),
        initial_value="720p",
      )
      self._button = self._server.gui.add_button(
        "Record", icon=viser.Icon.PLAYER_RECORD
      )
      self._button.on_click(lambda _: self._request())
      self._status = self._server.gui.add_html("")
    self._show("Not recording")

  def toggle(self) -> None:
    """Start or stop. Runs on the viewer's loop thread."""
    if self.recording:
      self.stop()
    else:
      self._start()

  def capture(self, sim_time: float) -> None:
    """Write a frame if sim_time has crossed the next frame boundary."""
    if self._writer is None:
      return

    # First frame, or the environment was reset and the clock restarted
    if self._frames == 0 or sim_time < self._last_time:
      self._next_time = sim_time
    self._last_time = sim_time
    if sim_time < self._next_time:
      return

    frame = self._render()
    if frame is None:
      return

    # A step can cross more than one frame interval, above 1x speed or after a stall.
    # Repeating the frame is what keeps the file on real time
    behind = (sim_time - self._next_time) * self._fps
    repeats = min(int(behind) + 1, MAX_REPEATS)
    for _ in range(repeats):
      self._writer.add_image(frame)
    self._frames += repeats
    # Advance the grid rather than re-anchoring on sim_time, which would round every
    # frame up to the next control step and play the video back fast
    self._next_time += repeats / self._fps
    if behind >= MAX_REPEATS:
      # Further behind than the cap can pay back. Drop the missing frames
      self._next_time = sim_time

    if self._frames % 10 < repeats:
      seconds = self._frames / self._fps
      self._show(f"Recording {self._frames} frames, {seconds:.1f} s", color="#e74c3c")

  def stop(self, error: str | None = None) -> None:
    """Close the file and release the renderer. Safe to call when idle."""
    if self._writer is None:
      return
    frames, path = self._frames, self._path
    try:
      self._writer.close()
    except Exception as err:
      error = error or str(err)
    self._writer = None
    self._path = None
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None

    self._button.label = "Record"
    self._button.icon = viser.Icon.PLAYER_RECORD
    self._size_input.disabled = False
    self._folder_input.disabled = False

    if error is not None:
      print(f"[ERROR]: Recording stopped: {error}")
      self._show(f"Stopped: {error}", color="#e74c3c")
      return
    seconds = frames / self._fps
    print(f"[INFO]: Saved {path} ({frames} frames, {seconds:.1f} s)")
    self._show(f"Saved <strong>{path}</strong><br/>{frames} frames, {seconds:.1f} s")

  def _start(self) -> None:
    width, height = SIZES[self._size_input.value]
    folder = Path(self._folder_input.value).expanduser()
    path = folder / f"{self._name}-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    try:
      folder.mkdir(parents=True, exist_ok=True)
      self._renderer = self._build_renderer(width, height)
      writer = media.VideoWriter(path, shape=(height, width), fps=self._fps)
      writer.__enter__()
    except Exception as err:
      if self._renderer is not None:
        self._renderer.close()
        self._renderer = None
      print(f"[ERROR]: Could not start recording: {err}")
      self._show(f"Failed: {err}", color="#e74c3c")
      return

    self._writer = writer
    self._path = path
    self._frames = 0
    self._button.label = "Stop"
    self._button.icon = viser.Icon.PLAYER_STOP
    # The frame size is baked into the open file and the renderer, and the folder
    # names a file already being written
    self._size_input.disabled = True
    self._folder_input.disabled = True
    print(f"[INFO]: Recording to {path}")
    self._show(f"Recording to {path}", color="#e74c3c")

  def _build_renderer(self, width: int, height: int) -> OffscreenRenderer:
    """One renderer per recording, so it picks up the env selected right now."""
    env = self._env.unwrapped
    sim = env.sim
    cfg = replace(
      env.cfg.viewer, width=width, height=height, env_idx=int(self._env_idx())
    )
    renderer = OffscreenRenderer(
      # A copy: the renderer overrides extent, shadows and the offscreen buffer size
      # on the model it is handed, and the viser scene is reading the live one
      model=copy.copy(sim.mj_model),
      cfg=cfg,
      scene=env.scene,
      sim_model=sim.model,
      expanded_fields=sim.expanded_fields,
    )
    renderer.initialize()
    return renderer

  def _render(self) -> np.ndarray | None:
    assert self._renderer is not None
    env = self._env.unwrapped
    try:
      self._renderer.update(
        env.sim.data, debug_vis_callback=getattr(env, "update_visualizers", None)
      )
      return self._renderer.render()
    except Exception as err:
      self.stop(error=str(err))
      return None

  def _show(self, message: str, color: str | None = None) -> None:
    style = f"color:{color};" if color else ""
    self._status.content = (
      '<div style="font-size:0.85em;line-height:1.25;padding:0 1em 0.5em 1em;'
      f'word-break:break-all;{style}">{message}</div>'
    )
