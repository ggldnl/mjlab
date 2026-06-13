"""Interactive viser tool to preview and crop LAFAN1 G1 motions into clips.

The Unitree-retargeted LAFAN1 CSVs are long, continuous performances (e.g.
``walk1_subject1`` is ~260 s of one subject walking, turning, idling). For the
latent-skill work we want *atomic* clips: a steady forward walk, a run, a jump,
a crouch. ``manual_crop.py`` carves these out by hard-coded frame
ranges, but finding a clean window means guessing a range, rendering an MP4,
and iterating -- slow and blind.

This is the interactive alternative. It spins up a `viser <https://viser.studio>`_
web app (open the printed URL in a browser) that plays the real G1 model back
through the motion, with a scrub bar and start/end markers. Drag the playhead
to a clean stretch, set the start and end, name the clip, and hit save. The
crop is written as a Unitree-convention CSV (the exact rows of the source), so
it drops straight into ``mjlab.scripts.csv_to_npz`` -- or tick "also convert to
NPZ" to run that conversion on the spot.

A LAFAN1 CSV is already the G1 generalized coordinate (base position, base
quaternion in xyzw, then the 29 joint angles, at 30 fps), which is exactly the
compiled tracking model's ``qpos`` layout (modulo the ``xyzw`` -> ``wxyz``
quaternion reorder). So previewing is just: set ``qpos`` per frame, run forward
kinematics, push the body poses to viser. What you see is the same model
``csv_to_npz`` replays, so the crop you pick is faithful.

License: the retargeted data derives from Ubisoft's LAFAN1 (CC BY-NC-ND 4.0,
non-commercial). Cite LAFAN1 (Harvey et al., SIGGRAPH 2020) and the Unitree
retargeting release.

Run with:
  uv run python \
      src/mjlab/tasks/tracking/scripts/datasets/interactive_crop.py
  # point it at a specific data dir / preselect a motion / pick the output dir:
  uv run python \
      src/mjlab/tasks/tracking/scripts/datasets/interactive_crop.py \
      --data-dir data/lafan1_g1 --motion walk1_subject1 \
      --output-dir data/lafan1_g1/skills
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
import tyro
import viser
from mjviser import ViserMujocoScene

import mjlab
from mjlab.scene import Scene
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

# CSV column layout (Unitree generalized coordinate): base position (3), base
# quaternion in xyzw (4), then the 29 G1 joint angles. The compiled tracking
# model's qpos is base position (3), base quaternion in wxyz (4), then the same
# 29 joints in the same order, so the only fix-up is the quaternion reorder.
_QUAT_XYZW_TO_WXYZ = [6, 3, 4, 5]
_EXPECTED_COLS = 36


def _build_model() -> mujoco.MjModel:
  """Compile the G1 tracking scene to the model csv_to_npz replays."""
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device="cpu")
  return scene.compile()


def _list_motions(data_dir: Path) -> list[str]:
  return sorted(p.stem for p in data_dir.glob("*.csv"))


class Cropper:
  """Holds viewer state and wires up the viser GUI."""

  def __init__(
    self,
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    data_dir: Path,
    output_dir: Path,
    fps: float,
    output_fps: float,
  ) -> None:
    self.server = server
    self.data_dir = data_dir
    self.output_dir = output_dir
    self.fps = fps
    self.output_fps = output_fps

    self.mj_model = mj_model
    self.mj_data = mujoco.MjData(mj_model)
    self.scene = ViserMujocoScene(server, mj_model, num_envs=1)
    self.scene.create_scene_gui()

    # Motion state.
    self.lines: list[str] = []  # Raw CSV rows, for lossless cropping.
    self.frames: np.ndarray = np.zeros((0, _EXPECTED_COLS), dtype=np.float32)
    self.n = 0
    self.frame = 0
    self._syncing = False  # Guards programmatic slider writes from on_update.
    self._last_render = -1

    self._build_gui()

  # GUI.

  def _build_gui(self) -> None:
    motions = _list_motions(self.data_dir)
    gui = self.server.gui

    with gui.add_folder("Motion"):
      self.dd_motion = gui.add_dropdown(
        "Clip",
        options=motions or ["<none found>"],
        initial_value=(motions or ["<none found>"])[0],
      )
      self.html_info = gui.add_html("")

      @self.dd_motion.on_update
      def _(_) -> None:
        self.load(self.dd_motion.value)

    with gui.add_folder("Playback"):
      self.cb_play = gui.add_checkbox("Play", initial_value=False)
      self.cb_loop = gui.add_checkbox(
        "Loop crop", initial_value=True, hint="Loop within the crop range."
      )
      self.sl_speed = gui.add_slider(
        "Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
      )
      self.sl_frame = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)

      @self.sl_frame.on_update
      def _(_) -> None:
        if not self._syncing:
          self.frame = int(self.sl_frame.value)

    with gui.add_folder("Crop"):
      self.num_start = gui.add_number(
        "Start frame", initial_value=1, min=1, max=1, step=1
      )
      self.num_end = gui.add_number("End frame", initial_value=1, min=1, max=1, step=1)
      self.btn_start_here = gui.add_button("Set start = playhead")
      self.btn_end_here = gui.add_button("Set end = playhead")
      self.html_crop = gui.add_html("")

      @self.num_start.on_update
      def _(_) -> None:
        self._update_crop_info()

      @self.num_end.on_update
      def _(_) -> None:
        self._update_crop_info()

      @self.btn_start_here.on_click
      def _(_) -> None:
        self.num_start.value = self.frame + 1  # 1-indexed.

      @self.btn_end_here.on_click
      def _(_) -> None:
        self.num_end.value = self.frame + 1  # 1-indexed.

    with gui.add_folder("Save"):
      self.txt_name = gui.add_text("Clip name", initial_value="clip")
      self.cb_npz = gui.add_checkbox(
        "Also convert to NPZ",
        initial_value=False,
        hint="Run csv_to_npz on the saved crop (slow; needs a GPU ideally).",
      )
      self.btn_save = gui.add_button("Save clip")
      self.html_status = gui.add_html("")

      @self.btn_save.on_click
      def _(_) -> None:
        self.save()

    if motions:
      self.load(motions[0])
    else:
      self.html_info.content = (
        f"<p>No CSVs found in <code>{self.data_dir}</code>. Download some with "
        "<code>download.py</code> first.</p>"
      )

  # Motion loading.

  def load(self, name: str) -> None:
    path = self.data_dir / f"{name}.csv"
    if not path.exists():
      self.html_info.content = f"<p>Missing <code>{path}</code>.</p>"
      return
    text = path.read_text()
    self.lines = [ln for ln in text.splitlines() if ln.strip()]
    self.frames = np.array(
      [[float(x) for x in ln.split(",")] for ln in self.lines], dtype=np.float32
    )
    if self.frames.ndim != 2 or self.frames.shape[1] != _EXPECTED_COLS:
      self.html_info.content = (
        f"<p>Expected {_EXPECTED_COLS} columns, got "
        f"{self.frames.shape[1] if self.frames.ndim == 2 else '?'}.</p>"
      )
      return

    self.n = self.frames.shape[0]
    self.frame = 0
    duration = self.n / self.fps
    self.html_info.content = (
      f"<p><b>{name}</b><br>{self.n} frames &middot; {duration:.1f} s "
      f"@ {self.fps:g} fps</p>"
    )

    self._syncing = True
    self.sl_frame.max = max(self.n - 1, 1)
    self.sl_frame.value = 0
    self._syncing = False

    self.num_start.max = self.n
    self.num_end.max = self.n
    self.num_start.value = 1
    self.num_end.value = self.n
    self.txt_name.value = name
    self._update_crop_info()
    self._render(force=True)

  def _update_crop_info(self) -> None:
    if self.n == 0:
      return
    start = int(self.num_start.value)
    end = int(self.num_end.value)
    n_clip = max(end - start + 1, 0)
    self.html_crop.content = (
      f"<p>{start}&ndash;{end} &middot; {n_clip} frames &middot; "
      f"{n_clip / self.fps:.2f} s "
      f"(t = {(start - 1) / self.fps:.2f}&ndash;{(end - 1) / self.fps:.2f} s)</p>"
    )

  # Rendering.

  def _render(self, force: bool = False) -> None:
    if self.n == 0:
      return
    if not force and self.frame == self._last_render:
      return
    row = self.frames[self.frame]
    qpos = self.mj_data.qpos
    qpos[0:3] = row[0:3]
    qpos[3:7] = row[_QUAT_XYZW_TO_WXYZ]
    qpos[7:] = row[7:]
    mujoco.mj_forward(self.mj_model, self.mj_data)
    self.scene.update_from_mjdata(self.mj_data)
    self._last_render = self.frame

  def step(self, dt: float) -> None:
    """Advance playback by wall-clock ``dt`` and render the current frame."""
    if self.n == 0:
      return
    if self.cb_play.value:
      lo, hi = 0, self.n - 1
      if self.cb_loop.value:
        lo = int(self.num_start.value) - 1
        hi = int(self.num_end.value) - 1
      advance = dt * self.fps * self.sl_speed.value
      nxt = self.frame + advance
      if nxt > hi:
        nxt = lo + (nxt - hi - 1) if hi > lo else lo
      self.frame = int(np.clip(round(nxt), lo, hi))
      self._syncing = True
      self.sl_frame.value = self.frame
      self._syncing = False
    self._render()

  # Saving.

  def save(self) -> None:
    if self.n == 0:
      self.html_status.content = "<p>Nothing loaded.</p>"
      return
    start = int(self.num_start.value)
    end = int(self.num_end.value)
    if not (1 <= start <= end <= self.n):
      self.html_status.content = (
        f"<p>Invalid range {start}&ndash;{end} for {self.n} frames.</p>"
      )
      return

    name = self.txt_name.value.strip() or "clip"
    self.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = self.output_dir / f"{name}.csv"
    # Write the exact source rows (start..end, 1-indexed inclusive) so the crop
    # is lossless and stays byte-faithful to the LAFAN1 release.
    out_csv.write_text("\n".join(self.lines[start - 1 : end]) + "\n")

    msg = (
      f"<p>Saved {end - start + 1} frames &rarr; <code>{out_csv}</code><br>"
      f"Convert: <code>uv run -m mjlab.scripts.csv_to_npz "
      f"--input-file {out_csv} --output-name {name} "
      f"--input-fps {self.fps:g} --output-fps {self.output_fps:g}</code></p>"
    )

    if self.cb_npz.value:
      self.html_status.content = msg + "<p>Converting to NPZ&hellip;</p>"
      try:
        from mjlab.scripts import csv_to_npz

        csv_to_npz.main(
          input_file=str(out_csv),
          output_name=name,
          output_dir=self.output_dir,
          input_fps=self.fps,
          output_fps=self.output_fps,
          device="cuda:0",
          render=False,
          upload_to_wandb=False,
          line_range=None,
        )
        msg += f"<p>Wrote <code>{self.output_dir / f'{name}.npz'}</code>.</p>"
      except Exception as e:  # noqa: BLE001 - surface any failure in the UI.
        msg += f"<p>NPZ conversion failed: {e}</p>"

    self.html_status.content = msg
    print(f"Saved crop {start}-{end} -> {out_csv}")


def main(
  data_dir: Path = Path("data/lafan1_g1"),
  output_dir: Path = Path("data/lafan1_g1/skills"),
  motion: str | None = None,
  fps: float = 30.0,
  output_fps: float = 50.0,
  port: int = 8080,
) -> None:
  """Launch the interactive LAFAN1 cropping viewer.

  Args:
    data_dir: Directory holding the LAFAN1 G1 CSVs to browse.
    output_dir: Directory to write saved clips (and NPZs) into.
    motion: Clip to preselect (stem, e.g. ``walk1_subject1``). Defaults to the
      first CSV found.
    fps: Frame rate of the source CSVs (LAFAN1 is 30).
    output_fps: Output frame rate used in the printed/optional NPZ conversion.
    port: Port for the viser web server.
  """
  if not data_dir.is_dir():
    raise SystemExit(
      f"No data dir at {data_dir}. Download LAFAN1 first with "
      "download.py, or pass --data-dir."
    )

  mj_model = _build_model()
  server = viser.ViserServer(port=port, label="LAFAN1 cropper")
  cropper = Cropper(server, mj_model, data_dir, output_dir, fps, output_fps)
  if motion is not None:
    cropper.dd_motion.value = motion
    cropper.load(motion)

  print(f"\nLAFAN1 cropper running at http://localhost:{port} -- Ctrl-C to quit.")
  last = time.time()
  try:
    while True:
      now = time.time()
      cropper.step(now - last)
      last = now
      time.sleep(1.0 / 60.0)
  except KeyboardInterrupt:
    print("\nShutting down.")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
