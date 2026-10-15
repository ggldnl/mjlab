"""Throwaway: play the converted T1 jump clips back kinematically, no physics.

Every clip is written straight into qpos frame by frame, so what you see is the retargeted
reference itself, not what a policy makes of it. Prints each clip's numbers as it starts.

    red sphere      root at takeoff
    green sphere    root at the last frame, the goal the policy is paid for
    blue sphere     root at the first frame

Keys in the viewer

    space       pause
    right/left  next/previous clip
    r           restart the clip
    t           toggle the training trim (the clip as the tracker sees it)

Run

1. Every clip, ordered by distance, as converted.

    uv run python -m mjlab.tasks.bridging.config.t1.skills.jump.view_clips

2. Only clips whose name contains a string, trimmed like training, at half speed.

    uv run python -m mjlab.tasks.bridging.config.t1.skills.jump.view_clips \
      --match KIT --trimmed True --speed 0.5
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

import mjlab
from mjlab.asset_zoo.robots.booster_t1.t1_constants import get_spec
from mjlab.scripts import csv_to_npz
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.jump_continuous_env_cfg import (
  LEAD_IN,
)
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.motion_lib import (
  discover_motion_files,
  trim_lead_in,
)
from mjlab.tasks.bridging.config.t1.skills.jump.dataset import DEFAULT_OUTPUT_DIR

KEY_SPACE, KEY_RIGHT, KEY_LEFT, KEY_R, KEY_T = 32, 262, 263, 82, 84


def build_model() -> mujoco.MjModel:
  spec = get_spec()
  spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], rgba=[0.6, 0.6, 0.6, 1]
  )
  spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
  return spec.compile()


def load_clip(path: Path, trimmed: bool, lead_in_steps: int) -> dict[str, np.ndarray]:
  clip = dict(np.load(path))
  return trim_lead_in(clip, lead_in_steps) if trimmed else clip


def describe(path: Path, clip: dict[str, np.ndarray], meta: dict | None) -> str:
  root = clip["body_pos_w"][:, 0]
  frames = len(root)
  fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
  takeoff, land = int(clip["takeoff_step"]), int(clip["land_step"])
  lines = [
    f"\n{path.stem}",
    f"  {frames} frames, {frames / fps:.2f} s at {fps:g} Hz",
    f"  distance {np.linalg.norm(root[-1, :2] - root[0, :2]):.2f} m, "
    f"apex {root[:, 2].max() - root[0, 2]:+.2f} m, root height {root[0, 2]:.2f} m",
    f"  takeoff {takeoff} ({takeoff / fps:.2f} s), land {land} ({land / fps:.2f} s), "
    f"flight {(land - takeoff) / fps:.2f} s",
  ]
  if meta is not None:
    lines.append(
      f"  float {meta['stance_float']:+.3f} m, sink {meta['ground_penetration']:+.3f} m, "
      f"source {meta.get('source', '?')} frames {meta.get('source_frames', '?')}"
    )
  return "\n".join(lines)


def main(
  motion_dir: Path = DEFAULT_OUTPUT_DIR,
  match: str = "",
  trimmed: bool = False,
  speed: float = 1.0,
) -> None:
  """Play the clips.

  Args:
    motion_dir: Where dataset.py wrote the clips.
    match: Only clips whose file name contains this.
    trimmed: Start each clip where training does, LEAD_IN seconds before the crouch.
    speed: Playback speed, 1 is real time.
  """
  paths = [Path(p) for p in discover_motion_files(motion_dir) if match in Path(p).name]
  if not paths:
    raise SystemExit(f"No clips in {motion_dir} matching '{match}'.")
  manifest_path = motion_dir / "manifest.json"
  manifest = (
    {e["file"]: e for e in json.loads(manifest_path.read_text())}
    if manifest_path.exists()
    else {}
  )

  model = build_model()
  data = mujoco.MjData(model)
  free = next(
    j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
  )
  root_adr = int(model.jnt_qposadr[free])
  joint_adr = [
    int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
    for n in csv_to_npz.robot_joint_names("booster_t1")
  ]

  state = {"index": 0, "frame": 0, "paused": False, "trimmed": trimmed, "reload": True}

  def on_key(key: int) -> None:
    if key == KEY_SPACE:
      state["paused"] = not state["paused"]
    elif key in (KEY_RIGHT, KEY_LEFT):
      step = 1 if key == KEY_RIGHT else -1
      state["index"] = (state["index"] + step) % len(paths)
      state["reload"] = True
    elif key == KEY_R:
      state["frame"] = 0
    elif key == KEY_T:
      state["trimmed"] = not state["trimmed"]
      state["reload"] = True

  clip: dict[str, np.ndarray] = {}
  fps = 50.0
  with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    while viewer.is_running():
      if state["reload"]:
        path = paths[state["index"]]
        fps = 50.0
        clip = load_clip(path, bool(state["trimmed"]), int(round(LEAD_IN * fps)))
        fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
        print(describe(path, clip, manifest.get(path.name)))
        print(f"  [{state['index'] + 1}/{len(paths)}] trimmed={state['trimmed']}")
        state["frame"], state["reload"] = 0, False

      t = int(state["frame"])
      data.qpos[root_adr : root_adr + 3] = clip["body_pos_w"][t, 0]
      data.qpos[root_adr + 3 : root_adr + 7] = clip["body_quat_w"][t, 0]
      data.qpos[joint_adr] = clip["joint_pos"][t]
      mujoco.mj_forward(model, data)

      root = clip["body_pos_w"][:, 0]
      takeoff = max(int(clip["takeoff_step"]), 0)
      markers = (
        (root[0], (0, 0, 1)),
        (root[takeoff], (1, 0, 0)),
        (root[-1], (0, 1, 0)),
      )
      viewer.user_scn.ngeom = 0
      for pos, rgb in markers:
        mujoco.mjv_initGeom(
          viewer.user_scn.geoms[viewer.user_scn.ngeom],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=np.array([0.04, 0.0, 0.0]),
          pos=np.array([pos[0], pos[1], 0.04]),
          mat=np.eye(3).flatten(),
          rgba=np.array([*rgb, 1.0]),
        )
        viewer.user_scn.ngeom += 1
      viewer.sync()

      if not state["paused"]:
        state["frame"] = (t + 1) % len(root)
      time.sleep(1.0 / (fps * max(speed, 1e-3)))


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
