"""Retarget SMPL-X human motion onto a humanoid robot, with GMR.

AMASS clips are SMPL-X body parameters, not robot joint angles, so nothing in mjlab can
read them. Every other adapter here sidesteps that by taking motions somebody else already
retargeted. This one does the retargeting.

GMR (Ze et al., "GMR: General Motion Retargeting", ICRA 2026) is not a learned model. It
loads a MuJoCo model of the robot and solves, per frame, for the pose that puts the robot's
key bodies on the human's, using mink's differential IK: one small QP, ten iterations, warm
started from the previous frame. That runs at 35 to 70 frames per second on one CPU core.
No training, no GPU, no dataset of paired poses.

Output is a Unitree generalized coordinate CSV: root position, root quaternion in xyzw,
then the 29 joint angles. That is what csv_to_npz reads and what the bridging skills'
converters read, so a retargeted clip enters either pipeline with nothing in between.

What happens to a clip, in order:

    1. Evaluate the SMPL-X body model to get human joint positions per frame, resampled to
       the retargeting rate.
    2. Solve each frame for the robot pose that matches them, scaled by the subject's
       height so a tall performer does not stretch the robot.
    3. Scatter the solved angles into the Unitree CSV column order by joint name.
    4. Replay the CSV through csv_to_npz, so MuJoCo forward kinematics fill in the per body
       poses in the body order the tracking task expects.

The CSV is kept by default, because a raw mocap trial is usually longer than the skill you
want out of it. Crop it with the tracking task's manual_crop.py before converting, or pass
--line-range to cut the window here.

Two one time installs, neither of which is an mjlab dependency:

    uv pip install "general_motion_retargeting @ git+https://github.com/YanjieZe/GMR"

and the SMPL-X body models, which are license gated: register at
https://smpl-x.is.tue.mpg.de, download the models, and lay them out so that
<smplx-dir>/smplx/SMPLX_NEUTRAL.npz exists.

Run

1. Retarget one AMASS clip to the G1.

    uv run python -m mjlab.tasks.tracking.scripts.datasets.gmr.retarget       --input-path data/amass_atomic/kick/CMU__CMU_10_10_01_stageii.npz

2. Retarget every clip in a directory.

    uv run python -m mjlab.tasks.tracking.scripts.datasets.gmr.retarget       --input-path data/amass_atomic/kick --output-dir data/gmr_g1

3. Convert only a frame window, when the trial holds more than the skill.

    uv run python -m mjlab.tasks.tracking.scripts.datasets.gmr.retarget       --input-path data/amass_atomic/kick/CMU__CMU_10_10_01_stageii.npz       --line-range "(120, 260)"

4. Retarget to a different robot. GMR ships configs for the T1, H1, N1 and a dozen more.

    uv run python -m mjlab.tasks.tracking.scripts.datasets.gmr.retarget       --input-path <clip> --robot booster_t1_29dof
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tyro
from tqdm import tqdm

import mjlab
from mjlab.scripts import csv_to_npz

# The 29 joints of a Unitree generalized coordinate CSV, in column order. GMR solves for the
# same joints but hands them back in its own model's order, so they are scattered into this
# order by name rather than copied across
CSV_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

SMPLX_DIR = Path("data") / "body_models"
"""Holds the SMPL-X models. smplx.create looks for <smplx-dir>/smplx/SMPLX_<GENDER>.npz"""

# GMR's own scripts align to 30 Hz before solving. Higher costs solver time for detail the
# source mocap does not have; csv_to_npz interpolates up to the control rate afterwards
RETARGET_FPS = 30.0

INSTALL_HINT = (
  "GMR is not installed. It is an optional dependency, used only by this script:\n"
  '  uv pip install "general_motion_retargeting @ git+https://github.com/YanjieZe/GMR"\n'
  "It also needs the SMPL-X body models, which are license gated: register at "
  "https://smpl-x.is.tue.mpg.de and lay them out as <smplx-dir>/smplx/SMPLX_NEUTRAL.npz."
)


def _import_gmr() -> tuple[Any, Any, Any]:
  """Import GMR, or explain how to install it."""
  try:
    from general_motion_retargeting import (  # ty: ignore[unresolved-import]
      GeneralMotionRetargeting,
    )
    from general_motion_retargeting.utils.smpl import (  # ty: ignore[unresolved-import]
      get_smplx_data_offline_fast,
      load_smplx_file,
    )
  except ImportError as exc:
    raise SystemExit(f"{INSTALL_HINT}\n\nImport failed with: {exc}") from exc
  return GeneralMotionRetargeting, load_smplx_file, get_smplx_data_offline_fast


def _qpos_addresses(model: Any) -> list[int]:
  """Where each Unitree CSV joint sits in GMR's qpos vector.

  GMR returns the robot's raw qpos, so the joint columns follow whatever order its XML
  declares. Looking the addresses up by name means a reordered or renamed model fails here
  with a clear message instead of silently producing a limb swapped motion.
  """
  import mujoco as mj

  address = {}
  for joint in range(model.njnt):
    if model.jnt_type[joint] in (mj.mjtJoint.mjJNT_FREE, mj.mjtJoint.mjJNT_BALL):
      continue
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint)
    address[name] = int(model.jnt_qposadr[joint])

  missing = [name for name in CSV_JOINT_NAMES if name not in address]
  if missing:
    raise SystemExit(
      f"GMR's robot model is missing {len(missing)} of the 29 Unitree joints, starting "
      f"with '{missing[0]}'. Retarget to unitree_g1, or extend CSV_JOINT_NAMES for the "
      "robot you meant."
    )
  return [address[name] for name in CSV_JOINT_NAMES]


def retarget_clip(
  smplx_file: Path,
  output_csv: Path,
  smplx_dir: Path,
  robot: str,
  retarget_fps: float,
  verbose: bool,
) -> float:
  """Retarget one SMPL-X clip, writing a Unitree CSV. Returns the CSV's fps."""
  gmr_cls, load_smplx_file, align_fps = _import_gmr()

  if not (smplx_dir / "smplx").is_dir():
    raise SystemExit(
      f"No SMPL-X models under {smplx_dir / 'smplx'}. Register at "
      "https://smpl-x.is.tue.mpg.de and put SMPLX_NEUTRAL.npz there, or point "
      "--smplx-dir at an existing copy."
    )

  smplx_data, body_model, smplx_output, human_height = load_smplx_file(
    str(smplx_file), str(smplx_dir)
  )
  frames, fps = align_fps(
    smplx_data, body_model, smplx_output, tgt_fps=int(round(retarget_fps))
  )

  # The height sets the human to robot scale, so the IK targets land at the robot's own
  # proportions rather than the performer's
  retarget = gmr_cls(
    actual_human_height=human_height,
    src_human="smplx",
    tgt_robot=robot,
    verbose=verbose,
  )
  addresses = _qpos_addresses(retarget.model)

  rows = np.zeros((len(frames), 7 + len(CSV_JOINT_NAMES)), dtype=np.float32)
  for i, frame in enumerate(tqdm(frames, desc=smplx_file.stem, ncols=100)):
    qpos = retarget.retarget(frame)
    rows[i, :3] = qpos[:3]
    rows[i, 3:7] = qpos[3:7][[1, 2, 3, 0]]  # wxyz to the CSV's xyzw
    rows[i, 7:] = qpos[addresses]

  output_csv.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_csv, rows, delimiter=",")
  print(f"  {len(frames)} frames @ {fps:g} fps -> {output_csv}")
  return float(fps)


def main(
  input_path: Path,
  output_dir: Path = Path("data") / "gmr_g1",
  smplx_dir: Path = SMPLX_DIR,
  robot: str = "unitree_g1",
  retarget_fps: float = RETARGET_FPS,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  keep_csv: bool = True,
  line_range: tuple[int, int] | None = None,
  convert: bool = True,
  verbose: bool = False,
) -> None:
  """Retarget SMPL-X clips onto a robot and convert them to mjlab motion npz files.

  Args:
    input_path: One SMPL-X npz, or a directory searched recursively for them.
    output_dir: Where the npz files land. CSVs go in output_dir/csv.
    smplx_dir: Holds the SMPL-X body models, as <smplx-dir>/smplx/SMPLX_NEUTRAL.npz.
    robot: A GMR target, such as unitree_g1, booster_t1_29dof or unitree_h1_2.
    retarget_fps: Rate the IK solves at. GMR's own scripts use 30.
    output_fps: Should match the env control rate, 1 / (timestep * decimation).
    device: Torch device for the MuJoCo replay in csv_to_npz.
    render: Render a video of the replay.
    keep_csv: Keep the intermediate CSVs. On by default, since cropping a skill out of a
      raw trial happens on the CSV.
    line_range: Frame window of the CSV to convert, 1-indexed and inclusive. Leave unset to
      convert the whole clip.
    convert: Retarget only, skipping the csv_to_npz replay, when False.
    verbose: Let GMR print its IK config and body tables.
  """
  if input_path.is_dir():
    clips = sorted(p for p in input_path.rglob("*.npz") if p.is_file())
    if not clips:
      raise SystemExit(f"No .npz clips under {input_path}")
  elif input_path.is_file():
    clips = [input_path]
  else:
    raise SystemExit(f"No such file or directory: {input_path}")

  csv_dir = output_dir / "csv"
  print(f"Retargeting {len(clips)} clip(s) to {robot}:")

  for clip in clips:
    name = clip.stem
    csv_path = csv_dir / f"{name}.csv"
    fps = retarget_clip(
      smplx_file=clip,
      output_csv=csv_path,
      smplx_dir=smplx_dir,
      robot=robot,
      retarget_fps=retarget_fps,
      verbose=verbose,
    )

    if convert:
      csv_to_npz.main(
        input_file=str(csv_path),
        output_name=name,
        output_dir=output_dir,
        input_fps=fps,
        output_fps=output_fps,
        device=device,
        render=render,
        upload_to_wandb=False,
        line_range=line_range,
      )
    if not keep_csv:
      csv_path.unlink(missing_ok=True)

  print(
    f"\nDone. Motions in {output_dir}/. Train one with:\n"
    "  uv run train Mjlab-Tracking-Flat-Unitree-G1 "
    f"--env.commands.motion.motion-file {output_dir}/<name>.npz --env.scene.num-envs 4096"
  )
  if keep_csv:
    print(
      "Crop a skill out of a longer trial first with:\n"
      "  uv run python src/mjlab/tasks/tracking/scripts/datasets/lafan1/interactive_crop.py "
      f"--data-dir {csv_dir} --motion <name>"
    )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
