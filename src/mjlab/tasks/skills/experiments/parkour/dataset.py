"""Build the parkour skill dataset: curated LAFAN1 clips, one folder per skill.

The Unitree-retargeted LAFAN1 files are long, multi-motion performances: one file
wanders through several behaviors with natural human transitions between them.
This script cuts them into short, single-behavior clips and files each clip under
the skill its source belongs to, so every skill gets its own reference set:

  data/lafan1_g1/raw/<clip>.csv                 the downloaded LAFAN1 clip
  data/lafan1_g1/segments/<skill>/<name>.csv    one curated cut, as CSV
  data/lafan1_g1/clips/<skill>/<name>.npz       the same cut replayed on the G1
  data/lafan1_g1/manifest.json                  every cut and its velocity label

`clips/<skill>/` is what a skill trains on: `parkour_env_cfg.py` points the
discriminator at that folder and it reads every npz in it.

A cut carries a velocity label, [v_fwd, v_lat, v_up, yaw_rate] in the robot's own
yaw frame, stored per frame in the npz. The label is not what conditions the
policy at training time -- the command box in `parkour_env_cfg.py` is -- it is what
tells you what that box should be, which is why this script ends by printing the
per-skill statistics to paste there.

How the cutting works, per skill (see `SkillSpec`):

- `plateau` (walk, run, sprint): keep the stretches where the yaw-frame command
  barely changes, and drop the ramps between them. A stretch that turns, speeds up
  or slows down is not one steady goal, so it is not one clip. Vertical velocity is
  deliberately left out of this test: it oscillates every stride and never plateaus.
- `takeoff` (jump): a jump is exactly the transient a plateau test throws away, so
  the opposite test applies -- find the bursts of upward root velocity and pad them,
  so the cut holds the crouch, the takeoff, the flight and the landing.

The two `plateau` knobs are coupled and worth understanding before retuning them:
the root surges once per stride, so `smooth_window` has to be long enough to see
through that surge and `change_tol` loose enough to tolerate what is left. Both
scale with speed. At running pace a walking-tight tolerance finds nothing but the
stretches where the human is standing still (which is how a run clip ends up
contributing zero run data), and at walking pace a running-loose one swallows the
whole clip, turns included.

Either way a cut is then accepted only if its label agrees with the skill it would
be filed under: `fwd_range` is a signed window, so a stretch where the human is
walking backwards or standing still inside a run clip does not become run data.

Run it (accept the dataset terms at
https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset once,
then export your token -- the repo is gated):

  export HF_TOKEN=hf_...  # $env:HF_TOKEN="..." on Windows PowerShell
  uv run python -m mjlab.tasks.skills.experiments.parkour.dataset

  # cut the CSVs but skip the (slow) replay, to check the segmentation first
  uv run python -m mjlab.tasks.skills.experiments.parkour.dataset --convert False

  # rebuild one skill after retuning its spec
  uv run python -m mjlab.tasks.skills.experiments.parkour.dataset --skills "('jump',)"

Any produced clip is also a valid tracking motion, so it can be eyeballed with:

  uv run play Mjlab-Tracking-Flat-Unitree-G1 --agent zero --no-terminations True
    --motion-file data/lafan1_g1/clips/run/run1_subject2_00.npz
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

# LAFAN1 is 30 FPS. A CSV row is base pos (3) + base quat xyzw (4) + 29 joints.
FPS = 30.0

# Gated HuggingFace repo; the G1 CSVs live under the `g1/` subfolder.
HF_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/g1"

# The label channels, in order, as stored in every clip's `command` array.
COMMAND_CHANNELS = ("v_fwd", "v_lat", "v_up", "yaw_rate")

# The fields of a clip npz that hold the motion itself (the tracking format).
MOTION_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)

# Padding, in seconds, added either side of a takeoff so the cut also holds the
# crouch and the landing -- as much a part of the jump as the flight is.
TAKEOFF_PAD = 0.35


@dataclass(frozen=True)
class SkillSpec:
  """How one skill's reference set is cut out of its source clips.

  This is the whole tuning surface of the dataset: which clips feed a skill, how
  they are cut, and which cuts count as that skill. The values below were read off
  the LAFAN1 clips channel by channel; they are meant to be retuned, and
  `--no-convert` is the fast way to do it.
  """

  clips: tuple[str, ...]
  """LAFAN1 clip names filed under this skill."""
  segmenter: Literal["plateau", "takeoff"] = "plateau"
  """`plateau` for steady locomotion, `takeoff` for jumps (see the module docstring)."""
  smooth_window: int = 31
  """Moving average, in frames, over the label before it is cut on. This is the knob
  that decides what counts as one behavior: a stride surges the root once per step,
  so the window has to be long enough to see through that and short enough not to
  smear a real change of speed. Faster gaits need a longer one (a run's surge is
  bigger), a jump needs a short one or its takeoff is averaged away."""
  fwd_range: tuple[float, float] = (-100.0, 100.0)
  """Accepted window for the cut's mean forward velocity [m/s]. Signed, so a
  negative window is backwards locomotion and a window starting above zero rejects
  the standing and backwards stretches found inside a clip."""
  max_abs_lat: float = 100.0
  """Reject a cut whose mean lateral velocity exceeds this [m/s]: sidestepping is
  not the same behavior as walking, even in a walk clip."""
  max_abs_yaw_rate: float = 100.0
  """Reject a cut whose mean turn rate exceeds this [rad/s]. LAFAN1 has plenty of
  spinning hops and sharp turns, and a reference set that mixes them in teaches the
  discriminator that turning on the spot is part of the behavior."""
  min_peak_up: float = 0.0
  """Reject a cut whose peak vertical velocity stays below this [m/s]. For the
  `takeoff` segmenter this doubles as the takeoff threshold: it is what separates a
  jump from a bouncy stride, so it is the one number that defines a jump here."""
  min_duration: float = 1.0
  """Reject a cut shorter than this [s]: too brief to be a behavior."""
  change_tol: tuple[float, float, float] = (2.0, 2.0, 4.0)
  """`plateau` only: the largest per-second change in [v_fwd, v_lat, yaw_rate] that
  still counts as steady. It scales with speed -- at walking pace a tolerance this
  loose swallows the whole clip, turns included, while at running pace a tight one
  finds nothing but the stretches where the human is standing still."""


# The four skills. Note that run and sprint overlap in speed on purpose: they are
# separated by style (their own folders, so their own discriminators) and by the
# command box each is trained on, not by a hard speed boundary in the data.
SKILLS: dict[str, SkillSpec] = {
  "walk": SkillSpec(
    clips=(
      "walk1_subject1",
      "walk1_subject2",
      "walk1_subject5",
      "walk3_subject1",
      "walk3_subject2",
    ),
    smooth_window=15,
    change_tol=(0.8, 0.8, 1.6),
    fwd_range=(0.3, 1.7),
    max_abs_lat=0.6,
    max_abs_yaw_rate=1.0,
  ),
  "run": SkillSpec(
    clips=("run1_subject2", "run1_subject5", "run2_subject1", "run2_subject4"),
    fwd_range=(1.5, 3.5),
    max_abs_lat=0.8,
    max_abs_yaw_rate=1.0,
  ),
  "sprint": SkillSpec(
    clips=("sprint1_subject2",),
    fwd_range=(2.8, 6.0),
    max_abs_lat=0.8,
    max_abs_yaw_rate=1.0,
  ),
  "jump": SkillSpec(
    clips=("jumps1_subject1", "jumps1_subject2", "jumps1_subject5"),
    segmenter="takeoff",
    smooth_window=5,
    fwd_range=(-1.0, 2.0),
    max_abs_lat=0.6,
    max_abs_yaw_rate=1.0,
    min_peak_up=0.6,
    min_duration=0.8,
  ),
}

# The 29 G1 joints the CSV's DOF columns map to, in order.
G1_JOINT_NAMES = [
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
]

##
# Download and labelling
##


def download(clip: str, raw_dir: Path) -> Path:
  """Fetch one G1 CSV into `raw_dir/<clip>.csv` (skips if already there)."""
  path = raw_dir / f"{clip}.csv"
  if path.exists():
    return path
  token = os.environ.get("HF_TOKEN")
  if not token:
    raise SystemExit(
      "The LAFAN1 repo is gated. Accept its terms on HuggingFace and set HF_TOKEN "
      "before running."
    )
  raw_dir.mkdir(parents=True, exist_ok=True)
  request = urllib.request.Request(
    f"{HF_BASE}/{clip}.csv", headers={"Authorization": f"Bearer {token}"}
  )
  print(f"Downloading {clip}.csv -> {path} ...")
  with urllib.request.urlopen(request) as response, open(path, "wb") as f:
    f.write(response.read())
  return path


def smooth(signal: np.ndarray, window: int) -> np.ndarray:
  """Centered per-channel moving average, edge-padded so length is preserved."""
  if window <= 1:
    return signal
  pad = window // 2
  padded = np.pad(signal, ((pad, pad), (0, 0)), mode="edge")
  kernel = np.ones(window) / window
  return np.stack(
    [
      np.convolve(padded[:, i], kernel, mode="valid")[: len(signal)]
      for i in range(signal.shape[1])
    ],
    axis=1,
  )


def label_command(rows: np.ndarray, smooth_window: int) -> np.ndarray:
  """The per-frame velocity label [v_fwd, v_lat, v_up, yaw_rate], yaw frame.

  Expressing it in the root's own yaw frame is what makes the label independent of
  where in the world the clip happens and which way the human faces: a backwards
  run comes out as a negative v_fwd rather than as a forward run pointing the other
  way, which is what lets `SkillSpec.fwd_range` reject it.
  """
  dt = 1.0 / FPS
  pos = rows[:, 0:3]
  x, y, z, w = rows[:, 3:7].T  # LAFAN1 stores the quaternion xyzw
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

  vel_world = np.gradient(pos, axis=0) / dt
  cos, sin = np.cos(yaw), np.sin(yaw)
  v_fwd = cos * vel_world[:, 0] + sin * vel_world[:, 1]
  v_lat = -sin * vel_world[:, 0] + cos * vel_world[:, 1]
  yaw_rate = np.gradient(np.unwrap(yaw)) / dt

  return smooth(
    np.stack([v_fwd, v_lat, vel_world[:, 2], yaw_rate], axis=1), smooth_window
  )


##
# Segmentation
##


def _runs(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
  """The [start, end) ranges where `mask` holds for at least `min_frames`."""
  ranges: list[tuple[int, int]] = []
  i, n = 0, len(mask)
  while i < n:
    if not mask[i]:
      i += 1
      continue
    j = i
    while j < n and mask[j]:
      j += 1
    if j - i >= min_frames:
      ranges.append((i, j))
    i = j
  return ranges


def plateau_segments(command: np.ndarray, spec: SkillSpec) -> list[tuple[int, int]]:
  """Cuts where the command holds still: steady goals, transitions dropped.

  Vertical velocity is excluded from the test on purpose -- it swings once per
  stride, so requiring it to be steady would reject every stretch of locomotion.
  """
  planar = command[:, [0, 1, 3]]  # v_fwd, v_lat, yaw_rate
  change = np.abs(np.gradient(planar, axis=0)) * FPS  # per second
  steady = np.all(change <= np.array(spec.change_tol), axis=1)
  return _runs(steady, int(spec.min_duration * FPS))


def takeoff_segments(command: np.ndarray, spec: SkillSpec) -> list[tuple[int, int]]:
  """Cuts around each burst of upward root velocity, padded and merged.

  A jump is exactly the transient a plateau test throws away, so it gets found the
  other way round: `min_peak_up` is the takeoff threshold, and the padding brings in
  the crouch before and the landing after. Consecutive hops merge into one cut,
  which is fine -- hopping is what the jump skill is being asked for.
  """
  pad = int(TAKEOFF_PAD * FPS)
  segments: list[tuple[int, int]] = []
  for start, end in _runs(command[:, 2] > spec.min_peak_up, 2):
    start, end = max(0, start - pad), min(len(command), end + pad)
    if segments and start <= segments[-1][1]:
      segments[-1] = (segments[-1][0], end)
    else:
      segments.append((start, end))
  min_frames = int(spec.min_duration * FPS)
  return [(s, e) for s, e in segments if e - s >= min_frames]


def cut(command: np.ndarray, spec: SkillSpec) -> list[tuple[int, int]]:
  """Every candidate cut of one clip, by whichever segmenter the skill uses."""
  if spec.segmenter == "plateau":
    return plateau_segments(command, spec)
  return takeoff_segments(command, spec)


def accept(command: np.ndarray, spec: SkillSpec) -> bool:
  """Whether one cut's label agrees with the skill it would be filed under."""
  mean = command.mean(axis=0)
  return bool(
    spec.fwd_range[0] <= mean[0] <= spec.fwd_range[1]
    and abs(mean[1]) <= spec.max_abs_lat
    and abs(mean[3]) <= spec.max_abs_yaw_rate
    and command[:, 2].max() >= spec.min_peak_up
  )


##
# Replay
##


def build_sim(device: str, output_fps: float) -> tuple[Simulation, Scene]:
  """The single-env G1 sim every cut is replayed through (expensive; build once)."""
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def replay(
  sim: Simulation, scene: Scene, csv_path: Path, output_fps: float
) -> dict[str, np.ndarray]:
  """Drive the robot through one CSV cut and record what the sim reports.

  The recorded state comes off the entity rather than out of the CSV, which is why
  this is a replay and not a format conversion: joint and body ordering, and the
  derived body velocities, are then exactly the ones the env will see at training
  time.
  """
  motion = MotionLoader(
    motion_file=str(csv_path),
    input_fps=int(FPS),
    output_fps=int(output_fps),
    device=sim.device,
  )
  robot: Entity = scene["robot"]
  joint_idx = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]
  log: dict[str, list[np.ndarray]] = {field: [] for field in MOTION_FIELDS}

  scene.reset()
  for _ in tqdm(
    range(motion.output_frames),
    desc=csv_path.stem,
    unit="frame",
    ncols=100,
    leave=False,
  ):
    (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel), _ = (
      motion.get_next_state()
    )

    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] = base_pos
    root_state[:, :2] += scene.env_origins[:, :2]
    root_state[:, 3:7] = base_rot
    root_state[:, 7:10] = base_lin_vel
    root_state[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_state)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_idx] = dof_pos
    joint_vel[:, joint_idx] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    data = robot.data
    log["joint_pos"].append(data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(data.body_link_ang_vel_w[0].cpu().numpy().copy())

  return {field: np.stack(frames, axis=0) for field, frames in log.items()}


def resample_command(command: np.ndarray, num_frames: int) -> np.ndarray:
  """Stretch a label sampled at `FPS` onto the replay's `num_frames` timeline."""
  src = np.linspace(0.0, 1.0, len(command))
  dst = np.linspace(0.0, 1.0, num_frames)
  return np.stack(
    [np.interp(dst, src, command[:, i]) for i in range(command.shape[1])], axis=1
  )


def save_clip(
  path: Path, motion: dict[str, np.ndarray], command: np.ndarray, fps: float
) -> None:
  """Write one reference clip: the replayed motion plus its velocity label.

  The motion fields are the tracking npz format, so a clip is also playable with
  `uv run play Mjlab-Tracking-Flat-Unitree-G1 --motion-file <clip>`.
  """
  np.savez(
    path,
    fps=np.array([fps]),
    command=resample_command(command, len(motion["joint_pos"])).astype(np.float32),
    joint_pos=motion["joint_pos"],
    joint_vel=motion["joint_vel"],
    body_pos_w=motion["body_pos_w"],
    body_quat_w=motion["body_quat_w"],
    body_lin_vel_w=motion["body_lin_vel_w"],
    body_ang_vel_w=motion["body_ang_vel_w"],
  )


##
# Reporting
##


def print_summary(skill: str, entries: list[dict[str, Any]]) -> None:
  """Per-skill label statistics, and the command box they suggest.

  Two statistics, because they answer different questions. The mean over a cut is
  what a *held* command looks like, so it bounds what is sensible to ask of the
  policy for seconds at a time -- that is the box for v_fwd, v_lat and yaw_rate.
  Vertical velocity averages to nothing over any cut (a jump comes back down), so
  what matters there is the peak, which is why v_up is reported separately.
  """
  if not entries:
    print(f"{skill}: no clips accepted")
    return

  means = np.array([e["command_mean"] for e in entries])
  duration = sum(e["num_frames"] for e in entries) / FPS
  print(f"{skill}: {len(entries)} clip(s), {duration:.1f} s of reference motion")
  for i, channel in enumerate(COMMAND_CHANNELS):
    column = means[:, i]
    print(
      f"  mean {channel:9s} min {column.min():6.2f}  avg {column.mean():6.2f}  "
      f"max {column.max():6.2f}"
    )
  peak_up = np.array([e["peak_up"] for e in entries])
  print(f"  peak v_up      min {peak_up.min():6.2f}  max {peak_up.max():6.2f}")
  print(
    "  suggested box: "
    f"v_fwd ({means[:, 0].min():.1f}, {means[:, 0].max():.1f}), "
    f"v_lat ({-abs(means[:, 1]).max():.1f}, {abs(means[:, 1]).max():.1f}), "
    f"yaw_rate ({-abs(means[:, 3]).max():.1f}, {abs(means[:, 3]).max():.1f})"
  )


##
# Entry point
##


def main(
  data_dir: Path = Path("data/lafan1_g1"),
  skills: tuple[str, ...] = tuple(SKILLS),
  convert: bool = True,
  device: str = "cuda:0",
  output_fps: float = 50.0,
) -> None:
  """Download, cut, label and replay the reference clips for each skill.

  `--convert False` stops after writing the segment CSVs, which is the fast way to
  check a `SkillSpec` before paying for the replay. `--skills "('run', 'jump')"`
  rebuilds only those skills, leaving the other folders alone.
  """
  unknown = set(skills) - set(SKILLS)
  if unknown:
    raise SystemExit(f"Unknown skill(s) {sorted(unknown)}; known: {sorted(SKILLS)}.")
  if convert and device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"

  raw_dir = data_dir / "raw"
  sim: Simulation | None = None
  scene: Scene | None = None
  manifest: dict[str, list[dict[str, Any]]] = {}

  for skill in skills:
    spec = SKILLS[skill]
    segment_dir = data_dir / "segments" / skill
    clip_dir = data_dir / "clips" / skill
    segment_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    for source in spec.clips:
      rows = np.loadtxt(download(source, raw_dir), delimiter=",")
      command = label_command(rows, spec.smooth_window)
      cuts = cut(command, spec)
      kept = [(s, e) for s, e in cuts if accept(command[s:e], spec)]
      print(
        f"{skill}/{source}: {len(rows)} frames -> {len(cuts)} cut(s), "
        f"{len(kept)} accepted"
      )

      for index, (start, end) in enumerate(kept):
        name = f"{source}_{index:02d}"
        segment_csv = segment_dir / f"{name}.csv"
        np.savetxt(segment_csv, rows[start:end], delimiter=",")

        entry: dict[str, Any] = {
          "name": name,
          "skill": skill,
          "source": source,
          "start": int(start),
          "end": int(end),
          "num_frames": int(end - start),
          "command_mean": command[start:end].mean(axis=0).round(4).tolist(),
          "peak_up": round(float(command[start:end, 2].max()), 4),
          "csv": str(segment_csv),
        }

        if convert:
          if sim is None:
            print("Building the G1 replay sim (first conversion)...")
            sim, scene = build_sim(device, output_fps)
          assert scene is not None  # set together with sim above
          motion = replay(sim, scene, segment_csv, output_fps)
          clip_dir.mkdir(parents=True, exist_ok=True)
          clip_path = clip_dir / f"{name}.npz"
          save_clip(clip_path, motion, command[start:end], output_fps)
          entry["clip"] = str(clip_path)

        entries.append(entry)

    manifest[skill] = entries

  manifest_path = data_dir / "manifest.json"
  if manifest_path.exists():
    # Only the rebuilt skills are replaced, so a partial run does not erase the rest.
    # Unknown keys are dropped rather than carried along: an older manifest sitting in
    # this directory is a different format, not a skill.
    previous = json.loads(manifest_path.read_text())
    kept = {k: v for k, v in previous.items() if k in SKILLS and k not in manifest}
    manifest = {**kept, **manifest}
  manifest_path.write_text(json.dumps(manifest, indent=2))

  print()
  for skill in skills:
    print_summary(skill, manifest[skill])
  print(f"\nWrote {sum(len(e) for e in manifest.values())} clip(s) under {data_dir}/")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
