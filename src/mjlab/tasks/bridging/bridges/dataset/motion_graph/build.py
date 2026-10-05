"""Build stitched clips: two filtered BABEL or LAFAN clips joined at a look-alike frame.

1. Keep the frames the kinematic filter accepted, describe each one, see graph.features.
2. Link frames to their closest look-alikes in other clips.
3. Pick links, every category pair equally likely. Each gives one clip: before_s of the
   first clip up to the link, then after_s of the second from it. The second piece is
   placed and turned so the root lines up, and the pose difference left at the seam fades
   out over blend_s.
4. Rebuild body poses and velocities. A clip with any frame the kinematic filter rejects
   is dropped, not split.

The loaders only cut windows that cross the seam from these clips (motions.crosses_seam),
so a window always holds a transition the recorded clips do not. The defaults leave room
for the longest window of the planner and the tracker, with their history and future.

Output: data/motion_graph/<robot>/{train,val}/stitched/*.npz, the BABEL and LAFAN layout
plus stitch_seams (first frame of the second piece), stitch_pieces (the two source clips)
and stitch_distance. A rerun replaces the previous output.

Run

1. Build. Needs the motion capture corpus.

    uv run python -m mjlab.tasks.bridging.bridges.dataset.motion_graph.build --robot g1

2. Inspect, Corpus Stitched in the dropdown.

    uv run python -m mjlab.tasks.bridging.bridges.dataset.view
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.tasks.bridging.bridges.dataset.motion_capture.filters import (
  BRIDGE_MOTION_FILTER,
  motion_bad_frames,
)
from mjlab.tasks.bridging.bridges.dataset.motion_graph import clips, graph
from mjlab.tasks.bridging.config import get_robot
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_box_minus,
  quat_box_plus,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)


@dataclass
class BuildCfg:
  robot: str = "g1"
  train_clips: int = 10000
  val_clips: int = 2000
  before_s: float = 0.4
  """First clip, up to the seam. Covers a window opening 0.3 s before it, plus history."""
  after_s: float = 2.7
  """Second clip, from the seam. Covers a 2 s window, plus the tracker's 32 step future."""
  blend_s: float = 0.2
  max_distance: float = 0.25
  """Feature distance under which two frames are a transition."""
  neighbors: int = 8
  stride: int = 2
  """Only every stride-th frame can be jumped from."""
  seed: int = 0
  device: str = "cuda:0"


def stitch(
  states: torch.Tensor, first: tuple[int, int], second: tuple[int, int], blend: int
) -> torch.Tensor:
  """qpos (T, 7 + J) of rows first, then rows second, both ranges inclusive.

  The first row of second is the look-alike of the last row of first, so it is dropped
  and the second piece starts at index first[1] - first[0] + 1.
  """
  joints = (states.shape[1] - 13) // 2

  def qpos(begin: int, end: int) -> tuple[torch.Tensor, ...]:
    s = states[begin : end + 1]
    return s[:, 0:3].clone(), s[:, 3:7].clone(), s[:, 13 : 13 + joints].clone()

  pos, quat, joint = qpos(*first)
  p, q, j = qpos(*second)

  # Turn and shift the second piece so its first root lands on the last one
  turn = quat_mul(yaw_quat(quat[-1]), quat_conjugate(yaw_quat(q[0])))
  turn = turn.expand(len(q), 4)
  p[:, :2] -= p[0, :2].clone()
  p = quat_apply(turn, p)
  p[:, :2] += pos[-1, :2]
  q = quat_mul(turn, q)

  # Fade the pose difference at the seam to zero
  k = torch.arange(len(q), dtype=torch.float32)
  fade = torch.where(k < blend, 0.5 * (1 + torch.cos(torch.pi * k / blend)), 0.0)
  p[:, 2] += fade * (pos[-1, 2] - p[0, 2])
  q = quat_box_plus(q, fade[:, None] * quat_box_minus(quat[-1:], q[:1]))
  j += fade[:, None] * (joint[-1] - j[0])

  return torch.cat(
    [
      torch.cat([pos, p[1:]]),
      torch.cat([quat, q[1:]]),
      torch.cat([joint, j[1:]]),
    ],
    dim=-1,
  )


def angular_velocity(quat: torch.Tensor, fps: float) -> torch.Tensor:
  """World angular velocity from (T, ..., 4) orientations, central differences."""
  step = quat_box_minus(quat[1:].reshape(-1, 4), quat[:-1].reshape(-1, 4))
  step = step.reshape(*quat[1:].shape[:-1], 3) * fps
  return torch.cat([step[:1], 0.5 * (step[1:] + step[:-1]), step[-1:]])


def finish(
  qpos: np.ndarray, model: mujoco.MjModel, fps: float, robot: str
) -> dict[str, np.ndarray] | None:
  """Every array a clip file holds, from its qpos. None when the filter rejects a frame."""
  data = mujoco.MjData(model)
  body_pos = np.zeros((len(qpos), model.nbody - 1, 3))
  body_quat = np.zeros((len(qpos), model.nbody - 1, 4))
  for t, row in enumerate(qpos):
    data.qpos[:] = row
    mujoco.mj_kinematics(model, data)
    body_pos[t], body_quat[t] = data.xpos[1:], data.xquat[1:]
  lin_vel = np.asarray(np.gradient(body_pos, axis=0)) * fps
  ang_vel = angular_velocity(torch.from_numpy(body_quat), fps).numpy()
  joint_pos = qpos[:, 7:]
  joint_vel = np.asarray(np.gradient(joint_pos, axis=0)) * fps

  state = np.concatenate(
    [
      body_pos[:, 0],
      body_quat[:, 0],
      lin_vel[:, 0],
      ang_vel[:, 0],
      joint_pos,
      joint_vel,
    ],
    axis=-1,
  )
  bad = motion_bad_frames(state, body_pos, body_quat, fps, BRIDGE_MOTION_FILTER, robot)
  if np.logical_or.reduce(tuple(bad.values())).any():
    return None
  arrays = {
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "body_pos_w": body_pos,
    "body_quat_w": body_quat,
    "body_lin_vel_w": lin_vel,
    "body_ang_vel_w": ang_vel,
  }
  arrays = {k: v.astype(np.float32) for k, v in arrays.items()}
  arrays["valid"] = np.ones(len(qpos), dtype=bool)
  return arrays


def build_split(cfg: BuildCfg, split: str, count: int, rng: np.random.Generator):
  files = clips.corpus_files("BABEL", cfg.robot, split)
  files += clips.corpus_files("LAFAN", cfg.robot, split)
  corpus = clips.load(files)
  fps = corpus.fps
  before = round(cfg.before_s * fps)
  after = round(cfg.after_s * fps)

  f = graph.frames(corpus, cfg.device)
  t = graph.transitions(f, cfg.max_distance, cfg.neighbors, cfg.stride, before, after)
  if len(t.source) == 0:
    raise SystemExit("No transitions. Raise --max-distance.")
  print(
    f"[{split}] {len(f.clip)} frames from {len(files)} clips, {len(t.source)} "
    f"transitions, distance p10/p50/p90 "
    f"{np.percentile(t.distance, [10, 50, 90]).round(2).tolist()}"
  )

  out = clips.corpus_root("Stitched", cfg.robot) / split / "stitched"
  for old in out.glob("*.npz"):
    old.unlink()

  model = get_robot(cfg.robot).get_spec().compile()
  states = f.states.cpu()
  blend = max(1, round(cfg.blend_s * fps))
  # Every link at most once, in random order, more likely for rare category pairs
  order = np.argsort(rng.exponential(size=len(t.source)) / graph.pair_weights(f, t))
  pairs: Counter[str] = Counter()
  written = dropped = 0
  for pick in order:
    if written == count:
      break
    source, target = int(t.source[pick]), int(t.target[pick])
    qpos = stitch(
      states, (source - before + 1, source), (target, target + after - 1), blend
    )
    arrays = finish(qpos.double().numpy(), model, fps, corpus.robot)
    if arrays is None:
      dropped += 1
      continue
    a, b = corpus.clips[f.clip[source]], corpus.clips[f.clip[target]]
    arrays["stitch_seams"] = np.asarray([before], dtype=np.int64)
    arrays["stitch_pieces"] = np.asarray(
      [f"{a.category}/{a.name}", f"{b.category}/{b.name}"]
    )
    arrays["stitch_distance"] = np.asarray([t.distance[pick]], dtype=np.float32)
    clips.save(out / f"{written:05d}.npz", corpus, arrays)
    pairs[f"{a.category} -> {b.category}"] += 1
    written += 1

  print(
    f"[{split}] wrote {written} clips to {out}, {written * (before + after - 1) / fps / 60:.1f} min, "
    f"{dropped} dropped by the filter, {len(pairs)} category pairs"
  )
  for pair, n in pairs.most_common(10):
    print(f"[{split}]   {n:5d}  {pair}")


def main(cfg: BuildCfg) -> None:
  rng = np.random.default_rng(cfg.seed)
  build_split(cfg, "train", cfg.train_clips, rng)
  build_split(cfg, "val", cfg.val_clips, rng)


if __name__ == "__main__":
  main(tyro.cli(BuildCfg, config=mjlab.TYRO_FLAGS))
