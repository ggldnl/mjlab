# pyright: reportPrivateImportUsage=false

from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.scripts import csv_to_npz
from mjlab.scripts.csv_to_npz import robot_joint_names
from mjlab.tasks.bridging.tests.retarget_amass import select_clip
from mjlab.tasks.tracking.mdp.commands import MotionLoader


def test_supported_robot_joint_orders() -> None:
  assert len(robot_joint_names("unitree_g1")) == 29
  assert len(robot_joint_names("booster_t1")) == 23


def test_select_clip_is_stable(tmp_path: Path) -> None:
  skill_dir = tmp_path / "walk"
  skill_dir.mkdir()
  (skill_dir / "b.npz").touch()
  (skill_dir / "a.npz").touch()

  assert select_clip(tmp_path, "walk", 0).name == "a.npz"
  assert select_clip(tmp_path, "walk", 1).name == "b.npz"


def test_motion_loader_reads_retargeted_clip(tmp_path: Path) -> None:
  path = tmp_path / "motion.npz"
  np.savez(
    path,
    robot=np.asarray("test_robot"),
    joint_names=np.asarray(("joint_a",)),
    body_names=np.asarray(("body",)),
    joint_pos=np.zeros((2, 1)),
    joint_vel=np.zeros((2, 1)),
    body_pos_w=np.zeros((2, 1, 3)),
    body_quat_w=np.zeros((2, 1, 4)),
    body_lin_vel_w=np.zeros((2, 1, 3)),
    body_ang_vel_w=np.zeros((2, 1, 3)),
  )

  motion = MotionLoader(str(path), torch.tensor((0,)))

  assert motion.time_step_total == 2
  assert motion.joint_pos.shape == (2, 1)
  assert motion.body_pos_w.shape == (2, 1, 3)


def test_t1_csv_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp"))
  rows = np.zeros((4, 7 + len(robot_joint_names("booster_t1"))))
  rows[:, 2] = 0.7
  rows[:, 6] = 1.0
  csv = tmp_path / "t1.csv"
  np.savetxt(csv, rows, delimiter=",")

  csv_to_npz.main(
    input_file=str(csv),
    output_name="t1",
    robot="booster_t1",
    output_dir=tmp_path,
    input_fps=30,
    output_fps=30,
    device="cpu",
  )

  with np.load(tmp_path / "t1.npz") as motion:
    assert str(motion["robot"]) == "booster_t1"
    assert motion["joint_pos"].shape[1] == 23
