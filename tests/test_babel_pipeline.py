import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from mjlab.datasets.babel.materialize import materialize
from mjlab.retargeting.gmr import retarget_manifest


def _write_manifest(path: Path) -> None:
  entries = [
    {
      "babel_sid": 7,
      "segment_id": f"segment-{index}",
      "segment_ids": [f"label-{index}"],
      "split": "train",
      "subset": "ACCAD",
      "source": "ACCAD/ACCAD/person/walk_poses.npz",
      "start_s": float(index),
      "end_s": float(index + 1),
      "categories": ["walk"],
      "label": "walk",
      "labels": ["walk"],
    }
    for index in range(2)
  ]
  path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))


def test_materialize_and_retarget_manifest(tmp_path: Path, monkeypatch: Any) -> None:
  manifest = tmp_path / "manifest.jsonl"
  _write_manifest(manifest)
  archive_dir = tmp_path / "archives"
  archive_dir.mkdir()
  with tarfile.open(archive_dir / "ACCAD.tar.bz2", "w:bz2") as archive:
    payload = b"smplx"
    member = tarfile.TarInfo("download/ACCAD/person/walk_stageii.npz")
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))

  input_dir = tmp_path / "amass"
  assert materialize(manifest, archive_dir, input_dir) == (1, 0)
  source = input_dir / "ACCAD/ACCAD/person/walk_poses.npz"
  assert source.read_bytes() == b"smplx"

  retarget_calls = []

  def fake_retarget(*args, **kwargs) -> float:
    retarget_calls.append(args[0])
    output = args[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, np.zeros((61, 30)), delimiter=",")
    return 30.0

  ranges = []

  def fake_convert(**kwargs) -> None:
    ranges.append(kwargs["line_range"])
    output = kwargs["output_dir"] / f"{kwargs['output_name']}.npz"
    np.savez(output, robot=np.asarray(kwargs["robot"]), joint_pos=np.zeros((3, 29)))

  monkeypatch.setattr(retarget_manifest.retarget, "retarget_clip", fake_retarget)
  monkeypatch.setattr(retarget_manifest.csv_to_npz, "main", fake_convert)
  monkeypatch.setattr(
    retarget_manifest.csv_to_npz,
    "robot_joint_names",
    lambda robot: tuple(f"joint_{index}" for index in range(29)),
  )

  output_dir = tmp_path / "retargeted"
  result = retarget_manifest.retarget_entries(
    manifest,
    input_dir,
    output_dir,
    tmp_path / "body_models",
    "unitree_g1",
    device="cpu",
  )

  assert result == (2, 0)
  assert retarget_calls == [source]
  assert ranges == [(1, 30), (31, 60)]
  motions = sorted((output_dir / "train/walk").glob("*.npz"))
  assert len(motions) == 2
  with np.load(motions[0]) as motion:
    assert str(motion["babel_source"]) == "ACCAD/ACCAD/person/walk_poses.npz"
    assert motion["babel_categories"].tolist() == ["walk"]
    assert motion["babel_segment_ids"].tolist() == ["label-0"]
    assert motion["babel_labels"].tolist() == ["walk"]

  assert retarget_manifest.retarget_entries(
    manifest,
    input_dir,
    output_dir,
    tmp_path / "body_models",
    "unitree_g1",
    device="cpu",
  ) == (0, 2)
  assert len(retarget_calls) == 1
