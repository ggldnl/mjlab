import json
from pathlib import Path

from mjlab.datasets.babel.build_manifest import build_manifest


def test_build_manifest_selects_safe_dense_and_single_action_labels(
  tmp_path: Path,
) -> None:
  records = {
    "1": {
      "babel_sid": 1,
      "feat_p": "ACCAD/person/motion_poses.npz",
      "dur": 5.0,
      "frame_ann": {
        "labels": [
          {
            "seg_id": "walk",
            "proc_label": "walk",
            "act_cat": ["walk"],
            "start_t": 0.0,
            "end_t": 2.0,
          },
          {
            "seg_id": "mixed",
            "proc_label": "walk and jump",
            "act_cat": ["walk", "jump"],
            "start_t": 2.0,
            "end_t": 4.0,
          },
        ]
      },
    },
    "2": {
      "babel_sid": 2,
      "feat_p": "Transitionsmocap/Transitions_mocap/person/turn_poses.npz",
      "dur": 3.0,
      "seq_ann": {
        "mul_act": False,
        "labels": [{"proc_label": "turn", "act_cat": ["turn"]}],
      },
    },
    "3": {
      "feat_p": "MPIHDM05/MPI_HDM05/person/mixed_poses.npz",
      "dur": 4.0,
      "seq_ann": {
        "mul_act": True,
        "labels": [{"proc_label": "walk", "act_cat": ["walk"]}],
      },
    },
    "4": {
      "feat_p": "CMU/person/walk_poses.npz",
      "dur": 2.0,
      "seq_ann": {
        "mul_act": False,
        "labels": [{"proc_label": "walk", "act_cat": ["walk"]}],
      },
    },
    "5": {
      "babel_sid": 5,
      "feat_p": "MPIHDM05/MPI_HDM05/person/crouch_poses.npz",
      "dur": 2.0,
      "seq_ann": {
        "mul_act": False,
        "labels": [{"proc_label": "crouch", "act_cat": ["crouch"]}],
      },
    },
    "6": {
      "feat_p": "Transitionsmocap/Transitions_mocap/person/airkick_stand_poses.npz",
      "dur": 2.0,
      "seq_ann": {
        "mul_act": False,
        "labels": [{"proc_label": "transition", "act_cat": ["transition"]}],
      },
    },
  }
  for split in ("train", "val", "test"):
    data = records if split == "train" else {}
    (tmp_path / f"{split}.json").write_text(json.dumps(data), encoding="utf-8")

  manifest = build_manifest(tmp_path)

  assert [
    (entry["source"], entry["start_s"], entry["end_s"]) for entry in manifest
  ] == [
    ("ACCAD/person/motion_poses.npz", 0.0, 2.0),
    ("MPI_HDM05/MPI_HDM05/person/crouch_poses.npz", 0.0, 2.0),
    ("Transitions_mocap/Transitions_mocap/person/turn_poses.npz", 0.0, 3.0),
  ]


def test_build_manifest_merges_allowed_labels_and_subtracts_denied_time(
  tmp_path: Path,
) -> None:
  records = {
    "1": {
      "babel_sid": 1,
      "feat_p": "ACCAD/person/motion_poses.npz",
      "dur": 6.0,
      "frame_ann": {
        "labels": [
          {
            "seg_id": "walk",
            "proc_label": "walk",
            "act_cat": ["walk"],
            "start_t": 0.0,
            "end_t": 3.0,
          },
          {
            "seg_id": "turn",
            "proc_label": "turn",
            "act_cat": ["turn"],
            "start_t": 3.0,
            "end_t": 6.0,
          },
          {
            "seg_id": "jump",
            "proc_label": "jump",
            "act_cat": ["jump"],
            "start_t": 2.0,
            "end_t": 4.0,
          },
        ]
      },
    }
  }
  for split in ("train", "val", "test"):
    data = records if split == "train" else {}
    (tmp_path / f"{split}.json").write_text(json.dumps(data), encoding="utf-8")

  manifest = build_manifest(tmp_path)

  assert [(entry["start_s"], entry["end_s"]) for entry in manifest] == [
    (0.0, 2.0),
    (4.0, 6.0),
  ]
  assert [entry["labels"] for entry in manifest] == [["walk"], ["turn"]]


def test_build_manifest_can_keep_every_category_except_denied_time(
  tmp_path: Path,
) -> None:
  records = {
    "1": {
      "feat_p": "ACCAD/person/motion_poses.npz",
      "dur": 5.0,
      "frame_ann": {
        "labels": [
          {
            "proc_label": "sit",
            "act_cat": ["sit"],
            "start_t": 0.0,
            "end_t": 3.0,
          },
          {
            "proc_label": "throw",
            "act_cat": ["throw"],
            "start_t": 3.0,
            "end_t": 5.0,
          },
          {
            "proc_label": "jump",
            "act_cat": ["jump"],
            "start_t": 1.0,
            "end_t": 2.0,
          },
        ]
      },
    }
  }
  for split in ("train", "val", "test"):
    data = records if split == "train" else {}
    (tmp_path / f"{split}.json").write_text(json.dumps(data), encoding="utf-8")

  manifest = build_manifest(tmp_path, allow_all=True)

  assert [(entry["start_s"], entry["end_s"]) for entry in manifest] == [
    (0.0, 1.0),
    (2.0, 5.0),
  ]
  assert manifest[1]["categories"] == ["sit", "throw"]
