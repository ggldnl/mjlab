# Tracking motion datasets

Tooling to fetch human-motion data, curate *atomic* single-skill clips (walk,
run, sprint, jump, crouch, ...), and feed them into the tracking pipeline. Every
path ends at `mjlab.scripts.csv_to_npz`, which replays a Unitree generalized
coordinate CSV (base position, base quaternion in xyzw, then the 29 G1 joint
angles) through MuJoCo forward kinematics and writes the `.npz` the
`Mjlab-Tracking-Flat-Unitree-G1` task trains on.

```
download ─▶ curate / crop ─▶ csv_to_npz ─▶ train
```

## Scripts

| Script                       | Source | Output | Notes |
|------------------------------| --- | --- | --- |
| `lafan1/download.py`         | Unitree-retargeted LAFAN1 (HuggingFace) | G1 CSVs | Long continuous performances; crop them (below). |
| `lafan1/interactive_crop.py` | LAFAN1 CSVs | cropped CSV / NPZ | **Interactive viser viewer** to scrub a clip and save a single-skill crop. |
| `lafan1/manual_crop.py`      | LAFAN1 CSVs | per-skill NPZs | Batch, reproducible slicing by hard-coded frame ranges. |
| `openhe/download.py`         | openhe `g1-retargeted-motions` (HuggingFace) | per-skill NPZs | Already atomic; 23-DOF source remapped to 29-DOF G1. |
| `phuma/download.py`          | PHUMA (HuggingFace) | per-skill NPZs | Already atomic, 29-DOF G1, physics-curated; large (~3.4 GB). |
| `amass/download.py`          | AMASS (license-gated MPI portal) | SMPL-X `.npz` | Robot-agnostic; needs your own retargeting before `csv_to_npz`. |
| `asap/download.py`           | ASAP `LeCAR-Lab/ASAP` (GitHub) | 23-DOF joblib `.pkl` | Short single-skill jumps; the bridging jump tasks convert them. |

All paths default to CWD-relative dirs under `data/`; run them from the repo
root with `uv run python src/mjlab/tasks/tracking/scripts/datasets/<dataset>/<script>.py`
and the result will end up in `data/<dataset>`.

## Fetching one clip from a task

Every `download.py` also exposes `fetch(name, output_dir) -> Path`, which
downloads one clip unless it is already cached and returns where it landed.
Tasks that track a single clip call that instead of carrying their own copy of
the URL, so there is one cache per dataset and one place to fix when a release
moves. `MOTIONS` in each script is the list of names `fetch` accepts.

```python
from mjlab.tasks.tracking.scripts.datasets.lafan1 import download as lafan1

csv = lafan1.fetch("fight1_subject2")
```

Converting is not shared. What a task does with a clip after it arrives is
specific enough that at most a couple of tasks want the same thing, and those
import each other directly.

## Which one do I want?

- **Want clean atomic skills with the least effort:** `openhe/download.py` or
  `phuma/download.py` — they ship single-skill G1 clips ready for `csv_to_npz`,
  but the quality is low.
- **Want to carve your own skills out of LAFAN1's long takes:**
  `lafan1/download.py`, then `lafan1/interactive_crop.py` or
  `lafan1/manual_crop.py`.
- **Want robot-agnostic source to retarget yourself:** `amass/download.py`.

## Interactive cropping (`lafan1/interactive_crop.py`)

```sh
uv run python src/mjlab/tasks/tracking/scripts/datasets/lafan1/interactive_crop.py \
    --data-dir data/lafan1_g1 --output-dir data/lafan1_g1/bridging
```

Open the printed `http://localhost:8080` URL. Pick a clip, scrub the timeline,
press **Set start = playhead** / **Set end = playhead** at the ends of a clean
single-skill stretch, name it, and **Save clip**. The crop is written as a CSV
of the exact source rows (so it is lossless and drops straight into
`csv_to_npz`); tick *Also convert to NPZ* to run that conversion immediately.

## Licensing

These datasets are research-only and carry their own terms — most notably
LAFAN1 is **CC BY-NC-ND 4.0 (non-commercial)** and AMASS / PHUMA are
non-commercial research licenses. Cite the originating papers (LAFAN1: Harvey
et al., SIGGRAPH 2020; AMASS: Mahmood et al., ICCV 2019; PHUMA: Lim et al.,
arXiv:2510.26236) and the relevant retargeting release in any publication. See
each script's module docstring for the specifics.
