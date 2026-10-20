# Kinematic diffusion bridge

The bridge has two stages:

1. The planner generates a kinematic trajectory from A to B.
2. The universal tracker executes that trajectory in MuJoCo.

Both stages use the selected robot's retargeted BABEL and original LAFAN data, plus
stitched pairs of those clips.
The planner uses contiguous kinematic windows. The tracker learns to follow those
windows and their recorded post-B continuation with physics randomization and
perturbations. Training selects `g1` or `t1`; motion metadata is checked against
that selection before use.

Inspect the held-out trajectories exactly as the tracker reads them:

```sh
uv run python -m mjlab.tasks.bridging.bridges.diffusion.dataset.view
```

Blue is the stored trajectory, red is its pre-correction height, and the two
colored lines connect the recorded root and sole samples without interpolation.

## Build the motion dataset

Both stages read `data/babel_retargeted/<robot>/` and `data/lafan_retargeted/<robot>/`.
One command selects the clips, retargets them, grounds them and marks the frames the
kinematic filter rejects:

```sh
uv run python -m mjlab.tasks.bridging.bridges.dataset.motion_capture.build --robot g1
```

What is taken and why is in `bridges/dataset/motion_capture/filters.py`. Each output
folder also holds `selection.jsonl`, the clips and intervals taken, and `qa.jsonl`.

The default `bridge` profile is robot specific. G1 keeps its broad motion coverage,
including fast gaits, lowered poses and short flights. T1 keeps upright, slower
motion: torso tilt at most about 20 degrees, root height at least 0.60 m above the
floor, horizontal speed at most 1 m/s, vertical speed 0.35 m/s, angular speed
2 rad/s, joint speed 6 rad/s, root acceleration 6 m/s² and joint acceleration
150 rad/s². Flights with both soles more than 3 cm above the floor for 0.04 s or
longer are rejected. These are conservative kinematic limits, not proof of
dynamic feasibility. Single support walking and turns remain eligible.

For an existing T1 corpus, refresh the masks without retargeting:

```sh
uv run python -m mjlab.tasks.bridging.bridges.dataset.motion_capture.build --robot t1 --filter-only True
uv run python -m mjlab.tasks.bridging.bridges.dataset.motion_graph.build --robot t1
```

This also refreshes existing T1 stitched clips, so their old masks cannot retain
difficult transitions. Both training stages reject every window overlapping a
rejected frame, including its history and continuation. Frames keep their original
indices; filtering never joins the motion across a gap. Rejection counts are
printed per criterion and may overlap. The `quality`, `locomotion` and `none`
profiles remain explicit overrides with their existing limits.

Retrain the T1 tracker and planner on these masks in new run directories. Existing
checkpoints do not change when the data is filtered. Evaluate the new tracker on
held-out T1 windows before testing the Sokoban handoffs. G1 data, filter limits,
training settings and checkpoints are unaffected by the T1 command.

Then join pairs of filtered clips at a look-alike frame, into `data/motion_graph/<robot>/`.
Both stages only cut windows from these that cross the seam:

```sh
uv run python -m mjlab.tasks.bridging.bridges.dataset.motion_graph.build --robot g1
```

The stitch builder applies the selected robot's bridge limits again after blending,
including velocities and accelerations introduced at the seam. Rebuild with
`--robot t1` to generate new transitions from the more restricted source motion;
this replaces that robot's previous stitched output.

T1 stitches default to 1 s before the seam so the planner's 50-frame history fits.
Older 0.4 s lead-ins cannot supply a seam-crossing planner window. After refreshing
an older T1 corpus, rebuild its stitches before planner training. G1 keeps its
existing 0.4 s default; `--before-s` remains an explicit override for either robot.

Inspect any of the three corpora, as the samplers draw them:

```sh
uv run python -m mjlab.tasks.bridging.bridges.dataset.view
```

## Package layout

```text
diffusion/
├── dataset/motions.py       BABEL loading and window sampling
├── planner/                 diffusion model, checkpoints, and pretraining
├── tracker/                 universal tracker task and evaluation
├── execution/               runtime planner and tracker adapters
├── evaluation/kinematic.py  held-out planner evaluation
└── train.py                 frozen-tracker planner improvement
```

## Train the tracker

```sh
uv run train Mjlab-G1-Diffusion-Universal-Tracker
uv run train Mjlab-T1-Diffusion-Universal-Tracker
```

Checkpoints are written to
`logs/rsl_rl/g1_diffusion_universal_tracker/<run>/`.

Evaluate a checkpoint on held-out BABEL clips:

```sh
uv run python -m mjlab.tasks.bridging.bridges.diffusion.tracker.evaluate --checkpoint logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_2999.pt
```

## Pretrain the planner

Inspect a saved T1 planner checkpoint as kinematic motion in Viser:

```sh
uv run python -m mjlab.tasks.bridging.bridges.diffusion.planner.view --robot t1 --checkpoint logs/rsl_rl/t1_diffusion_kinematic_planner/<run>/model_4000.pt
```

`--robot` selects the held-out motion corpus and display model and must match the
checkpoint. It defaults to `g1`. Custom `--motions` paths are still supported and
their robot metadata is checked. This viewer displays generated plans without
running the universal tracker or physics.

The planner predicts positions only: the root and joint step per frame, and the
root orientation. The steps are summed from A, and the gap left at B is spread
over the path, so B is hit exactly. Velocities are computed from the positions.

It is conditioned on 50 frames up to A (one gait cycle at 50 Hz) and 25 frames from
B on (one step). Every frame also carries foot channels: a contact flag per foot,
predicted between A and B, and the sole position and yaw, given on the boundary
frames only. Contacts and soles come from forward kinematics, computed on the
history and on B's continuation separately, exactly as at deployment. A loss keeps
a sole still where the planner says it is planted. Windows of 134 frames keep about
60% of the starts the old 4 and 1 frame context had. At deployment skill A must have
run for 50 frames and skill B must supply 25 consecutive target states.

Training stretches each demonstrated duration by 0.8 to 1.25. Every window is
expressed in the heading frame of A, so a global rotation or translation of a
window changes nothing. Half of the windows shift A and its history in xy by at
most 1 cm, with the shift vanishing smoothly at B. The shift moves only the root,
so it also slides the feet; keep it small. Half are mirrored left to right. The
ranges are exposed as command-line options.

```sh
uv run python -m mjlab.tasks.bridging.bridges.diffusion.planner.train --robot g1
uv run python -m mjlab.tasks.bridging.bridges.diffusion.planner.train --robot t1
```

Checkpoints are written to
`logs/rsl_rl/g1_diffusion_kinematic_planner/<run>/`.

Evaluate a checkpoint on held-out contiguous BABEL windows:

```sh
uv run python -m mjlab.tasks.bridging.bridges.diffusion.evaluation.kinematic --checkpoint logs/rsl_rl/g1_diffusion_kinematic_planner/<run>/model_30000.pt
```

Resume a tracker checkpoint with:

```powershell
uv run train Mjlab-G1-Diffusion-Universal-Tracker --agent.resume True --agent.load-run '^2026-09-28_22-50-18_overnight$' --agent.load-checkpoint '^model_2999\.pt$' --agent.max-iterations 3000 --agent.run-name post-b
```

## Improve the planner with a frozen tracker

Run planner improvement with:

```sh
uv run train Mjlab-G1-Diffusion-Planner-Improvement --agent.max-iterations 12 --agent.tracker-checkpoint logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_5999.pt --agent.planner-checkpoint logs/rsl_rl/g1_diffusion_kinematic_planner/<run>/model_30000.pt
```

Use `Mjlab-T1-Diffusion-Planner-Improvement` with T1 tracker and planner
checkpoints for the T1 pipeline.

The tracker checkpoint is required and remains frozen. The planner checkpoint is
optional; without it, the task first pretrains the planner for 30,000 updates.

No rollout dataset is needed. Each condition contains pre-A and post-B sequences
sampled from different contiguous BABEL clips. B is placed relative to A and the
duration is sampled using velocity and acceleration limits measured from BABEL.
The planner generates candidates, hard kinematic gates reject invalid paths, and
the frozen tracker executes every survivor.

Successful executions and stable misses relabelled to their achieved endpoint are
added to a bounded replay and used to continue planner training. Failed paths are
kept only as diagnostics. The tracker is never updated by this task.

Output is written to
`logs/rsl_rl/g1_diffusion_planner_improvement/<run>/`. `model_N.pt` means that N
planner-improvement cycles completed. One cycle evaluates a candidate batch and
runs 1,000 planner optimizer updates.

## Run walk to kick

Use a unified planner-improvement checkpoint:

```sh
uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.walk2kick --bridge diffusion --bridge-checkpoint logs/rsl_rl/g1_diffusion_planner_improvement/<run>/model_30.pt
```

The unified checkpoint contains the improved planner and its frozen tracker. To
test independently pretrained stages instead, pass both checkpoints:

```sh
uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.walk2kick --bridge diffusion --bridge-checkpoint logs/rsl_rl/g1_diffusion_kinematic_planner/<run>/model_30000.pt --tracker-checkpoint logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_2999.pt
```
