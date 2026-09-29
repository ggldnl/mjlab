# Kinematic diffusion bridge

The bridge has two stages:

1. The planner generates a kinematic trajectory from A to B.
2. The universal tracker executes that trajectory in MuJoCo.

Both stages use the G1 retargeted BABEL data under
`data/babel_retargeted/unitree_g1_locomotion_v1`. The planner uses contiguous kinematic
windows. The tracker learns to follow those windows and their recorded post-B
continuation with physics randomization and perturbations.

Inspect the held-out trajectories exactly as the tracker reads them:

```sh
uv run python -m mjlab.tasks.bridging.bridges.diffusion.dataset.view
```

Blue is the stored trajectory, red is its pre-correction height, and the two
colored lines connect the recorded root and sole samples without interpolation.

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
```

Checkpoints are written to
`logs/rsl_rl/g1_diffusion_universal_tracker/<run>/`.

Evaluate a checkpoint on held-out BABEL clips:

```sh
uv run python -m mjlab.tasks.bridging.bridges.diffusion.tracker.evaluate --checkpoint logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_2999.pt
```

## Pretrain the planner

Training rescales each demonstrated duration by 0.95 to 1.05 and scales its
velocities consistently. Half of the windows receive a bounded start-state
perturbation that smoothly vanishes at B, and half are mirrored left to right.
The ranges are exposed as command-line options.

```sh
uv run python -m mjlab.tasks.bridging.bridges.diffusion.planner.train
```

Checkpoints are written to
`logs/rsl_rl/g1_kinematic_diffusion_planner/<run>/`.

Evaluate a checkpoint on held-out contiguous BABEL windows:

```sh
uv run python -m mjlab.tasks.bridging.bridges.diffusion.evaluation.kinematic --checkpoint logs/rsl_rl/g1_kinematic_diffusion_planner/<run>/model_30000.pt
```

Resume a tracker checkpoint with:

```powershell
uv run train Mjlab-G1-Diffusion-Universal-Tracker --agent.resume True --agent.load-run '^2026-09-28_22-50-18_overnight$' --agent.load-checkpoint '^model_2999\.pt$' --agent.max-iterations 3000 --agent.run-name post-b
```

## Improve the planner with a frozen tracker

Run planner improvement with:

```sh
uv run train Mjlab-G1-Diffusion-Planner-Improvement --agent.max-iterations 12 --agent.tracker-checkpoint logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_5999.pt --agent.planner-checkpoint logs/rsl_rl/g1_kinematic_diffusion_planner/<run>/model_30000.pt
```

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
uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.walk2kick --bridge diffusion --bridge-checkpoint logs/rsl_rl/g1_kinematic_diffusion_planner/<run>/model_30000.pt --tracker-checkpoint logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_2999.pt
```
