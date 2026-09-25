# Kinematic diffusion bridge

The bridge has two independent stages:

1. The planner inpaints a kinematic G1 trajectory between full boundary states
   A and B. It copies their poses and velocities exactly and may use the motion
   immediately before A and after B as boundary context.
2. A universal tracker realizes that trajectory under physics and is explicitly
   scored on matching B's root pose, joint pose, root velocities, and joint
   velocities at the requested tick.

The planner trains directly on retargeted LAFAN1 kinematics. The tracker trains
on physical rollouts of the LAFAN1 experts so its references remain kinematic
while its observations and endpoint errors come from the simulated robot.

## Package layout

```text
diffusion/
├── dataset/
│   └── motions.py       LAFAN1 loading, splits, encoding, and windows
├── planner/
│   ├── model.py         temporal denoiser
│   ├── process.py       masked diffusion process
│   ├── bridge.py        checkpoint loading and A to B generation
│   └── train.py         planner training entry point
├── execution/
│   ├── tracker.py       TextOp UniTracker observation and action adapter
│   ├── protomotions.py  G1 BONES deployment adapter and download helper
│   ├── learned_tracker.py learned tracker deployment adapter
│   └── runtime.py       plan execution and handoff gate
├── tracker/
│   ├── command.py       dynamic rollout windows and hybrid path reference
│   ├── actions.py       next-reference joint targets plus learned residual
│   ├── env_cfg.py       observations, physics randomization, rewards, metrics
│   └── evaluate.py      held-out exact-arrival benchmark
└── evaluation/
    ├── kinematic.py     held-out plan errors
    └── physics.py       UniTracker ceiling and simulated arrival errors
```

The interactive programs live with the other experiments:

- `tests/experiments/unitracker_viewer.py` shows the physical robot, reference
  ghost, root path, and live tracking errors on a LAFAN1 clip.
- `tests/experiments/diffusion_plan_viewer.py` shows the generated kinematic
  walk to bridge to kick sequence without bridge physics.

## Workflow

All commands run from the repository root.

### 1. Build the dynamic tracker corpus

The planner uses kinematic LAFAN1 directly. The learned tracker deliberately
does not: it trains on the physical trajectories produced by the existing
per-clip LAFAN1 expert policies. Collect or refresh that corpus with:

```sh
uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker
```

This writes `data/bridge/tracker.npz`. Windows are sampled only inside a single
contiguous rollout and are at most two seconds long.

### 2. Train the universal tracker

```sh
uv run train Mjlab-G1-Diffusion-Universal-Tracker \
  --env.scene.num-envs 4096
```

There is one training run with all mechanisms enabled. The actor receives four
physical state frames, three actions, the first five future reference frames
densely, five longer sparse frames, exact B error, phase, and time remaining.
At a reset the state history is the recorded physical continuation ending at A;
after that it rolls forward with the simulated robot. Future references continue
past B rather than freezing there, which makes a nonzero target velocity
consistent with the commanded motion. The actor outputs a residual around the
next reference joint pose. Rewards combine route tracking, a phase-ramped
endpoint objective, and exact deadline scoring.

This completes the tracker side of the boundary contract. Before training the
planner, its post-B target context must be extended to cover the tracker's
32-tick far lookahead; the current planner configuration still controls that
length independently.

Watch held-out windows with the moving blue reference and orange endpoint:

```sh
uv run play Mjlab-G1-Diffusion-Universal-Tracker \
  --checkpoint-file \
  logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_30000.pt \
  --viewer viser
```

Quantify exact arrival on held-out physical windows:

```sh
uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.evaluate \
  --checkpoint \
  logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_30000.pt \
  --sources "('dance1_subject1',)"
```

The report includes survival, strict arrival at the downstream handoff
tolerances, 2x/4x/8x pass rates, and per-channel p50/p90/p95/max errors.

### 3. Compare the off-the-shelf trackers

Download the released ONNX checkpoint if it is missing:

```sh
uv run python -m mjlab.tasks.unitracker.scripts.download_checkpoint
```

Inspect tracking on the same motion domain used by the planner:

```sh
uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.tests.experiments.unitracker_viewer
```

The solid robot should remain close to the reference ghost. The live error line
uses the same channels as bridge handoff. If this fails badly, fix or fine-tune
the tracker before spending compute on the diffusion model.

The released checkpoint uses the pelvis as its motion anchor. This differs from
the `torso_link` value in TextOp's public training configuration but agrees with
its deployment code and is required for stable tracking. The previous-action
observation must also start at zero after reset.

The orange curve is the complete pelvis path for the clip, not an error marker.
It is hidden by default because long clips wind over the same area. Pass
`--show-path` to display it.

Measure every tracking channel over a complete clip and save the time series:

```sh
uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.tests.experiments.unitracker_evaluate \
  --motion data/lafan1_g1/motions/walk1_subject1.npz
```

The table reports mean, percentiles, maximum, tolerance pass rate, and the worst
frame for each handoff channel. The NPZ is written to
`data/unitracker/evaluation/` for later tracker comparisons.

Evaluate the official ProtoMotions G1 BONES deployment tracker on the same
clip:

```sh
uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.execution.protomotions

uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.tests.experiments.protomotions_evaluate \
  --motion data/lafan1_g1/motions/dance1_subject1.npz
```

The adapter follows the checkpoint's published contract: torso orientation,
pelvis-local angular velocity, `xyzw` quaternions, future offsets 1, 2, 4, and
8, previous absolute PD-target feedback, 50 Hz control, and 1 kHz physics. Its
NPZ is written to `data/protomotions/evaluation/`.

### 4. Train the diffusion planner

```sh
uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.planner.train
```

Checkpoints are written under
`logs/rsl_rl/g1_kinematic_diffusion_bridge/<run>/`.

### 5. Test held-out kinematic generation

```sh
uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.evaluation.kinematic \
  --checkpoint logs/rsl_rl/g1_kinematic_diffusion_bridge/<run>/model_30000.pt
```

`exact_boundary_rate` must be `1.0`. The remaining values measure the generated
interior against the held-out demonstrated solution. They are useful regression
metrics, not a requirement that diffusion reproduce the only demonstrated path.

### 6. Inspect generated trajectories

```sh
uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.tests.experiments.diffusion_plan_viewer \
  --checkpoint logs/rsl_rl/g1_kinematic_diffusion_bridge/<run>/model_30000.pt
```

Use the Viser controls to change B, bridge duration, and the walk trigger, then
regenerate. Look for foot sliding, penetration, discontinuities, and implausible
root or joint motion.

### 7. Measure physical execution

```sh
uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.evaluation.physics \
  --checkpoint logs/rsl_rl/g1_kinematic_diffusion_bridge/<run>/model_30000.pt
```

The first table is UniTracker on untouched held-out LAFAN1 paths. This is the
tracker ceiling. The second table is UniTracker on diffusion plans. Do not blame
the planner when the first table already fails the terminal tolerances.

### 8. Run the complete transition

```sh
uv run python -m \
  mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick \
  --bridge diffusion \
  --bridge-checkpoint logs/rsl_rl/g1_kinematic_diffusion_bridge/<run>/model_30000.pt \
  --tracker-checkpoint \
  logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_30000.pt
```

The planner is an inbetweener, not a reachability oracle. An arbitrary A and B
can still be incompatible with the requested duration. Exact equality of the
constructed endpoint does not imply that the simulated robot reached it; the
physics evaluation and runtime capture gate keep those claims separate.
