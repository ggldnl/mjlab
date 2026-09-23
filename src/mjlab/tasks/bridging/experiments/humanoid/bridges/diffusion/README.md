# Diffusion bridge

The model learns 50 Hz G1 state and joint action windows from the physical tracker
rollouts in `data/bridge/tracker.npz`. It receives four recent dynamic states, an
end dynamic state, and a duration in control ticks. It generates the whole path
in one diffusion sample. Root motion is expressed in the current yaw frame.

History states and the target state are visible to the denoiser during training
and sampling. Root poses and joint positions one tick after A and one tick before B are fixed from
their velocities. The returned state path copies its first and final states from
the inputs exactly. Other intermediate states and actions are model predictions; physical
feasibility and simulated arrival are not guaranteed by endpoint inpainting.
Both 5,000- and 30,000-update checkpoints tested on 2026-09-22 reached the
strict terminal tolerances in 0 of 64 paired held-out physics rollouts. See
[REPORT.md](REPORT.md)
before treating generated endpoints as a successful handoff.

Run:

```sh
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.train
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.sample --checkpoint logs/rsl_rl/g1_diffusion_bridge/<run>/model_30000.pt
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.evaluate --checkpoint logs/rsl_rl/g1_diffusion_bridge/<run>/model_30000.pt
```

`sample` draws A, B and their actual duration from a held-out rollout. It saves
`states`, `actions`, `demonstration`, and `fps` to
`data/bridge/diffusion_sample.npz`.
`evaluate` checks endpoint equality and compares terminal position and velocity
consistency against recorded paths. It does not run the simulator.

`runtime.py` registers this package as the `diffusion` bridge, so the walk to kick
viewer can drive it:

```sh
uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick --bridge diffusion
```

It samples one path when the bridge window opens and replays its actions until the
window closes. The walk to kick runner supplies four observed states. The window
duration is rounded to ticks and clamped into the checkpoint's trained range.
`--bridge-checkpoint` picks a run; the default is the newest one under
`logs/rsl_rl/g1_diffusion_bridge`.
For an inference ablation, `--diffusion-replan-interval 5` samples again every
five ticks while enough time remains. It is experimental and has not passed
the physical handoff test.

Use the batched physics check before judging a checkpoint by its pinned endpoints.
It puts generated and recorded actions from the same held-out start states into
parallel mjlab environments:

```sh
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.evaluate_physics --checkpoint logs/rsl_rl/g1_diffusion_bridge/<run>/model_30000.pt --batch 64
```

Add `--candidates 16 --batch 4` to test whether any of 16 samples for each
of four held-out goals survives the strict physics gate.

Record a walk to kick rollout and inspect simulated G1, the bridge plan, and the
subsequent kick reference in Viser:

```sh
uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick --bridge diffusion --viewer none --entry 0 --diagnostic-path data/bridge/walk2kick-entry0.npz
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.view_rollout --path data/bridge/walk2kick-entry0.npz
```

For an exact-entry control experiment, add `--exact-entry-baseline True` to the
walk to kick command. This places the simulated robot at the recorded B state,
restores B's previous action, and runs the kick policy; it is a diagnostic upper
bound, not a bridge.

For direct inference:

```python
from pathlib import Path
import torch
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.bridge import DiffusionBridge

bridge = DiffusionBridge.load(Path("model_30000.pt"), device="cuda:0")
# history: (batch, 4, 71), target: (batch, 71)
duration = torch.tensor([40], device="cuda:0")
path = bridge.generate(history, target, duration)
# path.states[i, :duration[i] + 1] and path.actions[i, :duration[i]] are valid
```

The data and model must use the same G1 joint order and 50 Hz control rate.
Duration must be between the trained `min_steps` and `max_steps` (15 to 60 by default).
The training duration is sampled from actual contiguous rollout windows between
`min_steps` and `max_steps` (15 to 60 by default).
