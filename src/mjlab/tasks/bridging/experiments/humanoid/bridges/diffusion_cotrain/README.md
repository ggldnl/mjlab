# Alternating planner and tracker co-training

This package reuses the diffusion planner and PPO tracker architectures without
changing `bridges/diffusion`. It removes the static generated-reference dataset
from this training path.

Build the physical tracker corpus first:

```sh
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker
```

If no compatible tracker checkpoint exists, pretrain one on physically recorded
corpus trajectories:

```sh
uv run train Mjlab-G1-CoTrain-Tracker-Pretrain
```

Then alternate tracker PPO updates and planner updates. Existing compatible
planner and tracker checkpoints can be used directly; a fresh pretrain is not
required:

```sh
uv run train Mjlab-G1-Diffusion-Tracker-CoTrain \
  --env.commands.path.planner-checkpoint \
  logs/rsl_rl/g1_kinematic_diffusion_bridge/<run>/model_50000.pt \
  --agent.tracker-checkpoint \
  logs/rsl_rl/<tracker_experiment>/<run>/model_<iteration>.pt
```

Recorded references remain the anchor distribution. Cross-trajectory endpoints
are drawn from distinct physical rollouts in `data/bridge/tracker.npz`, including
different windows from the same source policy. Their plans are generated online,
starting at zero probability and ramping to 50 percent. Tracker rollouts label a
small outcome critic. During each planner phase PPO is frozen, the outcome critic
is frozen after fitting, and its gradient updates only the last denoising step.
High-outcome generated paths provide a self-imitation anchor.

Each tracker checkpoint `model_N.pt` is accompanied by a compatible diffusion
checkpoint `planner_N.pt`. Both can be passed to the existing `walk2kick`
runtime explicitly.
