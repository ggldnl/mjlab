# Transition policies: Lee et al. (ICLR 2019)

This package ports the default **TP with dense proximity reward** architecture
and transition learning code from
[youngwoon/transition](https://github.com/youngwoon/transition), commit
`79ab8e412b21df47779c97b4bef796d5be4a7660`, to PyTorch and the existing G1 skills.
The upstream MIT notice is retained in `LICENSE.upstream`.

The earlier adapted baseline has been replaced. Its goal conditioning, sigmoid
predictor, larger networks, fixed duration handoff, measured success reward
bonus and actor warm start are removed. **Old checkpoints require retraining.**

## What the controller learns

Given the physical state and the identity of the next primitive, the transition
policy outputs a joint action and a stop/continue decision. It learns to enter
any state from which the frozen next primitive succeeds. There is no requested
endpoint in the policy or predictor inputs, no endpoint loss and no explicit
state selector. The proximity predictor supplies rewards during training; the
actor and its termination head control execution at inference.

The primitive sequence is supplied by the environment, corresponding to the
upstream `meta_oracle=True` default. This package does not learn a meta policy.
One destination controller is trained in each pair experiment. It is not
conditioned on predecessor identity; broader predecessor distributions can
reuse that destination controller architecture.

## Architecture and source mapping

| Local implementation | Upstream reference |
| --- | --- |
| `model.py`: action/termination trunk | `rl/transition_policy.py`, `_build`: two 32-unit tanh layers, shared by action and stop heads |
| `model.py`: action output | Gaussian mean head, learned state independent log standard deviation initialized to zero |
| `model.py`: termination output | Two-class categorical softmax head; sample in training, argmax at evaluation |
| `model.py`: value network | Separate two 32-unit tanh layers and scalar output |
| `model.py`: initialization | Per-output normalized Gaussian weights, scale 1 for hidden/value layers and 0.01 for action/termination heads |
| `model.py`: predictor | `rl/proximity_predictor.py`: two 96-unit ReLU layers, linear scalar output, Xavier initialization |
| `model.py`: statistics | `baselines/common/mpi_running_mean_std.py`: independent policy/predictor running sums, squared sums, epsilon 0.01, variance floor 0.01, normalized inputs clipped to [-5, 5] |
| `learning.py`: PPO | `rl/trainer.py`, `policy_loss_ppo` and `_update_one_trans_policy` |
| `learning.py`: replay/predictor | `rl/proximity_predictor.py`, `Replay`; `rl/trainer.py`, `_update_one_proximity_predictor_network` |
| `rollout.py`: action and handoff timing | `rl/rollouts.py`, `traj_segment_generator` |
| `model.py`: rewards, labels, GAE | `rl/rollouts.py` and `rl/trainer.py` |

Predictor training uses **raw linear outputs**, with the sum of the separate
success and failure mean squared losses. Outputs are clipped to [0, 1] only
when used for rewards. The predictor has no primitive one-hot input; each
predictor belongs to one destination primitive. The actor and critic append
the destination identity after normalizing physical observations.

## Exact transition timing and rewards

The implementation follows the supplied code's default behavior, including
its details that a simplified next-state potential formula would miss:

1. At each decision, evaluate proximity on the **current, pre-action state**
2. Sample the Gaussian action and categorical termination decision
3. If termination is chosen, record it and hand over **without applying that
   sampled transition action**
4. Otherwise apply the action and record current proximity minus previous
   proximity; previous proximity starts at zero for every transition
5. Add the final current proximity as a terminal bonus, as enabled by upstream
   `proximity_dense_diff_rew_final_bonus=True`

There is **no measured downstream success bonus in PPO rewards**. Measured
success labels predictor examples after executing the incoming frozen skill.
The predictor remains unchanged throughout collection.

The upstream maximum is 100 decision records. The last record is forced to
terminate, so at most 99 transition actions are applied. This is a maximum,
not a scheduled handover time. At 50 Hz, it permits at most 1.98 s of transition
motion. Simulator failure after an action also closes the trace.

Successful terminal states get label 1. The upstream label loop gives the
immediately preceding action state label 1 as well, then 0.95, 0.95 squared,
and so on backward. Failed traces receive zero. Success and failure have
separate buffers, each bounded to one million online states. Successful
primitive demonstrations remain in a permanent initiation buffer. Success
sampling splits equally between collected and online states when both exist.

## Optimization defaults

Defaults follow `rl/config.py`:

* PPO learning rate 1e-4, clip 0.2, Gaussian entropy coefficient 1e-3
* 10 PPO epochs, minibatches of 64, up to 256 minibatches per epoch
* Transition GAE with gamma 0.99 and lambda 0.98; value loss coefficient 1
* PPO probability ratio includes termination; action probability is ignored
  for terminal records. Categorical entropy is not added to the objective
* Policy statistics update before copying the PPO reference distribution
* Predictor learning rate 1e-4, 5 epochs, one success and one failure minibatch
  per step, gradient norm limit 10 for both optimizers
* Adam epsilon placement matches the authors' MpiAdam; missing minibatches
  advance Adam with zero gradients as in the reference
* 10,000 simulator steps per collection, rounded up to complete G1 batches

`--lr-decay True` follows the configuration default. The authors' Walker
`ours` launch script overrides it to False; use `--lr-decay False` for that
experiment setting. Bootstrap collection, iteration count and checkpoint/log
frequency are operational G1 settings rather than claims about paper results.

## G1 environment adaptation

The supported scenes are `walk2jump`, `walk2kick` and `jump2walk`. They load
existing frozen G1 checkpoints and reuse the existing robot action convention.
Physical observations contain root pose and world velocities, joint positions
and velocities, and the previous action. Root position is relative to the
scene origin. Quaternions are WXYZ. These are G1 observations, not a claim to
reproduce Walker2d/Jaco observation vectors or their acceleration channels.

The existing entry table initializes successful demonstration executions and
sets the incoming reference phase and scene geometry required by G1 tracking
skills. Its endpoint state is also retained for **evaluation diagnostics** and
for competing goal-driven bridges. It is never fed to the transition networks.
The incoming reference is held until handoff, then the unchanged frozen skill
executes it. This reference setup is part of the G1 environment; the transition
policy does not search the entry table or choose a reference phase.

Training keeps that reference, and the kick's ball, fixed in the world for the
whole episode. Scoring keeps that scene only for walk2kick, where the ball
decides where the kick must start. The walk2jump and jump2walk transitions are
scored with the incoming reference placed on the robot when they hand over
(`TransitionScene.place_incoming`), the same as an instant switch.

Training supports independent handoff times in a batch. Once an environment
hands over, its incoming reference progresses while the other environments
continue their transitions. Success is checked over the configured downstream
horizon using falls and the incoming skill's native termination conditions.
Walk and jump use two seconds or the remaining reference, whichever is shorter.
Kick uses the full remaining reference and also requires foot–ball contact after
handoff and peak forward ball speed greater than 0.5 m/s. Contact before handoff,
a missed ball, a weak or backward strike, or a native failure produces a failure
label. The same predicate labels bootstrap demonstrations and online transitions.
These binary outcomes train the proximity predictor; ball speed is not added to
the PPO reward. The threshold matches the scoring package's effective kick rule.

This is a source-based port of the default transition architecture and learner,
not a reproduction of the original full-task experiments. It uses PyTorch,
single-host vector simulation, G1 scenes and their success signals instead of
TensorFlow 1, MPI, Walker2d and Jaco. It implements the default softmax/LSGAN/dense
configuration, not the upstream optional sigmoid, WGAN, sparse reward or learned
meta-controller ablations. Initial random draws are not bitwise identical across
frameworks. These differences must be reported with experimental results.

## Train and play

Use the repository environment and existing G1 assets. The entry table can be
built with `uv run python -m mjlab.tasks.bridging.selector --robot g1`.

```sh
uv run train Mjlab-G1-Walk2Jump-Transition
uv run train Mjlab-G1-Walk2Kick-Transition
uv run train Mjlab-G1-Jump2Walk-Transition
uv run play Mjlab-G1-Walk2Jump-Transition --viewer viser
```

On Windows, run both walk2kick and jump2walk sequentially overnight:

```powershell
.\scripts\train_transitions_overnight.ps1
```

Defaults are 1000 iterations per task, 64 environments and seed 0. Override with
`-Iterations 1000 -NumEnvs 64 -Seed 0`. Training uses TensorBoard logging and the
usual task checkpoint directories under `logs/rsl_rl`. Console logs also go to
`logs/overnight`. The second job runs even if the first exits with an error.
Keep the terminal open and disable automatic sleep while training.

The standard player's Info panel shows the active walk, transition or jump
policy, the gap at the switch request, the current XY gap and the transition
time. The nominal jump ghost stays fixed in the world. Its reference is anchored
3 m ahead at reset by default; it is not moved when the switch is requested.
Walking requests the switch at the configured gap (typically 0.4 m). The ghost
is not a state input or a required endpoint of the transition policy.

Standard training defaults to 64 environments, 1000 updates and saves every
25 updates. The default keeps one incoming reference phase, `scene.entry`,
fixed, so the G1 tracker setup does not change behind the state-only actor.
The request distance is also fixed at 0.4 m throughout a run. Use
`--agent.transition.distance 0.6` with the standard launcher, or `--distance 0.6`
with the standalone trainer, to choose another fixed distance before training.
Initial state perturbations remain enabled. The distance is saved in the
checkpoint and used for automatic playback and scoring; it is not an actor input.
The simulator triggers at control steps, and batched training uses the first
robot's trigger time, so individual measured gaps can differ slightly.
Use separate runs to compare different phases. Explicit `--entries` can change
the sampled reference distribution, but does not add phase inputs to the actor.
Standalone training exposes the same learner:

```sh
uv run python -m mjlab.tasks.bridging.baseline.train --scene.pair walk2jump
```

Pin `--scene.source-checkpoint`, `--scene.target-checkpoint` and
`--scene.selector-path` for experiments. Prefix these with `--agent.transition`
when using the standard train launcher. Successful primitive trajectories are
collected first; `--initiation-start` and `--initiation-end` select the portion
retained as initiation examples. Their defaults are 0 and 1, matching the
upstream configuration. The original Walker script uses skill-specific ends,
including 0.4 for jumping; this requires validation for the G1 motions.

A short pipeline smoke run, not a trained baseline:

```sh
uv run python -m mjlab.tasks.bridging.baseline.train --entries "(0,)" --num-envs 2 --iterations 1 --bootstrap-batches 1 --num-rollouts 1 --num-trans-batches 1 --ppo-epochs 1 --proximity-epochs 1 --max-transition-steps 10 --output logs/baseline/faithful_smoke/model.pt
```

Standalone runs refuse to overwrite existing outputs. Checkpoints include
network/statistics state, primitive identity, source revision, configuration,
training seed range and skill/table hashes. They support inference, not optimizer
and replay resume. Old version 1 adapted checkpoints are rejected explicitly.
Previous walk2kick checkpoints trained with a distance distribution or kick
tracking-only labels must be retrained. Earlier version 2 walk2jump and jump2walk
checkpoints trained at multiple distances remain valid for playback and scoring.
Their automatic playback uses the middle training distance; the scorer retains
the nine-distance evaluation grid. No fixed-distance retraining is required for
those two pairs.

## Evaluate

```sh
uv run python -m mjlab.tasks.bridging.baseline.evaluate --checkpoint logs/baseline/walk2jump/seed_0/model.pt --methods "('transition','direct')" --output data/baseline/walk2jump.json
```

By default, evaluation uses a 2 s maximum and held-out seeds 10000–10029.
To compare diffusion, use `--methods "('transition','direct','diffusion')"`
and pin its planner and tracker checkpoints. Choose `--durations` within that
bridge's supported range, for example `"(0.4,0.8,1.2)"`. Shorter limits are
explicit evaluation interventions; the standard training limit remains 100
records. Training seeds, including all collected episodes, cannot overlap the
evaluation seeds.

Methods replay the same source request and scene. The transition chooses its
own endpoint and early handoff; the bridge receives the externally selected
endpoint. Consequently, **endpoint error is a diagnostic, not the transition
policy's training objective or success criterion**. Report that information
difference when comparing methods.

The evaluator retains raw JSON/CSV trials, downstream success rates and Wilson
intervals, physical endpoint errors, joint command discontinuities in radians,
actual transition duration and downstream RMS errors. Quality normalizes errors
against a paired nominal frozen-skill execution. Paired conditional metrics
retain only cases where both methods succeed and report the retained seeds;
interpret them alongside unconditional success rates. Protocol version 3 uses
learned termination and must not be pooled with earlier fixed-duration results.
