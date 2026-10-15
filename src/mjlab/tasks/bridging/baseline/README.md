# Dedicated transition baseline

Train one transition policy for one ordered pair of frozen skills, then compare
it with the general bridge under the same initial conditions and deadline.
The supplied scene adapter supports G1 `walk2jump`, `walk2kick` and `jump2walk`.
Training, inference and evaluation live here; a named viewer entry point also
lives alongside the existing G1 transition scripts. Existing skills and bridge
implementations are reused without changes.

## Literature and architecture

The thesis explicitly selects **Lee et al., Composing Complex Skills by Learning
Transition Policies, ICLR 2019**, in chapter 3, “Transition and Setup Policies”,
and chapter 7, “Baselines”. See the [paper and project](https://youngwoon.github.io/transition/)
and the authors' [description of proximity reward induction](https://sites.google.com/view/transitions-iclr2019).

This is an adaptation to the thesis protocol, rather than a reproduction of the
original experiments:

* The source and destination policies stay frozen
* The actor, value network and proximity predictor are separate MLPs with hidden
  sizes 256 and 128 and ELU activations
* Inputs are the current root pose, root velocities, joint positions and
  velocities, and previous action. Root position is relative to the scene origin,
  so the actor can learn the fixed arena geometry. No target state, entry index,
  future motion, reference features, or deadline enters these networks
* The actor outputs joint position actions in the same convention as the skills
* Initial predictor examples come from actual destination executions, starting
  at nominal entries, perturbed entries, and direct source interruptions
* Successful transition endpoints receive proximity 1; earlier states of that
  transition receive `0.95 ** steps_remaining`. Failed attempts receive 0.
  The predictor uses balanced regression minibatches and retains bootstrap
  examples while adding recent on policy attempts
* PPO receives `D(next_state) - D(state)` plus a terminal measured success bonus
  of 1. Full finite episode returns use discount 1, preserving the telescoping
  proximity reward. The predictor stays fixed during each PPO collection/update
* The controller hands over at the requested deadline. There is no learned
  termination network or action blending

The original transition actor has two 32 unit tanh layers, a Gaussian action
output, and a two class softmax output for learned termination. Its separate
proximity predictor has two 96 unit ReLU layers. Lee et al. already train the
transition policies with PPO. They share one transition policy across all
predecessors of a destination skill, rather than specializing to an ordered pair.
See appendix B.4 and B.5 of the [paper](https://openreview.net/forum?id=rygrBhC5tQ).

The larger networks, terminal measured success bonus, per pair specialization,
and external deadline control are adaptation choices. The proximity approach comes from Lee et al.; this is
not the discriminator and Q network architecture of Byun and Perrault or the
target value reward of Tidd et al. The predictor is a training signal, not a
calibrated estimate of the probability that a skill will succeed.

By default, one actor trains across **all selected entries of its destination
skill**. Use `--entries "(0,)"` for a fixed entry ablation. Each batch samples one
entry, trigger distance, and duration. Source states come from executing the
source policy, not from the bridge's motion corpus. Small initial joint position
and velocity perturbations provide repeatable differences between seeds.

## Prerequisites

Run from the repository root with the repository's existing environment. A CUDA
GPU, trained source/destination policies, their motion assets and a selector
table are required. If necessary, build the table first:

```sh
uv run python -m mjlab.tasks.bridging.selector --robot g1
```

Policies default to the newest checkpoints in their existing experiment folders.
For a controlled experiment, pin `--scene.source-checkpoint` and
`--scene.target-checkpoint` when training. Their resolved paths and SHA256 hashes,
and the selector table's hash, are saved. Evaluation reloads those same skills
and rejects changed files or incompatible scene settings.

## Train

Use the standard mjlab launchers:

```sh
uv run train Mjlab-G1-Walk2Jump-Transition
uv run train Mjlab-G1-Walk2Kick-Transition
uv run train Mjlab-G1-Jump2Walk-Transition
```

These registered tasks use the existing proximity reward PPO trainer through a
custom runner, with the standard RSL logger, W&B, timestamped experiment folders,
saved configuration and numbered checkpoints. The PPO objective and actor
inputs are unchanged. The auxiliary proximity predictor still needs its own
bootstrap and updates; this is not an unmodified RSL PPO runner.

The defaults are now **1000 iterations**, 64 environments and a checkpoint every
25 iterations. For a shorter experiment:

```sh
uv run train Mjlab-G1-Walk2Jump-Transition --agent.max-iterations 100 --agent.run-name quick
```

Training writes to `logs/rsl_rl/g1_walk2jump_transition/<timestamp>/` and logs
to the configured W&B project, `mjlab` by default. Set `--agent.wandb-project`
and `--agent.run-name` as for other tasks. `--agent.logger tensorboard` works
without W&B. Algorithm and scene options live under `--agent.transition`, for
example `--agent.transition.warm-start-updates 2000` or
`--agent.transition.scene.source-checkpoint PATH`. Set parallelism with
`--env.scene.num-envs`.

Inspect these curves:

| Curve | Meaning |
| --- | --- |
| `Transition/success_rate` | Fraction of the training batch with valid downstream execution |
| `Train/mean_reward` | Transition return, averaged over recent batches |
| `Loss/ppo`, `Loss/proximity` | PPO objective and predictor fit; low loss alone does not establish successful transitions |
| `Transition/handoff_jump_rad` | Mean RMS joint command change at handoff |
| `Target/*` | Eight endpoint error channels |
| `Downstream/*` | Incoming tracking or walking goal RMS errors |
| `Perf/*` | Collection time, update time and throughput |

The console reports bootstrap progress, each iteration and an ETA after the
first update. Bootstrap cost is paid once and does not decrease with the
iteration limit. A shorter run is a time budget, not a convergence guarantee.
Use separate evaluation seeds for the final comparison.

Watch the newest checkpoint with the standard player:

```sh
uv run play Mjlab-G1-Walk2Jump-Transition --viewer viser
```

Or choose a checkpoint explicitly, including an older baseline checkpoint:

```sh
uv run play Mjlab-G1-Walk2Jump-Transition --checkpoint-file logs/baseline/walk2jump/seed_0/model.pt --viewer viser
```

Standard playback runs walk, transition and jump automatically using the saved
scene, trigger distances and durations. Its checkpoint selector can load newer
saves while training continues. The manual viewer in the next section remains
available for interruptions at arbitrary walking phases. `train --video` and
`play --video` use the standard recorders; `--eval-video` is not supported by
this scene bound controller. Training currently uses one GPU per run.
Checkpoints contain inference weights and provenance, not optimizer and replay
state, so `--agent.resume` is rejected explicitly.

The standalone trainer remains compatible:

```sh
uv run python -m mjlab.tasks.bridging.baseline.train --scene.pair walk2jump
uv run python -m mjlab.tasks.bridging.baseline.train --scene.pair walk2kick
uv run python -m mjlab.tasks.bridging.baseline.train --scene.pair jump2walk
```

Defaults: 64 parallel environments, 1000 PPO updates, distances 0.2/0.4/0.6 m,
durations 0.4/0.8/1.2 s, straight walking at 0.8 m/s. Bootstrap collection runs at
least two batches per entry. Walking after a jump must use the velocity recorded
with its selector entries; change `--scene.walk-speed` if your entries differ.

The default output for the first command is
`logs/baseline/walk2jump/seed_0/model.pt`, with `config.json` and `training.jsonl`
alongside it. Run multiple training seeds, for example `--seed 1000` and
`--seed 2000`, using distinct evaluation seeds. A new launch refuses to overwrite
an existing run. Checkpoints contain inference weights and provenance, not
optimizer/replay state for resuming interrupted training.

For a standalone run already in progress, inspect its live records in PowerShell:

```powershell
Get-Content logs/baseline/walk2jump/seed_0/training.jsonl -Tail 10 -Wait
```

The old process keeps its original iteration budget and has no W&B logger.
Changes to defaults apply to new runs. Its saved model can still be watched
with either player. New standalone records also include reward, timing,
downstream errors and handoff diagnostics.

For a short pipeline check, not a trained research baseline:

```sh
uv run python -m mjlab.tasks.bridging.baseline.train --entries "(0,)" --num-envs 2 --iterations 2 --bootstrap-batches 2 --proximity-updates 2 --ppo-epochs 1 --output logs/baseline/smoke/model.pt
```

Look for improving measured `success_rate` in the training log, then choose
hyperparameters on a separate validation seed set. Predictor loss alone is not
evidence of a useful transition. The launch fails if bootstrap collection finds
no successful or no failed destination execution, because the initiation
boundary would otherwise be unidentifiable.

If training never finds a successful transition, an optional initialization uses
only successful frozen destination executions collected during bootstrap:

```sh
uv run python -m mjlab.tasks.bridging.baseline.train --scene.pair jump2walk --entries "(0,)" --distances "(0.6,)" --warm-start-updates 2000 --output logs/baseline/jump2walk_initialized/model.pt
```

This fits the actor to the incoming skill's actions over those successful
trajectories before PPO starts. It uses no bridge demonstrations. The default
is zero initialization updates, preserving existing training behavior. Record
this choice when comparing trained baselines. Low predictor loss with zero
measured transition success is not a working baseline.

## Use

Watch the trained walk to jump transition with a manual switch button:

```sh
uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.walk2jump_transition
```

This loads `logs/baseline/walk2jump/seed_0/model.pt`; override `--checkpoint`
for another training run. Open the Viser address printed in the terminal. The
viewer starts paused: press **Play**, then **Start transition** at any walking
phase. The next control tick uses the transition actor without resetting or
teleporting the robot. The jump reference waits at the selected entry until
the transition budget expires, then the frozen jump policy takes control.
There is no learned termination signal. The target ghost and status display
show the requested entry and the active controller. Use the viewer's **Reset**
button to repeat with the same seed. Changing the incoming entry restarts the
trial on the next simulation tick; only trained entries are offered.

For the same automatic distance trigger as the bridge viewer:

```sh
uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.walk2jump_transition --automatic True --distance 0.4 --duration-s 0.8 --seed 10000
```

The controls accept manual interruptions at any walking phase, but the actor's
training still samples specific interruption distances. Requests outside that
distribution are robustness tests, not guaranteed successful transitions. The
target placement and walking command remain those used for training. A fair
numerical comparison replays the exact same source state and target to both
methods, as the evaluator below does; visually clicking at roughly the same
time does not establish matched conditions.

The generic player supports every baseline pair and defaults to automatic
triggering:

```sh
uv run python -m mjlab.tasks.bridging.baseline.play --checkpoint logs/baseline/walk2jump/seed_0/model.pt --distance 0.4 --duration-s 0.8
```

Use `--viewer none` to run a single headless trial. For another pair, add
`--scene.pair walk2kick` or `--scene.pair jump2walk`. Choose a trained destination
entry with `--scene.entry 1`. Any changed training scene settings must also be
supplied at playback/evaluation.

For another controller, `TransitionPolicy.load(path, device)` returns the actor
and its metadata. Call the actor on a batch consisting of the 13 root state
values, all joint positions, all joint velocities, and the last raw action, in
the same joint order as the skills. Quaternions are WXYZ; velocities are in world
coordinates; subtract the scene origin from root position. The actor returns
the deterministic mean action. Keep the actor's saved normalization and give
control to the destination skill when the external deadline expires. Reference
placement and destination phase remain the controller's responsibility.

## Evaluate numerically

This evaluates the transition policy, direct switch, and diffusion bridge on a
matched grid, including a nominal destination rollout for every seed/distance:

```sh
uv run python -m mjlab.tasks.bridging.baseline.evaluate --checkpoint logs/baseline/walk2jump/seed_0/model.pt --output data/baseline/walk2jump_entry0.json
```

Defaults use entry 0, 30 seeds (10000 through 10029), three distances and three
durations. For the thesis's original three seed grid:

```sh
uv run python -m mjlab.tasks.bridging.baseline.evaluate --checkpoint logs/baseline/walk2jump/seed_0/model.pt --seeds "(10000,10001,10002)" --output data/baseline/walk2jump_entry0_three_seeds.json
```

For publication results, pin `--bridge-checkpoint` and `--tracker-checkpoint`.
When omitted, the existing bridge loaders choose their newest checkpoints;
the evaluator records the resolved paths and SHA256 hashes. Embedded tracker
checkpoints are supported. To compare another bridge, use for example
`--methods "('transition','direct','mixed')"` and its checkpoint. Learned
`imitation` and `cvae` bridges are also supported. When supplying one explicit
bridge checkpoint, compare one bridge architecture per invocation.

Without a trained bridge checkpoint, test just the baseline and direct switch:

```sh
uv run python -m mjlab.tasks.bridging.baseline.evaluate --checkpoint logs/baseline/walk2jump/seed_0/model.pt --methods "('transition','direct')" --output data/baseline/walk2jump_direct.json
```

Repeat with `--scene.entry 1`, etc., and for each skill pair and independent
training seed. Keep the same evaluation seeds for every comparison. Training
and bootstrap seeds cannot overlap evaluation seeds. The entry must have been
included in the baseline training set.

JSON contains all trials, nominal references, per cell summaries, paired
differences, and empirical reachability. Before the switch request, every method
is initialized from the same captured robot state, actions, recent history and
object states. `request_state` and `requested_target` make this matching auditable.
A CSV with the same stem contains the
individual trials, including raw error vectors. Existing results are never
silently replaced. Channel order and physical floors are stored in the JSON.

## Scores and interpretation

After handover, run the destination for two seconds, or until the remaining
reference ends. A switch fails if the source times out, the robot falls during
the source/transition/destination execution, or any of the destination skill's
native failure conditions fires. Failure is sticky; recovering later does not
erase it. Tracking reference completion is not failure. Nonfinite simulation
states stop the experiment instead of producing misleading scores.

For jump and kick, record RMS errors for torso position/orientation, mean body
position/orientation, and joint position/velocity. Use the existing tracking
command's error definitions, including its vector norms for joint errors. For
walking, use RMS errors in forward/lateral velocity and yaw rate instead.

On successful executions, the thesis quality score is:

```text
Q = 1 / mean_c(max(error[c], floor[c]) / max(nominal[c], floor[c]))
```

Tracking floors are 0.01 m, 0.02 rad, 0.01 m, 0.02 rad, 0.05 rad, 0.1 rad/s.
Walking floors are 0.05 m/s, 0.05 m/s, 0.05 rad/s. `Q = 1` matches nominal
execution; `Q = 0.5` means twice the average normalized error. Q can exceed one.
Each nominal rollout uses the same seed as its paired switch; nominal failure
stops evaluation so that a broken reference cannot distort the denominator.

Read these quantities together:

| Output | Meaning |
| --- | --- |
| `quality_mean`, `quality_std` | Quality conditional on that method's successful executions |
| `overall` | Per method successful execution diagnostics; use `paired` for comparisons |
| `success_rate`, `success_rate_ci95` | Survival/tracking success and Wilson 95% interval |
| `endpoint_errors`, `endpoint_rate` | Eight physical handoff channels and fraction within twice the training tolerances |
| `duration_mean_s` | Actual time between request and handover |
| `action_jump_mean_rad` | RMS change in the applied joint position command, after action scaling, offsets and clipping |
| `action_jump_max_mean_rad` | Largest absolute command change, averaged over retained executions |
| `target_error_mean` | Mean of each of the eight physical target error channels |
| `downstream_rms_mean` | Mean of each downstream tracking or velocity goal error |
| `paired` | Both methods' three metrics on exactly the same successful matched cases, plus paired differences |
| `matched_successes`, `attempted_pairs`, `retained_fraction` | Which fraction of the attempted cases contributes to the conditional comparison |
| `reachable` | Tested cells in which every seed succeeds; an empirical set, not a guarantee |
| `reach` | Largest tested successful distance per method and duration, or null if none |

The comparison uses only cases where both methods terminate in time and the
incoming skill remains valid for the same evaluation horizon. Failed attempts
and missing bridge terminations stay in `trials`, but do not enter the three
conditional metrics. An empty matched set has null metrics, not zero quality.
This selection does not establish comparative reliability. The retained seeds
and counts are reported for every cell and for the overall paired comparison.

The three quantities are taken from the same selected pairs, not from separate
successful subsets for each method. A completed reference counts as valid; a
missing bridge termination, a fall or a native skill failure does not. Target
box membership is not used for selection. This is the protocol in thesis
section 6.2.5. Reports with `protocol_version: 2` measure command changes in
radians; older reports used raw action differences and should not be mixed with
these results.

Endpoint error is useful for
the bridge's target reaching task, but a transition policy can enter a different
successful state and should not be declared a failure for missing the target box.
Survival/tracking success for kicking is not a separate measurement of ball
contact or goal scoring.

## Scope of the experiment

The policy specializes to a pair and the fixed scene/entry distribution used
for training. It does not generalize to arbitrary moved targets merely because
the general bridge does. Training batches synchronize their interruption time
to the first robot; their physical source states differ through initial noise.
Numerical evaluation uses one environment per trial, so the distance trigger is
individual and its actual distance is recorded.

All methods use a hard deadline with no capture grace. The transition policy
uses the full budget; a bridge may hand over earlier. At the deadline its native
termination signal is queried without taking another physics step. If it still
does not terminate, no incoming execution is forced and that attempt cannot
qualify. Direct switching has zero transition duration, so its
results are repeated across duration cells for alignment of the comparison grid.

Testing unseen skill pairs is the general bridge's generalization experiment;
this specialized baseline must be retrained for each pair. To test the premise
about corpus coverage, report the comparison separately for covered and poorly
covered transitions using a split defined before evaluation. This package does
not claim to estimate dataset coverage from a rollout.

## Feasibility pilot

A small jump to walk pilot completed on 2026-10-15 with the first walk entry,
a 0.6 m trigger distance, durations 0.4/0.8 s and test seeds 20000/20001/20002.
All six pairs qualified. Their recorded source states match exactly. The
baseline actor was initialized with 2000 supervised updates on successful
frozen walk executions and then received one PPO update. This checks that the
study can run; it is not a converged baseline or a final comparison of methods.

The report and trial CSV are `data/baseline/study_jump2walk_matched.json` and
`data/baseline/study_jump2walk_matched.csv`. Its preserved baseline is
`logs/baseline/study_jump2walk/model.pt`. At a duration of 0.8 s:

| Conditional mean over three matched seeds | Dedicated policy | Diffusion bridge |
| --- | ---: | ---: |
| Downstream velocity goal quality Q | 0.688 | 0.762 |
| RMS joint command discontinuity, rad | 0.067 | 0.154 |
| Target root position error, m | 0.116 | 0.025 |

The report contains all eight target errors and all downstream error channels.
In this pilot, better target accuracy does not imply a smaller handoff command
change. No benchmark conclusion should be drawn from these three seeds or from
this briefly trained baseline. The checked walk to jump checkpoint had zero
successful executions in its three test runs and therefore supplies no matched
comparison yet.

To repeat the feasibility comparison with new test seeds and a new output name:

```sh
uv run python -m mjlab.tasks.bridging.baseline.evaluate --checkpoint logs/baseline/study_jump2walk/model.pt --scene.pair jump2walk --methods "('transition','diffusion')" --distances "(0.6,)" --durations "(0.4,0.8)" --seeds "(20010,20011,20012)" --output data/baseline/jump2walk_repeat.json
```

Run the focused checks with:

```sh
uv run pytest tests/test_transition_baseline.py
uv run ty check src/mjlab/tasks/bridging/baseline
uv run pyright src/mjlab/tasks/bridging/baseline
```
