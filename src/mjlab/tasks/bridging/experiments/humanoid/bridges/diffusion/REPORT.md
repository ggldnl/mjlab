# Walk to kick diffusion bridge: diagnosis

## What is being tested

The current checkpoint samples a 40-tick (0.8 s) state and action sequence from
four observed G1 states and a specified terminal state B. The sampler overwrites
the first and final generated states with A and B. `DiffusionRuntime` then sends
the 40 generated joint actions to MuJoCo, and the kick policy takes over at the
deadline. The exact generated endpoint is a property of the array returned by
the sampler. It is not a guarantee that the simulated robot reaches B.

The existing imitation bridge's terminal limits are a useful strict test: root
position 0.05 m, root orientation 0.05 rad, root linear velocity 0.15 m/s, root
angular velocity 0.30 rad/s, lower joint position 0.08 rad, lower joint velocity
0.80 rad/s, upper joint position 0.05 rad, and upper joint velocity 0.75 rad/s.
The kick tracking metrics must also stay near the exact-entry baseline over at
least the first second. Survival alone does not meet that criterion.
The accepted post-handoff gate is a baseline envelope rather than fixed
absolute tracking limits. It should be estimated from repeated exact-B
rollouts for each entry (for example using a high percentile of each metric),
then applied to the bridge rollouts at both 0.5 and 1.0 s. The single baseline
runs below establish scale but are not enough to set a statistical envelope.

## Reproduced walk to kick results

These are one run per selected kick entry using the 5,000-update checkpoint
`2026-09-22_12-36-20/model_5000.pt`, 50 Hz control, 0.8 s bridge, and the same
walk trigger (0.49 m from target). The bridge was supplied **observed** history
rather than synthetic rewound states. GPU rollouts can vary; these are diagnostic
examples, not success-rate estimates.

| Entry | B foot contact (L,R) | Diffusion terminal root position / orientation error | Strict terminal pass | First kick action jump RMS |
|---|---|---:|---|---:|
| 0 | 1,1 | 0.117 m / 0.518 rad | No | 0.768 |
| 1 | 1,1 | 0.235 m / 0.287 rad | No | 0.926 |
| 2 | 1,0 | 0.387 m / 0.391 rad | No | 1.242 |
| 3 | 1,0 | 0.247 m / 0.391 rad | No | 1.155 |

The following values are averages over the first 0.5 s of kick tracking.
`anchor` and `body` are position errors; `joint` is position error. Velocity
error is included because a policy can briefly recover a pose while moving far
from the intended trajectory.

| Entry | Diffusion: anchor / body / joint / joint velocity | Exact-B baseline: same metrics | Diffusion fell by 1 s? |
|---|---|---|---|
| 0 | 0.119 m / 0.063 m / 0.773 rad / 10.533 rad/s | 0.017 / 0.037 / 0.332 / 2.198 | No |
| 1 | 0.267 m / 0.109 m / 1.018 rad / 18.486 rad/s | 0.027 / 0.042 / 0.349 / 3.058 | Yes |
| 2 | 0.618 m / 0.389 m / 1.479 rad / 23.515 rad/s | 0.021 / 0.044 / 0.322 / 3.262 | Yes |
| 3 | 0.473 m / 0.439 m / 1.822 rad / 31.657 rad/s | 0.026 / 0.049 / 0.344 / 4.554 | Yes |

For the baseline I teleported the robot to the **recorded dynamic state B**,
restored B's recorded previous action, and then ran the same kick policy and
reference. All four entries remained upright for at least 1 s. This is an
upper-bound diagnostic because it bypasses the transition and contact history,
but it establishes that the selected entry states and kick policy can track.
The first kick action jump in this baseline was 0.074, 0.110, 0.117, and
0.193 RMS for entries 0–3, respectively.

I repeated entries 0–3 with the complete 30,000-update checkpoint, one run per
entry. None passed the strict terminal test. Root position/orientation errors
were respectively 0.156 m / 0.483 rad, 0.254 m / 0.558 rad, 0.284 m /
0.466 rad, and 0.296 m / 0.286 rad. Entry 0 fell by 1 s of kick tracking;
entries 1–3 had fallen by 0.5 s. The first kick action jumps were 0.976,
1.152, 1.525, and 1.150 RMS. This checkpoint improves same-clip averages
while still failing the cross-skill handoff at every tested entry.

The actual generated path and simulated bridge trajectory separate early. At
entry 0, root position error grows from 0 at A to 0.030 m after 10 ticks,
0.060 m after 20, and 0.117 m at B. Joint position RMS at B is 0.210 rad.
Across entries 0–3, the generated paths' finite-difference root positions
disagree with their predicted linear velocities by about 0.66–0.70 m/s on
average. Joint finite differences disagree with predicted joint velocities by
about 0.65–0.71 rad/s RMS. The last executed generated action differs from B's
recorded previous action by 0.62–0.72 RMS. Pinning endpoint poses and adjacent
poses removes endpoint finite-difference jumps, but does not make the interior
trajectory dynamically coherent or the actions capable of executing it.

## Batched physics isolation test

`evaluate_physics.py` samples held-out windows from the tracker corpus, creates
two mjlab environments per window, restores exactly the same state and previous
action at A, then replays either generated actions or the original recorded
actions for 40 control ticks. The sampler uses 50 DDIM steps and the experiment
uses seed 0. This uses 128 parallel environments for a batch of 64 paired
windows. In the 5,000-update checkpoint test:

| Replay | Strict terminal pass | Mean root position | Mean root orientation | Mean root speed error |
|---|---:|---:|---:|---:|
| Diffusion actions | 0/64 | 0.183 m | 0.455 rad | 0.758 m/s |
| Recorded actions | 64/64 | <0.001 m | <0.001 rad | <0.001 m/s |

The paired recorded replay is nearly exact, so the environment state restore,
action indexing, recorded data, and simulation settings are sufficient to
reproduce a feasible transition. The dominant failure is the sampled action
sequence's physical execution. The held-out test uses A and B from the same
source rollout, a *much easier* condition than the cross-skill walk-to-kick
handoff. A better walk-to-kick score cannot be expected until the held-out
physics test improves substantially.

With the same 64-window physics gate, longer offline training produced:

| Updates | Denoising loss near checkpoint | Strict physical arrival | Mean root position / orientation error |
|---:|---:|---:|---:|
| 5,000 | 0.0987 | 0/64 | 0.183 m / 0.455 rad |
| 10,000 | 0.0782 | 0/64 | 0.148 m / 0.400 rad |
| 15,000 | 0.0673 | 0/64 | 0.137 m / 0.354 rad |
| 30,000 | 0.0511 | 0/64 | 0.114 m / 0.322 rad |

The default 30,000 updates improve averages but do not solve strict physical
arrival. Lower denoising loss is not a sufficient checkpoint criterion.
In one walk-to-kick entry-0 run with the 15,000-update checkpoint, the robot
missed B by 0.238 m and 0.604 rad and fell during the first 0.5 s of kick.
This single stochastic rollout does not establish a regression rate, but it
shows that improving same-clip replay does not automatically improve the
cross-skill transition.

As a small test of multimodal sampling, I drew 16 independent generated plans
for each of four held-out A/B windows and rolled all 64 out in parallel.
Best-of-16 had 0/4 windows with a passing candidate. The recorded replay passed
3/4 in that particular 68-environment run; one window had excess joint velocity
after state-only reset. This is too small to establish a
general best-of-N rate, but it rules out an easy win on those four examples.

## Pipeline review

### Data

The training set contains 123,448 states from eight physically executed G1
tracker sources based on LAFAN1. The evaluation set contains 17,617 states.
Window boundaries are cut from contiguous, feasible rollouts, so the recorded
action replay succeeds. The corpus does not include the kick entry trajectory
or goal-conditioned walk-to-kick transitions. B at entries 2 and 3 has one foot
in the air; contact mode and the next kick reference frames are not supplied to
the diffusion model. The target entry's recorded previous action is available
in the selector, but the present model does not condition on it. More kinematic
clips alone would not address errors between generated states and actions.

As a coverage check, I encoded each state in its own yaw frame using root
height/orientation/velocity and joint position/velocity, standardized each
channel by the training corpus, and measured nearest training-state distance.
The four kick entries have distances 5.33, 6.15, 5.57, and 6.55. Against 256
random held-out tracker states, these lie at roughly the 93rd, 95th, 94th, and
96th percentile of nearest-neighbor distance. This is only a pose-space
proxy, not a feasibility metric, but it confirms that B is near the sparse
edge of the current corpus even before conditioning on the transition or kick
continuation.

The high-value new data are **closed-loop rollouts of this bridge** and recovery
from its visited states, including failed approaches and the first second of
the entering policy. Keep a paired record of current state, previous action,
contacts, requested B, remaining time, plan, executed action, and kick tracking
errors. This targets the actual distribution shift shown above.

### Architecture and training

The present model independently denoises root pose, root velocity, joint pose,
joint velocity, and joint actions. Its loss is reconstruction error on the
recorded tensor. No simulator or dynamics consistency term validates that
sampled actions produce sampled states. Only positions immediately beside A
and B are integrated from their boundary velocities. The interior ticks
can violate kinematics and contact dynamics. A 5,000-update checkpoint is also
an early training point; the default training command requests 30,000 updates.
Longer training must be judged by *physical rollout*, not denoising loss or
endpoint equality.

The model is **not** BeyondMimic's controller. BeyondMimic trains a
proprioception-conditioned VAE decoder with DAgger to imitate motion-tracking
policies, then diffuses state and latent trajectories and decodes each current
action using up-to-date observations. It replans over a short horizon. Their
paper explicitly identifies irregular raw action targets and inference latency
as reasons for latent actions, and says precise objectives and motion starts
and ends remain difficult. The direct action diffuser here omits those two
feedback stages. [BeyondMimic](https://arxiv.org/html/2508.08241).
This differs from asking a CVAE student to infer B directly from teachers
that never saw B. In BeyondMimic the diffusion planner selects future latent
intent; the DAgger-trained decoder maps that intent plus current proprioception
to the next action. A weak tracking teacher remains a real bottleneck, so
replicating that architecture here still requires a teacher/decoder accuracy
gate on the motions and perturbed states that matter for handoff.

Diffuser supports conditioning trajectories by inpainting, but it is a planner
and uses replanning and guidance; fixing a goal variable does not enforce
humanoid physics. [Diffuser](https://proceedings.mlr.press/v162/janner22a.html).
Diffusion Policy executes only a short part of each sampled action sequence
before observing again and replanning. This bridge plays all 40 actions from
one plan. [Diffusion Policy](https://robots-that-learn.github.io/resources/diffusion_policy_2023.pdf).
PDP uses diffusion as a *physics-based policy* with perturbed-state recovery,
rather than relying on a single offline kinematic sample.
[PDP](https://zhaomingxie.github.io/projects/PDP/PDP.pdf).

I added an experimental `--diffusion-replan-interval 5` option to isolate the
effect of replanning without changing training. In one entry-0 trial with the
5,000-update checkpoint, it still missed B by 0.214 m and 0.722 rad. This
is not a replacement for a feedback controller: every new plan is still
executed by actions that may not produce its predicted states, and independent
samples can introduce action discontinuities.

### Inference and handoff wiring

The walk-to-kick call previously supplied one state to `DiffusionRuntime`, which
invented four history states by constant-velocity rewind. Training saw true
observed history. That mismatch is fixed in the transition runner; runtime
now requires the checkpoint's observed history length. The bridge also supports
per-environment timers and partial reset, allowing batched use with different
window starts. Neither change is sufficient to make the 5,000-update model
physically accurate.

The handoff at tick 40 is indexed correctly: action 0 leads out of A, action
39 is the final bridge action, and on the next call the kick policy supplies
the next action. The failure is already present before that call. The first
kick action discontinuity then amplifies it. The current bridge has no capture
check or corrective controller; at deadline it hands off even if all eight
terminal channels miss tolerance. Holding the kick reference at entry during
the bridge and letting it advance after handoff is consistent with the test's
intended timing.

## What to change to reach a usable handoff

1. **Keep the physics gate.** Evaluate each checkpoint on at least dozens of
   held-out paired windows in vectorized mjlab. Require substantial strict
   terminal success before running walk-to-kick. The present 0/64 fails this
   gate. Repeat at multiple durations and seeds. Do not select on exact
   generated endpoints: they are assigned by construction.
2. **Condition on the handoff contract.** Include A's previous action,
   B's previous action, foot
   contact and foot pose/velocity, and several post-B kick reference frames or
   phase. Record the corresponding fields in bridge training. This prevents
   two state-identical but contact/action-incompatible completions from being
   treated as interchangeable. Action continuity should be measured at both
   boundaries.
3. **Give execution feedback.** The most direct architecture is a diffusion
   planner for a short reference/latent trajectory plus a G1 controller that
   receives the *actual* state, last action, reference, and remaining B error
   every 20 ms. Train the controller on generated plans and its own perturbed
   rollouts with mjlab's parallel environments. Replan the diffusion trajectory
   on a short receding horizon, keeping the fixed requested deadline. The
   controller should optimize terminal dynamic-state error and subsequent kick
   tracking, not just pose imitation. A state/action diffusion model could also
   be trained with a dynamics-aware objective, but endpoint pinning alone is
   insufficient.
4. **Use a staged acceptance test.** First compare generated vs recorded
   replay on held-out same-clip windows. Then test cross-clip goals within the
   tracker corpus. Finally test entries 0–3 from walking approaches. At each
   entry, compare terminal error and the first 0.5/1.0 s kick metrics against
   an exact-entry baseline over multiple trials, and count falls. A kick policy
   that recovers from a large deviation must still be counted as a failed
   handoff.

### Can this use the standard mjlab task structure?

Yes. Offline denoising is supervised minibatch training over an existing
dataset; it already batches on the GPU and does not benefit from stepping
MuJoCo environments merely to read windows. A standard registered mjlab task
becomes useful for **closed-loop controller training, data aggregation, and
evaluation**. The task would reset a batch of robots to sampled A states,
select B and a deadline, run the diffusion sampler to produce a reference,
and expose actual/reference errors, contacts, previous action, and remaining
time to a policy. A custom runner could alternate diffusion updates with
vectorized on-policy rollouts; PPO can train the controller, but a standard
PPO actor cannot by itself train a multi-step denoiser. The added
`evaluate_physics.py` uses the existing task configuration to run 128
environments now. This is the appropriate parallelization boundary for the
current codebase.

## Reproduce and inspect

```sh
uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.evaluate_physics \
  --checkpoint logs/rsl_rl/g1_diffusion_bridge/2026-09-22_12-36-20/model_5000.pt \
  --batch 64 --duration 40

uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick \
  --bridge diffusion --viewer none --entry 0 --steps 450 \
  --bridge-checkpoint logs/rsl_rl/g1_diffusion_bridge/2026-09-22_12-36-20/model_5000.pt \
  --diagnostic-path data/bridge/walk2kick-entry0.npz

uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.view_rollout \
  --path data/bridge/walk2kick-entry0.npz
```

The Viser slider overlays blue simulated G1 with orange planned G1 during the
bridge and orange kick reference after handoff. The blue and orange root paths
and target frame show where execution starts diverging. The readout shows
per-tick root/joint errors during the bridge and kick metrics afterward.
Run walk-to-kick again with `--exact-entry-baseline True` and a different
diagnostic file for a control experiment. The baseline option changes the
physical initial condition at handoff, so it is for diagnosis only.
