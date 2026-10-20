# Absolute G1 skill scores

Run these commands from the repository root. Each setting averages exactly 50
seeded runs, including failures. Individual scorers save absolute scores;
the comparison runner below also computes nominal normalized scores.

## Run the complete comparison

```sh
uv run python -m mjlab.tasks.bridging.scoring.compare --output data/bridging/comparison
```

This runs, in order:

1. Nominal jump, kick and walk: 50 runs each
2. Instant switches (no-op) on all three pairs: 50 runs each, one request
3. Transition policies on all three pairs: 50 runs each, one request
4. The bridge on those three pairs: 50 runs for each of the 45 distance/duration settings
5. Best bridge parameter selection and normalization of all three methods against the matching nominal skill

The full comparison contains 7,200 scored trials. Source recording warmups are
additional simulation time. Distance and duration are bridge parameters only.
Nominal execution and both baselines have neither; transition policies have
learned stopping and their trained maximum duration.

## Where the incoming entry is

Each run gets one switch request. All methods of a pair share it, except the
walk2kick transition:

| Method | Request | Incoming entry |
| --- | --- | --- |
| No-op | timed walk, or first landing | on the robot at the switch |
| Bridge | timed walk, or first landing | entry sequence `d` ahead of the robot at the request, facing its heading, `T` seconds to reach it |
| Transition, walk2jump and jump2walk | timed walk, or first landing | on the robot when the transition hands over |
| Transition, walk2kick | first step within 0.4 m of the world fixed entry | fixed in the world, as in training |

- Timed walk: start at the reset pose, as in transition training, walk straight
  along its heading, switch after 3 s plus a seeded 0 to 1.25 s. The kick's ball
  is parked behind the robot during this walk
- On the robot: the recorded entry state goes on the robot's root position and
  heading. The jump or kick reference moves with it, the kick's ball moves to
  where that reference expects it. Walk has no reference
- walk2kick transition: the ball sits 3 m ahead of the reset pose and the kick
  reference is slid until its ball lies on the real one. The reference stays
  frozen on its entry frame while the walk and the transition run, the walk heads
  for the entry pose, the transition takes over within its training distance

Jump and kick start their clip at the entry frame once control reaches them.

The runner selects the newest transition checkpoint in each pair's training
directory and copies it into the output directory before evaluation. It also
resolves the planner and tracker once. To use specific models:

```sh
uv run python -m mjlab.tasks.bridging.scoring.compare --output data/bridging/comparison_pinned --walk2jump-checkpoint logs/rsl_rl/g1_walk2jump_transition/RUN/model.pt --walk2kick-checkpoint logs/rsl_rl/g1_walk2kick_transition/RUN/model.pt --jump2walk-checkpoint logs/rsl_rl/g1_jump2walk_transition/RUN/model.pt --bridge-checkpoint PATH --tracker-checkpoint PATH
```

Replace RUN and PATH with actual paths. All stages use the incoming skill,
reference entry, walking command and selector saved with the corresponding
transition checkpoint. An optional `--entry` must have been used in training.
All scores use the same failure penalties. All three methods replay the same
source recording for each pair.

`OUTPUT/results.json` contains every nominal report, both baselines,
every bridge grid setting, every trial, absolute measurements, failure adjusted
scores and normalized results. It is replaced atomically after every completed
50-run setting and on stage/status changes. A failure or interruption keeps
completed settings. An unfinished batch has no published score. Individual stage
JSON and CSV files are also saved. Existing output directories are rejected;
this command does not resume an interrupted comparison.

Normalization uses the failure adjusted means over all 50 trials:

- Error metrics: nominal mean / post-switch mean
- Ball speed: post-switch mean / nominal mean
- Each ratio is clipped to [0, 1]; walk's two metric scores are averaged equally

Zero nominal error scores 1 only if the post-switch error is also zero, otherwise
0. A zero nominal ball speed cannot be a useful denominator and stops
normalization with an error, retaining the raw report. A 1e-12 denominator floor
handles a zero post-switch error with positive nominal error.

`comparison/PAIR` holds, each with raw means, success rate and normalized score:

- `no_op` and `transition`: the single cell of each baseline
- `best_bridge`: the selected distance and duration
- `best_bridge_by_distance`: the best duration at each of the nine distances
- `bridge`: all 45 normalized bridge cells

Selection maximizes the normalized score, then the unclipped mean ratio, success
rate, shorter duration and shorter distance.

Best parameters are selected using these evaluation runs. Treat this as an
exploratory maximum, rather than an independent estimate on unseen trials.

## Action discontinuity

```sh
uv run python -m mjlab.tasks.bridging.scoring.discontinuity --comparison data/bridging/comparison --output data/bridging/discontinuity
```

Replays a finished comparison: same checkpoints (hash checked), seeds, entry and
`PAIR/source_REQUEST.pt` recordings, with the bridge at the comparison's selected
`(d, T)`. Add `--bridge grid` for all 45 settings, `--pair walk2kick` for one pair.

Every scoring trial records the joint position target applied at each control step,
`default + scale * action` after clipping, from the target held at the request to the
end of its measurement. Values are the change over one control step in radians, RMS
over the joints unless stated:

| Value | Step |
| --- | --- |
| `seam_rad` | largest change at a seam: the request (bridge, transition) or the handoff |
| `request_jump_rad` | first step after the request |
| `handoff_jump_rad` | first incoming step; the instant switch's only seam |
| `handoff_joint_jump_rad` | largest single joint change at the first incoming step |
| `peak_step_rad` | largest change from the request to the end of the trial |
| `mean_step_rad` | mean change over the same steps |

Statistics cover every trial that reached the step, failed or not. Ratios divide a
method's mean by the nominal mean of the same value. `score_difference` is the change
in the failure adjusted means against the comparison; near zero means the same trials.

`OUTPUT/results.json` holds every report and, under `discontinuity/PAIR`, `nominal`,
`no_op`, `transition`, every `bridge` cell, `best_bridge` and, for a grid,
`best_bridge_by_distance`. `OUTPUT/actions.csv` has one row per setting. Per trial
values are also in each report and in `trials.csv` as `action_*` columns, also for
plain scoring and comparison runs.

## Nominal skills

```sh
uv run python -m mjlab.tasks.bridging.scoring.nominal --skill walk
uv run python -m mjlab.tasks.bridging.scoring.nominal --skill jump
uv run python -m mjlab.tasks.bridging.scoring.nominal --skill kick
```

Nominal execution starts at the selected incoming reference entry. It has no
distance or duration parameters. Use `--entry` to choose another recorded entry.

## Skill pairs

```sh
uv run python -m mjlab.tasks.bridging.scoring.transitions.walk2jump
uv run python -m mjlab.tasks.bridging.scoring.transitions.walk2kick
uv run python -m mjlab.tasks.bridging.scoring.transitions.jump2walk
```

The default method is the diffusion bridge. Each command evaluates the 45
distance/duration combinations: 0.2 to 1.0 m in 0.1 m steps, and 0.4 to 1.2 s in
0.2 s steps. This is 2,250 trials per pair and reference entry.
For one setting, append `--distance 0.4 --duration 0.8`.
Pin models with `--bridge-checkpoint`, `--tracker-checkpoint`,
`--source-checkpoint` and `--target-checkpoint`.

For the learned transition policy:

```sh
uv run python -m mjlab.tasks.bridging.scoring.transitions.walk2jump --method transition --transition-checkpoint logs/rsl_rl/g1_walk2jump_transition/RUN/model.pt
```

For the instant-switch baseline:

```sh
uv run python -m mjlab.tasks.bridging.scoring.transitions.walk2jump --method no-op --transition-checkpoint logs/rsl_rl/g1_walk2jump_transition/RUN/model.pt
```

The no-op switches directly from the last source action to the incoming action,
without a transition action, pause, interpolation or reset at the switch. The
incoming policy reads the source's previous-action history. The incoming skill
starts on the robot wherever the request finds it. Transition time is zero. The optional
transition checkpoint pins the skill and scene settings; its actor is not used.
Raw results are saved under `PAIR/no-op`, with `termination: instant_switch`
and `transition_s: 0`.

The transition policy acts until its termination head stops it or the training
limit forces a handover. Neither baseline accepts `--distance` or `--duration`.
The network receives its usual observation, with no target state or bridge
duration input. Replace the module and checkpoint for the other pairs. The
walk2kick checkpoint must use the effective-kick success labels.

Only the walk2kick transition is asked at a distance, its fixed training distance
before the entry. `report.json` records it as `request_placement_m`, null for every
other method. The walk2jump and jump2walk transitions are asked at the shared
request, with no distance, and the incoming skill starts where they hand over.
Their observation holds the root position relative to the scene origin. The timed
walk starts at the reset pose, like training, so it keeps that position close to
where training asked.

Passing `--transition-checkpoint` with `--method bridge` also pins the source and
incoming skills, selector, entry and walking command to that training setup.
For matching nominal runs, pass the same `--selector`, `--entry`,
`--target-checkpoint` and `--walk-speed`. Reports contain resolved model paths
and hashes: compare those before comparing results. Training seeds are rejected
when they overlap the scoring seed range.

## Measurements and failures

| Incoming skill | Absolute measurement | Better |
| --- | --- | --- |
| Walk | Mean planar command velocity error, m/s | Lower |
| Walk | Mean yaw rate command error, rad/s | Lower |
| Jump | Mean body position tracking error, m | Lower |
| Kick | Peak forward ball speed after foot contact, m/s | Higher |

Jump uses the same root/heading aligned reference body positions as its tracking
task. Native tracking termination checks still detect excessive world position
or orientation error. Kick direction comes from its reference: there is no
requested ball speed command in this skill.

Measurement starts when control reaches the incoming skill. Walk is measured
for two seconds; jump and kick use the remaining reference clip. Each environment
hands over independently. A fall, native incoming termination, failed source,
missing handoff, jump without landing, kick without contact or kick at or below
0.5 m/s counts as failure. Contact during the transition also fails a kick trial.

Failed walking trials receive at least 2 m/s velocity error and 2 rad/s yaw error;
failed jumps receive at least 1 m tracking error. Larger observed errors remain
larger. Failed kicks score zero, even if a partial rollout moved the ball.
These penalties can be changed using `--failure-errors.velocity-error-m-s`,
`--failure-errors.yaw-rate-error-rad-s` and
`--failure-errors.body-position-error-m`.

Raw measurements remain available, including partial measurements before failure.
They are separate from the failure adjusted scores averaged over all 50 runs.
Missing raw measurements are blank/null, never silently treated as successful zero
error. Nonfinite simulation output aborts the report instead of publishing a mean.

## Reproducibility and output

Default seeds are 10000 through 10049. Each seed perturbs initial joint positions
by Gaussian noise with 0.01 rad standard deviation and velocities by 0.02 rad/s.
The simulation resets the whole batch with seed 10000, which prints
`Setting seed: 10000`. Each of its 50 environments then gets an independent
perturbation generator with its own seed. The scorer prints the full seed range;
individual seeds are also stored in each trial. These are 50 parallel trials,
not 50 sequential environment resets.
For walk2jump and walk2kick, the timed walk switches after 3 s plus a random 0 to
1.25 s drawn from that run's seed, about one gait cycle, so each run switches at a
different point of its stride. The walk2kick transition's approach starts back on
the line to the entry by 3 s of walking plus a seeded 0 to 1 m, for the same
reason. Jumping sources run through their first landing. Each request keeps its
robot state, action history and recent physical history. The robot is never
teleported after the request. Source failures remain in all averages. Methods
under the same output root that use the same request reuse
`PAIR/source_REQUEST.pt` (timed, approach or landing), so their request states are
identical. A mismatch in skills, seeds, scene settings or walking sampling is
rejected. Use a fresh output directory for this protocol. Each walking trial
records `source_walk_s`, the walking time up to its request, in JSON and CSV;
`report.json` also holds the 50 seeded offsets. This affects scoring only; training
and nominal skill initialization are unchanged.

Results go under `data/bridging/scoring/nominal/SKILL` or
`data/bridging/scoring/PAIR/METHOD`:

- `report.json`: settings, model hashes, all trials, physical states, means,
  sample standard deviations, raw measurement counts and failure reasons
- `trials.csv`: one row per trial, raw measurements and failure adjusted scores
- `averages.csv`: one row per setting, absolute means and success counts
- `source.pt`: the replayed source request, a copy of `PAIR/source_REQUEST.pt`

The report is saved after every completed setting. Existing output directories
are never overwritten; use `--output PATH` for another run. Individual pair
commands do not select a best setting or divide by nominal scores; the complete
comparison command performs those steps.
