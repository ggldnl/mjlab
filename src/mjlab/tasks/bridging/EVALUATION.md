# Chapter 7 evaluation

## Standalone scoring scripts

The current standalone transition comparison is `scripts/score_bridge.py`.
It records one fixed bank of 50 runs per source skill. Each transition has
exactly 50 direct switches, with no entry point, distance or duration parameter.
The direct switch restores the recorded state without moving the robot towards
an entry. Tracking policies start at reference frame zero.
The tracking reference and kick ball are placed before execution, without
changing the recorded robot state.

Direct switches are scored against the same entry point nominals as the bridge,
averaged over the entry points of the incoming skill. Runs before
`quick_score_v3` used a nominal started from the skill's own initialization
instead, which inflates the direct walking score: walking from a stand has a
larger velocity error than walking from an entry point in the gait.

The bridge runs all 50 recordings for every entry point, distance and duration.
`averages.csv` contains one direct row per transition and one bridge row per
grid cell and entry. `best_entries.csv` keeps the highest mean successful score
entry for each distance and duration, with that same entry's success rate.
Across all three transitions the default experiment has 150 direct switches
and 5,400 bridge trials. Reusing the source bank does not add new source noise.

`scripts/score_demos.py` evaluates 50 full episodes per demo and method; it does
not sweep a transition grid. Its workers keep setup, stepping, resets and
cleanup inside inference mode. Resuming requires unchanged pinned code and
checkpoints, so use a new output directory after updating either script.

The module runner below is an older, separate evaluation path; its direct
grid protocol does not implement the standalone script's entry free baseline.

## Module runner

Run from the repository root with the trained checkpoints and selector tables present:

```powershell
uv run python -m mjlab.tasks.bridging.evaluate --output data/bridging/results/chapter7
```

The runner evaluates G1 walk to kick, walk to jump and jump to walk, then G1
soccer and the default T1 Sokoban puzzle. It exports the four measured tables to
the output directory and to `Masters-Thesis/thesis/tables/ch7_*.tex`.
Transition policy columns remain unmeasured. The filter ablation is separate.

Individual stages:

```powershell
uv run python -m mjlab.tasks.bridging.evaluate --suite transitions --output data/bridging/results/chapter7
uv run python -m mjlab.tasks.bridging.evaluate --suite demos --output data/bridging/results/chapter7
uv run python -m mjlab.tasks.bridging.evaluate --suite tables --output data/bridging/results/chapter7
```

Completed transition reports can be reused if their grid, seeds, selector and
checkpoint hashes still match. Existing incomplete reports and demo results are
not replaced. Use a new output directory for another measurement.

## Protocol

The default grid is 0.2, 0.4 and 0.6 m, with bridge durations of 0.4, 0.8 and
1.2 s, and 50 source seeds, 0 through 49. All selector entries are evaluated. Skills
use the checkpoints recorded in their selector table. G1 planner and tracker
checkpoints come from the soccer configuration; T1 planner and tracker paths
are resolved once and saved with the demo results.

Each seed adds joint position noise with standard deviation 0.01 rad and joint
velocity noise with standard deviation 0.02 rad/s to the source initialization.
Walking runs for at least one second and the planner history length. Jumping
runs until flight and landing have both been observed in every environment.
The actual source history and previous actions are recorded, then the recording
is rotated and translated to the requested distance before the fixed entry.
This is trial initialization; neither method teleports the robot during a switch.
Direct and bridge executions restore the same recording. The `.pt` request
files retain the source history, object states, target and previous actions.

Distances and source seeds run in independent parallel environments. The same
source seed uses the same initial perturbation across distances. Diffusion
sampling uses seed 0 for each batch, and the JSON records each trial's batch
index. A direct switch is measured once per entry, distance and source seed;
the bridge is also measured at each duration. The full transition experiment
has 1,800 direct trials, 5,400 bridge trials and 1,800 nominal executions.
The aggregate table includes all entries. Each distance and duration cell instead
uses the entry with the highest successful mean score, breaking ties by success
rate. Both numbers in the cell come from that entry's 50 trials.

The bridge must hand over at its deadline. Falling during the bridge fails the
trial. After handover, the incoming skill's own failure conditions apply.
Walking runs for two seconds; tracking skills run to the end of their reference.
The nominal execution starts in the recorded entry with its recorded previous
action. Incoming failure conditions apply to nominal executions too.

## Measures

* Walking: mean Euclidean error of the command vector `[vx, vy, yaw_rate]` in
  the robot frame
* Jumping: mean tracked body position distance, using the skill's reference
  aligned to the robot's horizontal position and heading, as in its training
* Kicking: peak horizontal ball speed after registered contact; a miss measures zero

The score is nominal error divided by trial error for walking and jumping,
and trial speed divided by nominal speed for kicking. Scores are not clipped.
Only successful trials enter the score mean. Success rates include every trial.
A zero denominator or failed nominal execution leaves a successful trial's
score undefined; a group containing an undefined successful score is shown as
`--`, as is a group without successes. No numerical floor is introduced.

## Demos

Soccer selects the highest mean score entry, distance and duration for each
transition from the measured grid. A score tie uses
success rate, then shorter duration. If a transition has no defined successful
score, its existing setting is retained and that fallback is recorded.
The evaluated configuration is saved as `soccer.yml`.

Each demo runs 50 seeds with each method. Each run resets the full scene and
controller, then adds the same small joint position and velocity perturbations
as the transitions. Direct and bridge runs use identical starts for each seed.
The four demo and method groups run in separate processes, reusing loaded
policies within each group. Success means completing the
controller's physical task: soccer must score a goal and finish kick recovery;
Sokoban must finish the supplied plan with all boxes settled on goals. Sokoban
keeps its existing visual walls without collisions. Bridge failures and falls
during a bridge or within two seconds after handover are attributed to that switch. Other failures,
including timeouts and missed task targets, count elsewhere or at an action.
This is a temporal attribution rule, not proof that a switch caused a fall.
Every result retains the initial scene and full robot state, phase changes, elapsed simulation time
and stop reason. Solving time is averaged only over completed Sokoban runs.

The JSON files contain every trial and resolved checkpoint hashes. These are
small seeded comparisons, not estimates with narrow statistical uncertainty.
