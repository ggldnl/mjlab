# T1 Sokoban

The T1 uses three skills: standard flat walking, conditioned turns in place,
and forward pushing. The controller requests only forward speed from walking;
direction changes use the turn policy. Train the predefined mjlab flat task:

```sh
uv run train Mjlab-Velocity-Flat-Booster-T1 --env.scene.num-envs 4096 --agent.run-name sokoban_flat --agent.logger tensorboard
```

Standard walks log under `logs/rsl_rl/t1_velocity`. This is the stock task,
including its rewards, command ranges and velocity curriculum, without the
LAFAN style reward. The natural variant remains available through the explicit
`--walk-task Mjlab-T1-Walk-Natural` override. Other stock T1 velocity tasks share
the `t1_velocity` experiment, so pin the flat checkpoint when recording and running.

The turn policy observes a scalar remaining angle in radians, plus the requested
angular speed. Training targets cover minus 180 to plus 180 degrees. Heading and
turn rate rewards train the rotation, foot stepping rewards encourage lifting
the feet, and slip penalties discourage pivoting on planted feet.
`TurnCommand.set_angle()` latches a relative angle; `set_heading()` latches a world
heading using the shortest rotation. The new heading API and controller fixes
keep the existing T1 turn observation and reward contracts.

Existing T1 turn and push checkpoints can be kept. The cube face observation fix
preserves the features used during forward push training, while selecting the
equivalent face for other grid directions. It does not require push retraining.
The diffusion planner and universal tracker formats are unchanged, so those
checkpoints can also be kept. If no trained turn exists yet, train it:

```sh
uv run train Mjlab-T1-Turn --agent.run-name sokoban_turn
```

After walk training, build the entry points with the selector, pinning the
checkpoints the demo will run. The demo reads `data/selector/t1/states.npz` and
refuses sets recorded with another task, or with another checkpoint when one is
pinned. Old `data/sokoban/*_entries.npz` files are no longer read.

```powershell
$walk = "logs/rsl_rl/t1_velocity/<flat-run>/model_29999.pt"
$turn = "logs/rsl_rl/t1_turn/<tested-run>/model_3000.pt"
$push = "logs/rsl_rl/t1_push/<tested-run>/model_1200.pt"
uv run python -m mjlab.tasks.bridging.selector --robot t1 --checkpoints walk $walk turn_left $turn turn_right $turn push $push
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --level example --walk-checkpoint $walk --turn-checkpoint $turn --push-checkpoint $push --bridge diffusion --viewer viser
```

Both commands choose the latest checkpoints in the registered experiments.
For repeatable runs, pin the same paths in both. The selector records 32 future
states after every entry point.

`--robot g1` selects `Mjlab-G1-Walk-Straight`, `Mjlab-G1-Turn` and `Mjlab-G1-Push`.
Only `--bridge no-op` works on the G1 for now: the selector's G1 table holds no
Sokoban sets. To bridge on the G1, train the skills and add `walk`, `turn_left`,
`turn_right` and `push` sets for those tasks to the G1 table in `selector/specs.py`.

```powershell
uv run train Mjlab-G1-Walk-Straight --agent.run-name sokoban_forward
uv run train Mjlab-G1-Turn --agent.run-name sokoban_turn
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --robot g1 --level example --bridge no-op --viewer viser
```

`--walk-task` and `--turn-task` override the defaults.

## Levels and plans

Drop a level into `levels/<name>.txt` using the ASCII convention below.
Select it with `--level <name>` (omit the extension). The default level is
`default`; `example` and `six_boxes` are also included. The demo loads the
matching `plans/<name>.json` automatically.

Generate or replace a plan with the existing solver:

```sh
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --level example --solver mjlab.tasks.bridging.config.t1.demos.sokoban.solver:solve --dry True
```

This requires Unified Planning and a compatible planner engine. Solver output
is checked against the board before saving to `plans/example.json`. Any solver
adapter passed through `--solver module:function` uses the same save convention.
Use `--dry True` to generate only the plan, then omit `--solver` to replay it.
Explicit `--map` and `--plan` paths remain supported; solver output for an
external map is saved in `plans/` using the map file's stem.

## Six box puzzle

`levels/six_boxes.txt` contains the supplied 22 by 11 puzzle. Its six `*` symbols
were converted to ordinary boxes (`$`), `X` to walls (`#`), and exterior
spaces to walls. Interior spaces and all six goals are preserved.
`plans/six_boxes.json` supplies a validated solution, so this puzzle does not
require an external planning engine. The small default smoke test remains available.

Run these commands from the repository root in PowerShell. First validate the
supplied plan without creating a simulation:

```powershell
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --level six_boxes --dry True
```

The diffusion bridge needs a T1 planner and a T1 universal tracker. Build the
T1 motion corpus and stitched transitions before training either stage:

```powershell
uv run python -m mjlab.tasks.bridging.bridges.dataset.motion_capture.build --robot t1
uv run python -m mjlab.tasks.bridging.bridges.dataset.motion_graph.build --robot t1
uv run train Mjlab-T1-Diffusion-Universal-Tracker --agent.run-name sokoban
uv run python -m mjlab.tasks.bridging.bridges.diffusion.planner.train --robot t1 --output logs/rsl_rl/t1_diffusion_kinematic_planner/sokoban
```

The corpus builder needs the local BABEL labels, AMASS CMU/KIT tarballs and
SMPL-X body model described in its module docstring. It downloads LAFAN if
missing and keeps existing retargeted clips. See the
[diffusion workflow](../../../../bridges/diffusion/README.md) for dataset inspection
and optional planner improvement with a frozen tracker.

Build the skill entry points, then launch the puzzle in Viser:

```powershell
uv run python -m mjlab.tasks.bridging.selector --robot t1
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --level six_boxes --bridge diffusion --bridge-checkpoint logs/rsl_rl/t1_diffusion_kinematic_planner/sokoban/model_30000.pt --viewer viser
```

The selector and demo select the newest local `t1_velocity`, `t1_turn` and `t1_push`
checkpoints. The demo selects the newest T1 universal tracker checkpoint.
To pin a run, supply `--walk-checkpoint`, `--turn-checkpoint`, `--push-checkpoint`, and, on the demo,
`--tracker-checkpoint`, and pin the same skill checkpoints on the selector.
Open the local Viser URL printed at startup to watch the T1.

To test the policies before training the bridge, run the same map and plan with
`--bridge no-op`; no entry points or diffusion checkpoints are needed.
The grid solution does not guarantee physical completion: this puzzle includes
1 m corridors around 1 m boxes, with no geometric clearance.

Train the push skill:

```sh
uv run train Mjlab-T1-Push --env.scene.num-envs 4096
uv run play Mjlab-T1-Push
uv run play Mjlab-T1-Push --env.commands.push.min-cells 3 --env.commands.push.max-cells 3
```

The robot starts at the center of the cell behind a 1 m cube, facing its rear
face, arms straight forward. The command latches a destination one to four cells
ahead and the policy reads what is left of it every step: remaining distance,
lane offset, the push axis in its heading frame and a reference box speed that
ramps up, holds 0.4 m/s and brakes onto the target. It accepts no twist.

The flat walking rewards stay on, driven by a hidden pace command derived from
that speed, so the robot walks steadily at the box speed and stands once the box
is in place. Both feet off the ground, contact other than the hands, the trunk
closer than an arm length to the box, and a box off lane, tipped or turned are
penalized. See `skills/push/__init__.py` for the details and the metrics to watch.
The same task runs on the G1 as `Mjlab-G1-Push`.

The push rest pose holds the arms forward while walk holds them down. The scene
keeps the walk pose and shifts the push joint observations, last action and
output action onto the push pose, so neither policy sees a change of offsets.

Validate the built in three cell plan without checkpoints:

```sh
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --dry True
```

`levels/example.txt` and `plans/example.json` in this package contain a small puzzle
with two pushes in different directions and walking around the box in between.
Use `--level example` for repeated skill handoffs.

Supply an ASCII map using standard Sokoban symbols: `#` wall, space floor,
`@` robot, `$` box, `.` goal, `*` box on goal, `+` robot on goal. Maps must be
rectangular and enclosed by walls. Each cell is exactly 1 m. World x follows
columns to the right, world y follows rows upward. Cell centers are at
`(column + 0.5, row + 0.5)`, counting rows from the bottom.

The solver is supplied externally. Its adapter takes a `Board` and returns
`Action` objects or dictionaries with `skill`, `direction`, and `cells`:

```json
[{"skill": "push", "direction": "E", "cells": 3}]
```

Directions are E, W, N, S. Walks may span multiple cells but are executed with
cell waypoints. Consecutive forward pushes of the same box are merged into one
variable distance push. The plan is checked against every crossed cell and
must put every box on a goal before simulation starts. Boxes keep their index
in `Board.boxes` throughout execution.

```sh
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --map puzzle.txt --plan plan.json --bridge no-op
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --map puzzle.txt --solver my_solver:solve --bridge no-op
```

The encoding from `axioms_usage.ipynb` is available as
`mjlab.tasks.bridging.config.t1.demos.sokoban.solver:solve`. It uses the explicit
move version of the notebook with zero cost moves and unit cost pushes. This
optional path requires `unified-planning` and a compatible planner engine
such as `up-symk` in the execution environment. They are not required for
exported plans or the built in demo. The notebook's experimental axiom fork
is not required. Export the notebook's printed plan to a `.txt` file and pass
it with `--plan` to use an existing solution. Both `move` / `push-box(x,y,z)`
and axiom `push-box(l,x,y,z)` formats are accepted. Implicit walks to pushing
faces are expanded into free cell paths. `loc-x-y` uses the notebook's
coordinates, with rows counted from the top.

The no-op mode is an immediate switching baseline. The learned bridge uses the
existing T1 diffusion planner and universal tracker:

```sh
uv run python -m mjlab.tasks.bridging.selector --robot t1
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --map puzzle.txt --plan plan.json --bridge diffusion --bridge-checkpoint planner.pt --tracker-checkpoint tracker.pt
```

For the bundled default level, build the entry points and run with the latest T1 checkpoints:

```sh
uv run python -m mjlab.tasks.bridging.selector --robot t1
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --bridge diffusion
```

Entry points are recorded skill states, not trained bridge checkpoints. The planner and
tracker must already be trained. Pin the skill checkpoints on both commands to keep
them fixed. The no-op baseline skips the
posture transition and can fall during a push even when each skill works on its own.

Cube orientation observations use the equivalent face nearest the push axis, so a
quarter turn of the push direction keeps the box features in the trained range.

The selector's `walk` set stands still after a short warmup, `turn_left` and
`turn_right` are the first steps of a quarter turn each way, and `push` is the
start of a one cell push, before the hands touch the box. The demo picks the turn
set from the sign of the turn it requests. Inspect them with
`python -m mjlab.tasks.bridging.selector.view --robot t1`. The control frequency
and state width are checked by the demo. Controller tolerances and speeds can be adjusted through
`--controller.position-tolerance`, `--controller.heading-tolerance`, and the
other controller fields.

Each entry point comes with the states that follow it, at least the planner's
future window. State layout is root position (3), root quaternion wxyz (4),
world linear velocity (3), world angular velocity (3), joint positions, joint
velocities, with the ground at height zero. The controller rotates and translates
them to the required position; it does not reset or teleport the robot during
handoffs.

The walk checkpoint defaults to `Mjlab-T1-Walk-Natural`; change `--walk-task`
if needed. The shared scene preserves its actor observation contract in
`walk`, the turn contract in `turn`, and the push contract in `push`.
Only push reads the selected box. All policies must use identical joint actions
and control frequency.

Walking stops and hands off to the turn skill when a direction change is needed.
At startup, a standing robot runs the turn policy directly if the first action
needs a different heading. There is no walk to turn bridge before any walking
has occurred. Later skill handoffs still use the selected bridge.
After turning settles, it hands back to straight walking. Walk never receives a
sideways or angular command. If heading drifts outside tolerance, it stops before
using the turn skill again. Outside the arrival tolerance, the controller turns
toward the measured waypoint bearing and walks forward. This corrects lateral
drift and overshoots without sideways or backward walking. Once it arrives, it
turns to the plan's cardinal heading before completing the waypoint. A turn
target stays latched while the robot rotates, even if position drift changes the
waypoint bearing. This is local correction within the planned cell route, not
obstacle replanning. Each waypoint requires position, heading and
low measured linear and angular speed for ``--controller.settle-seconds``.
Cell arrival defaults to 0.25 m position error and 0.20 rad heading error,
with 0.15 m/s settling speed and 0.2 seconds of settling.
The display includes robot position, target errors and speeds.
``--controller.position-gain`` controls the position correction strength. Increase
it if a walking checkpoint stands still at small commanded velocities near a cell
center; ``walk-speed`` caps the requested forward velocity.

Learned bridge handoffs require the observed skill entry tolerances. The planned
``--controller.bridge-seconds`` sets the generated path length, not the handoff
time. If the path ends before the robot reaches its entry, the tracker keeps
correcting toward the endpoint until ``--controller.bridge-timeout`` expires.
``--controller.action-timeout`` is a safety limit for each cell waypoint or push,
not a delay before issuing the next action.

The walk only needs to arrive near the pushing cell. The push entry is placed
relative to the physical box center and its nearest pushing face, one metre behind
it by default (`--controller.push-standoff`). The bridge corrects the remaining
position, heading and posture error. Push starts only when bridge feedback is ready
and measured box relative errors meet `--controller.entry-position-tolerance`
(0.08 m) and `--controller.entry-heading-tolerance` (0.10 rad).
Wider cell tolerances do not weaken these handoff checks. The no-op baseline cannot
perform this correction and fails if alignment is required.
The controller advances only when the box is aligned and
settled at the requested grid destination. Placement errors and timeouts stop
execution and are reported. It does not silently snap boxes to cells or replan.
The Viser view shows active skill, bridge phase, plan progress and the current
completion condition. A translucent cyan robot shows the bridge's full target
state, with the recorded root height, orientation and joint pose, as in the
soccer demo. Only the robot is duplicated; boxes and walls are hidden in the
ghost model. An orange sphere marks the entry on the floor and an orange arrow
shows its heading. The ghost and markers use the actual latched bridge target,
including the physical box offset for pushing. They remain visible during the
entering skill and after a failed handoff, and clear on reset or completion.
They are also shown in the native viewer. If hidden in
Viser, turn on Debug Viz > Enabled in the scene display settings.
Native and headless modes are also available.

Exactly 1 m boxes in exactly 1 m corridors have no geometric clearance. Start
with open puzzles while tuning lateral and orientation precision, then test
narrow corridors. Symbolic solvability alone does not establish physical
feasibility for a humanoid walking around a box.

