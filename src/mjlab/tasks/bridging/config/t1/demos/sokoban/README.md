# T1 Sokoban

For the G1, run the same script with `--robot g1` and G1 checkpoints.

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

Record skill entry windows, then launch the puzzle in Viser:

```powershell
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.record --out data/sokoban/t1_entries.npz
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --level six_boxes --entries data/sokoban/t1_entries.npz --bridge diffusion --bridge-checkpoint logs/rsl_rl/t1_diffusion_kinematic_planner/sokoban/model_30000.pt --viewer viser
```

The recorder and demo select the newest local `t1_walk_natural` and `t1_push`
checkpoints. The demo selects the newest T1 universal tracker checkpoint.
To pin a run, supply `--walk-checkpoint`, `--push-checkpoint`, and, on the demo,
`--tracker-checkpoint`. Use the same skill checkpoints for recording and execution.
Open the local Viser URL printed at startup to watch the T1.

To test the policies before training the bridge, run the same map and plan with
`--bridge no-op`; no entry recording or diffusion checkpoints are needed.
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
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.record --out data/sokoban/t1_entries.npz
uv run python -m mjlab.tasks.bridging.config.t1.demos.sokoban.run --map puzzle.txt --plan plan.json --entries data/sokoban/t1_entries.npz --bridge diffusion --bridge-checkpoint planner.pt --tracker-checkpoint tracker.pt
```

The recorder uses trained walk and push checkpoints. It records stationary
walking after a short warmup and the initial push rollout. Increase `--frames`
if the planner uses a longer future window, and inspect the recordings before
using them as entry targets. The saved joint order and frequency are checked
by the demo. Controller tolerances and speeds can be adjusted through
`--controller.position-tolerance`, `--controller.heading-tolerance`, and the
other controller fields.

Entry NPZ files contain `walk` and `push` arrays with shape `(time, state)`.
Record consecutive states from successful T1 skill rollouts at the same
control frequency and joint order as the scene. State layout is root position
(3), root quaternion wxyz (4), world linear velocity (3), world angular
velocity (3), joint positions, joint velocities. Include at least the planner's
future window. Ground height is zero. Use a stationary walk entry and a push
entry at the center of the cell behind the box, before advancing into contact.
The controller rotates/translates these recordings to the required position;
it does not reset or teleport the robot during handoffs.

The walk checkpoint defaults to `Mjlab-T1-Walk-Natural`; change `--walk-task`
if needed. The shared scene preserves its actor observation contract in
`walk`, and the push contract in `push`. Only push reads the selected box.
Both policies must use identical joint actions and control frequency.

The controller verifies cell center and heading before pushing, checks the
entry again after bridging, and advances only when the box is aligned and
settled at the requested grid destination. Placement errors and timeouts stop
execution and are reported. It does not silently snap boxes to cells or replan.
The Viser view shows active skill, bridge phase, plan progress and the current
completion condition. Native and headless modes are also available.

Exactly 1 m boxes in exactly 1 m corridors have no geometric clearance. Start
with open puzzles while tuning lateral and orientation precision, then test
narrow corridors. Symbolic solvability alone does not establish physical
feasibility for a humanoid walking around a box.

