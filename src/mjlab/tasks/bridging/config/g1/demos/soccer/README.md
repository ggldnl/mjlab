# G1 soccer: jump and shoot

One physical robot approaches a visual fallen G1, jumps over it, resumes locomotion,
then kicks a stationary ball towards the goal. The controller executes
`walk/run -> bridge -> jump -> bridge -> walk/run -> bridge -> kick` without
resetting the robot at handoff. Walk, jump and kick each read their own observation
group in the shared scene. Faster walk commands use the existing velocity policy;
they do not guarantee a running gait.

Run these commands from the repository root:

```powershell
uv run -m mjlab.tasks.bridging.selector --robot g1 --sets "('walk', 'jump', 'kick')"
uv run -m mjlab.tasks.bridging.config.g1.demos.soccer.run
```

To view the field and run the skills with immediate handoffs, use:

```powershell
uv run -m mjlab.tasks.bridging.config.g1.demos.soccer.run --bridge no-op --viewer viser
```

Open the local Viser URL printed at startup. Use `--viewer native` for a desktop
MuJoCo window. The camera uses a closer view from the side, centered on the path
from the robot spawn to the goal. Episode recordings use this same configured
camera rather than the browser's current camera position.
No-op mode skips the diffusion planner and tracker entirely, but
still needs the walk, jump and kick checkpoints and the selector entry points.
It is a visual baseline; immediate handoffs can fail physically.

The first command builds the walk, jump and kick entry points in
`policies.selector_path`. The walk set is recorded walking at 0.8 m/s, so it must
match `speed_after_jump`; the demo refuses it otherwise. Run it again when changing
a skill checkpoint, and pin the same checkpoints the demo uses with
`--checkpoints walk <path> jump <path> kick <path>`. The demo also requires trained
walk, jump, kick, planner and tracker checkpoints. Null checkpoint paths select the
available latest models. Pin explicit paths for reproducible experiments.

The default viewer is Viser. Its Info panel displays the active skill or bridge,
controller phase, speed and goal status. Green markers show entry locations;
orange markers show bridge trigger locations. A translucent cyan robot shows the
upcoming handoff's target pose, including its orientation and joints, and stays at
that entry during the bridge. It updates to the recovery target during jump and
the kick target during the final approach. Use Reset Environment to restart.

## Tune `config.yml`

`scene` controls the robot spawn, ball, fallen robot and goal. Positions are in
metres, rotations are roll/pitch/yaw in degrees, and `fallen_joints` angles are
in radians. The fallen robot keeps the G1 appearance and its joints are frozen,
with all collisions disabled so it cannot tip over the jumping robot. The ball
starts at rest and remains physically free so
it moves when kicked; the controller flags premature movement rather than moving
it back into place.

Each reset samples `fallen_position_jitter` (symmetric XY metre bounds),
`fallen_yaw_jitter_degrees`, and `fallen_joint_jitter` (a map of joint angle
bounds in radians). Height, roll and pitch remain fixed. Joint variations are
baked into the visual geometry, then stay frozen for the episode.
The default pose turns hips and shoulders so knee and elbow bends stay along
the floor. Wide shoulder, hip, knee, elbow and waist variations create sprawled,
asymmetric poses, with a full range of ground headings and a lower torso height.
Keep the fixed hip and shoulder rotations when increasing these joint bounds;
arbitrary pitch and yaw variations can lift limbs off the floor.
Entry locations and clearance checks use the sampled obstacle position. Set the
bounds to zero and the joint map to `{}` to disable all variation; `--seed`
controls the reproducible sequence of poses.

Each handoff has independent settings:

| Setting | Meaning |
| --- | --- |
| `heading_degrees` | Travel direction; null aims jump/kick towards the goal and recovery along the jump direction |
| `entry_index` | Zero-based selector entry for jump/kick; frame index in the locomotion recording for recovery |
| `start_distance` | Metres before the entry along the travel direction at which the bridge starts |
| `duration` | Seconds available for the bridge to reach the entry |
| `tolerance_scale` | Multiplier for the tracker's endpoint tolerances |
| `require_endpoint` | Fail the learned handoff when the endpoint check fails |

Entry positions come from the task and the selected reference. The kick clip's
ball target is placed at the actual ball, aligned towards the goal; its pre-strike
robot pose determines the entry. Jump centers the reference's takeoff-to-landing
span on the sampled fallen robot. Takeoff is half the flight distance before its
center, and the selected entry also accounts for the reference's run-up before
takeoff. Rotation and motion scaling apply before this placement.

Recovery starts at the positioned jump reference's final root location and uses
an entry point of the selector's walk set, recorded at `speed_after_jump`,
preserving a moving target. There
are no independent entry offsets to tune. Changing an entry index changes its
pose and run-up while preserving the task alignment.

The approach crosses a trigger plane without commanding a stop. Lateral and
heading alignment are checked before entry. Jump exit requires an observed flight,
obstacle clearance, the recovery trigger, and optionally a landing. Configure
these gates in `controller`. The reference must have enough clearance for the
fallen robot's geometry; centering the flight does not increase its jump length.

The pitch has alternating turf stripes, touchlines, a halfway line, center circle,
and penalty and goal areas. Its scoring end follows the configured goal position
and heading; pitch length extends back from that goal. Markings and turf are visual
and use the existing ground for contact.

The goal has white posts and a crossbar, rear supports, and a visible mesh net.
Only the front frame collides; the net is decorative and does not catch the ball.
The goal is a physical frame with a goal-plane crossing check. Goal heading points
in the scoring direction. `controller.require_goal: false` accepts a confirmed
ball launch instead of a goal, which is useful while tuning. Either success waits
for the kick reference's recovery to finish before marking the demo complete.
While the viewer continues, the kick policy keeps balancing at its final reference
pose; the controller never freezes the last actuator command after success.

## Compare bridges and save diagnostics

```powershell
uv run -m mjlab.tasks.bridging.config.g1.demos.soccer.run --bridge mixed
uv run -m mjlab.tasks.bridging.config.g1.demos.soccer.run --bridge no-op
uv run -m mjlab.tasks.bridging.config.g1.demos.soccer.run --viewer none --diagnostics data/soccer/result.yml
uv run -m mjlab.tasks.bridging.config.g1.demos.soccer.run --config path/to/config.yml --dry True
```

`no-op` directly switches policies and bypasses endpoint enforcement; it is a
baseline. Learned bridges use the configured duration and tracker. Diagnostics
include each handoff's measured start distance, duration, endpoint errors and
speed, plus final controller status. Headless failure returns an error.

The walk-to-jump and walk-to-kick settings use per-entry calibration results
in `data/bridging/calibration`, with the measured models, 50 planner sample steps
and a 0.8 m/s source command. Re-record the locomotion entry after updating from
the original 1 m/s setting. Jump-to-walk retains its initial parameters because
there is no calibration for that pair yet.

Jump uses entry 3 (frame 87), 0.6 m and 0.8 s from
`g1_walk_to_jump_diffusion_complete.yaml` (six entries, nine combinations and
three seeds per entry). Kick uses entry 3 (frame 124), 0.2 m and 0.4 s from
`g1_walk_to_kick_diffusion.yaml` (six entries, four combinations and one seed).
The corresponding `_entries.yaml` files contain compact per-entry recommendations.

Calibration ranks post-handoff tracking against nominal tracking. Its successful
trials can still miss the strict endpoint box, so `require_endpoint` is false for
the two calibrated transitions; endpoint errors remain in the diagnostics. It
remains true for jump-to-walk. Calibration alone does not establish success in the
obstacle scene or scoring a goal.

These are the best measured parameters for entry 3 of each skill. Other entries
score higher overall; entry 3 is chosen here to demonstrate a more dynamic handoff.

The physical trial with the current config completed all handoffs, scored a goal,
finished kick recovery and stayed upright for five more seconds under the kick
policy. Minimum root height during those five seconds was 0.786 m. Diagnostics
are in `data/soccer/kick_recovery_match.yml`. Task alignment is also checked for
translated balls, randomized obstacles and rotated approaches.
