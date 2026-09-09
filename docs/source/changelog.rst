=========
Changelog
=========

Upcoming version (not yet released)
-----------------------------------

.. admonition:: Breaking API changes
   :class: attention

   - ``CollisionCfg`` now requires ``contype``, ``conaffinity``, ``condim``,
     and ``priority`` to be explicit instead of silently defaulting to
     MuJoCo's values, and dict values for these fields must cover every
     matched geom (add a catch-all ``".*"`` entry).

Fixed
^^^^^

- A static box from ``get_box_cfg`` floated half its own height above the ground.
  ``get_box_spec`` lifted the geom by half a height inside the body so that a box
  left at the default pose rested on the floor, and ``get_box_cfg`` lifted it again
  through ``init_state``. A box with a mass never showed it, because its freejoint
  takes the pose from qpos and overwrites the offset; a static one has no freejoint,
  so mjlab wraps it in a mocap body and the two lifts add. The lift inside the body
  is gone and ``init_state`` is the only one left, which leaves ``Mjlab-G1-Push``
  exactly where it was.

Added
^^^^^

- ``Mjlab-G1-Climb`` is a new skill: the G1 climbs onto a 0.65 m box, crosses it
  and steps down the far side, tracking one OmniRetarget clip. Getting on and
  getting off are one skill because the source motion is one motion, and a policy
  that stopped on top would end its episode in a state no other skill here has an
  entry point for.

  The obstacle ships with the motion. OmniRetarget (arXiv 2509.26633) retargets
  the human climb and the box together and preserves the contacts between them, so
  the clip is only physical against that box at that pose; a clip retargeted
  without the obstacle has no defined relationship to one placed in simulation, and
  its hands land centimetres above or below the face they push off.
  ``skills/climb/dataset.py`` reads the box out of the scene URDF the release
  ships, rotates it with the clip when the clip is moved to the origin, and writes
  it into the manifest, so the environment places the obstacle the converter
  measured rather than one somebody typed. Its height is measured against whatever
  the subject was standing on at the opening frame, not against the source floor,
  which is what makes it survive the move: the two G1 models do not stand at the
  same foot height, and a difference of two surfaces cancels an offset that applies
  to the box top and the floor alike. Driving the converted reference through
  MuJoCo with the box in place puts the deepest contact at 2.8 cm and the median at
  0.5 cm, against 1.6 cm and 1.5 cm for the same clip's contacts with the floor, so
  the box is registered at least as well as the ground is.

  The approach walk is cropped out. The parkour controller drives the robot up to
  the obstacle with the walk and hands over facing it, so the reference opens
  standing 0.26 m from the near face. That number is what a hand-over has to
  deliver and the converter prints it. What a hand-over cannot promise exactly is
  the angle it arrives at, and that is ``climb_env_cfg.APPROACH_YAW_RANGE``: the
  box and the reference are one rigid thing and neither moves, because every
  reference observation is expressed in the robot's own frame and rotating the
  whole scene together leaves all of them unchanged. What is perturbed is where
  the robot spawns relative to them.

- ``tracking/scripts/datasets/omniretarget/download.py`` fetches the OmniRetarget
  robot-terrain set: 29 climbing scenes, each at five height scales, with the box
  each was solved against. Five of those scenes never get the subject onto the box
  despite the name, and are listed in ``NOT_CLIMBS`` so nobody converts one and
  looks for the bug in their environment.

- ``Mjlab-G1-Kick`` was rebuilt as a motion tracking task and no longer shapes a
  kick out of reward terms. The reference is one of the thirteen human kicks
  published by PAiD (arXiv 2602.05310), which are already retargeted to the 29
  joint G1: ``skills/kick/dataset.py`` downloads one, permutes its columns from
  Isaac Lab's breadth-first joint order into mjlab's, and replays it, which is
  the whole conversion. Driving mjlab's G1 with that permutation reproduces the
  body positions stored in the clip to a maximum of 0.00 mm, so the two models
  are the same robot.

  Where the ball goes is searched rather than configured. The G1's foot collides
  through seven capsules along the sole and nothing else, so the converter takes
  that geometry, scores every position near the strike on how fast the sole
  closes on the ball, and requires the support foot to stay 5 cm clear of it at
  every frame and every corner of the spawn scatter. Both halves are needed: put
  the ball one radius ahead of the striking ankle, which is the obvious rule, and
  in the narrower clips the support foot plants on it during the walk in and
  knocks it away before the swing arrives. Score on clearance alone instead and
  the ball drifts to the outside edge of the swing, where a 6.0 m/s foot delivers
  4.0 m/s. Scoring on the closing speed gives 4.4.

  The default clip runs in: 1.05 m of approach at 1.98 m/s. That follows from the
  same measurement rather than from taste, because the clips that walk in slowly
  also plant their feet close together, and a 0.22 m ball has nowhere to go in
  them. Verified per foot in simulation, driving the reference exactly with the
  full spawn scatter on: the striking foot contacts in 64 of 64 environments and
  the support foot in none, sending the ball away at 4.2 m/s on average.

  The two ball rewards are latched, a contact bonus and a saturating term on ball
  speed along the clip's kick direction, and neither anneals the tracking reward
  away. Reference state initialization is squeezed into the frames before the
  strike, because the frames around and after it put the foot inside the ball and
  a reset there strikes it by teleport. The viewer carries two sliders that move
  the ball while the reference ghost plays, and a position found that way can be
  pinned with ``Clip(..., ball=(x, y))``, which the converter checks rather than
  trusts.

- Added ``tracking/scripts/datasets/asap/download.py``, the one that was missing:
  ASAP's G1-retargeted clips were being fetched by the jump's own converter,
  which meant the repository URL, the file naming scheme and the clip manifest
  lived inside a task. They live in the dataset folder now, next to LAFAN1's,
  openhe's, PHUMA's and AMASS's. Every ``download.py`` also gained a public
  ``fetch(name, output_dir) -> Path``, which downloads one clip unless it is
  cached and returns where it landed, so a task that tracks a single clip asks
  for it by name instead of carrying a copy of the URL. ``front_kick`` and both
  jumps now do. Existing caches under ``data/lafan1_g1`` and ``data/asap/raw``
  are reused as they are.

- ``Mjlab-G1-Jump`` has its own ``dataset.py``, so the task that tracks a clip
  is also the task that fetches and converts it. It is the clip name and nothing
  else; the conversion is the continuous jump's, the download is ASAP's, the
  same way ``punch_combo`` is a frame window over ``front_kick``'s converter.
  Converting one clip no longer replaces the motion directory's manifest, it
  merges into it, so a single clip run cannot leave the other clips unordered
  for ``discover_motion_files``.

- Added ``Mjlab-G1-Jump``, a jump that tracks one ASAP clip end to end instead of
  covering a range of distances. The clip is ``jump_forward_level3``, 1.54 m of
  forward displacement, picked because it lands where a 1.5 m jump lands with no
  horizontal stretch applied and because it is the only one of the five whose
  sole clears the floor at both reset landmarks. One clip means one phase: the
  policy reads the reference at inference the way the front kick and the punch
  combo do, and there is no distillation phase that can quietly fail to recover
  its teacher. The environment is the continuous jump's with the goal terms, the
  stretch and the teacher observation group taken out, subtracted rather than
  restated so the two cannot drift apart in their reward tolerances, their
  sensors or their curriculum.

- Added ``Mjlab-G1-Kick``, a G1 that walks to a loose ball and strikes it at a
  target on the ground. It is a port of RoboNaldo (arXiv 2606.11092, MIT), which
  trains a G1 to shoot and publishes both its reference motion and the mechanism
  that makes an approach possible.

  An episode is two phases and one event. The ball spawns out of reach, so the
  policy walks to it under dense approach and facing terms plus a foot air time
  reward. When the ball is predicted to pass within 0.25 m of the striking area
  inside 0.4 s, a trigger latches and the reference clock starts at the clip's
  entry frame, so the wind up and the strike play out from wherever the robot is
  standing. The policy's own locomotion covers the approach, whatever its
  length, and the clip is snapped in at the moment it becomes relevant. The
  prediction is on the ball's velocity relative to the robot's, which is what
  makes it fire for a still ball the robot is walking at as well as for a rolling
  one.

  The imitation reward is never annealed. RoboNaldo runs ``motion_weight`` at 1.0
  through every stage of its curriculum and scales only its global anchor terms
  when the soccer reward turns on; a policy paid only for where the ball ends up
  is doing reward shaped kicking and looks like it. The curriculum moves the task
  weights and the ball's spawn instead, widening it from a step away to two
  metres and finally rolling it in, which needed a new ``event_curriculum``
  because nothing in mjlab moved an event's parameters before.

  The goal is a target on the ground rather than a launch velocity: a launch
  velocity only means something relative to a heading the robot has already
  committed to, while a target survives a ball the robot had to chase. The
  reward is the closest the ball has come to it, latched, with a short burst when
  a shot actually lands.

  Not ported: RoboNaldo's terrain curriculum, its sim-to-real regularization set,
  its lidar staleness model, and its five separate chained training runs.

- Added ``tasks/tracking/scripts/datasets/gmr/retarget.py``, which retargets
  SMPL-X human motion onto a robot with GMR (Ze et al., ICRA 2026) and writes
  the Unitree generalized coordinate CSV the rest of the pipeline already reads.
  GMR fits the robot to the human's key bodies with a per frame differential IK
  solve, so it needs no training and runs at 35 to 70 frames per second on CPU.
  This is the step the AMASS downloader has always ended one short of: until
  now every motion in mjlab came from somebody else's retargeting. GMR and the
  SMPL-X body models stay optional dependencies, installed only to run it.

- The AMASS downloader now curates a ``kick`` skill, and can select clips by
  their path inside a subset as well as by their file name. That reaches CMU
  subjects 10 and 11, its soccer session, whose trials are numbered rather than
  named, and ACCAD's martial arts kicks.

- Added the ``demos/parkour`` package, which walks a G1 down a procedurally
  generated obstacle course, switching skills at every obstacle. ``course`` draws
  the course from a seed, ``pool`` wraps each frozen policy as a ``Skill`` that
  knows how it has to be spoken to, ``bridge`` aims the bridge at a commanded
  pose, ``arena`` builds the environment and also serves the course on its own
  with no robot and no policies loaded, ``controller`` holds the rules and the
  phase machine, and ``run`` is the entry point.

  Two kinds of obstacle, both solid, both turned, both coloured from the palette.
  A hurdle is 0.20 m tall and 1.00 m long and is jumped; the jump clip covers
  about 1.5 m, so the bar plus the floor its take-off and landing need fits
  inside that. A box is climbed, and every box is identical because the climb
  skill's clip was retargeted together with its obstacle and is only physical
  against that box at that pose. Its size therefore comes out of the skill's own
  manifest rather than out of the config: 0.65 m tall, 1.48 m along the approach,
  0.70 m across.

  Which follows through to the approach. The clip fixes where the box sits
  relative to the robot at frame zero, so the pose the robot has to arrive in is
  that relationship inverted onto the real obstacle rather than a take-off
  distance somebody chose, and the switch waits on heading and cross-track error
  as well as on reachability. ``ARRIVE_SLACK`` still applies: a crossing travels
  along one line, so a demanded pose off that line stays off it and the
  controller waits rather than firing into a miss. The tolerance on the angle is
  the climb's own ``APPROACH_YAW_RANGE``, the spread its reference state
  initialization was trained across.

  The climb reads its obstacle, and on a course there are several. Rather than
  copy the term, the arena rebinds it to one that swaps the scene's entry for the
  obstacle the controller is pointing at, calls the skill's own function and puts
  the entry back: the observation of a frozen policy is bound to its checkpoint
  and a second implementation of it could drift.

  Every number the generator reads lives in ``config.yml`` beside the package,
  ``--config`` points at another one, and ``--viewer`` takes ``viser``,
  ``native`` or ``none``. ``--scene True`` draws the course and stops;
  ``--dry True`` prints the rules, the course and the plan and stops. Both work
  with no checkpoints at all, and ``run`` refuses to start, before building
  anything, when a skill the plan needs has no entry table rows.

- ``Mjlab-G1-Jump`` now trains in two phases in a single run, so that the
  clips shape the skill while it is learned and nothing reads them once it is.
  Phase one is unchanged: PPO against a tracking reward over five retargeted
  clips, stretched horizontally, with the goal in the observation and in the
  reward. Phase two freezes that policy as a teacher and regresses a student
  onto it over the student's own rollouts, in the same process, with no
  checkpoint written in between. The student reads the goal and its own body and
  nothing else, 101 numbers against the teacher's 176, and the 75 that go are all
  reference. It is what the checkpoint holds at the end and what ``play`` and the
  composition arena load. The split is ``agent.tracking_iterations`` of
  ``agent.max_iterations``. Retraining is required: existing checkpoints hold a
  teacher and no student, and are refused with a message saying so.

  This makes the jump conditionable the way the walk and the run are. A hand-over
  into it sets a distance rather than placing a trajectory and winding it to a
  frame, so the entry table's frame column no longer decides what the skill does.

- Added ``RslRlTeacherStudentRunnerCfg`` and ``MjlabTeacherStudentRunner``,
  which run PPO with a privileged observation and then distil the result into a
  policy reading a smaller one, as two phases of a single ``learn`` call. For any
  skill whose observation is only learnable with information it will not have at
  inference. ``MjlabOnPolicyRunner`` gained a ``MODEL_KEYS`` class attribute so
  both runners share its checkpoint handling, and it now serves a
  ``student_state_dict`` to callers asking for an actor, so a distilled policy
  loads through every existing inference path unchanged.

- Added a ``distance`` control to the jump in the humanoid transition arena, so
  ``walk2jump`` can be asked how far the robot has to cover instead of always
  jumping the 1.55 m that was pinned in ``anchor_jump``. The slider spans the
  0.35 to 2.55 m the five clips reach once stretched. To carry it,
  ``Actor.enter`` now also receives the skill's control values: a distance picks
  which clip plays and how far it is stretched, and the anchor reads that when it
  places the reference, so it has to be known at placement rather than written a
  step later the way a ``condition`` is. The entry frame is still an absolute step
  off a table recorded across all five clips, so a distance far from the default
  asks a frame chosen under one clip to mean something under another.

- Added the ``selector`` package, which finds where each skill can be entered
  instead of asking someone to write it down. ``selector.record`` drives every
  trained skill and writes what it does, ``selector.build`` clusters those
  states and keeps the spots most rollouts pass through, and ``selector.view``
  draws one skill's entries side by side in viser. Nothing in it is per-skill:
  no contacts, no gait, no clip landmarks, so the same code covers a jump, a
  kick or a backflip. A candidate is then measured against the floor and
  dropped if the robot is airborne in it, since a mid-flight state is a real
  part of a jump and no controller can put a body on a chosen ballistic arc.
  Grounded states read within a millimetre of the floor and airborne ones 13 cm
  or more, so that bound is a gap rather than a tuned number. The result is a
  small table at
  ``data/selector/entries.npz`` holding, per entry, the pose to aim at, the
  control step to enter the skill's phase at, and how reliably rollouts pass
  through it. This replaces the hand-written posture table it shipped with. It
  says where a skill can be joined, not whether it survives being restarted
  there, which still needs a rollout sweep.
- Added an ``entry`` sampling mode to ``JumpCommandCfg``, which starts an
  episode where the jump loads instead of at the clip's first frame, and made
  it the default for ``uv run play Mjlab-G1-Jump``. The ASAP clips open with
  one to two seconds of standing still before anything happens, and playing
  that back was faithful to the recording rather than anything the skill
  needed. The entry frame is found per clip from the root height, so it
  carries the goal: crouch depth runs from 8 cm on the shortest jump to 26 cm
  on the longest, and one fixed frame could not serve every distance.
  ``entry_landmark`` chooses between the frame the root first drops out of
  standing (``load``, the default) and the bottom of the crouch (``crouch``),
  and ``entry_offset`` backs either one off. No retraining is involved: the
  skill is trained with reference-state initialization over the whole clip, so
  these are frames it has always been reset into.
- ``entry_landmark`` defaults to ``load`` rather than the bottom of the crouch,
  because of where the retargeted clips sit relative to the floor. The clips
  are shifted vertically to stand on the ground, and ``dataset.py`` caps that
  shift with ``MAX_GROUND_PENETRATION`` measured on the ankle body origin. That
  proxy holds while the foot is flat and breaks when it pitches: standing, the
  ankle origin sits 3.6 cm above the sole; in the crouch the heel lifts and it
  grows to 7.7 cm. The clips therefore pass their own check while the sole is
  4.3 to 7.5 cm underground, and the crouch is the worst frame in every one of
  them. That is free while the clip is only a tracking target, and costs
  something at a reset, which writes the robot into the pose. ``load`` is the
  last frame before the sink begins.
- Added ``...skills.jump.entry``, which measures what each entry mode costs a
  checkpoint. On ``g1_jump`` the entry landmarks rank by how deep the reset
  pose sits: ``load`` enters with the sole 7 mm under and survives 100% of
  episodes, matching a standing start, while the bottom of the crouch enters
  64 mm under and survives 99.6%.
- Added ``GeomCfg``, exposed as the ``geoms`` field on ``EntityCfg``, a spec
  editor that matches geoms by name and patches their attributes. Supports
  ``group`` (so a geom can collide without being drawn) and all collision
  attributes; unset attributes are left untouched. Contribution by
  @bd-pmorais.
- Added ``diffuse``, ``specular``, ``ambient``, ``active``, and
  ``attenuation`` fields to ``LightCfg`` for configuring light color and
  falloff. Contribution by @bd-pmorais.
- Added light domain randomization functions: ``dr.light_diffuse``,
  ``dr.light_specular``, ``dr.light_ambient``, ``dr.light_attenuation``,
  ``dr.light_cutoff``, and ``dr.light_exponent``. Contribution by @bd-pmorais.
- Added a ``gui`` field to ``CommandTermCfg``, defaulting to ``True``. Set it
  to ``False`` to keep a command term from adding controls to the Viser
  viewer, for terms something else already drives.
- Added a ``guidance`` reward to ``Mjlab-G1-Bridge``. Every window is a
  contiguous slice of one tracker rollout, so the frames between its endpoints
  are a crossing this robot performed under this physics; the term pays for
  staying near that crossing, and its weight anneals to zero over
  ``BridgeCommandCfg.guide_steps``, after which the reward is what it was
  before. It is shaping, never an objective: there is no reference at inference
  and the start is perturbed off the recorded one during training, so scoring
  the crossing would pay for imitating one answer instead of for arriving. Both
  observation groups are untouched, so the policy reads only its own state and
  the gap to the target. ``Segments.draw`` also returns the position a window
  opened at, and ``Segments.path`` reads the rows between a window's endpoints.
- ``uv run play Mjlab-G1-Bridge`` now draws two ghosts: the target in amber
  standing where the window ends, and the recorded crossing in blue walking
  through it as the clock runs.
- Added ``...bridge.datasets.view``, a Viser viewer for the windows a bridge is
  trained on. It draws a start/target pair the way the command term does and
  replays the recording it was cut from as a ghost: green through the context
  either side, red across the masked stretch between the two states. It also
  lists what the corpus holds per source (states, rollouts, and how many windows
  each admits); a source with states but no windows contributed nothing to
  training and was previously visible nowhere. Previous/Next step through the
  windows drawn so far, and Play/Stop/Reset drive the playback.
- Added a ``priority`` argument to ``get_box_spec`` and ``get_box_cfg``,
  defaulting to ``0``. Raise it so the box's own ``friction`` wins against the
  terrain's instead of losing to MuJoCo's elementwise maximum, which is what
  lets a box be more slippery than the ground it sits on.
- ``tests/transitions/walk2kick.py`` stages a hand-over from the walk into
  ``Mjlab-G1-Kick``, the football kick. It is the first couple whose entering
  skill is both a clip tracker and a skill with an object, and the two together
  are what make it different from the pass: the clip runs in for a metre and
  strikes a ball fixed relative to the swing, so the swing only connects if the
  clip is laid down with its ball on the real one. That inverts the placement.
  Instead of predicting where the robot's momentum will carry it, the harness
  solves for where the robot has to stand,

  .. code-block:: text

     target = ball - R(heading) * (ball_in_clip - clip_root(entry frame))

  which is ``JumpCommand.anchor_to_robot`` read backwards, built out of that
  method's own ``anchor_yaw_for`` and ``clip_root_at`` so the two cannot drift
  apart. ``Actor.arrive`` therefore takes the entry frame now: the kick's window
  is fifty frames of run-up, so where the robot has to stand depends on how much
  of that run-up it is skipping, and the answer moves by half a metre across the
  window. Every other skill ignores the argument. The switch is the button, as
  it is for the jump and the strikes; the window is still solved from the
  geometry, which is a separate question from when to fire and is now written
  that way.

Changed
^^^^^^^

- ``selector.view`` and the transition harness draw a skill's entry states as
  the sequence the skill passes through rather than as a row of poses. They used
  to stand side by side, evenly spaced, which says every entry is the same
  distance on from the last and is false for every skill that accelerates
  through its window. ``EntryTable.trail`` now says how far apart they really
  are: the ground position is not in a state, since ``canonical`` drops it and it
  would not compose across two medoids taken from two rollouts anyway, so the gap
  between two entries is the mean of their two root velocities over the time
  between their frames, which is the same model the bridge places its targets
  with. The kick's six entries come out as half a metre of run-up and the jump's
  six as one tile the robot crouches on, both of which are the truth about those
  windows. ``selector.view --gap`` inserts daylight between entries for a window
  the skill barely moves through.

  The transitions draw the whole window with the target. At the switch the
  target ghost is joined by a faint one per entry, placed along that trail and
  hung off the target, so the aimed state sits in its own place in the line and
  the rest of the skill's approach runs back from it. For a skill whose target is
  fixed by an object, that means the object places the entire line: the kick's
  ghosts run back from the ball.

  For a couple whose entering skill has an object, that line is now drawn from
  the first step rather than at the switch, and a ``show the entry states``
  checkbox turns it off. A hand-over into an object skill is fired at a moment,
  and firing it late is a miss rather than a worse score: the robot walks through
  its own ball. Nothing in the arena knows when that moment is, because the button
  is what a controller would be replacing. What makes it visible is that the
  moment is really a place. ``Actor.arrive`` says where the robot has to stand for
  the skill to meet its object, and an object does not move, so evaluating it at
  each entry's own frame gives a line of poses fixed to the floor and the switch
  reduces to walking onto them. The viewer's line reports the metres still to go
  next to whoever is driving, so a number closing on zero is a switch worth firing
  and one that has bottomed out and started growing again is a robot that has gone
  past. Measured on ``walk2kick``, firing at step 130 leaves the robot 0.08 m from
  where the kick wants it and firing at 180 leaves it 0.68 m past.

  The line drawn before the switch is ``Actor.arrive`` per entry, not
  ``EntryTable.trail``. The trail integrates the entries' own velocities and is a
  few centimetres out over half a second, which is most of the box a ball skill
  was trained in; ``arrive`` reads the clip. A skill with nothing on the floor
  gets neither the line nor the checkbox, since its target moves with the robot
  every step and there is nothing to walk towards.

- ``PUSH`` declares an ``arrive``. ``arrive_at_box`` inverts ``box_in_reach`` the
  way ``_arrive_at_ball`` inverts ``_ball_in_reach``, so the push gets the entry
  ghosts and the standing-error diagnostic the other object skills have.

- The bridge's target now lands on one of the entry states that are drawn, rather
  than near them, and both the transition harness and the parkour demo were
  getting this wrong in their own way. Drawing the states an object skill needs
  and then aiming somewhere else is worse than drawing nothing: the operator times
  the switch against a line the switch does not use.

  In the harness the split was ``Config.commanded``, which decided whether
  ``Actor.arrive`` placed the target and defaulted off. It is gone. A skill that
  says where it needs the robot is aimed there; a skill that says nothing still
  gets the ballistic midpoint. The flag was off because commanding the arrival
  measured 0.104 m against 0.083 m on ``walk2pass``, which was never deciding
  anything against a box eight centimetres deep and was measuring the target on
  its own rather than against the entry states that have to agree with it.

  What the change buys exactly is the agreement, not a better number: the line
  drawn at the switch is the demanded one frozen, so the aimed entry's ghost is the
  target to the last printed digit rather than a few centimetres from it, on every
  couple. The standing error a hand-over inherits moved from 0.172 m to 0.059 and
  0.149 m over two runs of ``walk2pass``, which is spread rather than an
  improvement: that couple fires on geometry, so a small change in the solved window
  moves which entry it takes and which stride it fires on.

  In the parkour demo ``Controller.frame_of`` returned ``entries[0].frame``, the
  first row of the table, while ``nearest`` chose whichever entry was easiest to
  reach. So the approach pose was solved for one entry and the target's state came
  from another: a pose the skill really passes through, standing somewhere it
  never passes through it. ``__call__`` carried a comment claiming it re-solved at
  the chosen frame and it did re-solve, at the same wrong one. ``ready`` now picks
  the reach first and solves the alignment gate, the approach pose and the window
  at that entry's frame, and ``frame_of`` answers with the entry a hand-over would
  actually use. It hid because the climb has a single entry, where the two agree;
  the jump has six.

- The parkour demo's walk now holds station short of the approach pose instead of
  driving at it. It was cruising at ``approach_speed`` until something fired, which
  crossed the whole band the switch can fire in inside about twenty control steps;
  if the heading had not settled by then the robot walked through the spot and
  parked past it, and from there the window that would land a crossing on the pose
  comes out negative, which ``ready`` can never accept. Measured on a two obstacle
  course it sat 0.16 m past the spot with the solve reading -1.5 s for the whole
  run and cleared nothing. ``Controller.stand_off`` is the distance to hold: half
  the entry's own forward speed times the middle of the trained window, which is the
  ground a crossing covers when it starts from a robot that has stopped. Off the
  entry's speed and nothing else, because averaging in the robot's own does not
  converge, a robot stepping in place still having a root velocity that swings
  through most of a stride. ``walk.approach_gain`` and ``walk.reverse_limit`` in
  ``config.yml`` are the regulator, and the gain is high because a proportional
  command stalls where it meets the walk's own deadband: at 0.6 the robot stopped
  0.34 m out asking for 0.16 m/s it would not act on, and a crossing into an entry
  moving at 0.2 m/s covers only 0.12 m, so it has to stop inside that or no trained
  window reaches it.

  Measured on the same two obstacle course, the demo now fires at step 43 into the
  climb's entry over a 0.64 s window and arrives 0.054 m and 2.6 degrees off the
  pose it was promised, against a run that previously made no decision at all in
  8000 steps. It still goes down during the climb itself, which is a hand-over
  quality question rather than a controller one: the climb has a single entry, so
  there is no second state for the selector to prefer, and the arrival is 0.37 rad
  out at the worst joint.

- ``walk.lateral_gain`` in the parkour ``config.yml`` goes from 0.8 to 2.5. Holding
  station rather than cruising through, the walk has to close its cross-track error
  standing still, and at 0.8 an error of 0.13 m asks for 0.09 m/s sideways, which
  the policy stands through: the error moved 0.009 m in a thousand control steps and
  the alignment gate never opened. At 2.5 the same error asks for 0.33 m/s, a
  stride, still under the 0.35 limit and well inside the walk's trained range, and
  the error closes to 0.055 m against a tolerance of 0.080.

- ``Actor.ready`` is no longer called for a skill that declares ``arrive``.
  Solving the window to land the crossing on the demanded pose puts the object in
  its box by construction, so the precondition would be answering its own
  question. What the pass and the push still need the declaration for is to say
  they fire themselves rather than waiting for the button, which a precondition is
  a roundabout way of saying; the docstring says so rather than leaving it
  implied.

- The ``selector`` package no longer discovers where a skill can be entered. It
  clustered every state a skill visited and then filtered the clusters on
  coverage, dwell, clearance and progress, which cannot work: a pre-jump stand
  and a post-landing stand are the same state, so no rule reading the state
  alone keeps one and drops the other. ``jump`` kept six entries of which three
  sat after the landing, and ``climb`` kept one, mid-hop with both hands over a
  box the selector cannot see.

  Where a skill may be entered is now written down by hand, per skill, as a
  window over that skill's own timeline in ``selector/__init__.py``:

  .. code-block:: python

     WINDOWS = {"jump": Window(phase=(55, 100), states=6), ...}

  ``selector.build`` cuts the window into that many equal slices and keeps the
  medoid of each one, so the states are equally spaced along the window and the
  medoid discards the rollouts that drifted or fell without a threshold saying
  which. ``selector.query`` is unchanged and hands back the closest of them to
  where the robot is, which now means the phase of the skill matching the speed
  the robot arrives with. A skill with no window gets no entries and cannot be
  handed over to; ``walk`` and ``pass`` have one only because the parkour
  controller has to aim somewhere when it hands back.

  ``selector.filter`` and ``data/selector/candidates.npz`` are gone, together
  with the ``progress``, ``dwell_s``, ``hold_s`` and ``share`` columns.
  ``Entry`` carries ``seconds`` instead, and ``coverage``, ``spread`` and
  ``clearance`` are now diagnostics that no code reads. Existing
  ``data/selector/entries.npz`` files do not load; re-run ``selector.build``,
  which needs no new recording.

- The ``walk2jump`` transition hands over to ``Mjlab-G1-Jump`` instead of
  ``Mjlab-G1-Jump-Continuous``. The continuous jump deploys a distilled student
  that reads a goal and its own body and no reference at all, so a hand-over
  into it had a distance to write and no phase to resume: the entry table's
  frame column meant nothing to it. The single clip jump is a tracker like the
  front kick and the punch combo, so the jump actor is one line again, with no
  controls, no ``condition`` and ``anchor_clip`` for its ``enter``. The
  ``distance`` slider and ``--tell "{'distance': ...}"`` are gone with it, and
  the selector's ``jump`` rows were already recorded against this task.

- ``front_kick`` and ``punch_combo`` are gone as separate packages and are now
  two entries of ``skills/martial``, one package for every martial arts motion
  cut out of a LAFAN1 fight performance. They already shared a converter and an
  environment, with ``punch_combo`` contributing a frame window and a
  registration and nothing else, so the split was only in the directory names.
  A motion is now a line in ``MOTIONS`` in ``martial/dataset.py``: it gets a task
  named after it, its own clip directory and a ``g1_<name>`` log directory, and
  both ``SKILLS`` dictionaries pick it up from ``MARTIAL_TASK_IDS``. Task ids,
  clip directories and log directories are unchanged, so existing checkpoints
  and converted clips keep working. The converter builds its scene once and
  converts every motion asked for, ``--motion`` picks one, and the window
  override is ``--crop-source/--crop-start/--crop-end`` instead of the old
  ``--crop.*``, which ``tyro.conf.AvoidSubcommands`` had been dropping.
  ``g1_strike_env_cfg`` is ``g1_martial_env_cfg``.

- The goal conditioned jump is now ``Mjlab-G1-Jump-Continuous``, in
  ``skills/jump_continuous``, logging to ``g1_jump_continuous``. The name
  ``Mjlab-G1-Jump`` goes to the new single clip tracker above, which is what a
  jump means by default here now. Existing runs under ``logs/rsl_rl/g1_jump``
  are continuous jumps and have to move to ``g1_jump_continuous`` with the
  rename, or ``play`` will pick one by modification time for the wrong task.
  ``SKILLS`` in both the skills package and the bridge's dataset now carries
  both entries, and the bridge's skills dataset keeps recording the continuous
  jump by default, which is the skill it recorded before.

- The transition harness under ``bridging/experiments/humanoid/tests`` now
  calls the selector's own API instead of the compatibility shim that was
  deleted with the new selector, so ``tests/stage.py``, ``tests/handoff.py``
  and every ``tests/transitions`` script import and run again. Entries are
  ranked per control step by ``selector.nearest``, which orders them by the
  rate of change each would demand of the body as it is moving now rather than
  by a property of the entering skill alone. ``--mode`` says who picks: ``auto``
  takes the top of that ranking every step, out of the entries ``filter.py``
  accepted, and has no slider; ``manual`` puts the choice on the ``entry``
  slider over every candidate the clustering found, rejected ones included, in
  the order ``selector.view`` draws them. The bridge's window is the longer of
  the couple's own default and the shortest window the ranking says is
  feasible: an effort of 1 is a per-channel corpus maximum, so it is a floor to
  be cleared and not a window to ask for. Scoring a hand-over is the harness's
  own job now, so the discount comes from the entering skill's PPO config and a
  pass or fail needs ``--baseline``, read off the ``perfect`` column of
  ``tests/handoff.py``.
- The entering skill of a transition now resumes at the step its entry was
  recorded at instead of at its own first frame. ``Actor.enter`` takes that
  frame and a clip tracker winds its reference to it, so a hand-over into the
  middle of a motion no longer replays the run-up the bridge just crossed. A
  tracking policy has no privileged beginning, so any frame is a state it
  continues from once the robot is put in it, which is the bridge's job.
  Caveat: ``selector.record`` drives each skill's training config and stores
  the episode age, and the trackers train with ``sampling_mode="adaptive"``,
  which starts every episode at a random clip frame. So the recorded frame is
  an age offset by that random start, not a clip phase, until
  ``datasets.dataset.record`` stores the tracker's own ``time_steps``.
- Removed the leftovers of the deleted ``kick`` skill from the transition
  harness: the ``KICK`` actor and ``tests/transitions/walk2kick.py``. Nothing
  in ``tests`` could be imported while they named a package that is no longer
  there. ``walk2pass`` covers the ball skill that exists.
- Removed the bridge's warm-start machinery: ``warm_start.py``,
  ``BridgeRunnerCfg``, ``BridgeOnPolicyRunner`` and the ``--agent.warm-start``
  flag. It was measured to hurt (0.064 score against a cold start's 0.142 at a
  matched 60 iterations) and was off by default. ``Mjlab-G1-Bridge`` now
  registers a plain ``RslRlOnPolicyRunnerCfg`` with no custom runner class.
- Rewrote the bridge's docstrings and comments: shorter, tables instead of
  prose, and a "Run" section with the commands to type in the modules that have
  an entry point.
- Bumped ``rsl-rl-lib`` from 5.4.0 to 5.4.2.
- ``CollisionCfg`` and ``GeomCfg`` now share one write path, and mjlab warns
  when a ``GeomCfg`` collision patch is overwritten by a ``CollisionCfg``.
- Changed the default MuJoCo Warp render background to solid black
  (``0, 0, 0, 1``), matching MuJoCo's native renderer. Contribution by
  @bd-pmorais.
- ``Mjlab-G1-Bridge`` now draws every window as a contiguous segment of a
  single rollout instead of pairing two independently sampled states, so the
  displacement, velocity change, joint travel and time available of each window
  were demonstrated together under physics. Datasets must be rebuilt: the
  segment index needs the ``trajectory`` column, and the task refuses to load a
  file without it.
- ``BridgeCommandCfg`` takes a single ``sources`` filter in place of
  ``leaving`` and ``entering``, which described a pairing the task no longer
  builds, and ``BridgeCommand.place`` and ``open_window`` take a duration in
  seconds instead of a count of control ticks. ``BridgeCommand.steps_for`` is
  the only conversion.
- The bridge's arrival metric splits its joint channel into legs/torso and
  arms, giving eight channels instead of six, and scores each group on its
  worst joint. An arm left behind used to be a fraction of one channel out of
  six and is now a channel at zero, which the bottleneck term prices at most of
  the episode. Tolerances were recalibrated against the new windows.
- The bridge's positive ``air_time`` reward is replaced by a one-sided
  ``feet_chatter`` penalty. The old term paid for a gait whatever the target
  was, which fought every commanded crouch or planted stance; the new one
  charges only for a flight or a stance too short to be a step, so a planted
  foot and a real step both cost nothing.
- The bridge's start-state perturbation now widens over
  ``start_noise_steps`` rather than being applied at full scale from the first
  window.
- The bridge's arrival tolerances are now a requirement in physical units
  (metres, radians, and rates) chosen for what the next skill needs, instead of
  being derived from the corpus at half each channel's median gap. The derived
  values moved whenever the dataset was rebuilt, so no two runs shared a
  definition of success, and the two corpora produced different gaps and could
  not share one set at all. The gap measurement survives as a read-only report:
  ``evaluate --calibrate`` becomes ``evaluate --diagnose``, which says which
  channels a robot standing still already satisfies and which are far enough out
  that ``arrived`` will read zero, and no longer prints a block to paste back
  into the config.
- The transition arena builds its control panel from what each skill declares
  it can be told, as ``Actor.controls`` and ``Actor.condition``: walking gets a
  forward speed, a sideways speed and a heading, the strike skills get a ball
  speed and an aim over the range they were trained on, and the punch
  combination gets nothing because it is one clip with no goal. A skill's
  conditioning is applied only while that skill owns the world, which replaces
  the ``entering_speed`` special case that existed because walk and run share
  one velocity term and whichever wrote it last won.
- The hand-over trigger now chooses the bridge's window as well as the moment.
  It asks, of every duration the bridge was trained on, where the robot would
  end up, and fires as soon as one of them satisfies the entering skill's
  precondition. A fixed window could only wait for the world to drift into
  agreement with it, so a skill needing the robot further on had to be walked
  towards until the arithmetic worked out, and one already too close never
  fired at all. On walk to kick the switch picks 1.20 s from a metre out, where
  the entry's own 0.70 s would have had to wait another third of a metre.
- ``Actor.robot`` lets a skill patch the robot the arena builds. The kick reads
  its observation off two sites it adds to the striking foot, and the arena kept
  the bridge's plain robot, so that transition could not be staged at all.
- Added ``tests/handoff.py``, which measures what a hand-over is worth before
  a bridge exists. It interrupts the leaving skill mid-stride and runs the same
  hand-over twice: cold, with the entering skill taking over from wherever the
  robot was left, and into a teleport onto the entry state, which is an oracle
  bridge. The gap between them is the headroom a real bridge is competing for.
  On walk to punch_combo: cold falls after 17 steps for a discounted return of
  0.013, the oracle runs the full window for 0.152.
- Entry states for skills trained by imitation are read from the skill's own
  reference clip instead of being written by hand, resolved through the task's
  registered config so the selector and the policy cannot point at different
  files. ``punch_combo`` now opens from its clip's frame zero, a fighting guard
  68 degrees from the default pose at the worst joint, rather than from a stand.
- ``BridgeCommandCfg.dataset_path`` accepts ``None``, for a command aimed
  entirely from outside. The transition arena was loading the bridge's whole
  training corpus to learn a frame rate the environment already knew, which also
  meant a transition could not be staged before the corpus was built.
- Fixed ``front_kick`` and ``punch_combo`` pointing at clip directories the
  converted motions are no longer in, which left both tasks registered with an
  empty motion list.
- The selector is a hand-written lookup table again. ``selector/table.py``
  holds a short list of postures per skill in degrees and metres,
  ``selector/check.py`` reports whether each stands on the floor, stays inside
  its joint limits and keeps its mass over its feet, and ``selector.build`` and
  the fitted scorer are gone. Discovering an initiation set is a research
  problem of its own, and while it is open a bad selector and a bad bridge fail
  identically, so the bridge cannot be developed against it. The measured
  version, which scored each candidate by the discounted return the skill opened
  with, is in git history.
- The bridge now defaults to the human motion corpus
  (``data/bridge/tracker.npz``). ``datasets/skills.py``, which draws states from
  the skill pool's own rollouts, is deprecated and warns when run: a bridge
  trained and evaluated on the same five policies cannot show whether it learned
  bridging or learned the pool. It still loads, and ``SKILLS_DATASET`` names its
  path for reproducing older runs.

Fixed
^^^^^

- The selector's ``frame`` column is the frame of the skill's own reference the
  state was recorded at, so a tracker can actually be resumed there. It used to
  be the control steps since that row's episode reset, which for a tracker is
  not a clip index at all: these tasks reset into a *sampled* frame of the clip,
  so two rollouts one step old sit at two different frames. Winding the jump's
  clip to entry ``p28``'s frame 53 landed on a stand 16 cm taller than the crouch
  the entry actually holds, and the policy stood, crouched and jumped as if from
  scratch. ``dataset.record`` now records ``clip_phase`` alongside the age, in a
  new ``phase`` column, and ``build.py`` puts it on the entry. Rollout files
  written before the column still load; the entry table has to be rebuilt with
  ``selector.record``, ``selector.build`` and ``selector.filter`` for the frames
  to mean anything. Affects every tracker: the jump, the front kick, the punch
  combo, and the tracking corpus in ``datasets/tracker.py``.

- A clip tracker taking over from the bridge now resumes at the frame its entry
  was recorded at, rather than a whole window past it. The reference is wound to
  that frame when the switch fires, and a motion command advances one step per
  environment step no matter who is driving, so by the hand-over it had played
  the entire crossing forward: 35 frames of a 212 frame jump at the default 0.7 s
  window, handing the tracker a reference a third of a jump ahead of the robot
  the bridge had just delivered into the entry state. ``Run.resume`` repeats the
  placement at the hand-over with the same target, heading and frame, so only the
  phase moves and the arrival error is left where the bridge left it. Affects
  every tracking hand-over: the jump, the front kick and the punch combo.

- The two phase runner no longer dies at the phase boundary when logging to
  W&B. rsl-rl closes the logging writer at the end of every ``learn``, and for
  W&B closing means ``wandb.finish``, after which the module swaps ``wandb.log``
  for a stub that raises. The runner already stopped phase two from opening a
  second writer, so phase two inherited the closed one and raised on its first
  logged iteration, losing the whole tracking phase. Phase one now leaves the
  writer open and it is closed once, at the end of phase two. A run interrupted
  this way is recoverable without retraining: resume from the last tracking
  checkpoint with ``--agent.tracking-iterations`` set to its iteration, and
  phase one is skipped entirely.

- ``JumpCommand.solve_goal`` now rounds the reachability miss before comparing,
  so its documented tie-break on how little a clip is stretched actually runs.
  Any distance two clips both reach exactly left two residuals differing only in
  the last bits of a division, so the choice was made on float noise: at 1.55 m it
  picked level 4 shortened to 0.85 over level 3 stretched to 1.008. This also made
  the goal-only command disagree with the tracker about which jump a distance
  means; the two now agree across the whole commanded range.

- ``csv_to_npz`` now saves to a configurable local directory and only uploads
  outputs to Weights & Biases when ``--upload-to-wandb`` is supplied.
- Fixed bridge datasets built from more than one source silently losing almost
  every training window. Each source is recorded on its own, so its trajectory
  identifiers restart at zero and collide with the previous source's;
  ``Dataset.segments`` sorts rows by ``(trajectory, frame)``, and the collision
  interleaved rows of different sources at equal frames so no adjacent pair
  stepped by one. On a four-clip corpus this left 1123 of 58217 windows on the
  train split, one clip contributing none at all, and raised ``No contiguous
  rollout segment matches the requested sources and duration range`` on the
  eval split, which is what ``uv run play Mjlab-G1-Bridge`` and
  ``...bridge.evaluate`` read. ``load_dataset`` now pairs each identifier with
  its source. Datasets already on disk are repaired on load and need no
  rebuild.
- ``RayCastSensorCfg.include_geom_groups`` now raises on values outside
  ``[0, mjNGROUP)`` instead of silently excluding every geom.
- Geoms with a negative group no longer pick up group 5's visibility toggle in
  the Viser viewer.

Version 1.5.3 (July 22, 2026)
-----------------------------

Changed
^^^^^^^

- The Viser reward bar panel's term cap is now configurable via
  ``ViewerConfig.reward_bar_max_terms``, so environments with more than 20
  reward terms can show them all. Defaults to 20, preserving previous behavior.
  :issue:`1079`

Fixed
^^^^^

- Bumped ``pillow`` (12.3.0), ``onnx`` (1.22.0), and ``soupsieve`` (2.9.1) in the
  lockfile to pick up security fixes.
- Fixed raycast sensor debug visualization and observations lagging one step
  behind the sensed hits. ``sense()`` rebinds the hit tensors after the cache had
  already been repopulated by a pre-sense reward read, so ``.data`` returned the
  previous step's hits; the cache is now invalidated after ``postprocess_rays``.
  With ``ray_alignment="yaw"`` this made debug rays appear tilted by the foot's
  per-step motion instead of vertical. :issue:`998`
- Restored ONNX uploads and W&B run metadata for velocity and manipulation
  training when using RSL-RL's current ``WandbLogWriter`` logger name.
- The Viser reward bar panel no longer *silently* drops reward terms beyond
  ``max_terms``; it now emits a warning listing the hidden terms. Previously
  environments with more than 20 reward terms had the overflow disappear from
  the bar panel with no indication. :issue:`1079`
- Fixed the ``terrain_levels_vel`` curriculum promoting every env from level 0
  to level 1 on the initial reset, ignoring ``max_init_terrain_level=0``. Before
  the first step the robot sits at its spawn pose rather than a walked-to
  position, so the distance check was spurious; terrain levels are now frozen on
  that first reset. :issue:`1094`
- Fixed the velocity task's actor ``joint_pos`` observation not being biased by
  the ``encoder_bias`` domain randomization, so the encoder bias only affected
  actions and never the observed joint positions. The actor now observes biased
  joint positions while the critic keeps the true (unbiased) values as
  privileged information, matching the tracking task.
  See `discussion #1065 <https://github.com/mujocolab/mjlab/discussions/1065>`_.
- Hardened ``fit_terrain_normal`` against non-finite raycast hits. A single env
  with a diverged state produced a NaN/Inf covariance that made
  ``torch.linalg.eigh`` raise and abort the whole batch; such rows now fall back
  to the up vector. This stops the hard crash so a diverged env can be reset
  normally; it does not by itself make a diverged env's downstream reward finite.
  :issue:`912`
- Enabled ``obs_normalization`` on the Go1 velocity actor and critic to match
  the other velocity tasks. Without it, extreme-but-finite observations on rough
  terrain drove value/policy divergence that eventually surfaced as a
  ``normal expects all elements of std >= 0.0`` crash. Note that Go1 velocity
  checkpoints trained before this change carry no normalizer buffers and will no
  longer load; retrain from scratch. :issue:`870` :issue:`1044` :issue:`1053`
- Fixed ``ContactSensor`` air-time tracking accumulating float32 sim-clock
  differences, whose quantization error grows with the clock magnitude and made
  ``compute_first_contact`` / ``compute_first_air`` miss touchdowns on long runs.
  The exact float64 substep ``dt`` is now accumulated instead. :issue:`1101`
- Bumped ``mujoco-warp`` to 3.10.0.3, fixing a CUDA 700 illegal memory access in
  ``smooth.crb`` triggered by startup mass domain randomization (via
  ``set_const``) once ``num_envs >= 128`` on consumer Ada GPUs. :issue:`1108`

Version 1.5.2 (July 17, 2026)
-----------------------------

Fixed
^^^^^

- Fixed CUDA illegal memory accesses when domain randomization triggers
  ``set_const`` with multiple environments. ``actuator_acc0`` is now expanded
  per environment before MuJoCo Warp recomputes it.
- Fixed ``MaterialCfg.reflectance`` being ignored when building the MuJoCo
  spec. Contribution by @bd-pmorais.

Version 1.5.1 (July 15, 2026)
-----------------------------

Added
^^^^^

- Added ``MeshCfg``, a spec editor that matches mesh assets by name and edits
  their asset-level attributes. The first attribute is ``maxhullvert``, which
  caps the collision convex hull's vertex count to lower narrowphase cost.
- Added ``SimulationCfg.broadphase`` and ``SimulationCfg.broadphase_filter``
  to configure MuJoCo Warp's broadphase collision algorithm and
  bounding-volume filters.

Changed
^^^^^^^

- Enabled skybox rendering for camera sensors. Contribution by @bd-pmorais.
- Bumped the minimum ``mujoco-warp`` to 3.10.0.2, which fixes ``qfrc_constraint``
  being populated incorrectly across vectorized environments (:issue:`1086`).
  Earlier 3.10.0.x releases are no longer supported.
- Command delay on fusable actuators (ideal PD, DC motor) now applies one shared
  lag per environment across all fused actuators sharing a delay config, matching
  the built-in actuator path, rather than an independent lag per actuator group
  (:issue:`1035`).

Fixed
^^^^^

- Fixed ``TerrainGenerator`` overwriting custom geom names set by sub-terrain
  functions with the default ``terrain_{i}`` name. Only unnamed geoms are now
  auto-named.
- Fixed ``TorchArray`` not expanding world-shared model fields to ``nworld``
  with mujoco_warp 3.10.0.2, which allocates them as real size-1 arrays
  instead of stride-0 broadcast views. Multi-env indexing of fields like
  ``soft_joint_pos_limits`` raised ``IndexError`` during resets (:issue:`1093`).
- Fixed ``mdp.bad_orientation`` returning NaN when float32 rounding in
  ``quat_apply_inverse`` pushed the projected-gravity z-component slightly
  outside ``[-1, 1]``, making ``torch.acos`` return NaN and silently
  suppressing the termination for flipped robots. The argument is now clamped
  to ``[-1, 1]``.
- Fixed a crash when using command delay on ideal PD (or other custom)
  actuators whenever ``num_envs`` differed from the number of delayed targets,
  and fused ideal PD and DC motor actuators sharing a transmission and delay
  config into a single gather, delay, control-law evaluation, and control
  write, removing per-group host overhead (:issue:`1035`).

Version 1.5.0 (June 28, 2026)
-----------------------------

Added
^^^^^

- Added ``reduce="max"`` to ``MetricsTermCfg`` for reporting episode-peak values
  (e.g. peak power, peak contact force) without needing stateful wrapper classes.
- Added ``BuiltinDcMotorActuator``, a native MuJoCo ``<dcmotor>`` wrapper.
  Supports voltage / position / velocity input modes with back-EMF,
  configurable motor constants, and optional integral, slew, inductance,
  thermal, LuGre, and cogging extensions.
- Added ``scale_with_difficulty`` to ``HfRandomUniformTerrainCfg``. When
  enabled, the noise amplitude scales with difficulty (flat at 0, full
  ``noise_range`` at 1) so the terrain progresses in a curriculum. Defaults to
  ``False``, preserving the previous difficulty-independent behavior.
- Added material domain randomization functions for MuJoCo Warp RGB rendering:
  ``dr.mat_emission``, ``dr.mat_specular``, ``dr.mat_shininess``, and
  ``dr.mat_texrepeat``.

Changed
^^^^^^^

- Bumped ``rsl-rl-lib`` from 5.2.0 to 5.4.0.
- Bumped ``mujoco`` and ``mujoco-warp`` to 3.10, both pinned from PyPI. The
  ``py.mujoco.org`` nightly index and the ``mujoco-warp`` git pin are dropped, so
  resolution no longer breaks when nightly wheels are garbage-collected.

  .. warning::

     ``SimulationCfg.ls_parallel`` is deprecated and now ignored, since parallel
     linesearch was removed upstream in MuJoCo Warp. Setting it emits a
     ``DeprecationWarning``; remove it from any ``SimulationCfg`` you construct.
- Curriculum-mode terrain difficulty is now deterministic across rows
  and reaches the configured ``difficulty_range`` endpoints
  (:issue:`1027`).
- Heightfield terrains now color by absolute height with a diverging palette
  (cool below the ground plane, green at ground level, warm above) on a fixed
  scale, replacing the per-patch normalization. Color is now consistent across
  terrains, and low-amplitude terrain such as ``random_rough`` reads as gently
  tinted ground instead of high-contrast noise.
- ``BoxNestedRingsTerrainCfg`` now builds uniform-height concentric ridges
  whose separating gaps widen with difficulty, replacing the random per-ring
  heights. Rings are colored by height (like the other terrains) and the outer
  border matches the ring height.
- Terrain generation no longer prints timing information to stdout.

Fixed
^^^^^

- Fixed domain randomization events that target different ``axes`` of the same
  model field (e.g. two ``dr.geom_size`` events scaling axis 0 and axis 1
  separately) silently clobbering each other. Each event now writes back only
  the axes it targeted, so per-axis events compose (:issue:`1042`).
- Regenerated the bundled MuJoCo type stubs, which had drifted from the
  installed mujoco version. CI now regenerates them and fails if they are
  stale, so they stay in sync going forward. Run ``make stubs`` to update them
  (:issue:`1048`).
- Fixed ``select_gpus`` crashing when ``CUDA_VISIBLE_DEVICES`` contains MIG
  UUIDs instead of numeric indices.
- Fixed pyramid-stairs terrains (``BoxPyramidStairsTerrainCfg``,
  ``BoxInvertedPyramidStairsTerrainCfg``, and ``BoxOpenStairsTerrainCfg``)
  leaving an empty, geometry-free border at difficulty 0, where the step
  height collapses to zero. The flat border frame is now always generated as
  solid geometry flush with the ground (:issue:`1033`).
- Fixed ``HfPerlinNoiseTerrainCfg`` failing to compile at difficulty 0, where
  the target height collapses to zero and MuJoCo rejects the non-positive
  heightfield size.
- Fixed ``BoxRandomGridTerrainCfg`` producing NaN colors (and failing to build)
  at difficulty 0, where the grid height is zero and the color normalization
  divided by zero.
- Fixed the center platform z-fighting with surrounding geometry in
  ``BoxRandomGridTerrainCfg`` (grid cells were left underneath the platform) and
  ``BoxRandomSpreadTerrainCfg`` (the platform duplicated the floor surface).
- Fixed ``BoxNarrowBeamsTerrainCfg`` square platform corners protruding between
  the beams at high difficulty; the platform now shrinks to stay within the
  beams' angular coverage.
- Fixed ``BoxSteppingStonesTerrainCfg`` reconfiguring abruptly at a difficulty
  threshold, where the stone grid re-tiled as its spacing crossed an integer
  boundary, and leaving an oversized gap around the center platform. The grid is
  now difficulty-independent and the platform snaps to it as a clean island.
- Fixed ``train --video``, ``play``, and ``demo`` crashing with ``OpenGL
  platform library not loaded`` on headless Linux hosts that don't pre-set
  ``MUJOCO_GL``. The default is now applied in ``mjlab/__init__.py`` (Linux
  only) so it takes effect before mujoco's GL backend selection runs.
- Fixed motion tracking re-anchoring to a stale robot pose after a mid-episode
  motion resample. ``MotionCommand._update_command`` now calls ``sim.forward()``
  after resampling so relative body poses read the post-teleport state
  (:issue:`1068`).

Version 1.4.0 (May 26, 2026)
----------------------------

Added
^^^^^

- Added ``BuiltinPdActuator``, the implicit-integration version of
  ``IdealPdActuator``. Same interface (position + velocity targets,
  kp/kd gains), but expresses the PD as native MuJoCo ``<position>``
  and ``<velocity>`` elements so the ``implicit`` / ``implicitfast``
  integrators include the kp/kd derivatives in their velocity update.
  The actuator stays stable at gain/timestep combinations where
  explicit Python PD would diverge, which matters when you want to
  run a real motor's stiff on-board PD gains in sim. ``effort_limit``
  is enforced as a sum-clamp on the two PD terms via
  ``jnt_actfrcrange`` (or ``tendon_actfrcrange``). Supported by
  ``dr.pd_gains`` and ``dr.effort_limits``.
- Added ``mdp.projected_gravity_from_sensor``, an observation that derives
  projected gravity from a ``framezaxis`` up-vector sensor (negated) rather
  than from the root body orientation. Unlike ``mdp.projected_gravity``, it
  reflects the sensor's site frame, so it can observe IMU mounting domain
  randomization (e.g. via ``dr.site_quat``). Go1 and G1 ship an
  ``imu_upvector`` sensor for this.
- Added ``DebugVisualizer.add_box`` for drawing an axis-oriented box
  primitive, mirroring ``add_ellipsoid``. Supported by both the native
  and Viser viewers. ``size`` is the box half-extents (:issue:`992`).
- Added ``--log-root`` CLI option to ``train``, ``play``, and ``evaluate``
  scripts for choosing where training logs are stored. Defaults to
  ``logs/rsl_rl`` (unchanged behavior). Useful for directing outputs to a
  scratch disk or shared mount.
- ``RewardManager``, ``TerminationManager``, and ``MetricsManager`` now
  validate that every term function returns a tensor of shape
  ``(num_envs,)`` when evaluated, raising a clear ``ValueError``
  naming the offending term instead of silently broadcasting or crashing
  with an opaque error later during training.
- Added ``ContactSensor.primary_names`` property to expose the resolved
  primary names in the order they appear along the per-contact axis of the
  output tensors. This makes it possible to map a contact-data column back
  to the primary it belongs to (:issue:`914`).
- Added per-world mesh variant support via ``VariantEntityCfg``. Each
  world in a batched simulation can now use a different mesh asset for
  the same logical entity (e.g. world 0 holds a cube, world 1 a
  sphere). Variants are passed as a ``dict[str, Callable]`` of named
  spec callables; the optional ``assignment`` field controls how worlds
  map to variants and accepts ``None`` (uniform), a ``dict[str, float]``
  of per-variant weights, or a custom ``Callable[[int], Sequence[int]]``.
  Mesh-derived constants (collision bounds, body inertials, subtree
  mass, inverse weights) are compiled per-variant and stored as
  per-world arrays in the Warp model, so domain randomization, the
  native viewer, the offscreen renderer, and the Viser viewer all pick
  up the variant assignment automatically. Variants must share the
  same kinematic structure (same bodies, joints, joint types); only
  mesh geoms may differ. Assignment is fixed at simulation init. See
  :ref:`heterogeneous_worlds` for usage. With help from @XiangruiJiang.
- Per-world mesh variants now support per-variant materials and textures.
  Each variant can reference its own named material, which is automatically
  prefixed and scattered via ``geom_matid`` alongside the existing
  ``geom_dataid`` table. Variants without a material get ``matid = -1``.
  Contribution by @omarrayyann.
- Added ``dr.geom_matid`` to randomize which baked material each geom uses
  per environment, sampling uniformly from ``asset_cfg.material_names``.
  Contribution by @bd-pmorais.

Changed
^^^^^^^

- ``Entity`` now raises a clear error at construction when its spec contains
  more than one freejoint. An entity models a single system rooted at one
  body, so it has at most one freejoint; a second one was previously accepted
  silently and only surfaced later as a cryptic shape mismatch when writing
  root state. Model each detached floating body as its own entry in
  ``SceneCfg.entities`` instead.
- Changed ``compute_root_relative_mpkpe`` to re-anchor the reference to the
  robot's root each step, removing yaw drift as well as translation so it
  measures intrinsic body pose error.
- Changed ``compute_joint_velocity_error`` from an L2 norm to a per-joint
  RMS, so it no longer scales with the number of joints.
- Bumped ``mujoco`` to 3.8 and ``mujoco-warp`` to 3.8.0. The ``multiccd``
  enable flag was removed in mujoco 3.8 (it became default-on), so configs
  that listed ``"multiccd"`` in ``MujocoCfg.enableflags`` need to drop it.
- Camera segmentation now matches ``mujoco_warp``'s typed segmentation
  output. ``CameraSensorData.segmentation`` stores ``(object_id,
  object_type)`` pairs in shape ``[B, H, W, 2]`` instead of the previous
  legacy geom-id-only layout. Contribution by @tkelestemur.
- Sped up ``RayCaster`` post-processing by removing boolean-mask indexing
  operations and replacing them with ``masked_fill_`` plus a clamped-distance
  formulation of ``hit_pos_w`` that places misses at the world origin. This
  removes all CUDA syncs from the ray post-process, letting the CPU thread
  proceed while GPU-based sensing runs. Contribution by @bd-pdomanico.
- Bumped ``rsl-rl-lib`` from 5.0.1 to 5.2.0. This brings ``torch.compile`` support for
  PPO and Distillation, and optional std clamping and constant std in
  ``GaussianDistribution``. No code changes required on the mjlab side.
- ``TerrainEntityCfg`` debug visualization sites (environment origins,
  terrain origins, flat patches) are now off by default. Set
  ``debug_vis=True`` to re-enable them. The sites inflated ``nsite`` and
  caused a measurable slowdown in the per-step ``site_local_to_global``
  kernel (:issue:`942`).
- Task package load failures during ``mjlab`` import now print the full
  traceback (and the entry point's module path) to ``stderr`` instead of
  just the exception message, making it easier to pinpoint the source of
  import errors when running commands like ``list-envs`` (:issue:`910`).
  Contribution by @saikishor.
- Clarified ``ContactSensor`` shape conventions: per-contact fields
  (``found``, ``force``, ``torque``, ``dist``, ``pos``, ``normal``,
  ``tangent``) have shape ``[B, P * num_slots, ...]`` while per-primary
  air-time fields (``current_air_time``, ``last_air_time``,
  ``current_contact_time``, ``last_contact_time``) have shape ``[B, P]``,
  where ``P`` is the number of resolved primaries (:issue:`914`).
- Event functions now share a single ``resolve_env_ids`` helper to expand
  ``env_ids=None`` to all environments, replacing five copies of the same
  guard. ``push_by_setting_velocity`` and ``apply_external_force_torque``
  accept ``env_ids=None`` too, so they work as global-time interval terms.
  Documented when to use ``apply_external_force_torque`` (a constant,
  self-managed wrench) versus ``apply_body_impulse`` (transient, automatic
  impulses) versus ``push_by_setting_velocity`` (an instantaneous velocity
  kick).

Fixed
^^^^^

- Removed use of deprecated ``warp-lang`` symbols (``wp.context.runtime``
  and ``wp.context.Device``) that were dropped in newer ``warp-lang``
  releases, causing ``AttributeError: module 'warp' has no attribute
  'context'`` at import/runtime. mjlab now uses
  ``wp.get_cuda_driver_version()`` and ``wp.Device`` instead
  (:issue:`967`). Contribution by @rdeits.
- Fixed the tracking ``evaluate`` script scoring each metric against the
  next motion frame; the reference is now snapshotted before each step to
  match the reward.
- Fixed the tracking end-effector metrics silently scoring zero for an
  unknown body name; they now raise ``ValueError``.
- Fixed ``compute_mpkpe`` measuring root-relative instead of global error;
  it now uses the global reference ``body_pos_w`` (:issue:`1006`).
- Fixed heavy flicker in offscreen training videos on rough-terrain tasks.
  The renderer recomputed its context "neighbor" robots every frame from
  ``env_origins``, which the terrain curriculum mutates on reset, so the
  neighbor set kept changing and robots popped in and out. The neighbor
  set is now computed once and cached (:issue:`979`).
- Fixed command delay only applying to an actuator's position target.
  ``IdealPdActuator`` and ``DcMotorActuator`` also use velocity and effort, which
  arrived undelayed and out of sync; all command targets now share one delay.
  Zero-reference setups are unaffected.
- Fixed duplicate random seeds across nodes in multi-node training. The
  per-process seed offset in ``scripts/train.py`` now uses the global
  ``RANK`` instead of ``LOCAL_RANK``. Contribution by @bd-pdomanico.
- Fixed ``apply_body_impulse`` firing an impulse on the very first step (and
  the first step after every reset) instead of starting with a cooldown as
  documented. The cooldown is now sampled lazily on the first call so impulse
  timing is decorrelated from episode resets (:issue:`973`).
- Fixed ``dr.pd_gains`` and ``dr.effort_limits`` silently no-oping when
  passed an ``Operation`` object (e.g. ``dr.scale``) instead of a string.
  Both functions now accept ``Operation | str`` like every other DR event
  and raise ``ValueError`` for unsupported operations (:issue:`971`).
- Fixed ``ContactSensor`` with ``global_frame=True`` and
  ``reduce`` ∈ {``"none"``, ``"mindist"``, ``"maxforce"``} producing forces
  rotated onto the wrong axis. The contact-frame→world rotation matrix had
  its columns ordered ``[tangent, tangent2, normal]`` instead of
  ``[normal, tangent, tangent2]``, projecting the normal-force component
  onto a tangent direction. Contribution by @bd-pdomanico.
- Fixed ``extras["log"]`` entries written by reward terms (e.g. ``Metrics/*``
  values in velocity tasks) being silently discarded on any step where at
  least one environment resets. ``_reset_idx`` was clearing the dict after
  ``reward_manager.compute()`` had already populated it. The clear now
  happens at the top of ``step()`` and ``reset()`` so that all entries
  survive (:issue:`957`).
- Fixed ``ContactSensor.compute_first_contact`` and ``compute_first_air``
  occasionally missing events when a contact began or ended right at the
  last physics substep of a control step. ``current_contact_time`` /
  ``current_air_time`` accumulate in float32 and can drift a few ULPs past
  ``dt``, but the default ``abs_tol`` of ``1e-8`` sat at the noise floor
  and rejected the comparison. Raised the default to ``1e-6``, which stays
  well below typical control ``dt`` while comfortably covering float32
  accumulation noise (:issue:`933`). Contribution by @paLeziart.
- Fixed ``out_of_terrain_bounds`` using stale terrain dimensions. It read
  ``TerrainGeneratorCfg.num_cols`` directly, which is ignored in curriculum
  mode (the generator uses ``len(sub_terrains)`` columns instead), and it
  did not account for ``border_width``. The termination now reads the
  effective grid shape from ``terrain.terrain_origins`` and includes the
  border in the footprint, so robots no longer reset while still on valid
  terrain (or fail to reset after running off it) (:issue:`923`).
- ``ObservationManager`` now skips observation groups that end up with
  zero active terms (e.g. all terms set to ``None``) with a log message,
  instead of crashing later in ``torch.stack``/``torch.cat``. This lets
  a shared runner config define groups that become empty under certain
  runtime flags (e.g. model-specific terms all disabled for one variant).
  The whole group can still be set to ``None`` to disable it explicitly.
- Fixed a runtime broadcast error in ``ContactSensor`` when combining
  ``num_slots > 1`` with ``track_air_time=True`` and more than one primary.
  Air-time tracking now reduces ``found`` across slots so that a primary is
  considered in contact when any of its slots reports a match (:issue:`914`).
- Updated the ``create_new_task.ipynb`` Colab tutorial to import
  ``XmlActuatorCfg`` instead of the removed ``XmlVelocityActuatorCfg``.
  Added a regression test (``tests/test_notebooks.py``) that parses each
  notebook cell and verifies that every ``from mjlab... import X``
  reference resolves, so future renames in the mjlab public API can't
  silently rot the tutorials (:issue:`913`).
- Fixed ``ObservationManager`` silently sharing a single ``NoiseModelCfg``
  instance across observation groups that declared terms with the same
  name. ``_group_obs_class_instances`` was keyed by term name alone, so
  the last group processed in ``_prepare_terms`` overwrote earlier
  groups' instances. Symptoms included the wrong noise config being
  applied, shared per-episode state for ``NoiseModelWithAdditiveBias``
  (e.g. bias drawn from the wrong ``bias_noise_cfg``), and missed
  ``reset()`` calls for overwritten instances. Instances are now keyed
  by ``(group_name, term_name)`` so each group owns its own noise model.
- Fixed ``CurriculumManager.get_active_iterable_terms`` raising
  ``TypeError`` when a term's state was a dict. The dict branch indexed
  the output list by term name instead of appending to the local ``data``
  list. No in-tree caller currently invokes this method, so the bug was
  latent.

Version 1.3.0 (April 14, 2026)
------------------------------

Added
^^^^^

- Added ``ManagerBasedRlEnvCfg.auto_reset`` flag. When ``True`` (default),
  ``step()`` continues to reset done environments in place and returns the
  post-reset observation. When ``False``, ``step()`` skips the reset block
  and returns the terminal observation directly; the caller must call
  ``reset(env_ids=...)`` for done environments before the next ``step()``
  or a ``RuntimeError`` is raised. Enables access to the true terminal
  state for algorithms that need it. Note that mjlab's bundled ``train.py``
  uses rsl_rl's ``OnPolicyRunner``, which does not drive manual resets, so
  ``auto_reset=False`` is intended for custom training loops (:issue:`900`).
- Added ``ActuatorCfg.viscous_damping`` for passive velocity proportional
  damping (``f = -b·v``), distinct from the PD derivative gain ``damping``
  used by position and velocity actuators. Maps to ``<joint damping>`` for
  JOINT transmission and ``<tendon damping>`` for TENDON transmission.
  Defaults to ``None`` (preserves the XML value).
- Added :class:`~mjlab.managers.RecorderManager` for logging observations,
  actions, or arbitrary environment data during rollouts. Implement a
  :class:`~mjlab.managers.RecorderTerm` subclass and register it in the
  ``recorders`` dict on ``ManagerBasedRlEnvCfg``. The manager provides
  ``record_pre_reset``, ``record_post_reset``, and ``record_post_step``
  lifecycle hooks with no opinion on how data is stored.
- Added :func:`~mjlab.envs.mdp.curriculums.termination_curriculum` for
  scheduling changes to termination term parameters during training,
  matching the existing ``reward_curriculum`` pattern. Both now share a
  single internal engine with init-time validation of stage ordering,
  field existence, and param keys.
- Added ``reduce`` field to ``MetricsTermCfg``. Setting ``reduce="last"``
  reports the value from the final step of the episode rather than the
  episode mean, which is useful for binary success metrics.
- Added :class:`~mjlab.envs.mdp.actions.RelativeJointPositionAction` for
  joint position control relative to the current configuration. The target is
  ``current_pos + action * scale``, so a zero action holds the current
  configuration rather than commanding the default pose.
- Added :func:`~mjlab.envs.mdp.dr.pair_friction` for randomizing geom-pair
  friction overrides (``pair_friction`` in ``mjModel``), with an
  ``isotropic=True`` option that mirrors the symmetric tangent and roll
  axes so single-axis randomization does not leave the paired axis stale.
- Added ``STAIRS_TERRAINS_CFG`` terrain preset for progressive stair
  curriculum training and ``@terrain_preset`` decorator for composing
  terrain configurations from reusable presets.
- Added cartpole balance and swingup tasks (``Mjlab-Cartpole-Balance`` and
  ``Mjlab-Cartpole-Swingup``) with a :ref:`tutorial <tutorial-cartpole>`
  that walks through building an environment from scratch.
- Added :ref:`motion imitation <motion-imitation>` documentation with
  preprocessing instructions. The README now links here instead of the
  BeyondMimic repository, which produced incompatible NPZ files when used
  with mjlab (:issue:`777`).
- Added ``margin``, ``gap``, and ``solmix`` fields to ``CollisionCfg``
  for per geom contact parameter configuration (:issue:`766`).
- NaN guard now captures mocap body poses (``mocap_pos``, ``mocap_quat``)
  when the model has mocap bodies, enabling full state reconstruction in
  the dump viewer for fixed-base entities.
- Implemented ``ActionTermCfg.clip`` for clamping processed actions after
  scale and offset (:issue:`771`).
- Added ``qfrc_actuator`` and ``qfrc_external`` generalized force accessors
  to ``EntityData``. ``qfrc_actuator`` gives actuator forces in joint space
  (projected through the transmission). ``qfrc_external`` recovers the
  generalized force from body external wrenches (``xfrc_applied``)
  (:issue:`776`).
- Added ``RewardBarPanel`` to the Viser viewer, showing horizontal bars for
  each reward term with a running mean over ~1 second (:issue:`800`).
- Added ``per_substep`` flag to ``MetricsTermCfg`` for evaluating metrics
  once per physics substep inside the decimation loop. The per substep
  values are averaged within each environment step, so episode averages
  remain comparable to regular per step metrics.
- Added ``project-instinct/InstinctMJ`` to the research page's list of
  projects built on mjlab.
- Added a Checkpoints tab to the Viser play viewer for hot-swapping
  checkpoints without restarting. Works with local directories and W&B
  runs (:issue:`751`). Contribution by @omarrayyann.
- Added ``"segmentation"`` camera data type for per-pixel geom ID output
  alongside RGB and depth, and a multi-cube goal-conditioned lifting task
  (``Mjlab-Multi-Cube-Seg-Yam``) that uses it (:issue:`862`).
  Contribution by @pthangeda.

Changed
^^^^^^^

- Renamed the ``list_envs`` console script to ``list-envs`` for consistency
  with the other hyphenated entry points (``viz-nan``, ``export-scene``).
  Invoke via ``uv run list-envs``.
- ``ActuatorCfg.armature`` and ``ActuatorCfg.frictionloss`` now default to
  ``None`` instead of ``0.0``. ``None`` preserves the value defined in the
  XML. Previously, builtin actuators would silently overwrite XML joint and
  tendon properties with zero when these fields were not explicitly set.
  To restore the old behavior, pass ``armature=0.0`` or ``frictionloss=0.0``
  explicitly.
- Actuator delay is now configured inline on any ``ActuatorCfg`` subclass
  (e.g. ``BuiltinPositionActuatorCfg(..., delay_min_lag=2, delay_max_lag=5)``)
  instead of wrapping with ``DelayedActuatorCfg``. ``DelayedActuator``,
  ``DelayedActuatorCfg``, and ``DelayedBuiltinActuatorGroup`` are removed.
- Removed ``delay_target`` from ``ActuatorCfg``. Delay now always applies to
  the actuator's ``command_field`` automatically. Multi-target delay
  (``delay_target=("position", "velocity")``) is no longer supported.
- ``XmlPositionActuatorCfg``, ``XmlVelocityActuatorCfg``, ``XmlMotorActuatorCfg``,
  and ``XmlMuscleActuatorCfg`` are replaced by a single ``XmlActuatorCfg`` that auto
  detects the actuator type from XML. Pass ``command_field=...`` to override detection.
- Replaced the viser viewer internals with the ``mjviser`` package. Scene
  creation, mesh conversion, and overlay rendering (contacts, forces,
  inertia, tendons, joints, frames) are now provided by mjviser. The viewer
  exposes a new Visualization tab for overlay controls and a Groups tab for
  geom/site visibility. Debug visualization and warp tensor conversion remain
  in mjlab's ``MjlabViserScene`` subclass (:issue:`839`).
- In curriculum terrain mode, each terrain type now gets exactly one column
  (``num_cols`` is set to ``len(sub_terrains)``). The ``proportion`` field
  now controls robot spawning distribution across columns rather than column
  count. Random mode is unchanged (:issue:`811`).
- ``BoxSteppingStonesTerrainCfg`` stone size now decreases with difficulty,
  interpolating from the large end of ``stone_size_range`` at difficulty 0
  to the small end at difficulty 1 (:issue:`785`).
- Removed deprecated ``TerrainImporter`` and ``TerrainImporterCfg`` aliases.
  Use ``TerrainEntity`` and ``TerrainEntityCfg`` instead (:issue:`667`).
- ``Entity.clear_state()`` is deprecated. Use ``Entity.reset()`` instead.
  ``clear_state`` only zeroed actuator targets without resetting actuator
  internal state (e.g. delay buffers), which could cause stale commands
  after teleporting the robot to a new pose.
- Removed ``EntityData.generalized_force``. The property was bugged (indexed
  free joint DOFs instead of articulated DOFs) and the name was ambiguous.
  Use ``qfrc_actuator`` or ``qfrc_external`` instead (:issue:`776`).
- ``get_wandb_checkpoint_path`` now filters checkpoints server-side via the
  ``pattern`` parameter, avoiding unnecessary pagination and tolerance to
  corrupted metadata (:issue:`898`).

Fixed
^^^^^

- ``train`` and ``play`` now print a top-level usage message when invoked
  with ``-h`` / ``--help`` and no task argument, pointing users at
  ``list-envs`` and ``<TASK> --help`` (:issue:`905`).
- Fixed ghost geom filtering in the Viser viewer. Ghost geoms were selected
  by collision flags, so collision-disabled robot geoms appeared as ghosts.
  The viewer now uses visual alpha to determine which geoms to render.
- Scene now warns when an attached entity or terrain spec has non-default
  ``<option>`` fields (e.g. ``<flag contact="disable"/>``), which are
  silently dropped by ``MjSpec.attach()``. Use ``MujocoCfg`` to set
  simulation options instead (:issue:`885`).
- Fixed ``SceneEntityCfg`` names and IDs ordering mismatch when
  ``preserve_order=False`` (:issue:`876`). Contribution by @jsw7460.
- Fixed ONNX export path resolution in the velocity, manipulation, and
  tracking runners when a parent directory name contains the word
  ``"model"`` (:issue:`867`). Contribution by @gokulp01.
- ``export-scene`` now writes only referenced assets and places them
  correctly under the output directory. Previously, asset keys containing
  path traversal could write files outside the output directory, and all
  spec assets were included regardless of whether the scene XML referenced
  them (:issue:`858`).
- ``electrical_power_cost`` now uses ``qfrc_actuator`` (joint space) instead
  of ``actuator_force`` (actuation space) for mechanical power computation.
  Previously the reward was incorrect for actuators with gear ratios other
  than 1 (:issue:`776`).
- ``create_velocity_actuator`` no longer sets ``ctrllimited=True`` with
  ``inheritrange=1.0``. This caused a ``ValueError`` for continuous joints
  (e.g. wheels) that have no position range defined (:issue:`787`).
- ``write_root_com_velocity_to_sim`` no longer fails with tensor ``env_ids``
  on floating base entities (:issue:`793`).
- Joint limits for unlimited joints are now set to [-inf, inf] instead of
  [0, 0]. Previously the zero range caused incorrect clamping for entities
  with unlimited hinge or slide joints.
- Contact force visualization now copies ``ctrl`` into the CPU ``MjData``
  before calling ``mj_forward``. Actuators that compute torques in Python
  (``DcMotorActuator``, ``IdealPdActuator``) previously showed incorrect
  contact forces because the viewer ran with ``ctrl=0``
  (:issue:`786`).
- ``BoxSteppingStonesTerrainCfg`` no longer creates a large gap around the
  platform. Stones are now only skipped when their center falls inside the
  platform; edges that extend under the platform are allowed since the
  platform covers them (:issue:`785`).
- ``dr.pseudo_inertia`` no longer loads cuSOLVER, eliminating ~4 GB of
  persistent GPU memory overhead. Cholesky and eigendecomposition are now
  computed analytically for the small matrices involved (4x4 and 3x3)
  (:issue:`753`).
- Set terrain geom mass to zero so that the static terrain body does not
  inflate ``stat.meanmass``, which made force arrow visualization invisible
  on rough terrain (:issue:`734`, :issue:`537`).
- Native viewer now syncs ``qpos0`` when domain randomized, fixing incorrect
  body positions after ``dr.joint_default_pos`` randomization
  (:issue:`760`).
- ``command_manager.compute()`` is now called during ``reset()`` so that
  derived command state (e.g. relative body positions in tracking
  environments) is populated before the first observation is returned
  (:issue:`761`).
- ``RayCastSensor`` with ``ray_alignment="yaw"`` or ``"world"`` now correctly
  aligns the frame offset when attached to a site or geom with a local offset
  from its parent body. Previously only ray directions and pattern offsets were
  aligned, causing the frame position to swing with body pitch/roll
  (:issue:`775`).

Version 1.2.0 (March 6, 2026)
-----------------------------

.. admonition:: Breaking API changes
   :class: attention

   - ``randomize_field`` no longer exists. Replace calls with typed functions
     from the new ``dr`` module (e.g. ``dr.geom_friction``, ``dr.body_mass``).
   - ``EventTermCfg`` no longer accepts ``domain_randomization``. The
     ``@requires_model_fields`` decorator on each ``dr`` function takes care
     of field expansion automatically.
   - ``Scene.to_zip()`` is deprecated. Use ``Scene.write(path, zip=True)``.
   - ``RslRlModelCfg`` no longer accepts ``stochastic``, ``init_noise_std``,
     or ``noise_std_type``. Use ``distribution_cfg`` instead
     (e.g. ``{"class_name": "GaussianDistribution", "init_std": 1.0,
     "std_type": "scalar"}``). Existing checkpoints are automatically
     migrated on load.

Added
^^^^^

- Added ``"step"`` event mode that fires every environment step.
- Added ``apply_body_impulse`` event for applying transient external wrenches
  to bodies with configurable duration and optional application point offset.
- ONNX auto-export and metadata attachment for manipulation tasks (lift cube)
  on every checkpoint save, matching the velocity and tracking task behavior.
- Multi-frame ``RayCastSensor``: pass a tuple of ``ObjRef`` to ``frame`` for
  per-site raycasting with independent body exclusion. New properties:
  ``num_frames``, ``num_rays_per_frame``. New ``RayCastData`` fields:
  ``frame_pos_w`` and ``frame_quat_w``.
- ``RingPatternCfg`` ray pattern for concentric ring sampling around each
  frame.
- ``TerrainHeightSensor``, a ``RayCastSensor`` subclass that computes
  per-frame vertical clearance above terrain (``sensor.data.heights``).
  Velocity task configs now use it for ``feet_clearance``,
  ``feet_swing_height``, and ``foot_height``, replacing the previous
  world-Z proxy that was incorrect on rough terrain.
- Cloud training support via `SkyPilot <https://skypilot.readthedocs.io/>`_
  and Lambda Cloud, with documentation covering setup, monitoring, and
  cost management.
- W&B hyperparameter sweep scripts that distribute one agent per GPU
  across a multi-GPU instance.
- Contributing guide with documentation for shared Claude Code commands
  (``/update-mjwarp``, ``/commit-push-pr``).
- Added optional ``ViewerConfig.fovy`` and apply it in native viewer camera
  setup when provided.
- Native viewer now tracks the first non-fixed body by default (matching
  the Viser viewer behavior introduced in
  ``716aaaa58ad7bfaf34d2f771549d461204d1b4ba``).
- New ``dr`` module (``mjlab.envs.mdp.dr``) replacing ``randomize_field``
  with typed per-field domain randomization functions. Each function
  automatically recomputes derived fields via ``set_const``. Highlights:

  - Camera and light randomization: ``dr.cam_fovy``, ``dr.cam_pos``,
    ``dr.cam_quat``, ``dr.cam_intrinsic``, ``dr.light_pos``,
    ``dr.light_dir``. Camera and light names are now supported in
    ``SceneEntityCfg`` (``camera_names`` / ``light_names``).
  - ``dr.pseudo_inertia`` for physics-consistent randomization of
    ``body_mass``, ``body_ipos``, ``body_inertia``, and ``body_iquat``
    via the pseudo-inertia matrix parameterization (Rucker & Wensing
    2022). Replaces the removed ``dr.body_inertia`` /
    ``dr.body_iquat``.
  - ``dr.geom_size`` with automatic recomputation of ``geom_rbound``
    and ``geom_aabb`` for broadphase consistency.
  - ``dr.tendon_armature`` and ``dr.tendon_frictionloss``.
  - ``dr.body_quat``, ``dr.geom_quat``, and ``dr.site_quat`` with RPY
    perturbation composed onto the default quaternion.
  - Extensible ``Operation`` and ``Distribution`` types. Users can define
    custom operations and distributions as class instances and pass them
    anywhere a string is accepted. Built-in instances (``dr.abs``,
    ``dr.scale``, ``dr.add``, ``dr.uniform``, ``dr.log_uniform``,
    ``dr.gaussian``) are exported from the ``dr`` module.
  - ``dr.mat_rgba`` for per-world material color randomization. Tints
    the texture color, useful for randomizing appearance of textured
    surfaces. Material names are now supported in ``SceneEntityCfg``
    (``material_names``).
  - Fixed ``dr.effort_limits`` drifting on repeated randomization.
  - Fixed ``dr.body_com_offset`` not triggering ``set_const``.

- ``export-scene`` CLI script to export any task scene or asset_zoo entity
  (``g1``, ``go1``, ``yam``) to a directory or zip archive for inspection
  and debugging.

- ``yam_lift_cube_vision_env_cfg`` now randomizes cube color (``dr.geom_rgba``)
  on every reset when ``cam_type="rgb"``.

- The native viewer now reflects per-world DR changes to visual model fields
  on each reset. Geom appearance, body and site poses, camera parameters,
  and light positions are all synced from the GPU model before rendering.
  Inertia boxes (press ``I``) and camera frustums (press ``Q``) update
  correctly when the corresponding fields are randomized. See
  :doc:`randomization` for viewer-specific caveats.

- ``MaterialCfg.geom_names_expr`` for assigning materials to geoms by
  name pattern during ``edit_spec``.

- ``TerrainEntityCfg`` now exposes ``textures``, ``materials``, and
  ``lights`` as configurable fields (previously hardcoded). Set
  ``textures=()``, ``materials=()`` to use flat ``dr.geom_rgba``
  instead of the default checker texture.

- ``DebugVisualizer`` now supports ellipsoid visualization via
  ``add_ellipsoid``.

- Interactive velocity joystick sliders in the Viser viewer. Enable the
  joystick under Commands/Twist to override velocity commands with manual
  sliders for ``lin_vel_x``, ``lin_vel_y``, and ``ang_vel_z``
  (`#666 <https://github.com/mujocolab/mjlab/issues/666>`_).
- Per-term debug visualization toggles in the Viser viewer. Individual
  command term visualizers (e.g. velocity arrows) can now be toggled
  independently under Scene/Debug Viz.
- Viewer single-step mode: press RIGHT arrow (native) or click "Step"
  (Viser) to advance exactly one physics step while paused.
- Viewer error recovery: exceptions during stepping now pause the viewer
  and log the traceback instead of crashing the process.
- Native viewer runs forward kinematics while paused, keeping
  perturbation visuals accurate.
- Viewer speed multipliers use clean power-of-2 fractions (1/32x to 1x).

- Visualizers display the realtime factor alongside FPS.

- ``joint_torques_l2`` now respects ``SceneEntityCfg.actuator_ids``,
  allowing penalization of a subset of actuators instead of all of them
  (`#703 <https://github.com/mujocolab/mjlab/pull/703>`_). Contribution by
  `@saikishor <https://github.com/saikishor>`_.

- Terrain is now a proper ``Entity`` subclass (``TerrainEntity``). This
  allows domain randomization functions to target terrain parameters
  (friction, cameras, lights) via ``SceneEntityCfg("terrain", ...)``.
  ``TerrainImporter`` / ``TerrainImporterCfg`` remain as aliases but will be
  deprecated in a future version.
- Added ``upload_model`` option to ``RslRlBaseRunnerCfg`` to control W&B model
  file uploads (``.pt`` and ``.onnx``) while keeping metric logging enabled
  (`#654 <https://github.com/mujocolab/mjlab/pull/654>`_).
- ``Scene.write(output_dir, zip=False)`` exports the scene XML and mesh
  assets to a directory (or zip archive). Replaces ``Scene.to_zip()``.
- ``Entity.write_xml()`` and ``Scene.write()`` now apply XML fixups
  (empty defaults, duplicate nested defaults) and strip buffer textures
  that ``MjSpec.to_xml()`` cannot serialize.
- ``fix_spec_xml`` and ``strip_buffer_textures`` utilities in
  ``mjlab.utils.xml``.

Changed
^^^^^^^

- Native viewer now syncs ``xfrc_applied`` to the render buffer and draws
  arrows for any nonzero applied forces. Mouse perturbation forces are
  converted to ``qfrc_applied`` (generalized joint space) so they coexist
  with programmatic forces on ``xfrc_applied`` without conflict.
- ``ViewerConfig.OriginType.WORLD`` now configures a free camera at the
  specified lookat point instead of auto tracking a body. A new ``AUTO``
  origin type (now the default) preserves the previous auto tracking
  behavior.
- Upgraded ``rsl-rl-lib`` from 4.0.1 to 5.0.1. ``RslRlModelCfg`` now
  uses ``distribution_cfg`` dict instead of ``stochastic`` /
  ``init_noise_std`` / ``noise_std_type``. Existing checkpoints are
  automatically migrated on load.
- Reorganized the Viser Controls tab into a cleaner folder hierarchy:
  Info, Simulation, Commands, Scene (with Environment, Camera, Debug Viz,
  Contacts sub-folders), and Camera Feeds. The Environment folder is
  hidden for single-env tasks and the Commands folder is hidden when no
  command terms are active.
- Viser camera tracking is now enabled by default so the agent stays in
  frame on launch.
- Self collision and illegal contact sensors now use ``history_length`` to
  catch contacts across decimation substeps. Reward and termination functions
  read ``force_history`` with a configurable ``force_threshold``.
- Replaced the single ``scale`` parameter in ``DifferentialIKActionCfg`` with
  separate ``delta_pos_scale`` and ``delta_ori_scale`` for independent scaling
  of position and orientation components.
- Improved offscreen multi environment framing by selecting neighboring
  environments around the focused env instead of first N envs.
- Tuned tracking task viewer defaults for tighter camera framing.
- Disabled shadow casting on the G1 tracking light to avoid duplicate
  stacked shadows when robots are close.

Fixed
^^^^^

- Fixed actuator target resolution for entities whose ``spec_fn`` uses
  internal ``MjSpec.attach(prefix=...)``
  (`#709 <https://github.com/mujocolab/mjlab/issues/709>`_).
- Fixed viewer physics loop starving the renderer by replacing the single
  sim-time budget with a two-clock design (tracked vs actual sim time).
  Physics now self-corrects after overshooting, keeping FPS smooth at all
  speed multipliers.
- Bundled ``ffmpeg`` for ``mediapy`` via ``imageio-ffmpeg``, removing the
  requirement for a system ``ffmpeg`` install. Thanks to
  `@rdeits-bd <https://github.com/rdeits-bd>`_ for the suggestion.
- Fixed ``height_scan`` returning ~0 for missed rays; now defaults to
  ``max_distance``. Replaced ``clip=(-1, 1)`` with ``scale`` normalization
  in the velocity task config. Thanks to `@eufrizz <https://github.com/eufrizz>`_
  for reporting and the initial fix (`#642 <https://github.com/mujocolab/mjlab/pull/642>`_).
- Fixed ghost mesh visualization for fixed-base entities by extending
  ``DebugVisualizer.add_ghost_mesh`` to optionally accept ``mocap_pos`` and
  ``mocap_quat`` (`#645 <https://github.com/mujocolab/mjlab/pull/645>`_).
- Fixed viser viewer crashing on scenes with no mocap bodies by adding
  an ``nmocap`` guard, matching the native viewer behavior.
- Fixed offscreen rendering artifacts in large vectorized scenes by applying
  a render local extent override in ``OffscreenRenderer`` and restoring the
  original extent on close.
- Fixed ``RslRlVecEnvWrapper.unwrapped`` to return the base environment,
  ensuring checkpoint state restore and logging work correctly when wrappers
  such as ``VideoRecorder`` are enabled.

Version 1.1.1 (February 14, 2026)
---------------------------------

Added
^^^^^

- Added reward term visualization to the native viewer (toggle with ``P``) (`#629 <https://github.com/mujocolab/mjlab/pull/629>`_).
- Added ``DifferentialIKAction`` for task-space control via damped
  least-squares IK. Supports weighted position/orientation tracking,
  soft joint-limit avoidance, and null-space posture regularization.
  Includes an interactive viser demo (``scripts/demos/differential_ik.py``) (`#632 <https://github.com/mujocolab/mjlab/pull/632>`_).

Fixed
^^^^^

- Fixed ``play.py`` defaulting to the base rsl-rl ``OnPolicyRunner`` instead
  of ``MjlabOnPolicyRunner``, which caused a ``TypeError`` from an unexpected
  ``cnn_cfg`` keyword argument (`#626 <https://github.com/mujocolab/mjlab/pull/626>`_). Contribution by
  `@griffinaddison <https://github.com/griffinaddison>`_.

Changed
^^^^^^^

- Removed ``body_mass``, ``body_inertia``, ``body_pos``, and ``body_quat``
  from ``FIELD_SPECS`` in domain randomization. These fields have derived
  quantities that require ``set_const`` to recompute; without that call,
  randomizing them silently breaks physics (`#631 <https://github.com/mujocolab/mjlab/pull/631>`_).
- Replaced ``moviepy`` with ``mediapy`` for video recording. ``mediapy``
  handles cloud storage paths (GCS, S3) natively (`#637 <https://github.com/mujocolab/mjlab/pull/637>`_).

.. figure:: _static/changelog/native_reward.png
   :width: 80%

Version 1.1.0 (February 12, 2026)
---------------------------------

Added
^^^^^

- Added RGB and depth camera sensors and BVH-accelerated raycasting (`#597 <https://github.com/mujocolab/mjlab/pull/597>`_).
- Added ``MetricsManager`` for logging custom metrics during training (`#596 <https://github.com/mujocolab/mjlab/pull/596>`_).
- Added terrain visualizer (`#609 <https://github.com/mujocolab/mjlab/pull/609>`_). Contribution by
  `@mktk1117 <https://github.com/mktk1117>`_.

.. figure:: _static/changelog/terrain_visualizer.jpg
   :width: 80%

- Added many new terrains including ``HfDiscreteObstaclesTerrainCfg``,
  ``HfPerlinNoiseTerrainCfg``, ``BoxSteppingStonesTerrainCfg``,
  ``BoxNarrowBeamsTerrainCfg``, ``BoxRandomStairsTerrainCfg``, and
  more. Added flat patch sampling for heightfield terrains (`#542 <https://github.com/mujocolab/mjlab/pull/542>`_, `#581 <https://github.com/mujocolab/mjlab/pull/581>`_).
- Added site group visualization to the Viser viewer (Geoms and Sites
  tabs unified into a single Groups tab) (`#551 <https://github.com/mujocolab/mjlab/pull/551>`_).
- Added ``env_ids`` parameter to ``Entity.write_ctrl_to_sim`` (`#567 <https://github.com/mujocolab/mjlab/pull/567>`_).

Changed
^^^^^^^

- Upgraded ``rsl-rl-lib`` to 4.0.0 and replaced the custom ONNX
  exporter with rsl-rl's built-in ``as_onnx()`` (`#589 <https://github.com/mujocolab/mjlab/pull/589>`_, `#595 <https://github.com/mujocolab/mjlab/pull/595>`_).
- ``sim.forward()`` is now called unconditionally after the decimation
  loop. See :ref:`faq-sim-forward` for details (`#591 <https://github.com/mujocolab/mjlab/pull/591>`_).
- Unnamed freejoints are now automatically named to prevent
  ``KeyError`` during entity init (`#545 <https://github.com/mujocolab/mjlab/pull/545>`_).

Fixed
^^^^^

- Fixed ``randomize_pd_gains`` crash with ``num_envs > 1`` (`#564 <https://github.com/mujocolab/mjlab/pull/564>`_).
- Fixed ``ctrl_ids`` index error with multiple actuated entities (`#573 <https://github.com/mujocolab/mjlab/pull/573>`_).
  Reported by `@bwrooney82 <https://github.com/bwrooney82>`_.
- Fixed Viser viewer rendering textured robots as gray (`#544 <https://github.com/mujocolab/mjlab/pull/544>`_).
- Fixed Viser plane rendering ignoring MuJoCo size parameter (`#540 <https://github.com/mujocolab/mjlab/pull/540>`_).
- Fixed ``HfDiscreteObstaclesTerrainCfg`` spawn height (`#552 <https://github.com/mujocolab/mjlab/pull/552>`_).
- Fixed ``RaycastSensor`` visualization ignoring the all-envs toggle (`#607 <https://github.com/mujocolab/mjlab/pull/607>`_).
  Contribution by `@oxkitsune <https://github.com/oxkitsune>`_.

Version 1.0.0 (January 28, 2026)
--------------------------------

Initial release of mjlab.
