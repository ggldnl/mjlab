=========
Changelog
=========

Upcoming version (not yet released)
-----------------------------------

Added
^^^^^

- The parkour demo takes the hand-over controls the transition scripts have, under the same
  names, each overriding ``config.yml`` for one run: ``--hold-back`` (metres short of the
  pose a skill needs that the walk stops, which ``tests/stage.py`` calls ``fire_at``),
  ``--window`` (seconds the bridge gets), ``--blend-steps``, ``--count-in``, ``--entries``
  (which clip frame to enter a skill at, by name), ``--tolerances.*`` (per channel arrival
  tolerance), and a checkpoint flag per skill plus ``--bridge-checkpoint``. ``--bridge``
  already chose the architecture.

  Two of those are new behaviour rather than a newly exposed number. ``blend_steps`` ramps
  out of the parting policy's last action at every switch, which is the action seam;
  ``count_in`` marches the entering tracker's clip up to its entry frame during the crossing
  so it arrives there as control changes, which is the observation seam and leaves the
  rewind afterwards with nothing to do. ``count_in`` is on. ``blend_steps`` defaults to 0
  here and not to stage's 8: the climb loses the box at any ramp at all, while a course of
  hurdles clears at 8. The numbers are in ``config.yml``.

- ``demos/parkour/approach.py``, the geometry the parkour demo used to carry inside its
  controller: given an obstacle's pose, where the robot has to stand for a traversal skill
  to work, and where the walk stops short of that so the bridge can cover the rest. The box
  is a solve rather than a distance, because the climb's reference carries its own obstacle
  rigidly and only lines up at one pose, and it is solved once per obstacle instead of every
  control step.

- ``bridges/diffusion`` is no longer a stub. It is BeyondMimic's second stage over this
  repo's tracker rollouts: a denoising diffusion model fitted offline on 64 tick windows of
  the shared corpus, carrying states and actions together, and steered at inference by
  pinning the start of the window and pulling its deadline column onto the target. Nothing
  about a crossing enters training, so one frozen model answers crossings it was never
  trained on and a new kind of demand is a new cost function rather than a new run.

  Registered as ``Mjlab-G1-Diffusion-Bridge``, logging to ``logs/rsl_rl/g1_diffusion_bridge``
  and running in the imitation bridge's arena so the two are scored on the same windows by
  the same code. Train it with::

      uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.train

  ``uv run train Mjlab-G1-Diffusion-Bridge`` refuses with that message: there is no reward
  and no episode behind this architecture. Everything that loads a bridge by task id drives
  it unchanged, ``uv run play`` and ``bridges/evaluate.py --bridge diffusion`` included,
  because ``DiffusionRunner`` answers the same loader interface an rsl_rl runner does.
  Sampling is an inference choice and lives on ``diffusion/policy.py``'s ``ControlCfg``:
  ``replan_every``, ``sample_steps``, ``strength`` and ``hold``.

- ``body_pos_b``, every body's position in the root frame, is recorded again by
  ``dataset.entry_context`` and loaded onto ``Dataset``. The column was in the corpus on
  disk and in neither the recorder nor the loader, so ``selector/build.py`` read ``None``
  for it and dropped every entry it was about to write. The diffusion bridge predicts it
  beside the joint angles, which is what BeyondMimic's representation ablation asks for: a
  small angle error at the hip is centimetres at the foot and an angle loss cannot see that.

- Every transition script takes a checkpoint flag per skill, named after the skill rather
  than after its role: ``tests.transitions.walk2kick`` takes ``--walk-checkpoint`` and
  ``--kick-checkpoint``, ``walk2front_kick`` takes ``--front-kick-checkpoint``. Each
  defaults to the newest checkpoint under that skill's own experiment, which is what
  ``find_checkpoint`` already picked and what has silently picked the wrong run before, so
  naming one pins a comparison to a particular pair of policies. The bridge's own is
  ``--bridge-checkpoint``, renamed from ``--checkpoint`` now that it is one of three.
  ``tests.stage.config_for`` builds the flags from the couple, so a new transition script
  gets them by naming its two actors.

- The panel of every transition script carries a ``bridge`` dropdown, which swaps which
  architecture crosses. ``--bridge`` is now where it starts rather than the whole choice:
  every architecture with a checkpoint under its experiment is loaded, and the dropdown
  offers what loaded. One with no code yet, or nothing under its log directory, is dropped
  with a line saying which, since an untrained architecture is the normal state of one of
  them. ``--bridges`` narrows the set, and ``--bridges "('imitation',)"`` is the old
  behaviour.

  The three share one arena because they differ in their window term and their observation
  and in nothing else, so ``tests.stage.arena`` gives each its own observation group and
  builds the command from the most specific window among them, which serves the rest
  because the specific one subclasses the general one. Two that subclassed it in
  incompatible ways would be refused by name rather than served by whichever came first.
  The swap lands at the next hand-over, not immediately: one architecture's opening and
  another's follow through are not a crossing anybody performed. The Active line names
  whoever is driving, so the dropdown's position is not something to remember.

  Checkpoints are resolved before the simulator is built, which is also why a missing one
  is now a message in the first second of a run rather than after a minute of arena.

- The panel of every transition script carries a ``switch step`` slider and a
  ``fire on the step above`` checkbox, which pin the control step the hand-over fires on.
  Off by default: choosing the moment is what the button is for. On, the crossing repeats
  at the same instant every episode, which together with the ``entry`` slider is what a
  formal comparison needs, since the entry fixes the state being crossed to and the instant
  fixes the state being crossed from. A button pressed by hand lands on a different stride
  every time, so the bridge starts from a different velocity and phase and two runs differ
  in more than what is being compared. ``--auto N`` sets the slider and turns the checkbox
  on, so a run can start pinned.

  ``tests.transitions.walk2kick_robust_test`` had both controls and has lost them to
  ``tests.stage``, which is where they belonged: it is down to the one dropdown that swaps
  which kick catches the robot. Its ``--switch-step`` and ``--max-switch-step`` are gone,
  and ``--auto`` does that job for every couple.

Fixed
^^^^^

- The parkour walk stopped going forward after a tilted obstacle and crabbed sideways
  instead, which is what a robot that has stopped mid-walk looks like from outside.
  ``go_to`` drove at the mark along a diagonal and split the error in the approach frame, so
  a robot level with the mark but half a metre to the side of it had no forward distance to
  cover: the forward command read zero and the whole error was left to the sideways command,
  which the walk caps at 0.35 m/s and barely acts on. A traversal over a turned obstacle
  lands the robot in exactly that state, which is why it never happened before the first one.

  ``go_to`` now follows a carrot a lookahead ahead on the approach line, so it converges onto
  the line and arrives pointing down it, and it splits the error in the frame it is being
  pointed at rather than in the approach frame. Over five hurdles turned to the configured
  limit: 0 stalled steps out of 2835, against stalls on every leg before. Fixing the approach
  also fixed the climb, which now starts from a robot that arrived rather than one still
  crabbing: the demo clears a box and a hurdle end to end for the first time.

- ``selector.build`` dropped every entry of any skill whose window sits at the opening of
  its clip, which was climb, pass and walk: all three built zero entries, and the parkour
  demo then refused to start with a message telling you to record them, which would not
  have helped. Recording discards the ``settle`` steps after every reset and ``Dataset.frame``
  keeps the absolute step index, so a rollout's earliest recorded row is step 25 and not
  step 0. ``run_up`` only guarded against a negative frame, so for any entry within
  ``settle + segment_steps`` of the start it looked up frames that were never recorded and
  returned None.

  The medoid is now searched over the rows that can supply a run-up rather than picked
  first and discarded after, which is what threw whole windows away while thousands of
  usable rows sat beside them. climb, pass and walk get their first entries, jump and
  front_kick gain one each, and a slice with no usable row is reported with the cause and
  the two remedies instead of one line per dropped entry.

- ``demos.parkour.ENTRIES`` names the clip frame to enter each skill at rather than an index
  into the entry table. An index is not stable across a rebuild: adding one entry to the
  jump shifted every later one, so the demo's measured choice of frame 87 would have
  silently become frame 78.

- ``tests/entry_tolerances.py`` frames are back in step with the rebuilt table. Only kick
  124 is measured and it did not move; the rest are provisional baselines.

- The parkour hurdle was taller and longer than the jump can clear. Measured from the jump's
  own entry state, the lowest foot peaks at 0.185 m and is back down 0.7 m later, against a
  bar that was 0.20 m tall and 1.00 m long: the robot either tripped on it or landed on top
  of it, every time. It is now 0.10 m by 0.40 m, sized off that trace. The failure was
  invisible before because a traversal ended when its clip ran out, so a jump that hit the
  bar still counted as one that had happened.

- The parkour lane did not leave room for the hand-over. With ``first_run_up`` at 1.0 m the
  first hold point solved to x = -0.04, behind the start line, so the robot reversed into its
  own first switch. The run up and the spacing now clear the metre a hold point needs.

- The parkour demo entered the jump at the entry the selector rates easiest, which is the
  one standing still at the start of the crouch, and the bridge cannot reproduce a crouch
  pose it has not begun. Over one hurdle it arrived 0.135 m off and the robot fell; entered
  at the third entry instead it arrives 0.039 m off and clears. Reachability is not the same
  question as which phase of a skill to resume at.

- ``--bridge distillation`` works on the transition scripts and the parkour demo.
  ``tests.stage.arena`` built its aimed command by copying every field of the chosen
  architecture's window config into ``AimedCfg``, a hard-coded subclass of
  ``BridgeCommandCfg``: the distillation bridge's ``MaskedBridgeCommandCfg`` carries four
  more fields, so the copy was refused, and had it gone through the arena would have held a
  plain ``BridgeCommand`` with no answer for the observation term the student reads.
  ``tests.stage.aimed_cfg`` now derives the config and the command from whatever the
  architecture declares, reading the command class off ``build``'s return annotation, with
  ``Aimed`` first in the bases so its overrides still win and ``isinstance(command, Aimed)``
  still holds. The distillation arena's bridge observation is 408 wide against imitation's
  180, which is the three keyframe slots, and every bit of them reads off: a window aimed
  from outside has no recorded interior, so what the student gets is the deployment pattern
  it was trained for.

  A checkpoint saved before ``TRACKING_ITERATIONS`` holds a teacher and no student and is
  still refused by name.

Changed
^^^^^^^

- The parkour demo's controller is a plan rather than a phase machine. A course is compiled
  once into a flat list of actions, ``go_to``, ``cross`` and a traversal per obstacle, and
  the run is an index walking down that list; every action answers the same four questions
  and the loop asks nothing else. The walk is no longer asked to be accurate: it stops
  ``approach.hold_back`` short of the pose a skill needs and the bridge covers the rest
  inside a fixed ``approach.window_s``, which replaces the per step crossing solve, the
  alignment gate and the entry effort window sizing.

  A traversal now ends on the robot rather than on its clip. Both skills leave the ground
  and come back, so the lowest foot rising past ``traverse.lift_height`` arms the action and
  returning below ``traverse.land_height`` upright ends it. There is no bridge on the way
  out: a traversal ends standing at about zero velocity, which is inside the walk's own
  initiation set.

- The imitation bridge's actor and critic are LSTM rather than plain MLP
  (``Mjlab-G1-Imitation-Bridge``). A single frame of root and joint state does not say which
  foot is loaded or where in the stride the robot is, and no observation term carries
  contact, so the memory is there to infer phase. Config only: the MLP trunk keeps its
  ``hidden_dims`` and sits behind the recurrent encoder. Backpropagation through time
  reaches ``num_steps_per_env`` back, 0.48 s at the bridge's rate. Checkpoints trained
  before this change do not load into the new model.

- The bridge is now one architecture among several. ``bridge/`` became ``bridges/``, a
  package with one sub-package per architecture and the shared corpus beside them. What
  used to be the whole bridge is ``bridges/imitation``, under ``Mjlab-G1-Imitation-Bridge``
  and logging to ``g1_imitation_bridge``. ``bridges/diffusion`` and
  ``bridges/distillation`` are stubs: selectable, and refused by ``bridges.resolve`` with a
  message naming the package to write.

  ``logs/rsl_rl/g1_bridge`` was renamed to ``logs/rsl_rl/g1_imitation_bridge`` with its
  runs intact, so the newest checkpoint is still what every script picks up.

- Every script that drives a bridge takes ``--bridge``, which is a choice between the
  registered architectures: the transition scripts through ``tests.stage``, the parkour
  demo, ``tests.handoff``, ``tests.end2end.bridge_delivery`` and
  ``tests.end2end.error_shapes``. The name is the whole selection. ``bridges.resolve``
  turns it into the task id a policy loads from, the experiment name its newest checkpoint
  is found under, and the env config the arena is built on, so the two skills either side
  of a hand-over are untouched by it and two architectures are comparable on one
  transition. An architecture with no code behind it is refused before the simulator
  starts.

  ``tests.end2end.bridge_delivery`` and ``tests.end2end.error_shapes`` took a required
  checkpoint path, one of them hard-coded to a run under the old log directory. Both now
  default to the newest checkpoint of the chosen architecture.

- The corpus moved from ``bridge/datasets/`` to ``bridges/datasets/``, since what a bridge
  is asked to cross does not depend on how its policy is produced. Two architectures read
  the same windows, which is what makes their scores comparable.

- ``bridges/datasets/skills.py`` is deleted, along with ``SKILLS_DATASET``. It built the
  corpus out of the skill pool's own rollouts, which trained and served the bridge from
  one distribution and made it a function of the pool; ``tracker.py`` replaced it and had
  been the default for some time. ``tests.resume`` read its ``SKILLS`` roster for a skill
  name to task id map and now reads ``skills.SKILLS``, which is the same map without the
  second copy, taking each experiment name off the registered task instead of a hand
  written tuple.

- ``tests/entry_tolerances.py`` is back in step with the entry table. Every frame it
  registered was one or two off what ``data/selector/entries.npz`` holds, so aiming at
  most entries raised "No bridge tolerance profile". Five of the six skills in the table
  were affected, not just the kick: only ``kick`` 98, 115 and 140 and ``front_kick`` 40
  and 56 still resolved. The table is written by ``selector.build`` and is not in git, so
  it had been rebuilt and every entry had moved.

  Frames are now ``jump`` 68, 76, 87, 96, 106; ``kick`` 98, 106, 115, 124, 133, 140;
  ``front_kick`` 40, 56; ``punch_combo`` 37, 44, 51; ``pass`` 34; ``walk`` 55. The kick's
  measured entry is frame 124, between the 123 and 125 the arm insensitivity was measured
  at, so both measurements still describe it.

  Keying on the recorded frame is what made this an error rather than frame 123's profile
  quietly applied to frame 124, and it stays. The refusal now prints the frames the file
  does have and says a rebuilt table is the usual cause. The two tests that pinned a kick
  frame read one out of ``ENTRY_TOLERANCES`` instead, so the next rebuild is one edit.

- ``tests.transitions.walk2kick_robust_test`` takes every flag
  ``transitions.walk2kick`` does. It parsed its own ``--variants`` and ``--checkpoints``
  and then handed the same command line to ``stage.main``, which parses a different
  config, so ``--viewer``, ``--auto`` and ``--entry`` were rejected as unrecognized before
  the arena was built: the dropdown could only ever be watched on whichever entry the
  selector happened to pick. It now takes its own flags and passes the rest down, so a
  comparison can pin the entry both kicks are judged on. An unknown flag is still an
  error, from the second parse rather than the first.

  ``transitions.walk2kick`` says the variant script exists, which nothing did before.

- The bridge is back to commit 912afa51 and both later generations are parked. The
  reference tracker, which put the spliced reference in the observation and widened it to
  463, is ``bridge_experimental`` under ``Mjlab-G1-Bridge-Experimental``, logging to
  ``g1_bridge_experimental``. The segment work that followed 912afa51, which took the
  observation to 327, is in ``git stash@{0}``. ``bridge`` and ``Mjlab-G1-Bridge`` are the
  committed 180 wide task again: ``26 + 2J`` of command on 96 of proprioception.

  All three generations wrote to ``logs/rsl_rl/g1_bridge``, so telling their checkpoints
  apart needs the observation width in ``actor_state_dict``. The six tracker runs are
  deleted. What loads now is ``2026-09-12_00-10-00_clock``,
  ``2026-09-12_10-49-02_short-windows`` and ``2026-09-12_15-18-03``, all 180 wide. Prefer
  ``2026-09-12_00-10-00_clock/model_2800.pt``: ``short-windows`` trained on 0.1 to 0.2 s
  windows against this config's 0.3 to 1.2, and ``15-18-03`` diverged after iteration
  1600, taking policy std from 0.209 to 1.94.

  Callers came back with it. ``tests.stage.aim`` and ``demos.parkour.Bridge.aim`` write
  ``command.target`` and open the window with ``duration_s`` alone, the end of a window is
  ``out_of_patience``, and ``duration_s_range`` is read off the config. ``selector.build``
  reads ``body_pos_b`` off the dataset rather than assuming it, so a corpus from either
  bridge loads.

  Seven entries below describe the parked work and not the code in the tree: the three
  value clock, the corpus rate check, arrival paid once at the hand-over, ``mdp.approach``,
  the terminal segment, ``body_pos_b`` with ``Segments.draw`` bounds, and ``entropy_coef``
  0.001. This bridge has a two value clock, no rate check, arrival on a running best, no
  approach term, no segment, and ``entropy_coef`` 0.005. Restore ``stash@{0}`` to make them
  true again.

- The bridge clock carries the window length as well, so it is three values: seconds left,
  fraction spent, and the seconds the crossing was given. The total is recoverable from the
  other two, since seconds left is the total times one minus the fraction, but that
  division degenerates exactly at the hand-over where both go to zero, which is the tick
  the precision is wanted at. The observation is ``12 + S * (15 + 2J)`` wide.

  All three are seconds or dimensionless, never control ticks, which is what they always
  were: a tick count is a property of the decimation rather than of the task, so a policy
  conditioned on one would mean something different the moment the simulator was configured
  differently.

- The bridge refuses a corpus recorded at a rate the environment does not step at. ``fps``
  came from the dataset while the simulator stepped at its own rate and nothing compared
  them, so a mismatch would have converted every window between seconds and ticks with the
  wrong divisor: the clock would have reported a number the simulator did not agree with
  and every duration in the config would have quietly meant something else. They agree
  today at 50 Hz; now they have to.

- The bridge arrival is paid once, at the instant control transfers, and nowhere else.
  ``mdp.arrival`` used to pay the improvement on a running best, clamped at zero, which
  summed over an episode to the best arrival the crossing ever reached whenever it reached
  it. Leaving the target cost nothing because the clamp floored it, returning and beating
  the old best paid again, so a trajectory sweeping through the target region twice drew
  two samples and kept the better, and standing in the right place at the hand-over was
  worth nothing at all because time appeared nowhere in the sum. The result is visible: the
  robot arrives early, adjusts, and hands over mid-fidget, a stutter in what should read as
  one motion.

  ``mdp.approach`` takes over the dense half, paying the change in the arrival score every
  step with no clamp and against the previous tick rather than a running maximum. That is
  potential based shaping in the sense of Ng, Harada and Russell 1999: it telescopes to the
  score at the end minus the score at the start, leaves the optimal policy unchanged, and
  makes a round trip out and back cost exactly what it pays.

  ``patience_scale`` drops from 1.5 to 1.0, because "the last instant" names nothing while
  the episode runs half again as long as the window. The episode is now exactly the window,
  and the policy is told when that ends by the clock it already reads.

  ``arrived``, ``fixed_arrived`` and ``score`` now describe the hand-over instead of the
  best moment of the window. They will read lower, and that is the measurement changing
  rather than the policy getting worse. ``arrival_s`` is kept and is paid nothing: read
  against ``window_s`` it is the overshoot detector, and a robot doing nothing scores 0.32.

- The bridge sees the end of its window, not just its last state. ``segment_steps`` frames
  of the recorded crossing reach the observation, ``segment_samples`` of them evenly
  spaced, the last being the target. A target is a state, and a state says where to be
  without saying what the robot will be doing when it gets there; these are the frames the
  entering skill is about to continue, so a bridge that can see them can arrive moving the
  way that motion moves rather than arriving at a pose and stopping.

  Input only. Nothing is scored against the segment and the reward is unchanged. The last
  sampled frame is the target, so the previous layout is a suffix of this one and anything
  indexing from the end still lands where it did. The observation is
  ``11 + S * (15 + 2J)`` wide, 230 at the defaults, so earlier checkpoints cannot be
  resumed. ``place`` and ``open_window`` take a ``segment`` keyword; omitting it repeats
  the target, which is exactly the observation this task had before.

  This is what survives of a larger attempt. Scoring the segment instead of the target,
  with a fixed hand-over and a duration curriculum, was tried over three runs and never
  trained: the best of them had ``merge_error`` rising monotonically from 0.040 at
  iteration 0 to 0.117 at 800, against 0.033 for a robot that does nothing, with every
  objective term flat and the regularizers running 22x and 641x their old per tick cost.
  That work is reverted. The segment as an input is the part worth keeping.

- ``entropy_coef`` drops from 0.005 to 0.001. At 0.005 a run diverged: once the task
  gradient flattened the entropy bonus was the largest term left, and over iterations 1800
  to 2075 the policy std went 0.209 to 0.362 while every arrival metric came apart with it.
  Safe to lower, because exploration was never the binding constraint here; reopening std
  from 0.177 to 0.30 mid-run walked straight back to the identical optimum. Watch
  ``Policy/mean_std``.

- The bridge corpus records body positions in the root frame, and ``Segments.draw`` accepts
  ``min_steps`` and ``max_steps``. Neither is read by the bridge today. Both are there for
  the next attempt at scoring a trajectory, and the corpus already carries the column.

- The selector records the ten frames before each entry, so a live hand-over can hand the
  bridge the entering skill's own run-up rather than a single pose. An entry whose
  recording does not reach back that far is dropped. Rebuild the table with
  ``selector.build``.

- The bridge tolerance curriculum is replaced. It was one multiplier range shared by all
  eight channels, log-uniform, both bounds sliding from (5, 10) to (0.5, 4) on the
  environment step counter. Three things were wrong with it. The eight draws were
  independent, so half of all windows asked the arms for more precision than the legs,
  which is the case where a perfect arm is worth nothing. ``arm_joint_pos`` ended up asked
  for 0.025 to 0.2 rad against a kick that shrugs at 0.8, up to 32 times tighter than any
  consumer has wanted, while carrying the joint highest reward weight. And it tightened on
  a step counter whether or not anything was being learned, which across four runs it was
  not: the worst channel sat at 5.5 times its limit for thousands of iterations.

  In its place, each channel has a band in multiples of the baseline. Root and legs use
  ``core_band``, 4x down to 0.6x; the arms use ``support_band``, 16x down to 4x. The two
  do not overlap and a draw never leaves its row, so the arms are never asked for more
  precision than the legs, for any window, at any point. Each window picks one channel to
  be strict and relaxes the other seven toward the wide end by ``focus_relax``, so a
  window asks one question; the policy is told which, for free, since the observation
  already carries the requested tolerances. A channel's band position moves only when the
  windows that focused it are met outside ``success_band``, which is Florensa's reverse
  curriculum per channel: above the upper rate the channel is solved and is asked for
  more, below the lower it is past what the policy can do and is asked for less.

  The baseline ``Tolerances`` is unchanged and is now only the unit, not the requirement.
  Two of its values no longer cover the kick, measured at 8 directions rather than 3, and
  the band floor of 0.6x covers both. ``tolerance_initial_range``, ``tolerance_final_range``
  and ``tolerance_steps`` are gone. ``level_<channel>`` and ``focus_rate_<channel>`` are
  logged. The band positions are evidence rather than a step count, so a resume does not
  recover them: read ``level_<channel>`` off the run and pass ``level_init``.

- Every entry in ``tests/entry_tolerances.py`` now asks 0.2 rad and 3.0 rad/s on the arms,
  which is the tightest the bridge is ever trained to deliver there. The unmeasured skills
  asked for the baseline, 0.05 rad, which after the band change is something no training
  window contains. This is what the bridge can do, not a robustness claim about those
  skills.

- The bridge reads a clock again: the command carries the seconds left of the crossing
  and the fraction of it already spent, so the observation is 26 + 2J wide and earlier
  checkpoints cannot be resumed. It had none, and a target that is a pose and a momentum
  at one moment cannot be reached without one: a 0.3 second window and a 1.2 second one
  were the same question, so the policy could only ever learn one average approach. That
  is what it learned. Freezing the tolerance curriculum, reopening the policy noise,
  removing the start perturbation and keeping the guidance reward alive each left the
  arrival error within a few percent of where it started, because all four addressed the
  search and none of them the missing input.

  Timing was removed once before because a demo had to estimate how long a crossing would
  take and estimated it badly. The clock does not bring that back: a caller fixes the
  window it asks for rather than solving for one, and ``arrival_s`` read against it says
  whether the crossing used the time it was given.

- The two leg arrival tolerances tighten: ``leg_joint_vel`` from 1.50 to 0.80 rad/s and
  ``leg_joint_pos`` from 0.10 to 0.08 rad. ``leg_joint_vel`` was the one channel declared
  looser than a skill accepts, so a bridge could meet the requirement and still hand over
  a robot that does not track. Both now sit a fifth below what ``entry_margin`` measured
  on the kick, which held its clip at 0.10 rad and 1.00 rad/s and left it by 0.15 and
  1.50: the measured values were the last rung that passed rather than the edge of
  anything, and they were measured one channel at a time. The other six channels were
  already tighter than the kick needs and are unchanged, leaving every channel covered
  with at least 1.22x margin. Retrain to pick this up; requests spanning 0.5x to 4x of
  the baseline move with it.

- The bridge logs ``reach_*`` per channel, plus ``worst_channel`` and ``channels_met``.
  ``reach_*`` is that channel's arrival error over its requirement, so 1.0 is the limit
  whatever the units were and the eight are comparable to each other; ``worst_channel``
  is the largest of them and ``channels_met`` counts how many of the eight are inside.
  All three are against the fixed baseline, like ``score``, so a falling curve means the
  crossing improved rather than the curriculum letting go.

- ``arrival_score`` aggregates log distances and squashes once, as
  ``benchmarks/objective-proposal.md`` specified, instead of blending a per channel
  ``exp(-z^2)``. The Gaussian was flat to machine zero a few tolerances out, so the
  worst channel carried none of the gradient and the bottleneck term was a constant:
  measured on a trained bridge leaving six tolerances of leg joint position error, that
  channel held 0.00% of the objective's sensitivity and now holds 46%. Scores are not
  comparable across the change, and a crossing with every channel exactly on its limit
  moves from 0.368 to 0.591.

Added
^^^^^

- ``skills/tolerance.py``, which measures the other half of a hand-over: given a skill and
  one selector entry, how wrong a state it can be handed and still get back on its clip. One
  channel is displaced at a time along a ladder, over both signs and several directions, each
  case in its own env. A displacement counts as tolerated while the skill's own tracking
  error stays within ``--margin`` of what that entry produces undisturbed.

  Explicitly not a survival test, which is the distinction that makes the number worth
  having. A skill handed a bad state usually stays upright, wanders off its reference and
  finishes the motion as something else: a success for a fall check and a failure for
  composition. Termination is reported in its own column and is never the criterion. On the
  kick at entry 3 it never fired at all, so every limit there is a trajectory being lost.

  It prints a ``Tolerances`` block to paste into ``tests/entry_tolerances.py``, which closes
  the loop with ``bridges/evaluate.py``: measure what the skill accepts, then score what the
  bridge delivers against it.

  This is ``tests/end2end/entry_margin.py``, moved, generalised past the kick and made
  runnable again. It had been dead since the ``benchmarks`` package left the tree, and its
  displacement machinery is reimplemented here rather than imported. It also no longer builds
  its policy after placing the robot: ``RslRlVecEnvWrapper`` resets the env on construction,
  which put every displaced robot back on its reference, and the measurement read as a skill
  that tolerated the entire ladder on all eight channels.

- The kick at entry 3, clip frame 124, has a measured entry profile in
  ``tests/entry_tolerances.py`` instead of the provisional one: 0.06 m, 0.15 rad, 0.10 m/s,
  1.2 rad/s, 0.10 rad, 1.5 rad/s, 0.8 rad, 6.0 rad/s. The arm position limit is a floor,
  since that ladder ran out without failing.

Changed
^^^^^^^

- ``BridgeCommandCfg.landing_s``: how near the duration a window asked for an arrival has
  to be to count, in seconds. ``None`` keeps the old behaviour, the best moment of the whole
  window whenever it happened, and imitation stays on it. The distillation bridge sets
  0.15 s.

  This closes a gap between what the policy reads and what it is paid for. The observation
  carries a clock, seconds left of the crossing and the fraction spent, and nothing happened
  when it ran out: the arrival could be scored anywhere in ``patience_scale`` times the
  duration, so a third of every episode was time in which arriving was neither early nor
  late but untimed.

  Measured before the band existed, on the first full distillation run: the best moment
  already landed at 1.02 times the asked duration in the median, and scoring at exactly that
  duration instead moved the aggregate from 0.406 to 0.394. So the freedom was not being
  exploited, which is not the same as saying it costs nothing to leave open. The same run
  put each channel's own minimum a median of 4 to 8 control steps from the scored instant,
  and a band is what forces the eight to coincide rather than letting the aggregate pick a
  compromise between them.

  A band and not a single instant, because the original argument against a fixed deadline
  holds: a target carries momentum, so it is a state the robot passes through, and a
  crossing that went through it perfectly three ticks early is a good crossing.

  Safe in the distillation task and not in imitation, which is why it is opt in. Distillation
  covers the approach with ``guidance`` densely and never anneals it, so gating the arrival
  term costs no early gradient. imitation relies on arrival being dense from the first step,
  which is the whole argument for paying the improvement rather than a terminal score.

- ``arrival_score``'s channel weights now read root before legs before arms, 6 to 3 to 1,
  where they used to read the reverse: the four joint channels carried 2.0 and 1.5 against
  the root's 1.0. The old ordering was argued from which channel is easy rather than which
  one matters. The measurement settled it: against the kick's measured envelope at entry 3,
  root linear velocity is the channel furthest outside on 74% of crossings and neither arm
  channel on any of them, while the weights had the arms at four times the root.

  Scores are not comparable across the change. A crossing exactly on every limit still reads
  0.591, since the weights are normalised by their own sum, but any uneven crossing moves.

- ``bridges.imitation.mdp.guidance`` takes ``bottleneck_weight``, defaulting to the 0.2 it
  was hard-coded at, so an architecture that keeps its reference can be held to its worst
  channel instead of to an average.

- The distillation teacher's tracking reward is a tracking objective rather than a hint:
  ``tolerance_scale`` 4.0 to 2.0 and ``bottleneck_weight`` 0.2 to 0.5, on top of the weight
  it already carried. The first full run plateaued at 6500 of 11000 iterations with its worst
  channel still four times its requirement, which is inside a kernel evaluated at four times
  that requirement and aggregated as an average: there was almost no gradient left where the
  errors actually were.

- ``UNITS`` moved next to ``CHANNELS`` in ``bridges/imitation/mdp/commands.py``. Both
  evaluators print channels in physical units and two copies were two chances to mislabel a
  number.

Added
^^^^^

- ``bridges/evaluate.py`` gained a hand-over section, which is what the script is actually
  for. A bridge is a means: what decides whether it worked is whether the policy taking over
  can resume from where it was left, and that policy tolerates some envelope of error. The
  section scores the delivery against each envelope, reporting the share of crossings that
  land inside it as written and, for a given share, the multiple the envelope would have to
  be widened by. That multiple is taken from the worst channel of each crossing, so it is a
  statement about all eight at once rather than eight separate marginal ones, and the gap
  between it and the per channel column is the price of needing every channel right at the
  same moment.

  Each envelope also gets ``blocks the hand-over``: the share of crossings where that
  channel is the one furthest outside. That is the fix list, in order.

  Envelopes come from ``tests/entry_tolerances.py`` and are deduplicated by value, since
  printing one row per skill would suggest one measurement per skill where there is
  currently one profile shared by all nineteen entries.

- ``bridges/evaluate.py``, one delivery report for every architecture. Takes ``--bridge``,
  draws windows from the eval split of the shared corpus, and reports the gap left standing
  at the best moment of each crossing in the units the gap is measured in: metres, metres
  per second, radians, radians per second, with the angular channels also in degrees. A
  robot holding its default pose is scored beside it in the same run, because the channel
  errors have no absolute meaning and are only ever a number next to another number.

  Per channel it gives the requirement, the median, the ninth decile, the median as a
  multiple of the requirement so the eight are comparable across their units, and the share
  of crossings that met it. Then the joints: which one is worst in its group how often, and
  its error on the crossings where it is. That last table is not per joint medians, which do
  not reconcile with a channel that is a worst-joint maximum and read like a contradiction.

  ``--start-noise`` scores the policy from a perturbed start rather than exactly on a corpus
  row, which is the condition a bridge actually runs in, and collapses the perturbation ramp
  so the setting takes effect inside one evaluation.

  Every run writes ``logs/benchmarks/<bridge>/<run>_<checkpoint>.md``, one file per
  architecture per checkpoint. ``bridges.imitation.evaluate`` stays as it is: it carries the
  statue diagnosis of the corpus, which is about the corpus rather than about a policy.

- ``bridges/distillation`` is written, so ``--bridge distillation`` has a task behind it:
  ``Mjlab-G1-Distillation-Bridge``, logging to ``logs/rsl_rl/g1_distillation_bridge``. It is
  MaskedMimic's second stage applied to the bridge, and it runs as one job in two phases on
  ``MjlabTeacherStudentRunner``. Phase one is PPO on a teacher that reads the recorded
  crossing frame by frame, which makes it a tracking problem rather than a two point
  boundary value problem. Phase two freezes that teacher and regresses a student onto it
  over the student's own rollouts, with the student reading only a randomly masked subset of
  the same crossing.

  The window's interior becomes ``keyframes`` constraint slots, evenly spaced strictly
  inside it, each carrying the gap to the recorded state at that tick, the seconds until it,
  and two bits: ``core`` for the root and the legs, ``arms`` for the shoulders, elbows and
  wrists. Masked channels are zeroed and the bits say so. The target is never in there: it
  reaches the policy through the base command as it always did, so with every bit off the
  student's observation is the imitation bridge's observation followed by a block of zeros.
  That bare pattern is drawn outright on ``bridge_prob`` of windows and is the only one play
  shows, because it is the question the student is deployed on.

  Widths on the G1 with three keyframes: student 408, teacher and critic 482, against
  imitation's 180.

  The environment is ``bridges.imitation.env_cfg.bridge_env_cfg``, not a copy: the robot,
  the terrain, the sensor, the action term, every reward, every termination and the corpus
  come from there, and ``MaskedBridgeCommandCfg`` carries every field of
  ``BridgeCommandCfg`` across. So ``score``, ``fixed_arrived``, ``reach_*`` and
  ``worst_channel`` are computed by the same code against the same baseline and a number
  from one architecture is comparable to a number from the other. Two metrics are added:
  ``visible_slots`` and ``bridge_pattern``.

  Two deliberate differences from imitation, both following from the teacher being thrown
  away after phase one. ``guidance`` is weighted 8 rather than 2, and
  ``MaskedBridgeCommand.guide_scale`` holds it at one instead of annealing it to zero: the
  network that has to run without a reference is the student, which this reward never
  touches.

  No latent. MaskedMimic's student is a conditional VAE and that is their answer to the
  multimodality the bare mask leaves; this student is a deterministic regression and will
  aim between two equally good crossings. The package docstring says what to measure before
  writing one.

  ``tests.stage`` and the parkour demo drive it, which they could not when it was written.
  See the ``--bridge distillation`` entry under Fixed.

- ``skills.recover``, which finetunes a tracking skill to get back on its reference fast
  after a bad hand-over at one entry point, rather than to survive a wide reset anywhere in
  the clip the way ``skills.finetune`` does. Three pieces. A share of every reset batch
  becomes a rehearsal: it resets inside the frames around a chosen selector entry, with the
  per-channel noise a hand-over actually delivers, while the rest of the batch resets the
  way the task always did, which is what keeps the other nine tenths of the clip in the
  training distribution. A potential shaping term on the tracking error pays per step for
  closing it, which is the only part of the reward that reads at all at the error the bridge
  leaves: the task's summed ``exp(-e^2/s^2)`` kernels are pinned near zero there, so nothing
  told the policy that a smaller error was better until it was nearly back. The shaping
  telescopes over an episode, so by Ng, Harada and Russell it cannot move the converged
  skill, only supply gradient through the transient. And ``motion_far`` is opened at a
  rehearsal reset and closed back to the task's own threshold over a quarter second, without
  which the reset offset alone trips the termination before the policy has acted and the
  episode collects the termination penalty for a state it was handed.

  ``benchmarks.kick.transitions`` reports what the run trains: ``recovery_steps`` to get the
  worst tracked body back within ``recovered_m`` of the reference, the peak and final errors,
  and ``track_error_auc_ms``, the integral over the kick. It also takes ``--kick-variants``,
  which resolves one checkpoint per named version of the skill, so comparing the baseline
  against the finetune is ``--kick-variants "('base','recover')"`` and each row carries the
  ``kick_label`` it was run under.

- Added ``tests.end2end.error_shapes``, which records the joint errors a trained bridge
  actually arrives with, and a ``--shapes`` flag on ``entry_margin`` that displaces the
  four joint group channels along one of them instead of along a random direction. Both
  are renormalized so the worst joint of the group lands on the same rung, so the two
  ladders ask for the same channel error and differ only in how it is spread over the
  rest of the group. The bridge's misses are about twice as concentrated as a random
  draw: worst joint over median joint is 4.0 against 1.9. ``cases.intervention`` takes
  the recorded direction as a keyword after ``arms``, so existing positional callers are
  unaffected.

- Added ``tests.end2end.reset_std``, which copies a checkpoint with the policy's action
  noise reopened and nothing else touched. A converged policy and one that has stopped
  exploring both read as a plateau; resuming from the copy keeps the learned mean
  behaviour and widens only the search around it, so the two can be told apart.

- Added ``tests.end2end.bridge_delivery``, which walks the robot at the ball, hands
  over to the bridge, and scores the arrival against a selector entry using the
  tolerances ``entry_margin`` measured, one column per window the bridge is given.
  The kick's target is fixed by the ball rather than by the window, so a duration is
  swept by solving ``crossing_time`` for when to hand over rather than by moving the
  target. Prints the paired statue baseline alongside, on the same walk and the same
  hand-over tick.

- Added ``tests.entry_margin``, which measures per channel how far a skill can be
  displaced at one selector entry and still track its clip. It grades the skill's own
  body tracking error against the reference rather than against an undisturbed sibling
  rollout, since the strike is chaotic and two rollouts of one entry part company within
  half a second. Every case runs as its own env in parallel, over both signs and several
  random directions, and a limit is read from the bottom of the ladder up. Reports a
  tolerance per channel in that channel's own unit.

- Added a resumable walk2kick entry 3 experiment with a one second bridge allowance,
  measured and jointly validated tolerance profiles, whole skill kick finetuning,
  pinned checkpoint provenance, raw rollouts and a combined before and after report.
  The finetune's reset widths are read off the baseline measurement rather than
  configured: each channel is opened to the chosen quantile of the error the bridge
  actually left there, times a margin, floored so a satisfied channel is still trained
  and capped so a delivery the skill cannot be trained to absorb is reported instead of
  producing an untrainable run. The after stage repeats the bridge sweep under the
  baseline profile as well as its own, since the bridge reads its request and acts on
  it, so a plain before against after would carry a changed bridge as well as a changed
  kick. The report ends with the two side by side.

- Transition tests now select hardcoded bridge tolerances by skill and entry frame.
  Kick entries relax arm position and velocity while retaining strict root and leg
  limits. Handoff output reports both requested and fixed scores. Explicit tolerance
  overrides remain available; parkour uses the same profiles.

- The restored bridge samples tolerance profiles independently per target and channel,
  with a configurable curriculum from broad to tighter requests. Reward, observation
  and arrival checks share each window's profile. ``open_window``, ``place`` and the
  handoff wrappers accept explicit physical tolerances. Fixed baseline metrics remain
  available to compare progress as the requests change. The command stays ``24 + 2J``
  wide; the shared error based width adaptation is replaced by profile sampling.

.. admonition:: Breaking API changes
   :class: attention

   - The selector's automatic scoring is removed: ``selector.entries``,
     ``selector.rank`` and ``selector.selector`` are deleted, along with the
     fitted scorer they produced. ``selector.query`` is renamed to
     ``selector.reach``; ``nearest``, ``best``, ``Cost`` and ``RateCost`` are
     gone and ``reach(table, skill, index, state, seconds)`` replaces them.
     ``stage.Config.mode`` is removed and the entry slider is always shown.

   - ``finetune.Config.position_scale`` and ``velocity_scale`` are replaced by
     ``finetune.Config.scales``, a per-channel ``Scales``. Checkpoints trained
     before this used a uniform 4 and 2; the new defaults differ per channel.

   - ``CollisionCfg`` now requires ``contype``, ``conaffinity``, ``condim``,
     and ``priority`` to be explicit instead of silently defaulting to
     MuJoCo's values, and dict values for these fields must cover every
     matched geom (add a catch-all ``".*"`` entry).

   - ``Mjlab-G1-Bridge`` has no deadline any more, and its observation changed
     width from ``17 + 2J`` to ``24 + 2J``. The two clock channels are gone,
     replaced by the best reward score and eight reward width scales,
     so existing bridge checkpoints do not load. ``mdp.approach`` and
     ``mdp.deadline_reached`` are removed; the termination is now
     ``mdp.out_of_patience``. On the command term, ``deadline`` is ``patience``,
     ``duration_s`` is ``patience_s``, ``reached`` is gone, ``score`` is now the
     best moment of the window rather than the state at a deadline, and
     ``errors_now`` no longer latches anything: ``advance`` does.

   - The bridge observation changes again from ``24 + 2J`` to ``16 + 3J``. The
     adaptive reward widths are removed and replaced by the target preceding
     action gap. ``BridgeCommand.open_window`` and ``place`` now require that
     target action context. Bridge checkpoints from the earlier objective do not
     load into this task.

Fixed
^^^^^

- A recurrent policy driven outside the training loop kept its hidden state across the
  auto-reset, so it entered every episode after the first remembering the last one. PPO
  clears it on every done while collecting rollouts and inference did not, which for the
  bridge means entering a crossing with memory of a different one. ``reset_policy`` in
  ``mjlab.utils.torch`` is that call, now made by the viewer loop and by the three places
  that drive a bridge policy in their own loop: ``bridges/evaluate.py``,
  ``bridges/imitation/evaluate.py`` and ``tests/end2end/error_shapes.py``. No-op for a
  feedforward model, so nothing else changes.

- Nothing under ``mjlab.tasks`` imported at all. The corpus package was renamed from
  ``bridges/datasets/`` to ``bridges/dataset/`` without its importers following, so thirteen
  modules named a package that does not exist and ``import mjlab.tasks`` stopped at the
  first of them. Paths updated, ``tests/test_bridge_dataset.py`` included.

- The parkour demo ended every course on the way up the first box. Its fall check
  measured the torso's lean against vertical and gave it 60 degrees, but a climb
  mount doubles the body over on purpose: the reference itself reaches 78 degrees
  at frame 147, so a robot tracking the clip correctly was called down at frame
  133 of 455. The lean is now measured against whatever the traversal's own
  reference is doing, so the check stays live through a climb instead of being
  switched off for it. Height is still measured against the world and is what
  covers a fall off a box.

- The parkour demo could not end a traversal. ``Controller.past`` asks for a metre
  of floor beyond the obstacle's far face and neither clip covers that: the climb
  ends 0.13 m past its box and the jump lands with nothing left to travel. So
  every traversal ran to ``TRAVERSE_PATIENCE`` instead, and 300 steps ran out with
  the climb's reference still on top of the box, handing the walk a robot up
  there. A traversal now ends when its clip does, ``past`` stays as the answer for
  a traversal skill with no clip, and patience is counted past the end of a
  reference rather than in total.

- The parkour demo's controller had no ``reset``, so the viewer's reset button put
  the robot back on the start line and left the phase machine where it was: the
  run carried on from whichever obstacle it had reached, under whichever skill was
  driving. ``Controller.reset`` restarts the course, and ``Bridge.reset`` forgets
  the target the last crossing was aimed at.

- Bridge training now terminates and pays a success event at the first
  simultaneous entry into all fixed state and preceding-action tolerances.
  Progress uses a long-tail closest score initialized from the actual perturbed
  start, while imitation shaping and alive reward are removed. Duration and start
  noise advance only after measured strict success. Reward and termination state
  is refreshed after the final physics substep, and PPO timeouts bootstrap from
  the terminal observation captured before auto reset.

- Bridge arrival increments now retain their intended reward amount after timestep
  scaling. Reward widths remain fixed within each episode, and fixed-tolerance
  evaluation keeps its own best state. The command observes the actual reward
  baseline and eight width scales, increasing its width to ``24 + 2J``. Earlier
  bridge checkpoints require retraining; skill checkpoint layouts are unchanged.

- Selector recordings now retain the preceding action, reference placement, clip
  identity and scale. Handoffs move the recorded robot and reference together,
  preserving tracking offsets. Re-run ``selector.record`` and ``selector.build``
  before using older entries for handoffs. The oracle now uses the same target as
  the bridge and compares recorded versus outgoing preceding actions.

- The bridge was optimizing the wrong term and never arrived. ``arrival`` was an
  8 channel kernel under a ``progress ** 3`` ramp, which put most of its mass in
  the last fifth of a window, and 70% of what was left rode on whichever single
  channel happened to be worst. The broad ``approach`` term meant to cover the
  rest of the window collected more than the objective did: at convergence over a
  12868 iteration run, ``approach`` was worth 0.238 an episode and ``arrival``
  0.114, ``alive`` alone was worth 0.169, root linear velocity error had not
  improved since iteration 0 (0.520 to 0.556), and ``Metrics/bridge/arrived`` read
  0.000 for the whole run. The policy had correctly learned what it was paid for,
  which was to hover near the target and not fall over.

  ``arrival`` now pays how much the arrival score beat the best already reached
  this window, so an episode's total is the best arrival the crossing ever
  managed. It is dense, it has no instant to hit, and it pays nothing for
  returning to a target already passed, which matters because a target carrying
  momentum is one the robot cannot stay on: scored at a fixed tick, a crossing
  that went through the target perfectly three ticks early read as a miss.
  ``approach`` is deleted rather than reweighted, since the broad early gradient
  it existed for is what the tolerance curriculum already provides.

  The deadline goes with it. Nothing is scored at an instant, so the policy is no
  longer told the time and no longer trades accuracy for punctuality; the duration
  a window's ends were drawn at now only buys ``patience_scale`` times as much
  patience before an unsuccessful crossing is abandoned. How long a crossing takes
  became an output, reported as ``Metrics/bridge/arrival_s``.

- The transition viewer still had a ``duration_s`` slider and a "solve the
  duration" checkbox, which now offered a window the bridge is not given. Both
  are gone from the panel, and the lines the run prints no longer say the bridge
  gets a number of seconds: a duration still places a target that nothing else
  fixes and still buys patience, so it is reported as how far ahead the target was
  placed. ``Run.verdict`` dropped its "outside the range it trained on" clause for
  the same reason and keeps the acceleration check, which was always physics.
  ``Config.duration_s`` stays for a run that wants to pin it.

- A hand-over fired on the clock rather than on the crossing. ``Bridge.done`` and
  the transition arena both switched skills when the window's last tick elapsed,
  which handed over at the same moment whether the bridge had arrived or not. Both
  now end the bridge phase on arrival or on patience running out, and say which:
  ``Bridge.succeeded`` is the branch a controller should read, and the demo prints
  ``arrived`` or ``gave up`` with the best score it managed.

- ``BridgeCommand`` only updated itself from a reward term, so in the transition
  arena and the parkour demo, neither of which has a bridge reward manager, none
  of it ran. That was survivable while the state it kept was only reported; it is
  not now that the best score of the window is half of what the policy reads, so
  ``_update_command`` calls ``advance`` too and the call is idempotent within a
  step. The tolerance curriculum is off wherever there is no corpus, which is
  inference, so the observed score means the same thing there as in training.

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

- ``skills.finetune``, which widens a tracking skill's reset noise and continues
  training it, so the skill survives being handed a robot instead of resetting into
  one. The reset already places the robot on the reference plus noise, and that noise
  was far narrower than what a hand-over delivers: measured at kick entry 3, the
  bridge misses the worst leg joint by 4.3 arrival tolerances while the reset perturbs
  it by 0.5 and the skill stops working past 2. Noise width is one knob per group in
  multiples of the arrival tolerances, ramped from the task's own values, with the
  task's curricula pinned at the stage the loaded checkpoint was trained at rather
  than rewound to their first stage. Nothing else about the task changes, so a
  finetuned skill stays independent of any bridge.

- ``joint_velocity_range`` on ``JumpCommandCfg``, joint velocity written at reset on
  top of the reference's own. Zero by default, which is what every tracking task had
  before: joint velocity was the one channel a reset never perturbed, and it is one a
  hand-over cannot deliver.

- ``--kick-checkpoints`` in the fixed entry walk2kick transitions script, which runs
  the whole sweep against several kick policies and records which one each trial used.
  A robustness finetune is then read as the difference between two groups of otherwise
  identical trials, controls included.

- Skill finetuning takes one noise scale per bridge channel instead of one for
  positions and one for velocities, defaulted to what the bridge measurably delivers
  at kick entry 3. Root orientation drops to twice its tolerance, which is the pitch
  limit, so the ramp stops over-stressing the channel the robot actually falls in.

- ``tests.transitions.walk2kick_robust_test`` loads a baseline and a finetuned kick
  together and adds a dropdown to swap between them plus a slider to pin the switch
  instant, so two policies can be watched on the same crossing. ``stage.main`` takes
  an optional panel hook and ``Run.switch_at`` is mutable where ``Config.auto`` is not.

- The selector no longer chooses an entry. ``selector.query`` is now ``selector.reach``
  and reports what a named entry would cost to reach, in table order, without ranking
  anything. The parkour demo names its entry per skill in ``controller.ENTRIES`` and the
  staging arena puts it on a slider, so ``stage.Config.mode`` is gone and manual is the
  only behaviour. ``nearest`` and ``best`` are removed.

- Group scoped joint disturbances in the resumption benchmark. ``leg_joint_pos``,
  ``leg_joint_vel``, ``arm_joint_pos`` and ``arm_joint_vel`` use the bridge arm mask
  and default to the bridge arrival tolerance for that channel, so a scale is a
  multiple of the requirement. Every trial also records the eight bridge channel
  errors it injected. ``benchmarks.tolerance`` turns a sweep into the per-channel
  displacement the next skill actually survives, graded on outcome quality rather
  than the skill's own generous success bar.

- ``benchmarks.objective``, an offline audit of the bridge arrival objective against
  recorded crossings. Reports per channel how close it ever got, where it ended, and
  what share of the reward's sensitivity it carries, alongside the same share under
  the potential in ``objective-proposal.md``. ``--relax`` rescores against multiplied
  tolerances.

- ``--repair-channels`` in the fixed entry walk2kick transitions script. Replays a real
  bridge crossing, puts the named channels back on target and leaves every other error
  as the bridge produced it, which separates a channel that caused a bad handoff from
  one that came along with it.

- Restricted bridge capability benchmark with fixed start and target pairs,
  recorded action reachability controls, isolated training runs and evaluation
  on reserved starts. The kick package builds cases for selector entry 3.

- Reusable skill resumption benchmarks with shared disturbance and action-sensitivity
  trials, separate climb and kick outcome evaluators, paired replay checks, saved
  trajectories and summaries that distinguish baseline failures from incomplete runs.

- ``tests.resumption`` compares uninterrupted skill execution with reconstruction
  of its entry in the same environment, reporting observation, action and motion
  differences. It supports kick and climb without a bridge checkpoint.

- A Record box in the Viser viewer's Controls tab writes the episode to an mp4
  while it plays, so ``play`` and the transition scripts under
  ``bridging/experiments/humanoid/tests`` can produce a video without a rerun. The
  panel holds the output folder, a frame size and one button; the file is
  ``<folder>/<name>-<timestamp>.mp4``, under the run's ``videos/play`` for
  ``play`` and under ``videos/`` for a transition. This is separate from
  ``play --video``, which records a fixed number of steps from the start with no
  way to say when.

  Frames come from an offscreen MuJoCo camera rather than from the browser, so
  they carry the debug visualizers (the bridge's target ghost, the entry states)
  and cost a few ms each. They are taken on sim time rather than wall clock, so
  the file plays back at real speed however slowly the viewer runs, and the speed
  buttons record as slow motion or fast forward.

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
