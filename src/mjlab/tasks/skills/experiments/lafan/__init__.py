"""Goal-conditioned skills carved from LAFAN1 command plateaus.

The Unitree-retargeted LAFAN1 clips are long, multi-motion performances: a
single file wanders through several behaviors with natural human transitions
between them (walk-to-run, run-to-jump, jump-to-recover). We turn that into an
asset instead of an annoyance.

The idea:

  1. Label every frame of a clip with a command: the root velocity expressed in
     the robot's own (yaw) frame, ``[v_forward, v_lateral, yaw_rate]``.
  2. Segment the clip on *plateaus* of that command, not on behaviors. A plateau
     is a stretch where the command barely changes, i.e. a near-constant goal.
     The ramps between plateaus are the transitions; we deliberately discard
     them.

Each surviving plateau is a clean, stationary goal signal, exactly what a
goal-conditioned skill wants to track. ``build_dataset.py`` produces one CSV
slice per plateau (which drops straight into ``mjlab.scripts.csv_to_npz``) plus
a manifest labeling each with its mean command.

License: LAFAN1 is CC BY-NC-ND 4.0 (non-commercial); the Unitree retargeting
release is gated on HuggingFace. Cite Harvey et al., SIGGRAPH 2020.
"""
