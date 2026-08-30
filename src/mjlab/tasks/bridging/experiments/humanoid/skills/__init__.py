"""The individual G1 skills, one sub-package each.

    walk      commanded-velocity locomotion, mjlab's flat G1 task under a different name
    run       same mjlab flat G1 task, with different curriculum for higher forward speed
    jump      goal-conditioned jump, motion tracking against retargeted clips
    kick      a standing strike on a football at a commanded launch velocity
    push      walk into a heavy box and drive it a commanded distance
    karate    a strike out of a standstill, tracking a crop of a LAFAN1 fight performance
    backflip  the same, against a reference that was built rather than recorded

Run:

    uv run train Mjlab-G1-Walk --env.scene.num-envs 4096

Swap in Run, Jump, Kick, Push, Karate or Backflip for the other six. Each sub-package's
__init__.py says what its task needs (policies where the robot interacts with objects in
the scene require different observations).
"""
