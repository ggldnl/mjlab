"""The individual G1 skills, one sub-package each.

    walk         commanded-velocity locomotion, mjlab's flat G1 task under a different name
    run          same mjlab flat G1 task, with different curriculum for higher forward speed
    jump         goal-conditioned jump, motion tracking against retargeted clips
    passing      a standing shove of a football at a commanded launch velocity
    kick         the same, with the ball out of reach and the toe doing the work
    push         walk into a heavy box and drive it a commanded distance
    front_kick   a kick out of a standstill, tracking a crop of a LAFAN1 fight performance
    punch_combo  the same recipe on a different crop, and it borrows front_kick's code
    backflip     the same again, against a reference that was built rather than recorded

Run:

    uv run train Mjlab-G1-Walk --env.scene.num-envs 4096

Swap in Run, Jump, Pass, Kick, ... for the others. Each sub-package's __init__.py says
what its task needs (policies where the robot interacts with objects in the scene
require different observations).
"""
