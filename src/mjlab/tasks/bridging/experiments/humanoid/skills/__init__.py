"""The individual G1 skills, one sub-package each.

    walk         commanded velocity locomotion, mjlab's flat G1 task under another name
    run          the same task, curriculum retuned for forward speed
    jump         goal conditioned jump, motion tracking against retargeted clips
    passing      a standing shove of a football at a commanded launch velocity
    kick         WIP
    push         walk into a heavy crate and drive it at a commanded velocity
    front_kick   a kick out of a standstill, tracking a crop of a LAFAN1 fight performance
    punch_combo  the same recipe on a different crop, borrowing front_kick's code

Run

    uv run train Mjlab-G1-Walk --env.scene.num-envs 4096

Swap in Run, Jump, Pass, ... for the others.
Each sub-package docstring says what its task needs: a skill that touches an object in the
scene carries different observations from one that does not, so their checkpoints are not
interchangeable.
"""
