"""The individual G1 skills, one sub-package each.

    walk   commanded-velocity locomotion, mjlab's flat G1 task under our name
    run    the same, retuned and curriculumed for forward speed
    jump   goal-conditioned jump, motion tracking against retargeted clips
    kick   a standing strike on a football at a commanded launch velocity
    push   walk into a heavy box and drive it a commanded distance

Run:

    uv run train Mjlab-G1-Walk --env.scene.num-envs 4096
    uv run play Mjlab-G1-Walk

Swap in Run, Jump, Kick or Push for the other four. Each sub-package's __init__.py says
what its task needs.
"""
