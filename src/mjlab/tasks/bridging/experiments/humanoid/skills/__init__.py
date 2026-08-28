"""The individual G1 skills, one sub-package each.

    skills/walk   commanded-velocity locomotion, mjlab's flat G1 task under our name
    skills/run    the same, retuned and curriculumed for forward speed
    skills/jump   goal-conditioned jump, motion tracking against retargeted clips
    skills/kick   a standing strike on a football at a commanded launch velocity
    skills/push   walk into a heavy box and drive it to a commanded distance

Train any of them the ordinary way:

    uv run train Mjlab-G1-Walk --env.scene.num-envs 4096
    uv run play Mjlab-G1-Walk
"""
