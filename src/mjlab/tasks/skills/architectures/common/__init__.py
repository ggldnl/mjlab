"""Pieces more than one architecture uses.

    nets.py        the actor an architecture trains, and control of its exploration
    imitation.py   the adversarial referee arch_1, arch_2 and arch_3 all play against

Anything used by exactly one architecture lives in that architecture's folder instead.
Nothing here knows about skills, pools or transitions, and dependencies only ever point
this way: an architecture may import from here, and this never imports an architecture.
"""
