"""Skill bridging: compose independently trained skills.

The problem: a controller says which skill should be running, the skills are frozen and
know nothing about each other, and something has to carry the robot from
one to the next without breaking it.

Interfaces:

    skill.py        the frozen policies being composed, and the pool they live in
    controller.py   who decides which skill should be running
    meta.py         MetaPolicy: the pool plus the machinery that carries out a switch
    experiment.py   what an experiment hands an architecture

Tools:

    view.py         which channels of the observation an architecture may work on
    windows.py      recorded stretches of skill behavior, and harvested restart states
    spaces.py       the unit conversions that keep every number of roughly unit size
    buffers.py      fixed-capacity storage
    utils.py        where checkpoints live

Folders:

    architectures/  the concrete meta policies (arch_0, arch_1, ...)
    experiments/    the arenas and skill pools (cartpole, diffdrive, parkour, ...)

Entry points:

    skill.py                    watch one skill on its own
    experiments/*/inspect.py    watch the rollout windows an experiment trains on
    experiments/*/demo.py       watch the experiment running with a given architecture
    experiments/*/train.py      train a particular architecture on a given experiment
"""
