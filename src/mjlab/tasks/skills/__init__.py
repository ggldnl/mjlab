"""
Skill bridging: compose independently trained skills with a transition bridge.

    skill.py        the frozen policies being composed
    meta.py         the meta policy: pool + switching machinery
    controller.py   steers the meta policy (which skill should be running and when a switch fires)

Utilities.

    buffers.py      buffers to store skills rollouts
    utils.py        generic utilities
    inspect.py      viewer to watch the windows a bridge is trained on
    view.py         what part of a skill's observation the bridging machinery is allowed to see
    windows.py      stretches of recorded skill behavior around a hand-over (contained in a buffer)

Folders.

    architectures/  the concrete meta policies (arch_0 direct hand-off, arch_1 ...)
    experiments/    the experiments (diffdrive, cartpole, parkour, ...)

"""
