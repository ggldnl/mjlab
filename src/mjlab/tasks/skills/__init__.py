"""
Skill bridging: compose independently trained skills with a transition bridge.

    skill.py        the frozen policies being composed
    controller.py   which skill should be running and when a switch fires
    meta.py         the meta policy: pool + switching machinery, and the driver that
                    pairs it with a controller
    architectures/  the concrete meta policies (arch_0 direct hand-off, arch_1 ...)

"""
