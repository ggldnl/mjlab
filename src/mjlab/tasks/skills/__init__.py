"""
Skill bridging: compose independently trained skills with a transition bridge.

    skill.py        the frozen policies being composed
    bridge.py       what drives the robot between two skills
    controller.py   which skill should be running and when a switch fires
    main.py         the reusable loop that runs the three together

"""