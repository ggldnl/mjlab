"""The selector: which states are worth handing a skill, at which goal.

The bridge takes the robot from wherever skill A was interrupted to a target state.
Something has to say what that target should be. This package answers the half of that
question that does not depend on the interruption: for skill B at goal g, here are a few
states worth aiming at, ranked.

Run:

    1. Profile the skills. One npz per skill under data/selector.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.profile

    2. Look at what it picked.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.viewer \
         --skill jump

    profile.py   builds it, one npz per skill
    features.py  what "the same situation" means
    viewer.py    looks at the result

Not here: picking one of the offered candidates at the moment of the switch, which is where
the cost of actually getting there comes in. That is a separate component.
"""
