"""
Skill bridging: compose independently trained experts with a transition bridge.

Three pieces, one per module, in dependency order:

    expert.py       the frozen policies being composed, and the pool they live in
    bridge.py       what drives the robot between two experts, and when to hand over
    controller.py   which expert should be running, and so when a switch fires

Everything is batched over the envs of a vectorized mjlab env and nothing gathers or
scatters. A method that concerns a subset of envs takes a boolean mask of shape
(num_envs,) and returns full width tensors; entries outside the mask are meaningless
and the caller discards them. Expert ids are int64 tensors of shape (num_envs,), one
id per env, with NO_EXPERT marking absence.
"""

from mjlab.tasks.registry import register_mjlab_task
