"""Checks on the parkour experiment that are neither training nor demonstration.

A test here builds a specific situation, runs it, and prints a number. It is not a unit
test and does not belong under `tests/` at the repository root: everything in this folder
needs trained checkpoints, a simulator and a minute of wall clock, which is the opposite
of what a test suite is for. What they are is the answer to "does this actually work",
asked of one piece at a time.

    uv run python -m mjlab.tasks.skills.experiments.parkour.tests.walk2jump
"""
