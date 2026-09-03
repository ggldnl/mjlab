"""Unitree G1 skills, and the machinery that composes them.

skills/    the individual policies: walk, run, jump, ...
bridge/    one policy that gets from any dynamic state to any other
selector/  which entry states are worth aiming the bridge at, scored and ranked
tests/     drive two skills back to back and measure the hand-over
demos/     complex scenarios where skills are switched by a controller
"""
