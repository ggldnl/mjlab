"""Unitree G1 skills, and the machinery that composes them.

skills/    the individual policies: walk, run, jump, kick, push
bridge/    one policy that gets from any skill's exit state to any skill's entry state
selector/  which entry states are worth aiming the bridge at
tests/     drive two skills back to back and measure the hand-over
demos/     placeholder for scenarios built out of the above
"""
