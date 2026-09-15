"""Unitree G1 skills, and the machinery that composes them.

skills/    the individual policies: walk, run, jump, ...
bridges/   one policy that gets from any dynamic state to any other, one sub-package per
           architecture, plus the corpus they share
selector/  representative entry states sampled from each skill's rollouts
tests/     drive two skills back to back and measure the hand-over
demos/     complex scenarios where skills are switched by a controller

Every script that drives a bridge takes --bridge to pick the architecture. See
bridges/__init__.py.
"""
