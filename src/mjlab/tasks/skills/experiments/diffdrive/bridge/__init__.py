"""
The bridge between two diffdrive skills: the baseline and the learned policy.

* InstantBridge   the do-nothing baseline (baseline.py).
* LearnedBridge   the trained bridging policy (policy.py), wrapped to fit the
                  Bridge interface. Imported lazily by callers that need it, so
                  this package stays light to import (no torch / onnxruntime here).

The training side of the learned bridge (environment, MDP terms, rollout
harvesting) lives in the sibling modules and is wired into mjlab as the
Mjlab-Bridge-Diffdrive task by bridge_env_cfg.py.
"""

from __future__ import annotations

from mjlab.tasks.skills.experiments.diffdrive.bridge.baseline import InstantBridge

__all__ = ["InstantBridge"]
