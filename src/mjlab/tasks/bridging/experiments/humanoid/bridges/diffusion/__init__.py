"""The diffusion bridge. Not implemented.

A stub, so --bridge diffusion is already a choice every transition script and the parkour
demo offer. Selecting it now stops at bridges.resolve with a message rather than somewhere
deeper, and nothing here is imported until something asks for it.

Writing it

1. Add env_cfg.py and whatever mdp terms it needs, the way bridges/imitation does
2. register_mjlab_task under BRIDGE_TASK_ID, logging to BRIDGE_EXPERIMENT
3. Point BRIDGE.env_cfg at the builder, which is what makes resolve stop refusing it

Read the corpus from bridges/dataset rather than collecting one. A second corpus would
make the two architectures incomparable, which is the whole reason to have both.
"""

from __future__ import annotations

from mjlab.tasks.bridging.experiments.humanoid.bridges import BridgeSpec

BRIDGE_TASK_ID = "Mjlab-G1-Diffusion-Bridge"
BRIDGE_EXPERIMENT = "g1_diffusion_bridge"

BRIDGE = BridgeSpec(
  kind="diffusion",
  task_id=BRIDGE_TASK_ID,
  experiment=BRIDGE_EXPERIMENT,
)
"""env_cfg is None, which is what marks this a stub. See bridges.resolve."""
