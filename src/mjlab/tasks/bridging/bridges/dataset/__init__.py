"""The kinematic corpus every bridge trains on. Shared by every architecture.

    dataset.py        shared format, loader, and rollout driver
    motion_capture/   retargeted and filtered BABEL and LAFAN
    motion_graph/     pairs of filtered BABEL and LAFAN clips stitched at a look-alike frame
    view.py           per source counts, and a window replayed as a ghost

The corpus is kinematic. A planner produces kinematic trajectories too, and the universal
tracker is what makes them physical.

Run

1. Build the motion capture corpus, then the stitched clips on top of it.

    uv run python -m mjlab.tasks.bridging.bridges.dataset.motion_capture.build --robot g1
    uv run python -m mjlab.tasks.bridging.bridges.dataset.motion_graph.build --robot g1

2. Inspect it.

    uv run python -m mjlab.tasks.bridging.bridges.dataset.view
"""
