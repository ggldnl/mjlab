"""Where a bridge's start and target states come from. Shared by every architecture.

    dataset.py   the format, and the rollout driver every source uses
    tracker.py   builds the corpus by driving motion trackers over LAFAN1 clips
    view.py      per source counts, and a window replayed as a ghost

One corpus, held here rather than inside an architecture, because what a bridge is asked
to cross does not depend on how its policy is produced. An architecture points its command
term at DEFAULT_DATASET and reads the same windows as every other one, so two of them are
comparable.

Both ends of a window are cut out of one tracker rollout a fixed time apart, which makes
the pair reachable by construction. A retargeted human clip is a description, not a state a
G1 is ever in, so nothing here reads motion capture directly.

Run

1. Build the corpus.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker

2. Inspect it.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.view
"""
