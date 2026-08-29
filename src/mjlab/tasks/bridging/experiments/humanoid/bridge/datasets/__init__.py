"""Where the two endpoints of a window come from. One module per source.

    dataset.py   what a dataset is, plus the rollout driver both sources share
    skills.py    rollouts of the trained skills
    tracker.py   rollouts of a motion tracker following LAFAN1 clips

The bridge does not care which one it gets. mdp/commands.py reads a table of states and
a control rate, nothing else, so switching source is one flag:

    uv run train Mjlab-G1-Bridge \
      --env.commands.bridge.dataset-path data/bridge/tracker.npz

Two sources because one of them cannot answer the question. Train the bridge on the
skills' own rollouts and test it against those same skills, and there is no way to tell a
bridge that learned bridging from one that memorized five policies. The tracker dataset
is made of motion the skill pool never produced, so it separates the two.
"""
