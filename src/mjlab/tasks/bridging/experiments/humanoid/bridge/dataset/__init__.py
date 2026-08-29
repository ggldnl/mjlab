"""Where a window's two endpoints come from. One module per way of producing them.

    dataset.py    frames of the trained skills' own rollouts

The bridge does not care which of these it is handed -- `mdp/commands.py` reads a bank of
states and a control rate and nothing else -- which is the point of giving them a folder
of their own. Whether a bridge has to be trained on rollouts of the very policies it will
serve, or whether one built from motion capture generalises to a pool it has never seen,
is a question that needs two banks to answer.
"""
