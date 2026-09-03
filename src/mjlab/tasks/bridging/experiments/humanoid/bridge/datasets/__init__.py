"""Training data.

dataset.py   dataset interface
skills.py    dataset built out of rollouts of the trained skills (deprecated)
tracker.py   dataset built out of rollouts of a motion tracker following LAFAN1 clips
view.py      visualization of the training data

Two ways exist for building the dataset: using rollouts of the trained skills and
using rollouts of a motion tracker following human motion data retargeted on the G1
(LAFAN1). The former is deprecated since it ties the bridge to the skill pool.
By training the bridge on diverse human motion data we hopefully make it skill
independent and robust. Adding a new skill with such a bridge should not require
retraining.
"""
