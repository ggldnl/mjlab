"""The diffusion bridge. BeyondMimic's second stage, over this repo's tracker rollouts.

Task id Mjlab-G1-Diffusion-Bridge, checkpoints under logs/rsl_rl/g1_diffusion_bridge.

    trained on   64 tick windows of the shared corpus, states and actions together
    given        the last 4 ticks the robot lived, a target dynamic state, a deadline
    out          29 joint position targets, redrawn every 5 control ticks

Not a policy. There is no reward, no episode and no simulator anywhere in training: the
model is fitted offline on recorded windows and learns one thing, what a second of G1 motion
looks like under physics. Everything about a crossing is applied afterwards, while the model
is being sampled, by pinning the start and pulling the deadline column onto the target. One
frozen model therefore answers crossings nobody trained it on, and a new kind of demand is a
new cost function rather than a new training run.

Why this and not another regression

The map from (start state, target state, duration) to a motion is not a function. A humanoid
has many ways to redistribute momentum and place its feet that all arrive at the same state
at the same time, the corpus shows exactly one of them per window, and the state space is
continuous enough that no two windows ever share a start. A network fitted by squared error
over that is fitting the mean of several valid strategies, and the mean of two ways to cross
a gap is usually a way to fall into it. A generative model is the standard answer: it samples
one mode rather than averaging over all of them, and a boundary condition is imposed on the
sample rather than trained into a regression target.

The one thing to expect from it, stated plainly because it is the paper's own limitation:
guidance steers well inside a neighbourhood of the learned motion manifold and struggles to
drag a sample across it. A crossing between two postures the corpus never connects is the
case this is weakest at, and the corpus is the lever, not the guidance strength.

Layout

    data.py        the feature layout, the canonicalization, and the window index
    model.py       the temporal U-Net denoiser
    diffusion.py   the noise schedule, the training loss, the guided sampler
    guidance.py    what a crossing is asked for, applied while sampling
    policy.py      the receding horizon controller and the loader shim
    train.py       the offline training script
    env_cfg.py     the mjlab task, which is the imitation bridge's arena

The corpus is not here. Every architecture reads the same one, from bridges/dataset.

Run

1. Build the corpus.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker

2. Inspect it: per source counts, then a window replayed as a ghost.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.view

3. Train the model. Offline, no simulator, one GPU.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.train

4. Score it against a robot that does nothing, and against the other architectures.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.evaluate \
      --bridge diffusion

5. Watch it. Amber ghost is the target.

    uv run play Mjlab-G1-Diffusion-Bridge

Knobs worth turning first

Sampling is decided at inference and not baked into the checkpoint, so all four live on
policy.ControlCfg and a script sets them by replacing policy.DEFAULT_CONTROL before the
runner is built.

    replan_every   ticks executed per plan. Lower is more closed loop and slower
    sample_steps   denoising steps per plan. The main cost of a control step
    strength       how hard the deadline column is pinned. One is inpainting
    hold           how hard the columns after the deadline are held on the target

`uv run train Mjlab-G1-Diffusion-Bridge` does not work and says so. The model is not trained
by reinforcement learning and the task is registered only so that everything which loads a
bridge by task id can load this one.
"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges import BridgeSpec
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.env_cfg import (
  diffusion_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.policy import (
  DiffusionRunner,
)
from mjlab.tasks.registry import register_mjlab_task

BRIDGE_TASK_ID = "Mjlab-G1-Diffusion-Bridge"
BRIDGE_EXPERIMENT = "g1_diffusion_bridge"


def diffusion_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Almost all of this is unread.

  register_mjlab_task wants an rl config and every loader passes one to the runner, so one
  exists. DiffusionRunner ignores it: there is no actor, no critic and no algorithm behind
  this architecture. What is actually used is experiment_name, which is where `uv run play`
  looks for a checkpoint, and clip_actions, which the vectorized environment wrapper reads.

  The training knobs live on diffusion/train.py, which is the script that fits the model.
  """
  return RslRlOnPolicyRunnerCfg(
    experiment_name=BRIDGE_EXPERIMENT,
    save_interval=5_000,
    max_iterations=0,
  )


register_mjlab_task(
  task_id=BRIDGE_TASK_ID,
  env_cfg=diffusion_env_cfg(),
  play_env_cfg=diffusion_env_cfg(play=True, split="eval"),
  rl_cfg=diffusion_runner_cfg(),
  runner_cls=DiffusionRunner,
)

BRIDGE = BridgeSpec(
  kind="diffusion",
  task_id=BRIDGE_TASK_ID,
  experiment=BRIDGE_EXPERIMENT,
  env_cfg=diffusion_env_cfg,
)
"""What bridges.resolve hands to a transition script that asked for this architecture."""
