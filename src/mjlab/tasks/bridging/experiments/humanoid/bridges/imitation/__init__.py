"""The imitation bridge. One PPO policy that drives the robot from a start dynamic state
to a target dynamic state.

Task id Mjlab-G1-Imitation-Bridge, checkpoints under logs/rsl_rl/g1_imitation_bridge.

    in     state (root velocities, gravity, joint angles and rates, last action)
           + per channel gap to the target state
           + the best reward score reached so far and eight requested tolerance scales
    out    29 joint position targets

One episode is one window: teleport onto a start state, then get to the target state. There
is no deadline and no clock in the observation. The policy is paid the best arrival score it
reaches, whenever it reaches it, and a window that is going nowhere is abandoned after
BridgeCommandCfg.patience_scale times the duration its two ends were drawn at.

Which means how long a crossing takes is measured rather than commanded. Read
Metrics/bridge/arrival_s for it. Use BridgeCommand.arrived_now for a live handoff;
arrived records any success during the window and score reports its best fixed-tolerance
match. The reward baseline best uses the profile frozen for that window. fixed_arrived
reports success against the baseline independently of the requested profile.

Endpoints come from tracker rollouts, never from motion capture. A retargeted human clip is
a description, not a state a G1 is ever in. Both endpoints come from one rollout a fixed
time apart, which makes the pair reachable by construction. The motion recorded between
them is used as a learning signal (mdp.guidance, annealed to zero), never as an input: the
policy reads only its own state and the gap to the target, in training and at inference
alike.

Run

1. Build the corpus. It is shared, so it lives one level up. Look at bridges/dataset.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker

2. Inspect it: per source counts, then a window replayed as a ghost.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.view

3. Train.

    uv run train Mjlab-G1-Imitation-Bridge --env.scene.num-envs 4096

4. Score it against a robot that does nothing.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.evaluate

5. Watch it. Amber ghost is the target, blue ghost is the recorded crossing.

    uv run play Mjlab-G1-Imitation-Bridge

Layout

    mdp/           commands (the window), rewards, terminations
    env_cfg.py     the mjlab task
    evaluate.py    scoring against the do nothing baseline

The corpus is not here. Every architecture reads the same one, from bridges/dataset.

API

Two methods on the command term drive the bridge from outside:

    place(env_ids, start, target, duration_s)    teleport onto a start, then cross
    open_window(env_ids, duration_s)             cross from wherever the robot already is

Durations are seconds, not control ticks. Tick counts change with the decimation. The
duration is how long the crossing is expected to take and buys patience_scale times that
much patience; it is not a deadline and the policy never sees it, so a caller with no
opinion should pass the middle of duration_s_range.

Both methods accept tolerances as a keyword: a Tolerances object, an (8,) tensor shared
by the selected environments, or an (N, 8) tensor in env_ids order. Values use physical
units and CHANNELS order. For example:

    command.open_window(ids, seconds, tolerances=Tolerances(arm_joint_pos=0.2))

An omitted profile uses BridgeCommandCfg.tolerances. Keep that baseline unchanged between
training and inference because it normalizes the observation. It is a unit, not the
requirement: what training asks for is a multiple of it, drawn from a per channel band.

Root and legs are drawn from core_band, 4x the baseline down to 0.6x, and the arms from
support_band, 16x down to 4x. The two do not overlap, so the arms are never asked for more
precision than the legs, which is the ordering every measured skill wants. Each window
picks one channel to be strict and relaxes the other seven toward the wide end of their
bands by focus_relax, so a window asks one question. A channel's band position moves only
when the windows that focused it are met between success_band's two rates, which is
Florensa's reverse curriculum applied per channel: above the upper rate the channel is
solved and is asked for more, below the lower it is past what the policy can do and is
asked for less. Watch level_<channel> and focus_rate_<channel>.

Transition scripts accept named limits, for example --tolerances.arm-joint-pos
0.2. The parkour Bridge.aim method accepts the same tolerances keyword.

The observation is 26 + 2J wide: the clock added two values and checkpoints from before
it cannot be resumed. Restored bridge checkpoints load with the same baseline, but need
training on varied requests before their precision conditioning can be assessed. Kernel
shape, channel weights, guidance, alive and termination rules are unchanged. Monitor
fixed_arrived, score and err_* for comparable progress; arrived and
requested_score describe the sampled requests, and tol_* records their physical limits.

The channel to read first is reach_*, which is err_* over that channel's requirement. One
is the limit on every channel, so the eight are comparable to each other whatever their
units, and worst_channel is the largest of them: the one thing standing between the
crossing and an arrival. channels_met counts how many of the eight are inside, so it runs
0 to 8 and reaching 8 is what fixed_arrived reports as 1. All three are against the fixed
baseline, so a falling curve means the crossing improved rather than the curriculum
letting go.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges import BridgeSpec
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.env_cfg import (
  bridge_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

BRIDGE_TASK_ID = "Mjlab-G1-Imitation-Bridge"
BRIDGE_EXPERIMENT = "g1_imitation_bridge"


def bridge_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO config. Copied from the jump: same robot, same rate, both goal conditioned.

  Three values differ from the rsl_rl defaults:

      init_std 0.6            1.0 is too much per joint noise at this action scale, and
                              the shortest window (0.3 s) is too short to recover from it
      num_learning_epochs 4   fewer chances per iteration for the KL schedule to ratchet
      desired_kl 0.015        the learning rate down to the rsl_rl floor of 1e-5, where a
                              run looks plateaued but is only crawling

  Check the learning rate first when a run stalls.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=4,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.015,
      max_grad_norm=1.0,
    ),
    experiment_name=BRIDGE_EXPERIMENT,
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=BRIDGE_TASK_ID,
  env_cfg=bridge_env_cfg(),
  play_env_cfg=bridge_env_cfg(play=True, split="eval"),
  rl_cfg=bridge_ppo_runner_cfg(),
)

BRIDGE = BridgeSpec(
  kind="imitation",
  task_id=BRIDGE_TASK_ID,
  experiment=BRIDGE_EXPERIMENT,
  env_cfg=bridge_env_cfg,
)
"""What bridges.resolve hands to a transition script that asked for this architecture."""
