"""Distillation bridge env config. The imitation bridge's environment, read three ways.

    actor     student. Proprioception, the gap to the target, and the interior of the
              window behind a random mask
    teacher   the same, with the interior unmasked and the recorded crossing frame by
              frame. Phase one trains this one
    critic    the teacher's terms without observation noise

Everything else is imitation's, taken from bridge_env_cfg rather than restated: the robot,
the terrain, the contact sensor, the action term, every reward, every termination and the
whole corpus. Two architectures asked the same question on the same windows and scored by
the same code is the reason bridges/dataset sits one level up, and copying the environment
here would be the fastest way to lose it.

Four things differ.

The command term is MaskedBridgeCommandCfg, which is BridgeCommandCfg plus the keyframe
mask. Every field of the imitation config is carried across, so play mode, the split, the
tolerance curriculum and the start perturbation are whatever imitation set them to.

guidance becomes a tracking objective rather than a hint: weighted up, never annealed,
evaluated at half the width and held to its worst channel. In imitation it is shaping that
has to go away, because the policy it shapes is the one that gets deployed and that policy
has no reference. Here the reference is the teacher's whole job and the teacher is never
deployed. See MaskedBridgeCommand.guide_scale and the three constants below.

The arrival is scored in a band around the duration the window asked for rather than
anywhere in it, so the clock the policy reads is a deadline it is also paid against. See
LANDING_S, and BridgeCommandCfg.landing_s for why imitation does not do this.

Play forces the deployment mask, so watching the task is watching the bridge problem rather
than watching a tracker.

Run

1. Build the corpus. Shared, so it lives one level up.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker

2. Train. One command, both phases.

    uv run train Mjlab-G1-Distillation-Bridge --env.scene.num-envs 4096

3. Watch the student.

    uv run play Mjlab-G1-Distillation-Bridge
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.distillation import mdp
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.env_cfg import (
  COMMAND,
  bridge_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  BridgeCommandCfg,
)

GUIDANCE_WEIGHT = 8.0
"""What the teacher is paid for staying on the recorded crossing.

A per step rate, against an arrival worth at most 8 once. Over the longest window that puts
the two within reach of each other, which is the intent: the teacher should be a tracker
that also arrives, not an arriver that is vaguely nudged toward a motion.

imitation runs this at 2 and anneals it to zero. Both differences are deliberate and both
come from the same fact: the network this reward shapes is thrown away after phase one.
"""

GUIDANCE_TOLERANCE_SCALE = 2.0
"""How wide the teacher's tracking kernel is, as a multiple of the arrival requirements.

imitation uses 4. That is right for a hint: it says be somewhere near this motion, and a
policy that will lose the reference should not be paid for matching it precisely. It is
wrong for a tracking objective, and the first full run showed how wrong. The teacher
plateaued at 6500 of 11000 iterations with its worst channel still four times its
requirement, which is inside a kernel evaluated at four times it: there was almost no
gradient left where the errors actually were.
"""

LANDING_S = 0.15
"""How near the asked duration this task scores an arrival, in seconds.

imitation leaves this off and scores the best moment of the whole window. Turning it on
here is what makes the clock in the observation mean something: the policy reads seconds
left and a fraction spent, and until now nothing happened when they ran out.

0.15 s is 8 control steps at this rate, and is a starting point rather than a measurement.
It is wide enough to cover p10 to p90 of where the previous run's best moments actually
landed, 0.94 to 1.19 of the asked duration, so it removes the untimed part of the patience
overrun without immediately punishing the timing the policy already has. Tighten it once a
run holds inside it.

Safe here and not in imitation for one reason: guidance covers the approach densely and
never anneals, so gating the arrival term costs this task no early gradient. imitation
relies on arrival being dense from the first step, which is the whole argument for paying
the improvement, and a band there would bring back the sparse reward that argument fixed.
"""

GUIDANCE_BOTTLENECK = 0.5
"""How much of the teacher's tracking score is its single worst channel.

imitation uses 0.2, nearly an average, for the same reason it uses a wide kernel: a hint
should ask whether the robot is on the motion at all. A tracker has to be paid to fix the
channel that is wrong rather than the seven that are right, which is what a bottleneck
term is, and it is why the arrival term has run at 0.7 from the start. 0.5 sits between
the two: still enough average to tell a policy that is bad everywhere which way to move.
"""


def distillation_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path = DEFAULT_DATASET,
  sources: tuple[str, ...] | None = None,
  keyframes: int = 3,
) -> ManagerBasedRlEnvCfg:
  """Build the masked distillation environment.

  Args:
    play: no observation noise, no start perturbation, no episode length cap, and the
      deployment mask on every window.
    split: "train" or "eval". Split by recording environment, not by frame.
    dataset_path: which corpus to draw windows from.
    sources: restrict windows to these skills or clips. None means any.
    keyframes: constraint slots inside the window. Changing it changes the observation
      width, so a checkpoint cannot be loaded against a different value.
  """
  cfg = bridge_env_cfg(
    play=play, split=split, dataset_path=dataset_path, sources=sources
  )

  # Field by field off the imitation config, so nothing here decides what a window is. A
  # field added to BridgeCommandCfg arrives in this task without being mentioned twice
  base = cfg.commands[COMMAND]
  assert isinstance(base, BridgeCommandCfg)
  shared = {f.name: getattr(base, f.name) for f in fields(base)}
  # Overridden after the copy rather than passed beside it, because it is a field
  # BridgeCommandCfg already has and the copy already carried imitation's value for it
  shared["landing_s"] = LANDING_S
  cfg.commands[COMMAND] = mdp.MaskedBridgeCommandCfg(
    **shared,
    keyframes=keyframes,
    # Play asks the only question the student is deployed on. In training the bare pattern
    # is one draw among several, which is what makes the others a curriculum for it
    bridge_prob=1.0 if play else mdp.MaskedBridgeCommandCfg.bridge_prob,
  )

  interior = {
    "keyframes": ObservationTermCfg(
      func=mdp.keyframes, params={"command_name": COMMAND, "masked": True}
    )
  }
  privileged = {
    "keyframes": ObservationTermCfg(
      func=mdp.keyframes, params={"command_name": COMMAND, "masked": False}
    ),
    "reference": ObservationTermCfg(
      func=mdp.dense_reference, params={"command_name": COMMAND}
    ),
  }

  # Order matters and is taken from the groups imitation already built, not restated. A
  # checkpoint is tied to its term list in order, and the same numbers shuffled do not
  # fail, they act on nonsense
  noisy = cfg.observations["actor"].terms
  clean = cfg.observations["critic"].terms
  cfg.observations = {
    # The student, and the policy that gets deployed. Its group is called actor because
    # that is what every inference path asks for
    "actor": ObservationGroupCfg(
      terms={**noisy, **interior},
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    # Phase one's actor and phase two's frozen labeller. Same noise the student sees, so
    # the teacher is asked to act on the quality of signal that actually exists
    "teacher": ObservationGroupCfg(
      terms={**noisy, **privileged},
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms={**clean, **privileged},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  assert cfg.rewards is not None
  cfg.rewards["guidance"] = replace(
    cfg.rewards["guidance"],
    weight=GUIDANCE_WEIGHT,
    params={
      "command_name": COMMAND,
      "tolerance_scale": GUIDANCE_TOLERANCE_SCALE,
      "bottleneck_weight": GUIDANCE_BOTTLENECK,
    },
  )
  return cfg
