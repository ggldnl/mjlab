"""Seed the bridge actor from a locomotion policy instead of from noise.

Run:

    uv run train Mjlab-G1-Bridge \
      --agent.warm-start logs/rsl_rl/g1_walk/<run>/model_950.pt

Optional. Without the flag nothing changes and the actor initializes the ordinary way.

##
# What gets copied
##

The walk actor and the bridge actor are the same network except at the input: both are
MLPs of (512, 256, 128) ending in 29 joint targets, but the walk reads 99 numbers and the
bridge reads proprioception plus a 75-number target. So `mlp.0` has the wrong shape, and so
does the observation normalizer in front of it.

What transfers is whole MLP layers whose name and shape match, which here means the two
deeper hidden layers and the output head. Those map an internal representation to joint
targets and are the part that knows what a stance looks like. The input projection has to
be relearned. Matching is by name, never by position: a positional copy would write the
walk's first layer into the bridge's if the widths ever coincided, and the observation
channels are in a different order. What was copied and what was skipped is printed.

##
# Measured, and it currently hurts
##

At a matched 60 iterations on 2048 envs:

    cold start   score 0.142   0.715 of windows survive to the deadline
    warm start   score 0.064   0.468

Roughly half on both numbers. 60 iterations is early and this may invert, but it has a
plausible mechanism: the copied layers were fitted to consume the output of the walk's own
`mlp.0`, and `mlp.0` is exactly what cannot come with them. Three trained layers hanging
off a random projection of a different 171-number observation are confidently wrong in a
coordinate system that no longer exists, and PPO has to undo them first.

If that is right, the load-bearing layer is the input one, and transferring it means
aligning observation channels by term name and width, then zeroing the columns under the
target block. The bridge would start as the walk policy reading the state it recognizes and
ignoring the goal. Not built here, because channel matching by name fails silently when it
fails at all.

##
# What a warm start can and cannot do
##

It changes where the policy starts, not where it ends. PPO drifts from its initialization
over a few hundred iterations, so if the reward pays for reaching the target however the
robot gets there, an unphysical gait stays an attractor and the run finds its way back to
one. A warm start buys a better first few hundred iterations and a prior that decays.

What does not decay is a reward term: a penalty that forbids the artifact directly (action
rate, action acceleration, foot slip, feet air time) or a regularizer keeping the bridge's
actions near a frozen locomotion policy's on the same state. Both are still open. If a run
comes out hopping, read `action_rate` before reaching for this flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from rsl_rl.env import VecEnv

from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg


@dataclass(kw_only=True)
class BridgeRunnerCfg(RslRlOnPolicyRunnerCfg):
  warm_start: str | None = None
  """Checkpoint of a locomotion policy to seed the actor from, or None.

  Applied when the runner is built, so `resume` still wins: run_train loads the resumed
  checkpoint afterwards and overwrites everything this put there."""


def warm_start_actor(actor: torch.nn.Module, path: Path, device: str) -> None:
  """Copy every parameter of `path`'s actor that fits `actor`, and say what happened."""
  if not path.exists():
    raise SystemExit(f"No warm-start checkpoint at {path}.")
  saved = torch.load(path, map_location=device, weights_only=False)
  if "actor_state_dict" not in saved:
    raise SystemExit(
      f"{path} holds {sorted(saved)} and no 'actor_state_dict'. Warm starting needs an "
      f"rsl_rl checkpoint written by `uv run train`, not an exported policy."
    )

  source = saved["actor_state_dict"]
  target = actor.state_dict()

  # Whole layers of the MLP and nothing else. Two rules, both written after a looser one
  # did damage.
  #
  # Only mlp.*, because everything outside it is training state rather than function.
  # obs_normalizer.count is a scalar so it matches any actor's, and copying it claims the
  # bridge's normalizer has seen a hundred million samples of a distribution it has never
  # seen. Its mean and variance did not match, were left at zero and one, and then barely
  # move again. distribution.std_param is a converged policy's exploration, which is the
  # wrong amount for a task that has not started; init_std in the runner config should win.
  #
  # Whole layers, because a layer's bias was fitted next to its weight. mlp.0.bias fits this
  # actor and mlp.0.weight does not, and half a layer is an arbitrary offset on a random
  # projection
  groups: dict[str, list[str]] = {}
  for name in target:
    groups.setdefault(name.rsplit(".", 1)[0], []).append(name)

  copied: list[str] = []
  skipped: list[str] = []
  for group, names in groups.items():
    fits = group.startswith("mlp.") and all(
      name in source and source[name].shape == target[name].shape for name in names
    )
    if fits:
      for name in names:
        target[name].copy_(source[name])
      copied.append(group)
    else:
      why = "not part of the function" if not group.startswith("mlp.") else "shape"
      skipped.append(f"{group} ({why})")

  if not copied:
    raise SystemExit(
      f"Nothing in {path} fits this actor, so a warm start would be a no-op. The two "
      f"networks have to agree on hidden sizes and action count."
    )
  actor.load_state_dict(target)
  print(f"[warm-start] {path}")
  print(f"[warm-start] copied: {', '.join(copied)}")
  print(f"[warm-start] left at init: {', '.join(skipped)}")


class BridgeOnPolicyRunner(MjlabOnPolicyRunner):
  """The ordinary runner, with the actor optionally seeded before the first iteration.

  The critic is left alone on purpose. A locomotion critic estimates the return of a
  different reward on a different task, so its output head would be confidently wrong at a
  scale PPO has to unlearn first. The actor is where a gait lives.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Popped rather than left in place: this is our argument, not rsl_rl's, and it should
    # not end up in the config a checkpoint carries around
    warm_start = train_cfg.pop("warm_start", None)
    super().__init__(env, train_cfg, log_dir, device)
    if warm_start:
      warm_start_actor(self.alg.actor, Path(warm_start), device)
