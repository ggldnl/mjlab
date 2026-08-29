"""Starting the bridge from a locomotion policy's weights instead of from noise.

    uv run train Mjlab-G1-Bridge --agent.warm-start logs/rsl_rl/g1_walk/<run>/model_950.pt

Optional. Without it nothing changes and the actor is initialized the ordinary way.

##
# What actually gets copied, and what does not
##

The walk actor and the bridge actor are the same network except at the input: both are
MLPs of (512, 256, 128) ending in 29 joint targets, but the walk reads 99 numbers and the
bridge reads its proprioception plus a 75-number target. So `mlp.0` has a different shape
and cannot be copied, and neither can the observation normalizer that sits in front of it.

What transfers is the function and not the training state: whole layers of the MLP whose
name and shape match, which here means the two deeper hidden layers and the output head.
Those map an internal representation to joint targets and are the part that knows what a
stance looks like. The input projection -- where a locomotion policy's reading of its own
state actually begins -- has to be relearned, and so does the observation normalizer.

Matching by name rather than by position is deliberate. A positional copy would happily
write the walk's first layer into the bridge's if the widths ever coincided, and the
observation channels are in a different order, so the result would be a policy confidently
reading its joint velocities as somebody else's velocity command. Nothing silently
misaligns here: what was copied and what was skipped is printed.

##
# Measured, and it currently hurts
##

At a matched 60 iterations on 2048 envs, cold start reaches score 0.142 with 0.715 of
windows surviving to their deadline; warm start reaches 0.064 and 0.468. Roughly half, on
both numbers. Sixty iterations is early and this may invert, but it is the opposite of the
expected direction and it has a plausible mechanism.

The mechanism is that the layers being copied were fitted to consume the output of the
walk's own `mlp.0`, and `mlp.0` is exactly the layer that cannot come with them. Three
trained layers hanging off a random projection of a different 171-number observation are
not a neutral starting point -- they are confidently wrong in a coordinate system that no
longer exists, and PPO has to undo them before it can use them. A freshly initialized
network is at least unbiased.

If this is right, the load-bearing part of a warm start here is the input layer, and
transferring it means aligning observation channels: copy the walk's `mlp.0` columns into
the bridge's for the proprioception terms the two share, by term name and width, and zero
the columns under the 75-number target block. The bridge would then begin as the walk
policy reading the state it recognizes and ignoring the goal, and learn the goal from
there. That is a real warm start and it is not built here, because matching channels by
name is the kind of code that fails silently if it fails at all.

##
# What a warm start can and cannot be expected to do
##

It changes where the policy starts, not where it ends. PPO drifts from its initialization
over a few hundred iterations, so if the reward pays for reaching the target however the
robot gets there, an unphysical gait remains an attractor and the run will find its way
back to one. A warm start buys a better first few hundred iterations and a prior that
decays.

The thing that does not decay is a term in the reward: either a penalty that forbids the
artifact directly (action rate, action acceleration, foot slip, feet air time) or a
persistent regularizer that keeps the bridge's actions near a frozen locomotion policy's
on the same state. Both of those are still open here. If a run comes out hopping, read
`action_rate` before reaching for this flag.
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

  Applied when the runner is built, so a genuine `resume` still wins: `run_train` loads
  the resumed checkpoint afterwards and overwrites everything this put there. Seeding a
  run that is being continued would be meaningless anyway."""


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

  # Whole layers of the MLP and nothing else. Two rules, and both were written after
  # watching a looser one do damage.
  #
  # Only `mlp.*`, because everything outside it is training state rather than function.
  # `obs_normalizer.count` is a scalar and so it matches any actor's, and copying it says
  # the bridge's normalizer has already seen a hundred million samples of an input
  # distribution it has in fact never seen -- its mean and variance, which did not match
  # and were left at zero and one, then barely move again. `distribution.std_param` is a
  # converged policy's exploration, which is the wrong amount for a task that has not
  # started; `init_std` is set deliberately in the runner config and should win.
  #
  # Whole layers, because a layer's bias was fitted next to its weight. `mlp.0.bias` fits
  # this actor and `mlp.0.weight` does not, and half a layer is not a warm start, it is an
  # arbitrary offset on a random projection.
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

  The critic is deliberately left alone. A locomotion critic estimates the return of a
  different reward on a different task, and its output head would be confidently wrong at
  a scale PPO has to unlearn before it can learn anything. The actor is the part where a
  gait lives.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Popped rather than left in place: it is this class's argument, not rsl_rl's, and it
    # should not end up in the config a checkpoint carries around.
    warm_start = train_cfg.pop("warm_start", None)
    super().__init__(env, train_cfg, log_dir, device)
    if warm_start:
      warm_start_actor(self.alg.actor, Path(warm_start), device)
