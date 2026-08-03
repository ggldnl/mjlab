"""The unit conversions the bridge family runs on, in one place.

Everything a bridge touches arrives in whatever units the experiment happens to use.
The diffdrive commands wheel velocities in the tens of rad/s and observes body speeds
of a couple of m/s in the same vector. Neural networks, Gaussian policies and
cross-entropy losses all assume numbers of roughly unit size, and every single failure
mode this module exists to prevent is a number that was a hundred or a thousand times
bigger than the thing it was being compared against.

So there are exactly two conversions, and the whole family shares them:

- `ActionSpace` turns raw env actions into normalized ones of roughly unit size, and
  back. The bridge's policy learns entirely in normalized units; the raw ones only ever
  appear in the single line that calls `env.step`.
- `StateSpace` turns a viewed observation (see view.py) into a standardized one of
  roughly unit size. The actor, the critic, the discriminator and the switch-decider all
  read this and nothing else.

Both are measured once, from the skills themselves, before any training starts, and
both print what they measured. That printout is the thing to read first when a run
misbehaves: if a channel's spread is reported as 0.001 while its neighbour's is 15, the
conversion below is the only reason the run will not fall over.

Why measuring over the *whole pool* matters. A bridge starts wherever a source skill
left the robot and has to end up where the target skill lives, so the numbers it sees
span both. Fitting the conversion on the target skill alone (which is what this code
used to do, inside the discriminator) makes every state the bridge starts from read as
an extreme outlier, and the discriminator then separates real from fake on the first
batch without learning anything.

The relative floor. A skill often holds a channel almost perfectly constant: the
diffdrive's `drive` keeps lateral velocity and roll rate at zero to within a
thousandth, and its commanded wheel velocity is a literal constant. Dividing by that
channel's own spread does the exact opposite of what normalizing is for, turning an
ordinary deviation into one of a thousand units. The floor is expressed as a fraction of
the *largest* spread in the vector rather than as an absolute number, so it adapts to
whatever units the experiment uses: a channel that barely moves relative to its
neighbours is flattened rather than amplified.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.view import StateView
from mjlab.tasks.skills.windows import WindowPlan, as_tensor, start_skill

# How small a channel's spread may get, as a fraction of the largest spread in the same
# vector, before it is treated as constant and flattened instead of amplified. 2% is
# well below any channel that carries real motion and well above the 1e-3 residue a
# near-constant channel leaves behind.
MIN_RELATIVE_SPREAD = 0.02


def _floor_spread(spread: torch.Tensor) -> torch.Tensor:
  """Raise every channel's spread to at least a fraction of the largest one."""
  largest = spread.max().clamp_min(1e-6)
  return spread.clamp_min(MIN_RELATIVE_SPREAD * largest)


class ActionSpace(nn.Module):
  """Raw env action units on one side, the bridge's normalized units on the other.

  Fitted so that the range every skill in the pool actually commands maps to roughly
  [-1, 1]. That is a *scale*, not a limit: the policy is free to output past +-1, which
  is how a bridge can brake harder than any skill in the pool ever does.

  An `nn.Module` with buffers rather than a plain dataclass so it travels with the
  architecture's checkpoint. A bridge loaded without the conversion it was trained under
  would command wildly wrong numbers, silently.
  """

  center: torch.Tensor
  half_range: torch.Tensor

  def __init__(self, action_dim: int) -> None:
    super().__init__()
    # Identity until `fit` runs, so an untrained architecture is still callable.
    self.register_buffer("center", torch.zeros(action_dim))
    self.register_buffer("half_range", torch.ones(action_dim))

  @torch.no_grad()
  def fit(self, raw_actions: torch.Tensor) -> None:
    """Set the conversion from a batch of raw actions, shaped (N, action_dim)."""
    low = raw_actions.amin(dim=0)
    high = raw_actions.amax(dim=0)
    self.center.copy_((high + low) / 2.0)
    self.half_range.copy_(_floor_spread((high - low) / 2.0))

  def normalize(self, raw: torch.Tensor) -> torch.Tensor:
    """Raw env action -> normalized action of roughly unit size."""
    return (raw - self.center) / self.half_range

  def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
    """Normalized action -> the raw action `env.step` expects."""
    return self.center + normalized * self.half_range

  def describe(self) -> str:
    low = (self.center - self.half_range).tolist()
    high = (self.center + self.half_range).tolist()
    ranges = [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(low, high, strict=True)]
    return f"raw action range per channel {ranges}"


class StateSpace(nn.Module):
  """A viewed observation on one side, standardized numbers on the other.

  One-way on purpose: nothing in the bridge family ever needs to turn a standardized
  observation back into a raw one, and offering the inverse would only invite somebody
  to mix the two.

  The clip is what stops a single channel from deciding everything. Even after the
  relative floor, a bridge dropped into a state no skill ever visits can land many
  spreads out on one channel; without the clip that one number dominates the first layer
  of every network reading it.
  """

  mean: torch.Tensor
  spread: torch.Tensor

  def __init__(self, dim: int, clip: float = 5.0) -> None:
    super().__init__()
    self.dim = dim
    self.clip = clip
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("spread", torch.ones(dim))

  @torch.no_grad()
  def fit(self, viewed_obs: torch.Tensor) -> None:
    """Set the conversion from a batch of viewed observations, shaped (N, dim)."""
    self.mean.copy_(viewed_obs.mean(dim=0))
    self.spread.copy_(_floor_spread(viewed_obs.std(dim=0)))

  def standardize(self, viewed: torch.Tensor) -> torch.Tensor:
    """Viewed observation -> standardized observation of roughly unit size."""
    centered = (viewed - self.mean) / self.spread
    return centered.clamp(-self.clip, self.clip)

  def describe(self) -> str:
    return (
      f"per-channel spread {[f'{s:.3f}' for s in self.spread.tolist()]}, "
      f"clipped at +-{self.clip:g}"
    )


@torch.no_grad()
def measure_spaces(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  view: StateView,
  plan: WindowPlan,
  clip: float = 5.0,
  obs_group: str = "actor",
) -> tuple[ActionSpace, StateSpace]:
  """Roll every skill in the pool and fit both conversions to what they produce.

  One pass per skill, from that skill's own start state, for as long as the window plan
  says a hand-over may fall. That covers the whole stretch of each skill's life a bridge
  can be dropped into, which is exactly the range the conversions have to span.
  """
  device = env.device
  action_dim = env.action_manager.total_action_dim
  active = torch.ones(env.num_envs, dtype=torch.bool, device=device)

  observed: list[torch.Tensor] = []
  commanded: list[torch.Tensor] = []

  for skill_id in range(len(pool)):
    skill = pool[skill_id]
    spec = plan[skill]
    obs, _ = env.reset()
    obs = start_skill(env, spec, obs)
    skill.reset(active)
    # The latest a hand-over may fall is the furthest into this skill a bridge will ever
    # see it, so that is how far we watch.
    for _ in range(spec.interrupt_range[1]):
      raw_action = skill.act(obs, active)
      observed.append(view(as_tensor(obs[obs_group])))
      commanded.append(raw_action)
      obs, _, _, _, _ = env.step(raw_action)

  all_obs = torch.cat(observed)
  all_actions = torch.cat(commanded)

  action_space = ActionSpace(action_dim).to(device)
  action_space.fit(all_actions)
  state_space = StateSpace(view.dim, clip).to(device)
  state_space.fit(all_obs)

  print(
    f"[spaces] measured over {len(pool)} skill(s), "
    f"{all_obs.shape[0]} samples each of state and action"
  )
  print(f"[spaces] action: {action_space.describe()}")
  print(f"[spaces] state:  {state_space.describe()}")
  return action_space, state_space
