"""Tests for the skill/controller/meta-policy composition interfaces."""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg
from mjlab.tasks.skills.architectures.arch_0 import Arch0
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.meta import ComposedPolicy, MetaPolicy, run_episode
from mjlab.tasks.skills.skill import NO_SKILL, Skill, SkillPool


@pytest.fixture(scope="module")
def device():
  return get_test_device()


class ConstantSkill(Skill):
  """A stateless skill that always outputs the same action."""

  def __init__(self, name: str, value: float, action_dim: int) -> None:
    self.name = name
    self.value = value
    self.action_dim = action_dim
    self.act_calls: list[torch.Tensor] = []

  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    self.act_calls.append(active.clone())
    num_envs = active.shape[0]
    return torch.full((num_envs, self.action_dim), self.value, device=active.device)


class SwitchAfterNCallsController(Controller):
  """Switches from skill 0 to skill 1 on the (n+1)-th call to decide, then stays."""

  def __init__(self, pool: SkillPool, n: int) -> None:
    super().__init__(pool)
    self.n = n
    self.calls = 0
    self.reset_calls: list[torch.Tensor] = []

  def decide(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
    skill_id = 0 if self.calls < self.n else 1
    self.calls += 1
    return torch.full_like(target, skill_id)

  def reset(self, mask: torch.Tensor) -> None:
    self.reset_calls.append(mask.clone())


class PatientMeta(MetaPolicy):
  """Test-double meta policy: a bridge that outputs a constant action and hands over
  after `patience` active steps, recording its begin/act calls.

  Stands in for a real architecture so the base `MetaPolicy` switching skeleton
  (fresh-adopt, switch detection, hand-over, reset absorption) can be tested without
  building any networks.
  """

  def __init__(
    self, env: ManagerBasedRlEnv, pool: SkillPool, value: float, patience: int
  ) -> None:
    self.value = value
    self.patience = patience
    self.action_dim = env.action_manager.total_action_dim
    self._counter: torch.Tensor | None = None
    self.begin_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    self.act_calls: list[torch.Tensor] = []
    super().__init__(env, pool)

  def begin_switch(
    self, switching: torch.Tensor, source: torch.Tensor, target: torch.Tensor
  ) -> None:
    self.begin_calls.append((switching.clone(), source.clone(), target.clone()))
    if self._counter is None:
      self._counter = torch.zeros_like(switching, dtype=torch.long)
    self._counter = torch.where(
      switching, torch.zeros_like(self._counter), self._counter
    )

  def bridge_step(
    self,
    obs: VecEnvObs,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    self.act_calls.append(active.clone())
    assert self._counter is not None
    self._counter = torch.where(active, self._counter + 1, self._counter)
    num_envs = active.shape[0]
    actions = torch.full((num_envs, self.action_dim), self.value, device=active.device)
    handover = active & (self._counter >= self.patience)
    return actions, handover


def _make_env(device: str, num_envs: int, episode_length_s: float) -> ManagerBasedRlEnv:
  cfg = cartpole_balance_env_cfg()
  cfg.episode_length_s = episode_length_s
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def test_skill_pool_act_batches_by_assignment(device):
  action_dim = 1
  skill_a = ConstantSkill("a", value=1.0, action_dim=action_dim)
  skill_b = ConstantSkill("b", value=-1.0, action_dim=action_dim)
  pool = SkillPool([skill_a, skill_b])

  assignment = torch.tensor([0, 1, NO_SKILL, 0], device=device)
  obs: VecEnvObs = {}
  actions = pool.act(obs, assignment)

  expected = torch.tensor([[1.0], [-1.0], [0.0], [1.0]], device=device)
  assert torch.equal(actions, expected)


def test_run_episode_switches_bridges_and_resets(device):
  num_envs = 2
  # Short enough episodes that at least one auto-reset happens during the run.
  env = _make_env(device, num_envs=num_envs, episode_length_s=0.5)  # 10 steps
  action_dim = env.action_manager.total_action_dim

  skill_0 = ConstantSkill("skill_0", value=0.5, action_dim=action_dim)
  skill_1 = ConstantSkill("skill_1", value=-0.5, action_dim=action_dim)
  pool = SkillPool([skill_0, skill_1])

  switch_after = 3
  controller = SwitchAfterNCallsController(pool, n=switch_after)
  bridge_patience = 2
  meta = PatientMeta(env, pool, value=0.0, patience=bridge_patience)

  run_episode(env, controller, meta, num_steps=14)

  # Exactly one switch fired: skill 0 -> skill 1. Post-reset re-decide calls should
  # not fire another one, since the controller already committed to skill 1.
  assert len(meta.begin_calls) == 1
  switching, source, target = meta.begin_calls[0]
  assert switching.all()
  # Control was coming from skill 0, so that -- not the episode's initial NO_SKILL --
  # is the transition's source by the time `begin_switch` is called.
  assert (source == 0).all()
  assert (target == 1).all()

  # skill_0 was driven for the `switch_after` pre-switch steps, never afterwards.
  # Unlike the old design there is no phantom initial decide, so every one of those
  # calls is a real step with skill 0 active.
  skill_0_active_counts = sum(int(m.sum()) for m in skill_0.act_calls)
  assert skill_0_active_counts == num_envs * switch_after

  # The bridge was engaged for exactly `bridge_patience` steps before handover.
  bridge_active_counts = sum(int(m.sum()) for m in meta.act_calls)
  assert bridge_active_counts == num_envs * bridge_patience

  # skill_1 took over for the remainder and pool.act still evaluated it every step.
  assert len(skill_1.act_calls) == 14

  # At least one auto-reset happened, and it was propagated to the controller.
  assert len(controller.reset_calls) >= 1
  env.close()


def test_arch0_is_a_pure_cut_over(device):
  """The no-bridge baseline must hand over instantly and drive with the target
  skill's own actions, so a run with it is indistinguishable from no bridge at all."""
  num_envs = 2
  env = _make_env(device, num_envs=num_envs, episode_length_s=100.0)
  action_dim = env.action_manager.total_action_dim

  skill_0 = ConstantSkill("skill_0", value=0.5, action_dim=action_dim)
  skill_1 = ConstantSkill("skill_1", value=-0.5, action_dim=action_dim)
  pool = SkillPool([skill_0, skill_1])
  meta = Arch0(env, pool)

  obs, _ = env.reset()
  source = torch.zeros(num_envs, dtype=torch.int64, device=device)
  target = torch.ones(num_envs, dtype=torch.int64, device=device)
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  actions, handover = meta.bridge_step(obs, source, target, active)

  # Target skill's action, and control released the very same step.
  assert torch.equal(actions, torch.full_like(actions, skill_1.value))
  assert handover.all()

  # Driven end to end it never lingers mid-bridge past the switch step.
  controller = SwitchAfterNCallsController(pool, n=3)
  run_episode(env, controller, Arch0(env, pool), num_steps=10)
  env.close()


def test_composed_policy_is_drivable_by_a_viewer(device):
  """A caller that owns `env.step` (a viewer) must be able to drive the composition.

  It never hands the done flags back, so `ComposedPolicy` has to notice episode
  boundaries itself, and its zero-argument `reset()` is what a viewer's reset button
  calls after resetting the env.
  """
  num_envs = 2
  # Short episodes, so boundaries are absorbed without anyone reporting them.
  env = _make_env(device, num_envs=num_envs, episode_length_s=0.5)
  action_dim = env.action_manager.total_action_dim

  pool = SkillPool(
    [
      ConstantSkill("skill_0", value=0.5, action_dim=action_dim),
      ConstantSkill("skill_1", value=-0.5, action_dim=action_dim),
    ]
  )
  controller = SwitchAfterNCallsController(pool, n=3)
  meta = PatientMeta(env, pool, value=0.0, patience=2)

  obs, _ = env.reset()
  policy = ComposedPolicy(env, controller, meta)
  for _ in range(12):
    obs, _, _, _, _ = env.step(policy(obs))

  assert len(meta.begin_calls) == 1
  assert meta.target.shape == (num_envs,)

  # The reset button: env first, then the composition, which must start over rather
  # than resume mid-transition.
  env.reset()
  policy.reset()
  assert not meta._bridging.any()
  assert (meta._source == NO_SKILL).all()
  assert (meta.target == NO_SKILL).all()

  # ...and it keeps running afterwards.
  obs = env.observation_manager.compute()
  for _ in range(5):
    obs, _, _, _, _ = env.step(policy(obs))
  env.close()
