"""Smoke tests for the diffdrive experiment: env cfgs, controller FSM, success_fns,
task registration, and the KUKA arm asset (shared scaffolding for table_tennis)."""

import math
from typing import cast

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_0 import (
  DRIVE_STRAIGHT,
  TURN,
  DiffdriveController,
)
from mjlab.tasks.skills.architectures.arch_0 import (
  drive_straight_env_cfg,
  turn_env_cfg,
)
from mjlab.tasks.skills.architectures.arch_0 import (
  straight_success_fn,
  turn_success_fn,
)
from mjlab.tasks.skills.skill import NO_SKILL, Skill, SkillPool

NUM_ENVS = 4


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.mark.parametrize("cfg_fn", [drive_straight_env_cfg, turn_env_cfg])
def test_diffdrive_env_cfg_builds_and_steps(device: str, cfg_fn):
  cfg = cfg_fn()
  cfg.scene.num_envs = NUM_ENVS
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  obs, _ = env.reset()
  action_dim = env.action_manager.total_action_dim
  assert action_dim == 2  # left_wheel, right_wheel.

  action = torch.zeros(NUM_ENVS, action_dim, device=device)
  for _ in range(19):
    env.step(action)
  obs, reward, terminated, time_out, _ = env.step(action)

  actor_obs = obs["actor"]
  assert isinstance(actor_obs, torch.Tensor)
  assert torch.isfinite(actor_obs).all()
  assert torch.isfinite(reward).all()
  assert terminated.dtype == torch.bool
  assert time_out.dtype == torch.bool
  env.close()


def test_diffdrive_env_cfg_play_mode_has_no_corruption():
  cfg = drive_straight_env_cfg(play=True)
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.episode_length_s > 1e9


def test_diffdrive_tasks_are_registered():
  from mjlab.tasks.registry import list_tasks

  tasks = list_tasks()
  assert "Mjlab-Diffdrive-DriveStraight" in tasks
  assert "Mjlab-Diffdrive-Turn" in tasks


class _ConstantSkill(Skill):
  def __init__(self, name: str) -> None:
    self.name = name

  def act(self, obs, active: torch.Tensor) -> torch.Tensor:
    del obs
    return torch.zeros(active.shape[0], 2, device=active.device)


def test_diffdrive_controller_alternates_on_fixed_timers():
  pool = SkillPool([_ConstantSkill("drive_straight"), _ConstantSkill("turn")])
  controller = DiffdriveController(pool, straight_steps=3, turn_steps=2)

  target = torch.full((2,), NO_SKILL, dtype=torch.int64)
  history = []
  for _ in range(9):
    target = controller.decide(cast(ManagerBasedRlEnv, None), target)
    history.append(target.clone())

  # Ignore the very first (slightly short) phase; every phase after that must
  # last exactly `straight_steps`/`turn_steps` calls and alternate skill ids.
  values = torch.stack(history)[:, 0].tolist()
  # First switch to TURN happens once the initial phase's timer expires.
  first_turn_idx = values.index(TURN)
  assert values[first_turn_idx : first_turn_idx + 2] == [TURN, TURN]
  assert values[first_turn_idx + 2 : first_turn_idx + 5] == [
    DRIVE_STRAIGHT,
    DRIVE_STRAIGHT,
    DRIVE_STRAIGHT,
  ]


def test_diffdrive_controller_reset_forces_drive_straight():
  pool = SkillPool([_ConstantSkill("drive_straight"), _ConstantSkill("turn")])
  controller = DiffdriveController(pool, straight_steps=1, turn_steps=100)

  target = torch.full((2,), NO_SKILL, dtype=torch.int64)
  target = controller.decide(cast(ManagerBasedRlEnv, None), target)
  target = controller.decide(cast(ManagerBasedRlEnv, None), target)
  assert target[0].item() == TURN

  controller.reset(torch.tensor([True, False]))
  target = controller.decide(cast(ManagerBasedRlEnv, None), target)
  assert target[0].item() == DRIVE_STRAIGHT


def test_success_fns_run_on_a_real_env(device: str):
  cfg = turn_env_cfg()
  cfg.scene.num_envs = NUM_ENVS
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()

  turn_success = turn_success_fn(env)
  straight_success = straight_success_fn(env)
  assert turn_success.shape == (NUM_ENVS,)
  assert straight_success.shape == (NUM_ENVS,)
  assert turn_success.dtype == torch.bool
  assert straight_success.dtype == torch.bool
  env.close()


def test_kuka_arm_compiles_with_attachment_site():

  from mjlab.asset_zoo.robots.kuka_iiwa_14.kuka_constants import get_kuka_robot_cfg
  from mjlab.entity import Entity

  robot = Entity(get_kuka_robot_cfg())
  model = robot.spec.compile()
  assert model.nq == 7
  assert model.nu == 7

  site_id = model.site("attachment_site").id
  body_id = model.site_bodyid[site_id]
  assert model.body(body_id).name == "link7"
  assert math.isclose(float(model.site_pos[site_id][2]), 0.045, abs_tol=1e-6)
