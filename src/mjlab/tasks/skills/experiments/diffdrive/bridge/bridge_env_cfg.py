"""The Mjlab-Bridge-Diffdrive task: train a bridge to join the next skill's tube.

The robot lives on a bare plane (the corridors are scored analytically, so no wall
geometry is needed for physics). Each episode the BridgeCommand term drops it at a
harvested interrupt state and names a goal in the next skill's window; the policy is
rewarded for sliding into that goal without leaving the corridors. The trained policy
is exported to ONNX on every checkpoint so the deployed LearnedBridge can load it.

    uv run train Mjlab-Bridge-Diffdrive --env.scene.num-envs 4096
    uv run play Mjlab-Bridge-Diffdrive --checkpoint-file <logdir>/<name>.onnx
"""

from __future__ import annotations

from typing import cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.skills.config.diffdrive.diffdrive_env_cfg import get_diffdrive_cfg
from mjlab.tasks.skills.experiments.diffdrive.bridge import mdp
from mjlab.tasks.skills.experiments.diffdrive.experiment import CONFIG
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

ENTITY = "robot"


def _export_selector_to_onnx(
  selector: mdp.MergeSelector, in_dim: int, path: str, filename: str
) -> None:
  """Export the selector net to ONNX (a copy, so training keeps its live net on device)."""
  import copy
  import os

  import torch

  net = copy.deepcopy(selector).to("cpu").eval()
  os.makedirs(path, exist_ok=True)
  torch.onnx.export(
    net,
    (torch.zeros(1, in_dim),),
    os.path.join(path, filename),
    export_params=True,
    opset_version=18,
    input_names=["obs"],
    output_names=["scores"],
    dynamic_axes={},
    dynamo=False,
  )


class BridgeOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner that co-trains the selector and exports the actor and selector to ONNX.

  Each iteration, right after the PPO update (where autograd is enabled, unlike the
  inference-mode rollout), it drives the selector's own REINFORCE step. The actor and the
  selector are both exported on every checkpoint, so the deployed bridge can pick a merge
  frame with the selector and then drive to it with the actor.
  """

  def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self._bridge = cast(
      mdp.BridgeCommand,
      self.env.unwrapped.command_manager.get_term(mdp.COMMAND_NAME),
    )
    base_update = self.alg.update

    def update() -> dict:
      losses = base_update()
      losses.update(self._bridge.update_selector())
      return losses

    # Replace the PPO update with the wrapped one. The dynamic set is deliberate: it keeps
    # the monkeypatch off the type checker, which would otherwise read a plain assignment
    # as shadowing the base method.
    setattr(self.alg, "update", update)  # noqa: B010

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    policy_dir, filename, _ = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      _export_selector_to_onnx(
        self._bridge.selector,
        self._bridge.window_dim,
        str(policy_dir),
        filename.replace(".onnx", "_selector.onnx"),
      )
    except Exception as exc:  # export must never break training
      print(f"[WARN] ONNX export failed (training continues): {exc}")


def bridge_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  observations = {
    "actor": ObservationGroupCfg(
      {"bridge": ObservationTermCfg(func=mdp.bridge_observation)},
    ),
    "critic": ObservationGroupCfg(
      {"bridge": ObservationTermCfg(func=mdp.bridge_observation)},
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "twist": mdp.TwistActionCfg(entity_name=ENTITY),
  }

  commands: dict[str, CommandTermCfg] = {
    "bridge": mdp.BridgeCommandCfg(entity_name=ENTITY),
  }

  rewards = {
    "tracking": RewardTermCfg(func=mdp.tracking, weight=CONFIG.track_weight),
    "effort": RewardTermCfg(func=mdp.effort, weight=-CONFIG.effort_weight),
  }

  terminations = {
    "tracked_to_end": TerminationTermCfg(func=mdp.tracked_to_end),
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={ENTITY: get_diffdrive_cfg()},
      num_envs=1,
      env_spacing=12.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name=ENTITY,
      body_name="base",
      distance=6.0,
      elevation=-75.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=CONFIG.timestep, integrator="implicitfast"),
    ),
    decimation=CONFIG.decimation,
    episode_length_s=4.0,
  )
  if play:
    cfg.episode_length_s = 1.0e10
  return cfg


def bridge_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="bridge_diffdrive",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )
