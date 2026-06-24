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

from typing import Any

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


def _export_to_onnx(module: Any, path: str, filename: str) -> None:
  """Export an rsl_rl MLP model (actor or critic) to ONNX, like the actor export."""
  import os

  import torch

  onnx_model = module.as_onnx(verbose=False)
  onnx_model.to("cpu")
  onnx_model.eval()
  os.makedirs(path, exist_ok=True)
  torch.onnx.export(
    onnx_model,
    onnx_model.get_dummy_inputs(),
    os.path.join(path, filename),
    export_params=True,
    opset_version=18,
    input_names=onnx_model.input_names,
    output_names=onnx_model.output_names,
    dynamic_axes={},
    dynamo=False,
  )


class BridgeOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner that also exports the actor and critic to ONNX on every checkpoint.

  The critic is exported too because the deployed bridge selects its merge target by
  reading the critic's value over the candidate states of the next skill's tube.
  """

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    policy_dir, filename, _ = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      _export_to_onnx(
        self.alg.critic, str(policy_dir), filename.replace(".onnx", "_critic.onnx")
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
