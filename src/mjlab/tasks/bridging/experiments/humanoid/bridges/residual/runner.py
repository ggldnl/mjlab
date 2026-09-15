"""Runner that keeps the trained imitation bridge frozen under the residual."""

from __future__ import annotations

from dataclasses import asdict

from rsl_rl.env import VecEnv

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  find_checkpoint,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation import (
  IMITATION_EXPERIMENT,
  imitation_ppo_runner_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.residual.mdp import (
  ResidualJointPositionAction,
)


class ResidualRunner(MjlabOnPolicyRunner):
  """Train one actor while a second, pretrained actor supplies the coarse motion."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    obs_groups = train_cfg.get("obs_groups", {}).get("actor", ("actor",))
    super().__init__(env, train_cfg, log_dir, device)
    if not isinstance(env, RslRlVecEnvWrapper):
      raise TypeError("ResidualRunner requires RslRlVecEnvWrapper")
    action = env.unwrapped.action_manager.get_term("joint_pos")
    if not isinstance(action, ResidualJointPositionAction):
      raise TypeError("ResidualRunner requires ResidualJointPositionAction")

    checkpoint = find_checkpoint(
      (IMITATION_EXPERIMENT,),
      str(action.cfg.imitation_checkpoint)
      if action.cfg.imitation_checkpoint is not None
      else None,
      hint=" Train the imitation bridge first",
    )
    base_cfg = asdict(imitation_ppo_runner_cfg())
    base_cfg["obs_groups"] = {"actor": obs_groups, "critic": obs_groups}
    self.imitation_runner = MjlabOnPolicyRunner(env, base_cfg, device=device)
    self.imitation_runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    action.set_base_policy(self.imitation_runner.get_inference_policy(device=device))
    print(f"imitation {checkpoint}")

  def save(self, path: str, infos: dict | None = None) -> None:
    """Keep the exact frozen base actor beside the correction actor."""
    base = self.imitation_runner.alg.get_policy().state_dict()
    super().save(path, {**(infos or {}), "imitation_actor_state_dict": base})

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    state = infos.get("imitation_actor_state_dict") if infos else None
    if state is not None:
      self.imitation_runner.alg.get_policy().load_state_dict(state, strict=True)
    return infos
