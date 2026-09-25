"""Load one frozen oracle as the DAgger teacher."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from rsl_rl.algorithms import Distillation

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.algorithm import (
  GoalDistillation,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  upper_body_mask,
)


def _actor_state(checkpoint: Path) -> dict[str, torch.Tensor]:
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  if "actor_state_dict" in saved:
    state = saved["actor_state_dict"]
  elif "model_state_dict" in saved:
    state = {
      key.replace("actor.", "mlp.").replace(
        "actor_obs_normalizer.", "obs_normalizer."
      ): value
      for key, value in saved["model_state_dict"].items()
      if key.startswith(("actor.", "actor_obs_normalizer."))
      or key in ("std", "log_std")
    }
  else:
    raise ValueError(f"{checkpoint} has no tracker actor")
  if "std" in state:
    state["distribution.std_param"] = state.pop("std")
  if "log_std" in state:
    state["distribution.log_std_param"] = state.pop("log_std")
  return state


class GoalRunner(MjlabOnPolicyRunner):
  """Train the goal CVAE only against an explicitly selected oracle checkpoint."""

  MODEL_KEYS = ("student", "teacher")

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir=None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    del registry_name
    checkpoint = Path(train_cfg.pop("teacher_checkpoint", ""))
    super().__init__(env, train_cfg, log_dir, device)
    algorithm = cast(GoalDistillation, self.alg)
    robot = env.unwrapped.scene["robot"]
    upper = upper_body_mask(tuple(robot.joint_names), device)
    algorithm.action_weights[upper] = algorithm.upper_action_weight
    if log_dir is not None:
      if not checkpoint.is_file():
        raise FileNotFoundError(
          f"Set --agent.teacher-checkpoint to a trained oracle checkpoint: {checkpoint}"
        )
      algorithm.teacher.load_state_dict(_actor_state(checkpoint), strict=True)
      algorithm.teacher.eval()
      algorithm.teacher_loaded = True
      print(f"[goal] oracle teacher: {checkpoint}")

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if not cast(Distillation, self.alg).teacher_loaded:
      raise ValueError("Goal CVAE training requires a trained oracle checkpoint")
    super().learn(num_learning_iterations, init_at_random_ep_len)
