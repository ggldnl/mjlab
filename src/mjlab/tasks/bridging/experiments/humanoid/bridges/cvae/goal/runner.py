"""Load one frozen oracle as the DAgger teacher."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from rsl_rl.algorithms import Distillation

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.algorithm import (
  GoalDistillation,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.runner import (
  _actor_state,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  upper_body_mask,
)


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
