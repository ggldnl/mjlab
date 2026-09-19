"""Runner that loads a trajectory tracker as the DAgger teacher."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from rsl_rl.algorithms import Distillation

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker import (
  _trained_on,
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


class CvaeRunner(MjlabOnPolicyRunner):
  """Run online DAgger with a frozen tracker policy."""

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
    self.teacher_checkpoint = train_cfg.pop("teacher_checkpoint", None)
    super().__init__(env, train_cfg, log_dir, device)
    if self.teacher_checkpoint is not None:
      self._load_teacher(Path(self.teacher_checkpoint))

  def _load_teacher(self, checkpoint: Path) -> None:
    if not checkpoint.exists():
      raise FileNotFoundError(f"No teacher checkpoint at {checkpoint}")
    trained_on = _trained_on(checkpoint.parent)
    motion_cfg = self.env.unwrapped.cfg.commands["motion"]
    requested = Path(getattr(motion_cfg, "motion_file", ""))
    if (
      trained_on is not None
      and requested
      and trained_on.resolve() != requested.resolve()
    ):
      raise ValueError(
        f"Teacher was trained on {trained_on}, but this run requests {requested}"
      )
    algorithm = cast(Distillation, self.alg)
    algorithm.teacher.load_state_dict(_actor_state(checkpoint), strict=True)
    algorithm.teacher_loaded = True
    print(f"[cvae] teacher: {checkpoint}")

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if not cast(Distillation, self.alg).teacher_loaded:
      raise ValueError(
        "Training needs --agent.teacher-checkpoint pointing to the tracker trained "
        "for --env.commands.motion.motion-file"
      )
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if (load_cfg or {}).get("actor"):
      load_cfg = {"student": True, "iteration": False}
    return super().load(path, load_cfg, strict, map_location)
