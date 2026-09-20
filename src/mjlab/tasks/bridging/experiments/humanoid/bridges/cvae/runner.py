"""Train one CVAE student from balanced, source-specific tracker groups."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import torch
from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.pool import (
  PooledEnv,
  RoutedTeacher,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.teachers import (
  DEFAULT_MANIFEST,
  Teacher,
  load_teachers,
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
  """Run online DAgger with one frozen tracker per parallel group."""

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
    manifest = Path(train_cfg.pop("teacher_manifest", DEFAULT_MANIFEST))
    self._teachers: tuple[Teacher, ...] = ()
    self._pool: PooledEnv | None = None
    if log_dir is not None:
      self._teachers = load_teachers(manifest, validate=True)
      motion_cfg = env.unwrapped.cfg.commands["motion"]
      requested = Path(getattr(motion_cfg, "motion_file", ""))
      if requested.resolve() != self._teachers[0].motion.resolve():
        raise ValueError("The first CVAE motion must match the first teacher")
      groups = [env]
      for index, teacher in enumerate(self._teachers[1:], start=1):
        cfg = copy.deepcopy(env.unwrapped.cfg)
        cfg.commands["motion"].motion_file = str(teacher.motion)
        cfg.commands["bridge"].sources = (teacher.source,)
        if cfg.seed is not None:
          cfg.seed += index
        group = ManagerBasedRlEnv(cfg, device=device)
        groups.append(RslRlVecEnvWrapper(group, clip_actions=env.clip_actions))
      self._pool = PooledEnv(tuple(groups))
    super().__init__(self._pool or env, train_cfg, log_dir, device)
    if self._pool is not None:
      self._load_teachers()

  def _load_teachers(self) -> None:
    algorithm = cast(Distillation, self.alg)
    template = algorithm.teacher
    models = []
    for teacher in self._teachers:
      model = copy.deepcopy(template)
      model.load_state_dict(_actor_state(teacher.checkpoint), strict=True)
      model.eval()
      models.append(model)
      print(f"[cvae] {teacher.source}: {teacher.checkpoint}")
    assert self._pool is not None
    algorithm.teacher = cast(
      MLPModel, RoutedTeacher(models, self._pool.group_size).to(self.device)
    )
    algorithm.teacher_loaded = True

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if not cast(Distillation, self.alg).teacher_loaded:
      raise ValueError("Training needs a validated CVAE teacher manifest")
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      if self._pool is not None:
        self._pool.close()

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if (load_cfg or {}).get("actor"):
      load_cfg = {"student": True, "iteration": False}
    infos = super().load(path, load_cfg, strict, map_location)
    if self._pool is not None and load_cfg is None:
      step = self._pool.unwrapped.common_step_counter
      for group in self._pool.groups[1:]:
        group.unwrapped.common_step_counter = step
    return infos
