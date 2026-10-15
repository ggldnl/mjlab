"""Registered transition tasks using mjlab launchers and the RSL logging stack."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from rsl_rl.utils.logger import Logger

from mjlab.rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.bridging.baseline.evaluate import load_for_scene
from mjlab.tasks.bridging.baseline.model import TransitionPolicy
from mjlab.tasks.bridging.baseline.play import Controller, PlayCfg
from mjlab.tasks.bridging.baseline.scene import (
  Pair,
  SceneCfg,
  TransitionScene,
  scene_env_cfg,
)
from mjlab.tasks.bridging.baseline.train import LearningCfg, TrainCfg, train
from mjlab.tasks.registry import register_mjlab_task


@dataclass
class TransitionRunnerCfg(RslRlBaseRunnerCfg):
  seed: int = 0
  max_iterations: int = 1000
  save_interval: int = 25
  transition: LearningCfg = field(default_factory=LearningCfg)


class TransitionRunner:
  """Keep the proximity reward PPO update and use standard logging and playback."""

  # The motion reference belongs to the frozen incoming skill
  requires_motion_artifact = False

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise ValueError("Transition training currently supports one GPU per run")
    self.env, self.cfg, self.device = env, train_cfg, device
    settings = dict(train_cfg["transition"])
    scene_settings = dict(settings["scene"])
    scene_settings["device"] = device
    settings["scene"] = SceneCfg(**scene_settings)
    self.settings = LearningCfg(**settings)
    if train_cfg["experiment_name"] != f"g1_{self.settings.scene.pair}_transition":
      raise ValueError("Select the task for the requested skill pair")
    self.scene: TransitionScene | None = None
    self.model: TransitionPolicy | None = None
    self.metadata: dict = {}
    self.checkpoint = Path("model.pt")
    self.current_learning_iteration = 0
    logger_cfg = {**train_cfg, "algorithm": {"rnd_cfg": None}}
    self.logger = Logger(
      log_dir=log_dir,
      cfg=logger_cfg,
      env_cfg=env.cfg,
      num_envs=env.num_envs,
      is_distributed=False,
      gpu_world_size=1,
      gpu_global_rank=0,
      device=device,
    )

  def add_git_repo_to_log(self, path: str) -> None:
    self.logger.git_status_repos.append(path)

  def _scene(self, cfg: SceneCfg) -> TransitionScene:
    return TransitionScene(cfg, self.env.num_envs, wrapped=self.env)

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    del init_at_random_ep_len
    if self.model is not None:
      raise ValueError(
        "These checkpoints contain inference weights, not optimizer and replay state. "
        "Start a new run rather than using --agent.resume"
      )
    if self.logger.log_dir is None:
      raise ValueError("Training requires a log directory")
    self.scene = self._scene(self.settings.scene)
    values = asdict(self.settings)
    values["scene"] = self.scene.cfg
    cfg = TrainCfg(
      **values,
      num_envs=self.env.num_envs,
      iterations=num_learning_iterations,
      seed=self.cfg["seed"],
      save_interval=self.cfg["save_interval"],
      output=Path(self.logger.log_dir) / "model.pt",
    )
    self.logger.init_logging_writer()
    try:
      train(cfg, scene=self.scene, on_update=self._log, on_save=self._save_update)
    finally:
      self.logger.stop_logging_writer()

  def _log(self, record: dict, model: TransitionPolicy) -> None:
    assert self.scene is not None
    self.model = model
    self.current_learning_iteration = record["iteration"]
    self.logger.cfg["num_steps_per_env"] = record["physics_steps"]
    self.logger.ep_extras.append(
      {
        "Transition/success_rate": record["success_rate"],
        "Transition/handoff_jump_rad": record["action_jump_rad"],
        "Transition/trigger_distance_m": record["distance"],
        "Transition/duration_s": record["duration_s"],
        **{
          f"Target/{name}": value
          for name, value in zip(
            self.scene.run.command.error_names, record["target_errors"], strict=True
          )
        },
        **{
          f"Downstream/{name}": value
          for name, value in zip(
            self.scene.channels, record["downstream_rms"], strict=True
          )
        },
      }
    )
    self.logger.rewbuffer.append(record["reward_mean"])
    self.logger.lenbuffer.append(record["transition_steps"])
    self.logger.log(
      it=record["iteration"] - 1,
      start_it=0,
      total_it=self.cfg["max_iterations"],
      collect_time=record["collect_s"],
      learn_time=record["learn_s"],
      loss_dict={"ppo": record["ppo_loss"], "proximity": record["proximity_loss"]},
      learning_rate=self.settings.learning_rate,
      action_std=model.log_std.detach().clamp(-5, 1).exp(),
      rnd_weight=None,
    )

  def _save_update(self, model: TransitionPolicy, metadata: dict) -> None:
    self.model, self.metadata = model, metadata
    assert self.logger.log_dir is not None
    self.save(str(Path(self.logger.log_dir) / f"model_{metadata['iteration']}.pt"))

  def save(self, path: str, infos=None) -> None:
    del infos
    if self.model is None:
      raise ValueError("No transition policy to save")
    self.model.save(Path(path), self.metadata)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self, path: str, load_cfg=None, strict: bool = True, map_location=None
  ) -> dict:
    del strict, map_location
    if load_cfg is None:
      raise ValueError("Transition checkpoints support inference, not training resume")
    _, metadata = TransitionPolicy.load(Path(path), self.device)
    saved = metadata["config"]
    scene_values = dict(saved["scene"])
    for name in ("source_checkpoint", "target_checkpoint", "selector_path"):
      scene_values[name] = Path(scene_values[name])
    scene_values["device"] = self.device
    if scene_values["pair"] != self.settings.scene.pair:
      raise ValueError("Checkpoint skill pair differs from the selected task")
    entries = saved.get("entries", [scene_values["entry"]])
    if scene_values["entry"] not in entries:
      scene_values["entry"] = entries[0]
    settings = {
      item.name: saved.get(item.name, getattr(self.settings, item.name))
      for item in fields(LearningCfg)
    }
    settings["scene"] = SceneCfg(**scene_values)
    self.settings = LearningCfg(**settings)
    self.model, scene_cfg, self.metadata = load_for_scene(
      Path(path), self.settings.scene
    )
    self.scene = self._scene(scene_cfg)
    self.current_learning_iteration = self.metadata["iteration"]
    self.checkpoint = Path(path)
    return self.metadata

  def get_inference_policy(self, device=None) -> Controller:
    del device
    if self.model is None or self.scene is None:
      raise ValueError("Load a transition checkpoint before playback")
    self.env.unwrapped.manager_visualizers["transition_target"] = self.scene.run
    return Controller(
      self.scene,
      self.model,
      PlayCfg(
        checkpoint=self.checkpoint,
        scene=self.scene.cfg,
        seed=self.cfg["seed"],
        distance=self.settings.distances[len(self.settings.distances) // 2],
        duration_s=self.settings.durations[len(self.settings.durations) // 2],
      ),
    )

  def set_eval_video_callback(self, *args, **kwargs) -> None:
    raise ValueError(
      "Use --video for training recordings or play --video for this controller"
    )


TASKS: dict[Pair, str] = {
  "walk2jump": "Mjlab-G1-Walk2Jump-Transition",
  "walk2kick": "Mjlab-G1-Walk2Kick-Transition",
  "jump2walk": "Mjlab-G1-Jump2Walk-Transition",
}

for _pair, _task_id in TASKS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=scene_env_cfg(_pair),
    play_env_cfg=scene_env_cfg(_pair, num_envs=1),
    rl_cfg=TransitionRunnerCfg(
      experiment_name=f"g1_{_pair}_transition",
      transition=LearningCfg(scene=SceneCfg(pair=_pair)),
    ),
    runner_cls=TransitionRunner,
  )
