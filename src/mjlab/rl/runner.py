import copy
import os
from pathlib import Path
from typing import cast

import torch
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  MODEL_KEYS: tuple[str, ...] = ("actor", "critic")
  """Which config entries hold model configs. Distillation names its two differently."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in self.MODEL_KEYS:
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore common_step_counter to preserve curricula state.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    # A distilled policy is somebody's actor. The student is what gets deployed, and every
    # inference path here asks for an actor: play, and the composition arena's Policy. They
    # are asking for "the thing that acts", which in a distillation checkpoint is under
    # another name, so the name is what is fixed rather than each caller.
    #
    # Read side only. Writing an actor_state_dict into these files instead would be the
    # shorter fix and a trap: rsl-rl decides a checkpoint came out of reinforcement learning
    # by looking for that key, so a distillation resuming from its own checkpoint would load
    # the student's weights into the teacher and carry on distilling against them.
    if (load_cfg or {}).get("actor") and "actor_state_dict" not in loaded_dict:
      if "student_state_dict" in loaded_dict:
        loaded_dict["actor_state_dict"] = loaded_dict["student_state_dict"]

    # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if load_iteration and infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos


def _keep_writer() -> None:
  """Stand in for the logger's writer setup and teardown, so a writer outlives one phase.

  Installed over both ends. A phase change reruns the learning loop, and the loop opens a
  writer on entry and closes it on exit; the writer belongs to the run, not to a phase."""


class MjlabTeacherStudentRunner(MjlabOnPolicyRunner):
  """Train with privileged observations, then distil into a policy that runs without them.

  One run, two phases, one checkpoint at the end holding a student that any inference path
  can load as an actor. See RslRlTeacherStudentRunnerCfg for what the phases are.

  Which shape this runner is built in depends on what it is for, and it decides that from
  how it is used rather than from a flag:

    constructed       the deployment shape. A PPO-shaped actor reading the "actor" group,
                      which is the student's observation. That is what `play` and the
                      composition arena want, and neither of them calls `learn`
    learning          `learn` rebuilds the algorithm for phase one, runs it, rebuilds again
                      for phase two, and runs that

  Rebuilding rather than reconfiguring, because the two phases are not the same shape:
  different algorithms, different models, different observations. RSL-RL builds all of that
  in one call from a config dict, so the honest way to change phase is to build the next one.
  """

  MODEL_KEYS: tuple[str, ...] = ("actor", "critic", "teacher")

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Kept before anything runs, because constructing an algorithm consumes the config it is
    # given: RSL-RL pops the class names out of the model and algorithm sections and
    # rewrites obs_groups in place. A second phase built from the leftovers would fail
    self._blueprint = copy.deepcopy(train_cfg)
    self._resume: tuple[str, str | None, bool] | None = None
    super().__init__(env, train_cfg, log_dir, device)

  ##
  # Phases.
  ##

  def _rebuild(self, section: dict) -> None:
    """Swap in a freshly built algorithm, described by `section`.

    `section` names the algorithm config, the models it needs and the observation groups
    they read. Everything else is carried over from the blueprint, so the runner keeps its
    step count, its save interval and its logging across the change.
    """
    cfg = copy.deepcopy(self._blueprint)
    # The blueprint is taken before the base class runs, so it is missing what that injects:
    # multi_gpu, which construct_algorithm reads. Anything the runner added and the blueprint
    # does not know about is carried over, with the blueprint winning where both have a key
    for key, value in self.cfg.items():
      cfg.setdefault(key, value)
    cfg.update(section)
    for key in self.MODEL_KEYS + ("student",):
      if key not in cfg:
        continue
      for opt in ("cnn_cfg", "distribution_cfg"):
        if cfg[key].get(opt) is None:
          cfg[key].pop(opt, None)
      if cfg[key].get("rnn_type") is None:
        for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
          cfg[key].pop(opt, None)
    self.cfg = cfg
    alg_class = cast("type[PPO]", resolve_callable(cfg["algorithm"]["class_name"]))
    self.alg = alg_class.construct_algorithm(
      self.env.get_observations(), self.env, cfg, self.device
    )

  def _tracking_section(self) -> dict:
    """Phase one: PPO, with the actor reading the teacher's observation group."""
    plan = self._blueprint
    return {
      "algorithm": copy.deepcopy(plan["algorithm"]),
      "actor": copy.deepcopy(plan["teacher"]),
      "critic": copy.deepcopy(plan["critic"]),
      "obs_groups": {"actor": [plan["teacher_obs_group"]], "critic": ["critic"]},
    }

  def _distillation_section(self) -> dict:
    """Phase two: the student reads "actor", the teacher reads what it was trained on."""
    plan = self._blueprint
    return {
      "algorithm": copy.deepcopy(plan["distillation"]),
      "student": copy.deepcopy(plan["actor"]),
      "teacher": copy.deepcopy(plan["teacher"]),
      "obs_groups": {"student": ["actor"], "teacher": [plan["teacher_obs_group"]]},
    }

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Run phase one, hand its actor over as the teacher, run phase two."""
    tracking = int(self._blueprint["tracking_iterations"])
    resume_at, past_tracking = self._resume_point()
    trained = None

    if not past_tracking:
      self._rebuild(self._tracking_section())
      self._apply_resume()
      left = max(tracking - max(resume_at, 0), 0)
      print(f"[INFO]: Phase 1 of 2, tracking. {left} iterations.")
      if left:
        # Phase one must not close the writer on its way out. rsl-rl closes it at the end
        # of every learn, and for W&B closing means wandb.finish, after which the module
        # swaps wandb.log for a stub that raises: phase two would inherit a dead writer and
        # die on its first logged iteration. Closed once instead, at the end of phase two
        self.logger.stop_logging_writer = _keep_writer  # ty: ignore[invalid-assignment]
        try:
          super().learn(left, init_at_random_ep_len)
        finally:
          del self.logger.stop_logging_writer
      tracker = self.alg
      assert isinstance(tracker, PPO)
      trained = tracker.actor.state_dict()
      self.current_learning_iteration = max(self.current_learning_iteration, tracking)

    self._rebuild(self._distillation_section())
    if trained is not None:
      # Straight across, with no file in between. The teacher is the network phase one just
      # finished training, built from the same config against the same observation group, so
      # the two state dicts are the same shape by construction
      distillation = self.alg
      assert isinstance(distillation, Distillation)
      distillation.teacher.load_state_dict(trained)
      distillation.teacher_loaded = True
    else:
      self._apply_resume()

    done = max(self.current_learning_iteration, tracking)
    left = max(num_learning_iterations - done, 0)
    print(f"[INFO]: Phase 2 of 2, distillation. {left} iterations.")
    if left:
      # Phase one already opened the writer on this run's directory, and opening a second
      # one would be a duplicate events file at best and a second W&B run at worst. Unless
      # phase one did not run at all, which is what a resume straight into phase two looks
      # like, and then there is nothing open yet to protect
      if getattr(self.logger, "writer", None) is not None:
        self.logger.init_logging_writer = _keep_writer  # ty: ignore[invalid-assignment]
      super().learn(left, init_at_random_ep_len=False)
    elif getattr(self.logger, "writer", None) is not None:
      # There is no phase two to run, so the loop that would have closed phase one's
      # writer never runs either. Close it here, or a W&B run is left open
      self.logger.stop_logging_writer()

  ##
  # Checkpoints.
  ##

  def _resume_point(self) -> tuple[int, bool]:
    """Where a resume lands: which iteration, and whether it is already past phase one."""
    if self._resume is None:
      return 0, False
    path, map_location, _ = self._resume
    saved = torch.load(path, map_location=map_location, weights_only=False)
    return int(saved.get("iter", 0)), "student_state_dict" in saved

  def _apply_resume(self) -> None:
    """Read the deferred checkpoint into the phase that has just been built."""
    if self._resume is None:
      return
    path, map_location, strict = self._resume
    print(f"[INFO]: Resuming from: {path}")
    self._resume = None
    super().load(path, None, strict, map_location)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load now for inference, or defer to `learn` when resuming a training run.

    The two callers want different things and are already telling them apart. Inference asks
    for an actor by name, and this runner is constructed in the deployment shape, so that one
    is read straight away. A resume asks for everything and cannot be served yet: which
    models to read into depends on which phase the checkpoint came from, and no phase has
    been built at the point `train` calls this.
    """
    if load_cfg is None:
      self._resume = (path, map_location, strict)
      return {}
    if load_cfg.get("actor"):
      saved = torch.load(path, map_location=map_location, weights_only=False)
      if "student_state_dict" not in saved:
        raise ValueError(
          f"{path} was saved during the tracking phase. It holds a teacher that reads the "
          f"privileged observation and no student, so there is no policy in it to deploy. "
          f"Train past tracking_iterations, or name a later checkpoint."
        )
    return super().load(path, load_cfg, strict, map_location)
