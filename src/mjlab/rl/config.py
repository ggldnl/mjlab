"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple


@dataclass
class RslRlModelCfg:
  """Config for a single neural network model (Actor or Critic)."""

  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the network."""
  activation: str = "elu"
  """The activation function."""
  obs_normalization: bool = False
  """Whether to normalize the observations. Default is False."""
  cnn_cfg: dict[str, Any] | None = None
  """CNN encoder config. When set, class_name should be "CNNModel".

  Passed to ``rsl_rl.modules.CNN``. Common keys: output_channels,
  kernel_size, stride, padding, activation, global_pool, max_pool.
  """
  distribution_cfg: dict[str, Any] | None = None
  """Distribution config dict passed to rsl_rl. Example::

    {"class_name": "GaussianDistribution",
     "init_std": 1.0, "std_type": "scalar"}

  ``None`` means deterministic output (use for critic).
  """
  rnn_type: str | None = None
  """RNN type ("lstm" or "gru"). When set, class_name should be "RNNModel"."""
  rnn_hidden_dim: int = 256
  """Hidden state dimension for the RNN."""
  rnn_num_layers: int = 1
  """Number of stacked RNN layers."""
  class_name: str = "MLPModel"
  """Model class name resolved by RSL-RL (MLPModel, CNNModel, or RNNModel)."""


@dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm."""

  num_learning_epochs: int = 5
  """The number of learning epochs per update."""
  num_mini_batches: int = 4
  """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
  learning_rate: float = 1e-3
  """The learning rate."""
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  """The learning rate schedule."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  entropy_coef: float = 0.005
  """The coefficient for the entropy loss."""
  desired_kl: float = 0.01
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  value_loss_coef: float = 1.0
  """The coefficient for the value loss."""
  use_clipped_value_loss: bool = True
  """Whether to use clipped value loss."""
  clip_param: float = 0.2
  """The clipping parameter for the policy."""
  normalize_advantage_per_mini_batch: bool = False
  """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """The optimizer to use."""
  share_cnn_encoders: bool = False
  """Share CNN encoders between actor and critic."""
  class_name: str = "PPO"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlDistillationAlgorithmCfg:
  """Config for supervised distillation of a trained teacher into a student.

  The student acts, the teacher labels the states the student visits, and the loss is
  regression onto the teacher's action. That is DAgger rather than behaviour cloning, and
  the difference is the whole point: a student trained on the teacher's own rollouts is
  never shown the states its own mistakes lead to.
  """

  num_learning_epochs: int = 1
  """Passes over each batch. One is usual: the data is on-policy and thrown away."""
  gradient_length: int = 15
  """Steps to accumulate before an optimizer step."""
  learning_rate: float = 1e-3
  """The learning rate."""
  max_grad_norm: float | None = None
  """Gradient clipping, or None for none."""
  loss_type: Literal["mse", "huber"] = "mse"
  """Regression loss against the teacher's action."""
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """The optimizer to use."""
  class_name: str = "Distillation"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """Directory name used to group runs under ``{log_root}/{experiment_name}/``.
  The log root defaults to ``logs/rsl_rl`` and can be overridden with
  ``--log-root`` on the CLI."""
  run_name: str = ""
  """Optional label appended to the timestamped run directory
  (e.g. ``2025-01-27_14-30-00_{run_name}``). Also becomes the
  display name for the run in wandb."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  wandb_tags: Tuple[str, ...] = ()
  """Tags for the wandb run. Default is empty tuple."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""
  upload_model: bool = True
  """Whether to upload model files (.pt, .onnx) to W&B on save. Set to
  False to keep metric logging but avoid storage usage. Default is True."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  """The actor configuration."""
  critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  """The critic configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """The algorithm configuration."""


@dataclass
class RslRlTeacherStudentRunnerCfg(RslRlBaseRunnerCfg):
  """One run, two phases: learn something with privileged observations, then distil it.

  For a skill whose observation is only learnable with information it will not have at
  inference. A motion tracker is the case this was written for: the reference makes the task
  learnable and is not available once the policy has to act on a goal alone.

  Phase one is ordinary PPO, with the actor reading `teacher_obs_group` instead of "actor".
  Phase two freezes that actor as a teacher and regresses a student, reading "actor", onto
  it over the student's own rollouts. The split is `tracking_iterations` of the run's
  `max_iterations`; the rest is distillation.

  The environment exposes both observations at once, so nothing is recorded and replayed and
  the teacher never leaves the process. The student's group is "actor" because the student is
  what gets deployed: `play`, and anything else that loads a policy for inference, asks for
  an actor and gets the student without knowing any of this happened.
  """

  class_name: str = "MjlabTeacherStudentRunner"
  """The runner class name."""

  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.1,
        "std_type": "scalar",
      }
    )
  )
  """The student, and the policy that gets deployed. Called actor because that is what it
  is by the end, and what every inference path asks for.

  A small init_std, unlike a policy trained by a gradient. Phase two is regression, and the
  exploration a policy gradient needs is here just noise on the states being labelled."""

  critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  """Phase one's critic. Unused in phase two, which has no value function."""

  teacher: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  """Phase one's actor, and phase two's frozen teacher. One config for both, because they
  are the same network: phase two takes the weights directly rather than through a file."""

  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """Phase one."""

  distillation: RslRlDistillationAlgorithmCfg = field(
    default_factory=RslRlDistillationAlgorithmCfg
  )
  """Phase two."""

  teacher_obs_group: str = "teacher"
  """Which environment observation group the teacher reads. The student reads "actor"."""

  tracking_iterations: int = 10_000
  """How many of `max_iterations` go to phase one. The remainder go to phase two.

  Phase two is much cheaper than phase one: it is supervised, its target is a function the
  teacher already computes, and it converges in a fraction of the iterations a policy
  gradient needs. Splitting one budget rather than configuring two keeps the total honest."""
