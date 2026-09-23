from rsl_rl.env import VecEnv

from mjlab.rl.runner import MjlabOnPolicyRunner


class OracleRunner(MjlabOnPolicyRunner):
  """PPO runner that accepts the tracking task's optional registry argument."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    del registry_name
    super().__init__(env, train_cfg, log_dir, device)
