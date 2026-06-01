import torch


def randomize_target_height(
  env,
  env_ids: torch.Tensor,
  height_range: tuple[float, float],
) -> None:
  """Sample a fixed target body height for each env at episode reset.

  The sampled values are stored on `env.target_heights` (shape: num_envs,)
  and read back by the `track_target_height` reward.
  """
  low, high = height_range
  if not hasattr(env, "target_heights"):
    env.target_heights = torch.zeros(env.num_envs, device=env.device)
  env.target_heights[env_ids] = (
    torch.rand(len(env_ids), device=env.device) * (high - low) + low
  )
