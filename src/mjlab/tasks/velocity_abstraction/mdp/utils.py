import jax.numpy as jnp


def quat_to_yaw(quat: jnp.ndarray) -> jnp.ndarray:
    """Extract yaw (rotation around z) from a unit quaternion [w, x, y, z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))


def yaw_to_rot2d(yaw: jnp.ndarray) -> jnp.ndarray:
    """2x2 rotation matrix that rotates a 2D vector by yaw."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.array([[c, -s], [s, c]])


def gaussian_reward(squared_error: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Maps a squared error to [0, 1]: perfect match -> 1, diverging -> 0.
    sigma controls how much error is tolerated before the reward drops sharply.
    """
    return jnp.exp(-squared_error / (sigma**2))