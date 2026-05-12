import jax.numpy as jnp
from mujoco import mjx

from mjlab.envs.mdp.abstractions import Abstraction


class AbstractionManager:
    # Owns a list of abstractions and aggregates their rewards each step.
    # This is the only entry point the environment needs to call.

    def __init__(self, abstractions: list[Abstraction]):
        self.abstractions = abstractions

    def step(
        self, data: mjx.Data, command: jnp.ndarray
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        # Run every abstraction and sum their weighted rewards.
        # Also returns a per-abstraction breakdown for logging.
        total_reward = jnp.zeros(())
        breakdown = {}

        for abstraction in self.abstractions:
            target, reward = abstraction(data, command)
            total_reward = total_reward + reward
            breakdown[target.name] = reward

        return total_reward, breakdown