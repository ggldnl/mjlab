from abc import ABC, abstractmethod
from typing import NamedTuple
import jax.numpy as jnp
from mujoco import mjx


class AbstractionTarget(NamedTuple):
    # Holds whatever the abstraction decided the reference state should be.
    # 'values' is a plain dict of jnp arrays so it works as a JAX pytree.
    name: str
    values: dict


class Abstraction(ABC):
    # Base class for all abstractions.
    # Subclasses define *what* the target is and *how* to score it.

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def compute_target(self, data: mjx.Data, command: jnp.ndarray) -> AbstractionTarget:
        # Derive the reference state from the current simulation state and command.
        # This should encode domain knowledge (kinematics, morphology, etc.).
        pass

    @abstractmethod
    def compute_reward(self, data: mjx.Data, target: AbstractionTarget) -> jnp.ndarray:
        # Score how well the current state matches the target.
        # Should return a value in [0, 1] before weighting.
        pass

    def __call__(
        self, data: mjx.Data, command: jnp.ndarray
    ) -> tuple[AbstractionTarget, jnp.ndarray]:
        # Compute target and weighted reward in one shot.
        target = self.compute_target(data, command)
        reward = self.weight * self.compute_reward(data, target)
        return target, reward


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