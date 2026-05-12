import jax.numpy as jnp
from mujoco import mjx

from mjlab.envs.mdp.abstraction import Abstraction, AbstractionTarget
from mjlab.tasks.velocity_abstraction.mdp.utils import quat_to_yaw, yaw_to_rot2d, gaussian_reward


class FootPlacementAbstraction(Abstraction):
    """
    Encodes where each foot should land based on morphology and commanded velocity.

    Uses the Raibert heuristic: the ideal landing spot for a foot is its
    nominal resting position offset forward by velocity * stance_duration / 2.
    This accounts for the robot needing to "catch up" with its own momentum.

    The nominal offsets encode morphology — they are fixed by the robot's geometry.
    The velocity term encodes the task — it shifts targets based on the command.

    command layout: [vx, vy, yaw_rate], only vx and vy are used here.
    """

    def __init__(
        self,
        base_body_id: int,
        foot_body_ids: list[int],              # one id per foot, e.g. [FR, FL, RR, RL]
        nominal_foot_offsets: jnp.ndarray,     # shape (n_feet, 3), base frame, at rest stance
        stance_duration: float,                # approximate duration of one stance phase (s)
        placement_sigma: float = 0.05,         # acceptable xy placement error (m)
        weight: float = 1.0,
    ):
        super().__init__(weight=weight)
        self.base_body_id = base_body_id
        self.foot_body_ids = foot_body_ids
        self.nominal_offsets = nominal_foot_offsets  # (n_feet, 3)
        self.stance_duration = stance_duration
        self.placement_sigma = placement_sigma

    def compute_target(self, data: mjx.Data, command: jnp.ndarray) -> AbstractionTarget:
        vel_cmd = command[:2]  # [vx, vy] in base frame

        base_pos = data.xpos[self.base_body_id]  # (3,)
        yaw = quat_to_yaw(data.xquat[self.base_body_id])
        rot = yaw_to_rot2d(yaw)

        # Raibert offset: each foot shifts forward by v * T_stance / 2 to stay under COM
        raibert_xy = vel_cmd * (self.stance_duration / 2.0)  # (2,) in base frame

        # Apply the offset to every nominal foot position simultaneously
        nominal_xy = self.nominal_offsets[:, :2]            # (n_feet, 2) in base frame
        adjusted_xy = nominal_xy + raibert_xy[None, :]      # (n_feet, 2), broadcast shift

        # Rotate all adjusted positions from base frame to world frame, then translate
        target_xy_world = (rot @ adjusted_xy.T).T + base_pos[:2]  # (n_feet, 2)

        # Flat terrain assumed; z=0 for all feet (can be swapped for terrain estimation)
        target_z = jnp.zeros((len(self.foot_body_ids), 1))
        target_positions = jnp.concatenate([target_xy_world, target_z], axis=1)  # (n_feet, 3)

        return AbstractionTarget(
            name="foot_placement",
            values={"target_foot_positions": target_positions},
        )

    def compute_reward(self, data: mjx.Data, target: AbstractionTarget) -> jnp.ndarray:
        target_positions = target.values["target_foot_positions"]  # (n_feet, 3)

        # Gather actual foot positions from simulation state
        actual_positions = jnp.stack(
            [data.xpos[fid] for fid in self.foot_body_ids]
        )  # (n_feet, 3)

        # Score XY placement only — z error on flat terrain just reflects leg extension
        xy_errors = jnp.sum(
            (actual_positions[:, :2] - target_positions[:, :2]) ** 2, axis=1
        )  # (n_feet,)

        foot_rewards = gaussian_reward(xy_errors, sigma=self.placement_sigma)  # (n_feet,)
        return jnp.mean(foot_rewards)