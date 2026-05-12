import jax.numpy as jnp
from mujoco import mjx

from mjlab.envs.mdp.abstractions import Abstraction, AbstractionTarget
from mjlab.tasks.velocity_abstraction.mdp.abstractions.utils import quat_to_yaw, yaw_to_rot2d, gaussian_reward


class BasePoseAbstraction(Abstraction):
    """
    Encodes where the base *should* be given the velocity command and target height.

    The "target" is not a single future position but a desired kinematic state:
    the velocity the base should carry and the height it should maintain.
    This is derived purely from geometry — no reward engineering required.

    command layout: [vx, vy, yaw_rate] expressed in the base frame.
    """

    def __init__(
        self,
        base_body_id: int,
        target_height: float,         # meters above ground
        vel_sigma: float = 0.5,       # acceptable velocity error (m/s)
        height_sigma: float = 0.05,   # acceptable height error (m)
        yaw_rate_sigma: float = 0.5,  # acceptable yaw rate error (rad/s)
        weight: float = 1.0,
    ):
        super().__init__(weight=weight)
        self.base_body_id = base_body_id
        self.target_height = target_height
        self.vel_sigma = vel_sigma
        self.height_sigma = height_sigma
        self.yaw_rate_sigma = yaw_rate_sigma

    def compute_target(self, data: mjx.Data, command: jnp.ndarray) -> AbstractionTarget:
        vx_cmd, vy_cmd, yaw_rate_cmd = command[0], command[1], command[2]

        # Rotate commanded velocity from base frame to world frame using current yaw
        yaw = quat_to_yaw(data.xquat[self.base_body_id])
        rot = yaw_to_rot2d(yaw)
        target_vel_world = rot @ jnp.array([vx_cmd, vy_cmd])  # (2,)

        return AbstractionTarget(
            name="base_pose",
            values={
                "target_linear_velocity": target_vel_world,       # world frame [vx, vy]
                "target_yaw_rate": yaw_rate_cmd,                  # rad/s
                "target_height": jnp.array(self.target_height),   # m
            },
        )

    def compute_reward(self, data: mjx.Data, target: AbstractionTarget) -> jnp.ndarray:
        # data.cvel[body] = [wx, wy, wz, vx, vy, vz] in world frame
        cvel = data.cvel[self.base_body_id]
        actual_lin_vel = cvel[3:5]   # vx, vy
        actual_yaw_rate = cvel[2]    # wz
        actual_height = data.xpos[self.base_body_id][2]

        # Squared error for each component of the target state
        vel_error = jnp.sum((actual_lin_vel - target.values["target_linear_velocity"]) ** 2)
        yaw_error = (actual_yaw_rate - target.values["target_yaw_rate"]) ** 2
        height_error = (actual_height - target.values["target_height"]) ** 2

        # Each component contributes equally; all in [0, 1]
        r_vel = gaussian_reward(vel_error, sigma=self.vel_sigma)
        r_yaw = gaussian_reward(yaw_error, sigma=self.yaw_rate_sigma)
        r_height = gaussian_reward(height_error, sigma=self.height_sigma)

        return (r_vel + r_yaw + r_height) / 3.0