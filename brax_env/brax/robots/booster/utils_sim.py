from ml_collections import config_dict
from jax import numpy as jp
import jax
from .utils import BoosterUtils as _BoosterUtils


class BoosterUtils(_BoosterUtils):
    """Simulation-specific Booster utils."""

    TRACKING_SIGMA = 0.1


    @staticmethod
    def reward_config() -> config_dict.ConfigDict:
        return config_dict.create(
            scales=config_dict.create(
                survival=0.25,
                tracking_lin_vel=0.0,  # 1.2
                tracking_lin_vel_x=1.0,
                tracking_lin_vel_y=1.0,
                tracking_ang_vel=2.0,  # original 0.5
                base_height=0.2,
                orientation=-5.0,
                torques=0.0,  # -2.0e-4 / 2
                torque_tiredness=0.0,  # -1.0e-2 / 2
                power=0.0,  # -2.0e-3 / 2
                lin_vel_z=0.0,  # -2.0
                ang_vel_xy=0.0,  # -0.2
                dof_vel=0.0,  # -1.0e-4
                dof_acc=0.0,  # -1.0e-7
                root_acc=0.0,  # -1.0e-4
                action_rate=0.0,  # -1.0 / 2
                dof_pos_limits=0.0,  # -1.0
                collision=0.0,  # -1.0 * 10.0
                feet_slip=-0.1,
                feet_vel_z=0.0,  # -1.0 (disabled in Isaac config)
                feet_yaw_diff=-1.0,
                feet_yaw_mean=-1.0,
                feet_roll=-0.1 * 10.0,  # original -0.1
                feet_distance=-1.0 * 10.0,  # original -1.0
                feet_swing=3.0,
            ),
            tracking_sigma=BoosterUtils.TRACKING_SIGMA,
            base_height_target=0.65,
            swing_period=0.2,
            foot_collision_radius=0.1115,  # radius of the foot geom
        )

    def wm_noise_to_actor_noise(wm_noise: jp.ndarray, actor_state: jp.ndarray, key: jp.ndarray) -> jp.ndarray:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        actor_noise = jp.zeros_like(actor_state)
        actor_noise = actor_noise.at[BoosterUtils.actor_gravity_idxs].set(wm_noise[BoosterUtils.wm_gravity_idxs]    * jax.random.normal(k1, (3)))
        actor_noise = actor_noise.at[BoosterUtils.actor_base_ang_vel_idxs].set(wm_noise[BoosterUtils.wm_base_ang_vel_idxs]    * jax.random.normal(k2, (3)))
        actor_noise = actor_noise.at[BoosterUtils.actor_q_idxs].set(wm_noise[BoosterUtils.wm_q_idxs]  * jax.random.normal(k3, (BoosterUtils.action_size)))
        actor_noise = actor_noise.at[BoosterUtils.actor_qd_idxs].set(wm_noise[BoosterUtils.wm_qd_idxs]  * jax.random.normal(k4, (BoosterUtils.action_size)))
        return actor_noise


__all__ = ["BoosterUtils"]
