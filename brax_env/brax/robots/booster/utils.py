from brax.base import System
from etils import epath
from brax.io import mjcf
from jax import numpy as jp
import jax
import dill
from pathlib import Path


class BoosterUtils:
    """Utility functions for the Go1."""

    """
    Properties
    """

    action_size = 12

    fallen_pitch = 0.785
    fallen_roll = 0.785

    KP = jp.array([50.0, 50.0, 50.0, 50.0, 30.0, 30.0, 
                  50.0, 50.0, 50.0, 50.0, 30.0, 30.0,])

    KD = jp.array([3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 
                  3.0, 3.0, 3.0, 3.0, 1.0, 1.0,])

    STANDING_JOINT_ANGLES_L = jp.array([-0.2, 0.0, 0.0 , 0.4, -0.25, 0.0])
    STANDING_JOINT_ANGLES_F = jp.array([-0.2, 0.0, 0.0 , 0.4, -0.25, 0.0])

    ALL_STANDING_JOINT_ANGLES = jp.concatenate([
        STANDING_JOINT_ANGLES_L,
        STANDING_JOINT_ANGLES_F,
    ])


    LOWER_JOINT_LIMITS = jp.array([-1.8, -0.3, -1.0, 0.0, -0.87, -0.44, 
                                   -1.8, -1.57, -1.0, 0.0, -0.87, -0.44])

    UPPER_JOINT_LIMITS = jp.array([1.57, 1.57, 1.0, 2.34, 0.35, 0.44, 
                                   1.57, 0.3, 1.0, 2.34, 0.35, 0.44])

    MOTOR_TORQUE_LIMIT = jp.tile(jp.array([45.0, 45.0, 30.0, 65.0, 24.0, 15.0]), 2)

    MOTOR_VEL_LIMIT = jp.array([
                                12.5, 10.9, 10.9, 11.7, 18.8, 12.4,
                                12.5, 10.9, 10.9, 11.7, 18.8, 12.4
                                ])
    soft_joint_pos_limit_factor = 0.95
    c = (LOWER_JOINT_LIMITS + UPPER_JOINT_LIMITS) / 2
    r = UPPER_JOINT_LIMITS - LOWER_JOINT_LIMITS
    soft_lowers = c - 0.5 * r * soft_joint_pos_limit_factor
    soft_uppers = c + 0.5 * r * soft_joint_pos_limit_factor

    policy_output_scale = 1.0

    fallen_roll = 0.785
    fallen_pitch = 0.785
    action_size = 12
    hip_indices = jp.array([1, 2, 7, 8])
    feet_indices = jp.array([6, 12])
    knee_indices = jp.array([4, 10])

    terminate_height_min = 0.3
    terminate_height_max = 0.8
    terminate_lin_vel_max = 10.0
    terminate_ang_vel_max = 10.0

    ALL_VEL_LIMIT = jp.array([
                            2.0, 2.0, 2.0,
                            1.0, 1.0, 1.0,
                            12.5, 10.9, 10.9, 11.7, 18.8, 12.4,
                            12.5, 10.9, 10.9, 11.7, 18.8, 12.4
                            ])

    UPPER_ALL_POS_LIMIT = jp.array([
                                100.0, 100.0, 0.8,
                                1.0, 1.0, 1.0, 1.0,
                                   1.57, 1.57, 1.0, 2.34, 0.35, 0.44, 
                                   1.57, 0.3, 1.0, 2.34, 0.35, 0.44
                                ])
    LOWER_ALL_POS_LIMIT = jp.array([
                                -100.0, -100.0, 0.0,
                                -1.0, -1.0, -1.0, -1.0,
                                -1.8, -0.3, -1.0, 0.0, -0.87, -0.44, 
                                -1.8, -1.57, -1.0, 0.0, -0.87, -0.44
                                ])
    """constant: the velocity limit for the motors"""

    xml_quat_idxs = jp.s_[3:7]
    xml_q_idxs = jp.s_[7:19]
    xml_base_vel_idxs = jp.s_[0:3]
    xml_rpy_rate_idxs = jp.s_[3:6]
    xml_qd_idxs = jp.s_[6:18]
    xml_h_idxs = jp.s_[2]

    actor_gravity_idxs = jp.s_[0:3]
    actor_base_ang_vel_idxs = jp.s_[3:6]
    actor_q_idxs = jp.s_[6:18]
    actor_qd_idxs = jp.s_[18:30]
    actor_commands_idxs = jp.s_[30:33]
    actor_cos_phase_idx = jp.s_[33]
    actor_sin_phase_idx = jp.s_[34]
    actor_last_action_idxs = jp.s_[35:47]

    priv_gravity_idxs = jp.s_[0:3]
    priv_base_ang_vel_idxs = jp.s_[3:6]
    priv_q_idxs = jp.s_[6:18]
    priv_qd_idxs = jp.s_[18:30]
    priv_commands_idxs = jp.s_[30:33]
    priv_cos_phase_idx = jp.s_[33]
    priv_sin_phase_idx = jp.s_[34]
    priv_last_action_idxs = jp.s_[35:47]
    priv_quat_idxs = jp.s_[47:51]
    priv_base_lin_vel_idxs = jp.s_[51:54]
    priv_base_lin_vel_x_idx = jp.s_[51]
    priv_base_lin_vel_y_idx = jp.s_[52]
    priv_base_lin_vel_z_idx = jp.s_[53]
    priv_h_idx = jp.s_[54]

    wm_gravity_idxs = jp.s_[0:3]
    wm_base_ang_vel_idxs = jp.s_[3:6]
    wm_q_idxs = jp.s_[6:18]
    wm_qd_idxs = jp.s_[18:30]
    wm_commands_idxs = jp.s_[30:33]
    wm_cos_phase_idx = jp.s_[33]
    wm_sin_phase_idx = jp.s_[34]
    wm_last_action_idxs = jp.s_[35:47]
    wm_quat_idxs = jp.s_[47:51]
    wm_base_lin_vel_idxs = jp.s_[51:54]
    wm_base_lin_vel_x_idx = jp.s_[51]
    wm_base_lin_vel_y_idx = jp.s_[52]
    wm_base_lin_vel_z_idx = jp.s_[53]
    wm_h_idx = jp.s_[54]
    wm_gait_process_idx = jp.s_[55]
    wm_gait_frequency_idx = jp.s_[56]

    observation_size = 47
    priv_observation_size = 55
    wm_observation_size = 57

    actor_state_limits = jp.ones((observation_size, 2))
    actor_state_limits = actor_state_limits.at[:, 0].set(-1.0)
    actor_state_limits = actor_state_limits.at[actor_gravity_idxs, 0].set(-1.0)
    actor_state_limits = actor_state_limits.at[actor_gravity_idxs, 1].set(1.0)
    actor_state_limits = actor_state_limits.at[actor_base_ang_vel_idxs, 0].set(-1.5)
    actor_state_limits = actor_state_limits.at[actor_base_ang_vel_idxs, 1].set(1.5)
    actor_state_limits = actor_state_limits.at[actor_q_idxs, 0].set(LOWER_JOINT_LIMITS - 0.25)
    actor_state_limits = actor_state_limits.at[actor_q_idxs, 1].set(UPPER_JOINT_LIMITS + 0.25)
    actor_state_limits = actor_state_limits.at[actor_qd_idxs, 0].set(-20.0)
    actor_state_limits = actor_state_limits.at[actor_qd_idxs, 1].set(20.0)
    actor_state_limits = actor_state_limits.at[actor_commands_idxs, 0].set(-1.0)
    actor_state_limits = actor_state_limits.at[actor_commands_idxs, 1].set(1.0)
    actor_state_limits = actor_state_limits.at[actor_cos_phase_idx, 0].set(-1.0)
    actor_state_limits = actor_state_limits.at[actor_cos_phase_idx, 1].set(1.0)
    actor_state_limits = actor_state_limits.at[actor_sin_phase_idx, 0].set(-1.0)
    actor_state_limits = actor_state_limits.at[actor_sin_phase_idx, 1].set(1.0)
    actor_state_limits = actor_state_limits.at[actor_last_action_idxs, 0].set(-1.0)
    actor_state_limits = actor_state_limits.at[actor_last_action_idxs, 1].set(1.0)

    wm_state_limits = jp.ones((wm_observation_size, 2))
    wm_state_limits = wm_state_limits.at[:, 0].set(-1.0)
    wm_state_limits = wm_state_limits.at[wm_gravity_idxs, 0].set(-1.0)
    wm_state_limits = wm_state_limits.at[wm_gravity_idxs, 1].set(1.0)
    wm_state_limits = wm_state_limits.at[wm_base_ang_vel_idxs, 0].set(-1.5)
    wm_state_limits = wm_state_limits.at[wm_base_ang_vel_idxs, 1].set(1.5)
    wm_state_limits = wm_state_limits.at[wm_q_idxs, 0].set(LOWER_JOINT_LIMITS - 0.25)
    wm_state_limits = wm_state_limits.at[wm_q_idxs, 1].set(UPPER_JOINT_LIMITS + 0.25)
    wm_state_limits = wm_state_limits.at[wm_qd_idxs, 0].set(-20.0)
    wm_state_limits = wm_state_limits.at[wm_qd_idxs, 1].set(20.0)
    wm_state_limits = wm_state_limits.at[wm_commands_idxs, 0].set(-1.0)
    wm_state_limits = wm_state_limits.at[wm_commands_idxs, 1].set(1.0)
    wm_state_limits = wm_state_limits.at[wm_cos_phase_idx, 0].set(-1.0)
    wm_state_limits = wm_state_limits.at[wm_cos_phase_idx, 1].set(1.0)
    wm_state_limits = wm_state_limits.at[wm_sin_phase_idx, 0].set(-1.0)
    wm_state_limits = wm_state_limits.at[wm_sin_phase_idx, 1].set(1.0)
    wm_state_limits = wm_state_limits.at[wm_last_action_idxs, 0].set(-1.0)
    wm_state_limits = wm_state_limits.at[wm_last_action_idxs, 1].set(1.0)
    wm_state_limits = wm_state_limits.at[wm_quat_idxs, 0].set(-1.0)
    wm_state_limits = wm_state_limits.at[wm_quat_idxs, 1].set(1.0)
    wm_state_limits = wm_state_limits.at[wm_base_lin_vel_x_idx, 0].set(-0.2)
    wm_state_limits = wm_state_limits.at[wm_base_lin_vel_x_idx, 1].set(2.5)
    wm_state_limits = wm_state_limits.at[wm_base_lin_vel_y_idx, 0].set(-0.5)
    wm_state_limits = wm_state_limits.at[wm_base_lin_vel_y_idx, 1].set(0.5)
    wm_state_limits = wm_state_limits.at[wm_base_lin_vel_z_idx, 0].set(-0.5)
    wm_state_limits = wm_state_limits.at[wm_base_lin_vel_z_idx, 1].set(0.5)
    wm_state_limits = wm_state_limits.at[wm_h_idx, 0].set(0.0)
    wm_state_limits = wm_state_limits.at[wm_h_idx, 1].set(0.8)
    wm_state_limits = wm_state_limits.at[wm_gait_process_idx, 0].set(0.0)
    wm_state_limits = wm_state_limits.at[wm_gait_process_idx, 1].set(1.0)
    wm_state_limits = wm_state_limits.at[wm_gait_frequency_idx, 0].set(1.0)
    wm_state_limits = wm_state_limits.at[wm_gait_frequency_idx, 1].set(2.0)

    priv_state_limits = jp.ones((priv_observation_size, 2))
    priv_state_limits = priv_state_limits.at[:, 0].set(-1.0)
    priv_state_limits = priv_state_limits.at[priv_gravity_idxs, 0].set(-1.0)
    priv_state_limits = priv_state_limits.at[priv_gravity_idxs, 1].set(1.0)
    priv_state_limits = priv_state_limits.at[priv_base_ang_vel_idxs, 0].set(-1.5)
    priv_state_limits = priv_state_limits.at[priv_base_ang_vel_idxs, 1].set(1.5)
    priv_state_limits = priv_state_limits.at[priv_q_idxs, 0].set(LOWER_JOINT_LIMITS - 0.25)
    priv_state_limits = priv_state_limits.at[priv_q_idxs, 1].set(UPPER_JOINT_LIMITS + 0.25)
    priv_state_limits = priv_state_limits.at[priv_qd_idxs, 0].set(-20.0)
    priv_state_limits = priv_state_limits.at[priv_qd_idxs, 1].set(20.0)
    priv_state_limits = priv_state_limits.at[priv_commands_idxs, 0].set(-1.0)
    priv_state_limits = priv_state_limits.at[priv_commands_idxs, 1].set(1.0)
    priv_state_limits = priv_state_limits.at[priv_cos_phase_idx, 0].set(-1.0)
    priv_state_limits = priv_state_limits.at[priv_cos_phase_idx, 1].set(1.0)
    priv_state_limits = priv_state_limits.at[priv_sin_phase_idx, 0].set(-1.0)
    priv_state_limits = priv_state_limits.at[priv_sin_phase_idx, 1].set(1.0)
    priv_state_limits = priv_state_limits.at[priv_last_action_idxs, 0].set(-1.0)
    priv_state_limits = priv_state_limits.at[priv_last_action_idxs, 1].set(1.0)
    priv_state_limits = priv_state_limits.at[priv_quat_idxs, 0].set(-1.0)
    priv_state_limits = priv_state_limits.at[priv_quat_idxs, 1].set(1.0)
    priv_state_limits = priv_state_limits.at[priv_base_lin_vel_x_idx, 0].set(-0.2)
    priv_state_limits = priv_state_limits.at[priv_base_lin_vel_x_idx, 1].set(2.5)
    priv_state_limits = priv_state_limits.at[priv_base_lin_vel_y_idx, 0].set(-0.5)
    priv_state_limits = priv_state_limits.at[priv_base_lin_vel_y_idx, 1].set(0.5)
    priv_state_limits = priv_state_limits.at[priv_base_lin_vel_z_idx, 0].set(-0.5)
    priv_state_limits = priv_state_limits.at[priv_h_idx, 0].set(0.0)
    priv_state_limits = priv_state_limits.at[priv_h_idx, 1].set(0.8)


    # TODO
    CACHE_PATH = None#epath.resource_path('brax') / 'robots/go1/.cache'

    @staticmethod
    def get_system(used_cached: bool = False) -> System:
        """Returns the system for the Go1."""

        if used_cached:
            sys = BoosterUtils._load_cached_system(approx_system=False)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/booster/T1_locomotion.xml'

            sys = mjcf.load(path)

        return sys


    @staticmethod
    def get_approx_system(used_cached: bool = False) -> System:
        """Returns the approximate system for the Go1."""

        if used_cached:
            sys = BoosterUtils._load_cached_system(approx_system=True)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/booster/T1_locomotion.xml'
            sys = mjcf.load(path)

        return sys

    @staticmethod
    def _cache_system(approx_system: bool) -> System:
        """Cache the system for the Go1 to avoid reloading the xml file."""
        sys = BoosterUtils.get_system()
        Path(BoosterUtils.CACHE_PATH).mkdir(parents=True, exist_ok=True)
        with open(BoosterUtils._cache_path(approx_system), 'wb') as f:
            dill.dump(sys, f)
        return sys

    @staticmethod
    def _load_cached_system(approx_system: bool) -> System:
        """Load the cached system for the Go1."""
        try:
            with open(BoosterUtils._cache_path(approx_system), 'rb') as f:
                sys = dill.load(f)
        except FileNotFoundError:
            sys = BoosterUtils._cache_system(approx_system)
        return sys

    @staticmethod
    def _cache_path(approx_system: bool) -> epath.Path:
        """Get the path to the cached system for the Go1."""
        if approx_system:
            path = BoosterUtils.CACHE_PATH / 'T1_locomotion.pkl'
        else:
            path = BoosterUtils.CACHE_PATH / 'T1_locomotion.pkl'
        return path

    @staticmethod
    def normalize_state(state: jp.ndarray, state_limits: jp.ndarray) -> jp.ndarray:
        return (2 * (state - state_limits[:, 0])
                / (state_limits[:, 1] - state_limits[:, 0])
                - 1)

    @staticmethod
    def denormalize_state(state: jp.ndarray, state_limits: jp.ndarray) -> jp.ndarray:
        return ((state + 1) * (state_limits[:, 1] - state_limits[:, 0]) / 2
                + state_limits[:, 0])



