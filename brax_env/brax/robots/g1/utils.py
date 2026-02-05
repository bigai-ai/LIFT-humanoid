from brax.base import System
from etils import epath
from brax.io import mjcf
from jax import numpy as jp
import jax
import dill
from pathlib import Path
from ml_collections import config_dict


class g1Utils:
    """Utility functions for the Go1."""

    """
    Properties
    """



    # ===== Values read directly from the provided G1-12DoF MJCF =====

    # Position gains (kp): default g1 joints kp=75, ankle_pitch kp=20, ankle_roll kp=2
    # Order per leg: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
    KP = jp.array([
        40.17923847137318, 99.09842777666113, 40.17923847137318, 99.09842777666113, 28.50124619574858,  28.50124619574858,   # left leg
        40.17923847137318, 99.09842777666113, 40.17923847137318, 99.09842777666113, 28.50124619574858,  28.50124619574858,   # right leg
    ])

    # Damping (kd): default g1 joints damping=2, ankle_pitch damping=1, ankle_roll damping=0.2
    KD = jp.array([
        2.5578897650279457,  6.3088018534966395,  2.5578897650279457,  6.3088018534966395, 1.814445686584846, 1.814445686584846,    # left leg
        2.5578897650279457,  6.3088018534966395,  2.5578897650279457,  6.3088018534966395, 1.814445686584846, 1.814445686584846,    # right leg
    ])

    # Standing pose from custom numeric "init_qpos" (after the 7-DoF free base):
    # repeated 6D per leg: [-0.1, 0, 0, 0.3, -0.2, 0]
    STANDING_JOINT_ANGLES_L = jp.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0])
    STANDING_JOINT_ANGLES_F = jp.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0])

    ALL_STANDING_JOINT_ANGLES = jp.concatenate([
        STANDING_JOINT_ANGLES_L,
        STANDING_JOINT_ANGLES_F,
    ])

    # Joint angle limits from each joint's <range="lower upper">
    LOWER_JOINT_LIMITS = jp.array([
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,   # left leg
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,   # right leg
    ])

    UPPER_JOINT_LIMITS = jp.array([
        2.8798,  2.9671,  2.7576,  2.8798,    0.5236,   0.2618,   # left leg
        2.8798,  2.9671,  2.7576,  2.8798,    0.5236,   0.2618,   # right leg
    ])

    # Motor torque limits from actuatorfrcrange (absolute values):
    # hip_pitch/hip_yaw: ±88, hip_roll/knee: ±139, ankle_pitch/ankle_roll: ±50
    MOTOR_TORQUE_LIMIT = jp.array([88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * 2)

    # ===== Not present in this MJCF (set here only if you still need placeholders) =====
    # No velocity limits are defined in the XML; enforce in controller if required.
    MOTOR_VEL_LIMIT = jp.array([
                                32, 20, 32, 20, 37, 37,
                                32, 20, 32, 20, 37, 37
                                ])
    ALL_VEL_LIMIT = None

    # Whole-state limits (including free base) are not defined in the XML.
    UPPER_ALL_POS_LIMIT = None
    LOWER_ALL_POS_LIMIT = None

    # set up slices for the state space, defined in the xml file
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


    # set up observation normalization limits
    # get the size of the state space using the slice above
    observation_size = 47
    priv_observation_size = 55
    wm_observation_size = 57

    actor_state_limits = jp.ones((observation_size, 2))
    actor_state_limits = actor_state_limits.at[:, 0].set(-1.)
    actor_state_limits = actor_state_limits.at[actor_gravity_idxs, 0].set(-1.)
    actor_state_limits = actor_state_limits.at[actor_gravity_idxs, 1].set(1.)
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
    wm_state_limits = wm_state_limits.at[:, 0].set(-1.)
    wm_state_limits = wm_state_limits.at[wm_gravity_idxs, 0].set(-1.)
    wm_state_limits = wm_state_limits.at[wm_gravity_idxs, 1].set(1.)
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
    priv_state_limits = priv_state_limits.at[:, 0].set(-1.)
    priv_state_limits = priv_state_limits.at[priv_gravity_idxs, 0].set(-1.)
    priv_state_limits = priv_state_limits.at[priv_gravity_idxs, 1].set(1.)
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


    action_scale = jp.array([0.54754645, 0.35066146, 0.54754645, 0.35066146, 0.43857732, 0.43857732, 
            0.54754645, 0.35066146, 0.54754645, 0.35066146, 0.43857732, 0.43857732])

    #action_max = jp.abs(UPPER_JOINT_LIMITS - ALL_STANDING_JOINT_ANGLES)
    #action_min = jp.abs(LOWER_JOINT_LIMITS - ALL_STANDING_JOINT_ANGLES)
    #max_range = jp.maximum(action_min, action_max)
    #policy_output_scale = max_range / action_scale
    policy_output_scale = 1.0

    terminate_height_min = 0.3
    terminate_height_max = 1.0
    terminate_lin_vel_max = 10.0
    terminate_ang_vel_max = 10.0
    soft_joint_pos_limit_factor = 0.95
    c = (LOWER_JOINT_LIMITS + UPPER_JOINT_LIMITS) / 2
    r = UPPER_JOINT_LIMITS - LOWER_JOINT_LIMITS
    soft_lowers = c - 0.5 * r * soft_joint_pos_limit_factor
    soft_uppers = c + 0.5 * r * soft_joint_pos_limit_factor

    fallen_roll = 0.785
    fallen_pitch = 0.785
    action_size = 12
    hip_indices = jp.array([1, 2, 7, 8])
    feet_indices = jp.array([6, 12])
    knee_indices = jp.array([4, 10])

    reward_config = config_dict.create(
          scales=config_dict.create(
              survival=0.25,
              tracking_lin_vel=0.0,#1.2,
              tracking_lin_vel_x=2.0,
              tracking_lin_vel_y=1.0,
              tracking_ang_vel=2.0,  # original 0.5
              base_height=0.2,
              orientation=-5.0,
              torques=0.0,#-2.0e-4 / 2,  # original -2.0e-4
              torque_tiredness=0.0,#-1.0e-2 / 2,  # original -1.0e-2
              power=0.0,#-2.0e-3 / 2,  # original -2.0e-3
              lin_vel_z=0.0,#-2.0,
              ang_vel_xy=0.0,#-0.2,
              dof_vel=0.0,#-1.0e-4,
              dof_acc=0.0,#-1.0e-7,
              root_acc=0.0,#-1.0e-4,
              action_rate=0.0,#-1.0 / 2,  # original -1.0
              dof_pos_limits=0.0,#-1.0,
              feet_slip=-0.1 * 10,
              feet_vel_z=0.0,#-1.0,     # disabled in Isaac config
              feet_yaw_diff=-1.0,
              feet_yaw_mean=-1.0,
              feet_roll=-0.1 * 10.0,  # original -0.1
              feet_distance=-1.0 * 10.0,  # original -1.0
              feet_swing=3.0,
              feet_height=-20,
              joint_deviation_hip=-1.0,

          ),
          tracking_sigma = 0.1,
          base_height_target = 0.5,
          swing_period = 0.6,
          max_foot_height=0.08,
          foot_collision_radius = 0.1115, # the radius of the geom of foot

      )
    """constant: the velocity limit for the motors"""


    # TODO
    CACHE_PATH = None#epath.resource_path('brax') / 'robots/go1/.cache'

    @staticmethod
    def get_system(used_cached: bool = False) -> System:
        """Returns the system for the Go1."""

        if used_cached:
            sys = g1Utils._load_cached_system(approx_system=False)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/g1/g1_locomotion.xml'

            sys = mjcf.load(path)

        return sys


    @staticmethod
    def get_approx_system(used_cached: bool = False) -> System:
        """Returns the approximate system for the Go1."""

        if used_cached:
            sys = g1Utils._load_cached_system(approx_system=True)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/g1/g1_locomotion.xml'
            sys = mjcf.load(path)

        return sys

    @staticmethod
    def _cache_system(approx_system: bool) -> System:
        """Cache the system for the Go1 to avoid reloading the xml file."""
        sys = g1Utils.get_system()
        Path(g1Utils.CACHE_PATH).mkdir(parents=True, exist_ok=True)
        with open(g1Utils._cache_path(approx_system), 'wb') as f:
            dill.dump(sys, f)
        return sys

    @staticmethod
    def _load_cached_system(approx_system: bool) -> System:
        """Load the cached system for the Go1."""
        try:
            with open(g1Utils._cache_path(approx_system), 'rb') as f:
                sys = dill.load(f)
        except FileNotFoundError:
            sys = g1Utils._cache_system(approx_system)
        return sys

    @staticmethod
    def _cache_path(approx_system: bool) -> epath.Path:
        """Get the path to the cached system for the Go1."""
        if approx_system:
            path = g1Utils.CACHE_PATH / 'g1_locomotion.pkl'
        else:
            path = g1Utils.CACHE_PATH / 'g1_locomotion.pkl'
        return path

    def normalize_state(state: jp.ndarray, state_limits: jp.ndarray) -> jp.ndarray:
        return (2*(state - state_limits[:, 0])
                / (state_limits[:, 1] - state_limits[:, 0])
                - 1)

    def denormalize_state(state: jp.ndarray, state_limits: jp.ndarray) -> jp.ndarray:
        return ((state + 1)*(state_limits[:, 1] - state_limits[:, 0])/2
                + state_limits[:, 0])

    def wm_noise_to_actor_noise(wm_noise: jp.ndarray, actor_state: jp.ndarray, key: jp.ndarray) -> jp.ndarray:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        actor_noise = jp.zeros_like(actor_state)
        actor_noise = actor_noise.at[g1Utils.actor_gravity_idxs].set(wm_noise[g1Utils.wm_gravity_idxs]    * jax.random.normal(k1, (3)))
        actor_noise = actor_noise.at[g1Utils.actor_base_ang_vel_idxs].set(wm_noise[g1Utils.wm_base_ang_vel_idxs]    * jax.random.normal(k2, (3)))
        actor_noise = actor_noise.at[g1Utils.actor_q_idxs].set(wm_noise[g1Utils.wm_q_idxs]  * jax.random.normal(k3, (g1Utils.action_size)))
        actor_noise = actor_noise.at[g1Utils.actor_qd_idxs].set(wm_noise[g1Utils.wm_qd_idxs]  * jax.random.normal(k4, (g1Utils.action_size)))
        return actor_noise


