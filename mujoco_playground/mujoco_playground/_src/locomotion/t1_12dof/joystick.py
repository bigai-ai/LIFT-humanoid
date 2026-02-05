# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Booster T1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.t1_12dof import base as t1_base
from mujoco_playground._src.locomotion.t1_12dof import t1_constants as consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=1.0,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                linvel=0.1,
                gyro=0.2,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                survival=0.25,
                tracking_lin_vel_x=1.0,
                tracking_lin_vel_y=1.0,
                tracking_ang_vel=2.0,  # original 0.5
                base_height=-20.0,
                orientation=-10.0,
                torques=-2.0e-4 / 2,  # original -2.0e-4
                torque_tiredness=-1.0e-2 / 2,  # original -1.0e-2
                power=-2.0e-3 / 2,  # original -2.0e-3
                lin_vel_z=-2.0,
                ang_vel_xy=-0.2,
                dof_vel=-1.0e-4,
                dof_acc=-1.0e-7,
                root_acc=-1.0e-4,
                action_rate=-1.0 / 2,  # original -1.0
                dof_pos_limits=-1.0,
                collision=-1.0 * 10.0,  # original -1.0
                feet_slip=-0.1,
                feet_vel_z=0.0,     # disabled in Isaac config
                feet_yaw_diff=-1.0,
                feet_yaw_mean=-1.0,
                feet_roll=-0.1 * 10.0,  # original -0.1
                feet_distance=-1.0 * 10.0,  # original -1.0
                feet_swing=3.0,
            ),
            tracking_sigma = 0.25,
            base_height_target = 0.68,
            swing_period = 0.2,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        lin_vel_x=[-1.0, 1.0],
        lin_vel_y=[-0.8, 0.8],
        ang_vel_yaw=[-1.0, 1.0],
    )


class Joystick(t1_base.T1LowDimEnv):
    """Track a joystick command."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

        # Note: First joint is freejoint.
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        hip_indices = []
        hip_joint_names = ["Hip_Roll", "Hip_Yaw"]
        for side in ["Left", "Right"]:
            for joint_name in hip_joint_names:
                hip_indices.append(
                    self._mj_model.joint(f"{side}_{joint_name}").qposadr - 7
                )
        self._hip_indices = jp.array(hip_indices)

        knee_indices = []
        for side in ["Left", "Right"]:
            knee_indices.append(
                self._mj_model.joint(f"{side}_Knee_Pitch").qposadr - 7
            )
        self._knee_indices = jp.array(knee_indices)

        # fmt: off
        self._weights = jp.array([
            0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # Left leg.
            0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # Right leg.
        ])
        # fmt: on

        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._left_feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in consts.LEFT_FEET_GEOMS]
        )
        self._right_feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in consts.RIGHT_FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        self._left_foot_box_geom_id = self._mj_model.geom("left_foot").id
        self._right_foot_box_geom_id = self._mj_model.geom("right_foot").id

        force_range = self._mj_model.actuator_forcerange          # (nact, 2)
        force_limited = self._mj_model.actuator_forcelimited      # (nact,)
        hi = jp.array(force_range[:, 1])
        #   unlimited → treat as very large so the penalty goes to zero
        self._torque_limits = jp.where(force_limited, hi, jp.full_like(hi, 1e6))

    def _reset_if_outside_bounds(self, state: mjx_env.State) -> mjx_env.State:
        qpos = state.data.qpos
        new_x = jp.where(jp.abs(qpos[0]) > 9.5, 0.0, qpos[0])
        new_y = jp.where(jp.abs(qpos[1]) > 9.5, 0.0, qpos[1])
        qpos = qpos.at[0:2].set(jp.array([new_x, new_y]))
        state = state.replace(data=state.data.replace(qpos=qpos))
        return state

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # qpos[7:]=*U(0.5, 1.5)
        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(
            qpos[7:] * jax.random.uniform(key, (12,), minval=0.5, maxval=1.5)
        )

        # d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        # Phase, freq=U(1.25, 1.75)
        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (1,), minval=1.0, maxval=2.0)
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        phase = jp.array([0, jp.pi])

        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Phase related.
            "phase_dt": phase_dt,
            "phase": phase,
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            "filtered_linvel": jp.zeros(3),
            "filtered_angvel": jp.zeros(3),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        left_feet_contact = jp.array([
            geoms_colliding(data, geom_id, self._floor_geom_id)
            for geom_id in self._left_feet_geom_id
        ])
        right_feet_contact = jp.array([
            geoms_colliding(data, geom_id, self._floor_geom_id)
            for geom_id in self._right_feet_geom_id
        ])
        contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])

        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info["rng"], push1_rng, push2_rng = jax.random.split(
            state.info["rng"], 3
        )
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
            == 0
        )
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        # state = self._reset_if_outside_bounds(state)

        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )
        state.info["motor_targets"] = motor_targets

        linvel = self.get_local_linvel(data)
        state.info["filtered_linvel"] = (
            linvel * 1.0 + state.info["filtered_linvel"] * 0.0
        )
        angvel = self.get_gyro(data)
        state.info["filtered_angvel"] = (
            angvel * 1.0 + state.info["filtered_angvel"] * 0.0
        )

        left_feet_contact = jp.array([
            geoms_colliding(data, geom_id, self._floor_geom_id)
            for geom_id in self._left_feet_geom_id
        ])
        right_feet_contact = jp.array([
            geoms_colliding(data, geom_id, self._floor_geom_id)
            for geom_id in self._right_feet_geom_id
        ])
        contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
        state.info["phase"] = jp.where(
            jp.linalg.norm(state.info["command"]) > 0.01,
            state.info["phase"],
            jp.ones(2) * jp.pi,
        )
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return (
            fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        )

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        )

        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        phi = info["phase"][0]          # [-π, π)
        phase = jp.array([jp.cos(phi), jp.sin(phi)])
        # if command near zero, set phase to zero
        phase *= jp.linalg.norm(info["command"]) > 0.01

        linvel = self.get_local_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
            linvel
            + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linvel
        )
        # We will disable noisy_linvel
        noisy_linvel = jp.zeros_like(noisy_linvel)

        state = jp.hstack([
            noisy_gravity,  # 3
            noisy_gyro,  # 3
            info["command"],  # 3
            phase, # 2-dimensional 
            noisy_joint_angles - self._default_pose,  # 12
            noisy_joint_vel * 0.1,  # 12
            info["last_act"],  # 12
        ])

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[2]

        privileged_state = jp.hstack([
            state,
            gyro,  # 3
            accelerometer,  # 3
            gravity,  # 3
            linvel,  # 3
            global_angvel,  # 3
            joint_angles - self._default_pose,
            joint_vel,
            root_height,  # 1
            data.actuator_force,
            contact,  # 2
            feet_vel,  # 4*3
            info["feet_air_time"],  # 2
        ])

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:

        cmd = info["command"]
        lin_f = info["filtered_linvel"]
        ang_f = info["filtered_angvel"]

        return {
            # -------- positive terms --------
            "survival": jp.array(1.0),
            "tracking_lin_vel_x": self._reward_tracking_lin_vel_axis(0, cmd, lin_f),
            "tracking_lin_vel_y": self._reward_tracking_lin_vel_axis(1, cmd, lin_f),
            "tracking_ang_vel": self._reward_tracking_ang_vel(cmd, ang_f),
            "feet_swing": self._reward_feet_swing(info["phase"], contact, cmd),

            # -------- penalties ------------ (signed handled by scale)
            "base_height": self._reward_base_height(data),
            "orientation": self._cost_orientation(self.get_gravity(data)),
            "torques": self._cost_torques(data.actuator_force),
            "torque_tiredness": self._cost_torque_tiredness(data.actuator_force),
            "power": self._cost_energy(data.qvel[6:], data.actuator_force),
            "lin_vel_z": self._cost_lin_vel_z(lin_f),
            "ang_vel_xy": self._cost_ang_vel_xy(ang_f),
            "dof_vel": self._cost_dof_vel(data.qvel[6:]),
            "dof_acc": self._cost_dof_acc(data.qacc[6:]),
            "root_acc": self._cost_root_acc(data),
            "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "collision": self._cost_collision(data),
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "feet_vel_z": self._cost_feet_vel_z(data),
            "feet_roll": self._cost_feet_roll(data),
            "feet_yaw_diff": self._cost_feet_yaw_diff(data),
            "feet_yaw_mean": self._cost_feet_yaw_mean(data),
            "feet_distance": self._cost_feet_distance(data, info),
        }


    # Tracking rewards.

    def _reward_tracking_lin_vel_axis(
        self, axis: int, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        """Axis–wise linear‑velocity tracker (matches Isaac Gym x & y trackers)."""
        err = jp.square(command[axis] - local_linvel[axis])
        return jp.exp(-err / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        local_angvel: jax.Array,
    ) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - local_angvel[2])
        return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

    # Base-related rewards.

    def _cost_lin_vel_z(self, local_linvel) -> jax.Array:
        return jp.square(local_linvel[2])

    def _cost_ang_vel_xy(self, local_angvel) -> jax.Array:
        return jp.sum(jp.square(local_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torso_zaxis[:2]))

    def _reward_base_height(self, data: mjx.Data) -> jax.Array:
        h = data.qpos[2]
        return jp.square(h - self._config.reward_config.base_height_target)

    # Energy related rewards.

    def _cost_torques(self, torques: jp.ndarray) -> jp.ndarray:
        return jp.sum(jp.square(torques))

    def _cost_energy(self, qvel: jp.ndarray, qfrc_actuator: jp.ndarray) -> jp.ndarray:
        power = qvel * qfrc_actuator
        return jp.sum(jp.where(power > 0.0, power, 0.0))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        c1 = jp.sum(jp.square(act - last_act))
        return c1

    def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qacc))

    def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qvel))

    # Other rewards.

    def _cost_joint_pos_limits(self, qpos: jp.ndarray) -> jp.ndarray:
        below = qpos < self._soft_lowers
        above = qpos > self._soft_uppers
        return jp.sum((below | above).astype(jp.float32))

    def _reward_survival(self) -> jax.Array:
        return jp.array(1.0)

    def _cost_collision(self, data: mjx.Data) -> jax.Array:
        return geoms_colliding(
            data, self._left_foot_box_geom_id, self._right_foot_box_geom_id
        )

    # Pose-related rewards.

    def _cost_joint_deviation_hip(
        self, qpos: jax.Array, cmd: jax.Array
    ) -> jax.Array:
        cost = jp.sum(
            jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices])
        )
        cost *= jp.abs(cmd[1]) > 0.1
        return cost

    def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
        return jp.sum(
            jp.abs(
                qpos[self._knee_indices] - self._default_pose[self._knee_indices]
            )
        )

    def _cost_pose(self, qpos: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

    # Feet related rewards.

    def _cost_feet_slip(
        self,
        data: mjx.Data,
        contact: jp.ndarray,
        info: dict[str, Any],
    ) -> jp.ndarray:
        del info  # Unused here.
        v = data.sensordata[self._foot_linvel_sensor_adr]  # shape (2, 3)
        speed2 = jp.sum(jp.square(v), axis=-1)             # per‑foot |v|²
        return jp.sum(speed2 * contact)

    def _cost_feet_height(
        self,
        swing_peak: jax.Array,
        first_contact: jax.Array,
        info: dict[str, Any],
    ) -> jax.Array:
        del info  # Unused.
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact)

    def _reward_feet_air_time(
        self,
        air_time: jax.Array,
        first_contact: jax.Array,
        commands: jax.Array,
        threshold_min: float = 0.2,
        threshold_max: float = 0.5,
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        air_time = (air_time - threshold_min) * first_contact
        air_time = jp.clip(air_time, max=threshold_max - threshold_min)
        reward = jp.sum(air_time)
        reward *= cmd_norm > 0.1  # No reward for zero commands.
        return reward

    def _cost_feet_distance(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        del info  # Unused.
        left_foot_pos = data.site_xpos[self._feet_site_id[0]]
        right_foot_pos = data.site_xpos[self._feet_site_id[1]]
        base_xmat = data.site_xmat[self._site_id]
        base_yaw = jp.arctan2(base_xmat[1, 0], base_xmat[0, 0])
        feet_distance = jp.abs(
            jp.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1])
            - jp.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
        )
        return jp.clip(0.2 - feet_distance, min=0.0, max=0.1)

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        lin_vel_x = jax.random.uniform(
            rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        # With 10% chance, set everything to zero.
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(3),
            jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
        )

    def _cost_torque_tiredness(self, torques: jax.Array) -> jax.Array:
        # Σ (τ / τ_max)²  – clipped at 1 so the term stays O(1)
        frac = jp.clip(jp.abs(torques) / self._torque_limits, 0.0, 1.0)
        return jp.sum(jp.square(frac))

    def _cost_root_acc(self, data: mjx.Data) -> jax.Array:
        # Root‑link 6‑D acceleration² (free‑joint entries of qacc)
        return jp.sum(jp.square(data.qacc[:6]))

    # ----- feet kinematics ----------------------------------------------------
    def _feet_site_xmat(self, data: mjx.Data) -> jax.Array:
        """Return the (2,3,3) rotation matrices of the foot *sites*."""
        return data.site_xmat[self._feet_site_id].reshape(2, 3, 3)

    def _feet_roll_yaw(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        """Return (roll, yaw) angles of both feet, radians in [‑π, π]."""
        R = self._feet_site_xmat(data)
        # roll  = atan2(R32, R33)           (x‑rotation)
        # yaw   = atan2(R21, R11)           (z‑rotation)
        roll = jp.arctan2(R[:, 2, 1], R[:, 2, 2])
        yaw  = jp.arctan2(R[:, 1, 0], R[:, 0, 0])
        return roll, yaw

    def _cost_feet_roll(self, data: mjx.Data) -> jax.Array:
        roll, _ = self._feet_roll_yaw(data)
        return jp.sum(jp.square(roll))

    def _cost_feet_yaw_diff(self, data: mjx.Data) -> jax.Array:
        _, yaw = self._feet_roll_yaw(data)
        diff = jp.fmod(yaw[1] - yaw[0] + jp.pi, 2 * jp.pi) - jp.pi
        return jp.square(diff)

    def _cost_feet_yaw_mean(self, data: mjx.Data) -> jax.Array:
        _, feet_yaw = self._feet_roll_yaw(data)
        base_R = data.site_xmat[self._site_id]
        base_yaw = jp.arctan2(base_R[1, 0], base_R[0, 0])
        mean_yaw = jp.mean(feet_yaw)
        err = jp.fmod(base_yaw - mean_yaw + jp.pi, 2 * jp.pi) - jp.pi
        return jp.square(err)

    def _cost_feet_vel_z(self, data: mjx.Data) -> jax.Array:
        # use the same foot linear‑velocity sensors already wired for slip
        vz = data.sensordata[self._foot_linvel_sensor_adr][:, 2]
        return jp.sum(jp.square(vz))

    def _reward_feet_swing(self,
                           phase: jp.ndarray,
                           feet_contact: jp.ndarray,
                            commands: jax.Array,
                           ) -> jp.ndarray:
        gait = jp.fmod(phase[0] + jp.pi, 2 * jp.pi) / (2 * jp.pi)  # scalar ∈[0,1)
        half_window = 0.5 * self._config.reward_config.swing_period

        left_swing  = jp.abs(gait - 0.25) < half_window
        right_swing = jp.abs(gait - 0.75) < half_window

        # Reward when the corresponding foot is **not** in contact
        swing_reward = (left_swing & ~feet_contact[0]) + (right_swing & ~feet_contact[1])
        # if command near zero, no swing reward
        swing_reward *= jp.linalg.norm(commands) > 0.01
        return swing_reward
