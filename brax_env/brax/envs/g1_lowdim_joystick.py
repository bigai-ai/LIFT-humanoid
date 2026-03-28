from brax.robots.g1.utils import g1Utils
from brax.envs.base import RlwamEnv, State

from brax import actuator
from brax import kinematics
from brax.generalized.base import State as GeneralizedState
from brax.generalized import dynamics
from brax.generalized import integrator
from brax.generalized import mass
from brax import base
from brax.math import rotate, inv_rotate, quat_to_eulerzyx, eulerzyx_to_quat, quat_to_euler, euler_to_quat
from brax.generalized.pipeline import step as pipeline_step

from jax import numpy as jp
from typing import Optional, Any, Tuple, Callable, Dict
import jax
import flax
from jax import lax



@flax.struct.dataclass
class ControlCommand:
    """Output of the low level controller which includes gait control and
    inverse kinematics. """
    q_des: jp.ndarray
    qd_des: jp.ndarray
    Kp: jp.ndarray
    Kd: jp.ndarray


class G1LowDimJoystick(RlwamEnv):
    """ G1LowDimJoystick environment"""

    def __init__(
        self,
        policy_repeat=10,
        initial_yaw_range=(-0.0, 0.0),
        contact_time_const=0.02,
        contact_time_const_range=None,
        contact_damping_ratio=1.0,
        friction_range=(0.6, 0.6),
        ground_roll_range=(0.0, 0.0),
        ground_pitch_range=(0.0, 0.0),
        joint_damping_perc_range=(1.0, 1.0),
        joint_gain_range=(1.0, 1.0),
        link_mass_perc_range=(1.0, 1.0),
        forces_in_q_coords=False,
        backend='generalized',
        used_cached_systems=False,
        mini_ankle_dist=0.06,
        vel_x_command=1.5,
        **kwargs
    ):

        self.sim_dt = 1/500  # simulation dt; 1000 Hz

        # determines high level policy freq; (1/sim_dt)/policy_repeat Hz
        self.policy_repeat = 10
        self.policy_repeat_pd = policy_repeat
        sys = g1Utils.get_system(used_cached_systems) # load 'robots/go1/xml/go1.xml'
        sys = sys.replace(dt=self.sim_dt)

        # normally this is use by Brax as the number of times to step the
        # physics pipeline for each environment step. However we have
        # overwritten the pipline_step function with our own behaviour which
        # steps the physics self.policy_repeat times. So we set this to 1.
        n_frames = 1
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._initial_yaw_range = initial_yaw_range
        if contact_time_const_range is None:
            self._contact_time_const_range = (contact_time_const,
                                              contact_time_const)
        else:
            self._contact_time_const_range = contact_time_const_range
        self._contact_damping_ratio = contact_damping_ratio
        self._friction_range = friction_range
        self._ground_roll_range = ground_roll_range
        self._ground_pitch_range = ground_pitch_range
        self._joint_damping_perc_range = joint_damping_perc_range
        self._joint_gain_range = joint_gain_range
        self._link_mass_perc_range = link_mass_perc_range


        if forces_in_q_coords:
            self._qfc_fn = lambda state, forces: forces
        else:
            self._qfc_fn = lambda state, forces: state.con_jac.T @ forces

        self.reward_config = g1Utils.reward_config

        # set up slices for the action space, defined by the table above
        self.mini_ankle_dist = mini_ankle_dist
        self.commands = jp.array([vel_x_command, 0.0, 0.0, 0.0])

    def reset(self, rng: jp.ndarray) -> State:

        # randomize initial yaw
        rng, rng_yaw = jax.random.split(rng)
        initial_yaw = jax.random.uniform(
            rng_yaw, shape=(),
            minval=self._initial_yaw_range[0]*180/jp.pi,
            maxval=self._initial_yaw_range[1]*180/jp.pi
        )
        initial_quat = eulerzyx_to_quat(jp.array([0.0, 0.0, initial_yaw]))

        # initialize system with initial q and qd
        q = self.sys.init_q  # init_q is defined in the xml file
        q = q.at[g1Utils.xml_quat_idxs].set(initial_quat)
        qd = jp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(q, qd)

        # domain randomization
        domain_rand_rngs = jax.random.split(rng, 7)
        self._contact_time_const = jax.random.uniform(
            domain_rand_rngs[0], shape=(),
            minval=self._contact_time_const_range[0],
            maxval=self._contact_time_const_range[1]
        )
        self._friction = jax.random.uniform(
            domain_rand_rngs[1], shape=(),
            minval=self._friction_range[0],
            maxval=self._friction_range[1]
        )
        self._ground_roll = jax.random.uniform(
            domain_rand_rngs[2], shape=(),
            minval=self._ground_roll_range[0],
            maxval=self._ground_roll_range[1]
        )
        self._ground_pitch = jax.random.uniform(
            domain_rand_rngs[3], shape=(),
            minval=self._ground_pitch_range[0],
            maxval=self._ground_pitch_range[1]
        )
        self._joint_damping = jax.random.uniform(
            domain_rand_rngs[4], shape=self.sys.dof.damping.shape,
            minval=self._joint_damping_perc_range[0] * self.sys.dof.damping,
            maxval=self._joint_damping_perc_range[1] * self.sys.dof.damping
        )
        self._joint_gain = jax.random.uniform(
            domain_rand_rngs[5], shape=self.sys.actuator.gain.shape,
            minval=self._joint_gain_range[0],
            maxval=self._joint_gain_range[1]
        )
        self._link_mass_perc = jax.random.uniform(
            domain_rand_rngs[6], shape=self.sys.link.inertia.mass.shape,
            minval=self._link_mass_perc_range[0],
            maxval=self._link_mass_perc_range[1]
        )


        # initialize metrics

        # we use info to pass along quantities for domain randomization
        feet_air_time = jp.zeros(2)
        last_contacts = jp.array([False, False], dtype=jp.bool_)
        last_root_vel = jp.zeros(6)
        last_last_actions = jp.zeros(self.action_size)
        last_actions = jp.zeros(self.action_size)
        actions = jp.zeros(self.action_size)
        info = {
            'contact_time_const': self._contact_time_const,
            'contact_damping_ratio': self._contact_damping_ratio,
            'friction': self._friction,
            'ground_roll': self._ground_roll,
            'ground_pitch': self._ground_pitch,
            'joint_damping': self._joint_damping,
            'joint_gain': self._joint_gain,
            'link_mass_perc': self._link_mass_perc,
        }

        # compute mass matrix and bias + passive forces
        sys = self.sys
        x, xd = kinematics.forward(sys, q, qd)
        rew_info = {
            'gait_process': jp.zeros(()),
            'gait_frequency': jp.ones(())*1.5,
            'last_last_actions': last_last_actions,
            'last_actions': last_actions,
            'rigid_state_pos': x.pos,
            'rigid_state_lin_vel': xd.vel,
            'rigid_state_ang_vel': xd.ang,
            'rigid_state_rot': x.rot,
            'rigid_state_qdd': pipeline_state.qdd,

        }

        empty_cmd = ControlCommand(
            q_des=jp.zeros((12,)),
            qd_des=jp.zeros((12,)),
            Kp=jp.zeros((12,)),
            Kd=jp.zeros((12,)),
        )
        info['cmd'] = empty_cmd

        # initial observations, reward, done, and u
        norm_actor_state, norm_critic_state = self._get_obs(
            pipeline_state,
            rew_info,
            jp.zeros(self.action_size),
            jp.zeros(self.action_size),
        )

        norm_wm_state = self._get_wm_state(pipeline_state, rew_info, jp.zeros(self.action_size), jp.zeros(self.action_size))

        # compute cmd for info
        _, rew_components = self.compute_reward(
            norm_wm_state, norm_wm_state, jp.zeros(self.action_size), jp.zeros(self.action_size), rew_info, 0.0)

        metrics = {k: jp.zeros(()) for k, v in rew_components.items()}

        metrics.update(
            step_count=jp.zeros(()),
            forward_vel=jp.zeros(()),
            )

        rew_info.update({
            'gait_process': jp.fmod(rew_info['gait_process'] + self.sim_dt*self.policy_repeat * rew_info['gait_frequency'], 1.0), 
        })


        obs = {
            'state': norm_actor_state,
            'privileged_state': norm_critic_state,
            'wm_state': norm_wm_state,
        }
        reward, done = jp.zeros(2)
        u = jp.zeros(self.action_size)

        return State(pipeline_state, obs, reward, done, metrics,
                     info=info, u=u, rew_info=rew_info, torque=jp.zeros(self.action_size))


    def low_level_control(self, scaled_action: jp.ndarray,
                          unused_norm_obs: jp.ndarray) -> jp.ndarray:
        # Here we simply return the action as we are treating as "control
        # inputs" to the env the actions. Low level PD torque control is
        # instead absorbed into the appoximate dynamics.
        return scaled_action

    def step(self, state: State, action: jp.ndarray) -> State:

        # overwrite system contact properties with the environment's
        rew_info = state.rew_info
        sys = self._update_system_properties(state)

        # get observations from state
        prev_norm_wm_state = self._get_wm_state(state.pipeline_state, rew_info, state.u, state.torque)

        def f(info_list, _):
            pipeline_state, _, _ = info_list
            torque, cmd = self.torque_pd_control(action, g1Utils.KP, g1Utils.KD, pipeline_state.q[g1Utils.xml_q_idxs], pipeline_state.qd[g1Utils.xml_qd_idxs])
            pipeline_state = pipeline_step(sys, pipeline_state, torque)
            return (pipeline_state, torque, cmd), _

        (new_pipeline_state, torque, cmd), _ = jax.lax.scan(f, (state.pipeline_state, state.torque, state.info["cmd"]),
                                             (), self.policy_repeat_pd)
        # get new observations and compute reward
        sys = self.sys

        x, xd = new_pipeline_state.x, new_pipeline_state.xd
        rew_info.update({
            'rigid_state_pos': x.pos,
            'rigid_state_lin_vel': xd.vel,
            'rigid_state_ang_vel': xd.ang,
            'rigid_state_rot': x.rot,
            'rigid_state_qdd': new_pipeline_state.qdd,
        })

        norm_actor_state, norm_critic_state = self._get_obs(
            new_pipeline_state,
            rew_info,
            action,
            torque,
        )
        # Use current action for wm_state to align with mujoco_playground envs
        new_norm_wm_state = self._get_wm_state(new_pipeline_state, rew_info, action, torque)
        new_wm_state = g1Utils.denormalize_state(new_norm_wm_state, g1Utils.wm_state_limits)


        reward, rew_components = self.compute_reward(
            norm_wm_state=new_norm_wm_state, prev_norm_wm_state=prev_norm_wm_state, torques=torque, action=action, info=rew_info, valid_step=1.0)
        metrics = {k: v for k, v in rew_components.items()}
        metrics.update(
            forward_vel=new_wm_state[g1Utils.wm_base_lin_vel_x_idx],
            step_count=state.metrics['step_count'] + 1,
        )

        # compute dones for resets
        done = self.is_done(new_norm_wm_state, rew_info)
        info = state.info
        info['cmd'] = cmd
        new_rew_info = {}
        new_rew_info['last_last_actions'] = rew_info['last_actions']
        new_rew_info['last_actions'] = action
        new_rew_info.update({
            'rigid_state_pos': x.pos,
            'rigid_state_lin_vel': xd.vel,
            'rigid_state_ang_vel': xd.ang,
            'rigid_state_rot': x.rot,
            'rigid_state_qdd': new_pipeline_state.qdd,
            'gait_frequency': rew_info['gait_frequency'],
            'gait_process': jp.fmod(rew_info['gait_process'] + self.sim_dt*self.policy_repeat * rew_info['gait_frequency'], 1.0), 

        })

        obs_dict = {
            'state': norm_actor_state,
            'privileged_state': norm_critic_state,
            'wm_state': new_norm_wm_state,
        }

        return State(new_pipeline_state, obs_dict, reward, done, metrics,
                     info=info, u=action, rew_info=new_rew_info, torque=torque)


    def is_done(self, next_norm_obs: jp.ndarray, info: Dict) -> jp.ndarray:
        """Returns the done signal."""
        done = 1.0 - self._is_healthy(next_norm_obs)
        return done


    def _is_healthy(self, next_norm_wm_state: jp.ndarray) -> jp.ndarray:
        """Returns the healthy signal."""
        next_wm_state = g1Utils.denormalize_state(next_norm_wm_state, g1Utils.wm_state_limits)
        quat = next_wm_state[g1Utils.wm_quat_idxs] / jp.linalg.norm(next_wm_state[g1Utils.wm_quat_idxs])

        roll, pitch, yaw = quat_to_eulerzyx(next_wm_state[g1Utils.wm_quat_idxs])

        base_vel_body = next_wm_state[g1Utils.wm_base_lin_vel_idxs]
        base_vel_global = rotate(base_vel_body, quat)
        ang_vel_body = next_wm_state[g1Utils.wm_base_ang_vel_idxs]
        ang_vel_global = rotate(ang_vel_body, quat)

        # Combine all conditions into one single condition using jp.logical_and:
        condition = next_wm_state[g1Utils.wm_h_idx] > g1Utils.terminate_height_min
        condition = jp.logical_and(condition, next_wm_state[g1Utils.wm_h_idx] < g1Utils.terminate_height_max)
        condition = jp.logical_and(condition, jp.all(jp.abs(base_vel_global) < g1Utils.terminate_lin_vel_max))
        condition = jp.logical_and(condition, jp.all(jp.abs(ang_vel_global) < g1Utils.terminate_ang_vel_max))
        condition = jp.logical_and(condition, jp.abs(roll) < g1Utils.fallen_roll)
        condition = jp.logical_and(condition, jp.abs(pitch) < g1Utils.fallen_pitch)
        condition = jp.logical_and(condition, jp.all(next_wm_state[g1Utils.wm_q_idxs] < g1Utils.UPPER_JOINT_LIMITS*1.5))
        condition = jp.logical_and(condition, jp.all(next_wm_state[g1Utils.wm_q_idxs] > g1Utils.LOWER_JOINT_LIMITS*1.5))
        condition = jp.logical_and(condition, jp.all(next_wm_state[g1Utils.wm_qd_idxs] < g1Utils.MOTOR_VEL_LIMIT*4))
        condition = jp.logical_and(condition, jp.all(next_wm_state[g1Utils.wm_qd_idxs] > -g1Utils.MOTOR_VEL_LIMIT*4))

        # Now use jp.where with the single composite condition.
        is_healthy = jp.where(condition, 1.0, 0.0)


        return is_healthy

    def is_done_in_wm(self, next_norm_wm_state: jp.ndarray, info: Dict) -> jp.ndarray:
        """Returns the done signal."""
        is_done = self.is_done(next_norm_wm_state, info)
        return is_done

    def _update_system_properties(self, state: State):
        """Updates the system properties used for physics simulation with
        values that were set by the domain randomization"""
        sys = self.sys

        contact_time_const = state.info['contact_time_const']
        contact_damping_ratio = state.info['contact_damping_ratio']
        friction = state.info['friction']
        ground_roll = state.info['ground_roll']
        ground_pitch = state.info['ground_pitch']
        ground_quat = eulerzyx_to_quat(jp.array([ground_roll, ground_pitch, 0.0]))
        new_geoms = [
            g.replace(
                solver_params=g.solver_params.at[0, 0].set(contact_time_const)
            ) for g in sys.geoms
        ]
        new_geoms = [
            g.replace(
                solver_params=g.solver_params.at[0, 1].set(contact_damping_ratio)
            ) for g in new_geoms
        ]
        new_geoms = [
            g.replace(
                friction=g.friction.at[:].set(friction)
            ) for g in new_geoms
        ]
        new_geoms[0] = new_geoms[0].replace(
            transform=new_geoms[0].transform.replace(
                rot=new_geoms[0].transform.rot.at[0, :].set(ground_quat)
            )
        )

        joint_damping = state.info['joint_damping']
        new_dof = sys.dof.replace(
            damping=sys.dof.damping.at[:].set(joint_damping)
        )

        joint_gain = state.info['joint_gain']
        new_actuator = sys.actuator.replace(
            gain=sys.actuator.gain.at[:].set(joint_gain)
        )

        link_mass_perc = state.info['link_mass_perc']
        new_link = sys.link.replace(
            inertia=sys.link.inertia.replace(
                mass=sys.link.inertia.mass.at[:].set(
                    link_mass_perc * sys.link.inertia.mass
                ),
                i=sys.link.inertia.i.at[:, :, :].set(
                    jp.expand_dims(link_mass_perc, axis=(1, 2)) * sys.link.inertia.i
                )
            )
        )

        sys = sys.replace(geoms=new_geoms, dof=new_dof, actuator=new_actuator,
                          link=new_link)
        return sys


    def dynamics_contact_integrate_only(
        self, norm_wm_state: jp.ndarray, desired_q: jp.ndarray,
        ext_forces: jp.ndarray,
        norm_wm_state_next: Optional[jp.ndarray] = None
    ) -> jp.ndarray:



        wm_state = g1Utils.denormalize_state(norm_wm_state, g1Utils.wm_state_limits)
        wm_state_next = g1Utils.denormalize_state(norm_wm_state_next, g1Utils.wm_state_limits)

        # get q and qd from obs
        q, qd = self.q_and_qd_from_wm_state(wm_state)

        # compute mass matrix and bias + passive forces
        sys = self.sys
        x, xd = kinematics.forward(sys, q, qd)
        state = GeneralizedState.init(q, qd, x, xd)
        state = dynamics.transform_com(sys, state)
        state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
        state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))
        qf_smooth = dynamics.forward(sys, state, jp.zeros(sys.qd_size()))
        qf_constraint = self._qfc_fn(state, ext_forces)
        state = state.replace(qf_constraint=qf_constraint)

        def f(state_torque, _):
            state, _ = state_torque
            torques_pd, _ = self.torque_pd_control(desired_q,  g1Utils.KP, g1Utils.KD, state.q[g1Utils.xml_q_idxs], state.qd[g1Utils.xml_qd_idxs])
            tau = actuator.to_tau(sys, torques_pd, state.q, state.qd)
            state = state.replace(qf_smooth=(qf_smooth + tau))
            state = integrator.integrate(sys, state)
            torques_pd = torques_pd.astype(desired_q.dtype)

            return (state, torques_pd), None

        (state, torques) , _ = jax.lax.scan(f, (state, jp.zeros_like(desired_q)), (), self.policy_repeat)

        norm_actor_state_new, norm_critic_state_new, norm_wm_state_new = self._get_obs_approx(
            state,
            wm_state_next,
            desired_q,
            torques,
        )

        return (
            norm_wm_state_new,
            state.x.pos,
            state.xd.vel,
            state.xd.ang,
            state.x.rot,
            state.qdd,
            torques,
            norm_actor_state_new,
            norm_critic_state_new,
        )

    def mbpo_dynamics(self, norm_wm_state: jp.ndarray, desired_q: jp.ndarray,
                      ext_forces: Optional[jp.ndarray] = None,
                      norm_wm_state_next: Optional[jp.ndarray] = None):
        norm_wm_state_new = norm_wm_state + ext_forces # TODO: clock may bug
        wm_state_next = g1Utils.denormalize_state(norm_wm_state_next, g1Utils.wm_state_limits)
        wm_state_prev = g1Utils.denormalize_state(norm_wm_state, g1Utils.wm_state_limits)
        # get q and qd from obs
        q, qd = self.q_and_qd_from_wm_state(wm_state_next)

        # compute mass matrix and bias + passive forces
        sys = self.sys
        x, xd = kinematics.forward(sys, q, qd)
        state = GeneralizedState.init(q, qd, x, xd)
        state = dynamics.transform_com(sys, state)
        state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
        state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))
        q = wm_state_prev[g1Utils.wm_q_idxs]
        qd = wm_state_prev[g1Utils.wm_qd_idxs]
        torques, _ = self.torque_pd_control(desired_q,  g1Utils.KP, g1Utils.KD, q, qd)
        return norm_obs_new, state.x.pos, state.xd.vel, state.xd.ang, state.x.rot, state.qdd, torques


    def q_and_qd_from_wm_state(self, wm_state: jp.ndarray):
        q = self.sys.init_q
        # normalize quat
        quat = wm_state[g1Utils.wm_quat_idxs] / jp.linalg.norm(wm_state[g1Utils.wm_quat_idxs])
        q = q.at[g1Utils.xml_quat_idxs].set(quat)
        q = q.at[g1Utils.xml_q_idxs].set(wm_state[g1Utils.wm_q_idxs])
        q = q.at[g1Utils.xml_h_idxs].set(wm_state[g1Utils.wm_h_idx])

        base_vel_body = wm_state[g1Utils.wm_base_lin_vel_idxs]
        base_vel_global = rotate(base_vel_body, quat)
        ang_vel_body = wm_state[g1Utils.wm_base_ang_vel_idxs]
        qd = jp.zeros(self.sys.qd_size())
        qd = qd.at[g1Utils.xml_base_vel_idxs].set(base_vel_global)
        qd = qd.at[g1Utils.xml_rpy_rate_idxs].set(ang_vel_body)
        qd = qd.at[g1Utils.xml_qd_idxs].set(wm_state[g1Utils.wm_qd_idxs])

        return q, qd

    def make_ssrl_dynamics_fn(self, fn_type) -> Callable:

        fn = {
            'contact_integrate_only': self.dynamics_contact_integrate_only,
            'mbpo': self.mbpo_dynamics
        }[fn_type]

        def dynamics_fn(norm_wm_state: jp.ndarray, desired_q: jp.ndarray, pred: jp.ndarray):
            wm_state = g1Utils.denormalize_state(norm_wm_state, g1Utils.wm_state_limits)
            gait_process = wm_state[g1Utils.wm_gait_process_idx]
            gait_frequency = wm_state[g1Utils.wm_gait_frequency_idx]
            new_gait_process = jp.fmod(gait_process + self.sim_dt*self.policy_repeat * gait_frequency, 1.0)

            # TODO: COMMAND,
            wm_state = wm_state.at[g1Utils.wm_cos_phase_idx].set(
                jp.cos(2 * jp.pi * new_gait_process)
                * (gait_frequency > 1.0e-8).astype(new_gait_process.dtype)
                * (jp.linalg.norm(wm_state[g1Utils.wm_commands_idxs]) > 0.01).astype(new_gait_process.dtype)
            )
            wm_state = wm_state.at[g1Utils.wm_sin_phase_idx].set(
                jp.sin(2 * jp.pi * new_gait_process)
                * (gait_frequency > 1.0e-8).astype(new_gait_process.dtype)
                * (jp.linalg.norm(wm_state[g1Utils.wm_commands_idxs]) > 0.01).astype(new_gait_process.dtype)
            )
            wm_state = wm_state.at[g1Utils.wm_gait_process_idx].set(new_gait_process)


            norm_wm_state_next = g1Utils.normalize_state(wm_state, g1Utils.wm_state_limits)


            return fn(norm_wm_state, desired_q, pred, norm_wm_state_next)

        return dynamics_fn

    def _get_wm_state(self, pipeline_state: base.State,
                 info: dict,
                 last_action: jp.ndarray, # TODO
                 last_torque: jp.ndarray, # TODO
                 ) -> jp.ndarray:
        """uses metrics to compute phase and desired velocity"""

        quat, q, base_vel_body, ang_vel_body, qd, q_all, qd_all = self._get_basic_obs(pipeline_state)

        x, xd = pipeline_state.x, pipeline_state.xd

        gait_process = info['gait_process']
        gait_frequency = info['gait_frequency']

        foot_speed = xd.vel[g1Utils.feet_indices]
        cos_phase = jp.cos(2*jp.pi*gait_process) * (gait_frequency > 1.0e-8).astype(gait_process.dtype) * (jp.linalg.norm(self.commands[:3]) > 0.01).astype(gait_process.dtype)
        sin_phase = jp.sin(2*jp.pi*gait_process) * (gait_frequency > 1.0e-8).astype(gait_process.dtype) * (jp.linalg.norm(self.commands[:3]) > 0.01).astype(gait_process.dtype)
        phase = jp.array([cos_phase, sin_phase])
        phase = jp.where(
            jp.linalg.norm(self.commands[:3]) > 0.01,
            phase,
            jp.zeros(2),
        )
        #torques, _ = self.torque_pd_control(last_action, q, qd)
        torques = last_torque

        projected_gravity = inv_rotate(jp.array([0.0, 0.0, -1.0]), quat)

        h = pipeline_state.q[g1Utils.xml_h_idxs]



        wm_state = jp.zeros((g1Utils.wm_observation_size,))
        wm_state = wm_state.at[g1Utils.wm_gravity_idxs].set(projected_gravity)
        wm_state = wm_state.at[g1Utils.wm_base_ang_vel_idxs].set(ang_vel_body)
        wm_state = wm_state.at[g1Utils.wm_q_idxs].set(q)
        wm_state = wm_state.at[g1Utils.wm_qd_idxs].set(qd)
        wm_state = wm_state.at[g1Utils.wm_commands_idxs].set(jax.lax.stop_gradient(self.commands[:3]))
        wm_state = wm_state.at[g1Utils.wm_cos_phase_idx].set(jax.lax.stop_gradient(phase[0]))
        wm_state = wm_state.at[g1Utils.wm_sin_phase_idx].set(jax.lax.stop_gradient(phase[1]))
        wm_state = wm_state.at[g1Utils.wm_last_action_idxs].add(jax.lax.stop_gradient(last_action))
        wm_state = wm_state.at[g1Utils.wm_quat_idxs].set(quat)
        wm_state = wm_state.at[g1Utils.wm_base_lin_vel_idxs].set(base_vel_body)
        wm_state = wm_state.at[g1Utils.wm_h_idx].set(h)
        wm_state = wm_state.at[g1Utils.wm_gait_process_idx].set(jax.lax.stop_gradient(gait_process))
        wm_state = wm_state.at[g1Utils.wm_gait_frequency_idx].set(jax.lax.stop_gradient(gait_frequency))

        norm_wm_state = g1Utils.normalize_state(wm_state, g1Utils.wm_state_limits)

        return norm_wm_state

    def _get_obs(self, pipeline_state: base.State,
                 info: dict,
                 last_action: jp.ndarray, # TODO
                 last_torque: jp.ndarray, # TODO
                 ) -> jp.ndarray:
        """uses metrics to compute phase and desired velocity"""

        quat, q, base_vel_body, ang_vel_body, qd, q_all, qd_all = self._get_basic_obs(pipeline_state)

        x, xd = pipeline_state.x, pipeline_state.xd


        gait_process = info['gait_process']
        gait_frequency = info['gait_frequency']

        foot_speed = xd.vel[g1Utils.feet_indices]
        cos_phase = jp.cos(2*jp.pi*gait_process) * (gait_frequency > 1.0e-8).astype(gait_process.dtype) * (jp.linalg.norm(self.commands[:3]) > 0.01).astype(gait_process.dtype)
        sin_phase = jp.sin(2*jp.pi*gait_process) * (gait_frequency > 1.0e-8).astype(gait_process.dtype) * (jp.linalg.norm(self.commands[:3]) > 0.01).astype(gait_process.dtype)
        phase = jp.array([cos_phase, sin_phase])
        phase = jp.where(
            jp.linalg.norm(self.commands[:3]) > 0.01,
            phase,
            jp.zeros(2),
        )
        #torques, _ = self.torque_pd_control(last_action, q, qd)
        torques = last_torque

        projected_gravity = inv_rotate(jp.array([0.0, 0.0, -1.0]), quat)

        h = pipeline_state.q[g1Utils.xml_h_idxs]

        actor_state = jp.zeros((g1Utils.observation_size,))
        actor_state = actor_state.at[g1Utils.actor_gravity_idxs].add(jax.lax.stop_gradient(projected_gravity))
        actor_state = actor_state.at[g1Utils.actor_base_ang_vel_idxs].add(jax.lax.stop_gradient(ang_vel_body))
        actor_state = actor_state.at[g1Utils.actor_q_idxs].add(jax.lax.stop_gradient(q))
        actor_state = actor_state.at[g1Utils.actor_qd_idxs].add(jax.lax.stop_gradient(qd))
        actor_state = actor_state.at[g1Utils.actor_commands_idxs].set(jax.lax.stop_gradient(self.commands[:3]))
        actor_state = actor_state.at[g1Utils.actor_cos_phase_idx].set(jax.lax.stop_gradient(phase[0]))
        actor_state = actor_state.at[g1Utils.actor_sin_phase_idx].set(jax.lax.stop_gradient(phase[1]))
        actor_state = actor_state.at[g1Utils.actor_last_action_idxs].add(jax.lax.stop_gradient(last_action))

        privileged_state = jp.zeros((g1Utils.priv_observation_size,))
        privileged_state = privileged_state.at[g1Utils.priv_gravity_idxs].add(jax.lax.stop_gradient(projected_gravity))
        privileged_state = privileged_state.at[g1Utils.priv_base_ang_vel_idxs].add(jax.lax.stop_gradient(ang_vel_body))
        privileged_state = privileged_state.at[g1Utils.priv_q_idxs].add(jax.lax.stop_gradient(q))
        privileged_state = privileged_state.at[g1Utils.priv_qd_idxs].add(jax.lax.stop_gradient(qd))
        privileged_state = privileged_state.at[g1Utils.priv_commands_idxs].set(jax.lax.stop_gradient(self.commands[:3]))
        privileged_state = privileged_state.at[g1Utils.priv_cos_phase_idx].set(jax.lax.stop_gradient(phase[0]))
        privileged_state = privileged_state.at[g1Utils.priv_sin_phase_idx].set(jax.lax.stop_gradient(phase[1]))
        privileged_state = privileged_state.at[g1Utils.priv_last_action_idxs].add(jax.lax.stop_gradient(last_action))
        privileged_state = privileged_state.at[g1Utils.priv_quat_idxs].add(jax.lax.stop_gradient(quat))
        privileged_state = privileged_state.at[g1Utils.priv_base_lin_vel_idxs].add(jax.lax.stop_gradient(base_vel_body))
        privileged_state = privileged_state.at[g1Utils.priv_h_idx].set(jax.lax.stop_gradient(h))

        norm_actor_state = g1Utils.normalize_state(actor_state, g1Utils.actor_state_limits)
        norm_critic_state = g1Utils.normalize_state(privileged_state, g1Utils.priv_state_limits)

        return norm_actor_state, norm_critic_state

    def _get_obs_approx(self, pipeline_state: base.State,
                wm_state_next: jp.ndarray,
                 last_action: jp.ndarray,
                 last_torque: jp.ndarray,
                 ) -> jp.ndarray:
        """uses the next observation to compute phase and desired velocity
        (this is OK since these obervations do not depend on the dynamics of
        the system)"""

        quat, q, base_vel_body, ang_vel_body, qd, q_all, qd_all = self._get_basic_obs(pipeline_state)
        #x, xd = kinematics.forward(self.sys, q_all, qd_all)
        #foot_speed = xd.vel[g1Utils.feet_indices]
        cos_phase = wm_state_next[g1Utils.wm_cos_phase_idx]
        sin_phase = wm_state_next[g1Utils.wm_sin_phase_idx]
        gait_process = wm_state_next[g1Utils.wm_gait_process_idx]
        gait_frequency = wm_state_next[g1Utils.wm_gait_frequency_idx]
        command = wm_state_next[g1Utils.wm_commands_idxs]
        phase = jp.array([cos_phase, sin_phase])


        projected_gravity = inv_rotate(jp.array([0.0, 0.0, -1.0]), quat)
        gait = jp.array([
            gait_process, gait_frequency
        ])
        h = pipeline_state.q[g1Utils.xml_h_idxs]

        actor_state = jp.zeros((g1Utils.observation_size,))
        actor_state = actor_state.at[g1Utils.actor_gravity_idxs].add(jax.lax.stop_gradient(projected_gravity))
        actor_state = actor_state.at[g1Utils.actor_base_ang_vel_idxs].add(jax.lax.stop_gradient(ang_vel_body))
        actor_state = actor_state.at[g1Utils.actor_q_idxs].add(jax.lax.stop_gradient(q))
        actor_state = actor_state.at[g1Utils.actor_qd_idxs].add(jax.lax.stop_gradient(qd))
        actor_state = actor_state.at[g1Utils.actor_commands_idxs].set(jax.lax.stop_gradient(command))
        actor_state = actor_state.at[g1Utils.actor_cos_phase_idx].set(jax.lax.stop_gradient(phase[0]))
        actor_state = actor_state.at[g1Utils.actor_sin_phase_idx].set(jax.lax.stop_gradient(phase[1]))
        actor_state = actor_state.at[g1Utils.actor_last_action_idxs].add(jax.lax.stop_gradient(last_action))

        privileged_state = jp.zeros((g1Utils.priv_observation_size,))
        privileged_state = privileged_state.at[g1Utils.priv_gravity_idxs].add(jax.lax.stop_gradient(projected_gravity))
        privileged_state = privileged_state.at[g1Utils.priv_base_ang_vel_idxs].add(jax.lax.stop_gradient(ang_vel_body))
        privileged_state = privileged_state.at[g1Utils.priv_q_idxs].add(jax.lax.stop_gradient(q))
        privileged_state = privileged_state.at[g1Utils.priv_qd_idxs].add(jax.lax.stop_gradient(qd))
        privileged_state = privileged_state.at[g1Utils.priv_commands_idxs].set(jax.lax.stop_gradient(command))
        privileged_state = privileged_state.at[g1Utils.priv_cos_phase_idx].set(jax.lax.stop_gradient(phase[0]))
        privileged_state = privileged_state.at[g1Utils.priv_sin_phase_idx].set(jax.lax.stop_gradient(phase[1]))
        privileged_state = privileged_state.at[g1Utils.priv_last_action_idxs].add(jax.lax.stop_gradient(last_action))
        privileged_state = privileged_state.at[g1Utils.priv_quat_idxs].add(jax.lax.stop_gradient(quat))
        privileged_state = privileged_state.at[g1Utils.priv_base_lin_vel_idxs].add(jax.lax.stop_gradient(base_vel_body))
        privileged_state = privileged_state.at[g1Utils.priv_h_idx].set(jax.lax.stop_gradient(h))



        wm_state = jp.zeros((g1Utils.wm_observation_size,))
        wm_state = wm_state.at[g1Utils.wm_gravity_idxs].set(projected_gravity)
        wm_state = wm_state.at[g1Utils.wm_base_ang_vel_idxs].set(ang_vel_body)
        wm_state = wm_state.at[g1Utils.wm_q_idxs].set(q)
        wm_state = wm_state.at[g1Utils.wm_qd_idxs].set(qd)
        wm_state = wm_state.at[g1Utils.wm_commands_idxs].set(jax.lax.stop_gradient(command))
        wm_state = wm_state.at[g1Utils.wm_cos_phase_idx].set(jax.lax.stop_gradient(phase[0]))
        wm_state = wm_state.at[g1Utils.wm_sin_phase_idx].set(jax.lax.stop_gradient(phase[1]))
        wm_state = wm_state.at[g1Utils.wm_last_action_idxs].add(jax.lax.stop_gradient(last_action))
        wm_state = wm_state.at[g1Utils.wm_quat_idxs].set(quat)
        wm_state = wm_state.at[g1Utils.wm_base_lin_vel_idxs].set(base_vel_body)
        wm_state = wm_state.at[g1Utils.wm_h_idx].set(h)
        wm_state = wm_state.at[g1Utils.wm_gait_process_idx].set(jax.lax.stop_gradient(gait_process))
        wm_state = wm_state.at[g1Utils.wm_gait_frequency_idx].set(jax.lax.stop_gradient(gait_frequency))

        norm_wm_state = g1Utils.normalize_state(wm_state, g1Utils.wm_state_limits)

        norm_actor_state = g1Utils.normalize_state(actor_state, g1Utils.actor_state_limits)
        norm_critic_state = g1Utils.normalize_state(privileged_state, g1Utils.priv_state_limits)

        return norm_actor_state, norm_critic_state, norm_wm_state

    def _get_basic_obs(self, pipeline_state: base.State) -> jp.ndarray:
        "Returns basic observations without phase and desired velocity"

        positions = pipeline_state.q
        velocities = pipeline_state.qd

        # quat orientation of the base
        quat = positions[g1Utils.xml_quat_idxs]

        # joint angles
        q = positions[g1Utils.xml_q_idxs]

        # linear velocity of the base in the body frame
        base_vel_global = velocities[g1Utils.xml_base_vel_idxs]
        base_vel_body = inv_rotate(base_vel_global, quat)

        # angular velocity of the base in the body frame
        ang_vel_body = velocities[g1Utils.xml_rpy_rate_idxs]

        # joint speeds
        qd = velocities[g1Utils.xml_qd_idxs]

        return quat, q, base_vel_body, ang_vel_body, qd, positions, velocities


    def compute_reward(self, norm_wm_state: jp.ndarray, prev_norm_wm_state: jp.ndarray,
                       torques: jp.ndarray,
                       action: jp.ndarray,
                    info:dict,
                    valid_step: jp.ndarray,
                       ) -> jp.ndarray:
        wm_state = g1Utils.denormalize_state(norm_wm_state, g1Utils.wm_state_limits)
        prev_wm_state = g1Utils.denormalize_state(prev_norm_wm_state, g1Utils.wm_state_limits)
        reward, reward_components = self._reward_normalized(wm_state=wm_state, prev_wm_state=prev_wm_state, torques=torques, action=action,
                                                    info=info, valid_step=valid_step)

        return reward, reward_components

    def _reward_normalized(self, wm_state: jp.ndarray,
                           prev_wm_state: jp.ndarray,
                           torques: jp.ndarray,
                           action: jp.ndarray,
                            info: dict,
                            valid_step: jp.ndarray,
                            ) -> jp.ndarray:
        def do_main(args):
            wm_state, prev_wm_state, torques, action, info = args
            gait_process = wm_state[g1Utils.wm_gait_process_idx]
            gait_frequency = wm_state[g1Utils.wm_gait_frequency_idx]
            last_actions = info['last_actions']
            last_last_actions = info['last_last_actions']
            rigid_state_pos = info['rigid_state_pos']
            rigid_state_lin_vel = info['rigid_state_lin_vel']
            _ = info['rigid_state_ang_vel']
            rigid_state_rot = info['rigid_state_rot']
            rigid_state_qdd = info['rigid_state_qdd']

            quat = wm_state[g1Utils.wm_quat_idxs] / jp.linalg.norm(wm_state[g1Utils.wm_quat_idxs])

            base_lin_vel = wm_state[g1Utils.wm_base_lin_vel_idxs]
            base_ang_vel = wm_state[g1Utils.wm_base_ang_vel_idxs]
            base_euler_xyz = quat_to_euler(quat)
            projected_gravity = inv_rotate(jp.array([0.0, 0.0, -1.0]), quat)
            contact = rigid_state_pos[g1Utils.feet_indices, 2] < self.mini_ankle_dist

            commands = wm_state[g1Utils.wm_commands_idxs]

            feet_pos = rigid_state_pos[g1Utils.feet_indices, :3]
            feet_quat = rigid_state_rot[g1Utils.feet_indices, :4]
            feet_euler_xyz_0 = quat_to_euler(feet_quat[0])
            feet_euler_xyz_1 = quat_to_euler(feet_quat[1])
            feet_euler_xyz = jp.array([feet_euler_xyz_0, feet_euler_xyz_1])
            feet_roll = (feet_euler_xyz[:,0] + jp.pi) % (2 * jp.pi) - jp.pi
            feet_yaw = (feet_euler_xyz[:,2] + jp.pi) %  (2 * jp.pi) - jp.pi

            reward_components = {
                # -------- positive terms --------

                "survival": jp.array(1.0),
                "tracking_lin_vel": self._reward_tracking_lin_vel(commands, base_lin_vel),
                "tracking_lin_vel_x": self._reward_tracking_lin_vel_axis(0, commands, base_lin_vel),
                "tracking_lin_vel_y": self._reward_tracking_lin_vel_axis(1, commands, base_lin_vel),
                "tracking_ang_vel": self._reward_tracking_ang_vel(commands, base_ang_vel),
                "feet_swing": self._reward_feet_swing(gait_process=gait_process, gait_frequency=gait_frequency, feet_contact=contact, commands=commands),
                "feet_height": self._cost_feet_height(feet_pos[:, -1], contact, info),

                # -------- penalties ------------ (signed handled by scale)
                "base_height": self._reward_base_height(height=rigid_state_pos[0, 2]),
                "orientation": self._cost_orientation(projected_gravity),
                "torques": self._cost_torques(torques),
                "torque_tiredness": self._cost_torque_tiredness(torques),
                "power": self._cost_energy(qvel=wm_state[g1Utils.wm_qd_idxs], torques=torques),
                "lin_vel_z": self._cost_lin_vel_z(base_lin_vel),
                "ang_vel_xy": self._cost_ang_vel_xy(base_ang_vel),
                "dof_vel": self._cost_dof_vel(wm_state[g1Utils.wm_qd_idxs]),
                "dof_acc": self._cost_dof_acc(rigid_state_qdd[6:]),
                "root_acc": self._cost_root_acc(rigid_state_qdd),
                "action_rate": self._cost_action_rate(act=action, last_act=last_actions, last_last_act=last_last_actions),
                "dof_pos_limits": self._cost_joint_pos_limits(wm_state[g1Utils.wm_q_idxs]),
                "feet_slip": self._cost_feet_slip(feet_vel=rigid_state_lin_vel[g1Utils.feet_indices], contact=contact),
                "feet_vel_z": self._cost_feet_vel_z(feet_vel=rigid_state_lin_vel[g1Utils.feet_indices]),
                "feet_roll": self._cost_feet_roll(feet_roll=feet_roll),
                "feet_yaw_diff": self._cost_feet_yaw_diff(feet_yaw=feet_yaw),
                "feet_yaw_mean": self._cost_feet_yaw_mean(feet_yaw=feet_yaw, base_yaw=base_euler_xyz[2]),
                "feet_distance": self._cost_feet_distance(feet_pos=feet_pos, base_yaw=base_euler_xyz[2]),
                "joint_deviation_hip": self._cost_joint_deviation_hip(
                   wm_state[g1Utils.wm_q_idxs]
                ),
            }

            rewards_comp = {k: v * self.reward_config.scales[k] for k, v in reward_components.items()}

            reward = jp.clip(sum(rewards_comp.values()) * self.sim_dt * self.policy_repeat, 0.0, 10000.0)
            rewards_comp['reward']  = reward
            return reward, rewards_comp
        def do_zero(args):
            args = args
            return 0.0, {
                "reward": jp.array(0.0),
                "survival": jp.array(0.0),
                "tracking_lin_vel": jp.array(0.0),
                "tracking_lin_vel_x": jp.array(0.0),
                "tracking_lin_vel_y": jp.array(0.0),
                "tracking_ang_vel": jp.array(0.0),
                "feet_swing": jp.array(0.0),
                "feet_height": jp.array(0.0),
                "base_height": jp.array(0.0),
                "orientation": jp.array(0.0),
                "torques": jp.array(0.0),
                "torque_tiredness": jp.array(0.0),
                "power": jp.array(0.0),
                "lin_vel_z": jp.array(0.0),
                "ang_vel_xy": jp.array(0.0),
                "dof_vel": jp.array(0.0),
                "dof_acc": jp.array(0.0),
                "root_acc": jp.array(0.0),
                "action_rate": jp.array(0.0),
                "dof_pos_limits": jp.array(0.0),
                "feet_slip": jp.array(0.0),
                "feet_vel_z": jp.array(0.0),
                "feet_roll": jp.array(0.0),
                "feet_yaw_diff": jp.array(0.0),
                "feet_yaw_mean": jp.array(0.0),
                "feet_distance": jp.array(0.0),
                "joint_deviation_hip": jp.array(0.0),
            }

        pred = valid_step > 0
        (reward, rewards_comp) = lax.cond(
            pred,
            do_main,
            do_zero,
            (wm_state, prev_wm_state, torques, action, info)
        )

        return reward, rewards_comp

    # Tracking rewards.
    def _reward_tracking_lin_vel(
        self, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        """Axis–wise linear‑velocity tracker (matches Isaac Gym x & y trackers)."""
        err = jp.sum(jp.square(command[:2] - local_linvel[:2]))
        return jp.exp(-err / self.reward_config.tracking_sigma)


    def _reward_tracking_lin_vel_axis(
        self, axis: int, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        """Axis–wise linear‑velocity tracker (matches Isaac Gym x & y trackers)."""
        err = jp.square(command[axis] - local_linvel[axis])
        return jp.exp(-err / self.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        local_angvel: jax.Array,
    ) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - local_angvel[2])
        return jp.exp(-ang_vel_error / self.reward_config.tracking_sigma)

    # Base-related rewards.

    def _cost_lin_vel_z(self, local_linvel) -> jax.Array:
        return jp.square(local_linvel[2])

    def _cost_ang_vel_xy(self, local_angvel) -> jax.Array:
        return jp.sum(jp.square(local_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torso_zaxis[:2]))

    def _reward_base_height(self, height: float) -> jax.Array:
        return jp.exp(-jp.abs(height - self.reward_config.base_height_target)*0.1)

    # Energy related rewards.

    def _cost_torques(self, torques: jp.ndarray) -> jp.ndarray:
        return jp.sum(jp.square(torques))

    def _cost_energy(self, qvel: jp.ndarray, torques: jp.ndarray) -> jp.ndarray:
        power = qvel * torques
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
    def _cost_feet_height(
        self,
        feet_height: jax.Array,
        contact: jp.ndarray,
        info: dict[str, Any],
    ) -> jax.Array:
        del info  # Unused.
        error = feet_height -  self.reward_config.max_foot_height
        return jp.sum(jp.square(error) * ~contact)

    def _cost_joint_pos_limits(self, qpos: jp.ndarray) -> jp.ndarray:
        below = qpos < g1Utils.soft_lowers
        above = qpos > g1Utils.soft_uppers
        return jp.sum((below | above).astype(qpos.dtype))

    def _reward_survival(self) -> jax.Array:
        return jp.array(1.0)


    # Feet related rewards.

    def _cost_feet_slip(
        self,
        feet_vel: jp.ndarray,
        contact: jp.ndarray,
    ) -> jp.ndarray:
        speed2 = jp.sum(jp.square(feet_vel), axis=-1)             # per‑foot |v|²
        return jp.sum(speed2 * contact)

    def _cost_feet_distance(
        self,
        feet_pos: jax.Array,
        base_yaw: jax.Array,
    ) -> jax.Array:
        left_foot_pos = feet_pos[0]
        right_foot_pos = feet_pos[1]
        feet_distance = jp.abs(
            jp.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1])
            - jp.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
        )
        return jp.clip(0.2 - feet_distance, min=0.0, max=0.1)

    def _cost_joint_deviation_hip(
        self, qpos: jax.Array
    ) -> jax.Array:
        cost = jp.sum(jp.square(qpos[g1Utils.hip_indices]))

        return cost

    def _cost_torque_tiredness(self, torques: jax.Array) -> jax.Array:
        # Σ (τ / τ_max)²  – clipped at 1 so the term stays O(1)

        frac = jp.clip(jp.abs(torques) / jp.full_like(g1Utils.MOTOR_TORQUE_LIMIT, 1e6), 0.0, 1.0)
        return jp.sum(jp.square(frac))

    def _cost_root_acc(self, qdd: jax.Array) -> jax.Array:
        # Root‑link 6‑D acceleration² (free‑joint entries of qacc)
        return jp.sum(jp.square(qdd[:6]))

    def _cost_feet_roll(self, feet_roll: jax.Array) -> jax.Array:
        return jp.sum(jp.square(feet_roll))

    def _cost_feet_yaw_diff(self, feet_yaw: jax.Array) -> jax.Array:
        diff = jp.fmod(feet_yaw[1] - feet_yaw[0] + jp.pi, 2 * jp.pi) - jp.pi
        return jp.square(diff)

    def _cost_feet_yaw_mean(self, feet_yaw: jax.Array, base_yaw: jax.Array) -> jax.Array:
        feet_mean_yaw = jp.fmod(jp.mean(feet_yaw) + jp.pi, 2 * jp.pi) - jp.pi
        err = jp.fmod(base_yaw - feet_mean_yaw + jp.pi, 2 * jp.pi) - jp.pi

        return jp.square(err)

    def _cost_feet_vel_z(self, feet_vel: jp.ndarray) -> jax.Array:
        # use the same foot linear‑velocity sensors already wired for slip
        vz = feet_vel[:, 2]
        return jp.sum(jp.square(vz))

    def _reward_feet_swing(self,
        gait_process: jp.ndarray,
        gait_frequency: float,
        feet_contact: jp.ndarray,
        commands: jax.Array,
        ) -> jp.ndarray:
        left_swing = (jp.abs(gait_process - 0.25) < 0.5 * self.reward_config.swing_period) & (gait_frequency > 1.0e-8)
        right_swing = (jp.abs(gait_process - 0.75) < 0.5 * self.reward_config.swing_period) & (gait_frequency > 1.0e-8)
        ref_rew = (left_swing & ~feet_contact[0]) + (right_swing & ~feet_contact[1])
        ref_rew *= jp.linalg.norm(commands) > 0.01

        return ref_rew



    def torque_pd_control(self, action: jp.ndarray,
                          kp: jp.ndarray,
                          kd: jp.ndarray,
                          q: jp.ndarray,
                          qd: jp.ndarray) -> Tuple[jp.ndarray, ControlCommand]:

        cmd = self.low_level_control_hardware(action, kp=kp, kd=kd)

        # torque control
        u = cmd.Kp*(cmd.q_des - q) + cmd.Kd*(cmd.qd_des - qd)

        u = jp.clip(u, -g1Utils.MOTOR_TORQUE_LIMIT, g1Utils.MOTOR_TORQUE_LIMIT)   
        return u, cmd

    def low_level_control_hardware(self, action: jp.ndarray,
                                    kp: jp.ndarray,
                                    kd: jp.ndarray,
                                   ) -> ControlCommand:

        q_des = action * g1Utils.action_scale + g1Utils.ALL_STANDING_JOINT_ANGLES
        qd_des = jp.zeros((g1Utils.action_size,))
        q_des = jp.clip(q_des,
                        g1Utils.LOWER_JOINT_LIMITS,
                        g1Utils.UPPER_JOINT_LIMITS)

        return ControlCommand(q_des, qd_des, kp, kd)

    @property
    def action_size(self) -> int:
        return g1Utils.action_size

    @property
    def controls_size(self) -> int:
        return g1Utils.action_size

    @property
    def observation_size(self) -> int:
        return {
            "state": g1Utils.observation_size,
            "privileged_state": g1Utils.priv_observation_size,
            "wm_state": g1Utils.wm_observation_size,
        }

    @property
    def dt(self) -> jp.ndarray:
        """The timestep used for each env step."""
        return self.sim_dt * self.policy_repeat
