# Copyright 2025 LIFT-Humanoid Authors.
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

"""Physics Informed Model-Based Policy Optimization.
"""

from brax import envs
import lift_utils.types as types
from brax_env.brax_env_utils import Evaluator, actor_step
from lift_utils import replay_buffers
from lift_utils import gradients
from lift_utils import running_statistics

from world_model import wm_networks as wm_networks
from world_model import wm_losses as wm_losses
from world_model import wm_base as base

from policy_pretrain import losses as sac_losses
from policy_pretrain import sac_networks
from policy_pretrain import train as sac_train

from flax import linen as nn

from absl import logging
from typing import Callable, Optional, Tuple, Sequence, Union, Any
import optax
from jax import numpy as jp
import jax
import functools
import math
import time

_PMAP_AXIS_NAME = 'i'
def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)

def _slice_horizon(trans, idx, H):
    return jax.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, idx, H), trans)

def _gather_dataset(trans, idxs, H):
    return jax.vmap(lambda i: _slice_horizon(trans, i, H))(idxs)

def train(
    environment: envs.Env,
    low_level_control_fn: Callable,
    dynamics_fn: Callable,
    reward_fn: Callable,
    model_output_dim: int,
    episode_length: int,
    num_timesteps: Optional[int] = None,
    num_epochs: Optional[int] = None,
    model_trains_per_epoch: int = 4,
    training_steps_per_model_train: int = 250,
    env_steps_per_training_step: int = 1,
    hallucination_updates_per_training_step: Union[int, Callable] = 1,
    model_rollouts_per_hallucination_update: int = 400,
    sac_grad_updates_per_hallucination_update: int = 20,
    init_exploration_steps: int = 5000,
    clear_model_buffer_after_model_train: bool = True,
    action_repeat: int = 1,
    obs_history_length: int = 1,
    priv_obs_history_length: int = 1,
    wm_obs_history_length: int = 1,
    num_envs: int = 1,  # TODO remove num_envs as option or freeze at 1
    num_evals: int = 1,
    num_eval_envs: int = 1,
    policy_normalize_observations: bool = False,
    model_learning_rate: float = 1e-3,
    model_training_batch_size: int = 256,
    model_training_max_sgd_steps_per_epoch: Optional[int] = None,
    model_training_max_epochs: int = 1000,
    model_training_convergence_criteria: float = 0.01,
    model_training_consec_converged_epochs: int = 6,
    model_training_abs_criteria: Optional[float] = None,
    model_training_test_ratio: float = 0.2,
    model_training_weight_decay: bool = True,
    model_training_stop_gradient: bool = False,
    mean_loss_over_horizon: bool = False,
    model_loss_horizon: int = 10,
    model_horizon_fn: Callable[[int], int] = lambda epoch: 1,
    model_check_done_condition: bool = True,
    max_env_buffer_size: Optional[int] = None,
    max_model_buffer_size: Optional[int] = None,
    sac_learning_rate: float = 1e-4,
    sac_discounting: float = 0.99,
    sac_batch_size: int = 256,
    real_ratio: float = 0.06,
    sac_reward_scaling: float = 1.0,
    sac_tau: float = 0.005,
    sac_fixed_alpha: Optional[float] = None,
    sac_init_log_alpha: Optional[float] = 0.0,
    sac_training_state: Optional[base.SACTrainingState] = None,
    wm_training_state: Optional[base.WM_TrainingState] = None,
    seed: int = 0,
    deterministic_in_env: bool = False,
    deterministic_eval: bool = False,
    model_network_factory: Callable = wm_networks.make_model_network,
    policy_network_factory: types.NetworkFactory[
        sac_networks.SACNetworks] = sac_networks.make_sac_networks,
    hallucination_max_std: Optional[float] = -1.0,  # if <= 0, use the same from the provided sac_network_factory # noqa: E501
    progress_fn: Callable[[int, types.Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    load_ac_optimizer_state: bool = True,
    load_alpha: bool = False,
    max_devices_per_host: Optional[int] = None,
    plot_model_rollouts: bool = True,
    entropy_rate: float = 0.0,
    target_entropy_coef: float = 0.0,
    robot_config: Optional[Any] = None,
):

    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()
    logging.info(
        'local_device_count: %s; total_device_count: %s',
        local_devices_to_use,
        device_count,
    )
    # make sure either num_timesteps or num_epochs is given
    assert num_timesteps is not None or num_epochs is not None
    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)

    env = envs.training.wrap(
        environment, episode_length=episode_length,
        action_repeat=action_repeat,
        obs_history_length=obs_history_length,
        priv_obs_history_length=priv_obs_history_length,
        wm_obs_history_length=wm_obs_history_length,
        )

    env_steps_per_epoch = (model_trains_per_epoch
                           * training_steps_per_model_train
                           * env_steps_per_training_step
                           * num_envs
                           * action_repeat)

    if num_epochs is None:
        num_epochs = math.ceil(
            (num_timesteps - init_exploration_steps)
            // env_steps_per_epoch + 1
        )

    if max_env_buffer_size is None:
        max_env_buffer_size = (
            env_steps_per_epoch * num_epochs + init_exploration_steps)

    total_batches = max_env_buffer_size // model_training_batch_size
    model_max_train_batches = int(
        jp.ceil(total_batches * (1 - model_training_test_ratio)))
    model_max_test_batches = total_batches - model_max_train_batches + 1

    if isinstance(hallucination_updates_per_training_step, int):
        hallucination_updates_per_training_step_fn = (
            lambda epoch: hallucination_updates_per_training_step)
    elif callable(hallucination_updates_per_training_step):
        hallucination_updates_per_training_step_fn = (
            hallucination_updates_per_training_step)
    else:
        raise ValueError('hallucination_updates_per_training_step must be an '
                         'int or a callable')

    obs_size=env.observation_size['state']
    priv_obs_size=env.observation_size['privileged_state']
    wm_obs_size = env.observation_size['wm_state']
    wm_obs_size_per_step = wm_obs_size
    torque_dim = env.sys.qd_size()

    obs_hist_len=obs_history_length
    priv_obs_hist_len=priv_obs_history_length
    wm_obs_hist_len=wm_obs_history_length
    action_size=env.action_size

    # initialize model
    scale_fn = base.Scaler.transform
    print("obs_size, action_size", wm_obs_size, action_size)

    model_network = model_network_factory(obs_size=wm_obs_size, output_dim=model_output_dim)
    model_ensemble_size = model_network.ensemble_size
    model_num_elites = model_network.num_elites
    model_optimizer = optax.adam(model_learning_rate)

    make_model = wm_networks.make_inference_fn(ensemble_model=model_network, preprocess_fn=scale_fn,
                                                wm_obs_size=wm_obs_size,
                                                wm_obs_size_per_step=wm_obs_size_per_step,
                                                torque_dim=torque_dim,
                                                wm_noise_to_actor_noise_fn=robot_config.wm_noise_to_actor_noise,
                                                dynamics_fn=dynamics_fn,
                                                reward_fn=reward_fn,
                                                plot_model_rollouts=plot_model_rollouts,
                                                robot_config=robot_config,                                                )
    local_key = jax.random.PRNGKey(seed)
    loss_key, local_key = jax.random.split(local_key)
    model_loss = wm_losses.make_losses(
        loss_key, 
        model_network, 
        scale_fn, 
        obs_size=obs_size,
        wm_obs_size=wm_obs_size,
        wm_obs_size_per_step=wm_obs_size_per_step,
        torque_dim=torque_dim,
        wm_obs_hist_len=wm_obs_hist_len,
        low_level_control_fn=low_level_control_fn,
        dynamics_fn=dynamics_fn,
        model_probabilistic=True,
        model_training_weight_decay=model_training_weight_decay,
        model_training_stop_gradient=model_training_stop_gradient,
        mean_loss_over_horizon=mean_loss_over_horizon,
        robot_config=robot_config,
        )

    local_key, wm_network_key = jax.random.split(local_key)
    # if model is EnsembleModel, initialize with ensemble_size
    dummy_X = jp.zeros((model_network.ensemble_size,
                        wm_obs_size_per_step*wm_obs_hist_len+action_size))
    model_params = model_network.init(wm_network_key, dummy_X)
    model_optimizer_state = model_optimizer.init(model_params['params'])


    scaler_params = base.Scaler.init(wm_obs_size_per_step*wm_obs_hist_len, action_size)

    training_state = base.WM_TrainingState(
        model_optimizer_state=model_optimizer_state,
        model_params=model_params,
        scaler_params=scaler_params,
        env_steps=jp.zeros(())
    )

    if device_count > 1:
        model_update = gradients.gradient_update_fn(model_loss, model_optimizer,
                                                    pmap_axis_name=_PMAP_AXIS_NAME,
                                                    has_aux=True)
    else:
        model_update = gradients.gradient_update_fn(model_loss, model_optimizer,
                                                    pmap_axis_name=None,
                                                    has_aux=True)

    # create the model_env
    done_fn = lambda *args: jp.zeros(())  # noqa: E731
    if model_check_done_condition:
        assert hasattr(env, 'is_done'), (
            'The environment must have an is_done method to check if the '
            'episode is done. Otherwise, set model_check_done_condition=False')
        done_fn = env.is_done_in_wm if hasattr(env, 'is_done_in_wm') else env.is_done
    model_env = base.ModelEnv(done_fn, obs_size,
                              priv_obs_size,
                              wm_obs_size,
                              action_size)
    model_env = envs.training.wrap(
        model_env, episode_length=episode_length,
        action_repeat=action_repeat,
        obs_history_length=obs_history_length,
        priv_obs_history_length=priv_obs_history_length,
        wm_obs_history_length=wm_obs_history_length,
        )

    # make the env policy
    normalize_fn = lambda x, y: x  # noqa: E731
    if policy_normalize_observations:
        normalize_fn = running_statistics.normalize
    env_network = policy_network_factory(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=normalize_fn)

    make_policy_env = sac_networks.make_inference_fn(env_network)
    make_policy_env = functools.partial(
        make_policy_env,
        robot_config=robot_config,
    )
    # update sac_network_factory to with the hallucination_max_std
    if hallucination_max_std is None or hallucination_max_std > 0:
        policy_network_factory = functools.partial(
            policy_network_factory, policy_max_std=hallucination_max_std)

    # initialize buffers
    dummy_obs = {
        'state': jp.zeros((obs_size * obs_hist_len,)),
        'privileged_state': jp.zeros((priv_obs_size * priv_obs_hist_len,)),
        'wm_state': jp.zeros((wm_obs_size_per_step * wm_obs_hist_len))
    }

    dummy_action = jp.zeros((action_size,))
    dummy_transition = types.SACTransition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.,
        discount=0.,
        next_observation=dummy_obs,
        extras={
            'state_extras': {
                'truncation': 0.
            },
            'policy_extras': {
                'mean': jp.zeros((action_size,)),
                'std': jp.ones((action_size,)),
                'log_prob': 0.
            }
        }
    )

    network_obs_size = {
        'state': env.observation_size['state'] * obs_history_length,
        'privileged_state': env.observation_size['privileged_state'] * priv_obs_history_length,
        'wm_state': env.observation_size['wm_state'] * wm_obs_history_length,
    }

    env_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_env_buffer_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=model_rollouts_per_hallucination_update)
    local_key, eb_key = jax.random.split(local_key)
    env_buffer_state = jax.pmap(env_buffer.init)(
        jax.random.split(eb_key, local_devices_to_use)
    )
    # initialize SAC
    init_model_horizon = model_horizon_fn(0)
    init_hallucination_updates_per_training_step = (
        hallucination_updates_per_training_step_fn(0))
    if init_model_horizon > 0:
        sac_max_replay_size = (
            training_steps_per_model_train
            * init_hallucination_updates_per_training_step
            * model_rollouts_per_hallucination_update
            * init_model_horizon)
    else:
        sac_max_replay_size = max_env_buffer_size
    if max_model_buffer_size is not None:
        sac_max_replay_size = min(sac_max_replay_size, max_model_buffer_size)
    local_key, eb_key = jax.random.split(local_key)

    # SAC initialization
    sac_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=sac_max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=sac_batch_size * sac_grad_updates_per_hallucination_update)

    sac_buffer_state = jax.pmap(sac_buffer.init)(
        jax.random.split(eb_key, local_devices_to_use)
    )

    def update_model_horizon(sac_buffer: replay_buffers.UniformSamplingQueue, epoch: int, key: jp.ndarray, last_sac_buffer_state: Any, last_model_horizon: int):

        model_horizon = model_horizon_fn(epoch)
        hallucination_updates_per_training_step = (
            hallucination_updates_per_training_step_fn(epoch))
        sac_buffer_state = last_sac_buffer_state

        if (model_horizon != last_model_horizon and model_horizon != 0):
            key_rb, key = jax.random.split(key)
            if model_horizon > 0:
                sac_max_replay_size = (
                    training_steps_per_model_train
                    * hallucination_updates_per_training_step
                    * model_rollouts_per_hallucination_update
                    * model_horizon)
            else:
                sac_max_replay_size = max_env_buffer_size
            if max_model_buffer_size is not None:
                sac_max_replay_size = min(sac_max_replay_size,
                                        max_model_buffer_size)

            sac_buffer = replay_buffers.UniformSamplingQueue(
                max_replay_size=sac_max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=(
                    sac_batch_size
                    * sac_grad_updates_per_hallucination_update))
            sac_buffer_state = jax.pmap(sac_buffer.init)(
                jax.random.split(key_rb, local_devices_to_use)
            )
            if not clear_model_buffer_after_model_train:
                current_size = sac_buffer.size(last_sac_buffer_state)
                # use pmap to do a 'prefix copy' on-device to avoid host for-loops and invalid slicing
                def _copy_prefix(old_state, new_state, size, ins_pos, samp_pos):
                    # old/new capacity (static compile-time constants)
                    cap_old = old_state.data.shape[0]
                    cap_new = new_state.data.shape[0]

                    # static min of old/new capacity (for safe alignment)
                    min_cap = min(cap_old, cap_new)

                    # align old to cap_new:
                    # - first min_cap rows come from old
                    # - remaining rows are filled from new (keep shape consistent)
                    old_head = old_state.data[:min_cap]          # static slice, legal
                    new_tail = new_state.data[min_cap:]          # static slice, legal (may be empty)
                    old_aligned = jp.concatenate([old_head, new_tail], axis=0)  # [cap_new, feat...]

                    # dynamic copy length: copy at most min_cap
                    size = jp.minimum(size, min_cap)  # size: int32[] (per-device scalar)

                    # build mask using new capacity (same leading dim as new_state.data)
                    idx = jp.arange(cap_new, dtype=jp.int32)     # [cap_new]
                    mask = idx < size                             # True means use the first size rows from old_aligned
                    mask = mask.reshape((cap_new,) + (1,) * (new_state.data.ndim - 1))

                    # elementwise select: first size rows from old_aligned, rest from new_state.data
                    new_data = jp.where(mask, old_aligned, new_state.data)

                    return new_state.replace(
                        data=new_data,
                        insert_position=ins_pos,
                        sample_position=samp_pos,
                    )


                sac_buffer_state = jax.pmap(_copy_prefix)(
                    last_sac_buffer_state,
                    sac_buffer_state,
                    current_size,  # per-device scalar
                    last_sac_buffer_state.insert_position,
                    last_sac_buffer_state.sample_position,
                )
        return sac_buffer, sac_buffer_state, key, model_horizon, hallucination_updates_per_training_step

    normalize_fn = lambda x, y: x  # noqa: E731
    if policy_normalize_observations:
        normalize_fn = running_statistics.normalize
    sac_network = policy_network_factory(
        observation_size=network_obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn)
    make_policy = sac_networks.make_inference_fn(sac_network)
    make_policy = functools.partial(
        make_policy,
        robot_config=robot_config,
    )
    make_q_network = sac_networks.make_q_inference_fn(sac_network)

    alpha_optimizer = optax.adamw(learning_rate=3e-4)
    policy_optimizer = optax.adamw(learning_rate=sac_learning_rate)
    q_optimizer = optax.adamw(learning_rate=sac_learning_rate)

    alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=sac_reward_scaling,
        discounting=sac_discounting,
        action_size=action_size,
        robot_config=robot_config,
    )
    if device_count == 1:
        alpha_update = gradients.gradient_update_fn(
            alpha_loss, alpha_optimizer, pmap_axis_name=None)
        critic_update = gradients.gradient_update_fn(
            critic_loss, q_optimizer, pmap_axis_name=None)
        actor_update = gradients.gradient_update_fn(
            actor_loss, policy_optimizer, pmap_axis_name=None)
    else:
        alpha_update = gradients.gradient_update_fn(
            alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
        critic_update = gradients.gradient_update_fn(
            critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
        actor_update = gradients.gradient_update_fn(
            actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)

    key_policy, key_q = jax.random.split(key)
    log_alpha = jp.asarray(sac_init_log_alpha, dtype=jp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    obs_shape = jax.tree_util.tree_map(
        lambda arr: types.Array(arr.shape, arr.dtype),
        dummy_obs
        )
    normalizer_params = running_statistics.init_state(obs_shape)

    sac_training_state_init = base.SACTrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        original_policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=types.UInt64(hi=0, lo=0),
        env_steps=types.UInt64(hi=0, lo=0),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params
    )
    load_dtype = dummy_obs['state'].dtype
    # fine-tune from sac_training_state, if provided
    if sac_training_state is not None:
        sac_training_state_init = sac_training_state_init.replace(
            policy_params = jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), sac_training_state.policy_params),
            original_policy_params = jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), sac_training_state.policy_params),
            q_params = jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), sac_training_state.q_params),
            target_q_params = jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), sac_training_state.target_q_params),
            normalizer_params = jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), sac_training_state.normalizer_params),

        )
        if load_ac_optimizer_state:
            sac_training_state_init = sac_training_state_init.replace(
                policy_optimizer_state = jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), sac_training_state.policy_optimizer_state),
                q_optimizer_state = jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), sac_training_state.q_optimizer_state),

            )
        if load_alpha:
            sac_training_state_init = sac_training_state_init.replace(
                alpha_optimizer_state = jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), sac_training_state.alpha_optimizer_state),
                alpha_params = jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), sac_training_state.alpha_params),

            )
    else:
        print("Training SAC from scratch.")
    if wm_training_state is not None:
        training_state = training_state.replace(
            model_optimizer_state=jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), wm_training_state.model_optimizer_state),
            model_params=jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), wm_training_state.model_params),
            scaler_params=jax.tree_util.tree_map(lambda x: jp.array(x, dtype=load_dtype), wm_training_state.scaler_params),
        )

    training_state = jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
    )
    sac_training_state = jax.device_put_replicated(
            sac_training_state_init, jax.local_devices()[:local_devices_to_use]
        )

    if not eval_env:
        eval_env = env
    else:
        eval_env = envs.training.wrap(
            eval_env, episode_length=episode_length,
            action_repeat=action_repeat,
            obs_history_length=obs_history_length,
            priv_obs_history_length=priv_obs_history_length,
            wm_obs_history_length=wm_obs_history_length,
            )

    local_key, eval_key = jax.random.split(local_key)
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy_env,
                          deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key)

    model_horizon = -1
    hallucination_updates_per_training_step=(hallucination_updates_per_training_step_fn(0))
    make_policy_render = functools.partial(make_policy_env, deterministic=deterministic_eval)

    # Run initial eval
    metrics = {}

    if num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((sac_training_state.normalizer_params,
            sac_training_state.policy_params)),
            training_metrics={})

        # 4) Log + report
        logging.info(metrics)
        progress_fn(0, metrics)

        policy_params_fn(
            0,
            make_policy_render,
            _unpmap((sac_training_state.normalizer_params,
            sac_training_state.policy_params)),
            metrics,
            wm_training_state=_unpmap(training_state))

    # Prefill the env buffer.
    def prefill_env_buffer(
        training_state: base.WM_TrainingState,
        sac_ts: base.SACTrainingState,
        env_state: envs.State,
        env_buffer_state: sac_train.ReplayBufferState,
        model_buffer_state: sac_train.ReplayBufferState,
        key: types.PRNGKey,
        deterministic: bool
    ) -> Tuple[base.WM_TrainingState, base.SACTrainingState, envs.State,
               sac_train.ReplayBufferState, types.PRNGKey]:

        def f(carry: Tuple[base.WM_TrainingState, base.SACTrainingState,
                           envs.State, sac_train.ReplayBufferState, types.PRNGKey],
              unused):
            del unused
            (training_state, sac_ts, env_state, model_buffer_state,
             key) = carry
            key, new_key = jax.random.split(key)
            (new_normalizer_params, env_state, model_buffer_state,
             transition) = get_experience_brax(
                sac_ts.normalizer_params,
                sac_ts.policy_params,
                make_policy_env,
                env_state, model_buffer_state, key, env,
                sac_buffer,
                deterministic)
            new_training_state = training_state.replace(
                env_steps=(training_state.env_steps
                           + action_repeat*num_envs))
            new_sac_ts = sac_ts.replace(
                normalizer_params=new_normalizer_params)
            return (new_training_state, new_sac_ts, env_state,
                    model_buffer_state, new_key), transition

        (training_state, sac_ts, env_state, model_buffer_state,
         key), transitions = jax.lax.scan(
            f,
            (training_state, sac_ts, env_state, model_buffer_state, key),
            (), length=init_exploration_steps)

        # we insert the transitions into the env buffer after the scan finishes
        # to ensure that they are inserted in order
        env_buffer_state = env_buffer.insert(env_buffer_state, transitions)

        return (training_state, sac_ts, env_state, env_buffer_state,
                model_buffer_state, key)
    prefill_env_buffer = jax.pmap(prefill_env_buffer, static_broadcasted_argnums=(6))

    def policy_update(
            buffer_state: sac_train.ReplayBufferState,
            training_state: base.SACTrainingState,
            training_key: types.PRNGKey,
            external_buffer_state: sac_train.ReplayBufferState = {},
            external_buffer_ratio: float = 0.0,
            target_entropy: float = 0.0,
    ) -> Tuple[base.SACTrainingState, sac_train.ReplayBufferState, types.Metrics, sac_train.ReplayBufferState, jp.ndarray, optax.OptState]:
        """Update the policy by doing grad_updates_per_step sgd_steps. If an
        external buffer is provided, external_buffer_ratio*batch_size samples are
        mixed into the batch."""

        def sgd_step(
            carry: Tuple[base.SACTrainingState, types.PRNGKey, jp.ndarray, optax.OptState],
            transitions: types.SACTransition
        ) -> Tuple[Tuple[base.SACTrainingState, types.PRNGKey, jp.ndarray, optax.OptState], types.Metrics]:
            training_state, key = carry

            key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

            alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
                training_state.alpha_params,
                training_state.policy_params,
                training_state.normalizer_params,
                transitions,
                target_entropy,
                key_alpha,
                optimizer_state=training_state.alpha_optimizer_state)
            alpha = jp.exp(training_state.alpha_params)
            if sac_fixed_alpha:
                alpha = sac_fixed_alpha
                alpha_params = jp.log(alpha).astype(jp.float32)
            critic_loss, q_params, q_optimizer_state = critic_update(
                training_state.q_params,
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.target_q_params,
                alpha,
                transitions,
                key_critic,
                optimizer_state=training_state.q_optimizer_state)

            actor_loss, policy_params, policy_optimizer_state = actor_update(
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.q_params,
                alpha,
                transitions,
                key_actor,
                optimizer_state=training_state.policy_optimizer_state)

            new_target_q_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - sac_tau) + y * sac_tau,
                training_state.target_q_params, q_params)

            dist_params = sac_network.policy_network.apply(
                training_state.normalizer_params, policy_params, transitions.observation
            )            
            action_mean = sac_network.parametric_action_distribution.mode(
                dist_params
            )
            action_std = sac_network.parametric_action_distribution.stddev(
                dist_params
            )
            dist_params_original = sac_network.policy_network.apply(
                training_state.normalizer_params, training_state.original_policy_params, transitions.observation
            )
            action_mean_original = sac_network.parametric_action_distribution.mode(
                dist_params_original
            )
            action_std_original = sac_network.parametric_action_distribution.stddev(
                dist_params_original
            )
            kl = jp.sum(
                jp.log(action_std/action_std_original)
                + 0.5 * (jp.square(action_std_original) + jp.square(action_mean - action_mean_original)) / jp.square(action_std)
                - 0.5,
                axis=-1
            )
            kl_mean = jp.mean(kl)
            if device_count > 1:
                kl_mean = jax.lax.pmean(kl_mean, axis_name=_PMAP_AXIS_NAME)

            metrics = {
                'critic_loss': critic_loss,
                'actor_loss': actor_loss,
                'kl_loss': kl_mean,
                'alpha_loss': alpha_loss,
                'alpha': jp.exp(alpha_params),
            }


            new_training_state = base.SACTrainingState(
                policy_optimizer_state=policy_optimizer_state,
                policy_params=policy_params,
                original_policy_params=training_state.original_policy_params,
                q_optimizer_state=q_optimizer_state,
                q_params=q_params,
                target_q_params=new_target_q_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
                alpha_optimizer_state=alpha_optimizer_state,
                alpha_params=alpha_params,
                normalizer_params=training_state.normalizer_params)
            return (new_training_state, key), metrics

        # sample buffer
        buffer_state, transitions = sac_buffer.sample(buffer_state)

        if env_buffer is not None:
            # sample external buffer
            batch_size = transitions.observation['state'].shape[0]
            ext_size = int(batch_size * external_buffer_ratio
                        // sac_grad_updates_per_hallucination_update * sac_grad_updates_per_hallucination_update)
            external_buffer_state, external_transitions = env_buffer.sample(
                external_buffer_state)
            external_transitions = jax.tree_util.tree_map(
                lambda x: x[:ext_size],
                external_transitions)
            transitions = jax.tree_util.tree_map(
                lambda x, y: jp.concatenate([x, y], axis=0),
                transitions, external_transitions)

            # shuffle external transitions into the batch
            training_key, key_shuf = jax.random.split(training_key)
            permuted_idxs = jax.random.permutation(key_shuf, batch_size + ext_size)
            transitions = jax.tree_map(lambda x: x[permuted_idxs], transitions)

        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        transitions = jax.tree_util.tree_map(
            lambda x: jp.reshape(x, (sac_grad_updates_per_hallucination_update, -1) + x.shape[1:]),
            transitions)

        # take sgd steps
        (training_state, _), metrics = jax.lax.scan(
            sgd_step,
            (training_state, training_key),
            transitions,
        )
        q_network = make_q_network((training_state.normalizer_params,
                                    training_state.q_params))


        q_value = jp.mean(q_network(transitions.observation, transitions.action))
        metrics['q_value'] = q_value
        metrics['buffer_current_size'] = sac_buffer.size(buffer_state)

        return training_state, buffer_state, metrics, external_buffer_state

    def policy_update_sac(
        training_state: base.SACTrainingState,
        sac_training_state: base.SACTrainingState,
        env_buffer_state: sac_train.ReplayBufferState,
        sac_buffer_state: sac_train.ReplayBufferState,
        key: types.PRNGKey,
        model_horizon: int,
        hallucination_updates_per_training_step: int,
        target_entropy: float,
    ) -> Tuple[base.SACTrainingState, base.SACTrainingState, types.Metrics, envs.State, jp.ndarray, optax.OptState]:

        def update(carry, unused_t):
            training_state, sac_training_state, sac_buffer_state, env_buffer_state, key = carry
            if model_horizon > 0:
                # sample model_rollouts_per_hallucination_update samples from env
                # buffer
                env_buffer_state, transitions = env_buffer.sample(env_buffer_state)

                # we do the following to get the first state of each trajectory from
                # the sampled transitions (in place of calling an env.reset function;
                # trajectories that are done will reset to their sampled state)
                obs_stack = transitions.observation
                keys = jax.random.split(key, obs_stack['state'].shape[0])

                def init_starting_state(obs_stack, key):
                    return envs.State(
                        pipeline_state=None,
                        obs=obs_stack,
                        reward=jp.zeros(()),
                        done=jp.zeros(()),
                        info={
                            'reward': jp.zeros(()),
                            'next_obs': obs_stack,
                            'first_pipeline_state': None,
                            'first_obs': obs_stack,
                            'first_metrics': {},
                            'truncation': jp.zeros(()),
                            'steps': jp.zeros(()),
                            'episode_done': jp.zeros(()),  
                            'episode_metrics': {          
                                'sum_reward': jp.zeros(()),
                                'length': jp.zeros(()),

                            },
                            'first_rew_info': {
                                'last_last_actions': jp.zeros(transitions.action.shape[1]),
                                'last_actions': jp.zeros(transitions.action.shape[1]),
                                'actions': jp.zeros(transitions.action.shape[1]),
                                'rigid_state_pos': jp.zeros((env.sys.qd_size()-6+1, 3)),
                                'rigid_state_lin_vel': jp.zeros((env.sys.qd_size()-6+1, 3)),
                                'rigid_state_ang_vel': jp.zeros((env.sys.qd_size()-6+1, 3)),
                                'rigid_state_rot': jp.zeros((env.sys.qd_size()-6+1, 4)),
                                'rigid_state_qdd':  jp.zeros((env.sys.qd_size(),)),
                                'ref_step': jp.zeros(()),
                                'ref_motion_init_step': jp.zeros(()),
                            },
                        },
                        rew_info = {
                                'last_last_actions': jp.zeros(transitions.action.shape[1]),
                                'last_actions': jp.zeros(transitions.action.shape[1]),
                                'actions': jp.zeros(transitions.action.shape[1]),
                                'rigid_state_pos': jp.zeros((env.sys.qd_size()-6+1, 3)),
                                'rigid_state_lin_vel': jp.zeros((env.sys.qd_size()-6+1, 3)),
                                'rigid_state_ang_vel': jp.zeros((env.sys.qd_size()-6+1, 3)),
                                'rigid_state_rot': jp.zeros((env.sys.qd_size()-6+1, 4)),
                                'rigid_state_qdd':  jp.zeros((env.sys.qd_size(),)),
                                'ref_step': jp.zeros(()),
                                'ref_motion_init_step': jp.zeros(()),

                            },
                        prev_obs=obs_stack,
                        torque=jp.zeros(transitions.action.shape[1]),
                    )
                env_states = jax.vmap(init_starting_state)(obs_stack, keys)
                # perform model_horizon step model rollouts from samples and add to
                # model buffer (the model buffer is the sac buffer)
                def f(carry, unused_t):
                    env_state, model_buffer_state, key = carry
                    key, new_key = jax.random.split(key)
                    env_state, model_buffer_state, transitions, sub_rewards = get_experience_model(
                        normalizer_params=sac_training_state.normalizer_params,
                        policy_params=sac_training_state.policy_params,
                        make_policy=make_policy,
                        model_scaler_params=training_state.scaler_params,
                        model_params=training_state.model_params,
                        make_model=make_model,
                        model_env=model_env, 
                        env_state=env_state, 
                        model_buffer_state=model_buffer_state,
                        model_replay_buffer=sac_buffer, 
                        policy_std=None,
                        output_info=True,
                        key=key)
                    return (env_state, model_buffer_state, new_key), (transitions, sub_rewards)


                (_, sac_buffer_state, key), (transitions, sub_rewards) = jax.lax.scan(
                    f, (env_states, sac_buffer_state, key), (),
                    length=model_horizon)

            # update policy using sac
            (sac_training_state, sac_buffer_state, metrics,
            env_buffer_state) = policy_update(
                sac_buffer_state, sac_training_state, key,
                env_buffer_state, real_ratio, target_entropy)

            if model_horizon > 0:
                returns = transitions.reward
                returns_sum = jp.sum(returns, axis=0)
                returns_mean = jp.mean(returns, axis=0)

                metrics.update({"policy/returns_in_model_sum": returns_sum})
                metrics.update({"policy/returns_in_model_mean": returns_mean})
                if plot_model_rollouts:
                    wm_state_seq = transitions.observation['wm_state'].reshape(
                        transitions.observation['wm_state'].shape[:-1] + (1, wm_obs_size_per_step)
                    )
                    single_nonorm_obs = robot_config.denormalize_state(wm_state_seq[:, :, -1, :robot_config.wm_observation_size].copy(), robot_config.wm_state_limits)
                    # protect not has this element in the object
                    if hasattr(robot_config, 'wm_gravity'):
                        for i in range(3):
                            metrics['generate_obs_mean/'+'gravity_'+str(i)] = jp.mean(single_nonorm_obs[:, :, robot_config.wm_gravity][:, :, i])
                            metrics['generate_obs_max/'+'gravity_'+str(i)] = jp.max(single_nonorm_obs[:, :, robot_config.wm_gravity][:, :, i])
                            metrics['generate_obs_min/'+'gravity_'+str(i)] = jp.min(single_nonorm_obs[:, :, robot_config.wm_gravity][:, :, i])

                    if hasattr(robot_config, 'wm_q_idxs'):
                        for i in range(env.action_size):
                            metrics['generate_obs_mean/'+'mean_dof_pos_'+str(i)] = jp.mean(single_nonorm_obs[:, :, robot_config.wm_q_idxs][:, :, i])
                            metrics['generate_obs_max/'+'max_dof_pos_'+str(i)] = jp.max(single_nonorm_obs[:, :, robot_config.wm_q_idxs][:, :, i])
                            metrics['generate_obs_min/'+'min_dof_pos_'+str(i)] = jp.min(single_nonorm_obs[:, :, robot_config.wm_q_idxs][:, :, i])

                    if hasattr(robot_config, 'wm_qd_idxs'):
                        for i in range(env.action_size):
                            metrics['generate_obs_mean/'+'mean_dof_vel'+str(i)] = jp.mean(single_nonorm_obs[:, :, robot_config.wm_qd_idxs][:, :, i])
                            metrics['generate_obs_max/'+'max_dof_vel'+str(i)] = jp.max(single_nonorm_obs[:, :, robot_config.wm_qd_idxs][:, :, i])
                            metrics['generate_obs_min/'+'min_dof_vel'+str(i)] = jp.min(single_nonorm_obs[:, :, robot_config.wm_qd_idxs][:, :, i])
                    if hasattr(robot_config, 'wm_base_lin_vel_idxs'):
                        for i in range(3):
                            metrics['generate_obs_mean/'+'mean_lin_vel_'+str(i)] = jp.mean(single_nonorm_obs[:, :, robot_config.wm_base_lin_vel_idxs][:, :, i])
                            metrics['generate_obs_max/'+'mean_lin_vel_'+str(i)] = jp.max(single_nonorm_obs[:, :, robot_config.wm_base_lin_vel_idxs][:, :, i])
                            metrics['generate_obs_min/'+'mean_lin_vel_'+str(i)] = jp.min(single_nonorm_obs[:, :, robot_config.wm_base_lin_vel_idxs][:, :, i])

                    if hasattr(robot_config, 'wm_base_ang_vel_idxs'):
                        for i in range(3):
                            metrics['generate_obs_mean/'+'mean_ang_vel_'+str(i)] = jp.mean(single_nonorm_obs[:, :, robot_config.wm_base_ang_vel_idxs][:, :, i])
                            metrics['generate_obs_max/'+'max_ang_vel_'+str(i)] = jp.max(single_nonorm_obs[:, :, robot_config.wm_base_ang_vel_idxs][:, :, i])
                            metrics['generate_obs_min/'+'min_ang_vel_'+str(i)] = jp.min(single_nonorm_obs[:, :, robot_config.wm_base_ang_vel_idxs][:, :, i])

                    if hasattr(robot_config, 'wm_quat_idxs'):
                        for i in range(4):
                            metrics['generate_obs_min/'+'min_quat_'+str(i)] = jp.min(single_nonorm_obs[:, :, robot_config.wm_quat_idxs][:, :, i])
                            metrics['generate_obs_max/'+'max_quat_'+str(i)] = jp.max(single_nonorm_obs[:, :, robot_config.wm_quat_idxs][:, :, i])
                            metrics['generate_obs_mean/'+'mean_quat_'+str(i)] = jp.mean(single_nonorm_obs[:, :, robot_config.wm_quat_idxs][:, :, i])

                    if hasattr(robot_config, 'wm_h_idx'):
                        metrics['generate_obs_mean/'+'mean_body_height'] = jp.mean(single_nonorm_obs[:, :, robot_config.wm_h_idx])
                        metrics['generate_obs_max/'+'max_body_height'] = jp.max(single_nonorm_obs[:, :, robot_config.wm_h_idx])
                        metrics['generate_obs_min/'+'min_body_height'] = jp.min(single_nonorm_obs[:, :, robot_config.wm_h_idx])

                    for i in range(env.sys.qd_size()):
                        metrics['generate_model/'+'model_output_mean_'+str(i)] = jp.mean(sub_rewards['model_output_mean'][:, :, i])

                    for i in range(sub_rewards['model_output_std'].shape[-1]):
                        metrics['generate_model/'+'model_output_std_'+str(i)] = jp.mean(sub_rewards['model_output_std'][:, :, i])


                    for i in range(sub_rewards['action_mean'].shape[-1]):
                        metrics['generate_model/'+'action'+str(i)+'_mean'] = jp.mean(transitions.extras["policy_extras"]['mean'][:, :, i])
                        metrics['generate_model/'+'action'+str(i)+'_std'] = jp.mean(transitions.extras["policy_extras"]['std'][:, :, i])
                    for i in sub_rewards.keys():
                        if "sub_" in i:
                            metrics['generate_reward_mean/'+i] = jp.mean(sub_rewards[i])

                    metrics['generate_model/'+'model_output_std_totalmean'] = jp.mean(sub_rewards['model_output_std'])


            return (training_state, sac_training_state, sac_buffer_state, env_buffer_state,
                    key), metrics

        (training_state, sac_training_state, sac_buffer_state, env_buffer_state,
        _), metrics = jax.lax.scan(
            update,
            (training_state, sac_training_state, sac_buffer_state, env_buffer_state, key),
            (), length=hallucination_updates_per_training_step)

        metrics = jax.tree_util.tree_map(jp.mean, metrics)

        return (training_state, sac_training_state, env_buffer_state,
                sac_buffer_state, metrics)
    def sim_training_step(
        training_state: base.WM_TrainingState,
        sac_training_state: base.SACTrainingState,
        env_state: envs.State,
        env_buffer_state: sac_train.ReplayBufferState,
        key: types.PRNGKey,
        sac_buffer_state: sac_train.ReplayBufferState,
        model_horizon: int,
        hallucination_updates_per_training_step: int,
        target_entropy: float,
    ) -> Tuple[base.WM_TrainingState, base.SACTrainingState, envs.State,
            sac_train.ReplayBufferState, sac_train.ReplayBufferState, types.Metrics]:

        # get experience
        def f(carry, unused_t):
            (ts, normalizer_params, env_state, sac_buffer_state,
            key) = carry
            key, new_key = jax.random.split(key)
            (normalizer_params, env_state, sac_buffer_state,
            transition) = get_experience_brax(
                normalizer_params, sac_training_state.policy_params,
                make_policy_env, env_state, sac_buffer_state,
                key, env, sac_buffer, deterministic_in_env)
            new_ts = ts.replace(
                env_steps=(ts.env_steps + action_repeat*num_envs))
            return (new_ts, normalizer_params, env_state,
                    sac_buffer_state, new_key), transition

        (training_state, normalizer_params, env_state,
        sac_buffer_state, key), transitions = jax.lax.scan(
            f,
            (training_state, sac_training_state.normalizer_params, env_state,
            sac_buffer_state, key), (),
            length=env_steps_per_training_step)

        env_buffer_state = env_buffer.insert(env_buffer_state, transitions)

        sac_training_state = sac_training_state.replace(
            normalizer_params=normalizer_params)
        sac_training_state = sac_training_state.replace(
            original_policy_params=sac_training_state.policy_params)
        # policy update
        (training_state, sac_training_state, env_buffer_state, sac_buffer_state,
        metrics) = policy_update_sac(
            training_state, sac_training_state,
            env_buffer_state, sac_buffer_state,
            key, model_horizon, hallucination_updates_per_training_step, target_entropy)


        return (training_state, sac_training_state, env_state, env_buffer_state,
                sac_buffer_state, metrics)
    sim_training_step = jax.pmap(sim_training_step, axis_name='i', static_broadcasted_argnums=(6,7,8))
    def prepare_data(transitions, test_transitions, key):
        E = model_network.ensemble_size
        H = model_loss_horizon
        B = model_training_batch_size
        max_train_batches = model_max_train_batches
        max_test_batches = model_max_test_batches

        # 1) add ensemble and horizon dims in one step:
        # orig: [N, ...] -> target intermediate shape: [E, N, H=1, ...]
        N = transitions.observation['privileged_state'].shape[0]
        def add_E_N_H(x):
            if isinstance(x, jp.ndarray):
                x = x[None, :, :, ...]                       # [1, N, 1, ...]
                return jp.broadcast_to(x, (E, N, H) + x.shape[3:])
            return x
        tr = jax.tree_map(add_E_N_H, transitions)               # [E, N, H, ...]


        N_test = test_transitions.observation['privileged_state'].shape[0]
        def add_E_N_H(x):
            if isinstance(x, jp.ndarray):
                x = x[None, :, :, ...]                       # [1, N, 1, ...]
                return jp.broadcast_to(x, (E, N_test, H) + x.shape[3:])
            return x
        tr_test = jax.tree_map(add_E_N_H, test_transitions)               # [E, N, H, ...]

        # 2) shuffle N axis for each ensemble (guarantee each model sees different order)
        # keys = jax.random.split(key, E)
        # def shuffle_one_e(arr, k):
        #     # arr: [N, H, ...] shuffle only along N axis
        #     perm = jax.random.permutation(k, arr.shape[0])
        #     return arr[perm, ...]
        # tr = jax.tree_map(lambda x: jax.vmap(shuffle_one_e, in_axes=(0,0))(x, keys), tr)  # still [E, N, H, ...]
        # tr_test = jax.tree_map(lambda x: jax.vmap(shuffle_one_e, in_axes=(0,0))(x, keys), tr_test)  # still [E, N, H, ...]

        # 3) form batches along N axis (put N first for reshape convenience)
        #    [E, N, H, ...] -> [N, E, H, ...] -> [num_batches, B, E, H, ...]
        num_batches = N // B
        def to_batches(x):
            x = jp.swapaxes(x, 0, 1)                           # [N, E, H, ...]
            x = x[:num_batches * B, ...]                        # discard tail, or can be padded
            x = x.reshape((num_batches, B) + x.shape[1:])       # [num_batches, B, E, H, ...]
            # need to swap axes once to get [num_batches, B, H, E, ...]
            return jp.swapaxes(x, 2, 3)                        # -> [num_batches, B, H, E, ...]
        tr = jax.tree_map(to_batches, tr)
        tr_test = jax.tree_map(to_batches, tr_test)

        # 4) pad to max_batches (pad front dimensions, don't pad back dimensions)
        pad_train_batches = max_train_batches - num_batches
        pad_test_batches = max_test_batches - num_batches

        # 4) pad to max_batches (pad front dimensions, don't pad back dimensions)
        pad_train_batches = max_train_batches - num_batches
        def pad_to_max(x):
            pad_cfg = ((0, max(0, pad_train_batches)), (0,0), (0,0), (0,0)) + tuple((0,0) for _ in x.shape[4:])
            return jp.pad(x, pad_cfg)
        tr = jax.tree_map(pad_to_max, tr)

        pad_test_batches = max_test_batches - num_batches
        def pad_to_max(x):
            pad_cfg = ((0, max(0, pad_test_batches)), (0,0), (0,0), (0,0)) + tuple((0,0) for _ in x.shape[4:])
            return jp.pad(x, pad_cfg)
        tr_test = jax.tree_map(pad_to_max, tr_test)
        return tr, tr_test
    prepare_data = jax.pmap(prepare_data)


    def model_training_epoch_jit(
        params: types.Params,
        model_optimizer_state: optax.OptState,
        scaler_params: base.ScalerParams,
        transitions: types.SACTransition,
        test_transitions: types.SACTransition,
        num_train_batches: int,
        num_test_batches: int
    ):
        def sgd_step_wm(carry, in_element):
            params, opt_state = carry
            transitions, i = in_element
            model_params = params.pop('params')
            (loss, aux), new_model_params, opt_state = jax.lax.cond(
                i < num_train_batches,
                lambda: model_update(
                    model_params,
                    params,
                    scaler_params,
                    transitions.observation['wm_state'],
                    transitions.next_observation['wm_state'],
                    transitions.action,
                    transitions.discount,
                    optimizer_state=opt_state
                ),
                lambda: ((0., (jp.zeros(model_ensemble_size), jp.zeros(wm_obs_size))),
                        model_params, opt_state),
            )
            loss_ensemble, loss_all_axiss = aux
            return (({'params': new_model_params, **params}, opt_state),
                    (loss, loss_ensemble, loss_all_axiss))

        # perform an sgd step for each batch
        ((new_params, new_opt_state),
        (train_total_losses, mean_losses, train_mean_losses_all_axiss)) = jax.lax.scan(
            sgd_step_wm, (params, model_optimizer_state),
            (transitions, jp.arange(transitions.observation['wm_state'].shape[0])))

        train_mean_losses = jp.mean(mean_losses, axis=-1)

        # compute lossses on the test set
        new_model_params = new_params.pop('params')

        def test(_, in_element):
            transitions, i = in_element
            test_total_loss, aux = jax.lax.cond(
                i < num_test_batches,
                lambda: model_loss(
                    new_model_params, new_params, scaler_params,
                    transitions.observation['wm_state'], transitions.next_observation['wm_state'],
                    transitions.action, transitions.discount),
                lambda: (0., (jp.zeros(model_ensemble_size), jp.zeros(wm_obs_size))),
            )
            test_mean_loss, test_mean_loss_all_axiss = aux
            return None, (test_total_loss, test_mean_loss, test_mean_loss_all_axiss)

        _, (test_total_losses, test_mean_losses, test_mean_losses_all_axiss) = jax.lax.scan(
            test, None,
            (test_transitions, jp.arange(test_transitions.observation['wm_state'].shape[0])))
        new_params = {'params': new_model_params, **new_params}
        return (new_params, new_opt_state, train_total_losses, train_mean_losses, train_mean_losses_all_axiss,
                test_total_losses, test_mean_losses, test_mean_losses_all_axiss)

    def model_training_epoch(
        params: types.Params,
        model_optimizer_state: optax.OptState,
        scaler_params: base.ScalerParams,
        transitions: types.SACTransition,
        test_transitions: types.SACTransition,
        num_train_batches: int,
        num_test_batches: int,
        key: types.PRNGKey
    ):
        (new_params, new_opt_state, train_total_losses, train_mean_losses, train_mean_losses_all_axiss,
        test_total_losses, test_mean_losses, test_mean_losses_all_axiss) = model_training_epoch_jit(
            params, model_optimizer_state, scaler_params, transitions,
            test_transitions, num_train_batches, num_test_batches)

        denom_test = jp.maximum(num_test_batches, 1)
        mean_test_total_loss = jp.sum(test_total_losses, axis=0) / denom_test
        mean_test_mean_loss = jp.sum(test_mean_losses, axis=0) / denom_test

        return (new_params, new_opt_state,
                train_total_losses[num_train_batches-1],
                train_mean_losses[num_train_batches-1],
                mean_test_total_loss,
                mean_test_mean_loss,
                jp.mean(train_mean_losses_all_axiss, axis=0), # mean loss of all batches
                jp.max(train_mean_losses_all_axiss, axis=0), # max loss of all batches
                jp.mean(test_mean_losses_all_axiss, axis=0), # mean loss of last batches
                jp.max(test_mean_losses_all_axiss, axis=0), # max loss of last batches
                )
    # pmap model_training_epoch
    model_training_epoch = jax.pmap(model_training_epoch, axis_name='i')

    def process_data(
        training_state: base.WM_TrainingState,
        env_buffer_state: sac_train.ReplayBufferState,
        key: types.PRNGKey
    ):
        capacity = env_buffer_state.data.shape[0]
        buf_size = env_buffer.size(env_buffer_state) - model_loss_horizon
        key_shuffle, key = jax.random.split(key)

        # the ramdom id of capacity samples
        perm = jax.random.permutation(key_shuffle, capacity)

        # the valid samples in perm
        perm_is_valid = (perm < buf_size).astype(jp.int32)

        # the valid pos in perm is a new value in valid_pos_in_perm
        valid_pos_in_perm = jp.cumsum(perm_is_valid) - 1

        # the valid rank in perm is a new value in valid_rank_in_perm, if the sample is valid, the rank is the valid pos, 
        # otherwise, is the capacity
        valid_rank_in_perm = jp.where(perm_is_valid == 1, valid_pos_in_perm,
                                    jp.full_like(valid_pos_in_perm, capacity))                                                      # [capacity]

        train_length, test_length, num_train_batches, num_test_batches = _calculate_dataset_lengths_jax(
            buf_size, model_training_batch_size, model_training_test_ratio
        )
        if model_training_max_sgd_steps_per_epoch is not None:
            num_train_batches = jp.minimum(
                num_train_batches,
                jp.array(model_training_max_sgd_steps_per_epoch, dtype=num_train_batches.dtype),
            )

        valid_rank = jp.full_like(valid_rank_in_perm, capacity)
        valid_rank = valid_rank.at[perm].set(valid_rank_in_perm)               # [capacity]
        # the index of the valid_rank represent the index of the sample in the random sequence
        # the value of the valid_rand is the order of the valid sample in the random sequence
        # if the sample is not valid, the value is the capacity
        # so the valid_rank is a new value in the valid_rank, the value is the order of the valid sample in the random sequence

        # devide the valid data into the train and test set
        train_mask = valid_rank < train_length
        test_mask  = (valid_rank >= train_length) & (valid_rank < train_length + test_length)

        # combine the train mask and the valid_rank to get the data in the original sequence
        def compress_by_mask(arr, mask, rank_like):
            cap = arr.observation['wm_state'].shape[0]

            # ~mask make the invalid sample (train or test) to the capacity, 
            # and make the valid sample become small
            # rank_like make the valid sample become small (exceed or less than the capacity)
            # and the earliest valid sample in the perm is small in the valid_rand. 
            sort_key = (~mask).astype(jp.int32) * cap + rank_like

            # get the order of the valid sample (train mask + capacity + random selection) in the original sequence
            order = jp.argsort(sort_key)
            return _gather_dataset(arr, order, model_loss_horizon)

        data = env_buffer_state.data
        all_data = env_buffer._unflatten_fn(data)
        transitions = compress_by_mask(all_data, train_mask, valid_rank)
        test_transitions  = compress_by_mask(all_data, test_mask,  valid_rank)

        # transitions is [batch, horizon, dim]
        train_obs = transitions.observation['wm_state'][:, 0, :]
        # reshape to batch, substeps, single wm_state dimension

        train_obs = train_obs.reshape((train_obs.shape[0], wm_obs_size_per_step))
        assert robot_config is not None
        train_act = transitions.action[:, 0, :] / robot_config.policy_output_scale


        weights = train_mask.astype(jp.float32)
        scaler_params = base.Scaler.fit_multi_device(train_obs, train_act,
                                                axis_name='i', weights=weights)
        training_state = training_state.replace(scaler_params=scaler_params)
        return training_state, num_train_batches, num_test_batches, transitions, test_transitions, train_length, test_length, train_mask, test_mask, perm, train_length, test_length

    process_data = jax.pmap(process_data, axis_name='i')

    def train_model(
        transitions: types.SACTransition,
        test_transitions: types.SACTransition,
        training_state: base.WM_TrainingState,
        num_train_batches: int,
        num_test_batches: int,
        key: types.PRNGKey
    ):
        model_params = training_state.model_params
        opt_state = training_state.model_optimizer_state
        best = 10e9 * jp.ones(model_ensemble_size)
        epochs_since_update = 0
        t = time.time()

        for epoch in range(model_training_max_epochs):
            key, epoch_key = jax.random.split(key)
            # epoch key to N devices
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            transitions_reshape, test_transitions_reshape = prepare_data(transitions, test_transitions, epoch_keys)
            key, epoch_key = jax.random.split(key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (model_params, opt_state, train_total_loss, train_mean_loss, test_total_loss, test_mean_loss, train_mean_losses_all_axiss, train_max_mean_losses_all_axiss,
                test_mean_losses_all_axiss, test_max_mean_losses_all_axiss) = model_training_epoch(
                model_params, opt_state, training_state.scaler_params, transitions_reshape,
                test_transitions_reshape, num_train_batches, num_test_batches, epoch_keys)
            train_total_loss = jp.mean(train_total_loss, axis=0)
            train_mean_loss = jp.mean(train_mean_loss, axis=0)
            test_total_loss = jp.mean(test_total_loss, axis=0)
            test_mean_loss = jp.mean(test_mean_loss, axis=0)
            train_mean_losses_all_axiss = jp.mean(train_mean_losses_all_axiss, axis=0)
            train_max_mean_losses_all_axiss = jp.mean(train_max_mean_losses_all_axiss, axis=0)
            test_mean_losses_all_axiss = jp.mean(test_mean_losses_all_axiss, axis=0)
            test_max_mean_losses_all_axiss = jp.mean(test_max_mean_losses_all_axiss, axis=0)

            print(f'Model epoch {epoch}: train total loss {train_total_loss}, '
                f'train mean loss {train_mean_loss}, '
                f'test mean loss {test_mean_loss}')

            # check absolute criteria
            if (model_training_abs_criteria is not None
                    and jp.sum(test_mean_loss < model_training_abs_criteria)
                    >= model_num_elites):
                break

            # check convergence criteria
            improvement = (best - test_mean_loss) / best
            best = jp.where(improvement > model_training_convergence_criteria,
                            test_mean_loss, best)
            if jp.any(improvement > model_training_convergence_criteria):
                epochs_since_update = 0
            else:
                epochs_since_update += 1
            if epochs_since_update >= model_training_consec_converged_epochs:
                break
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), model_params)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), opt_state)

        sec_per_epoch = (time.time() - t) / (epoch + 1)

        elite_idxs = jp.argsort(test_mean_loss)
        elite_idxs = elite_idxs[:model_num_elites]
        # convert to float32
        elite_idxs = elite_idxs.astype(jp.int32)
        # elite_idxs not trainable
        elite_idxs = jax.lax.stop_gradient(elite_idxs)

        # to N devices array
        elite_idxs = jax.device_put_replicated(elite_idxs, jax.local_devices()[:local_devices_to_use])   
        model_params = {
            'params': {
                **model_params['params'],
            },
            'elites': {'idxs': elite_idxs},
        }

        test_mean_loss = jp.mean(test_mean_loss)

        training_state = training_state.replace(
            model_params=model_params, model_optimizer_state=opt_state)

        metrics = {
                'train_total_loss': train_total_loss,
                'train_mean_loss': train_mean_loss,
                'test_total_loss': test_total_loss,
                'test_mean_loss': test_mean_loss,
                'train_epochs': epoch + 1,
                'sec_per_epoch': sec_per_epoch}
        single_step_mean_loss = 0.0
        for i in range(train_mean_losses_all_axiss.shape[-1]):
            single_step_mean_loss += train_mean_losses_all_axiss[i]
            # log index % env.wm_state_size
            metrics['train_mean_losses_axis_' + str(i % wm_obs_size_per_step)+ '_' + str(i // wm_obs_size_per_step)] = train_mean_losses_all_axiss[i]
            metrics['test_mean_losses_axis_' + str(i % wm_obs_size_per_step)+ '_' + str(i // wm_obs_size_per_step)] = test_mean_losses_all_axiss[i]
            if (i + 1) % wm_obs_size_per_step == 0:
                metrics['train_mean_loss_single_step'+'_' + str(i // wm_obs_size_per_step)] = single_step_mean_loss / wm_obs_size_per_step
                metrics['test_mean_loss_single_step'+'_' + str(i // wm_obs_size_per_step)] = single_step_mean_loss / wm_obs_size_per_step
                single_step_mean_loss = 0.0

        return training_state, metrics

    def sim_training_epoch_with_timing(
        training_state: base.WM_TrainingState,
        sac_training_state: base.SACTrainingState,
        sac_buffer_state: sac_train.ReplayBufferState,
        env_buffer_state: sac_train.ReplayBufferState,
        training_walltime: float,
        env_state: envs.State,
        model_horizon: int,
        hallucination_updates_per_training_step: int,
        key: types.PRNGKey,
        target_entropy: float=0.0,
    ):
        t = time.time()
        model_train_time = 0
        other_time = 0
        model_metrics = {}
        for _ in range(model_trains_per_epoch):
            # train model
            start_time = time.time()
            if model_training_max_epochs > 0:
                new_key, model_key = jax.random.split(key)
                model_keys = jax.random.split(model_key, local_devices_to_use)
                training_state, num_train_batches, num_test_batches, transitions, test_transitions, train_valid_len, test_valid_len, train_mask, test_mask, perm, train_length, test_length = process_data(training_state, env_buffer_state, model_keys)
                new_key, model_key = jax.random.split(new_key)
                training_state, model_metrics = train_model(
                transitions=transitions,
                test_transitions=test_transitions,
                training_state=training_state,
                num_train_batches=num_train_batches,
                num_test_batches=num_test_batches,
                key=model_key
                )
                model_train_time += time.time() - start_time
                new_key, buffer_key = jax.random.split(new_key)

                if clear_model_buffer_after_model_train:
                    sac_buffer_state = jax.pmap(sac_buffer.init)(
                        jax.random.split(buffer_key, local_devices_to_use)
                    )
            # do env steps, hallucinations, and sac training
            epoch_key, new_local_key = jax.random.split(new_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)

            training_state, sac_training_state, env_state, env_buffer_state, sac_buffer_state, policy_metrics = sim_training_step(
                    training_state,
                    sac_training_state,
                    env_state,
                    env_buffer_state,
                    epoch_keys,
                    sac_buffer_state,
                    model_horizon,
                    hallucination_updates_per_training_step,
                    target_entropy)
            other_time += time.time() - start_time

        policy_metrics = jax.tree_util.tree_map(jp.mean, policy_metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), policy_metrics)
        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = env_steps_per_epoch / epoch_training_time
        metrics = {
            'training/sps': sps,
            'training/walltime': training_walltime,
            'training/model_train_time': model_train_time,
            'training/other_time': other_time,
            'training/model_horizon': model_horizon,
            'training/hallucination_updates_per_training_step': (
                hallucination_updates_per_training_step),
            'training/env_buffer_size': env_buffer.size(env_buffer_state),
            **{f'model/{name}': value for name, value in model_metrics.items()},
            **{f'{name}': value for name, value in policy_metrics.items()}

        }
        return training_state, sac_training_state, env_buffer_state, sac_buffer_state, metrics, training_walltime, env_state, new_local_key

    local_key, env_key, prefill_key = jax.random.split(local_key, 3)
    env_keys = jax.random.split(env_key, num_envs)
    env_state = env.reset(env_keys)
    env_state = jax.device_put_replicated(
        env_state, jax.local_devices()[:local_devices_to_use]
    )

    prefill_keys = jax.random.split(prefill_key, num_envs)

    assert local_devices_to_use==1, "local_devices_to_use must be equal to 1 for env rollouts"
    (training_state, sac_training_state, env_state, env_buffer_state,
    sac_buffer_state, _) = prefill_env_buffer(
        training_state, sac_training_state, env_state,
        env_buffer_state, sac_buffer_state, prefill_keys,
        deterministic_in_env)

    env_buffer_size = env_buffer.size(env_buffer_state)
    logging.info('env buffer size after init exploration %s', env_buffer_size)

    # Training loop
    training_walltime = 0
    init_target_entropy = -target_entropy_coef * float(action_size)

    for epoch in range(num_epochs):
        current_target_entropy = min(0.0, init_target_entropy + entropy_rate * epoch)
        (sac_buffer, sac_buffer_state, local_key, model_horizon, hallucination_updates_per_training_step) = update_model_horizon(sac_buffer, epoch, local_key, sac_buffer_state, model_horizon)
        (training_state, sac_training_state, env_buffer_state, sac_buffer_state, training_metrics, training_walltime,
        env_state, new_local_key) = sim_training_epoch_with_timing(
            training_state=training_state,
            sac_training_state=sac_training_state,
            sac_buffer_state=sac_buffer_state,
            env_buffer_state=env_buffer_state,
            training_walltime=training_walltime,
            env_state=env_state,
            model_horizon=model_horizon,
            hallucination_updates_per_training_step=hallucination_updates_per_training_step,
            key=local_key,
            target_entropy=current_target_entropy,
            )

        metrics = evaluator.run_evaluation(
            _unpmap((sac_training_state.normalizer_params,
            sac_training_state.policy_params)),
            training_metrics)
        current_step = int(_unpmap(training_state.env_steps))
        continue_train_flag = progress_fn(current_step, metrics)

        policy_params_fn(
            current_step,
            make_policy_render,
            _unpmap((sac_training_state.normalizer_params,
            sac_training_state.policy_params)),
            metrics,
            wm_training_state=_unpmap(training_state))

        if not continue_train_flag:
            return metrics

    return metrics

def _calculate_dataset_lengths_jax(D, B, test_set_ratio):
    """All-JAX; safe under jit/pmap.
    D: jnp.int32[] (tracer ok)
    B: int or jnp.int32[]
    test_set_ratio: float or jnp.float32[]
    """
    D = jp.asarray(D, dtype=jp.int32)
    B = jp.asarray(B, dtype=jp.int32)
    ratio = jp.asarray(test_set_ratio, dtype=jp.float32)

    total_batches = jp.floor_divide(D, B)                             # int32

    test_batches = jp.floor(total_batches.astype(jp.float32) * ratio).astype(jp.int32)

    train_batches = total_batches - test_batches                      # int32

    train_length = train_batches * B                                  # int32
    test_length  = test_batches  * B                                  # int32
    return train_length, test_length, train_batches, test_batches

def get_experience_brax(
    normalizer_params: running_statistics.RunningStatisticsState,
    policy_params: types.Params,
    make_policy: Callable,
    env_state: envs.State,
    model_buffer_state: sac_train.ReplayBufferState,
    key: types.PRNGKey,
    env: envs.Env,
    model_replay_buffer: replay_buffers.UniformSamplingQueue,
    deterministic: bool
) -> Tuple[running_statistics.RunningStatisticsState,
           envs.State, sac_train.ReplayBufferState]:

    policy = make_policy((normalizer_params, policy_params),
                         deterministic=deterministic)
    env_state, transitions = actor_step(
        env, env_state, policy, key, extra_fields=('truncation',))

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation)

    model_buffer_state = model_replay_buffer.insert(model_buffer_state,
                                                    transitions)
    return normalizer_params, env_state, model_buffer_state, transitions

def get_experience_model(
    normalizer_params: running_statistics.RunningStatisticsState,
    policy_params: types.Params,
    make_policy: Callable,
    model_scaler_params: base.ScalerParams,
    model_params: types.Params,
    make_model: Callable,
    model_env: base.ModelEnv,
    env_state: envs.State,
    model_buffer_state: sac_train.ReplayBufferState,
    model_replay_buffer: replay_buffers.UniformSamplingQueue,
    policy_std: jp.ndarray,
    output_info: bool,
    key: types.PRNGKey
):
    policy = make_policy((normalizer_params, policy_params), output_info=output_info)
    model = make_model((model_scaler_params, model_params))

    env_state, transitions, sub_reward = model_actor_step(
        model_env, env_state, policy, model, key,
        extra_fields=('truncation',))
    model_buffer_state = model_replay_buffer.insert(model_buffer_state,
                                                transitions)
    return env_state, model_buffer_state, transitions, sub_reward


def model_actor_step(
    env: envs.Env,
    env_state: envs.State,
    policy: types.Policy,
    model: base.Model,
    key: types.PRNGKey,
    extra_fields: Sequence[str] = ()
) -> Tuple[envs.State, types.SACTransition]:
    key_policy, key = jax.random.split(key)
    actions, policy_extras = policy(env_state.obs, key_policy)
    obs_stack = env_state.obs
    model_keys = jax.random.split(key, obs_stack['state'].shape[0])
    next_wm_obs, reward, rew_info, torque, sub_reward, next_obs, next_priv_obs = jax.vmap(model)(obs_stack['wm_state'], actions, env_state.rew_info, model_keys)

    info = env_state.info
    obs_dict = {
        'state': next_obs,
        'privileged_state': next_priv_obs,
        'wm_state': next_wm_obs
    }

    info['next_obs'] = obs_dict
    info['reward'] = reward
    env_state = env_state.replace(rew_info=rew_info)
    env_state = env_state.replace(info=info)
    env_state = env_state.replace(torque=torque)

    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, types.SACTransition(
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={
            'policy_extras': policy_extras,
            'state_extras': state_extras,

        }), sub_reward


def make_linear_threshold_fn(
    start_epoch: int,
    end_epoch: int,
    start_model_horizon: int,
    end_model_horizon: int
) -> Callable[[int], float]:
    a = start_epoch
    b = end_epoch
    x = start_model_horizon
    y = end_model_horizon

    def f(epoch):
        return math.floor(min(max(x + (epoch - a)/(b - a)*(y - x), x), y))

    return f

def make_linear_threshold_fn_float(
    start_epoch: int,
    end_epoch: int,
    start_value: float,
    end_value: float
):
    a, b = start_epoch, end_epoch
    x, y = float(start_value), float(end_value)
    lo, hi = (min(x, y), max(x, y))

    if b == a:
        def f(epoch):
            return x if epoch <= a else y
        return f

    def f(epoch):
        t = (epoch - a) / (b - a)              
        v = x + t * (y - x)                    
        return min(max(v, lo), hi)             

    return f
