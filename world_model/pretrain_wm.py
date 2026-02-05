# Copyright 2025 The Brax Authors.
# Modifications Copyright 2025 LIFT-Humanoid Authors.
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

"""
    train the physics-informed world model using the SAC offline data.
"""
import time
from typing import Any, Callable, Optional, Tuple, Union

from absl import logging
from brax import envs
import sys

from lift_utils import gradients
from lift_utils import replay_buffers
sys.modules["brax.training.replay_buffers"] = replay_buffers

import lift_utils.types as types
from world_model import wm_base
from world_model import wm_losses as wm_losses
from world_model import wm_networks as wm_networks

from flax import linen as nn

import jax
import jax.numpy as jnp
import optax
import dill
import os 
import pickle

_PMAP_AXIS_NAME = None#'i'

def cast_arrays_only(x, desired_type):
    if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(desired_type)
    return x

def load_pkl_when_ready(path, timeout_s: float = 120.0, poll_s: float = 0.5):
    """Retries until a pickle file is fully written, then loads it."""
    deadline = time.time() + timeout_s if timeout_s > 0 else None
    last_err = None
    while True:
        try:
            if os.path.getsize(path) <= 0:
                raise EOFError("pickle file is empty")
            with open(path, "rb") as f:
                return dill.load(f)
        except (EOFError, pickle.UnpicklingError, OSError) as err:
            last_err = err
            if deadline is not None and time.time() > deadline:
                raise RuntimeError(
                    f"Timed out waiting for a complete pickle file: {path}"
                ) from last_err
            time.sleep(poll_s)

def _calculate_dataset_lengths(D: int, B: int, test_set_ratio: float):
    """Calculate the lengths of the training and test sets to be as close as
    possible to the test_set_ratio, ensuring that the training set length is
    divisible by B.

    Args:
    D: int, total number of samples in the dataset
    B: int, batch size
    test_set_ratio: float, ratio of the test set length to the total dataset
    """

    total_batches = D // B
    test_batches = int(jnp.floor(total_batches * test_set_ratio))
    train_batches = total_batches - test_batches
    train_length = train_batches * B
    test_length = test_batches * B

    return train_length, test_length, train_batches, test_batches

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)

def _init_training_state(
    key: types.PRNGKey,
    local_devices_to_use: int,
    model_network: Union[wm_networks.EnsembleModel, nn.Module],  # supports different network types
    model_optimizer: optax.GradientTransformation,
    wm_obs_size_per_step: int,
    wm_obs_hist_len: int, 
    action_size: int,
) -> wm_base.WM_TrainingState:
    """Inits the world model training state and replicates it over devices."""
    dummy_X = jnp.zeros((model_network.ensemble_size,
                      wm_obs_size_per_step * wm_obs_hist_len + action_size))
    model_params = model_network.init(key, dummy_X)
    model_optimizer_state = model_optimizer.init(model_params['params'])


    scaler_params = wm_base.Scaler.init(
        wm_obs_size_per_step * wm_obs_hist_len, action_size
    )


    training_state = wm_base.WM_TrainingState(
      model_optimizer_state=model_optimizer_state,
      model_params=model_params,
      scaler_params=scaler_params,
      env_steps=jnp.zeros(())
    )
    return jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )


def train(
    max_replay_size: Optional[int] = None,
    progress_fn: Callable[[int, types.Metrics], None] = lambda *args: None,
    model_use_env: Optional[envs.Env] = None,
    robot_config: Optional[Any] = None,
    data_path: str = '',
    model_training_max_epochs: int = 1,
    ensemble_size: int = 1,
    num_elites: int = 1,
    wm_learning_rate: float = 1e-3,
    hidden_size: int = 400,
    probabilistic: bool = True,
    model_training_batch_size: int = 200,
    model_training_test_ratio: float = 0.2,
    model_loss_horizon: int = 1,
    model_probabilistic: bool = True,
    model_training_weight_decay: bool = True,
    model_training_stop_gradient: bool = False,
    mean_loss_over_horizon: bool = False,
    model_training_consec_converged_epochs: int = 2,
    model_training_convergence_criteria: float = 0.01,
    ssrl_dynamics_fn: str = 'contact_integrate_only', #'mbpo'
    wm_obs_history_length: int = 1,
    seed: int = 0,
):
    """World model pretraining."""
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    device_count = local_devices_to_use * jax.process_count()
    logging.info(
        'local_device_count: %s; total_device_count: %s',
        local_devices_to_use,
        device_count,
    )

    obs_size = model_use_env.observation_size
    wm_obs_hist_len = wm_obs_history_length
    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)

    def _obs_dim(size):
        if isinstance(size, (tuple, list)):
            return int(size[0])
        return int(size)

    obs_state_size = _obs_dim(obs_size['state'])
    priv_obs_state_size = _obs_dim(obs_size['privileged_state'])
    wm_obs_size = _obs_dim(obs_size['wm_state'])
    action_size = model_use_env.action_size
    wm_obs_size_per_step = wm_obs_size

    dummy_obs = {
      "state": jnp.zeros(obs_size["state"]),
      "privileged_state": jnp.zeros(obs_size["privileged_state"]),
      "wm_state": jnp.zeros(obs_size["wm_state"]),
    }
    dummy_action = jnp.zeros((action_size,))
    dummy_extras={
        'state_extras': {'truncation': 0.0},
        'policy_extras': {
            'mean': jnp.zeros((action_size,)),
            'std': jnp.ones((action_size,)),
        },
    }

    dummy_transition = types.SACTransition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        next_observation=dummy_obs,
        extras=dummy_extras,
    )

    loss_key, key = jax.random.split(key)
    scale_fn = wm_base.Scaler.transform
    model_output_dim = (
        priv_obs_state_size
        if ssrl_dynamics_fn == 'mbpo'
        else model_use_env.sys.qd_size()
    )
    model_network = wm_networks.make_model_network(
       obs_size=wm_obs_size, 
       output_dim=model_output_dim,
       hidden_size=hidden_size,
       ensemble_size=ensemble_size,
       num_elites=num_elites,
       probabilistic=probabilistic,
       )

    model_loss = wm_losses.make_losses(
        loss_key, 
        model_network, 
        scale_fn, 
        obs_size=obs_state_size,
        wm_obs_size=wm_obs_size,
        wm_obs_size_per_step=wm_obs_size_per_step,
        torque_dim=model_use_env.sys.qd_size(),
        wm_obs_hist_len=wm_obs_hist_len,
        low_level_control_fn=model_use_env.low_level_control,
        dynamics_fn=model_use_env.make_ssrl_dynamics_fn(ssrl_dynamics_fn),
        model_probabilistic=model_probabilistic,
        model_training_weight_decay=model_training_weight_decay,
        model_training_stop_gradient=model_training_stop_gradient,
        mean_loss_over_horizon=mean_loss_over_horizon,
        robot_config=robot_config)
    model_optimizer = optax.adam(wm_learning_rate)

    model_update = gradients.gradient_update_fn(
       model_loss, model_optimizer, pmap_axis_name=None, has_aux=True)
    model_training_batch_size = model_training_batch_size // device_count


    # Use the replay buffer capacity as the upper bound for batching so
    # prepare_data padding never under-allocates when loading large buffers.
    max_wm_replay_size = max_replay_size
    total_batches = max_wm_replay_size // model_training_batch_size
    model_max_train_batches = int(
          jnp.ceil(total_batches * (1 - model_training_test_ratio)))
    model_max_test_batches = total_batches - model_max_train_batches + 1
    model_ensemble_size = ensemble_size
    wm_replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_wm_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=model_training_batch_size,
    )

    @jax.jit
    def prepare_data(
      transitions: types.SACTransition,
      test_transitions: types.SACTransition,
      key: types.PRNGKey
  ):
        # (num_samples, horizon, dim) -> (ensemble_size, num_samples, horizon, dim)
        # by duplicating the data for each model in the ensemble
        def duplicate_for_ensemble(transitions: types.SACTransition):
            num_samples = transitions.observation['wm_state'].shape[0]
            transitions = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0),
                                       transitions)
            transitions = jax.tree_map(
                lambda x: jnp.broadcast_to(
                    x, (model_ensemble_size, num_samples) + x.shape[2:]),
                transitions)
            return transitions
        transitions = duplicate_for_ensemble(transitions)
        test_transitions = duplicate_for_ensemble(test_transitions)

        # shuffle the data for each model in the ensemble (the sequence of data
        # that each model sees will be different)
        def shuffle_subarr(subarr, key):
            permuted_idxs = jax.random.permutation(key, subarr.shape[0])
            return subarr[permuted_idxs]
        keys = jax.random.split(key, model_ensemble_size)
        transitions = jax.tree_map(lambda x: jax.vmap(shuffle_subarr)(x, keys),
                                   transitions)

        # put data into batches: reshape to
        # (num_batches, batch_size, ensemble_size, horizon, dim)
        transitions = jax.tree_map(
            lambda x: x.reshape(
                ((-1, model_training_batch_size, model_ensemble_size)
                 + x.shape[2:])),
            transitions)
        test_transitions = jax.tree_map(
            lambda x: x.reshape(
                ((-1, model_training_batch_size, model_ensemble_size)
                 + x.shape[2:])),
            test_transitions)


        # transpose to (num_batches, batch_size, horizon, ensemble_size, dim)
        transitions = jax.tree_map(lambda x: jnp.swapaxes(x, 2, 3), transitions)
        test_transitions = jax.tree_map(lambda x: jnp.swapaxes(x, 2, 3),
                                        test_transitions)

        # expand to (max_batches, batch_size, horizon, ensemble_size, dim)
        def expand(arr, leading_dim):
            expanded_array_shape = (leading_dim,) + arr.shape[1:]
            expanded_array = jnp.zeros(expanded_array_shape)
            expanded_array = expanded_array.at[:arr.shape[0]].set(arr)
            return expanded_array
        transitions = jax.tree_map(
            lambda x: expand(x, model_max_train_batches), transitions)
        test_transitions = jax.tree_map(
            lambda x: expand(x, model_max_test_batches), test_transitions)

        return transitions, test_transitions
    @jax.jit
    def model_training_epoch_jit(
      params: Any,
      model_optimizer_state: optax.OptState,
      scaler_params: wm_base.ScalerParams,
      transitions: types.SACTransition,
      test_transitions: types.SACTransition,
      num_train_batches: int,
      num_test_batches: int,
      ):
        param_dtype = jax.tree_util.tree_leaves(params['params'])[0].dtype
        zero_loss = jnp.array(0.0, dtype=param_dtype)
        zero_ensemble = jnp.zeros((model_ensemble_size,), dtype=param_dtype)
        zero_all_axiss = jnp.zeros((wm_obs_size,), dtype=param_dtype)

        def sgd_step(carry, in_element):
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
                lambda: ((zero_loss, (zero_ensemble, zero_all_axiss)),
                        model_params, opt_state),
            )
            loss_ensemble, loss_all_axiss = aux
            return (({'params': new_model_params, **params}, opt_state),
                    (loss, loss_ensemble, loss_all_axiss))

        # perform an sgd step for each batch
        ((new_params, new_opt_state),
         (train_total_losses, mean_losses, train_mean_losses_all_axiss)) = jax.lax.scan(
            sgd_step, (params, model_optimizer_state),
            (transitions, jnp.arange(transitions.observation['wm_state'].shape[0])))

        train_mean_losses = jnp.mean(mean_losses, axis=-1)

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
                lambda: (zero_loss, (zero_ensemble, zero_all_axiss)),
            )
            test_mean_loss, test_mean_loss_all_axiss = aux
            return None, (test_total_loss, test_mean_loss, test_mean_loss_all_axiss)

        _, (test_total_losses, test_mean_losses, test_mean_losses_all_axiss) = jax.lax.scan(
            test, None,
            (test_transitions, jnp.arange(test_transitions.observation['wm_state'].shape[0])))


        new_params = {'params': new_model_params, **new_params}

        return (new_params, new_opt_state, train_total_losses, train_mean_losses, train_mean_losses_all_axiss,
                test_total_losses, test_mean_losses, test_mean_losses_all_axiss)

    def model_training_epoch(
      params: Any,
      model_optimizer_state: optax.OptState,
      scaler_params: wm_base.ScalerParams,
      transitions: types.SACTransition,
      test_transitions: types.SACTransition,
      num_train_batches: int,
      num_test_batches: int,
      key: types.PRNGKey
  ):

        transitions, test_transitions = prepare_data(transitions, test_transitions, key)
        (new_params, new_opt_state, train_total_losses, 
         train_mean_losses, train_mean_losses_all_axiss,
         test_total_losses, test_mean_losses, test_mean_losses_all_axiss
         ) = model_training_epoch_jit(
            params, model_optimizer_state, 
            scaler_params, transitions, test_transitions, num_train_batches, num_test_batches)

        return (new_params, new_opt_state,
                train_total_losses[num_train_batches-1],
                train_mean_losses[num_train_batches-1],
                jnp.mean(train_mean_losses_all_axiss, axis=0), # mean loss of all batches
                jnp.max(train_mean_losses_all_axiss, axis=0), # max loss of all batches
                jnp.mean(test_total_losses[:num_test_batches]),
                jnp.mean(test_mean_losses[:num_test_batches], axis=0),
                jnp.mean(test_mean_losses_all_axiss, axis=0), # mean loss of last batches
                )

    def train_model(
      training_state: wm_base.WM_TrainingState,
      env_buffer_state: types.ReplayBufferState,
      model_training_batch_size: int,
      key: types.PRNGKey) -> Tuple[wm_base.WM_TrainingState, dict, int, types.SACTransition]:

        # Create a dataset using all transitions from the env buffer
        buffer_size = wm_replay_buffer.size(env_buffer_state)
        buffer_size = buffer_size
        data = env_buffer_state.data[:buffer_size]
        all_transitions = wm_replay_buffer._unflatten_fn(data)
        dataset = wm_base.Dataset(all_transitions, model_loss_horizon)

        # create shuffled idxs and split them into training and test sets where the
        # length of the train set is divisible by the batch size and the length of
        # the test set is as close as possible to the test ratio
        key_shuffle, key = jax.random.split(key)
        per_idxs = jax.random.permutation(key_shuffle, jnp.arange(buffer_size))
        # (train_length, num_train_batches) = _calculate_dataset_lengths(
        #     buffer_size, model_training_batch_size

        (train_length, test_length, num_train_batches,
         num_test_batches) = _calculate_dataset_lengths(
            buffer_size, model_training_batch_size, model_training_test_ratio)

        train_idxs = per_idxs[:train_length]
        test_idxs = per_idxs[train_length:(train_length + test_length)]

        transitions = dataset[train_idxs]
        test_transitions = dataset[test_idxs]

        # scale the data
        all_train_data = data[train_idxs]
        all_train_transitions = wm_replay_buffer._unflatten_fn(all_train_data)
        train_obs = all_train_transitions.observation['wm_state']
        train_obs = train_obs.reshape(
            (train_obs.shape[0], wm_obs_size_per_step)
        )
        train_act = all_train_transitions.action
        # mesure the diversity

        scaler_params = wm_base.Scaler.fit(train_obs, train_act)
        training_state = training_state.replace(scaler_params=scaler_params)



        return training_state, metrics, num_train_batches, num_test_batches, transitions, test_transitions

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    # Training state init
    training_state = _init_training_state(
        key=global_key,
        local_devices_to_use=local_devices_to_use,
        model_network=model_network, 
        model_optimizer=model_optimizer,
        wm_obs_size_per_step=wm_obs_size_per_step,
        wm_obs_hist_len=wm_obs_hist_len,
        action_size=action_size,
    )
    del global_key

    local_key, rb_key, eval_key = jax.random.split(local_key, 3)

    # Replay buffer init
    wm_buffer_state = jax.pmap(wm_replay_buffer.init)(
        jax.random.split(rb_key, local_devices_to_use)
    )
    # Run initial eval
    metrics = {}
    if process_id == 0:
        logging.info(metrics)

        render_rng = progress_fn(0, metrics,
                    _unpmap(training_state),
                    jax.random.PRNGKey(0)
                    )

    # Create and initialize the replay buffer.
    total_update = 0
    training_state = _unpmap(training_state)

    expected_epoch = 0
    expected_mini_epoch = 0   # if you need it later

    def pkl_name(epoch, mini_epoch):
        return f"{epoch}_{mini_epoch}.pkl"

    while True:
        fname = pkl_name(expected_epoch, expected_mini_epoch)
        pkl_path = os.path.join(data_path, fname)
        if not os.path.exists(pkl_path):
            time.sleep(0.5)
            continue
        print(f"load expected pkl: {fname}")
        wm_buffer_state = load_pkl_when_ready(pkl_path)

        expected_epoch += 1

        k, new_key = jax.random.split(key)
        wm_buffer_state = _unpmap(wm_buffer_state)
        model_training_batch_size = int(model_training_batch_size)
        training_state, wm_metrics, num_train_batches, num_test_batches, transitions, test_transitions = train_model(training_state, wm_buffer_state, model_training_batch_size, k)

        metrics.update(wm_metrics)
        params = training_state.model_params
        opt_state = training_state.model_optimizer_state
        scaler_params = training_state.scaler_params
        best = 10e9 * jnp.ones(model_ensemble_size)

        for epoch in range(model_training_max_epochs):
            key, subkey = jax.random.split(key)

            params, opt_state, train_total_loss, train_mean_loss, train_mean_losses_all_axiss, train_max_mean_losses_all_axiss, test_total_loss, test_mean_loss, test_mean_losses_all_axiss = \
                model_training_epoch(params, opt_state, scaler_params, transitions, test_transitions, num_train_batches, num_test_batches, subkey)
            # use the loss from the final epoch
            training_state = training_state.replace(
                model_params=params,
                model_optimizer_state=opt_state,
                scaler_params=scaler_params)
            print(f'Model epoch {total_update}: train total loss {train_total_loss}, '
                f'train mean loss {train_mean_loss}, '
                f'test mean loss {test_mean_loss}')
            metrics['train_total_loss'] = train_total_loss
            metrics['train_mean_loss'] = train_mean_loss
            metrics['test_mean_loss'] = test_mean_loss

            if total_update % 5 == 0:
                render_rng = progress_fn(total_update,
                            metrics,
                            training_state,
                            render_rng
                            )
            total_update += 1
            # check convergence criteria
            improvement = (best - test_mean_loss) / best
            best = jnp.where(improvement > model_training_convergence_criteria,
                            test_mean_loss, best)
            if jnp.any(improvement > model_training_convergence_criteria):
                epochs_since_update = 0
            else:
                epochs_since_update += 1
            if epochs_since_update >= model_training_consec_converged_epochs:
                break
