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

from world_model import wm_networks
import lift_utils.types as types
from world_model import wm_base
from jax import numpy as jp
import jax
from typing import Callable
from typing import Optional, Any
def make_losses(
    key: types.PRNGKey,
    ensemble_model: wm_networks.EnsembleModel,
    preprocess_fn: lambda x, y: (x, y),
    obs_size: int,
    wm_obs_size: int,
    wm_obs_size_per_step: int,
    torque_dim: int,
    wm_obs_hist_len: int,
    low_level_control_fn: Callable,
    dynamics_fn: Callable, 
    model_probabilistic: bool = True,
    model_training_weight_decay: bool = True,
    model_training_stop_gradient: bool = False,
    mean_loss_over_horizon: bool = False,  # used only for model evaluation,
    robot_config: Optional[Any] = None,
):

    def model_loss(model_params: types.Params,
                   other_vars: types.Params,
                   scaler_params: wm_base.ScalerParams,
                   obs_stack_r: types.Observation,
                   obs_next_stack_r: types.Observation,
                   actions_r: types.Action,
                   discount_r: jp.ndarray):

        # _r indicates actual data from the rollout; XX_r shapes:
        # (batch_size, horizon, ensemble_size, dim)

        obs_next, logvars = propagate_obs_batch(
            key,
            model_params, other_vars, scaler_params, obs_stack_r,
            actions_r, ensemble_model, preprocess_fn,
            wm_obs_size,
            wm_obs_size_per_step,
            torque_dim,
            wm_obs_hist_len,
            low_level_control_fn,
            dynamics_fn, 
            model_training_stop_gradient,
            robot_config,
            )
        obs_next_r = obs_next_stack_r

        error = obs_next - obs_next_r

        # Create a mask for NaN values in predictions and targets
        # nan_mask = jp.logical_not(jp.isnan(error))
        # # Apply NaN mask to error
        # error = jp.where(nan_mask, error, jp.zeros_like(error))

        # do not propagate the loss when a done is hit (discount = 0)
        discount_mask = jp.where(
            jp.expand_dims(jp.cumprod(discount_r, axis=1) == 0, axis=-1),
            jp.zeros_like(error), jp.ones_like(error))
        error = discount_mask * error

        # healthy_condition = jp.mean(error**2, axis=3, keepdims=True) < 10.0
        # error = jp.where(healthy_condition, error, jp.zeros_like(error))

        # compute the loss only for the mean, for each ensemble model, as an
        # auxiliary output
        if mean_loss_over_horizon:
            mean_loss = jp.mean(error**2, axis=(0, 2, 3))
        else:
            mean_loss = jp.mean(error**2, axis=(0, 1, 3))

        # loss for all models (average over batch, horizon, and dim; sum over
        # ensembles)
        if model_probabilistic:
            # Apply NaN mask to logvars as well
            # logvars = jp.where(nan_mask, logvars, jp.zeros_like(logvars))
            # logvars = jp.where(healthy_condition, logvars, jp.zeros_like(logvars))
            inv_vars = jp.exp(-logvars)

            mse_loss = jp.mean(inv_vars*error**2, axis=(0, 1, 3))
            var_loss = jp.mean(discount_mask*logvars, axis=(0, 1, 3))
            total_loss = jp.sum(mse_loss) + jp.sum(var_loss)
        else:
            # if not probabilistic, just use the mean loss
            total_loss = jp.sum(mean_loss)

        # add weight decay (L2 regularization) to the loss
        if model_training_weight_decay:
            for layer, decay in ensemble_model.weight_decays.items():
                weights = model_params[layer]['kernel']
                total_loss += 0.5 * decay * jp.sum(weights**2)

        return total_loss, (mean_loss, jp.mean(error**2, axis=(0, 1, 2)))

    return model_loss

def propagate_obs_batch(
        key: types.PRNGKey,
        model_params: types.Params,
        other_vars: types.Params,  # model variables, not variances
        scaler_params: wm_base.ScalerParams,
        obs_stack_r: types.Observation,
        actions_r: types.Action,
        ensemble_model: wm_networks.EnsembleModel,
        preprocess_fn: lambda x, y: (x, y),
        wm_obs_size: int,
        wm_obs_size_per_step: int,
        torque_dim: int,
        wm_obs_hist_len: int,
        low_level_control_fn: Callable,
        dynamics_fn: Callable, 
        model_training_stop_gradient: bool = False,
        robot_config: Optional[Any] = None,
        ):

    # obs_stack_r, obs_next_stack_r, actions_r shapes:
    # (batch_size, horizon, ensemble_size, dim)

    def propagate_obs(obs_stack_r, actions_r, key):

        # input shapes: (horizon, ensemble_size, dim)

        def outer(carry_outer, in_element_outer):

            # shape of carry items and in_element: (ensemble_size, dim)
            obs_stack = carry_outer
            actions_r = in_element_outer

            obs = obs_stack[:, :wm_obs_size]

            obs = jp.reshape(
                obs, (obs.shape[0], wm_obs_size_per_step)
            )
            if model_training_stop_gradient:
                obs = jax.lax.stop_gradient(obs)
            unscale_actions_r = actions_r / robot_config.policy_output_scale
            proc_obs_stack, proc_act_r = preprocess_fn(obs, unscale_actions_r,
                                                       scaler_params)
            x = jp.concatenate([proc_obs_stack, proc_act_r], axis=-1)
            means, logvars = ensemble_model.apply(
                {'params': model_params, **other_vars}, x, train=True, rngs={'dropout': key})

            # propagate the mean through the dynamics
            def inner(carry, in_element_unused):
                obs = carry
                u = jax.vmap(low_level_control_fn)(actions_r, obs)
                obs_next = jax.vmap(dynamics_fn)(obs, u, means)
                obs_next = obs_next[0]
                obs_next = obs_next.astype(obs.dtype)
                return obs_next, None
            obs_next, _ = jax.lax.scan(inner, obs_stack[:, :wm_obs_size], (), length=1)

            # update the obs stack
            obs_stack_next = jp.concatenate(
                [obs_next, obs_stack[:, :wm_obs_size*(wm_obs_hist_len-1)]],
                axis=-1
            )

            return obs_stack_next, (obs_next, logvars)

        _, (obs_next, logvars) = jax.lax.scan(
            outer, obs_stack_r[0, :, :], actions_r)

        return obs_next, logvars

    # output shape: (batch_size, horizon, ensemble_size, dim)
    key = jax.random.split(key, obs_stack_r.shape[0])
    return jax.vmap(propagate_obs)(obs_stack_r, actions_r, key)
