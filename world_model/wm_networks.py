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

from world_model import wm_base
from typing import Tuple, Dict, Callable
from flax import linen as nn
from flax import struct
from jax import numpy as jp
import jax
from flax import linen as nn
import lift_utils.types as types
from typing import Optional, Any
ModelParams = Tuple[wm_base.ScalerParams, types.Params]

class EnsembleDense(nn.Module):
    """Ensemble Dense module.
    (ensemble_size, input_size) -> (ensemble_size, features)
    """
    features: int
    ensemble_size: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', self.kernel_init,
                            (self.ensemble_size, x.shape[-1], self.features))
        bias = self.param('bias', self.bias_init,
                          (self.ensemble_size, self.features))
        return jax.vmap(jp.matmul)(x, kernel) + bias


class EnsembleModel(nn.Module):
    """Ensemble model.
    (ensemble_size, obs_size + action_size) -> (ensemble_size, output_dim)
        -> (means: (ensemble_size, output_dim)
            logvars: (ensemble_size, obs_size)
    """
    obs_size: int
    output_dim: int
    ensemble_size: int
    num_elites: int
    hidden_size: int
    probabilistic: bool
    weight_decays: Dict[str, jp.ndarray] = struct.field(
        default_factory=lambda: {
            'ed1': 0.000025,
            'ed2': 0.00005,
            'ed3': 0.000075,
            'ed4': 0.000075,
            'ed5': 0.0001,
        }
    )

    def setup(self):
        self.max_logvar = 0.5 * jp.ones((1, self.obs_size))
        self.min_logvar = -10. * jp.ones((1, self.obs_size))
        self.act = nn.swish

        self.ed1 = EnsembleDense(self.hidden_size, self.ensemble_size)
        self.ed2 = EnsembleDense(self.hidden_size, self.ensemble_size)
        self.ed3 = EnsembleDense(self.hidden_size, self.ensemble_size)
        self.ed4 = EnsembleDense(self.hidden_size, self.ensemble_size)
        # Probabilistic head emits [mean (torque seq), logvar (one-step state)]
        out_dim = (self.output_dim + self.obs_size if self.probabilistic
                   else self.output_dim)
        self.ed5 = EnsembleDense(out_dim, self.ensemble_size)

    @nn.compact
    def __call__(self, x, train=False, rngs=None):
        elite_idxs = self.variable('elites', 'idxs',  # noqa: F841
                                   lambda: jp.arange(self.num_elites))

        x = self.ed1(x)
        x = self.act(x)
        x = self.ed2(x)
        x = self.act(x)
        x = self.ed3(x)
        x = self.act(x)
        x = self.ed4(x)
        x = self.act(x)
        x = self.ed5(x)

        if self.probabilistic:
            mean, logvar = jp.split(x, [self.output_dim], axis=-1)
            logvar = self.max_logvar - nn.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + nn.softplus(logvar - self.min_logvar)
            return mean, logvar
        else:
            return x, jp.zeros((self.ensemble_size, self.obs_size))

def make_model_network(obs_size: int,
                       output_dim: int,
                       hidden_size: int = 200,
                       ensemble_size: int = 7,
                       num_elites: int = 5,
                       probabilistic: bool = False):
    """Creates a model network."""
    assert ensemble_size >= num_elites
    return EnsembleModel(obs_size, output_dim, ensemble_size, num_elites,
                         hidden_size, probabilistic)

def make_inference_fn(
    ensemble_model: EnsembleModel,
    preprocess_fn: lambda x, y: (x, y),
    wm_obs_size: int,
    wm_obs_size_per_step: int,
    torque_dim: int,
    wm_noise_to_actor_noise_fn: Callable,
    dynamics_fn: Callable,
    reward_fn: Callable,
    model_probabilistic: bool=True,
    plot_model_rollouts: bool=False,
    robot_config: Optional[Any] = None,
):

    def make_model(model_params: ModelParams):

        def model(obs_stack: types.Observation, action: types.Action, rew_info: dict,
                  key: types.PRNGKey):
            scaler_params, params = model_params
            flat_obs = obs_stack[:wm_obs_size]

            # use the last substep as the current normalized wm state
            last_wm_state = flat_obs[-wm_obs_size_per_step:]
            unscale_action = action / robot_config.policy_output_scale
            proc_obs, proc_act = preprocess_fn(last_wm_state, unscale_action,
                                               scaler_params)
            x = jp.concatenate([proc_obs, proc_act], axis=-1)
            # repeat x across ensemble dimension
            x = jp.tile(x, (ensemble_model.ensemble_size,) + (1,))

            means, logvars = ensemble_model.apply(params, x)
            key_normal, key_choice = jax.random.split(key)
            key_normal, k1 = jax.random.split(key_normal, 2)
            elite_idxs = jp.asarray(params['elites']['idxs'], dtype=jp.int32)
            idx = jax.random.choice(key_choice, elite_idxs)
            mean = means[idx]
            logvar = logvars[idx]
            norm_std = jp.sqrt(jp.exp(logvar))
            if not model_probabilistic:
                norm_std = jp.zeros_like(norm_std)
            # propagate the mean through the dynamics
            def f(carry, in_element_unused):
                obs = carry
                obs_next = dynamics_fn(obs, action, mean)
                pos = obs_next[1]
                vel = obs_next[2]
                ang = obs_next[3]
                rot = obs_next[4]
                qdd = obs_next[5]
                torque = obs_next[6]
                actor_state = obs_next[7]
                critic_state = obs_next[8]
                wm_state_next = obs_next[0]
                return wm_state_next, (torque, pos, vel, ang, rot, qdd, actor_state, critic_state)

            wm_state_next_mean, us = jax.lax.scan(
                f, flat_obs, (), length=1)
            torque = us[0][0]
            pos = us[1][0]
            vel = us[2][0]
            ang = us[3][0]
            rot = us[4][0]
            qdd = us[5][0]
            actor_state = us[6][0]
            critic_state = us[7][0]
            actor_noise = wm_noise_to_actor_noise_fn(norm_std[-wm_obs_size_per_step:], actor_state, k1)

            wm_state_next = wm_state_next_mean
            actor_state = actor_state + actor_noise
            rew_info = {
                **rew_info,
                'rigid_state_pos': pos,
                'rigid_state_lin_vel': vel,
                'rigid_state_ang_vel': ang,
                'rigid_state_rot': rot,
                'rigid_state_qdd': qdd,
            }
            # compute reward
            reward, reward_components = reward_fn(norm_wm_state=wm_state_next[-wm_obs_size_per_step:], prev_norm_wm_state=last_wm_state, torques=torque, action=action, info=rew_info, valid_step=1.0)

            rew_info['last_last_actions'] = rew_info['last_actions']
            rew_info['last_actions'] = action

            sub_reward = {}
            if plot_model_rollouts:
                for reward_components_key in reward_components.keys():
                    sub_reward.update({
                            'sub_'+reward_components_key:reward_components[reward_components_key]
                        }
                    )
                sub_reward.update({
                    'action_mean': us[0][0],
                    'model_output_mean': mean,
                    'model_output_std': jp.sqrt(jp.exp(logvar))
                })            
            return wm_state_next, reward, rew_info, torque, sub_reward, actor_state, critic_state

        return model

    return make_model
