# Copyright 2025 The Brax Authors.
# Modifications Copyright 2025 LIFT-Humanoid Authors
#
# This file is ADAPTED FROM the Brax project and includes local modifications.
# Original source licensed under the Apache License, Version 2.0 (the "License").
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Network definitions."""

import dataclasses
from flax import linen
import jax
import jax.numpy as jnp
from typing import Any, Callable, Literal, Mapping, Sequence

from lift_utils import types
from lift_utils import running_statistics


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
    obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
    return jax.tree_util.tree_flatten(obs_size)[0][-1]


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    last_layer_in_fp32: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            if i != len(self.layer_sizes) - 1:
                hidden = linen.Dense(
                    hidden_size,
                    name=f'hidden_{i}',
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype
                )(hidden)
            else:
                hidden = linen.Dense(
                    hidden_size,
                    name=f'hidden_{i}',
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                    dtype=jnp.float32 if self.last_layer_in_fp32 else self.dtype,
                    param_dtype=self.param_dtype
                )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
                if self.layer_norm:
                    hidden = linen.LayerNorm(dtype=jnp.float32, param_dtype=self.param_dtype)(hidden)
        return hidden


class Param(linen.Module):
    """Scalar parameter module."""

    init_value: float = 0.0
    size: int = 1

    @linen.compact
    def __call__(self):
        return self.param(
            'value', init_fn=lambda keys: jnp.full((self.size,), self.init_value)
        )


class LogParam(linen.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0
    size: int = 1

    @linen.compact
    def __call__(self):
        log_value = self.param(
            'log_value',
            init_fn=lambda key: jnp.full((self.size,), jnp.log(self.init_value)),
        )
        return jnp.exp(log_value)


class PolicyModuleWithStd(linen.Module):
    """Policy module with learnable mean and standard deviation."""

    param_size: int
    hidden_layer_sizes: Sequence[int]
    activation: ActivationFn
    kernel_init: jax.nn.initializers.Initializer
    layer_norm: bool
    noise_std_type: Literal['scalar', 'log']
    init_noise_std: float
    state_dependent_std: bool = False

    @linen.compact
    def __call__(self, obs):
        if self.noise_std_type not in ['scalar', 'log']:
            raise ValueError(
                f'Unsupported noise std type: {self.noise_std_type}. Must be one of'
                ' "scalar" or "log".'
            )

        outputs = MLP(
            layer_sizes=list(self.hidden_layer_sizes),
            activation=self.activation,
            kernel_init=self.kernel_init,
            layer_norm=self.layer_norm,
            activate_final=True,
        )(obs)

        mean_params = linen.Dense(
            self.param_size, kernel_init=self.kernel_init
        )(outputs)

        if self.state_dependent_std:
            log_std_output = linen.Dense(
                self.param_size, kernel_init=self.kernel_init
            )(outputs)
            if self.noise_std_type == 'log':
                std_params = jnp.exp(log_std_output)
            else:
                std_params = log_std_output

        else:
            if self.noise_std_type == 'scalar':
                std_module = Param(
                    self.init_noise_std, size=self.param_size, name='std_param'
                )
            else:
                std_module = LogParam(
                    self.init_noise_std, size=self.param_size, name='std_logparam'
                )
            std_params = std_module()

        return mean_params, std_params


def make_policy_network(
    param_size: int,
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    last_layer_in_fp32: bool = False,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    if distribution_type == 'tanh_normal':
        policy_module = MLP(
            layer_sizes=list(hidden_layer_sizes) + [param_size],
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
            dtype=dtype,
            param_dtype=param_dtype,
            last_layer_in_fp32=last_layer_in_fp32
        )
    elif distribution_type == 'normal':
        policy_module = PolicyModuleWithStd(
            param_size=param_size,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
            noise_std_type=noise_std_type,
            init_noise_std=init_noise_std,
            state_dependent_std=state_dependent_std,
        )
    else:
        raise ValueError(
            f'Unsupported distribution type: {distribution_type}. Must be one'
            ' of "normal" or "tanh_normal".'
        )

    def apply(processor_params, policy_params, obs):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[obs_key], normalizer_select(processor_params, obs_key)
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    obs_size = _get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, obs_size))

    def init(key):
        policy_module_params = policy_module.init(key, dummy_obs)
        return policy_module_params

    return FeedForwardNetwork(init=init, apply=apply)


def make_value_network(
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    value_obs_key: str = 'state',
    dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    last_layer_in_fp32: bool = False,
) -> FeedForwardNetwork:
    """Creates a value network."""
    value_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        dtype=dtype,
        param_dtype=param_dtype,
        last_layer_in_fp32=last_layer_in_fp32
    )

    def apply(processor_params, value_params, obs):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[value_obs_key], normalizer_select(processor_params, value_obs_key)
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return jnp.squeeze(value_module.apply(value_params, obs), axis=-1)

    critic_obs_size = _get_obs_state_size(obs_size, value_obs_key)
    dummy_obs = jnp.zeros((1, critic_obs_size))
    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs), apply=apply
    )


def make_q_network(
    obs_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    layer_norm: bool = False,
    value_obs_key: str = 'state',
    dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    last_layer_in_fp32: bool = False,
) -> FeedForwardNetwork:
    """Creates a value network."""

    class QModule(linen.Module):
        """Q Module."""

        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            for _ in range(self.n_critics):
                q = MLP(
                    layer_sizes=list(hidden_layer_sizes) + [1],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                    layer_norm=layer_norm,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    last_layer_in_fp32=last_layer_in_fp32
                )(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, q_params, obs, actions):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[value_obs_key], normalizer_select(processor_params, value_obs_key)
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    obs_size = _get_obs_state_size(obs_size, value_obs_key)
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


def normalizer_select(
    processor_params: running_statistics.RunningStatisticsState, obs_key: str
) -> running_statistics.RunningStatisticsState:
    return running_statistics.RunningStatisticsState(
        count=processor_params.count,
        mean=processor_params.mean[obs_key],
        summed_variance=processor_params.summed_variance[obs_key],
        std=processor_params.std[obs_key],
    )
