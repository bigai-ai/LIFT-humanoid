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

import lift_utils.types as types
from brax import envs

from jax import numpy as jp
from typing import Callable, Protocol, Tuple, Optional
import flax
import optax
import jax
from lift_utils import running_statistics
from jax import lax

class Dataset:
    def __init__(self, transitions: types.SACTransition, horizon: int):
        self._transitions = transitions
        self._h = horizon

    def __len__(self):
        return self._transitions.observation.shape[0] - self._h

    def __getitem__(self, idxs: jp.ndarray):
        def f(idx):
            transitions = jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(
                    x, idx, self._h),
                self._transitions)
            return transitions
        return jax.vmap(f)(idxs)


@flax.struct.dataclass
class ScalerParams:
    obs_mu: jp.ndarray
    obs_std: jp.ndarray
    act_mu: jp.ndarray
    act_std: jp.ndarray


class Scaler:
    @staticmethod
    def fit_multi_device(
        obs_local: jp.ndarray,          
        act_local: jp.ndarray,          
        eps: float = 1e-6,
        axis_name: str = 'i',
        weights: Optional[jp.ndarray] = None  
    ) -> "ScalerParams":
        """get the mean and std of the observation and action in multi device"""

        if obs_local.ndim == 3:
            N, H, D_obs = obs_local.shape
            obs_local = obs_local.reshape(N * H, D_obs)
        else:
            N, D_obs = obs_local.shape

        if act_local.ndim == 3:
            Na, Ha, D_act = act_local.shape
            act_local = act_local.reshape(Na * Ha, D_act)
        else:
            Na, D_act = act_local.shape

        assert obs_local.shape[0] == act_local.shape[0], \
            f"obs and act batch mismatch: {obs_local.shape[0]} vs {act_local.shape[0]}"
        M = obs_local.shape[0]  

        if weights is None:
            w = jp.ones((M,), dtype=obs_local.dtype)
        else:
            if weights.ndim == 2:    
                w = weights.reshape(-1).astype(obs_local.dtype)
            else:
                w = weights.astype(obs_local.dtype)
            assert w.shape[0] == M, f"weights length {w.shape[0]} != batch {M}"

        # wsum mean how many samples are used to compute the mean
        wsum_local   = jp.sum(w)                                  
        obs_sum_local   = jp.sum(w[:, None] * obs_local, axis=0)   # [D_obs]
        obs_sumsq_local = jp.sum(w[:, None] * (obs_local**2), axis=0)
        act_sum_local   = jp.sum(w[:, None] * act_local, axis=0)   # [D_act]
        act_sumsq_local = jp.sum(w[:, None] * (act_local**2), axis=0)

        wsum    = lax.psum(wsum_local, axis_name=axis_name)
        obs_sum = lax.psum(obs_sum_local, axis_name=axis_name)
        obs_sumsq = lax.psum(obs_sumsq_local, axis_name=axis_name)
        act_sum = lax.psum(act_sum_local, axis_name=axis_name)
        act_sumsq = lax.psum(act_sumsq_local, axis_name=axis_name)

        # wsum mean how many samples are used to compute the mean
        wsum = jp.maximum(wsum, jp.array(1.0, dtype=obs_local.dtype))

        obs_mu  = obs_sum / wsum 
        obs_ex2 = obs_sumsq / wsum
        obs_var = jp.maximum(obs_ex2 - obs_mu * obs_mu, 0.0)
        obs_std = jp.sqrt(obs_var) + eps

        act_mu  = act_sum / wsum
        act_ex2 = act_sumsq / wsum
        act_var = jp.maximum(act_ex2 - act_mu * act_mu, 0.0)
        act_std = jp.sqrt(act_var) + eps

        return ScalerParams(
            obs_mu=obs_mu,  obs_std=obs_std,
            act_mu=act_mu,  act_std=act_std
        )

    @staticmethod
    def init(obs_size: int, act_size: int):
        return ScalerParams(
            obs_mu=jp.zeros(obs_size), obs_std=jp.ones(obs_size),
            act_mu=jp.zeros(act_size), act_std=jp.ones(act_size))
    @staticmethod
    def fit(obs: jp.ndarray, act: jp.ndarray):
        obs_mu = jp.mean(obs, axis=0)
        obs_std = jp.std(obs, axis=0) + 1e-6
        act_mu = jp.mean(act, axis=0)
        act_std = jp.std(act, axis=0) + 1e-6
        return ScalerParams(obs_mu=obs_mu, obs_std=obs_std,
                            act_mu=act_mu, act_std=act_std)

    @staticmethod
    def transform(obs: jp.ndarray, act: jp.ndarray, params: ScalerParams):
        obs = (obs - params.obs_mu) / params.obs_std
        act = (act - params.act_mu) / params.act_std
        return obs, act

    @staticmethod
    def inverse_transform(obs: jp.ndarray, act: jp.ndarray,
                          params: ScalerParams):
        obs = obs * params.obs_std + params.obs_mu
        act = act * params.act_std + params.act_mu
        return obs, act




@flax.struct.dataclass
class WM_TrainingState:
    model_optimizer_state: optax.OptState
    model_params: types.Params
    scaler_params: ScalerParams
    env_steps: jp.ndarray

@flax.struct.dataclass
class SACTrainingState:
    policy_optimizer_state: optax.OptState
    policy_params: types.Params
    original_policy_params: types.Params
    q_optimizer_state: optax.OptState
    q_params: types.Params
    target_q_params: types.Params
    gradient_steps: types.UInt64
    env_steps: types.UInt64
    alpha_optimizer_state: optax.OptState
    alpha_params: types.Params
    normalizer_params: running_statistics.RunningStatisticsState


class ModelEnv(envs.Env):
    """Environment for the model which hallucinates transitions."""
    def __init__(self, done_fn: Callable, observation_size: int, priv_observation_size:int, 
                 wm_observation_size: int, action_size: int):
        self._done_fn = done_fn
        self._observation_size = observation_size
        self._priv_observation_size = priv_observation_size
        self._wm_observation_size = wm_observation_size

        self._action_size = action_size

    def reset(self, rng: jp.ndarray):
        return

    def step(self, state: envs.State, action: jp.ndarray) -> envs.State:
        next_obs = state.info['next_obs']

        reward = state.info['reward']
        done = self._done_fn(next_obs['wm_state'], state.rew_info)
        nstate = state.replace(obs=next_obs, reward=reward, done=done)

        return nstate

    @property
    def observation_size(self):
        return {
            'state': self._observation_size,
            'privileged_state': self._priv_observation_size,
            'wm_state': self._wm_observation_size,

        }


    @property
    def action_size(self):
        return self._action_size

    @property
    def backend(self):
        return None

class Model(Protocol):

    def __call__(
        self,
        observation: types.Observation,
        action: types.Action,
        key: types.PRNGKey,
    ) -> Tuple[types.Observation, types.Reward]:
        pass
