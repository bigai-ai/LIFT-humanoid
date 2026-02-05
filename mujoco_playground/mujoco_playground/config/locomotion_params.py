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
"""RL config for Locomotion envs."""

from ml_collections import config_dict

from mujoco_playground._src import locomotion


def brax_sac_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned Brax SAC config for the given environment."""
    env_config = locomotion.get_default_config(env_name)

    rl_config = config_dict.create(
        render=False,
        num_timesteps=5_000_000,
        num_evals=1000,
        reward_scaling=1.0,
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        discounting=0.9870596636685084,
        num_envs=1000,
        batch_size=1024,
        grad_updates_per_step=9,
        max_replay_size=1000 * 1000,
        min_replay_size=8192,
        tau=0.024184784277809342,
        network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            q_hidden_layer_sizes=(1024, 512, 256),
            policy_obs_key="state",
            value_obs_key="privileged_state",
            activation='swish',
            q_network_layer_norm=False,

        ),
        target_entropy_coef = 0.5,
        int_log_alpha=-3.348060147490869,
        entropy_rate=0.0,
        actor_learning_rate = 0.00010344851660896503,
        critic_learning_rate = 0.00010015193933286596,
        alpha_learning_rate =  0.009969445577213422,
    )

    if env_name in (
        "T1LowDimSimpRewJoystickFlatTerrain",
    ):

        rl_config.num_timesteps = 5_000_000_000
        rl_config.num_evals = 1000
        rl_config.num_eval_envs = 1024
        rl_config.discounting = 0.982017611613856
        rl_config.alpha_learning_rate = 0.0006570164638205332
        rl_config.actor_learning_rate = 0.00013164360719693946
        rl_config.critic_learning_rate = 0.0002126090725449083
        rl_config.tau = 0.017818858190597406
        rl_config.target_entropy_coef = 0.8963998146422688
        rl_config.int_log_alpha = -8.108259589476159
        rl_config.grad_updates_per_step = 17
        rl_config.max_replay_size = 1000 * 1000
        rl_config.num_envs = 4096
        rl_config.batch_size = 4096
        rl_config.reward_scaling = 3
    if env_name in (
        "G1LowDimJoystickFlatTerrain",
        "G1LowDimJoystickRoughTerrain",
        "T1LowDimJoystickFlatTerrain",
        "T1LowDimJoystickRoughTerrain",
        "T1LowDimSimFinetuneJoystickFlatTerrain",
        "T1LowDimSimFinetuneJoystickRoughTerrain",
        "T1LowDimRealFinetuneJoystickFlatTerrain",
        "T1LowDimRealFinetuneJoystickRoughTerrain",
    ):
        rl_config.num_timesteps = 5_000_000_000
        rl_config.num_evals = 1000
        rl_config.num_eval_envs = 1024

    if env_name in (
        "G1JoystickFlatTerrain",
        "G1JoystickRoughTerrain",
        "T1JoystickRoughTerrain",
    ):
        rl_config.num_timesteps = 5_000_000_000
        rl_config.num_evals = 1000
        rl_config.num_eval_envs = 1024
        rl_config.discounting = 0.9820125427256672
        rl_config.alpha_learning_rate = 0.0006441882558746206
        rl_config.actor_learning_rate = 0.0001699306824584787
        rl_config.critic_learning_rate = 0.00018773592095912688
        rl_config.tau = 0.0023896287473209837
        rl_config.target_entropy_coef = 0.3963871022148665
        rl_config.int_log_alpha = -3.004930256758307
        rl_config.grad_updates_per_step = 19
        rl_config.max_replay_size = 1000 * 1000
        rl_config.num_envs = 4096
        rl_config.batch_size = 16384
        rl_config.reward_scaling = 16

    if env_name in (
        "T1JoystickFlatTerrain",
    ):
        rl_config.num_timesteps = 5_000_000_000
        rl_config.num_evals = 1000
        rl_config.num_eval_envs = 1024
        rl_config.discounting = 0.9820980020719039
        rl_config.alpha_learning_rate = 0.0006413840226829715
        rl_config.actor_learning_rate = 0.00020500567475686062
        rl_config.critic_learning_rate = 0.0001365279781549369
        rl_config.tau = 0.006653397288400465
        rl_config.target_entropy_coef = 0.1741594908349707
        rl_config.int_log_alpha = -6.050352732435576
        rl_config.grad_updates_per_step = 16
        rl_config.max_replay_size = 1000 * 1000
        rl_config.num_envs = 4096
        rl_config.batch_size = 16384
        rl_config.reward_scaling = 32

    return rl_config

def brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned Brax PPO config for the given environment."""
    env_config = locomotion.get_default_config(env_name)

    rl_config = config_dict.create(
        num_timesteps=100_000_000,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=8192,
        batch_size=256,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(128, 128, 128, 128),
            value_hidden_layer_sizes=(256, 256, 256, 256, 256),
            policy_obs_key="state",
            value_obs_key="state",
        ),
    )

    if env_name in ("Go1JoystickFlatTerrain", "Go1JoystickRoughTerrain"):
        rl_config.num_timesteps = 200_000_000
        rl_config.num_evals = 10
        rl_config.num_resets_per_eval = 1
        rl_config.network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )

    elif env_name in ("Go1Handstand", "Go1Footstand"):
        rl_config.num_timesteps = 100_000_000
        rl_config.num_evals = 5
        rl_config.network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )

    elif env_name == "Go1Backflip":
        rl_config.num_timesteps = 200_000_000
        rl_config.num_evals = 10
        rl_config.discounting = 0.95
        rl_config.network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )

    elif env_name == "Go1Getup":
        rl_config.num_timesteps = 50_000_000
        rl_config.num_evals = 5
        rl_config.network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )

    elif env_name in ("G1JoystickFlatTerrain", "G1JoystickRoughTerrain"):
        rl_config.num_timesteps = 1_000_000_000
        rl_config.num_evals = 100
        rl_config.clipping_epsilon = 0.2
        rl_config.num_resets_per_eval = 1
        rl_config.entropy_cost = 0.005
        rl_config.network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )

    elif env_name in (
        "BerkeleyHumanoidJoystickFlatTerrain",
        "BerkeleyHumanoidJoystickRoughTerrain",
    ):
        rl_config.num_timesteps = 150_000_000
        rl_config.num_evals = 15
        rl_config.clipping_epsilon = 0.2
        rl_config.num_resets_per_eval = 1
        rl_config.entropy_cost = 0.005
        rl_config.network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )

    elif env_name in (
        "T1JoystickFlatTerrain",
        "T1JoystickRoughTerrain",
        "T1LowDimJoystickFlatTerrain",
        "T1LowDimJoystickRoughTerrain",
        "T1LowDimSimFinetuneJoystickFlatTerrain",
        "T1LowDimSimFinetuneJoystickRoughTerrain",
    ):
        rl_config.num_timesteps = 1_000_000_000
        rl_config.num_evals = 100
        rl_config.clipping_epsilon = 0.2
        rl_config.num_resets_per_eval = 1
        rl_config.entropy_cost = 0.005
        rl_config.network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )

    elif env_name in ("ApolloJoystickFlatTerrain",):
        rl_config.num_timesteps = 200_000_000
        rl_config.num_evals = 20
        rl_config.clipping_epsilon = 0.2
        rl_config.num_resets_per_eval = 1
        rl_config.entropy_cost = 0.005
        rl_config.network_factory = config_dict.create(
          policy_hidden_layer_sizes=(512, 256, 128),
          value_hidden_layer_sizes=(512, 256, 128),
          policy_obs_key="state",
          value_obs_key="privileged_state",
        )

    elif env_name in (
        "BarkourJoystick",
        "H1InplaceGaitTracking",
        "H1JoystickGaitTracking",
        "Op3Joystick",
        "SpotFlatTerrainJoystick",
        "SpotGetup",
        "SpotJoystickGaitTracking",
    ):
        pass  # use default config
    else:
        raise ValueError(f"Unsupported env: {env_name}")

    return rl_config


def rsl_rl_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned RSL-RL PPO config for the given environment."""

    rl_config = config_dict.create(
        seed=1,
        runner_class_name="OnPolicyRunner",
        policy=config_dict.create(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
            activation="elu",
            class_name="ActorCritic",
        ),
        algorithm=config_dict.create(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5,
            # mini batch size = num_envs*nsteps / nminibatches
            num_mini_batches=4,
            learning_rate=3.0e-4,  # 5.e-4
            schedule="fixed",  # could be adaptive, fixed
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        num_steps_per_env=24,  # per iteration
        max_iterations=100000,  # number of policy updates
        empirical_normalization=True,
        # logging
        save_interval=50,  # check for potential saves every this many iterations
        experiment_name="test",
        run_name="",
        # load and resume
        resume=False,
        load_run="-1",  # -1 = last run
        checkpoint=-1,  # -1 = last saved model
        resume_path=None,  # updated from load_run and chkpt
    )

    if env_name in (
        "Go1Getup",
        "BerkeleyHumanoidJoystickFlatTerrain",
        "G1Joystick",
        "Go1JoystickFlatTerrain",
    ):
        rl_config.max_iterations = 1000
    if env_name == "Go1JoystickFlatTerrain":
        rl_config.algorithm.learning_rate = 3e-4
        rl_config.algorithm.schedule = "fixed"

    return rl_config
