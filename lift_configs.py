from ml_collections import config_dict
from mujoco_playground._src import locomotion

_DEFAULT_WANDB_ENTITY = "xxx"

def pretrain_sac_config(env_name: str) -> config_dict.ConfigDict:
    """Returns tuned Brax SAC config for the given environment."""
    env_config = locomotion.get_default_config(env_name)
    pretrain_sac_config = config_dict.create(
        render=False,
        wandb_entity=_DEFAULT_WANDB_ENTITY,
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
        "T1LowDimRealFinetuneJoystickFlatTerrain",
        "T1LowDimRealFinetuneJoystickRoughTerrain",
    ):
        pretrain_sac_config.num_timesteps = 5_000_000_000
        pretrain_sac_config.num_evals = 1000
        pretrain_sac_config.num_eval_envs = 1024
        pretrain_sac_config.discounting = 0.982017611613856
        pretrain_sac_config.alpha_learning_rate = 0.0006570164638205332
        pretrain_sac_config.actor_learning_rate = 0.00013164360719693946
        pretrain_sac_config.critic_learning_rate = 0.0002126090725449083
        pretrain_sac_config.tau = 0.017818858190597406
        pretrain_sac_config.target_entropy_coef = 0.8963998146422688
        pretrain_sac_config.int_log_alpha = -8.108259589476159
        pretrain_sac_config.grad_updates_per_step = 17
        pretrain_sac_config.max_replay_size = 1000 * 1000
        pretrain_sac_config.num_envs = 4096
        pretrain_sac_config.batch_size = 4096
        pretrain_sac_config.reward_scaling = 3
    if env_name in (
        "G1LowDimJoystickFlatTerrain",
        "G1LowDimJoystickRoughTerrain",
        "T1LowDimJoystickFlatTerrain",
        "T1LowDimJoystickRoughTerrain",
        "T1LowDimSimFinetuneJoystickFlatTerrain",
        "T1LowDimSimFinetuneJoystickRoughTerrain",
    ):
        pretrain_sac_config.num_timesteps = 5_000_000_000
        pretrain_sac_config.num_evals = 1000
        pretrain_sac_config.num_eval_envs = 1024

    if env_name in (
        "G1JoystickFlatTerrain",
        "G1JoystickRoughTerrain",
        "T1JoystickRoughTerrain",
    ):
        pretrain_sac_config.num_timesteps = 5_000_000_000
        pretrain_sac_config.num_evals = 1000
        pretrain_sac_config.num_eval_envs = 1024
        pretrain_sac_config.discounting = 0.9820125427256672
        pretrain_sac_config.alpha_learning_rate = 0.0006441882558746206
        pretrain_sac_config.actor_learning_rate = 0.0001699306824584787
        pretrain_sac_config.critic_learning_rate = 0.00018773592095912688
        pretrain_sac_config.tau = 0.0023896287473209837
        pretrain_sac_config.target_entropy_coef = 0.3963871022148665
        pretrain_sac_config.int_log_alpha = -3.004930256758307
        pretrain_sac_config.grad_updates_per_step = 19
        pretrain_sac_config.max_replay_size = 1000 * 1000
        pretrain_sac_config.num_envs = 4096
        pretrain_sac_config.batch_size = 16384
        pretrain_sac_config.reward_scaling = 16

    if env_name in (
        "T1JoystickFlatTerrain",
    ):
        pretrain_sac_config.num_timesteps = 5_000_000_000
        pretrain_sac_config.num_evals = 1000
        pretrain_sac_config.num_eval_envs = 1024
        pretrain_sac_config.discounting = 0.9820980020719039
        pretrain_sac_config.alpha_learning_rate = 0.0006413840226829715
        pretrain_sac_config.actor_learning_rate = 0.00020500567475686062
        pretrain_sac_config.critic_learning_rate = 0.0001365279781549369
        pretrain_sac_config.tau = 0.006653397288400465
        pretrain_sac_config.target_entropy_coef = 0.1741594908349707
        pretrain_sac_config.int_log_alpha = -6.050352732435576
        pretrain_sac_config.grad_updates_per_step = 16
        pretrain_sac_config.max_replay_size = 1000 * 1000
        pretrain_sac_config.num_envs = 4096
        pretrain_sac_config.batch_size = 16384
        pretrain_sac_config.reward_scaling = 32

    return pretrain_sac_config


def pretrain_wm_config(env_name: str) -> config_dict.ConfigDict:
    """Returns world-model pretrain config for the given environment."""
    pretrain_policy_config = pretrain_sac_config(env_name)

    wm_config = config_dict.create(
        max_replay_size=pretrain_policy_config.max_replay_size,
        model_training_max_epochs=1,
        ensemble_size=1,
        num_elites=1,
        wm_learning_rate=1e-3,
        hidden_size=400,
        model_training_batch_size=200,
        model_training_test_ratio=0.2,
        model_loss_horizon=1,
        model_probabilistic=True,
        model_training_weight_decay=True,
        model_training_stop_gradient=False,
        mean_loss_over_horizon=False,
        model_training_consec_converged_epochs=2, # TODO: tune this
        model_training_convergence_criteria=0.01,
        ssrl_dynamics_fn='contact_integrate_only',
        wm_obs_history_length=1,
        seed=0,
    )
    return wm_config


def finetune_sac_config(env_name: str) -> config_dict.ConfigDict:
    pretrain_policy_config = pretrain_sac_config(env_name)
    wm_config = pretrain_wm_config(env_name)

    rl_config=config_dict.create(
        normalize_observations=pretrain_policy_config.normalize_observations,
        action_repeat=pretrain_policy_config.action_repeat,
        network_factory=pretrain_policy_config.network_factory,
        float32_compute_dtype=True,
        float32_param_dtype=True,
        last_layer_in_fp32=False,
        obs_history_length=1,
        priv_obs_history_length=1,
        wm_obs_history_length=1,

    )
    finetune_config = config_dict.create(
        run_name=None,
        save_policy=False,
        use_wandb=True,
        wandb_entity=_DEFAULT_WANDB_ENTITY,
        transmit=True,
        render_during_training=True,
        render_epoch_interval=3,
        render_seed=0,
        ac_training_state_path=None,
        wm_training_state_path=None,
        start_with_pretrain_policy=True,
        dynamics_fn=wm_config.ssrl_dynamics_fn,
        rl_config=rl_config,
        wm_config=config_dict.create(
            seed=0,
            episode_length=pretrain_policy_config.episode_length,
            num_epochs=300,
            model_trains_per_epoch=1,
            training_steps_per_model_train=1,
            env_steps_per_training_step=pretrain_policy_config.episode_length,
            model_rollouts_per_hallucination_update=400,
            sac_grad_updates_per_hallucination_update=20,
            init_exploration_steps=pretrain_policy_config.episode_length,
            clear_model_buffer_after_model_train=False,
            action_repeat=pretrain_policy_config.action_repeat,
            obs_history_length=rl_config.obs_history_length,
            priv_obs_history_length=rl_config.priv_obs_history_length,
            wm_obs_history_length=rl_config.wm_obs_history_length,
            num_envs=1,
            num_evals=301,
            num_eval_envs=1024,
            policy_normalize_observations=rl_config.normalize_observations,
            model_learning_rate=wm_config.wm_learning_rate,
            model_training_batch_size=wm_config.model_training_batch_size,
            model_training_max_sgd_steps_per_epoch=None,
            model_training_max_epochs=1000,
            model_training_convergence_criteria=0.01,
            model_training_consec_converged_epochs=6,
            model_training_abs_criteria=None,
            model_training_test_ratio=0.2,
            model_training_weight_decay=wm_config.model_training_weight_decay,
            model_training_stop_gradient=wm_config.model_training_stop_gradient,
            mean_loss_over_horizon=wm_config.mean_loss_over_horizon,
            model_loss_horizon=4,
            model_check_done_condition=True,
            max_env_buffer_size=60000,
            max_model_buffer_size=400000,
            sac_learning_rate=2e-4,
            sac_discounting=0.99,
            sac_batch_size=256,
            real_ratio=0.06,
            sac_reward_scaling=pretrain_policy_config.reward_scaling,
            sac_tau=0.001,
            sac_init_log_alpha=-7.1308988302963465,
            deterministic_in_env=True,
            deterministic_eval=True,
            hallucination_max_std=-1.0,
            load_ac_optimizer_state=True,
            load_alpha=False,
            entropy_rate=0.0,
            target_entropy_coef=0.5,
        ),

        finetune_env_config=config_dict.create(
            forces_in_q_coords=True,
        ),

        world_model_config=config_dict.create(
            hidden_size=wm_config.hidden_size,
            ensemble_size=wm_config.ensemble_size,
            num_elites=wm_config.num_elites,   
            model_probabilistic=wm_config.model_probabilistic
        ),
        linear_threshold_fn=config_dict.create(
            start_epoch=0,
            end_epoch=10,
            start_model_horizon=1,
            end_model_horizon=20
        ),
        hupts_fn=config_dict.create(
            start_epoch=0,
            end_epoch=4,
            start_hupts=10,
            end_hupts=1000
        ),

    )

    return finetune_config
