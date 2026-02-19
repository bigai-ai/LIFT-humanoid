import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
# use 64-bit precision to limit rollout dynamics mismatch
from jax import config
config.update("jax_enable_x64", False)
config.update("jax_default_matmul_precision", 'highest')
from lift_configs import finetune_sac_config
# set environment variables before loading anything from jax/brax
# os.environ["XLA_FLAGS"] = '--xla_gpu_deterministic_ops=true'
from omegaconf import OmegaConf
from datetime import datetime
import functools as ft
import wandb
import dill
import numpy as np

from brax.envs.g1_lowdim_joystick import G1LowDimJoystick
from brax.envs.t1_lowdim_sim_joystick import T1LowDimSimJoystick
from brax.envs.t1_lowdim_real_joystick import T1LowDimRealJoystick

from brax import envs
from brax.evaluate import evaluate
from brax.io import html
import jax
import jax.numpy as jp
from mujoco_playground import registry
from policy_pretrain import sac_networks
from world_model import wm_networks as wm_networks
from world_model import finetune_wm_ac
from world_model import wm_base as wm_base

from brax.robots.g1.utils import g1Utils
from brax.robots.booster.utils_sim import BoosterUtils as T1SimUtils
from brax.robots.booster.utils_real import BoosterUtils as T1RealUtils
from absl import app
from absl import flags
from absl import logging
from etils import epath

def int_multiply(x, y):
    return int(x * y)

def plot_wm_group(
    pred, target, ar_pred, labels, title, save_path,
    reset_interval: int | None = None,   # NEW
):
    import matplotlib.pyplot as plt
    import numpy as np                   # NEW

    dim = pred.shape[1]
    rows = dim
    fig_h = max(2.0 * rows, 4.0)

    fig, axes = plt.subplots(
        rows, 1, figsize=(12, fig_h), sharex=True
    )

    if rows == 1:
        axes = [axes]

    # NEW: indices of each reset point
    if reset_interval is not None and reset_interval > 0:
        T = target.shape[0]
        reset_idx = np.arange(0, T, reset_interval)
    else:
        reset_idx = None

    for i in range(rows):
        axes[i].plot(target[:, i], label="target", linewidth=1.5)
        axes[i].plot(pred[:, i], label="pred (1-step)", alpha=0.7)
        axes[i].plot(ar_pred[:, i], label="pred (AR)", linestyle="--", alpha=0.9)

        # NEW: draw small stars on reset points (use target y so it aligns perfectly)
        if reset_idx is not None:
            axes[i].scatter(
                reset_idx,
                target[reset_idx, i],
                marker="*",
                s=25,          # small star
                zorder=5,
                label="reset" if i == 0 else None,  # only add legend once
            )

        axes[i].set_ylabel(labels[i])
        axes[i].grid(True)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time step")
    axes[0].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_subreward_series(gt, ar, title, save_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(gt, label="gt", linewidth=1.5)
    ax.plot(ar, label="wm_ar", linestyle="--", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Policy step")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def get_wm_groups(robot_config, action_size):
    def _slice_len(slc):
        if isinstance(slc, slice):
            return slc.stop - slc.start
        return 1

    groups = {}


    if hasattr(robot_config, "wm_base_lin_vel_idxs"):
        groups["base_lin_vel"] = {
            "slice": robot_config.wm_base_lin_vel_idxs,
            "labels": ["vx", "vy", "vz"],
        }
    if hasattr(robot_config, "wm_base_ang_vel_idxs"):
        groups["base_ang_vel"] = {
            "slice": robot_config.wm_base_ang_vel_idxs,
            "labels": ["wx", "wy", "wz"],
        }
    if hasattr(robot_config, "wm_q_idxs"):
        groups["q"] = {
            "slice": robot_config.wm_q_idxs,
            "labels": [f"q_{i}" for i in range(action_size)],
        }
    if hasattr(robot_config, "wm_qd_idxs"):
        groups["qd"] = {
            "slice": robot_config.wm_qd_idxs,
            "labels": [f"qd_{i}" for i in range(action_size)],
        }
    if hasattr(robot_config, "wm_last_action_idxs"):
        groups["last_action"] = {
            "slice": robot_config.wm_last_action_idxs,
            "labels": [f"a_{i}" for i in range(action_size)],
        }
    if hasattr(robot_config, "wm_quat_idxs"):
        groups["quat"] = {
            "slice": robot_config.wm_quat_idxs,
            "labels": ["qw", "qx", "qy", "qz"],
        }

    return groups
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    None,
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)

_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_WANDB_ENTITY = flags.DEFINE_string("wandb_entity", None, "wandb entity (overrides lift_configs)")
_AC_TRAINING_STATE_PATH = flags.DEFINE_string(
    "ac_training_state_path",
    None,
    "Path to a SAC training state .pkl to preload.",
)
_WM_TRAINING_STATE_PATH = flags.DEFINE_string(
    "wm_training_state_path",
    None,
    "Path to a world model training state .pkl to preload.",
)
# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)


def _sanitize_wandb_entity(entity):
    if entity is None:
        return None
    entity = str(entity).strip()
    if entity in ("", "xxx", "your_wandb_entity"):
        return None
    return entity



def main(argv):
    del argv
    env_name = _ENV_NAME.value
    cfg = finetune_sac_config(env_name)
    if _WANDB_ENTITY.present:
        cfg.wandb_entity = _WANDB_ENTITY.value
    if _SEED.present:
        cfg.wm_config.seed = _SEED.value
    if _AC_TRAINING_STATE_PATH.present:
        cfg.ac_training_state_path = _AC_TRAINING_STATE_PATH.value
    if _WM_TRAINING_STATE_PATH.present:
        cfg.wm_training_state_path = _WM_TRAINING_STATE_PATH.value
    OmegaConf.register_new_resolver("int_multiply", int_multiply)
    print("JAX devices:", jax.devices())
    print("Local device count:", jax.local_device_count())
    use_wandb = cfg.use_wandb
    wandb_entity = _sanitize_wandb_entity(cfg.wandb_entity)
    env_kwargs = cfg.finetune_env_config.to_dict()
    eplen = cfg.wm_config.episode_length
    deterministic_eval = cfg.wm_config.deterministic_eval
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{_ENV_NAME.value}-{timestamp}"
    if _SUFFIX.value is not None:
        exp_name += f"-{_SUFFIX.value}"
    print(f"Experiment name: {exp_name}")

    # Set up logging directory
    logdir = epath.Path("logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    # start wandb
    if use_wandb:
        wandb_kwargs = {
            "project": "LIFT_finetune",
            "name": exp_name,
            "config": cfg,
        }
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        try:
            wandb.init(**wandb_kwargs)
        except Exception as exc:  # login/auth/network issues should not kill training
            logging.warning("W&B init failed, disabling W&B logging: %s", exc)
            use_wandb = False
    save_data_dir = os.fspath(logdir)
    os.makedirs(save_data_dir, exist_ok=True)
    generate_data_dir = os.path.join(save_data_dir, "generate_data")
    os.makedirs(generate_data_dir, exist_ok=True)
    policy_ckpt_dir = os.path.join(save_data_dir, "policy_ckpt")
    os.makedirs(policy_ckpt_dir, exist_ok=True)
    policy_pkl_dir = os.path.join(save_data_dir, "policy_pkl")
    os.makedirs(policy_pkl_dir, exist_ok=True)
    wm_pkl_dir = os.path.join(save_data_dir, "wm_pkl")
    os.makedirs(wm_pkl_dir, exist_ok=True)
    wm_fig_dir = os.path.join(save_data_dir, "wm_figures")
    os.makedirs(wm_fig_dir, exist_ok=True)
    base_data_dir = os.path.join(save_data_dir, "data")
    os.makedirs(base_data_dir, exist_ok=True)
    print(f"Running on GPU {os.environ['CUDA_VISIBLE_DEVICES']}")

    # create env fn
    env_dict = {
        'G1LowDimJoystickFlatTerrain': G1LowDimJoystick,
        'T1LowDimSimFinetuneJoystickFlatTerrain': T1LowDimSimJoystick,
        'T1LowDimSimFinetuneJoystickRoughTerrain': T1LowDimSimJoystick,
        'T1LowDimRealFinetuneJoystickFlatTerrain': T1LowDimRealJoystick,
        'T1LowDimRealFinetuneJoystickRoughTerrain': T1LowDimRealJoystick,
        }
    env_fn = ft.partial(env_dict[env_name], backend='generalized')
    env_fn = add_kwargs_to_fn(env_fn, **env_kwargs)
    env = env_fn()

    done_fn = lambda *args: jp.zeros(())  # noqa: E731
    if cfg.wm_config.model_check_done_condition:
        assert hasattr(env, 'is_done'), (
            'The environment must have an is_done method to check if the '
            'episode is done. Otherwise, set model_check_done_condition=False')
        done_fn = env.is_done_in_wm if hasattr(env, 'is_done_in_wm') else env.is_done
    model_env = wm_base.ModelEnv(
        done_fn,
        env.observation_size['state'],
        env.observation_size['privileged_state'],
        env.observation_size['wm_state'],
        env.action_size,
    )
    model_env = envs.training.wrap(
        model_env,
        episode_length=cfg.wm_config.episode_length,
        action_repeat=cfg.wm_config.action_repeat,
        obs_history_length=cfg.wm_config.obs_history_length,
        priv_obs_history_length=cfg.wm_config.priv_obs_history_length,
        wm_obs_history_length=cfg.wm_config.wm_obs_history_length,
    )

    robot_config_dict = {
        'G1LowDimJoystickFlatTerrain': g1Utils,
        'T1LowDimSimFinetuneJoystickFlatTerrain': T1SimUtils,
        'T1LowDimSimFinetuneJoystickRoughTerrain': T1SimUtils,
        'T1LowDimRealFinetuneJoystickFlatTerrain': T1RealUtils,
        'T1LowDimRealFinetuneJoystickRoughTerrain': T1RealUtils,
    }

    # progress functions
    best_reward = -float('inf')
    best_reward_domain_curr = -float('inf')
    best_params_domain_curr = None
    epoch = 0


    wm_obs_size = env.observation_size['wm_state'] * cfg.wm_config.wm_obs_history_length
    wm_obs_size_per_step = wm_obs_size
    torque_dim = env.sys.qd_size()

    model_output_dim = (
        env.observation_size['privileged_state']
        if cfg.dynamics_fn == 'mbpo'
        else env.sys.qd_size()
    )

    dynamics_fn = env.make_ssrl_dynamics_fn(cfg.dynamics_fn)

    # make networks
    policy_network_factory = ft.partial(
        sac_networks.make_sac_networks,
        policy_hidden_layer_sizes=cfg.rl_config.network_factory.policy_hidden_layer_sizes,
        q_hidden_layer_sizes=cfg.rl_config.network_factory.q_hidden_layer_sizes,             
        activation=cfg.rl_config.network_factory.activation,
        compute_dtype=jp.float32 if cfg.rl_config.float32_compute_dtype else jp.float64,
        param_dtype=jp.float32 if cfg.rl_config.float32_param_dtype else jp.float64,
        last_layer_in_fp32=cfg.rl_config.last_layer_in_fp32,
    )

    model_network_factory = ft.partial(
        wm_networks.make_model_network,
        hidden_size=cfg.world_model_config.hidden_size,
        ensemble_size=cfg.world_model_config.ensemble_size,
        num_elites=cfg.world_model_config.num_elites,
        probabilistic=cfg.world_model_config.model_probabilistic)

    model_network = model_network_factory(
        obs_size=wm_obs_size,
        output_dim=model_output_dim,
    )

    make_model = wm_networks.make_inference_fn(
        ensemble_model=model_network,
        preprocess_fn=wm_base.Scaler.transform,
        wm_obs_size=wm_obs_size,
        wm_obs_size_per_step=wm_obs_size_per_step,
        torque_dim=torque_dim,
        wm_noise_to_actor_noise_fn=robot_config_dict[env_name].wm_noise_to_actor_noise,
        dynamics_fn=dynamics_fn,
        reward_fn=env.compute_reward,
        plot_model_rollouts=False,
        robot_config=robot_config_dict[env_name],
    )
    make_model_subreward = wm_networks.make_inference_fn(
        ensemble_model=model_network,
        preprocess_fn=wm_base.Scaler.transform,
        wm_obs_size=wm_obs_size,
        wm_obs_size_per_step=wm_obs_size_per_step,
        torque_dim=torque_dim,
        wm_noise_to_actor_noise_fn=robot_config_dict[env_name].wm_noise_to_actor_noise,
        dynamics_fn=dynamics_fn,
        reward_fn=env.compute_reward,
        plot_model_rollouts=True,
        robot_config=robot_config_dict[env_name],
    )

    # NOTE: params are dynamic, function is static
    def build_jitted_model(wm_training_state):
        model = make_model(
            (wm_training_state.scaler_params,
            wm_training_state.model_params)
        )
        return jax.jit(model)

    def build_jitted_model_subreward(wm_training_state):
        model = make_model_subreward(
            (wm_training_state.scaler_params,
            wm_training_state.model_params)
        )
        return jax.jit(model)

    def ar_rollout_scan(
        model,
        obs_stack,              # (T, wm_obs_size)
        actions,                # (T, action_dim)
        rew_info_batch,         # pytree, leading dim T
        keys,                   # (T, 2)
        reset_interval: int,
    ):
        """
        Returns:
            ar_states_norm: (T, wm_obs_size)  normalized
        """

        def step_fn(carry, inputs):
            """
            carry:
                current_state_norm: (wm_obs_size,)
            inputs:
                action_t
                rew_info_t
                key_t
                gt_state_t
                t
            """
            current_state_norm = carry
            action_t, rew_info_t, key_t, gt_state_t, t = inputs

            # periodic reset
            current_state_norm = jp.where(
                (t % reset_interval) == 0,
                gt_state_t,
                current_state_norm,
            )

            pred_next, _, _, _, _, _, _ = model(
                current_state_norm,
                action_t,
                rew_info_t,
                key_t,
            )

            next_state_norm = pred_next.reshape(-1)
            return next_state_norm, next_state_norm


        T = obs_stack.shape[0]

        inputs = (
            actions,
            rew_info_batch,
            keys,
            obs_stack,
            jp.arange(T),
        )

        init_state = obs_stack[0]

        _, ar_states_norm = jax.lax.scan(
            step_fn,
            init_state,
            inputs,
        )

        return ar_states_norm

    def ar_reward_rollout_model_env(
        model,
        init_state,
        actions,   # (T, action_dim), policy steps
        keys,      # (T, 2)
        gt_obs,
        gt_prev_obs,
        gt_rew_info,
        gt_metrics,
        reset_interval: int,
    ):
        array_types = (jp.ndarray, np.ndarray)
        if hasattr(jax, "Array"):
            array_types = (jax.Array, jp.ndarray, np.ndarray)

        def _maybe_add_batch(x):
            if isinstance(x, array_types):
                return x[None, ...]
            return x

        def _maybe_remove_batch(x):
            if isinstance(x, array_types) and x.shape[:1] == (1,):
                return x[0]
            return x

        init_info = dict(init_state.info)
        init_info.setdefault('first_obs', init_state.obs)
        init_info.setdefault('first_rew_info', init_state.rew_info)
        init_info.setdefault('first_metrics', init_state.metrics)
        init_info.setdefault('first_pipeline_state', init_state.pipeline_state)
        init_info.setdefault('next_obs', init_state.obs)
        init_info.setdefault('reward', init_state.reward)
        if 'episode_metrics' not in init_info:
            zero = jp.zeros_like(init_state.reward)
            episode_metrics = {
                'sum_reward': zero,
                'length': zero,
            }
            for metric_name in init_state.metrics.keys():
                episode_metrics[metric_name] = zero
            init_info['episode_metrics'] = episode_metrics
        init_info.setdefault('steps', jp.zeros_like(init_state.reward))
        init_info.setdefault('truncation', jp.zeros_like(init_state.reward))
        init_info.setdefault('episode_done', jp.zeros_like(init_state.reward))

        init_state = init_state.replace(info=init_info)
        init_state = jax.tree_util.tree_map(_maybe_add_batch, init_state)

        def step_fn(env_state, inputs):
            action_t, key_t, gt_obs_t, gt_prev_obs_t, gt_rew_info_t, gt_metrics_t, step_idx = inputs
            reset_step = (reset_interval > 0) & ((step_idx % reset_interval) == 0)
            reset_info = dict(env_state.info)
            reset_info['first_obs'] = gt_obs_t
            reset_info['first_rew_info'] = gt_rew_info_t
            reset_info['first_metrics'] = gt_metrics_t
            reset_info['next_obs'] = gt_obs_t
            reset_state = env_state.replace(
                obs=gt_obs_t,
                prev_obs=gt_prev_obs_t,
                rew_info=gt_rew_info_t,
                metrics=gt_metrics_t,
                info=reset_info,
            )
            env_state = jax.lax.cond(
                reset_step,
                lambda _: reset_state,
                lambda _: env_state,
                operand=None,
            )
            wm_state = env_state.obs['wm_state'][0]
            rew_info_unbatched = jax.tree_util.tree_map(
                _maybe_remove_batch, env_state.rew_info
            )
            next_wm_obs, reward, rew_info, torque, sub_reward, next_obs, next_priv_obs = model(
                wm_state,
                action_t,
                rew_info_unbatched,
                key_t,
            )
            sub_reward = {k: v for k, v in sub_reward.items() if k.startswith('sub_')}
            info = dict(env_state.info)
            info['next_obs'] = {
                'state': next_obs[None, ...],
                'privileged_state': next_priv_obs[None, ...],
                'wm_state': next_wm_obs[None, ...],
            }
            info['reward'] = reward[None, ...]
            env_state = env_state.replace(
                info=info,
                rew_info=jax.tree_util.tree_map(_maybe_add_batch, rew_info),
                torque=torque[None, ...],
            )
            env_state = model_env.step(env_state, action_t[None, ...])
            return env_state, sub_reward

        final_state, sub_rewards = jax.lax.scan(
            step_fn,
            init_state,
            (
                actions,
                keys,
                gt_obs,
                gt_prev_obs,
                gt_rew_info,
                gt_metrics,
                jp.arange(actions.shape[0]),
            ),
        )
        return final_state, sub_rewards

    def progress(num_steps, metrics):
        metrics['steps'] = num_steps
        print("Steps / Eval: ", num_steps)
        if 'eval/episode_reward' in metrics:
            nonlocal best_reward
            print("Reward is ", metrics['eval/episode_reward'])
            best_reward = max(best_reward, metrics['eval/episode_reward'])
            metrics['eval/best_reward'] = best_reward
        if 'eval/episode_forward_vel' in metrics:
            metrics['eval/episode_forward_vel'] = (
                metrics['eval/episode_forward_vel']
                / (eplen / cfg.wm_config.action_repeat)
            )
        if use_wandb:
            wandb.log(metrics, step=int(num_steps))
        return True

    def policy_params_fn(current_step, make_policy, params, metrics, wm_training_state=None):
        nonlocal epoch

        # store the best policy when using domain curriculum
        nonlocal best_reward_domain_curr
        nonlocal best_params_domain_curr


        jitted_model = build_jitted_model(wm_training_state)
        jitted_model_subreward = build_jitted_model_subreward(wm_training_state)

        # save policies at each evaluation step
        # save policies at each evaluation step
        if cfg.save_policy:
            if 'eval/episode_reward' in metrics.keys():
                fname = f"{'LIFT'}_{exp_name}_step_{current_step:.0f}_rew_{metrics['eval/episode_reward']:.0f}.pkl"

            else:
                fname = f"{'LIFT'}_{exp_name}_step_{current_step:.0f}.pkl"
            path = os.path.join(policy_pkl_dir, fname)
            with open(path, 'wb') as f:
                dill.dump(params, f)

            if wm_training_state is not None:
                if 'eval/episode_reward' in metrics.keys():
                    wm_fname = f"{'WM'}_{exp_name}_step_{current_step:.0f}_rew_{metrics['eval/episode_reward']:.0f}.pkl"
                else:
                    wm_fname = f"{'WM'}_{exp_name}_step_{current_step:.0f}.pkl"
                wm_path = os.path.join(wm_pkl_dir, wm_fname)
                with open(wm_path, 'wb') as f:
                    dill.dump(wm_training_state, f)


        # render evals
        if (use_wandb and cfg.render_during_training and epoch % cfg.render_epoch_interval == 0):
            key = jax.random.PRNGKey(cfg.render_seed)
            eval_results = evaluate(
                params=params,
                env_unwrapped=env,
                make_policy=make_policy,
                episode_length=eplen,
                action_repeat=cfg.wm_config.action_repeat,
                key=key,
                obs_history_length=cfg.wm_config.obs_history_length,
                priv_obs_history_length=cfg.wm_config.priv_obs_history_length,
                wm_obs_history_length=cfg.wm_config.wm_obs_history_length,
                deterministic=deterministic_eval,
                jit=True,
            )
            pipeline_states = eval_results[1]
            eval_states = eval_results[4]
            if wm_training_state is not None and len(pipeline_states) > 1:

                eval_actions = eval_results[0]
                actions = jp.stack(eval_actions)
                actions_per_step = jp.repeat(actions, cfg.wm_config.action_repeat, axis=0)
                num_steps = min(len(pipeline_states), len(eval_states), actions_per_step.shape[0])
                if num_steps > 1:
                    actions_per_step = actions_per_step[:num_steps]
                    obs_stack = jp.stack([state.obs['wm_state'] for state in eval_states[:num_steps - 1]])
                    rew_info_list = [state.rew_info for state in eval_states[:num_steps - 1]]
                    rew_info_batch = jax.tree_util.tree_map(lambda *xs: jp.stack(xs), *rew_info_list)
                    model_keys = jax.random.split(
                        jax.random.PRNGKey(cfg.render_seed + int(current_step)),
                        num_steps - 1,
                    )
                    pred_wm_obs, _, pred_rew_info, _, _, _, _ = jax.vmap(jitted_model)(
                        obs_stack,
                        actions_per_step[:num_steps - 1],
                        rew_info_batch,
                        model_keys,
                    )
                    pred_wm_state = pred_wm_obs.reshape(
                        ((num_steps - 1), wm_obs_size_per_step)
                    )
                    robot_config = robot_config_dict[env_name]
                    pred_wm_state_dn = robot_config.denormalize_state(
                        pred_wm_state, robot_config.wm_state_limits
                    )

                    ground_truth_wm_state = jp.stack([
                        state.obs['wm_state'] for state in eval_states[1:num_steps]
                    ])
                    ground_truth_wm_state = ground_truth_wm_state.reshape(
                        ((num_steps - 1), wm_obs_size_per_step)
                    )
                    ground_truth_wm_state_dn = robot_config.denormalize_state(
                        ground_truth_wm_state, robot_config.wm_state_limits
                    )

                    # ============================
                    # Autoregressive WM rollout (JAX)
                    # ============================

                    T = obs_stack.shape[0]

                    ar_wm_state_norm = ar_rollout_scan(
                        model=jitted_model,
                        obs_stack=obs_stack[:T],
                        actions=actions_per_step[:T],
                        rew_info_batch=jax.tree_util.tree_map(lambda x: x[:T], rew_info_batch),
                        keys=model_keys[:T],
                        reset_interval=20,
                    )

                    # reshape to substeps
                    ar_wm_state_norm = ar_wm_state_norm.reshape(
                        (-1, wm_obs_size_per_step)
                    )


                    ar_wm_state_dn = robot_config.denormalize_state(
                        ar_wm_state_norm,
                        robot_config.wm_state_limits,
                    )

                    gt_step_count = min(
                        len(eval_states) - 1,
                        actions_per_step.shape[0],
                    )
                    gt_policy_steps = min(
                        actions.shape[0],
                        gt_step_count // cfg.wm_config.action_repeat,
                    )
                    gt_metrics_total = None
                    gt_metrics_sum = None
                    gt_obs_batched = None
                    gt_prev_obs_batched = None
                    gt_rew_info_batched = None
                    gt_metrics_batched = None
                    if gt_policy_steps > 0:
                        gt_steps = gt_policy_steps * cfg.wm_config.action_repeat
                        gt_state_list = [
                            eval_states[i * cfg.wm_config.action_repeat]
                            for i in range(gt_policy_steps)
                        ]
                        gt_obs_batched = jax.tree_util.tree_map(
                            lambda *xs: jp.stack(xs), *[s.obs for s in gt_state_list]
                        )
                        gt_prev_obs_batched = jax.tree_util.tree_map(
                            lambda *xs: jp.stack(xs), *[s.prev_obs for s in gt_state_list]
                        )
                        gt_rew_info_batched = jax.tree_util.tree_map(
                            lambda *xs: jp.stack(xs), *[s.rew_info for s in gt_state_list]
                        )
                        gt_metrics_batched = jax.tree_util.tree_map(
                            lambda *xs: jp.stack(xs), *[s.metrics for s in gt_state_list]
                        )
                        gt_obs_batched = jax.tree_util.tree_map(
                            lambda x: x[:, None, ...], gt_obs_batched
                        )
                        gt_prev_obs_batched = jax.tree_util.tree_map(
                            lambda x: x[:, None, ...], gt_prev_obs_batched
                        )
                        gt_rew_info_batched = jax.tree_util.tree_map(
                            lambda x: x[:, None, ...], gt_rew_info_batched
                        )
                        gt_metrics_batched = jax.tree_util.tree_map(
                            lambda x: x[:, None, ...], gt_metrics_batched
                        )
                        gt_metrics_list = [
                            state.metrics for state in eval_states[1:1 + gt_steps]
                        ]
                        gt_metrics_stack = jax.tree_util.tree_map(
                            lambda *xs: jp.stack(xs), *gt_metrics_list
                        )
                        gt_metrics_stack = jax.tree_util.tree_map(
                            lambda x: x.reshape(
                                (gt_policy_steps, cfg.wm_config.action_repeat)
                                + x.shape[1:]
                            ),
                            gt_metrics_stack,
                        )
                        gt_metrics_sum = jax.tree_util.tree_map(
                            lambda x: jp.sum(x, axis=1), gt_metrics_stack
                        )
                        gt_metrics_total = jax.tree_util.tree_map(
                            lambda x: jp.sum(x, axis=0), gt_metrics_sum
                        )

                    if gt_policy_steps > 0:
                        ar_actions = actions[:gt_policy_steps]
                        ar_keys = jax.random.split(
                            jax.random.PRNGKey(cfg.render_seed + int(current_step) + 1),
                            gt_policy_steps,
                        )
                        ar_final_state, ar_sub_rewards = ar_reward_rollout_model_env(
                            model=jitted_model_subreward,
                            init_state=eval_states[0],
                            actions=ar_actions,
                            keys=ar_keys,
                            gt_obs=gt_obs_batched,
                            gt_prev_obs=gt_prev_obs_batched,
                            gt_rew_info=gt_rew_info_batched,
                            gt_metrics=gt_metrics_batched,
                            reset_interval=20,
                        )
                        ar_reward_value = float(
                            jax.device_get(
                                ar_final_state.info['episode_metrics']['sum_reward']
                            )[0]
                        )
                        metrics['wm/ar_episode_reward'] = ar_reward_value
                        print("WM AR episode reward:", ar_reward_value)
                        if use_wandb:
                            wandb.log(
                                {'wm/ar_episode_reward': ar_reward_value},
                                step=int(current_step),
                            )
                    if gt_metrics_total is not None:
                        ar_sub_rewards_sum = jax.tree_util.tree_map(
                            lambda x: jp.sum(x[:gt_policy_steps], axis=0),
                            ar_sub_rewards,
                        )
                        gt_metrics_total = jax.device_get(gt_metrics_total)
                        ar_sub_rewards_sum = jax.device_get(ar_sub_rewards_sum)
                        subreward_log = {}
                        for key, value in ar_sub_rewards_sum.items():
                            if not key.startswith('sub_'):
                                continue
                            gt_key = key[4:]
                            if gt_key not in gt_metrics_total:
                                continue
                            pred_val = float(np.asarray(value))
                            gt_val = float(np.asarray(gt_metrics_total[gt_key]))
                            subreward_log[f'wm/ar_subreward/{gt_key}'] = pred_val
                            subreward_log[f'wm/gt_subreward/{gt_key}'] = gt_val
                            subreward_log[f'wm/subreward_error/{gt_key}'] = pred_val - gt_val
                        if subreward_log:
                            metrics.update(subreward_log)
                            if use_wandb:
                                wandb.log(subreward_log, step=int(current_step))

                        if gt_metrics_sum is not None:
                            step_dir = os.path.join(
                                wm_fig_dir, f"step_{int(current_step)}"
                            )
                            subreward_dir = os.path.join(step_dir, "subreward")
                            os.makedirs(subreward_dir, exist_ok=True)
                            gt_metrics_sum_np = jax.device_get(gt_metrics_sum)
                            ar_sub_rewards_np = jax.device_get(ar_sub_rewards)
                            for key, value in ar_sub_rewards_np.items():
                                if not key.startswith('sub_'):
                                    continue
                                gt_key = key[4:]
                                if gt_key not in gt_metrics_sum_np:
                                    continue
                                ar_series = np.asarray(value[:gt_policy_steps])
                                gt_series = np.asarray(
                                    gt_metrics_sum_np[gt_key][:gt_policy_steps]
                                )
                                fig_path = os.path.join(
                                    subreward_dir, f"{gt_key}.png"
                                )
                                plot_subreward_series(
                                    gt=gt_series,
                                    ar=ar_series,
                                    title=f"{gt_key} (step {current_step})",
                                    save_path=fig_path,
                                )
                                if use_wandb:
                                    wandb.log(
                                        {f"wm/subreward/{gt_key}": wandb.Image(fig_path)},
                                        step=int(current_step),
                                    )


                    step_dir = os.path.join(
                        wm_fig_dir, f"step_{int(current_step)}"
                    )
                    os.makedirs(step_dir, exist_ok=True)

                    wm_groups = get_wm_groups(robot_config, action_size=env.action_size)

                    for name, figure_cfg in wm_groups.items():
                        sl = figure_cfg["slice"]
                        labels = figure_cfg["labels"]

                        pred_g = pred_wm_state_dn[:, sl]
                        tgt_g = ground_truth_wm_state_dn[:, sl]
                        ar_g   = ar_wm_state_dn[:, sl]

                        # ensure numpy & (T, D)
                        pred_g = np.asarray(pred_g)
                        tgt_g  = np.asarray(tgt_g)
                        ar_g   = np.asarray(ar_g)

                        fig_path = os.path.join(step_dir, f"{name}.png")

                        plot_wm_group(
                            pred=pred_g,
                            target=tgt_g,
                            ar_pred=ar_g,
                            labels=labels,
                            title=f"{name} (step {current_step})",
                            save_path=fig_path,
                            reset_interval=20,   # NEW
                        )

                        if use_wandb:
                            wandb.log(
                                {f"wm/{name}": wandb.Image(fig_path)},
                                step=int(current_step),
                            )

            render_html = html.render(env.sys.replace(dt=env.dt),
                                      pipeline_states,
                                      height=500)
            wandb.log(
                {f"Render at step {current_step}": wandb.Html(render_html)})

        epoch += 1



    # preload sac policy
    if cfg.start_with_pretrain_policy and cfg.ac_training_state_path is not None:
        sac_ts_path = cfg.ac_training_state_path
        with open(sac_ts_path, 'rb') as f:
            sac_ts = dill.load(f)
        print("load sac_ts", sac_ts_path)
    else:
        sac_ts = None
        print("sac_ts is None")

    if cfg.wm_training_state_path is not None:
        wm_ts_path = cfg.wm_training_state_path
        with open(wm_ts_path, 'rb') as f:
            wm_ts = dill.load(f)
        print("load wm_ts", wm_ts_path)
    else:   
        wm_ts = None
        print("wm_ts is None")
    # perform training
    train_fn = ft.partial(finetune_wm_ac.train)
    train_fn = add_kwargs_to_fn(train_fn, **cfg.wm_config)
    model_horizon_fn = finetune_wm_ac.make_linear_threshold_fn(
        cfg.linear_threshold_fn.start_epoch,
        cfg.linear_threshold_fn.end_epoch,
        cfg.linear_threshold_fn.start_model_horizon,
        cfg.linear_threshold_fn.end_model_horizon)
    hupts_fn = finetune_wm_ac.make_linear_threshold_fn(
        cfg.hupts_fn.start_epoch,
        cfg.hupts_fn.end_epoch,
        cfg.hupts_fn.start_hupts,
        cfg.hupts_fn.end_hupts)
    output_dim = (env.observation_size['privileged_state'] if cfg.dynamics_fn == 'mbpo'
                    else env.sys.qd_size())
    state = train_fn(
        environment=env,
        low_level_control_fn=env.low_level_control,
        dynamics_fn=dynamics_fn,
        reward_fn=env.compute_reward,
        model_output_dim=output_dim,
        model_horizon_fn=model_horizon_fn,
        hallucination_updates_per_training_step=hupts_fn,
        sac_training_state=sac_ts,
        wm_training_state=wm_ts,
        model_network_factory=model_network_factory,
        policy_network_factory=policy_network_factory,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        robot_config=robot_config_dict[env_name]
    )



def add_kwargs_to_fn(partial_fn, **kwargs):
    """add the kwargs to the passed in partial function"""
    for param in kwargs:
        partial_fn.keywords[param] = kwargs[param]
    return partial_fn


def dict_mean(dict_list):
    """Take a list of dicts with the same keys and return a dict with the mean
    of each key"""
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def lin_interp(start, end, i, n):
    """Linear interpolation between start and end for zero-indexed i out of n
    total iterations"""
    return start + i / (n-1) * (end - start)


if __name__ == "__main__":
    app.run(main)
