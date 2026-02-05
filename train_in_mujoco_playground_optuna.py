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
"""Train a LIFT agent using JAX on the specified environment."""
import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
# TODO: track the performance of X64
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", False)
import functools
import warnings
import math

from absl import app
from absl import flags
from absl import logging
from policy_pretrain import sac_networks
import policy_pretrain.train as sac

from lift_configs import pretrain_sac_config
from mujoco_playground import registry
from mujoco_playground import wrapper

import optuna

from optuna.pruners import SuccessiveHalvingPruner, PatientPruner
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.storages import RDBStorage

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

# --- new flags ---
_STORAGE = flags.DEFINE_string(
    "storage",
    "sqlite:///optuna_sac.db",  # local single-file storage can be parallelized (multi-process/multi-script)
    "Optuna storage URL, e.g., sqlite:///file.db or postgresql://user:pw@host:5432/db",
)
_STUDY_NAME = flags.DEFINE_string(
    "study_name",
    "sac_tuning_leapcube",
    "Optuna study name (use the same to run workers in parallel).",
)
_SAMPLER = flags.DEFINE_string(
    "sampler",
    "cmaes",  # options: cmaes / tpe / random
    "Sampler type for Optuna: cmaes | tpe | random",
)
_POPSIZE = flags.DEFINE_integer("popsize", 12, "CMA-ES population size (≈ parallelism).")
_N_JOBS = flags.DEFINE_integer(
    "n_jobs", 1,
    "In-process parallel workers for study.optimize; prefer multi-process instead."
)

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)

_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_GRAD_UPDATES_PER_STEP = flags.DEFINE_integer(
    "grad_updates_per_step", 8, "Number of updates per batch sample"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.99, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")

_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_LOG_ALPHA = flags.DEFINE_float("log_alpha", 1e-3, "Log Alpha ")
_MAX_REPLAY_SIZE = flags.DEFINE_integer(
    "max_replay_size", 1000, "Max replay buffer size"
)

def get_rl_config(env_name: str):
    return pretrain_sac_config(env_name)


def main(argv):
    """Run training and evaluation for the specified environment."""

    del argv

    if _USE_TB.present and _USE_TB.value:
        raise ValueError("--use_tb is not supported in this Optuna runner.")
    if _LOAD_CHECKPOINT_PATH.present and _LOAD_CHECKPOINT_PATH.value:
        raise ValueError("--load_checkpoint_path is not supported in this Optuna runner.")

    def _resolve_robot_config(env_name: str):
        name = env_name.lower()
        if "g1" in name:
            from brax.robots.g1.utils import g1Utils as robot_config
            return robot_config
        if "t1" in name:
            from brax.robots.booster.utils import BoosterUtils as robot_config
            return robot_config
        return None

    # Load environment configuration
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    sac_params = get_rl_config(_ENV_NAME.value)

    if _NUM_TIMESTEPS.present:
        sac_params.num_timesteps = _NUM_TIMESTEPS.value
    if _PLAY_ONLY.present:
        sac_params.num_timesteps = 0
    if _NUM_EVALS.present:
        sac_params.num_evals = _NUM_EVALS.value
    if _REWARD_SCALING.present:
        sac_params.reward_scaling = _REWARD_SCALING.value
    if _EPISODE_LENGTH.present:
        sac_params.episode_length = _EPISODE_LENGTH.value
    if _NORMALIZE_OBSERVATIONS.present:
        sac_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
    if _ACTION_REPEAT.present:
        sac_params.action_repeat = _ACTION_REPEAT.value
    if _GRAD_UPDATES_PER_STEP.present:
        sac_params.grad_updates_per_step = _GRAD_UPDATES_PER_STEP.value
    if _DISCOUNTING.present:
        sac_params.discounting = _DISCOUNTING.value
    if _LEARNING_RATE.present:
        sac_params.actor_learning_rate = _LEARNING_RATE.value
        sac_params.critic_learning_rate = _LEARNING_RATE.value
        sac_params.alpha_learning_rate = _LEARNING_RATE.value
    if _NUM_ENVS.present:
        sac_params.num_envs = _NUM_ENVS.value
    if _NUM_EVAL_ENVS.present:
        sac_params.num_eval_envs = _NUM_EVAL_ENVS.value
    if _BATCH_SIZE.present:
        sac_params.batch_size = _BATCH_SIZE.value
    if _LOG_ALPHA.present:
        sac_params.int_log_alpha = _LOG_ALPHA.value
    if _MAX_REPLAY_SIZE.present:
        sac_params.max_replay_size = _MAX_REPLAY_SIZE.value

    if hasattr(sac_params, "network_factory"):
        if _POLICY_HIDDEN_LAYER_SIZES.present:
            sac_params.network_factory.policy_hidden_layer_sizes = list(
                map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
            )
        if _VALUE_HIDDEN_LAYER_SIZES.present:
            sac_params.network_factory.q_hidden_layer_sizes = list(
                map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
            )
        if _POLICY_OBS_KEY.present:
            sac_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
        if _VALUE_OBS_KEY.present:
            sac_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

    if _VISION.value:
        raise ValueError("not implement")



    def sample_sac_params(trial: optuna.Trial):
        """Sampler for SAC hyperparameters (CMA-ES friendly: float -> discretize)."""

        # ===== continuous params (float / log sampling) =====
        discounting = trial.suggest_float("discounting", 0.975, 0.995)
        alpha_learning_rate = trial.suggest_float("alpha_learning_rate", 1e-4, 2e-3, log=True)
        actor_learning_rate  = trial.suggest_float("actor_learning_rate",  1e-4, 2e-3, log=True)
        critic_learning_rate = trial.suggest_float("critic_learning_rate", 1e-4, 2e-3, log=True)
        tau = trial.suggest_float("tau", 1e-3, 5e-2, log=True)
        target_entropy_coef = trial.suggest_float("target_entropy_coef", 0.01, 1.0)

        # sample alpha then take log (fixes prior naming ambiguity)
        int_alpha = trial.suggest_float("int_alpha", 1e-5, 1.0, log=True)
        int_log_alpha = math.log(int_alpha)

        # ===== params that are int/powers/step-aligned: sample float then discretize =====
        # UTD: integer 2~20
        gups_f = trial.suggest_float("grad_updates_per_step_f", 2.0, 20.0)
        grad_updates_per_step = max(2, min(20, int(round(gups_f))))

        # Replay buffer：10**k,  k∈[4,6]
        buf_pow_f = trial.suggest_float("buffer_size_pow_f", 4.0, 5.0)
        max_replay_size = 10 ** int(round(buf_pow_f))

        # parallel env count: 2**k, k in [10,13]  -> 1024~8192
        num_env_pow_f = trial.suggest_float("num_env_pow_f", 10.0, 13.0)
        num_envs = 2 ** int(round(num_env_pow_f))
        ####
        # batch size: 2**k, k in [4,15] -> 16~32768 (your old comment said 128~4096; tighten if needed)
        bs_pow_f = trial.suggest_float("batch_size_pow_f", 4.0, 13.0)
        batch_size = 2 ** int(round(bs_pow_f))

        # reward scaling：2**k, k∈[1,5]
        rew_pow_f = trial.suggest_float("reward_scaling_pow_f", 1.0, 3.0, log=True)
        reward_scaling = 2 ** int(round(rew_pow_f))

        return {
            "num_envs": num_envs,
            "grad_updates_per_step": grad_updates_per_step,
            "max_replay_size": max_replay_size,
            "int_log_alpha": int_log_alpha,
            "discounting": discounting,
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "alpha_learning_rate": alpha_learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            "reward_scaling": reward_scaling,
            "target_entropy_coef": target_entropy_coef,
        }   

    N_TRIALS = 5000  # Maximum number of trials

    if not _NUM_TIMESTEPS.present:
        sac_params.num_timesteps = 80_000_000
    if not _NUM_EVALS.present:
        sac_params.num_evals = 20

    robot_config = _resolve_robot_config(_ENV_NAME.value)


    def objective(trial: optuna.Trial) -> float:
        is_pruned = False
        sample_result = sample_sac_params(trial)
        sac_params.num_envs = sample_result['num_envs']
        sac_params.grad_updates_per_step = sample_result['grad_updates_per_step']
        sac_params.max_replay_size = sample_result['max_replay_size']
        sac_params.int_log_alpha = sample_result['int_log_alpha']
        sac_params.discounting = sample_result['discounting']
        sac_params.target_entropy_coef = sample_result['target_entropy_coef']
        sac_params.actor_learning_rate = sample_result['actor_learning_rate']
        sac_params.critic_learning_rate = sample_result['critic_learning_rate']
        sac_params.alpha_learning_rate = sample_result['alpha_learning_rate']
        sac_params.batch_size = sample_result['batch_size']
        sac_params.tau = sample_result['tau']
        sac_params.reward_scaling = sample_result['reward_scaling']
        #sac_params.entropy_rate = sample_result['entropy_rate']

        training_params = dict(sac_params)
        if "wandb_entity" in training_params:
            del training_params["wandb_entity"]
        if "network_factory" in training_params:
            del training_params["network_factory"]


        network_fn = sac_networks.make_sac_networks

        if hasattr(sac_params, "network_factory"):
            network_factory = functools.partial(
                network_fn, **sac_params.network_factory
            )
        else:
            network_factory = network_fn


        if _DOMAIN_RANDOMIZATION.value:
            training_params["randomization_fn"] = registry.get_domain_randomizer(
                _ENV_NAME.value
            )
        env = registry.load(_ENV_NAME.value, config=env_cfg)


        num_eval_envs = (
            sac_params.num_envs
            if _VISION.value
            else sac_params.get("num_eval_envs", 128)
        )


        if "num_eval_envs" in training_params:
            del training_params["num_eval_envs"]

        train_fn = functools.partial(
            sac.train,
            **training_params,
            network_factory=network_factory,
            seed=_SEED.value,
            wrap_env_fn=None if _VISION.value else wrapper.wrap_for_brax_training,
            wrap_eval_env_fn=None if _VISION.value else wrapper.wrap_for_brax_evaluating,
            num_eval_envs=num_eval_envs,
            eval_at_start=True,
            deterministic_eval=True,
            robot_config=robot_config,
        )


        # Load evaluation environment
        eval_env = (
            None if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg)
        )

        # Progress function for logging
        def progress(num_steps, metrics, sac_ts, make_inference_fn, render_reset_rng):
            nonlocal is_pruned

            print("num_steps", num_steps, metrics['eval/episode_reward'])
            trial.report(metrics['eval/episode_reward'], num_steps)
            # Prune trial if need
            if trial.should_prune():
                is_pruned = True
                return False, render_reset_rng
            return True, render_reset_rng
        # Train or load the model
        metrics = train_fn(  # pylint: disable=no-value-for-parameter
            environment=env,
            progress_fn=progress,
            eval_env=None if _VISION.value else eval_env,
            buffer_data_path=None,

        )
        if is_pruned:
            raise optuna.exceptions.TrialPruned()
        return metrics['eval/episode_reward']

    if _SAMPLER.value.lower() == "cmaes":
        sampler = CmaEsSampler(
            with_margin=True,
            sigma0=0.3,
            popsize=_POPSIZE.value,         # <<< controls number of parallel trials per generation
            seed=_SEED.value
        )
    elif _SAMPLER.value.lower() == "tpe":
        sampler = TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True,              # <<< parallel-safe, avoids over-optimism
            n_startup_trials=max(20, _POPSIZE.value)  # give TPE enough startup trials
        )
    elif _SAMPLER.value.lower() == "random":
        sampler = RandomSampler(seed=_SEED.value)
    else:
        raise ValueError(f"Unknown sampler: {_SAMPLER.value}")

    base_pruner = SuccessiveHalvingPruner(min_resource=3, reduction_factor=3)
    pruner = PatientPruner(base_pruner, patience=2)


    # --- parallel storage (critical) ---
    storage = RDBStorage(
        url=_STORAGE.value,
        engine_kwargs={"pool_pre_ping": True},
        heartbeat_interval=60,   # <<< worker heartbeat for crash recovery
    )

    study_name = _STUDY_NAME.value
    if _SUFFIX.present and _SUFFIX.value:
        study_name = f"{study_name}-{_SUFFIX.value}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,      # <<< reuse parallel workers
    )

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=_N_JOBS.value)


    # Best result
    best_trial = study.best_trial
    print(best_trial)



if __name__ == "__main__":
    app.run(main)
