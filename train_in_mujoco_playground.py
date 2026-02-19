# Copyright 2025 The Brax Authors.
# Modifications Copyright 2025 LIFT Author
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
"""Train a LIFT agent using JAX on the specified environment."""

import os
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
# TODO: check the performance of X64
jax.config.update('jax_default_matmul_precision', 'highest')
jax.config.update("jax_enable_x64", False)
from datetime import datetime
import functools
import inspect
import json

import time
import warnings

from absl import app
from absl import flags
from absl import logging
from policy_pretrain import sac_networks
import policy_pretrain.train as sac

from etils import epath


import mediapy as media
from tensorboardX import SummaryWriter
import wandb

import mujoco
from mujoco_playground import registry
from mujoco_playground import wrapper
from lift_configs import pretrain_sac_config
import dill


# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_WANDB_ENTITY = flags.DEFINE_string(
    "wandb_entity", None, "wandb entity (overrides lift_configs)"
)
_RENDER = flags.DEFINE_boolean(
    "render",
    False,
    "render",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SAVE_BUFFER_DATA = flags.DEFINE_boolean(
    "save_buffer_data", False, "save sac buffer"
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

_MAX_REPLAY_SIZE = flags.DEFINE_integer("max_replay_size", 1000, "Max replay buffer size")


def _sanitize_wandb_entity(entity):
    if entity is None:
        return None
    entity = str(entity).strip()
    if entity in ("", "xxx", "your_wandb_entity"):
        return None
    return entity


def main(argv):
    """Run training and evaluation for the specified environment."""

    del argv

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
    sac_params = pretrain_sac_config(_ENV_NAME.value)

    wandb_entity = getattr(sac_params, "wandb_entity", None)
    if _WANDB_ENTITY.present:
        wandb_entity = _WANDB_ENTITY.value
    wandb_entity = _sanitize_wandb_entity(wandb_entity)

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
        sac_params.learning_rate = _LEARNING_RATE.value
    if _NUM_ENVS.present:
        sac_params.num_envs = _NUM_ENVS.value
    if _NUM_EVAL_ENVS.present:
        sac_params.num_eval_envs = _NUM_EVAL_ENVS.value
    if _BATCH_SIZE.present:
        sac_params.batch_size = _BATCH_SIZE.value

    if _POLICY_HIDDEN_LAYER_SIZES.present:
        sac_params.network_factory.policy_hidden_layer_sizes = list(
            map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
        )
    if _VALUE_HIDDEN_LAYER_SIZES.present:
        sac_params.network_factory.value_hidden_layer_sizes = list(
            map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
        )
    if _POLICY_OBS_KEY.present:
        sac_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
    if _VALUE_OBS_KEY.present:
        sac_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

    if _LOG_ALPHA.present:
        sac_params.int_log_alpha = _LOG_ALPHA.value

    if _MAX_REPLAY_SIZE.present:
        sac_params.max_replay_size = _MAX_REPLAY_SIZE.value
    if _VISION.value:
        raise ValueError("not implement")
    env = registry.load(_ENV_NAME.value, config=env_cfg)

    print(f"Environment Config:\n{env_cfg}")
    print(f"SAC Training Parameters:\n{sac_params}")

    # Generate unique experiment name
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

    # Initialize Weights & Biases if required
    use_wandb = _USE_WANDB.value and not _PLAY_ONLY.value
    if use_wandb:
        wandb_kwargs = {"project": "LIFT_policy_pretrain", "name": exp_name}
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        try:
            wandb.init(**wandb_kwargs)
            wandb.config.update(env_cfg.to_dict())
            wandb.config.update({"env_name": _ENV_NAME.value})
        except Exception as exc:  # login/auth/network issues should not kill training
            logging.warning("W&B init failed, disabling W&B logging: %s", exc)
            use_wandb = False

    # Initialize TensorBoard if required
    if _USE_TB.value and not _PLAY_ONLY.value:
        writer = SummaryWriter(logdir)

    # Set up checkpoint directory
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # Save environment configuration
    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    raw_training_params = dict(sac_params)
    raw_training_params.pop("wandb_entity", None)
    raw_training_params.pop("network_factory", None)
    allowed_train_args = set(inspect.signature(sac.train).parameters.keys())
    training_params = {
        k: v for k, v in raw_training_params.items() if k in allowed_train_args
    }
    dropped_train_args = sorted(
        set(raw_training_params.keys()) - set(training_params.keys())
    )
    if dropped_train_args:
        print(f"Ignoring unsupported SAC params: {dropped_train_args}")

    if _VISION.value:
        raise ValueError(f"not implement.")
    else:
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

    if _VISION.value:
        env = wrapper.wrap_for_brax_training(
            env,
            vision=True,
            num_vision_envs=env_cfg.vision_config.render_batch_size,
            episode_length=sac_params.episode_length,
            action_repeat=sac_params.action_repeat,
            randomization_fn=training_params.get("randomization_fn"),
        )

    num_eval_envs = (
        sac_params.num_envs
        if _VISION.value
        else sac_params.get("num_eval_envs", 128)
    )

    if "num_eval_envs" in training_params:
        del training_params["num_eval_envs"]
    print(sac.train)

    robot_config = _resolve_robot_config(_ENV_NAME.value)

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

    # Prepare for evaluation
    render_env = (
        None if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg)
    )

    jit_render_reset = jax.jit(render_env.reset)
    jit_render_step = jax.jit(render_env.step)

    rng = jax.random.PRNGKey(123)
    rng, render_reset_rng = jax.random.split(rng)

    video_path = logdir / "videos"
    video_path.mkdir(parents=True, exist_ok=True)

    policy_path = logdir / "policies"
    policy_path.mkdir(parents=True, exist_ok=True)
    if _SAVE_BUFFER_DATA.value == True:
        buffer_data_path = logdir / "buffer_data"
        buffer_data_path.mkdir(parents=True, exist_ok=True)
    else:
        buffer_data_path = None

    # Progress function for logging
    def progress(num_steps, metrics, sac_ts, make_inference_fn, render_reset_rng):

        # Log to Weights & Biases
        if use_wandb:
            wandb.log(metrics, step=num_steps)

        # Log to TensorBoard
        if _USE_TB.value and not _PLAY_ONLY.value:
            for key, value in metrics.items():
                writer.add_scalar(key, value, num_steps)
            writer.flush()

        print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")

        if _RENDER.value:
            os.environ["MUJOCO_GL"] = "osmesa"

            render_inference_fn = make_inference_fn((sac_ts.normalizer_params, sac_ts.policy_params), deterministic=True)
            jit_render_inference_fn = jax.jit(render_inference_fn)

            rng, render_reset_rng = jax.random.split(render_reset_rng)
            render_state = jit_render_reset(render_reset_rng)
            render_state0 = (
                jax.tree_util.tree_map(lambda x: x[0], render_state) if _VISION.value else render_state
            )
            render_rollout = [render_state0]

            # Run render rollout
            for _ in range(env_cfg.episode_length):
                act_rng, rng = jax.random.split(rng)

                ctrl, _ = jit_render_inference_fn(render_state.obs, act_rng)
                render_state = jit_render_step(render_state, ctrl)
                render_state0 = (
                    jax.tree_util.tree_map(lambda x: x[0], render_state)
                    if _VISION.value
                    else render_state
                )
                render_rollout.append(render_state0)
                if render_state0.done:
                    break

            # Render and save the rollout
            render_every = 2
            fps = 1.0 / eval_env.dt / render_every
            print(f"FPS for rendering: {fps}")

            render_traj = render_rollout[::render_every]

            scene_option = mujoco.MjvOption()
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

            render_frames = render_env.render(
                render_traj, camera="track", height=480, width=640, scene_option=scene_option
            )
            media.write_video(video_path / f"rollout{num_steps}.mp4", render_frames, fps=fps)

            print("Rollout video saved as: ", video_path / f"rollout{num_steps}.mp4")

        with open(policy_path / f"policy{num_steps}.pkl", 'wb') as f:
            dill.dump(sac_ts, f)
        return True, render_reset_rng

    # Load evaluation environment
    eval_env = (
        None if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg)
    )

    # Train or load the model
    _ = train_fn(  # pylint: disable=no-value-for-parameter
        environment=env,
        progress_fn=progress,
        eval_env=None if _VISION.value else eval_env,
        buffer_data_path=buffer_data_path,

    )

if __name__ == "__main__":
    app.run(main)
