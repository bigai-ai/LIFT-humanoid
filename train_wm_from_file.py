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
"""Train a PPO agent using JAX on the specified environment."""

from datetime import datetime
import functools
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"  # use no more than 90% of GPU memory
import jax
# TODO: check the performance of X64
jax.config.update('jax_default_matmul_precision', 'highest')
jax.config.update("jax_enable_x64", True)
import warnings
from absl import app
from absl import flags
from absl import logging
from lift_configs import finetune_sac_config, pretrain_wm_config
from world_model import pretrain_wm as sac_wm
from etils import epath
from brax.envs.g1_lowdim_joystick import G1LowDimJoystick
from brax.envs.t1_lowdim_sim_joystick import T1LowDimSimJoystick
from brax.envs.t1_lowdim_real_joystick import T1LowDimRealJoystick
from brax.robots.g1.utils import g1Utils
from brax.robots.booster.utils_sim import BoosterUtils as T1SimUtils
from brax.robots.booster.utils_real import BoosterUtils as T1RealUtils
from tensorboardX import SummaryWriter
import wandb
from mujoco_playground import registry
import dill
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "True"
os.environ["MUJOCO_GL"] = "egl"

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
    "G1LowDimJoystickFlatTerrain",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)

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
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DATA_PATH = flags.DEFINE_string(
    "data_path", None, "Path to load data of sac policy"
)
_SEED = flags.DEFINE_integer("seed", 0, "Random seed")
_POLICY_WAIT_INTERVAL = flags.DEFINE_float(
    "policy_wait_interval",
    5.0,
    "Seconds between scans of policies/ when waiting for policy*.pkl.",
)
_POLICY_WAIT_TIMEOUT = flags.DEFINE_integer(
    "policy_wait_timeout",
    0,
    "Max seconds to wait for first policy*.pkl. <=0 means wait forever.",
)

def main(argv):
    """Run training and evaluation for the specified environment."""

    del argv
    wm_params = pretrain_wm_config(_ENV_NAME.value)
    wandb_entity = getattr(wm_params, "wandb_entity", None)
    if _WANDB_ENTITY.present:
        wandb_entity = _WANDB_ENTITY.value
    if _SEED.present:
        wm_params.seed = _SEED.value
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
    if _USE_WANDB.value and not _PLAY_ONLY.value:
        wandb_kwargs = {"project": "LIFT_wm_pretrain", "name": exp_name}
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        wandb.init(**wandb_kwargs)
        wandb.config.update({"env_name": _ENV_NAME.value})

    # Initialize TensorBoard if required
    if _USE_TB.value and not _PLAY_ONLY.value:
        writer = SummaryWriter(logdir)

    # Set up checkpoint directory
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    training_params = dict(wm_params)
    if "wandb_entity" in training_params:
        del training_params["wandb_entity"]

    finetune_cfg = finetune_sac_config(_ENV_NAME.value)
    env_kwargs = finetune_cfg.finetune_env_config.to_dict()
    env_dict = {
        'G1LowDimJoystickFlatTerrain': G1LowDimJoystick,
        'G1LowDimJoystickRoughTerrain': G1LowDimJoystick,
        'T1LowDimSimFinetuneJoystickFlatTerrain': T1LowDimSimJoystick,
        'T1LowDimSimFinetuneJoystickRoughTerrain': T1LowDimSimJoystick,
        'T1LowDimRealFinetuneJoystickFlatTerrain': T1LowDimRealJoystick,
        'T1LowDimRealFinetuneJoystickRoughTerrain': T1LowDimRealJoystick,
    }
    robot_config_dict = {
        'G1LowDimJoystickFlatTerrain': g1Utils,
        'G1LowDimJoystickRoughTerrain': g1Utils,
        'T1LowDimSimFinetuneJoystickFlatTerrain': T1SimUtils,
        'T1LowDimSimFinetuneJoystickRoughTerrain': T1SimUtils,
        'T1LowDimRealFinetuneJoystickFlatTerrain': T1RealUtils,
        'T1LowDimRealFinetuneJoystickRoughTerrain': T1RealUtils,
    }
    if _ENV_NAME.value not in env_dict:
        raise ValueError(
            f"Unsupported env_name {_ENV_NAME.value} for model_use_env. "
            f"Expected one of {sorted(env_dict.keys())}."
        )
    env_fn = functools.partial(env_dict[_ENV_NAME.value], backend='generalized')
    env_fn = add_kwargs_to_fn(env_fn, **env_kwargs)
    model_use_env = env_fn()

    data_path = _DATA_PATH.value
    if not data_path:
        raise ValueError("--data_path is required and must point to a log dir with policies/ and buffer_data/.")
    data_path = os.path.abspath(os.path.expanduser(data_path))
    training_data_path = os.path.join(data_path, "buffer_data")
    if not os.path.isdir(training_data_path):
        raise FileNotFoundError(f"Missing buffer_data directory: {training_data_path}")


    wm_path = logdir / "wm_states"
    wm_path.mkdir(parents=True, exist_ok=True)

    # Progress function for logging
    def progress(num_steps, metrics, wm_ts, render_reset_rng):

        # Log to Weights & Biases
        if _USE_WANDB.value and not _PLAY_ONLY.value:
            wandb.log(metrics, step=num_steps)

        # Log to TensorBoard
        if _USE_TB.value and not _PLAY_ONLY.value:
            for key, value in metrics.items():
                writer.add_scalar(key, value, num_steps)
            writer.flush()
        with open(wm_path / f"wm_state{num_steps}.pkl", 'wb') as f:
            dill.dump(wm_ts, f)

        return render_reset_rng

    train_fn = functools.partial(
        sac_wm.train,
        **training_params,
        model_use_env=model_use_env,
        robot_config=robot_config_dict.get(_ENV_NAME.value),
        data_path=training_data_path,
        progress_fn=progress,
    )

    train_fn() # pylint: disable=no-value-for-parameter

def add_kwargs_to_fn(partial_fn, **kwargs):
    """add the kwargs to the passed in partial function"""
    for param in kwargs:
        partial_fn.keywords[param] = kwargs[param]
    return partial_fn


if __name__ == "__main__":
    app.run(main)
