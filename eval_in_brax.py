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
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

from jax import config as jax_config
jax_config.update("jax_enable_x64", False)

import time
from pathlib import Path
import argparse
import jax
from lift_utils import running_statistics
import sys
sys.modules['brax.training.acme.running_statistics'] = running_statistics
import dill

from policy_pretrain import sac_networks
from brax.io import html


def _build_sac_network(observation_size, action_size):
    normalize_fn = running_statistics.normalize
    return sac_networks.make_sac_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        policy_hidden_layer_sizes=(512, 256, 128),
        q_hidden_layer_sizes=(1024, 512, 256),
        activation='swish',
    )


def _looks_like_normalizer(obj) -> bool:
    if hasattr(obj, "mean") and hasattr(obj, "std"):
        return True
    if isinstance(obj, dict) and "mean" in obj and "std" in obj:
        return True
    return False


def _unpack_policy_state(sac_ts):
    if hasattr(sac_ts, "policy_params") and hasattr(sac_ts, "normalizer_params"):
        return sac_ts.normalizer_params, sac_ts.policy_params
    if isinstance(sac_ts, dict):
        if "normalizer_params" in sac_ts and "policy_params" in sac_ts:
            return sac_ts["normalizer_params"], sac_ts["policy_params"]
        if "processor_params" in sac_ts and "policy_params" in sac_ts:
            return sac_ts["processor_params"], sac_ts["policy_params"]
    if isinstance(sac_ts, (list, tuple)) and len(sac_ts) == 2:
        first, second = sac_ts
        if _looks_like_normalizer(first) and not _looks_like_normalizer(second):
            return first, second
        if _looks_like_normalizer(second) and not _looks_like_normalizer(first):
            return second, first
        return first, second
    raise ValueError(
        "Unsupported policy format. Expected TrainingState or (normalizer, policy_params)."
    )


# ---------- Common env kwargs base ----------
COMMON_KWARGS = dict(
    policy_repeat=10,
    initial_yaw_range=(-0.0, 0.0),
    contact_time_const=0.02,
    contact_damping_ratio=1.0,
    friction_range=(0.6, 0.6),
    ground_roll_range=(0.0, 0.0),
    ground_pitch_range=(0.0, 0.0),
    joint_damping_perc_range=(1.0, 1.0),
    joint_gain_range=(1.0, 1.0),
    link_mass_perc_range=(1.0, 1.0),
    forces_in_q_coords=True,
    vel_x_command=1.5,
)



def build_env(env_name: str):
    'Instantiate the chosen Brax environment with merged kwargs.'
    env_name = env_name.lower()
    if env_name == "g1":
        from brax.envs.g1_lowdim_joystick import G1LowDimJoystick
        kwargs = {**COMMON_KWARGS}
        env = G1LowDimJoystick(backend="generalized", **kwargs)
    elif env_name == "t1":
        from brax.envs.t1_lowdim_sim_joystick import T1LowDimSimJoystick
        kwargs = {**COMMON_KWARGS}
        env = T1LowDimSimJoystick(backend="generalized", **kwargs)
    else:
        raise ValueError("env must be one of {'g1','t1'}")
    return env


def _select_health_obs(obs):
    if isinstance(obs, dict):
        if "wm_state" in obs:
            return obs["wm_state"]
        if "privileged_state" in obs:
            return obs["privileged_state"]
    return obs


def _is_healthy(env, obs) -> bool:
    if not hasattr(env, "_is_healthy"):
        return True
    return bool(jax.device_get(env._is_healthy(obs)))


def evaluate_and_render(
    env,
    sac_network,
    processor_params,
    policy_params,
    max_steps=1000,
    render_height=500,
    out_name="evaluation_render.html",
):
    @jax.jit
    def jit_step(state, action):
        return env.step(state, action)

    @jax.jit
    def jit_reset(key):
        return env.reset(key)

    key = jax.random.PRNGKey(0)
    state = jit_reset(key)
    states = []
    rew = 0.0
    start_time = time.time()

    for i in range(1, max_steps + 1):
        logits = sac_network.policy_network.apply(
            processor_params, policy_params, state.obs
        )
        action = sac_network.parametric_action_distribution.mode(logits)

        state = jit_step(state, action)
        rew += state.reward
        states.append(state.pipeline_state)

        if not _is_healthy(env, _select_health_obs(state.obs)):
            break

        if i % 10 == 0:
            now = time.time()
            print(f"Step {i}")
            print(f"Time elapsed: {now - start_time:.2f}s")
            start_time = now

    print("Total reward:", float(jax.device_get(rew)))

    render_html = html.render(env.sys.replace(dt=env.dt), states, height=render_height)
    output_path = Path(__file__).parent / out_name
    with open(output_path, "w") as f:
        f.write(render_html)
    print(f"Render saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and render policy in Brax envs.")
    parser.add_argument(
        "--env",
        type=str,
        default="t1",
        choices=["t1", "g1"],
        help="Which environment to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/step_40000_rew_104.pkl",
        help="Path to a dill-pickled TrainingState or (normalizer, policy_params) tuple.",
    )
    parser.add_argument("--max_steps", type=int, default=1000, help="Max simulation steps.")
    parser.add_argument("--height", type=int, default=500, help="Render viewport height.")
    parser.add_argument("--out", type=str, default="", help="Output HTML filename.")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            f"(Set with --model path/to/file.pkl)"
        )
    with open(args.model, "rb") as f:
        sac_ts = dill.load(f)
        print("Loaded model:", type(sac_ts))

    processor_params, policy_params = _unpack_policy_state(sac_ts)
    env = build_env(args.env)
    sac_network = _build_sac_network(env.observation_size, env.action_size)

    out_name = args.out or (f"evaluation_render_{args.env}.html")
    evaluate_and_render(
        env,
        sac_network,
        processor_params,
        policy_params,
        max_steps=args.max_steps,
        render_height=args.height,
        out_name=out_name,
    )


# CUDA_VISIBLE_DEVICES=0 python eval_in_brax.py --env t1 --model models/policy45009000.pkl
if __name__ == "__main__":
    main()
