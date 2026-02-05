import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.distributions import Normal

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from convert_jax_to_torch_nn import (
    DEFAULT_ACTIVATION,
    DEFAULT_SAC_TRAINING_STATE_PATH,
    load_torch_policy,
)

def setup_gl_backend(render: bool, gl: str | None):
    """Set MuJoCo GL backend based on args; must run before importing mujoco."""
    if gl:                    # manual override
        os.environ["MUJOCO_GL"] = gl
    elif render:              # need windowed rendering
        os.environ["MUJOCO_GL"] = "glfw"
    else:                     # headless only
        os.environ.setdefault("MUJOCO_GL", "egl")

    # keep PYOPENGL_PLATFORM consistent with MUJOCO_GL to avoid EGL errors
    if os.environ["MUJOCO_GL"] == "egl":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    else:
        # for glfw / glx, remove PYOPENGL_PLATFORM to use default behavior
        os.environ.pop("PYOPENGL_PLATFORM", None)

class minmaxnormalizer():
    def __init__(self):
        self.device='cpu'
        # Normalization limits
        self.obs_limit_min = torch.full((1, 87), -1.0, device=self.device)  # Min values
        self.obs_limit_max = torch.full((1, 87), 1.0, device=self.device)   # Max values

        self.q_idxs = [i for i in range(11, 23, 1)]
        self.obs_limit_min[:, self.q_idxs] = torch.tensor([-1.8, -0.3, -1.0, 0.0, -0.87, -0.44, -1.8, -1.57, -1.0, 0.0, -0.87, -0.44], device=self.device) - 0.25
        self.obs_limit_max[:, self.q_idxs] = torch.tensor([1.57, 1.57, 1.0, 2.34, 0.35, 0.44, 1.57, 0.3, 1.0, 2.34, 0.35, 0.44], device=self.device) + 0.25

        self.wm_q_idxs = [i for i in range(60, 72, 1)]
        self.obs_limit_min[:, self.wm_q_idxs] = torch.tensor([-1.8, -0.3, -1.0, 0.0, -0.87, -0.44, -1.8, -1.57, -1.0, 0.0, -0.87, -0.44], device=self.device) - 0.25
        self.obs_limit_max[:, self.wm_q_idxs] = torch.tensor([1.57, 1.57, 1.0, 2.34, 0.35, 0.44, 1.57, 0.3, 1.0, 2.34, 0.35, 0.44], device=self.device) + 0.25

        self.forward_vel_idx = 57
        self.obs_limit_min[:, self.forward_vel_idx] = -0.2
        self.obs_limit_max[:, self.forward_vel_idx] = 2.5

        self.y_vel_idx = 58
        self.obs_limit_min[:, self.y_vel_idx] = -0.5
        self.obs_limit_max[:, self.y_vel_idx] = 0.5

        self.z_vel_idx = 59
        self.obs_limit_min[:, self.z_vel_idx] = -0.5
        self.obs_limit_max[:, self.z_vel_idx] = 0.5

        self.roll_rate_idx = 3
        self.pitch_rate_idx = 4
        self.turn_rate_idx = 5
        self.rpy_rate_idxs = [i for i in range(self.roll_rate_idx, self.turn_rate_idx+1, 1)]
        self.obs_limit_min[:, self.rpy_rate_idxs] = torch.tensor([-1.5, -1.5, -1.5], device=self.device)
        self.obs_limit_max[:, self.rpy_rate_idxs] = torch.tensor([1.5, 1.5, 1.5], device=self.device)

        self.wm_roll_rate_idx = 47
        self.wm_pitch_rate_idx = 48
        self.wm_turn_rate_idx = 49
        self.wm_rpy_rate_idxs = [i for i in range(self.wm_roll_rate_idx, self.wm_turn_rate_idx+1, 1)]
        self.obs_limit_min[:, self.wm_rpy_rate_idxs] = torch.tensor([-1.5, -1.5, -1.5], device=self.device)
        self.obs_limit_max[:, self.wm_rpy_rate_idxs] = torch.tensor([1.5, 1.5, 1.5], device=self.device)

        self.qd_idxs = [i for i in range(23, 35, 1)]
        self.obs_limit_min[:, self.qd_idxs] = -20.0
        self.obs_limit_max[:, self.qd_idxs] = 20

        self.wm_qd_idxs = [i for i in range(72, 84, 1)]
        self.obs_limit_min[:, self.wm_qd_idxs] = -20.0
        self.obs_limit_max[:, self.wm_qd_idxs] = 20

        self.wm_height_idx = 84
        self.obs_limit_min[:, self.wm_height_idx] = 0.0
        self.obs_limit_max[:, self.wm_height_idx] = 0.8


        self.wm_gravity_idxs = [i for i in range(50, 53, 1)]
        self.wm_quat_idxs = [i for i in range(53, 57, 1)]
        self.wm_base_vel_idxs = [i for i in range(57, 60, 1)]
        self.wm_gait_process_idx = 85
        self.wm_gait_frequency_idx = 86
        
        self.priv_rpy_rate_idxs = self.wm_rpy_rate_idxs
        self.priv_gravity_idxs = self.wm_gravity_idxs
        self.priv_base_lin_vel_idxs =  [i for i in range(53, 56, 1)]
        self.priv_global_ang_vel_idxs =  [i for i in range(56, 59, 1)]
        self.priv_q_idxs = [i for i in range(59, 71, 1)]
        self.priv_qd_idxs = [i for i in range(71, 83, 1)]
        self.priv_height_idx = 83


    def normalize_obs(self, obs):
        # Normalize observation
        normalized_obs = 2 * (obs - self.obs_limit_min) / (self.obs_limit_max - self.obs_limit_min) - 1
        return normalized_obs

    def denormalize_obs(self, normalize_obs):
        obs = (normalize_obs + 1)*(self.obs_limit_max - self.obs_limit_min)/2 + self.obs_limit_min
        return obs
        

class TanhBijector:
    """Tanh Bijector."""

    def forward(self, x):
        return torch.tanh(x)

    def inverse(self, y):
        # Clamping the input to avoid numerical issues with arctanh
        return torch.arctanh(torch.clamp(y, -0.999999, 0.999999))

    def forward_log_det_jacobian(self, x):
        # Computing the log of the absolute value of the Jacobian determinant
        return 2. * (torch.log(torch.tensor(2.0)) - x - F.softplus(-2. * x))


class NormalTanhDistribution:
    """Normal distribution followed by tanh."""

    def __init__(self, min_std=0.001, max_std=None):
        self.min_std = min_std
        self.max_std = max_std
        self.postprocessor = TanhBijector()

    def create_dist(self, parameters):
        loc, scale = torch.chunk(parameters, 2, dim=-1)
        if self.max_std is None:
            scale = F.softplus(scale) + self.min_std
        else:
            scale = torch.sigmoid(scale)
            scale = self.min_std + (self.max_std - self.min_std) * scale
        return Normal(loc, scale)

    def sample(self, parameters):
        dist = self.create_dist(parameters)
        return self.postprocessor.forward(dist.rsample())

    def mode(self, parameters):
        dist = self.create_dist(parameters)
        return self.postprocessor.forward(dist.mean)

def infer_max_steps(cfg, fallback=500):
    try:
        episode_length_s = cfg["rewards"]["episode_length_s"]
        control_dt = cfg["sim"]["dt"] * cfg["control"]["decimation"]
        return max(int(episode_length_s / control_dt), 1)
    except Exception:
        return fallback


def run_mujoco(policy, cfg, render: bool = False, lin_vel=0.6, max_steps: int | None = None, collect_data: bool = False):
    import mujoco  # Must set MUJOCO_GL before import.
    if render:
        import mujoco.viewer

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)

    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            raise ValueError(
                f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined"
            )
    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)
    normalizer = NormalTanhDistribution()

    if max_steps is None:
        max_steps = infer_max_steps(cfg)
    rewards_cfg = cfg.get("rewards", {})
    terminate_height = rewards_cfg.get("terminate_height", 0.3)

    actions = np.zeros((cfg["env"]["num_actions"]), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_process = 0.0
    gait_frequency = np.average(cfg["commands"]["gait_frequency"])
    lin_vel_y = ang_vel_yaw = 0.0
    lin_vel_x = lin_vel
    it = 0
    step = 0
    num_envs = 1

    data_buffers = None
    obs_minmax_normalizer = None
    if collect_data:
        obs_minmax_normalizer = minmaxnormalizer()
        data_dict = {
            "state": [],
            "priv_state": [],
            "wm_state": [],
            "actions": [],
            "torques": [],
            "contacts": [],
            "rewards": [],
            "timestamps": [],
        }
        data_buffers = [{key: [] for key in data_dict} for _ in range(num_envs)]

    if render:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        viewer.cam.elevation = -20

    start_x = float(mj_data.qpos.astype(np.float32)[0])
    sum_lin_vel_x = 0.0
    sum_lin_vel_y = 0.0
    sum_ang_vel_yaw = 0.0
    sim_steps = 0
    termination = "timeout"

    while True:
        base_pos = mj_data.qpos.astype(np.float32)[:3]
        dof_pos = mj_data.qpos.astype(np.float32)[7:]
        dof_vel = mj_data.qvel.astype(np.float32)[6:]
        quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
        quat_wxyz = mj_data.sensor("orientation").data.astype(np.float32)

        base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
        base_lin_vel = mj_data.sensor("linear-velocity").data.astype(np.float32)

        sum_lin_vel_x += float(base_lin_vel[0])
        sum_lin_vel_y += float(base_lin_vel[1])
        sum_ang_vel_yaw += float(base_ang_vel[2])
        sim_steps += 1

        projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        ang_vel_global = rotate(quat, base_ang_vel)
        if it % cfg["control"]["decimation"] == 0:
            if collect_data and step != 0:
                data_buffers[0]["state"].append(state)
                data_buffers[0]["wm_state"].append(wm_state[0])
                data_buffers[0]["priv_state"].append(priv_state)
                data_buffers[0]["actions"].append(actions.copy())
                data_buffers[0]["torques"].append(torque)
                data_buffers[0]["contacts"].append([0.0, 0.0])
                data_buffers[0]["rewards"].append(0)
                data_buffers[0]["timestamps"].append(0)

            state = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
            state[0:3] = projected_gravity
            state[3:6] = base_ang_vel
            state[6] = lin_vel_x
            state[7] = lin_vel_y
            state[8] = ang_vel_yaw
            state[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
            state[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
            state[11:23] = dof_pos - default_dof_pos 
            state[23:35] = dof_vel * 0.1
            state[35:47] = actions

            obs_torch = torch.tensor(state, dtype=torch.float32)

            if collect_data:
                wm_state = np.zeros(87)
                wm_state[:state.shape[0]] = state
                wm_state[obs_minmax_normalizer.wm_rpy_rate_idxs] = base_ang_vel
                wm_state[obs_minmax_normalizer.wm_gravity_idxs] = projected_gravity
                wm_state[obs_minmax_normalizer.wm_quat_idxs] = quat_wxyz
                wm_state[obs_minmax_normalizer.wm_base_vel_idxs] = base_lin_vel
                wm_state[obs_minmax_normalizer.wm_q_idxs] = dof_pos
                wm_state[obs_minmax_normalizer.wm_qd_idxs] = dof_vel
                wm_state[obs_minmax_normalizer.wm_height_idx] = base_pos[2]
                wm_state[obs_minmax_normalizer.wm_gait_process_idx] = gait_process
                wm_state[obs_minmax_normalizer.wm_gait_frequency_idx] = gait_frequency
                wm_state = torch.tensor(wm_state)
                wm_state = obs_minmax_normalizer.normalize_obs(wm_state)

                priv_state = np.zeros(84)
                priv_state[:state.shape[0]] = state
                priv_state[obs_minmax_normalizer.priv_rpy_rate_idxs] = base_ang_vel
                priv_state[obs_minmax_normalizer.priv_gravity_idxs] = projected_gravity
                priv_state[obs_minmax_normalizer.priv_base_lin_vel_idxs] = base_lin_vel
                priv_state[obs_minmax_normalizer.priv_global_ang_vel_idxs] = ang_vel_global
                priv_state[obs_minmax_normalizer.priv_q_idxs] = dof_pos
                priv_state[obs_minmax_normalizer.priv_qd_idxs] = dof_vel
                priv_state[obs_minmax_normalizer.priv_height_idx] = base_pos[2]
                priv_state = torch.tensor(priv_state)

            with torch.no_grad():
                dist = policy(obs_torch.unsqueeze(0))
                actions = normalizer.mode(dist).detach().cpu().numpy().squeeze(0)

            actions = np.clip(
                actions,
                -cfg["normalization"]["clip_actions"],
                cfg["normalization"]["clip_actions"],
            )
            dof_targets[:] = default_dof_pos + cfg["control"]["action_scale"] * actions
            step += 1

        torque = np.clip(
            dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
            mj_model.actuator_ctrlrange[:, 0],
            mj_model.actuator_ctrlrange[:, 1],
        )
        mj_data.ctrl = torque
        mujoco.mj_step(mj_model, mj_data)
        it += 1
        gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)

        if base_pos[2] < terminate_height:
            termination = "fell"
            break
        if step >= max_steps:
            termination = "timeout"
            break
        if render:
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()

    if render:
        viewer.close()

    episode_time_s = sim_steps * cfg["sim"]["dt"]
    distance_m = float(mj_data.qpos.astype(np.float32)[0] - start_x)
    mean_lin_vel_x = sum_lin_vel_x / max(sim_steps, 1)
    mean_lin_vel_y = sum_lin_vel_y / max(sim_steps, 1)
    mean_ang_vel_yaw = sum_ang_vel_yaw / max(sim_steps, 1)

    info = {
        "steps": step,
        "sim_steps": sim_steps,
        "episode_time_s": episode_time_s,
        "distance_m": distance_m,
        "mean_lin_vel_x": mean_lin_vel_x,
        "mean_lin_vel_y": mean_lin_vel_y,
        "mean_ang_vel_yaw": mean_ang_vel_yaw,
        "command_lin_vel": lin_vel,
        "tracking_error": abs(mean_lin_vel_x - lin_vel),
        "termination": termination,
        "final_height": float(base_pos[2]),
    }

    return step, data_buffers, info

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c



def rotate(quat, vec):
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat.
  """
  s, u = quat[-1], quat[:-1]
  r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
  r = r + 2 * s * np.cross(u, vec)
  return r

#  python -u play_mujoco_simp_train.py --lin_vel=1.0 --policy_path=mujoco_env/policy_110119000.pt --render
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="mujoco_env/T1.yaml",
        help="Path to task config yaml.",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default=None,
        help="Override MuJoCo xml path.",
    )
    parser.add_argument("--lin_vel", type=float, default=0.6, help="Commanded forward velocity.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max control steps per episode.",
    )
    parser.add_argument("--render", action="store_true", help="Open GUI render.")
    parser.add_argument(
        "--policy_path",
        type=str,
        default=None,
        help="Existing TorchScript policy path.",
    )
    parser.add_argument(
        "--sac_training_state_path",
        type=str,
        default=DEFAULT_SAC_TRAINING_STATE_PATH,
        help="SAC .pkl path for conversion.",
    )
    parser.add_argument(
        "--policy_save_dir",
        type=str,
        default=None,
        help="Directory for converted TorchScript policy.",
    )
    parser.add_argument(
        "--policy_step",
        type=int,
        default=None,
        help="Policy step for conversion output.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=DEFAULT_ACTIVATION,
        help="Actor activation.",
    )
    parser.add_argument(
        "--force_convert",
        action="store_true",
        help="Force reconversion to TorchScript.",
    )

    args = parser.parse_args()
    setup_gl_backend(render=args.render, gl=None)

    with open(args.yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    if args.xml_path:
        cfg["asset"]["mujoco_file"] = args.xml_path

    if args.policy_path:
        policy = torch.jit.load(args.policy_path, map_location="cpu")
        policy.eval()
        policy_path = args.policy_path
    else:
        policy, policy_path = load_torch_policy(
            sac_training_state_path=args.sac_training_state_path,
            save_dir=args.policy_save_dir,
            step=args.policy_step,
            activation=args.activation,
            force_convert=args.force_convert,
            device="cpu",
        )

    print(f"Loaded policy: {policy_path}")
    max_steps = args.max_steps if args.max_steps is not None else infer_max_steps(cfg)

    results = []
    for ep in range(args.num_episodes):
        _, _, info = run_mujoco(
            policy,
            cfg,
            render=args.render,
            lin_vel=args.lin_vel,
            max_steps=max_steps,
            collect_data=False,
        )
        results.append(info)
        print(
            f"Episode {ep + 1}/{args.num_episodes} steps={info['steps']} term={info['termination']} "
            f"time={info['episode_time_s']:.2f}s dist={info['distance_m']:.2f}m "
            f"mean_vx={info['mean_lin_vel_x']:.2f}m/s track_err={info['tracking_error']:.2f}m/s"
        )

    if results:
        steps_arr = np.array([r["steps"] for r in results], dtype=np.float32)
        time_arr = np.array([r["episode_time_s"] for r in results], dtype=np.float32)
        dist_arr = np.array([r["distance_m"] for r in results], dtype=np.float32)
        vx_arr = np.array([r["mean_lin_vel_x"] for r in results], dtype=np.float32)
        err_arr = np.array([r["tracking_error"] for r in results], dtype=np.float32)
        fall_rate = float(np.mean([r["termination"] == "fell" for r in results]))
        print("Summary")
        print(
            f"  episodes={len(results)} mean_steps={steps_arr.mean():.1f} "
            f"mean_time={time_arr.mean():.2f}s mean_dist={dist_arr.mean():.2f}m "
            f"mean_vx={vx_arr.mean():.2f}m/s mean_track_err={err_arr.mean():.2f}m/s "
            f"fall_rate={fall_rate * 100:.1f}%"
        )
