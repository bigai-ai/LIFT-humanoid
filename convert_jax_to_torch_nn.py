import re
from pathlib import Path

import dill
import numpy as np
import torch

from lift_utils.checkpoint import jax2torch_with_norm

DEFAULT_SAC_TRAINING_STATE_PATH = (
    "logs/"
    "models/"
    "T1LowDimJoystickRoughTerrain_policy110119000.pkl"
)
DEFAULT_ACTIVATION = "swish"
DEFAULT_NUM_STEPS = 110119000


def _infer_step_from_path(path: str) -> int | None:
    match = re.search(r"policy(\d+)\.pkl$", Path(path).name)
    if match:
        return int(match.group(1))
    return None


def _default_save_dir(sac_training_state_path: str) -> Path:
    return Path(sac_training_state_path).parent / "torchscript"


def convert_policy(
    sac_training_state_path: str = DEFAULT_SAC_TRAINING_STATE_PATH,
    save_dir: str | None = None,
    step: int | None = None,
    activation: str = DEFAULT_ACTIVATION,
    transmit: bool = False,
    robot_ip: str = "10.1.128.227",
) -> str:
    if step is None:
        step = _infer_step_from_path(sac_training_state_path) or DEFAULT_NUM_STEPS
    if save_dir is None:
        save_dir = str(_default_save_dir(sac_training_state_path))

    with open(sac_training_state_path, "rb") as f:
        sac_ts = dill.load(f)

    mean_np = np.array(sac_ts.normalizer_params.mean["state"], copy=True)
    std_np = np.array(sac_ts.normalizer_params.std["state"], copy=True)

    jax2torch_with_norm(
        sac_ts.policy_params["params"],
        mean_np,
        std_np,
        save_dir,
        str(step),
        activation,
        transmit,
        robot_ip,
    )

    return str(Path(save_dir) / f"policy_{step}.pt")


def load_torch_policy(
    sac_training_state_path: str = DEFAULT_SAC_TRAINING_STATE_PATH,
    save_dir: str | None = None,
    step: int | None = None,
    activation: str = DEFAULT_ACTIVATION,
    force_convert: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[torch.jit.ScriptModule, str]:
    if step is None:
        step = _infer_step_from_path(sac_training_state_path) or DEFAULT_NUM_STEPS
    if save_dir is None:
        save_dir = str(_default_save_dir(sac_training_state_path))

    policy_path = Path(save_dir) / f"policy_{step}.pt"
    if force_convert or not policy_path.exists():
        policy_path = Path(
            convert_policy(
                sac_training_state_path=sac_training_state_path,
                save_dir=save_dir,
                step=step,
                activation=activation,
            )
        )

    policy = torch.jit.load(str(policy_path), map_location=device)
    policy.eval()
    return policy, str(policy_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sac_training_state_path",
        type=str,
        default=DEFAULT_SAC_TRAINING_STATE_PATH,
        help="Path to the SAC policy .pkl file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save TorchScript policy.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Policy step used in output filename (policy_<step>.pt).",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=DEFAULT_ACTIVATION,
        help="Activation used by the actor network.",
    )
    parser.add_argument("--transmit", action="store_true", help="Send to robot IP.")
    parser.add_argument("--robot_ip", type=str, default="10.1.128.227")
    args = parser.parse_args()

    policy_path = convert_policy(
        sac_training_state_path=args.sac_training_state_path,
        save_dir=args.save_dir,
        step=args.step,
        activation=args.activation,
        transmit=args.transmit,
        robot_ip=args.robot_ip,
    )
    print(policy_path)
