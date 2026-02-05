import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

from lift_utils.tcp_server import send_checkpoint_until_success

ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'swish': nn.SiLU,
}

class MLPWithNorm(nn.Module):
    """
    MLP with input normalization ((x-mean)/std).
    mean/std are stored as buffers in the model and exported with TorchScript.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, mean: torch.Tensor, std: torch.Tensor, activation_cls=nn.ReLU):
        super().__init__()
        # register buffers (not updated by optimizer, saved with state_dict / TorchScript)
        self.register_buffer("obs_mean", mean.clone().detach())
        self.register_buffer("obs_std", std.clone().detach())

        layers = []
        prev_dim = input_dim
        for h in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_cls())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize
        x = (x - self.obs_mean) / self.obs_std
        return self.net(x)

def infer_mlp_shape_from_jax_params(params: dict):
    """Infer MLP structure from JAX params (supports hidden_0,... and optional output layer)."""
    hidden_layer_keys = sorted(
        [k for k in params.keys() if k.startswith("hidden_")],
        key=lambda x: int(x.split("_")[1])
    )
    if len(hidden_layer_keys) == 0:
        raise ValueError("No hidden_* layers found in policy params.")

    input_dim = params[hidden_layer_keys[0]]['kernel'].shape[0]
    hidden_dims = [params[k]['kernel'].shape[1] for k in hidden_layer_keys]

    if "output" in params and 'kernel' in params["output"]:
        output_dim = params["output"]["kernel"].shape[1]
        has_output = True
    else:
        # Fallback: no separate output layer, treat last hidden as output
        output_dim = hidden_dims[-1]
        has_output = False

    return hidden_layer_keys, input_dim, hidden_dims, output_dim, has_output

def load_normalizer_from_ts(sac_ts):
    """Extract mean/std from brax normalizer_params (state only)."""
    mean_np = np.array(sac_ts.normalizer_params.mean['state'], copy=True)
    std_np = np.array(sac_ts.normalizer_params.std['state'], copy=True)
    mean_t = torch.tensor(mean_np, dtype=torch.float32).view(1, -1)
    std_t = torch.tensor(std_np, dtype=torch.float32).view(1, -1)
    return mean_t, std_t

def jax2torch_with_norm(params: dict, mean_np: np.ndarray, std_np: np.ndarray,
                        save_dir: str, step: str, activate_actor: str, transmit: bool, robot_ip = "10.1.128.227"):
    """
    Convert JAX weights to PyTorch, embed the normalizer (mean/std), and export to TorchScript.
    The scripted model expects raw (unnormalized) observations as input.
    """
    mean_t = torch.tensor(mean_np, dtype=torch.float32).view(1, -1)
    std_t = torch.tensor(std_np, dtype=torch.float32).view(1, -1)
    hidden_layer_keys, input_dim, hidden_dims, output_dim, has_output = infer_mlp_shape_from_jax_params(params)

    # guard: ensure mean/std dims match input
    if mean_t.shape[-1] != input_dim or std_t.shape[-1] != input_dim:
        raise ValueError(f"Normalizer shape mismatch: mean/std dim={mean_t.shape[-1]}/{std_t.shape[-1]} vs input_dim={input_dim}")

    device = torch.device("cpu")
    act_cls_actor = ACTIVATION_MAP.get(activate_actor.lower(), nn.ReLU)
    model = MLPWithNorm(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        mean=mean_t.to(device),
        std=std_t.to(device),
        activation_cls=act_cls_actor
    ).to(device)

    # copy weights
    # note: model.net interleaves Linear/Act; only take Linear here
    torch_linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]

    # 1) each hidden_i
    for i, k in enumerate(hidden_layer_keys):
        w = torch.tensor(np.array(params[k]['kernel']), dtype=torch.float32).t()   # [out,in] for torch
        b = torch.tensor(np.array(params[k]['bias']), dtype=torch.float32)
        torch_linear_layers[i].weight.data.copy_(w)
        torch_linear_layers[i].bias.data.copy_(b)

    # 2) output layer (if a separate output exists)
    if has_output:
        w = torch.tensor(np.array(params["output"]["kernel"]), dtype=torch.float32).t()
        b = torch.tensor(np.array(params["output"]["bias"]), dtype=torch.float32)
        torch_linear_layers[-1].weight.data.copy_(w)
        torch_linear_layers[-1].bias.data.copy_(b)

    # save TorchScript (trace with raw obs input so it can consume raw obs directly)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    try:
        scripted = torch.jit.script(model)
    except Exception:
        example = torch.randn(1, input_dim, device=device)
        scripted = torch.jit.trace(model, example)

    out_path = os.path.join(save_dir, f"policy_{step}.pt")
    scripted.save(out_path)
    if transmit:
        send_checkpoint_until_success(
            ip=robot_ip,
            port=9001,
            file_path=out_path,
        )
    print(f"Saved TorchScript with embedded normalizer to: {out_path}")
    return scripted
