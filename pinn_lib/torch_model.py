"""Rebuild trained PINNs from the notebook's checkpoints and evaluate them (needs PyTorch, CPU is enough).

The checkpoint stores the weights, the training history and the TrainConfig, but not the network
architecture. The architecture is inferred from the weight shapes, and the hard-constraint output
transform from the run name (poisson_hard / heat_hard / burgers_hard_bc), exactly as in the notebook.
"""
import math
from typing import Callable, Dict, Optional

import numpy as np

from .pdes import heat_initial, burgers_initial

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:  # torch not installed: the pages fall back to the stored figures / results
    torch, nn, TORCH_OK = None, None, False


def torch_available() -> bool:
    return TORCH_OK


if TORCH_OK:
    class FourierFeatures(nn.Module):
        def __init__(self, in_dim: int, n_features: int, sigma: float, seed: int = 0):
            super().__init__()
            gen = torch.Generator().manual_seed(seed)
            self.register_buffer("B", torch.randn(in_dim, n_features, generator=gen) * sigma)
            self.out_dim = 2 * n_features

        def forward(self, z):
            proj = 2.0 * math.pi * z @ self.B
            return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    class MLP(nn.Module):
        def __init__(self, in_dim, out_dim=1, width=50, depth=4, fourier_features=0, fourier_sigma=1.0,
                     output_transform: Optional[Callable] = None):
            super().__init__()
            self.in_dim = in_dim
            self.register_buffer("lb", torch.zeros(1, in_dim))
            self.register_buffer("ub", torch.ones(1, in_dim))
            self.embed = FourierFeatures(in_dim, fourier_features, fourier_sigma) if fourier_features > 0 else None
            dims = [self.embed.out_dim if self.embed is not None else in_dim] + [width] * depth
            layers = []
            for d_in, d_out in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(d_in, d_out), nn.Tanh()]
            layers.append(nn.Linear(width, out_dim))
            self.net = nn.Sequential(*layers)
            self.output_transform = output_transform

        def forward(self, x):
            z = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
            if self.embed is not None:
                z = self.embed(z)
            raw = self.net(z)
            return raw if self.output_transform is None else self.output_transform(x, raw)


def problem_of(run_name: str) -> str:
    """Map a run name to its problem: poisson | heat | burgers | disk | lshape | inverse."""
    for prefix, prob in [("poisson", "poisson"), ("heat", "heat"), ("burgers", "burgers"), ("disk", "disk"),
                         ("lshape", "lshape"), ("inverse_heat", "inverse")]:
        if run_name.startswith(prefix):
            return prob
    return "unknown"


def _output_transform(run_name: str):
    if run_name.startswith("poisson_hard"):
        return lambda x, raw: x * (1.0 - x) * raw
    if run_name.startswith("heat_hard"):
        return lambda X, raw: heat_initial(X[:, 0:1], xp=torch) + X[:, 1:2] * X[:, 0:1] * (1.0 - X[:, 0:1]) * raw
    if run_name.startswith("burgers_hard_bc"):
        return lambda X, raw: burgers_initial(X[:, 0:1], xp=torch) + X[:, 1:2] * (1.0 - X[:, 0:1] ** 2) * raw
    return None


def load_checkpoint(path) -> dict:
    """Load a notebook checkpoint on CPU. Only load files you created yourself (pickle-based format)."""
    if not TORCH_OK:
        raise RuntimeError("PyTorch is not installed")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(run_name: str, state: dict):
    """Infer width/depth/Fourier features from the weights, build the MLP and load the weights."""
    sd = state["model"]
    idx = sorted(int(k.split(".")[1]) for k in sd if k.startswith("net.") and k.endswith(".weight"))
    first, last = sd[f"net.{idx[0]}.weight"], sd[f"net.{idx[-1]}.weight"]
    width, depth, out_dim = first.shape[0], len(idx) - 1, last.shape[0]
    if "embed.B" in sd:
        in_dim, n_feat = sd["embed.B"].shape
    else:
        in_dim, n_feat = first.shape[1], 0
    model = MLP(in_dim, out_dim, width, depth, fourier_features=n_feat, output_transform=_output_transform(run_name))
    model.load_state_dict(sd)
    return model.eval()


@torch.no_grad() if TORCH_OK else (lambda f: f)
def predict(model, X: np.ndarray, batch: int = 20000) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(np.asarray(X[i:i + batch]), dtype=torch.float32)
        out.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out).reshape(-1)


def summarize(state: dict) -> Dict[str, object]:
    """Small metadata dict shown in the explorer."""
    h = state.get("history", {})
    ex = {k: v[-1] for k, v in h.items() if k.startswith("extra_") and v}
    return dict(status=state.get("status", "?"), runtime_s=float(state.get("elapsed", float("nan"))),
                n_collocation=int(state["X_col"].shape[0]) if "X_col" in state else None,
                final_test_rel_l2=(h.get("rel_l2") or [float("nan")])[-1],
                last_step=(h.get("step") or [0])[-1], extras=ex)
