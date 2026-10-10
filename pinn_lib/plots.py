"""Matplotlib figure builders. Every function returns a Figure (no saving / showing)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})


def plot_1d_solutions(x, u_ref, preds, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(x, u_ref, "k-", lw=2.5, label="exact")
    for label, u in preds.items():
        a1.plot(x, u, "--", lw=1.5, label=label)
        a2.semilogy(x, np.abs(u - u_ref) + 1e-16, lw=1.2, label=label)
    a1.set(xlabel="x", ylabel="u", title=f"{title}: solution")
    a2.set(xlabel="x", ylabel="|u - u_exact|", title=f"{title}: absolute error")
    a1.legend()
    a2.legend()
    fig.tight_layout()
    return fig


def plot_field_panels(panels, extent, xlabel, ylabel, suptitle):
    """Side-by-side 2D fields (arrays indexed [i_x, i_y]); the last panel is an error map (log colour scale)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.9))
    axes = np.atleast_1d(axes)
    for i, (ax, (title, A)) in enumerate(zip(axes, panels)):
        A = np.asarray(A, dtype=np.float64)
        if i == len(panels) - 1:
            pos = A[np.isfinite(A) & (A > 0)]
            vmax = pos.max() if pos.size else 1.0
            vmin = max(pos.min(), vmax * 1e-5) if pos.size else 1e-6
            im = ax.imshow(np.clip(A, vmin, None).T, origin="lower", extent=extent, aspect="auto",
                           cmap="magma", norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)))
        else:
            im = ax.imshow(A.T, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_loss_curves(h, title):
    """Loss components, test error and (if present) the start of L-BFGS for one training history."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k in sorted(h):
        if k.startswith("loss_") and k != "loss_total":
            ax.semilogy(h["step"], h[k], lw=1.2, label=k[5:])
    if h.get("rel_l2"):
        ax.semilogy(h["step"], h["rel_l2"], "k--", lw=1.5, label="test rel. L2")
    if "lbfgs" in h.get("phase", []):
        ax.axvline(h["step"][h["phase"].index("lbfgs")], color="gray", ls=":", label="L-BFGS start")
    ax.set(xlabel="iteration", ylabel="value", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_cost_accuracy(df, title):
    """Runtime vs. relative L2 error, PINN vs classical, one marker per results row."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    style = {"PINN": ("tab:red", "o"), "classical": ("tab:blue", "s")}
    for kind, (color, mk) in style.items():
        d = df[(df["kind"] == kind) & df["rel_l2"].notna() & df["runtime_s"].notna() & (df["runtime_s"] > 0)]
        if len(d):
            ax.scatter(d["runtime_s"], d["rel_l2"], c=color, marker=mk, s=45, label=kind, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="runtime [s]", ylabel="relative L2 error", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bars(labels, means, stds, ylabel, title, log=False, color="tab:blue"):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    pos = np.arange(len(labels))
    ax.bar(pos, means, yerr=np.nan_to_num(stds), capsize=4, color=color)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set(ylabel=ylabel, title=title)
    fig.tight_layout()
    return fig
