import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

from pinn_lib import data, pdes, plots, torch_model, ui
from pinn_lib.metrics import max_abs_err, rel_l2

ui.page_setup("PINN explorer", "🧠")
st.title("PINN explorer")

if not torch_model.torch_available():
    st.error("PyTorch is not installed in this environment, so checkpoints cannot be loaded. "
             "Add `torch` to requirements.txt (see README). The other pages work without it.")
    st.stop()


@st.cache_resource(show_spinner="Loading checkpoint...")
def load_state(path_str, mtime):
    return torch_model.load_checkpoint(path_str)


@st.cache_data(show_spinner=False)
def reference_burgers(_ref_flag):
    ref = data.load_burgers_reference()
    if ref is None or ref.shape != pdes.B_XX.shape:
        from pinn_lib import solvers
        _, U = solvers.burgers_spectral_etdrk4(1024, pdes.B_T)
        return solvers.spectral_to_test_grid(U, 1024), "computed live (N=1024)"
    return ref, "stored notebook reference (N=2048)"


# ---------------- choose a checkpoint ----------------
st.sidebar.header("Checkpoint")
up = st.sidebar.file_uploader("Upload a checkpoint (.pt)", type=["pt"],
                              help="Only upload files you created yourself: checkpoints are pickle-based.")
tags = sorted({p.stem.split("__")[-1] for p in data.list_checkpoints()})
path, name = None, None
if up is not None:
    tmp = Path(tempfile.gettempdir()) / f"uploaded_{up.name}"
    tmp.write_bytes(up.getvalue())
    path, name = tmp, up.name.split("__")[0].removesuffix(".pt")
elif tags:
    tag = st.sidebar.selectbox("Notebook run", tags)
    cks = data.list_checkpoints(tag)
    path = st.sidebar.selectbox("Run", cks, format_func=lambda p: data.checkpoint_run_name(p))
    name = data.checkpoint_run_name(path)
else:
    st.info("No checkpoints found. Copy the notebook's `checkpoints/*.pt` into `data/checkpoints/` and redeploy, "
            "or upload one in the sidebar.")
    st.stop()

state = load_state(str(path), path.stat().st_mtime)
prob = torch_model.problem_of(name)
meta = torch_model.summarize(state)
model = torch_model.build_model(name, state)

st.subheader(name)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training time", f"{meta['runtime_s']:.0f} s")
c2.metric("Collocation points", meta["n_collocation"] if meta["n_collocation"] is not None else "-")
c3.metric("Test rel. L2 (last log)", f"{meta['final_test_rel_l2']:.2e}")
c4.metric("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
st.caption(f"Status: {meta['status']}")
for k, v in meta["extras"].items():
    st.caption(f"Learned parameter {k.replace('extra_', '')}: {v:.5f}" + (f" (true alpha = {pdes.HEAT_ALPHA})" if k == "extra_alpha" else ""))

# ---------------- prediction vs. reference ----------------
tab_pred, tab_hist = st.tabs(["Prediction vs. reference", "Training history"])

with tab_pred:
    if prob == "poisson":
        u = torch_model.predict(model, pdes.P1_X_TEST.reshape(-1, 1))
        st.metric("Relative L2 error (1001-point test grid)", f"{rel_l2(u, pdes.P1_U_TEST):.2e}")
        st.write(f"u(0) = {u[0]:+.2e}, u(1) = {u[-1]:+.2e}")
        st.pyplot(plots.plot_1d_solutions(pdes.P1_X_TEST, pdes.P1_U_TEST, {name: u}, "1D Poisson"), clear_figure=True)
    elif prob in ("heat", "inverse"):
        U = torch_model.predict(model, pdes.H_XT).reshape(pdes.H_XX.shape)
        st.metric("Relative L2 error (101 x 101 test grid)", f"{rel_l2(U, pdes.H_U):.2e}")
        st.pyplot(plots.plot_field_panels([("exact", pdes.H_U), ("PINN", U), ("|PINN - exact|", np.abs(U - pdes.H_U))],
                                          [0, 1, 0, 1], "x", "t", "Heat equation"), clear_figure=True)
    elif prob == "burgers":
        ref, ref_label = reference_burgers(0)
        U = torch_model.predict(model, pdes.B_XT).reshape(pdes.B_XX.shape)
        st.metric("Relative L2 error (513 x 101 test grid)", f"{rel_l2(U, ref):.2e}")
        st.caption(f"Reference: {ref_label}")
        it = st.slider("Time index (t = index/100)", 0, 100, 100)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, xlim in zip(axes, [(-1, 1), (-0.12, 0.12)]):
            ax.plot(pdes.B_X, ref[:, it], "k-", lw=2.5, label="reference")
            ax.plot(pdes.B_X, U[:, it], "r--", lw=1.6, label="PINN")
            ax.set(xlim=xlim, xlabel="x", ylabel="u", title=f"t = {pdes.B_T[it]:.2f}" + (" (zoom on the shock)" if xlim[0] > -1 else ""))
            ax.legend()
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.pyplot(plots.plot_field_panels([("reference", ref), ("PINN", U), ("|PINN - reference|", np.abs(U - ref))],
                                          [-1, 1, 0, 1], "x", "t", "Viscous Burgers"), clear_figure=True)
    elif prob in ("disk", "lshape"):
        mask = pdes.DISK_MASK if prob == "disk" else pdes.LSHAPE_MASK
        U_ref = pdes.DISK_U if prob == "disk" else pdes.LSHAPE_U
        U = np.full(pdes.S_XX.shape, np.nan)
        U[mask] = torch_model.predict(model, pdes.S_XY[mask.ravel()])
        st.metric("Relative L2 error (201 x 201 test grid, inside the domain)", f"{rel_l2(U[mask], U_ref[mask]):.2e}")
        st.pyplot(plots.plot_field_panels([("exact", U_ref), ("PINN", U), ("|PINN - exact|", np.abs(U - U_ref))],
                                          [-1, 1, -1, 1], "x", "y", "Disk" if prob == "disk" else "L-shape"),
                  clear_figure=True)
    else:
        st.warning(f"Don't know how to evaluate runs named '{name}'.")

with tab_hist:
    h = state.get("history", {})
    if not h.get("step"):
        st.info("This checkpoint has no logged history.")
    else:
        st.pyplot(plots.plot_loss_curves(h, name), clear_figure=True)
        if h.get("extra_alpha"):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.plot(h["step"], h["extra_alpha"], lw=1.6, label="alpha estimate")
            ax.axhline(pdes.HEAT_ALPHA, color="k", ls="--", label="true alpha")
            ax.set(yscale="log", xlabel="iteration", ylabel="alpha", title="Convergence of the diffusivity")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
