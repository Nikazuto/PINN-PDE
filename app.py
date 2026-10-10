import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from pinn_lib import data, ui
from pinn_lib.torch_model import torch_available

ui.page_setup("Home")
df, label = ui.sidebar_results()

st.title("When do PINNs work?")
st.caption("Physics-informed neural networks vs. classical numerical solvers: an honest comparison")

st.markdown(
    """
This dashboard presents the results of the notebook **pinn_project.ipynb** and lets you re-run the classical
solvers and inspect the trained PINNs interactively.

| Page | What it does |
|---|---|
| **Results** | Full results table, accuracy-vs-cost plots, headline PINN vs. classical comparison, notebook figures |
| **Classical solvers** | Run the finite-difference / Crank-Nicolson / spectral solvers live and watch the error change with the grid |
| **PINN explorer** | Load a trained checkpoint, compare its prediction with the exact or reference solution, see its training curves |
| **Inverse and ablation** | Recovery of the diffusivity from noisy data, and the Burgers ablation study |
"""
)

st.subheader("Experiments")
st.markdown(
    "1. 1D Poisson (soft vs. hard boundary conditions)  \n"
    "2. 1D heat equation vs. FTCS and Crank-Nicolson  \n"
    "3. Viscous Burgers vs. finite differences, with a spectral reference  \n"
    "4. Irregular 2D domains: unit disk and L-shape (corner singularity)  \n"
    "5. Inverse problem: recover the diffusivity alpha from noisy sparse data  \n"
    "6. Burgers ablation: hard BCs, Fourier features, adaptive weights, RAR, L-BFGS"
)

st.subheader("Headline comparison")
if df.empty:
    ui.no_results_message()
else:
    st.caption(f"Source: {label}")
    head = data.headline_table(df)
    if head.empty:
        st.warning("The results file has no experiment with both PINN and classical rows yet.")
    else:
        cols = st.columns(len(head))
        for col, (_, r) in zip(cols, head.iterrows()):
            with col:
                st.metric(r["experiment"], f"PINN {r['pinn_rel_l2']:.1e}", f"classical {r['classical_rel_l2']:.1e}",
                          delta_color="off")
                sp = r["classical_speedup_at_equal_accuracy"]
                st.caption("No classical run matched the PINN's error." if sp != sp
                           else f"A classical run at least as accurate was {sp:,.0f}x faster.")
        st.caption("Relative L2 error of the best PINN run vs. the best classical run for each forward problem. "
                   "Speed-ups compare the PINN's training time with the cheapest classical run that is at least as accurate.")

st.subheader("Data and environment status")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Result files", len(data.available_tags()))
c2.metric("Checkpoints", len(data.list_checkpoints()))
c3.metric("Figures", len(data.list_figures()))
c4.metric("PyTorch", "available" if torch_available() else "not installed")
if not torch_available():
    st.caption("Without PyTorch the PINN explorer is disabled; everything else still works.")
