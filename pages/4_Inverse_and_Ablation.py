import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from pinn_lib import data, pdes, plots, ui

ui.page_setup("Inverse and ablation", "🔬")
df, label = ui.sidebar_results()
st.title("Inverse problem and ablation study")

if df.empty:
    ui.no_results_message()
    st.stop()
st.caption(f"Source: {label}")

tab_inv, tab_abl = st.tabs(["Inverse problem (alpha)", "Burgers ablation"])

# ---------------- inverse ----------------
with tab_inv:
    st.markdown(
        r"Recover the diffusivity $\alpha$ (true value 0.05) in $u_t=\alpha u_{xx}$ from noisy sparse data. "
        "The PINN is not told the initial or boundary conditions; the classical Crank-Nicolson least-squares fit is.")
    inv = data.inverse_table(df)
    if inv.empty:
        st.info("No inverse-problem rows found in this results file.")
    else:
        inv["err_pinn_pct"] = 100 * (inv["alpha_pinn"] - pdes.HEAT_ALPHA).abs() / pdes.HEAT_ALPHA
        inv["err_classical_pct"] = 100 * (inv["alpha_classical"] - pdes.HEAT_ALPHA).abs() / pdes.HEAT_ALPHA
        st.dataframe(inv, hide_index=True)
        g = inv.groupby("noise_pct")[["err_pinn_pct", "err_classical_pct"]].agg(["mean", "std"])
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for col, lab, mk in [("err_pinn_pct", "PINN (no IC/BC knowledge)", "o-"),
                             ("err_classical_pct", "CN fit (knows IC/BC)", "s--")]:
            ax.errorbar(g.index.values, g[(col, "mean")].values, yerr=np.nan_to_num(g[(col, "std")].values),
                        fmt=mk, capsize=4, label=lab)
        ax.set(yscale="log", xlabel="noise level [% of std(u)]", ylabel="relative error of alpha [%]",
               title="Recovered alpha vs. noise (mean +- std over seeds)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption("Zero noise does not mean zero error: optimisation and approximation leave an error floor. "
                   "With the full forward model known, the classical fit is usually at least as accurate and much faster; "
                   "the PINN's advantage is that it needs less prior knowledge.")

# ---------------- ablation ----------------
with tab_abl:
    st.markdown("Each variant changes exactly one ingredient of the baseline PINN (soft IC/BC, fixed weights, "
                "uniform collocation, Adam only).")
    abl = data.ablation_table(df)
    if abl.empty:
        st.info("No ablation rows found in this results file.")
    else:
        show = abl.copy()
        show["rel_l2_mean"] = show["rel_l2_mean"].map(ui.fmt_sci)
        show["rel_l2_std"] = show["rel_l2_std"].map(ui.fmt_sci)
        show["runtime_mean"] = show["runtime_mean"].map(lambda v: f"{v:.0f} s")
        show["runtime_std"] = show["runtime_std"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f} s")
        st.dataframe(show, hide_index=True)
        c1, c2 = st.columns(2)
        c1.pyplot(plots.plot_bars(abl["variant"], abl["rel_l2_mean"], abl["rel_l2_std"], "test relative L2 error",
                                  "Accuracy", log=True), clear_figure=True)
        c2.pyplot(plots.plot_bars(abl["variant"], abl["runtime_mean"], abl["runtime_std"], "training time [s]",
                                  "Cost", color="tab:orange"), clear_figure=True)
        if (abl["n_seeds"] < 2).any():
            st.warning("Some variants have a single seed (quick mode). Differences smaller than a factor of about 2 "
                       "are not significant; error bars are missing for those variants.")
        base = abl[abl["variant"].str.startswith("Baseline")]
        if len(base):
            b = float(base["rel_l2_mean"].iloc[0])
            ratio = abl[["variant", "rel_l2_mean", "runtime_mean"]].copy()
            ratio["error vs. baseline"] = (ratio["rel_l2_mean"] / b).map(lambda v: f"{v:.2f}x")
            st.caption("Error relative to the baseline (below 1x is better); compare with the cost panel before "
                       "concluding that a variant helps.")
            st.dataframe(ratio[["variant", "error vs. baseline"]], hide_index=True)
