import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from pinn_lib import data, plots, ui

ui.page_setup("Results", "📊")
df, label = ui.sidebar_results()
st.title("Results")

if df.empty:
    ui.no_results_message()
    st.stop()

st.caption(f"Source: {label}")
ok, failed = data.split_failures(df)

tab_head, tab_table, tab_cost, tab_fig = st.tabs(["Headline", "Full table", "Accuracy vs. cost", "Notebook figures"])

with tab_head:
    head = data.headline_table(ok)
    if head.empty:
        st.info("No experiment has both PINN and classical rows.")
    else:
        show = head.copy()
        for c in ("pinn_rel_l2", "classical_rel_l2"):
            show[c] = show[c].map(ui.fmt_sci)
        show["pinn_runtime_s"] = show["pinn_runtime_s"].map(lambda v: f"{v:.1f}")
        show["classical_runtime_s"] = show["classical_runtime_s"].map(lambda v: f"{v:.3g}")
        show["classical_speedup_at_equal_accuracy"] = show["classical_speedup_at_equal_accuracy"].map(
            lambda v: "none reached PINN error" if v != v else f"{v:,.0f}x")
        st.dataframe(show, hide_index=True)
        st.caption("Per experiment: best PINN, best classical method, and how much faster the cheapest classical run "
                   "that is at least as accurate as the best PINN was. Treat PINN numbers as budget-limited: runs that "
                   "hit their time cap are not converged (see the status column).")
    capped = ok[ok["status"].str.contains("time cap", na=False)]
    if len(capped):
        st.warning(f"{len(capped)} run(s) hit their wall-clock cap, so their error is budget-limited: "
                   + ", ".join(capped["method"].head(8)) + (" ..." if len(capped) > 8 else ""))

with tab_table:
    exps = sorted(ok["experiment"].unique())
    c1, c2 = st.columns(2)
    sel_exp = c1.multiselect("Experiment", exps, default=exps)
    sel_kind = c2.multiselect("Kind", sorted(ok["kind"].unique()), default=sorted(ok["kind"].unique()))
    view = ok[ok["experiment"].isin(sel_exp) & ok["kind"].isin(sel_kind)].sort_values(["experiment", "kind", "rel_l2"])
    st.dataframe(view, hide_index=True)
    st.download_button("Download table as CSV", view.to_csv(index=False).encode(), "results_filtered.csv", "text/csv")
    if len(failed):
        with st.expander(f"{len(failed)} failed step(s)"):
            st.dataframe(failed[["experiment", "method", "notes"]], hide_index=True)

with tab_cost:
    forward = [e for e in ok["experiment"].unique() if e not in ("5_inverse", "6_ablation")]
    exp = st.selectbox("Experiment", forward or sorted(ok["experiment"].unique()),
                       format_func=lambda e: data.EXPERIMENT_TITLES.get(e, e))
    d = ok[ok["experiment"] == exp]
    st.pyplot(plots.plot_cost_accuracy(d, data.EXPERIMENT_TITLES.get(exp, exp)), clear_figure=True)
    st.caption("Lower-left is better. Classical points are different grid sizes; PINN points are separate training runs.")

with tab_fig:
    tag = label if label in data.available_tags() else None
    figs = data.list_figures(tag)
    if not figs:
        st.info("No pre-rendered figures found. Copy the notebook's `figures/*.png` into `data/figures/` "
                "to see them here.")
    for p in figs:
        st.image(str(p), caption=p.stem)
