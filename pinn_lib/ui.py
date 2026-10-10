"""Streamlit helpers shared by all pages."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# make `import pinn_lib` work from pages/ regardless of how Streamlit sets sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from . import data  # noqa: E402


def page_setup(title: str, icon: str = "📈"):
    st.set_page_config(page_title=f"PINN vs classical: {title}", page_icon=icon, layout="wide")


def sidebar_results():
    """Sidebar controls for choosing the results table. Returns (DataFrame, label)."""
    st.sidebar.header("Results data")
    up = st.sidebar.file_uploader("Upload results (.json / .csv)", type=["json", "csv"],
                                  help="Optional. Use the results_*.json or results_*.csv file written by the notebook.")
    if up is not None:
        try:
            return data.parse_uploaded_results(up.name, up.getvalue()), f"uploaded: {up.name}"
        except Exception as exc:
            st.sidebar.error(f"Could not read {up.name}: {exc}")
    tags = data.available_tags()
    if not tags:
        return data.load_results("__none__"), "none"
    tag = st.sidebar.selectbox("Notebook run", tags, help="RUN_TAG of the notebook (quick/full + gpu/cpu)")
    return data.load_results(tag), tag


def fmt_sci(v) -> str:
    try:
        return "-" if pd.isna(v) else f"{v:.2e}"
    except Exception:
        return str(v)


def no_results_message():
    st.info("No results found yet. Put the notebook's `results_<RUN_TAG>.json` (or `.csv`) into "
            "`data/results/` and redeploy, or upload the file in the sidebar.")
