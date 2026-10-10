"""Loading the notebook's outputs (results table, reference solution, checkpoints, figures).

Expected layout (the same sub-folders the notebook writes to Google Drive):

    data/results/results_<RUN_TAG>.json | .csv     e.g. results_quick_gpu.json
    data/results/burgers_reference_N2048.npz       optional (otherwise computed live, coarser)
    data/checkpoints/<run>__<RUN_TAG>.pt           e.g. burgers_adam_lbfgs_s1234__quick_gpu.pt
    data/figures/*_<RUN_TAG>.png                   optional pre-rendered notebook figures

Override the root with the environment variable PINN_DATA_DIR.
"""
import json
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

COLUMNS = ["experiment", "method", "kind", "rel_l2", "max_err", "runtime_s", "status", "notes"]


def data_dir() -> Path:
    return Path(os.environ.get("PINN_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


def results_dir() -> Path:
    return data_dir() / "results"


def checkpoint_dir() -> Path:
    return data_dir() / "checkpoints"


def figure_dir() -> Path:
    return data_dir() / "figures"


# ---------------- results table ----------------
def available_tags() -> List[str]:
    """RUN_TAGs found in results/ (e.g. 'quick_gpu', 'full_gpu')."""
    tags = set()
    for p in results_dir().glob("results_*.*"):
        if p.suffix in (".json", ".csv"):
            tags.add(p.stem[len("results_"):])
    return sorted(tags)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan if c in ("rel_l2", "max_err", "runtime_s") else ""
    for c in ("rel_l2", "max_err", "runtime_s"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("experiment", "method", "kind", "status", "notes"):
        df[c] = df[c].fillna("").astype(str)
    return df[COLUMNS].reset_index(drop=True)


def load_results(tag: str) -> pd.DataFrame:
    """Load results_<tag>.json (preferred) or .csv from the results folder."""
    js, csv = results_dir() / f"results_{tag}.json", results_dir() / f"results_{tag}.csv"
    if js.exists():
        with open(js) as f:
            return _normalise(pd.DataFrame(json.load(f)))
    if csv.exists():
        return _normalise(pd.read_csv(csv))
    return _normalise(pd.DataFrame(columns=COLUMNS))


def parse_uploaded_results(name: str, raw: bytes) -> pd.DataFrame:
    if name.lower().endswith(".json"):
        return _normalise(pd.DataFrame(json.loads(raw.decode("utf-8"))))
    import io
    return _normalise(pd.read_csv(io.BytesIO(raw)))


def split_failures(df: pd.DataFrame):
    """(good rows, failed rows). Failed steps are recorded by the notebook as method '[step]' with status 'failed'."""
    failed = df["status"].str.startswith("failed")
    return df[~failed].copy(), df[failed].copy()


# ---------------- findings (same logic as the notebook's auto_findings) ----------------
EXPERIMENT_TITLES = {
    "1_poisson": "1D Poisson",
    "2_heat": "1D heat equation",
    "3_burgers": "Viscous Burgers",
    "4a_disk": "Poisson on the unit disk",
    "4b_lshape": "Laplace on the L-shape",
    "5_inverse": "Inverse problem (alpha)",
    "6_ablation": "Burgers ablation",
}


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Best PINN vs. best classical run for each forward experiment, plus how much faster the cheapest
    classical run that is at least as accurate as the best PINN is."""
    ok, _ = split_failures(df)
    ok = ok[ok["rel_l2"].notna()]
    rows = []
    for exp in ["1_poisson", "2_heat", "3_burgers", "4a_disk", "4b_lshape"]:
        p = ok[(ok.experiment == exp) & (ok.kind == "PINN")]
        c = ok[(ok.experiment == exp) & (ok.kind == "classical")]
        if p.empty or c.empty:
            continue
        bp, bc = p.loc[p.rel_l2.idxmin()], c.loc[c.rel_l2.idxmin()]
        match = c[c.rel_l2 <= bp.rel_l2].sort_values("runtime_s")
        speed = (bp.runtime_s / max(match.iloc[0].runtime_s, 1e-9)) if len(match) else np.nan
        rows.append(dict(experiment=EXPERIMENT_TITLES.get(exp, exp), best_pinn=bp.method, pinn_rel_l2=bp.rel_l2,
                         pinn_runtime_s=bp.runtime_s, best_classical=bc.method, classical_rel_l2=bc.rel_l2,
                         classical_runtime_s=bc.runtime_s,
                         classical_speedup_at_equal_accuracy=speed))
    return pd.DataFrame(rows)


# ---------------- ablation / inverse parsing ----------------
def ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-variant mean/std over seeds from rows 'VARIANT (seed N)' of experiment 6_ablation."""
    ok, _ = split_failures(df)
    a = ok[(ok.experiment == "6_ablation") & ok.rel_l2.notna()].copy()
    if a.empty:
        return pd.DataFrame()
    m = a["method"].str.extract(r"^(.*) \(seed (\d+)\)$")
    a["variant"], a["seed"] = m[0], m[1]
    a = a[a["variant"].notna()]
    g = a.groupby("variant", sort=False).agg(rel_l2_mean=("rel_l2", "mean"), rel_l2_std=("rel_l2", "std"),
                                              runtime_mean=("runtime_s", "mean"), runtime_std=("runtime_s", "std"),
                                              n_seeds=("seed", "count"))
    return g.reset_index()


def inverse_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (noise, seed) with the alpha estimates of the PINN and the classical fit.

    Parsed from the 'notes' column ('alpha=0.05012 (rel err 0.24%) ...') of experiment 5_inverse.
    """
    ok, _ = split_failures(df)
    d = ok[ok.experiment == "5_inverse"].copy()
    if d.empty:
        return pd.DataFrame()
    pat = re.compile(r"\(noise (\d+(?:\.\d+)?)%, seed (\d+)\)")
    rows = {}
    for _, r in d.iterrows():
        m, a = pat.search(r["method"]), re.search(r"alpha=([0-9.eE+-]+)", r["notes"])
        if not (m and a):
            continue
        key = (float(m.group(1)), int(m.group(2)))
        rec = rows.setdefault(key, dict(noise_pct=key[0], seed=key[1]))
        if r["kind"] == "PINN":
            rec.update(alpha_pinn=float(a.group(1)), field_rel_l2=r["rel_l2"], t_pinn=r["runtime_s"])
        else:
            rec.update(alpha_classical=float(a.group(1)), t_classical=r["runtime_s"])
    out = pd.DataFrame(rows.values())
    return out.sort_values(["noise_pct", "seed"]).reset_index(drop=True) if len(out) else out


# ---------------- reference solution ----------------
def load_burgers_reference() -> Optional[np.ndarray]:
    p = results_dir() / "burgers_reference_N2048.npz"
    if p.exists():
        z = np.load(p)
        return z["U"]
    return None


# ---------------- checkpoints & figures ----------------
def list_checkpoints(tag: Optional[str] = None) -> List[Path]:
    paths = sorted(checkpoint_dir().glob("*.pt"))
    if tag:
        paths = [p for p in paths if p.stem.endswith(f"__{tag}")]
    return paths


def checkpoint_run_name(path: Path) -> str:
    return path.stem.split("__")[0]


def list_figures(tag: Optional[str] = None) -> List[Path]:
    paths = sorted(figure_dir().glob("*.png"))
    if tag:
        paths = [p for p in paths if p.stem.endswith(f"_{tag}")]
    return paths
