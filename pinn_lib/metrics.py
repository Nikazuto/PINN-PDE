"""Error metrics (identical to the notebook)."""
import numpy as np


def rel_l2(pred, ref) -> float:
    pred, ref = np.asarray(pred, dtype=np.float64).ravel(), np.asarray(ref, dtype=np.float64).ravel()
    ok = np.isfinite(ref) & np.isfinite(pred)
    return float(np.linalg.norm(pred[ok] - ref[ok]) / max(np.linalg.norm(ref[ok]), 1e-30))


def max_abs_err(pred, ref) -> float:
    d = np.abs(np.asarray(pred, dtype=np.float64).ravel() - np.asarray(ref, dtype=np.float64).ravel())
    return float(np.nanmax(d))
