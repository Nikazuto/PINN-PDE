"""Headless check of the numerical library (no Streamlit / PyTorch needed):  python tests/smoke_test.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from pinn_lib import pdes, solvers, plots, data
from pinn_lib.metrics import rel_l2

u = np.interp(pdes.P1_X_TEST, *solvers.solve_poisson_fd(200)); e = rel_l2(u, pdes.P1_U_TEST); assert e < 1e-3, e
x, U = solvers.heat_crank_nicolson(100, pdes.H_T); e = rel_l2(solvers.interp_columns(x, U, pdes.H_X), pdes.H_U); assert e < 1e-3, e
x, U = solvers.heat_ftcs(50, pdes.H_T); e = rel_l2(solvers.interp_columns(x, U, pdes.H_X), pdes.H_U); assert e < 1e-2, e
_, U = solvers.burgers_spectral_etdrk4(512, pdes.B_T); ref = solvers.spectral_to_test_grid(U, 512)
assert np.abs(ref[[0, -1], :]).max() < 1e-6
x, U = solvers.burgers_fd_cnab2(513, pdes.B_T); e = rel_l2(solvers.interp_columns(x, U, pdes.B_X), ref); assert e < 0.2, e
XX, YY, Ud, ins = solvers.fd_poisson_masked(100, pdes.disk_inside, pdes.disk_rhs, pdes.disk_exact, solvers.radial_projection)
assert rel_l2(Ud[ins], pdes.disk_exact(XX, YY)[ins]) < 0.05
XX, YY, Ul, ins = solvers.fd_poisson_masked(100, pdes.lshape_inside, lambda a, b: np.zeros_like(a), pdes.lshape_exact)
assert rel_l2(Ul[ins], pdes.lshape_exact(XX, YY)[ins]) < 0.05
plots.plot_field_panels([("a", pdes.H_U), ("b", pdes.H_U), ("err", np.abs(pdes.H_U) + 1e-9)], [0, 1, 0, 1], "x", "t", "t")
print("smoke test OK")
