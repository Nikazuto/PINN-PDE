import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from pinn_lib import data, pdes, plots, solvers, ui
from pinn_lib.metrics import max_abs_err, rel_l2

ui.page_setup("Classical solvers", "🧮")
st.title("Classical solvers, live")
st.caption("These run in the app, in NumPy/SciPy, using the same code and the same test grids as the notebook.")


# ---------------- cached computations ----------------
@st.cache_data(show_spinner=False)
def run_poisson(n):
    (x, u), t = solvers.timed(lambda: solvers.solve_poisson_fd(n), repeats=3)
    u_test = np.interp(pdes.P1_X_TEST, x, u)
    return u_test, t


@st.cache_data(show_spinner=False)
def run_heat(method, nx):
    solver = solvers.heat_ftcs if method == "FTCS" else solvers.heat_crank_nicolson
    (x, U), t = solvers.timed(lambda: solver(nx, pdes.H_T), repeats=1)
    return solvers.interp_columns(x, U, pdes.H_X), t


@st.cache_data(show_spinner=False)
def run_burgers_fd(nx):
    (x, U), t = solvers.timed(lambda: solvers.burgers_fd_cnab2(nx, pdes.B_T), repeats=1)
    return solvers.interp_columns(x, U, pdes.B_X), t


@st.cache_data(show_spinner="Computing a spectral reference solution (a few seconds)...")
def burgers_reference_live(N):
    _, U = solvers.burgers_spectral_etdrk4(N, pdes.B_T)
    return solvers.spectral_to_test_grid(U, N)


@st.cache_data(show_spinner=False)
def run_shape(domain, n):
    if domain == "Unit disk":
        args = (pdes.disk_inside, pdes.disk_rhs, pdes.disk_exact, solvers.radial_projection)
        exact = pdes.disk_exact
    else:
        args = (pdes.lshape_inside, lambda x, y: np.zeros_like(x), pdes.lshape_exact, None)
        exact = pdes.lshape_exact
    (XX, YY, U, ins), t = solvers.timed(lambda: solvers.fd_poisson_masked(n, *args), repeats=1)
    U_ex = np.where(ins, exact(XX, YY), np.nan)
    return U, U_ex, ins, t


tab_p, tab_h, tab_b, tab_s = st.tabs(["1D Poisson", "Heat equation", "Burgers", "Disk and L-shape"])

# ---------------- Poisson ----------------
with tab_p:
    st.markdown(r"Second-order central differences for $u''=f$, $u(0)=u(1)=0$; exact $u=\sin 2\pi x+0.1\sin 8\pi x$.")
    n = st.select_slider("Number of intervals n", [10, 25, 50, 100, 200, 500, 1000, 5000], value=50, key="p_n")
    u_test, t = run_poisson(n)
    c1, c2, c3 = st.columns(3)
    c1.metric("Relative L2 error", f"{rel_l2(u_test, pdes.P1_U_TEST):.2e}")
    c2.metric("Max abs error", f"{max_abs_err(u_test, pdes.P1_U_TEST):.2e}")
    c3.metric("Runtime (best of 3)", f"{t * 1e3:.3f} ms")
    st.pyplot(plots.plot_1d_solutions(pdes.P1_X_TEST, pdes.P1_U_TEST, {f"FD n={n}": u_test}, "1D Poisson"),
              clear_figure=True)
    if st.checkbox("Convergence table", key="p_conv"):
        rows = []
        for k in [10, 25, 50, 100, 200, 500, 1000, 5000]:
            u, tk = run_poisson(k)
            rows.append(dict(n=k, rel_l2=rel_l2(u, pdes.P1_U_TEST), runtime_ms=tk * 1e3))
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("The error drops 4x each time n doubles (second order).")

# ---------------- Heat ----------------
with tab_h:
    st.markdown(r"$u_t=0.05\,u_{xx}$ on $(0,1)^2$ with exact solution; scored on the $101\times101$ test grid.")
    c1, c2 = st.columns(2)
    method = c1.radio("Method", ["Crank-Nicolson", "FTCS"], horizontal=True)
    nx = c2.select_slider("Spatial intervals nx", [25, 50, 100, 200, 400], value=100, key="h_nx")
    if method == "FTCS" and nx > 200:
        st.warning("FTCS needs dt ~ dx^2, so very fine grids are slow; limited to nx <= 200.")
        nx = 200
    U, t = run_heat(method, nx)
    c1, c2, c3 = st.columns(3)
    c1.metric("Relative L2 error", f"{rel_l2(U, pdes.H_U):.2e}")
    c2.metric("Max abs error", f"{max_abs_err(U, pdes.H_U):.2e}")
    c3.metric("Runtime", f"{t * 1e3:.1f} ms")
    st.pyplot(plots.plot_field_panels([("exact u(x,t)", pdes.H_U), (method, U), ("|error|", np.abs(U - pdes.H_U))],
                                      [0, 1, 0, 1], "x", "t", "Heat equation"), clear_figure=True)
    if st.checkbox("Convergence table", key="h_conv"):
        rows = []
        for m_, grids in [("FTCS", [25, 50, 100, 200]), ("Crank-Nicolson", [25, 50, 100, 200, 400])]:
            for g in grids:
                Uk, tk = run_heat(m_, g)
                rows.append(dict(method=m_, nx=g, rel_l2=rel_l2(Uk, pdes.H_U), runtime_ms=tk * 1e3))
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# ---------------- Burgers ----------------
with tab_b:
    st.markdown(r"Viscous Burgers $u_t+uu_x=\nu u_{xx}$, $\nu=0.01/\pi$, $u(x,0)=-\sin\pi x$. "
                "Finite differences (Crank-Nicolson + Adams-Bashforth 2) vs. a Fourier spectral reference.")
    ref = data.load_burgers_reference()
    if ref is not None and ref.shape == pdes.B_XX.shape:
        ref_label = "notebook reference (N=2048, from data/results)"
    else:
        N = st.select_slider("No stored reference found; compute a spectral one with N =", [512, 1024, 2048], value=1024)
        ref = burgers_reference_live(N)
        ref_label = f"computed live (N={N})"
    st.caption(f"Reference: {ref_label}")
    nx = st.select_slider("FD grid points nx", [129, 257, 513, 1025, 2049], value=257, key="b_nx")
    U, t = run_burgers_fd(nx)
    c1, c2, c3 = st.columns(3)
    c1.metric("Relative L2 error", f"{rel_l2(U, ref):.2e}")
    c2.metric("Max abs error", f"{max_abs_err(U, ref):.2e}")
    c3.metric("Runtime", f"{t:.2f} s")
    it = st.slider("Time index (t = index/100)", 0, 100, 100)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, xlim in zip(axes, [(-1, 1), (-0.12, 0.12)]):
        ax.plot(pdes.B_X, ref[:, it], "k-", lw=2.5, label="reference")
        ax.plot(pdes.B_X, U[:, it], "b--", lw=1.5, label=f"FD nx={nx}")
        ax.set(xlim=xlim, xlabel="x", ylabel="u", title=f"t = {pdes.B_T[it]:.2f}" + (" (zoom on the shock)" if xlim[0] > -1 else ""))
        ax.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.pyplot(plots.plot_field_panels([("reference", ref), (f"FD nx={nx}", U), ("|error|", np.abs(U - ref))],
                                      [-1, 1, 0, 1], "x", "t", "Burgers"), clear_figure=True)
    if st.checkbox("Convergence table", key="b_conv"):
        rows = []
        for g in [129, 257, 513, 1025, 2049]:
            Ug, tg = run_burgers_fd(g)
            rows.append(dict(nx=g, rel_l2=rel_l2(Ug, ref), runtime_s=tg))
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# ---------------- Disk / L-shape ----------------
with tab_s:
    st.markdown("5-point finite differences on a masked Cartesian grid. On the disk, a staircase boundary "
                "(first order near the wall); on the L-shape, the exact solution has an unbounded gradient at the "
                "re-entrant corner, so the error peaks there.")
    c1, c2 = st.columns(2)
    domain = c1.radio("Domain", ["Unit disk", "L-shape"], horizontal=True)
    n = c2.select_slider("Grid intervals n (h = 2/n)", [20, 40, 100, 200], value=100, key="s_n")
    U, U_ex, ins, t = run_shape(domain, n)
    err = np.abs(U - U_ex)
    c1, c2, c3 = st.columns(3)
    c1.metric("Relative L2 (nodal)", f"{rel_l2(U[ins], U_ex[ins]):.2e}")
    c2.metric("Max abs error", f"{max_abs_err(U[ins], U_ex[ins]):.2e}")
    c3.metric("Runtime", f"{t * 1e3:.0f} ms")
    st.pyplot(plots.plot_field_panels([("exact", U_ex), ("FD", U), ("|FD - exact|", err)],
                                      [-1, 1, -1, 1], "x", "y", domain), clear_figure=True)
    if st.checkbox("Convergence table", key="s_conv"):
        rows = []
        for g in [20, 40, 100, 200]:
            Ug, Ue, ig, tg = run_shape(domain, g)
            rows.append(dict(n=g, rel_l2=rel_l2(Ug[ig], Ue[ig]), runtime_ms=tg * 1e3))
        st.dataframe(pd.DataFrame(rows), hide_index=True)
