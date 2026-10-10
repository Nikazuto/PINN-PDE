"""Classical solvers (numpy / scipy only), copied from the notebook."""
import math
import time
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_banded
from scipy.optimize import minimize_scalar

from .pdes import (BURGERS_NU, B_X, HEAT_ALPHA, burgers_initial, heat_initial, poisson_rhs)


def timed(fn: Callable, repeats: int = 3):
    """Run fn() `repeats` times; return (result of the last call, best wall-clock time in s)."""
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def interp_columns(x_src, U_src, x_dst):
    """Linear interpolation in x of every column (time slice) of U_src onto x_dst."""
    return np.stack([np.interp(x_dst, x_src, U_src[:, j]) for j in range(U_src.shape[1])], axis=1)


# ---------------- Exp 1 ----------------
def solve_poisson_fd(n: int):
    x = np.linspace(0.0, 1.0, n + 1)
    h = 1.0 / n
    ab = np.zeros((3, n - 1))
    ab[0, 1:], ab[1, :], ab[2, :-1] = 1.0 / h ** 2, -2.0 / h ** 2, 1.0 / h ** 2
    u = np.zeros(n + 1)
    u[1:-1] = solve_banded((1, 1), ab, poisson_rhs(x[1:-1]))
    return x, u


# ---------------- Exp 2 ----------------
def heat_ftcs(nx: int, t_out, alpha: float = HEAT_ALPHA, r: float = 0.4):
    x = np.linspace(0.0, 1.0, nx + 1)
    dx = x[1] - x[0]
    u = heat_initial(x)
    u[0] = u[-1] = 0.0
    U = np.empty((nx + 1, len(t_out)))
    U[:, 0] = u
    dt_max = r * dx ** 2 / alpha
    for k in range(1, len(t_out)):
        span = t_out[k] - t_out[k - 1]
        m = int(np.ceil(span / dt_max - 1e-12))
        lam = alpha * (span / m) / dx ** 2
        for _ in range(m):
            u[1:-1] = u[1:-1] + lam * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        U[:, k] = u
    return x, U


def heat_crank_nicolson(nx: int, t_out, alpha: float = HEAT_ALPHA, dt_over_dx: float = 1.0):
    x = np.linspace(0.0, 1.0, nx + 1)
    dx = x[1] - x[0]
    u = heat_initial(x)
    u[0] = u[-1] = 0.0
    U = np.empty((nx + 1, len(t_out)))
    U[:, 0] = u
    cache = {}
    for k in range(1, len(t_out)):
        span = t_out[k] - t_out[k - 1]
        m = int(np.ceil(span / (dt_over_dx * dx) - 1e-12))
        key = round(span / m, 14)
        if key not in cache:
            lam = alpha * (span / m) / dx ** 2
            ab = np.zeros((3, nx - 1))
            ab[0, 1:], ab[1, :], ab[2, :-1] = -lam / 2, 1.0 + lam, -lam / 2
            cache[key] = (ab, lam)
        ab, lam = cache[key]
        for _ in range(m):
            rhs = (1.0 - lam) * u[1:-1] + 0.5 * lam * (u[2:] + u[:-2])
            u[1:-1] = solve_banded((1, 1), ab, rhs)
        U[:, k] = u
    return x, U


# ---------------- Exp 3 ----------------
def burgers_spectral_etdrk4(N: int, t_out, nu: float = BURGERS_NU, dt_target: float = 2e-4):
    """Periodic Fourier pseudo-spectral Burgers solver with ETDRK4 (Kassam & Trefethen 2005)."""
    x = -1.0 + 2.0 * np.arange(N) / N
    k = 2.0 * math.pi * np.fft.rfftfreq(N, d=2.0 / N)
    dealias = k < (2.0 / 3.0) * k.max()
    Lop, g = -nu * k ** 2, -0.5j * k
    nonlin = lambda v: g * np.fft.rfft(np.fft.irfft(v, n=N) ** 2) * dealias
    span = t_out[1] - t_out[0]
    m = int(np.ceil(span / dt_target - 1e-12))
    dt = span / m
    E, E2 = np.exp(dt * Lop), np.exp(dt * Lop / 2)
    r = np.exp(1j * math.pi * (np.arange(1, 33) - 0.5) / 32)
    LR = dt * Lop[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))
    v = np.fft.rfft(burgers_initial(x))
    U = np.empty((N, len(t_out)))
    U[:, 0] = burgers_initial(x)
    for j in range(1, len(t_out)):
        for _ in range(m):
            Nv = nonlin(v)
            a = E2 * v + Q * Nv
            Na = nonlin(a)
            b = E2 * v + Q * Na
            Nb = nonlin(b)
            c = E2 * a + Q * (2 * Nb - Nv)
            Nc = nonlin(c)
            v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        U[:, j] = np.fft.irfft(v, n=N)
    return x, U


def spectral_to_test_grid(U, N: int):
    """Pick the spectral nodes that coincide with B_X (x = 1 wraps to x = -1 by periodicity)."""
    idx = (np.arange(len(B_X)) * (N // (len(B_X) - 1))) % N
    return U[idx, :]


def burgers_fd_cnab2(nx: int, t_out, nu: float = BURGERS_NU, cfl: float = 0.4):
    x = np.linspace(-1.0, 1.0, nx)
    dx = x[1] - x[0]
    u = burgers_initial(x)
    u[0] = u[-1] = 0.0
    U = np.empty((nx, len(t_out)))
    U[:, 0] = u
    adv = lambda v: -(0.5 * v[2:] ** 2 - 0.5 * v[:-2] ** 2) / (2.0 * dx)
    span = t_out[1] - t_out[0]
    m = int(np.ceil(span / (cfl * dx) - 1e-12))
    dt = span / m
    mu = nu * dt / dx ** 2
    ab = np.zeros((3, nx - 2))
    ab[0, 1:], ab[1, :], ab[2, :-1] = -mu / 2, 1.0 + mu, -mu / 2
    a_prev = None
    for j in range(1, len(t_out)):
        for _ in range(m):
            a_now = adv(u)
            a_ext = a_now if a_prev is None else 1.5 * a_now - 0.5 * a_prev
            rhs = (1.0 - mu) * u[1:-1] + 0.5 * mu * (u[2:] + u[:-2]) + dt * a_ext
            u[1:-1] = solve_banded((1, 1), ab, rhs)
            a_prev = a_now
        U[:, j] = u
    return x, U


# ---------------- Exp 4 ----------------
def fd_poisson_masked(n: int, inside_fn: Callable, f_fn: Callable, g_fn: Callable,
                      project_fn: Optional[Callable] = None):
    """5-point FD for -Laplace(u) = f on {inside_fn}, Dirichlet g, h = 2/n on [-1,1]^2."""
    xs = np.linspace(-1.0, 1.0, n + 1)
    h = xs[1] - xs[0]
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    inside = inside_fn(XX, YY)
    inside[0, :] = inside[-1, :] = inside[:, 0] = inside[:, -1] = False
    I, J = np.nonzero(inside)
    N = len(I)
    idx = -np.ones(XX.shape, dtype=np.int64)
    idx[I, J] = np.arange(N)
    rows, cols, vals = [np.arange(N)], [np.arange(N)], [np.full(N, 4.0)]
    rhs = h ** 2 * f_fn(XX[I, J], YY[I, J])
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        In, Jn = I + di, J + dj
        nb = inside[In, Jn]
        rows.append(np.nonzero(nb)[0]); cols.append(idx[In[nb], Jn[nb]]); vals.append(-np.ones(int(nb.sum())))
        xo, yo = XX[In[~nb], Jn[~nb]], YY[In[~nb], Jn[~nb]]
        if project_fn is not None:
            xo, yo = project_fn(xo, yo)
        np.add.at(rhs, np.nonzero(~nb)[0], g_fn(xo, yo))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    U = np.full(XX.shape, np.nan)
    U[I, J] = spla.spsolve(A.tocsc(), rhs)
    return XX, YY, U, inside


def radial_projection(x, y):
    return x / np.hypot(x, y), y / np.hypot(x, y)


# ---------------- Exp 5 (classical inverse fit) ----------------
def fit_alpha_classical(x_d, t_d, u_obs, nx: int = 100) -> float:
    t_grid = np.linspace(0.0, 1.0, 101)
    pts = np.stack([x_d, t_d], axis=1)

    def misfit(log_a: float) -> float:
        x, U = heat_crank_nicolson(nx, t_grid, alpha=math.exp(log_a))
        return float(np.mean((RegularGridInterpolator((x, t_grid), U)(pts) - u_obs) ** 2))

    res = minimize_scalar(misfit, bounds=(math.log(1e-3), math.log(1.0)), method="bounded", options={"xatol": 1e-5})
    return math.exp(res.x)
