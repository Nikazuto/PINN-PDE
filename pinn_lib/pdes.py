"""Exact solutions, right-hand sides, domain masks and the fixed test grids (numpy only).

All formulas are copied from the notebook so that errors computed in the dashboard are
directly comparable with the numbers stored in the notebook's results files.
"""
import math

import numpy as np

# ---------------- Exp 1: 1D Poisson ----------------
def poisson_exact(x, xp=np):
    return xp.sin(2 * math.pi * x) + 0.1 * xp.sin(8 * math.pi * x)


def poisson_rhs(x, xp=np):
    return -(2 * math.pi) ** 2 * xp.sin(2 * math.pi * x) - 0.1 * (8 * math.pi) ** 2 * xp.sin(8 * math.pi * x)


# ---------------- Exp 2 & 5: heat equation ----------------
HEAT_ALPHA = 0.05


def heat_initial(x, xp=np):
    return xp.sin(math.pi * x) + 0.5 * xp.sin(3 * math.pi * x)


def heat_exact(x, t, xp=np, alpha=HEAT_ALPHA):
    return (xp.exp(-alpha * math.pi ** 2 * t) * xp.sin(math.pi * x)
            + 0.5 * xp.exp(-9 * alpha * math.pi ** 2 * t) * xp.sin(3 * math.pi * x))


# ---------------- Exp 3 & 6: viscous Burgers ----------------
BURGERS_NU = 0.01 / math.pi


def burgers_initial(x, xp=np):
    return -xp.sin(math.pi * x)


# ---------------- Exp 4a: unit disk ----------------
def disk_exact(x, y, xp=np):
    return xp.sin(math.pi * x) * xp.cos(math.pi * y) + 0.25 * (x ** 2 - y ** 2)


def disk_rhs(x, y, xp=np):
    return 2 * math.pi ** 2 * xp.sin(math.pi * x) * xp.cos(math.pi * y)


def disk_inside(x, y):
    return x ** 2 + y ** 2 < 1.0 - 1e-12


# ---------------- Exp 4b: L-shape ----------------
def lshape_exact(x, y, xp=np):
    r = xp.sqrt(x ** 2 + y ** 2)
    theta = xp.remainder(xp.arctan2(y, x), 2 * math.pi)
    return r ** (2.0 / 3.0) * xp.sin(2.0 * theta / 3.0)


def lshape_inside(x, y):
    eps = 1e-12
    in_square = (np.abs(x) < 1 - eps) & (np.abs(y) < 1 - eps)
    in_cut = (x > -eps) & (y < eps)
    return in_square & ~in_cut


# ---------------- grids ----------------
def grid_points(a, b):
    """Tensor-product grid with 'ij' indexing: AA, BB (len(a), len(b)) and stacked points (N, 2)."""
    AA, BB = np.meshgrid(a, b, indexing="ij")
    return AA, BB, np.stack([AA.ravel(), BB.ravel()], axis=1)


# Exp 1 test grid
P1_X_TEST = np.linspace(0.0, 1.0, 1001)
P1_U_TEST = poisson_exact(P1_X_TEST)

# Exp 2 / 5 test grid (arrays are indexed [i_x, i_t])
H_X, H_T = np.linspace(0.0, 1.0, 101), np.linspace(0.0, 1.0, 101)
H_XX, H_TT, H_XT = grid_points(H_X, H_T)
H_U = heat_exact(H_XX, H_TT)

# Exp 3 / 6 test grid
B_X, B_T = np.linspace(-1.0, 1.0, 513), np.linspace(0.0, 1.0, 101)
B_XX, B_TT, B_XT = grid_points(B_X, B_T)

# Exp 4 test grid (201 x 201) and masks
S_G = np.linspace(-1.0, 1.0, 201)
S_XX, S_YY, S_XY = grid_points(S_G, S_G)
DISK_MASK, LSHAPE_MASK = disk_inside(S_XX, S_YY), lshape_inside(S_XX, S_YY)
DISK_U = np.where(DISK_MASK, disk_exact(S_XX, S_YY), np.nan)
LSHAPE_U = np.where(LSHAPE_MASK, lshape_exact(S_XX, S_YY), np.nan)
