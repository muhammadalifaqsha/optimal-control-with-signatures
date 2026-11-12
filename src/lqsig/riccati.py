"""Riccati benchmark: Itô coefficients, the Riccati ODEs and the optimal Brownian feedback."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from .model import LQModel


def ito_coefficients(model):
    """Itô form: dX = (a + Ā X + b1 u) dt + Σ_d (c_d + C_d X) dW^d."""
    a = model.b0 + 0.5 * sum(model.sigma2[:, dd, :] @ model.sigma0[:, dd] for dd in range(model.D))
    Abar = model.b2 + 0.5 * sum(model.sigma2[:, dd, :] @ model.sigma2[:, dd, :] for dd in range(model.D))
    return a, Abar, [model.sigma0[:, dd] for dd in range(model.D)], [model.sigma2[:, dd, :] for dd in range(model.D)]

def riccati_benchmark(model, n_grid=2001):
    """P(t), r(t) of the value function V(t,x) = x'P x + 2 r'x + s on a uniform grid with n_grid points;
    u* = -B(t)^{-1}(D(t) + b1'(P x + r))."""
    from scipy.integrate import solve_ivp
    N = model.N; a, Ab, c, Cm = ito_coefficients(model); b1 = model.b1
    def rhs(t, y):
        P = y[:N*N].reshape(N, N); r = y[N*N:N*N+N]
        At, Bt, Ct, Dt = (LQModel.poly(model.A, t), LQModel.poly(model.B, t),
                          LQModel.poly(model.C, t), LQModel.poly(model.Dlin, t))
        Bi = np.linalg.inv(Bt); z = Dt + b1.T @ r
        dP = -(At + P @ Ab + Ab.T @ P + sum(Cd.T @ P @ Cd for Cd in Cm) - P @ b1 @ Bi @ b1.T @ P)
        dr = -(Ct + Ab.T @ r + P @ a + sum(Cd.T @ P @ cd for Cd, cd in zip(Cm, c)) - P @ b1 @ Bi @ z)
        return np.concatenate([dP.ravel(), dr])
    yT = np.concatenate([model.E.ravel(), model.G])
    tg = np.linspace(model.T, 0, n_grid)
    sol = solve_ivp(rhs, (model.T, 0), yT, t_eval=tg, rtol=1e-11, atol=1e-11, method='DOP853')
    t = sol.t[::-1]; Y = sol.y[:, ::-1]
    return dict(t=t, P=Y[:N*N].T.reshape(-1, N, N), r=Y[N*N:N*N+N].T)

def riccati_feedback(model, steps):
    """Optimal Brownian feedback u* = -B^{-1}(D + b1'(P X + r)) on a uniform grid with `steps` steps."""
    ric = riccati_benchmark(model, n_grid=steps + 1)
    def ctrl(i, t, X, sig):
        Binv = np.linalg.inv(LQModel.poly(model.B, t))
        return -(LQModel.poly(model.Dlin, t) + (X @ ric['P'][i].T + ric['r'][i]) @ model.b1) @ Binv.T
    return ctrl
