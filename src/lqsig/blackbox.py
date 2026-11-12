"""The paper's route: the explicit tensor J^L(u) and recovery of the quadratic by polarisation (used for checks)."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from .tensors import TensorSpace, to_levels
from .shuffles import shuffle_product
from .model import build_p, build_q
from .state import SOLVERS
from .cost import theta_to_u


def concat_time(x, d):
    """x ⊗ 1 (append the time letter)."""
    out = [np.zeros(1)]
    for xm in x:
        y = np.zeros((xm.size, d)); y[:, 0] = xm; out.append(y.ravel())
    return out

def word_tensor(word, d, Lmax):
    t = [np.zeros(d**n) for n in range(Lmax + 1)]
    i = 0
    for a in word: i = i * d + a
    t[len(word)][i] = 1.0
    return t

def J_tensor_explicit(model, X_levels, u_levels, Lout):
    """J^L(u) of Prop. 3.6 built with explicit shuffles; X_levels[n], u_levels[k] are lists of level arrays."""
    d = model.d; J = [np.zeros(d**n) for n in range(Lout + 1)]
    add = lambda acc, x, c: [a + c * (x[i] if i < len(x) else 0) for i, a in enumerate(acc)]
    run = [np.zeros(d**n) for n in range(Lout)]
    for m in range(model.m_max + 1):
        om = word_tensor((0,) * m, d, m)
        if m < len(model.A):
            for n in range(model.N):
                for n2 in range(model.N):
                    if model.A[m][n, n2] != 0:
                        run = add(run, shuffle_product(om, shuffle_product(X_levels[n], X_levels[n2], Lout - 1, d), Lout - 1, d), model.A[m][n, n2])
        if m < len(model.B):
            for k in range(model.K):
                for k2 in range(model.K):
                    if model.B[m][k, k2] != 0:
                        run = add(run, shuffle_product(om, shuffle_product(u_levels[k], u_levels[k2], Lout - 1, d), Lout - 1, d), model.B[m][k, k2])
        if m < len(model.C):
            for n in range(model.N):
                if model.C[m][n] != 0: run = add(run, shuffle_product(om, X_levels[n], Lout - 1, d), 2 * model.C[m][n])
        if m < len(model.Dlin):
            for k in range(model.K):
                if model.Dlin[m][k] != 0: run = add(run, shuffle_product(om, u_levels[k], Lout - 1, d), 2 * model.Dlin[m][k])
    J = add(J, concat_time(run, d), 1.0)
    for n in range(model.N):
        for n2 in range(model.N):
            if model.E[n, n2] != 0: J = add(J, shuffle_product(X_levels[n], X_levels[n2], Lout, d), model.E[n, n2])
        if model.G[n] != 0: J = add(J, X_levels[n], 2 * model.G[n])
    return J

def blackbox_cost(model, theta, L, M, ES, solver='levelwise'):
    """<J^L(u), E[Ŵ_T]> for one θ, the way the paper evaluates it."""
    ts = TensorSpace(model.d, L); tM = TensorSpace(model.d, M); q = build_q(model)
    p0, pU = build_p(model, L, M)
    X = SOLVERS[solver]((p0 + np.tensordot(theta, pU, axes=1))[None], q, ts)[0]
    U = theta_to_u(theta, model, M)
    Lout = 2 * max(L, M) + model.m_max + 1
    J = J_tensor_explicit(model, [to_levels(X[n], ts) for n in range(model.N)],
                          [to_levels(U[k], tM) for k in range(model.K)], Lout)
    return sum(a @ b for a, b in zip(J, ES))

def polarisation_coefficients(f, P):
    """Recover c0, c_v, c_{v1 v2} of the quadratic f(θ) = Σ c_{v1v2} θ_v1 θ_v2 + Σ c_v θ_v + c0
    (sum over ordered pairs, so the cross term of v1≠v2 is 2 c_{v1v2}).
    NB: the cross formula subtracts the linear terms c_v1 + c_v2 as well."""
    e = np.eye(P); c0 = f(np.zeros(P))
    fp = np.array([f(e[i]) for i in range(P)]); fm = np.array([f(-e[i]) for i in range(P)])
    cv = (fp - fm) / 2; H = np.diag((fp + fm) / 2 - c0)
    for i in range(P):
        for j in range(i + 1, P):
            H[i, j] = H[j, i] = (f(e[i] + e[j]) - H[i, i] - H[j, j] - cv[i] - cv[j] - c0) / 2
    return c0, cv / 2, H          # in the convention f = c0 + 2 h·θ + θ'Hθ
