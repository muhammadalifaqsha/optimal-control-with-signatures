"""Solvers for the state tensors X^(n): Picard sweeps, level-wise forward substitution and the resolvent form."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


def apply_q(X, q, ts):
    """Y^(n) = Σ_n' X^(n') ⊗ q^(n,n'), truncated at ts.L.  X: (..., N, dim)."""
    Y = np.zeros_like(X); d = ts.d
    for m in range(ts.L):
        Xm = X[..., ts.sl(m)]
        Y[..., ts.sl(m + 1)] = np.einsum('...pw,npj->...nwj', Xm, q).reshape(*X.shape[:-1], d ** (m + 1))
    return Y

def truncate(X, ts, m):
    Y = X.copy(); Y[..., ts.offsets[m + 1]:] = 0.0
    return Y

def solve_state_picard(p, q, ts):
    """User's scheme: X = X0 ∅; for m = 1..L: X_new = trunc_m(p + X_old ⊗ q) (all n jointly), X = X_new."""
    X = np.zeros_like(p); X[..., 0] = p[..., 0]
    for m in range(1, ts.L + 1):
        X = truncate(p + apply_q(X, q, ts), ts, m)
    return X

def solve_state_levelwise(p, q, ts):
    """Forward substitution, one new level per step:  X_m = p_m + Σ_n' X_{m-1}^(n') ⊗ q^(n,n')."""
    X = np.zeros_like(p); d = ts.d
    X[..., ts.sl(0)] = p[..., ts.sl(0)]
    for m in range(1, ts.L + 1):
        Xm1 = X[..., ts.sl(m - 1)]
        X[..., ts.sl(m)] = p[..., ts.sl(m)] + np.einsum('...pw,npj->...nwj', Xm1, q).reshape(*p.shape[:-1], d**m)
    return X

def resolvent(q, ts):
    """R = Σ_k Q^{⊗k} (N×N matrix of tensors, Q[n',n] = q^(n,n')).  R[m]: (N, N, d^m)."""
    N, d = q.shape[0], ts.d
    R = [np.eye(N)[:, :, None]]
    for m in range(1, ts.L + 1):
        R.append(np.einsum('abw,nbj->anwj', R[-1], q).reshape(N, N, d**m))
    return R

def solve_state_resolvent(p, q, ts, R=None):
    """Closed form X = p ⊗ (I - Q)^{-1}:  X^(n) = Σ_a p^(a) ⊗ R[a, n].  R is control-independent."""
    R = resolvent(q, ts) if R is None else R
    X = np.zeros_like(p)
    for i in range(ts.L + 1):
        pi = p[..., ts.sl(i)]
        if not np.any(pi): continue
        for m in range(ts.L + 1 - i):
            X[..., ts.sl(i + m)] += np.einsum('...ai,anw->...niw', pi, R[m]).reshape(*p.shape[:-1], ts.d**(i + m))
    return X

SOLVERS = {'picard': solve_state_picard, 'levelwise': solve_state_levelwise, 'resolvent': solve_state_resolvent}
