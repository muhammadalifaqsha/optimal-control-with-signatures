"""Monte Carlo tools: signatures along simulated paths, Itô-Euler simulation, paired statistics and tables."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from .tensors import TensorSpace
from .model import LQModel
from .cost import theta_to_u
from .riccati import ito_coefficients


def signature_stream(dW, dt, Lsig):
    """Yield the flat signature of the time-augmented piecewise-linear path at t_0, t_1, ..., t_n.
    dW: (n_paths, n_steps, D).  Chen's identity S_{t+dt} = S_t ⊗ exp(Δ), evaluated level by level with a
    Horner scheme:  (S ⊗ e^Δ)_m = S_m + (((S_0 Δ/m + S_1) Δ/(m-1) + S_2) Δ/(m-2) + ... + S_{m-1}) Δ/1."""
    n, steps, D = dW.shape; d = D + 1
    S = [np.ones((n, 1))] + [np.zeros((n, d**m)) for m in range(1, Lsig + 1)]
    yield np.concatenate(S, axis=1)
    for i in range(steps):
        inc = np.concatenate([np.full((n, 1), dt), dW[:, i, :]], axis=1)
        new = [S[0]]
        for m in range(1, Lsig + 1):
            acc = S[0] * inc / m
            for j in range(1, m):
                acc = ((acc + S[j])[:, :, None] * inc[:, None, :]).reshape(n, -1) / (m - j)
            new.append(S[m] + acc)
        S = new
        yield np.concatenate(S, axis=1)

def simulate(model, dW, dt, control, sig_level=0):
    """Euler–Maruyama in Itô form.  control(i, t, X, sig) -> u (n, K).  Returns per-path cost, X path, u path."""
    n, steps, D = dW.shape; a, Ab, c, Cm = ito_coefficients(model)
    X = np.tile(model.X0, (n, 1)); cost = np.zeros(n); Xs = [X.copy()]; Us = []
    stream = signature_stream(dW, dt, sig_level) if sig_level > 0 else None
    for i in range(steps):
        t = i * dt; sig = next(stream) if stream is not None else None
        u = control(i, t, X, sig); Us.append(u)
        At, Bt, Ct, Dt = (LQModel.poly(model.A, t), LQModel.poly(model.B, t), LQModel.poly(model.C, t), LQModel.poly(model.Dlin, t))
        cost += dt * (np.einsum('pi,ij,pj->p', X, At, X) + np.einsum('pi,ij,pj->p', u, Bt, u) + 2 * X @ Ct + 2 * u @ Dt)
        dX = (a + X @ Ab.T + u @ model.b1.T) * dt
        for dd in range(D): dX += (c[dd] + X @ Cm[dd].T) * dW[:, i, dd:dd + 1]
        X = X + dX; Xs.append(X.copy())
    cost += np.einsum('pi,ij,pj->p', X, model.E, X) + 2 * X @ model.G
    return cost, np.array(Xs), np.array(Us)

def mean_ci(x):
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))

def fmt_mc(x, p=4):
    """Monte-Carlo samples x -> 'mean ± 95% CI half-width'."""
    m, ci = mean_ci(x)
    return f"{m:.{p}f} ± {ci:.{p}f}"

def l2_ci(d2):
    """L2(dP⊗dt) distance sqrt(E ∫|u - u_ref|² dt) from per-path squared distances d2, with a 95% CI
    (the CI of the mean of d2, endpoints mapped through the square root).  Returns (value, lower, upper)."""
    m, ci = mean_ci(d2)
    return np.sqrt(m), np.sqrt(max(m - ci, 0.0)), np.sqrt(m + ci)

def fmt_l2(d2, p=4):
    """'L2 distance ± 95% CI half-width' from per-path squared distances."""
    v, lo, hi = l2_ci(d2)
    return f"{v:.{p}f} ± {(hi - lo) / 2:.{p}f}"

def signature_control(theta, model, M):
    """Control callable for simulate / simulate_driver: u_t = <θ, Ŵ_t^{≤M}>."""
    U = theta_to_u(theta, model, M); dM = TensorSpace(model.d, M).dim
    return lambda i, t, X, sig: sig[:, :dM] @ U.T

def simulate_many(model, dW, dt, thetas, reference):
    """Itô Euler–Maruyama (as `simulate`) for many signature controls and one reference control, all on the SAME
    noise, in a single pass (the signature is computed once per step).  thetas: list of (θ, M); reference:
    callable(i, t, X, sig) -> u that needs no signature (e.g. riccati_feedback).
    Returns per-path costs (C, n) of the signature controls, per-path costs (n,) of the reference, per-path
    squared L2(dt) distances (C, n) of each control to the reference control, ∫|u_t - u_ref,t|² dt, and per-path
    ∫|u_ref,t|² dt (n,) of the reference."""
    n, steps, D = dW.shape; a, Ab, c, Cm = ito_coefficients(model); C, K = len(thetas), model.K
    Mmax = max(M for _, M in thetas); dimM = TensorSpace(model.d, Mmax).dim
    Th = np.stack([np.pad(theta_to_u(np.asarray(th, float), model, M), ((0, 0), (0, dimM - TensorSpace(model.d, M).dim)))
                   for th, M in thetas]).reshape(C * K, dimM)                # all controls as one matrix
    X = np.tile(model.X0.astype(float), (C, n, 1)); Xr = np.tile(model.X0.astype(float), (n, 1))
    cost, d2, cr, qr = np.zeros((C, n)), np.zeros((C, n)), np.zeros(n), np.zeros(n)
    stream = signature_stream(dW, dt, max(Mmax, 1))
    for i in range(steps):
        t = i * dt; sig = next(stream)[:, :dimM]
        u = (sig @ Th.T).reshape(n, C, K).transpose(1, 0, 2); ur = reference(i, t, Xr, None)
        At, Bt, Ct, Dt = (LQModel.poly(model.A, t), LQModel.poly(model.B, t), LQModel.poly(model.C, t), LQModel.poly(model.Dlin, t))
        cost += dt * (np.einsum('cni,ij,cnj->cn', X, At, X) + np.einsum('cni,ij,cnj->cn', u, Bt, u) + 2 * X @ Ct + 2 * u @ Dt)
        cr += dt * (np.einsum('pi,ij,pj->p', Xr, At, Xr) + np.einsum('pi,ij,pj->p', ur, Bt, ur) + 2 * Xr @ Ct + 2 * ur @ Dt)
        d2 += dt * np.sum((u - ur) ** 2, axis=2); qr += dt * np.sum(ur ** 2, axis=1)
        dX = (a + X @ Ab.T + u @ model.b1.T) * dt; dXr = (a + Xr @ Ab.T + ur @ model.b1.T) * dt
        for dd in range(D):
            dX += (c[dd] + X @ Cm[dd].T) * dW[None, :, i, dd:dd + 1]
            dXr += (c[dd] + Xr @ Cm[dd].T) * dW[:, i, dd:dd + 1]
        X = X + dX; Xr = Xr + dXr
    cost += np.einsum('cni,ij,cnj->cn', X, model.E, X) + 2 * X @ model.G
    cr += np.einsum('pi,ij,pj->p', Xr, model.E, Xr) + 2 * Xr @ model.G
    return cost, cr, d2, qr

def mc_grid(keys, cost, cref, d2, q2):
    """(L, M) -> per-path Monte-Carlo arrays of one control: cost, benchmark cost cref, paired difference diff,
    squared distance d2 = ∫|u - u_ref|² dt to the benchmark control, and q2 = ∫|u_ref|² dt."""
    return {k: dict(cost=cost[c], cref=cref, diff=cost[c] - cref, d2=d2[c], q2=q2) for c, k in enumerate(keys)}

def mc_mean(x):
    """(mean, lower, upper) with the 95% CI of the mean."""
    m, ci = mean_ci(x)
    return m, m - ci, m + ci

def rel_excess_ci(diff, cref):
    """Relative excess cost 100 E[J(u) - J(ref)] / E[J(ref)] in %, with a 95% CI (delta method on the paired samples)."""
    mc = cref.mean(); R = diff.mean() / mc; z = (diff - R * cref) / mc
    return 100 * R, 100 * 1.96 * z.std(ddof=1) / np.sqrt(len(z))

def rel_l2_ci(d2, q2):
    """Relative L2(dP⊗dt) error 100 ‖u - u_ref‖ / ‖u_ref‖ in %, with a 95% CI (delta method)."""
    a, b = d2.mean(), q2.mean(); f = np.sqrt(a / b); z = (d2 / b - a * q2 / b ** 2) / (2 * f)
    return 100 * f, 100 * 1.96 * z.std(ddof=1) / np.sqrt(len(z))

def _triple(v_ci):
    return v_ci[0], v_ci[0] - v_ci[1], v_ci[0] + v_ci[1]

def print_grid(mc, Ls, Ms, p=4, ref="u*"):
    """Monte-Carlo tables over the (L, M) grid, each ± 95% CI: excess cost over the benchmark, relative excess cost (%),
    L2(dP⊗dt) distance to the benchmark control, relative L2 error (%)."""
    def block(title, f, w):
        print(f"\n{title}     rows: L, cols: M = {Ms}")
        for L in Ls: print(f"L={L}: " + "  ".join(f"{f(mc[(L, M)]):>{w}}" for M in Ms))
    w = 2 * p + 12
    block(f"excess cost J(u^{{L,M}}) − J({ref})  [paired MC: mean ± 95% CI]", lambda v: fmt_mc(v['diff'], p), w)
    block(f"relative excess cost (J(u^{{L,M}}) − J({ref})) / J({ref})  [%, ± 95% CI]",
          lambda v: "%.4f ± %.4f" % rel_excess_ci(v['diff'], v['cref']), 22)
    block(f"L2(dP⊗dt) distance to {ref}  [± 95% CI]", lambda v: fmt_l2(v['d2'], p), w)
    block(f"relative L2 error ‖u^{{L,M}} − {ref}‖ / ‖{ref}‖  [%, ± 95% CI]",
          lambda v: "%.4f ± %.4f" % rel_l2_ci(v['d2'], v['q2']), 22)
