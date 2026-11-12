"""Fractional Brownian motion: Cholesky simulation, Duncan–Pasik-Duncan benchmark, Wong–Zakai simulation, empirical cost functionals and paired out-of-sample evaluation."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from .tensors import TensorSpace
from .model import LQModel
from .cost import theta_to_u
from .riccati import riccati_benchmark
from .montecarlo import signature_stream


def fbm_cholesky(H, n_steps, T):
    """Lower Cholesky factor of the covariance of the fBM increments on a uniform grid."""
    from scipy.linalg import toeplitz
    k = np.arange(n_steps, dtype=float); h = T / n_steps
    g = 0.5 * (np.abs(k + 1) ** (2 * H) + np.abs(k - 1) ** (2 * H) - 2 * k ** (2 * H)) * h ** (2 * H)
    return np.linalg.cholesky(toeplitz(g))

def fbm_increments(H, n_paths, n_steps, D, T, rng, chol=None, return_xi=False):
    """D independent fBMs: ΔB = chol @ ξ, ξ i.i.d. N(0,1).  Returns (n_paths, n_steps, D) [and ξ, same shape]."""
    chol = fbm_cholesky(H, n_steps, T) if chol is None else chol
    xi = rng.standard_normal((n_steps, n_paths * D))
    dB = (chol @ xi).reshape(n_steps, n_paths, D).transpose(1, 0, 2)
    return (dB, xi.reshape(n_steps, n_paths, D).transpose(1, 0, 2)) if return_xi else dB

def dpd_benchmark(model, H, n_steps, chol=None):
    """Duncan & Pasik-Duncan (SICON 2013, Thm 3.3 / Cor. 3.4), discretised on the simulation grid.
    For σ2 = 0 (additive noise σ0 dB^H, B^H a D-dim fBM) the optimal adapted control is
        u*_t = -B(t)^{-1} ( D(t) + b1' (P_t X_t + r_t + V_t) ),   V_t = E[ φ_t | F_t ],
        φ_t = ∫_t^T Φ_P(s,t) P(s) σ0 dB^H_s,   ∂_t Φ_P(s,t) = -(b2 - b1 B^{-1} b1' P)(t)' Φ_P(s,t),  Φ_P(s,s) = I,
    with the deterministic Riccati P and the affine term r (Section 5).  On the grid, φ_i = P_i σ0 ΔB_i + Γ_i φ_{i+1},
    and with ΔB = chol ξ the prediction is exact Gaussian conditioning:  E[ΔB_j | ΔB_0..ΔB_{i-1}] = Σ_{k<i} chol_{jk} ξ_k,
    hence V_i = Σ_{k<i} K[i,k] ξ_k with a deterministic kernel K."""
    from scipy.linalg import expm
    assert not np.any(model.sigma2), "the Duncan–Pasik-Duncan benchmark requires sigma2 = 0 (additive noise)"
    N, D, T = model.N, model.D, model.T; n = n_steps; dt = T / n
    chol = fbm_cholesky(H, n, T) if chol is None else chol
    ric = riccati_benchmark(model, n_grid=n + 1); P, r, t = ric['P'], ric['r'], ric['t']
    Binv = np.stack([np.linalg.inv(LQModel.poly(model.B, s)) for s in t])
    Acl = np.stack([model.b2 - model.b1 @ Binv[i] @ model.b1.T @ P[i] for i in range(n + 1)])
    K = np.zeros((n + 1, n, N, D))
    for i in range(n - 1, -1, -1):
        Gam = expm(0.5 * (Acl[i] + Acl[i + 1]).T * dt)                  # Φ_P(t_{i+1}, t_i)
        K[i] = np.einsum('nd,k->knd', P[i] @ model.sigma0, chol[i]) + np.einsum('nm,kmd->knd', Gam, K[i + 1])
    K = K[:n] * (np.arange(n)[None, :] < np.arange(n)[:, None])[:, :, None, None]     # keep k < i only
    Kmat = K.transpose(0, 2, 1, 3).reshape(n * N, n * D)
    def V_of(xi):            # xi: (paths, n, D) -> V: (paths, n, N)
        return (xi.reshape(xi.shape[0], n * D) @ Kmat.T).reshape(xi.shape[0], n, N)
    def control_factory(V, scale=1.0):
        def ctrl(i, s, X, sig):
            return -(LQModel.poly(model.Dlin, s) + (X @ P[i].T + r[i] + scale * V[:, i]) @ model.b1) @ Binv[i].T
        return ctrl
    return dict(P=P, r=r, t=t, K=K, chol=chol, V_of=V_of, control=control_factory, H=H)

def _wz_step_matrices(model, beta, dt):
    """Exact step of the linear ODE obtained by freezing the piecewise-linear driver slope β on one step:
    dX/ds = (b2 + Σ_d σ2_d β_d) X + const  ->  X' = E X + Φ·const,  E = e^{Mh}, Φ = ∫_0^h e^{sM} ds."""
    from scipy.linalg import expm
    N = model.N; n = beta.shape[0]
    M = model.b2[None] + np.einsum('pd,ndm->pnm', beta, model.sigma2)
    Aug = np.zeros((n, 2 * N, 2 * N)); Aug[:, :N, :N] = M * dt; Aug[:, :N, N:] = np.eye(N) * dt
    Ex = expm(Aug)
    return Ex[:, :N, :N], Ex[:, :N, N:]

def simulate_driver(model, dB, dt, control, sig_level=0, scheme='auto'):
    """State simulation for a general driver (e.g. fBM), SDE understood in the Stratonovich / rough-path
    (Wong–Zakai) sense, i.e. as the limit of ODEs driven by the piecewise-linear interpolation of the driver.
    scheme='euler' (valid only for sigma2 = 0, where all interpretations agree) or
    'wz' (exact solution of the piecewise-linear ODE on each step; control frozen at the left point).
    Cost: left-point Riemann sum.  control(i, t, X, sig) -> (n, K)."""
    if scheme == 'auto': scheme = 'euler' if not np.any(model.sigma2) else 'wz'
    if scheme == 'euler': assert not np.any(model.sigma2), "Euler is only consistent for sigma2 = 0"
    n, steps, D = dB.shape
    X = np.tile(model.X0.astype(float), (n, 1)); cost = np.zeros(n); Xs = [X.copy()]; Us = []
    stream = signature_stream(dB, dt, sig_level) if sig_level > 0 else None
    for i in range(steps):
        t = i * dt; sig = next(stream) if stream is not None else None
        u = control(i, t, X, sig); Us.append(u)
        At, Bt, Ct, Dt = (LQModel.poly(model.A, t), LQModel.poly(model.B, t), LQModel.poly(model.C, t), LQModel.poly(model.Dlin, t))
        cost += dt * (np.einsum('pi,ij,pj->p', X, At, X) + np.einsum('pi,ij,pj->p', u, Bt, u) + 2 * X @ Ct + 2 * u @ Dt)
        if scheme == 'euler':
            X = X + (model.b0 + X @ model.b2.T + u @ model.b1.T) * dt + dB[:, i] @ model.sigma0.T
        else:
            beta = dB[:, i] / dt
            Em, Ph = _wz_step_matrices(model, beta, dt)
            c = model.b0 + u @ model.b1.T + beta @ model.sigma0.T
            X = np.einsum('pnm,pm->pn', Em, X) + np.einsum('pnm,pm->pn', Ph, c)
        Xs.append(X.copy())
    cost += np.einsum('pi,ij,pj->p', X, model.E, X) + 2 * X @ model.G
    return cost, np.array(Xs), np.array(Us)

def simulate_many_driver(model, dB, dt, thetas, reference, scheme='auto'):
    """`simulate_many` for a general driver (e.g. fBM), with the same scheme as `simulate_driver`: Euler for σ2 = 0,
    otherwise the exact step of the piecewise-linear (Wong–Zakai) ODE.  The step matrices depend only on the noise
    path, so they are computed once per step and shared by all controls.  Same arguments and returns as simulate_many."""
    if scheme == 'auto': scheme = 'euler' if not np.any(model.sigma2) else 'wz'
    if scheme == 'euler': assert not np.any(model.sigma2), "Euler is only consistent for sigma2 = 0"
    n, steps, D = dB.shape; C, K = len(thetas), model.K
    Mmax = max(M for _, M in thetas); dimM = TensorSpace(model.d, Mmax).dim
    Th = np.stack([np.pad(theta_to_u(np.asarray(th, float), model, M), ((0, 0), (0, dimM - TensorSpace(model.d, M).dim)))
                   for th, M in thetas]).reshape(C * K, dimM)
    X = np.tile(model.X0.astype(float), (C, n, 1)); Xr = np.tile(model.X0.astype(float), (n, 1))
    cost, d2, cr, qr = np.zeros((C, n)), np.zeros((C, n)), np.zeros(n), np.zeros(n)
    stream = signature_stream(dB, dt, max(Mmax, 1))
    for i in range(steps):
        t = i * dt; sig = next(stream)[:, :dimM]
        u = (sig @ Th.T).reshape(n, C, K).transpose(1, 0, 2); ur = reference(i, t, Xr, None)
        At, Bt, Ct, Dt = (LQModel.poly(model.A, t), LQModel.poly(model.B, t), LQModel.poly(model.C, t), LQModel.poly(model.Dlin, t))
        cost += dt * (np.einsum('cni,ij,cnj->cn', X, At, X) + np.einsum('cni,ij,cnj->cn', u, Bt, u) + 2 * X @ Ct + 2 * u @ Dt)
        cr += dt * (np.einsum('pi,ij,pj->p', Xr, At, Xr) + np.einsum('pi,ij,pj->p', ur, Bt, ur) + 2 * Xr @ Ct + 2 * ur @ Dt)
        d2 += dt * np.sum((u - ur) ** 2, axis=2); qr += dt * np.sum(ur ** 2, axis=1)
        if scheme == 'euler':
            noise = dB[:, i] @ model.sigma0.T
            X = X + (model.b0 + X @ model.b2.T + u @ model.b1.T) * dt + noise[None]
            Xr = Xr + (model.b0 + Xr @ model.b2.T + ur @ model.b1.T) * dt + noise
        else:
            beta = dB[:, i] / dt
            Em, Ph = _wz_step_matrices(model, beta, dt)                     # (n, N, N), shared by all controls
            cb = model.b0 + beta @ model.sigma0.T                          # control-free constant (n, N)
            X = np.einsum('pnm,cpm->cpn', Em, X) + np.einsum('pnm,cpm->cpn', Ph, cb[None] + u @ model.b1.T)
            Xr = np.einsum('pnm,pm->pn', Em, Xr) + np.einsum('pnm,pm->pn', Ph, cb + ur @ model.b1.T)
    cost += np.einsum('cni,ij,cnj->cn', X, model.E, X) + 2 * X @ model.G
    cr += np.einsum('pi,ij,pj->p', Xr, model.E, Xr) + 2 * Xr @ model.G
    return cost, cr, d2, qr

class EmpiricalCostFunctionals:
    """Drop-in replacement for CostFunctionals when E[Ŵ_T] has no closed form (Section 3.5):
    the expected signature is replaced by the empirical measure of simulated driver paths.  Because shuffle
    identities hold pathwise,  <(1^{⊗m} ш v ш w)⊗1, Ê[Ŵ_T]> = Ê ∫ t^m/m! Ŵ^v_t Ŵ^w_t dt,  so only signatures up to
    level max(L, M) are needed (instead of level 2L+m+1); the dt-integral uses the trapezoid rule on the recorded grid."""
    def __init__(self, model, L, M, dB, dt, record_every=1, chunk=2000):
        d = model.d; Lb = max(L, M); tb = TensorSpace(d, Lb); self.L, self.M, self.d = L, M, d
        n_paths, n_steps, _ = dB.shape
        rec = list(range(0, n_steps + 1, record_every))
        if rec[-1] != n_steps: rec.append(n_steps)
        tg = np.array(rec) * dt; w = np.zeros(len(rec)); w[1:] += 0.5 * np.diff(tg); w[:-1] += 0.5 * np.diff(tg)
        recpos = {i: j for j, i in enumerate(rec)}; mm = model.m_max
        G = [np.zeros((tb.dim, tb.dim)) for _ in range(mm + 1)]; l = [np.zeros(tb.dim) for _ in range(mm + 1)]
        GT, lT = np.zeros((tb.dim, tb.dim)), np.zeros(tb.dim)
        for c0 in range(0, n_paths, chunk):
            for i, sig in enumerate(signature_stream(dB[c0:c0 + chunk], dt, Lb)):
                if i not in recpos: continue
                j = recpos[i]; SS = sig.T @ sig; s1 = sig.sum(0)
                for m in range(mm + 1):
                    c = w[j] * tg[j] ** m / factorial(m); G[m] += c * SS; l[m] += c * s1
                if i == n_steps: GT += SS; lT += s1
        dL, dM = TensorSpace(d, L).dim, TensorSpace(d, M).dim
        self.Grun = [g[:dL, :dL] / n_paths for g in G]; self.GrunU = [g[:dM, :dM] / n_paths for g in G]
        self.lrun = [v[:dL] / n_paths for v in l]; self.lrunU = [v[:dM] / n_paths for v in l]
        self.Gterm = GT[:dL, :dL] / n_paths; self.lterm = lT[:dL] / n_paths
        self._full = (G, l, GT, lT, n_paths)

    def restrict(self, L, M):
        """Functionals for smaller (L, M) from the same paths: just sub-blocks (no recomputation)."""
        assert L <= self.L or M <= self.M
        G, l, GT, lT, n = self._full
        out = EmpiricalCostFunctionals.__new__(EmpiricalCostFunctionals)
        dL, dM = TensorSpace(self.d, L).dim, TensorSpace(self.d, M).dim
        assert dL <= G[0].shape[0] and dM <= G[0].shape[0]
        out.L, out.M, out.d, out._full = L, M, self.d, self._full
        out.Grun = [g[:dL, :dL] / n for g in G]; out.GrunU = [g[:dM, :dM] / n for g in G]
        out.lrun = [v[:dL] / n for v in l]; out.lrunU = [v[:dM] / n for v in l]
        out.Gterm = GT[:dL, :dL] / n; out.lterm = lT[:dL] / n
        return out

def evaluate_controls(model, controls, dB, dt, reference=None):
    """Out-of-sample Monte Carlo with common random numbers.
    controls: name -> (callable, sig_level).  Returns name -> dict(cost=per-path costs, U=control paths),
    and, if `reference` names one of the controls, paired cost differences and L2(dP⊗dt) distances to it."""
    out = {}; names = list(controls)
    if reference is not None: names = [reference] + [k for k in names if k != reference]
    Uref = None
    for name in names:
        ctrl, lev = controls[name]
        c, _, U = simulate_driver(model, dB, dt, ctrl, sig_level=lev)
        out[name] = dict(cost=c)
        if reference is not None:
            if name == reference:
                Uref, cref = U, c; q2ref = np.sum(dt * np.sum(Uref ** 2, axis=2), axis=0)   # per path: ∫|u_ref|² dt
            out[name]['diff'] = c - cref
            out[name]['d2'] = np.sum(dt * np.sum((U - Uref) ** 2, axis=2), axis=0)     # per path: ∫|u - u_ref|² dt
            out[name]['L2'] = np.sqrt(out[name]['d2'].mean()); out[name]['cref'] = cref; out[name]['q2'] = q2ref
        del U                                  # control paths are not kept (memory)
    return {k: out[k] for k in controls}
