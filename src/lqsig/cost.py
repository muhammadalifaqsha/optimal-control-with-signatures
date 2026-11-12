"""The cost J^L as an exact quadratic in the signature coefficients θ, its closed-form minimiser and helpers."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from .tensors import TensorSpace, expected_signature_bm, from_levels
from .shuffles import pairing_matrix, running_functional
from .model import build_p, build_q
from .state import SOLVERS


class CostFunctionals:
    """Pairing matrices / vectors such that, for tensors x, y,
       x @ Grun[m] @ y = <(1^{⊗m} ш x ш y) ⊗ 1, E[Ŵ_T]>,   lrun[m] @ x = <(1^{⊗m} ш x) ⊗ 1, E[Ŵ_T]>,
       x @ Gterm  @ y = <x ш y, E[Ŵ_T]>,                    lterm  @ x = <x, E[Ŵ_T]>."""
    def __init__(self, D, T, L, M, m_max, ES=None):
        d = D + 1; Lb = max(L, M); self.d, self.L, self.M = d, L, M
        need = 2 * Lb + m_max + 1
        self.ES = expected_signature_bm(D, T, need) if ES is None else ES[:need + 1]
        self.Gamma = [running_functional(self.ES, m, 2 * Lb, d) for m in range(m_max + 1)]
        self.Grun  = [pairing_matrix(Gm, L, L, d) for Gm in self.Gamma]
        self.GrunU = [pairing_matrix(Gm, M, M, d) for Gm in self.Gamma]
        self.Gterm = pairing_matrix(self.ES, L, L, d)
        self.lrun  = [from_levels(Gm[:L + 1]) for Gm in self.Gamma]
        self.lrunU = [from_levels(Gm[:M + 1]) for Gm in self.Gamma]
        self.lterm = from_levels(self.ES[:L + 1])

def _quad(Y, G, W):
    return np.einsum('anv,nm,bmv->ab', Y @ G, W, Y, optimize=True)

def assemble_quadratic(model, Xall, Uall, F):
    """Xall: (1+P, N, dim_L) [row 0 = constant part, rows 1.. = ∂X/∂θ_i];  Uall: (1+P, K, dim_M).
       Returns (c0, h, H) with  J^L(θ) = c0 + 2 h·θ + θ' H θ."""
    S = np.zeros((Xall.shape[0], Xall.shape[0]))
    for m in range(len(F.Grun)):
        if m < len(model.A) and np.any(model.A[m]): S += _quad(Xall, F.Grun[m], model.A[m])
        if m < len(model.B) and np.any(model.B[m]): S += _quad(Uall, F.GrunU[m], model.B[m])
        if m < len(model.C) and np.any(model.C[m]):
            a = np.einsum('anv,v,n->a', Xall, F.lrun[m], model.C[m]); S[0, :] += a; S[:, 0] += a
        if m < len(model.Dlin) and np.any(model.Dlin[m]):
            a = np.einsum('anv,v,n->a', Uall, F.lrunU[m], model.Dlin[m]); S[0, :] += a; S[:, 0] += a
    if np.any(model.E): S += _quad(Xall, F.Gterm, model.E)
    if np.any(model.G):
        a = np.einsum('anv,v,n->a', Xall, F.lterm, model.G); S[0, :] += a; S[:, 0] += a
    S = 0.5 * (S + S.T)
    return S[0, 0], S[0, 1:], S[1:, 1:]

def control_basis(model, M):
    tM = TensorSpace(model.d, M); P = model.K * tM.dim
    U = np.zeros((1 + P, model.K, tM.dim))
    for k in range(model.K):
        U[1 + k * tM.dim: 1 + (k + 1) * tM.dim, k, :] = np.eye(tM.dim)
    return U

def state_affine(model, L, M, solver='levelwise'):
    """X(θ) = Xall[0] + Σ_i θ_i Xall[1+i]  (the solve is linear in p, so one batched solve suffices)."""
    ts = TensorSpace(model.d, L); q = build_q(model)
    p0, pU = build_p(model, L, M)
    return SOLVERS[solver](np.concatenate([p0[None], pU]), q, ts)

def solve_signature_lq(model, L, M, solver='levelwise', F=None):
    """Minimise J^L over u in (T^M)^K.  Returns dict with θ*, value J^L(θ*), and the quadratic."""
    F = CostFunctionals(model.D, model.T, L, M, model.m_max) if F is None else F
    assert F.L == L and F.M >= M, "CostFunctionals must have the same L and at least the same M"
    Xall = state_affine(model, L, M, solver)
    Uall = control_basis(model, M)
    Uall = np.pad(Uall, ((0, 0), (0, 0), (0, TensorSpace(model.d, F.M).dim - Uall.shape[2])))  # T^M ⊂ T^{F.M}
    c0, h, H = assemble_quadratic(model, Xall, Uall, F)
    theta = -np.linalg.solve(H, h)
    return dict(theta=theta, value=c0 + h @ theta, c0=c0, h=h, H=H, L=L, M=M)

def theta_to_u(theta, model, M):
    return theta.reshape(model.K, TensorSpace(model.d, M).dim)

def cost_of_theta(model, theta, M, L_eval, solver='levelwise', F=None):
    """J^{L_eval}(θ): the cost of a *fixed* signature control evaluated with a (high) state truncation."""
    F = CostFunctionals(model.D, model.T, L_eval, M, model.m_max) if F is None else F
    Xall = state_affine(model, L_eval, M, solver)
    Xth = Xall[0] + np.tensordot(theta, Xall[1:], axes=1)
    Uth = theta_to_u(theta, model, M)
    dimF = TensorSpace(model.d, F.M).dim                      # embed T^M into T^{F.M} (flat prefix)
    Uth = np.pad(Uth, ((0, 0), (0, dimF - Uth.shape[1])))
    c0, h, H = assemble_quadratic(model, np.stack([Xth, np.zeros_like(Xth)]),
                                  np.stack([Uth, np.zeros_like(Uth)]), F)
    return c0
