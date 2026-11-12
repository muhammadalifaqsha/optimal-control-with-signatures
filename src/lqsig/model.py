"""The controlled linear system and quadratic cost (LQModel) and its tensor-equation coefficients q and p."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from .tensors import TensorSpace


@dataclass
class LQModel:
    """dX^(n) = [b0^(n) + Σ_k b1^(n,k) u^(k) + Σ_n' b2^(n,n') X^(n')] dt
              + Σ_d [σ0^(n,d) + Σ_n' σ2^(n,d,n') X^(n')] ∘ dW^(d)
    J(u) = E[∫ X'A(t)X + u'B(t)u + 2C(t)'X + 2D(t)'u dt + X_T'E X_T + 2G'X_T],
    A(t) = Σ_m t^m/m! A[m]  (same for B, C, Dlin)."""
    T: float
    X0: np.ndarray; b0: np.ndarray; b1: np.ndarray; b2: np.ndarray
    sigma0: np.ndarray; sigma2: np.ndarray
    A: list; B: list; C: list; Dlin: list
    E: np.ndarray; G: np.ndarray
    def __post_init__(self):
        f = lambda z: np.array(z, dtype=float)
        self.X0, self.b0, self.b1, self.b2 = f(self.X0), f(self.b0), f(self.b1), f(self.b2)
        self.sigma0, self.sigma2, self.E, self.G = f(self.sigma0), f(self.sigma2), f(self.E), f(self.G)
        self.A = [f(a) for a in self.A]; self.B = [f(a) for a in self.B]
        self.C = [f(a) for a in self.C]; self.Dlin = [f(a) for a in self.Dlin]
        self.N, self.K = self.b1.shape; self.D = self.sigma0.shape[1]; self.d = self.D + 1
        assert self.b2.shape == (self.N, self.N) and self.sigma2.shape == (self.N, self.D, self.N)
    @property
    def m_max(self):
        return max(len(self.A), len(self.B), len(self.C), len(self.Dlin)) - 1
    @staticmethod
    def poly(coefs, t):
        return sum(t**m / factorial(m) * c for m, c in enumerate(coefs))

def build_q(model):
    """q[n, n', j]: coefficient of letter j in q^(n,n') = b2^(n,n') 1 + Σ_d σ2^(n,d,n') (d+1)."""
    q = np.zeros((model.N, model.N, model.d))
    q[:, :, 0] = model.b2
    q[:, :, 1:] = model.sigma2.transpose(0, 2, 1)
    return q

def build_p(model, L, M):
    """Affine parametrisation  p^u = p0 + Σ_i θ_i pU[i],  θ = (u^(1) coeffs, ..., u^(K) coeffs) in T^M.
    Returns p0 (N, dim_L) and pU (P, N, dim_L), P = K * dim(T^M)."""
    d = model.d; ts, tM = TensorSpace(d, L), TensorSpace(d, M)
    p0 = np.zeros((model.N, ts.dim))
    p0[:, 0] = model.X0
    if L >= 1:
        p0[:, ts.index((0,))] += model.b0
        for dd in range(model.D): p0[:, ts.index((dd + 1,))] += model.sigma0[:, dd]
    pU = np.zeros((model.K * tM.dim, model.N, ts.dim))
    for k in range(model.K):
        for iv, v in enumerate(tM.words()):
            if len(v) + 1 <= L:                       # u ⊗ 1 lands on level |v| + 1
                pU[k * tM.dim + iv, :, ts.index(v + (0,))] = model.b1[:, k]
    return p0, pU
