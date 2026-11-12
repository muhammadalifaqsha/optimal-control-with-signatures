"""Fully recurrent network (fRNN) of Horvath, Teichmann & Žurič (2021), adapted from hedging to control,
with an exact hand-written backward pass through the Wong–Zakai simulation (NumPy only)."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from .model import LQModel


def _mv(A, x):  return (A @ x[..., None])[..., 0]                         # batched A x

def _mtv(A, x): return (x[..., None, :] @ A)[..., 0, :]                      # batched A' x

def _expm_batched(A, order=12):
    """exp(A) for a stack of small matrices: Taylor series with scaling and squaring (one pass over the whole stack)."""
    nrm = np.abs(A).sum(-1).max(); s = max(0, int(np.ceil(np.log2(max(nrm, 1e-300) / 0.5))))
    A = A / 2 ** s; out = np.eye(A.shape[-1]) + A; term = A
    for k in range(2, order + 1):
        term = term @ A / k; out = out + term
    for _ in range(s): out = out @ out
    return out

def step_matrices(model, dB, dt, scheme='auto'):
    """Noise-only one-step maps X' = E X + Φ (b0 + σ0 β + b1 u), β = ΔB/dt: the exact piecewise-linear (Wong–Zakai) step
    of simulate_driver, or Euler for σ2 = 0.  Returns E, Φ (n, P, N, N) and the constant c (n, P, N)."""
    if scheme == 'auto': scheme = 'euler' if not np.any(model.sigma2) else 'wz'
    n_paths, n, D = dB.shape; N = model.N; beta = (dB / dt).transpose(1, 0, 2)
    if scheme == 'euler':
        E = np.broadcast_to(np.eye(N) + model.b2 * dt, (n, n_paths, N, N)); Ph = np.broadcast_to(np.eye(N) * dt, (n, n_paths, N, N))
    else:
        Mt = model.b2 + np.einsum('ipd,ndm->ipnm', beta, model.sigma2)
        Aug = np.zeros((n, n_paths, 2 * N, 2 * N)); Aug[..., :N, :N] = Mt * dt; Aug[..., :N, N:] = np.eye(N) * dt
        Ex = _expm_batched(Aug); E, Ph = Ex[..., :N, :N], Ex[..., :N, N:]
    return E, Ph, model.b0 + beta @ model.sigma0.T

class FRNNPolicy:
    """One feed-forward cell per time step.  Parameters (leading axis = time step): W1 (n, 2D, w), b1 (n, w), W2 (n, w, w),
    b2 (n, w), W3 (n, w, K + D), b3 (n, K + D).  Output columns: first K = control, last D = next hidden state."""
    def __init__(self, model, n_steps, width=12, seed=0):
        self.model, self.n, self.w = model, n_steps, width; D, K = model.D, model.K
        self.dt = model.T / n_steps; rng = np.random.default_rng(seed); n_in, n_out = 2 * D, K + D
        self.W1 = rng.normal(scale=1 / np.sqrt(n_in), size=(n_steps, n_in, width)); self.b1 = np.zeros((n_steps, width))
        self.W2 = rng.normal(scale=1 / np.sqrt(width), size=(n_steps, width, width)); self.b2 = np.zeros((n_steps, width))
        self.W3 = rng.normal(scale=1 / np.sqrt(width), size=(n_steps, width, n_out)); self.b3 = np.zeros((n_steps, n_out))
        self.W3[:, :, :K] *= 0.01                                   # start with an almost-zero control

    def params(self): return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
    def set_params(self, ps): self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = ps

    def cell(self, i, Wt, hid):
        """Cell F_i: (W_{t_i}, δ̃_i) -> (u_i, δ̃_{i+1}) and the cache for backward."""
        x = np.concatenate([Wt, hid], axis=1)
        a1 = np.tanh(x @ self.W1[i] + self.b1[i]); a2 = np.tanh(a1 @ self.W2[i] + self.b2[i])
        out = a2 @ self.W3[i] + self.b3[i]; K = self.model.K
        return out[:, :K], out[:, K:], (x, a1, a2)

    def cell_backward(self, i, dout, cache, grads):
        """dout = gradient w.r.t. (u_i, δ̃_{i+1}); accumulates into grads[...][i]; returns the gradient w.r.t. δ̃_i."""
        x, a1, a2 = cache; D = self.model.D
        grads[4][i] += a2.T @ dout; grads[5][i] += dout.sum(0)
        d2 = (dout @ self.W3[i].T) * (1 - a2 ** 2); grads[2][i] += a1.T @ d2; grads[3][i] += d2.sum(0)
        d1 = (d2 @ self.W2[i].T) * (1 - a1 ** 2); grads[0][i] += x.T @ d1; grads[1][i] += d1.sum(0)
        return (d1 @ self.W1[i].T)[:, D:]

def frnn_cost_and_grad(pol, dB, scheme='auto', need_grad=True):
    """Per-path cost of the fRNN control on the increments dB (same discretisation as simulate_driver: left-point running
    cost, control frozen on each step) and the exact gradient of the MEAN cost (backpropagation through time)."""
    model = pol.model; n_paths, n, D = dB.shape; dt = pol.dt; N, K = model.N, model.K
    E, Ph, cb = step_matrices(model, dB, dt, scheme)
    Wpath = np.concatenate([np.zeros((n_paths, 1, D)), np.cumsum(dB, axis=1)], axis=1)        # W_{t_0}, ..., W_{t_n}
    tg = np.arange(n) * dt
    Q = [LQModel.poly(model.A, t) for t in tg]; R = [LQModel.poly(model.B, t) for t in tg]
    q = [LQModel.poly(model.C, t) for t in tg]; rho = [LQModel.poly(model.Dlin, t) for t in tg]
    X = np.tile(model.X0.astype(float), (n_paths, 1)); hid = np.zeros((n_paths, D))
    Xs, Us, caches = [X], [], []; cost = np.zeros(n_paths)
    for i in range(n):
        u, hid, cache = pol.cell(i, Wpath[:, i], hid); Us.append(u); caches.append(cache)
        cost += dt * (((X @ Q[i]) * X).sum(1) + ((u @ R[i]) * u).sum(1) + 2 * X @ q[i] + 2 * u @ rho[i])
        X = _mv(E[i], X) + _mv(Ph[i], cb[i] + u @ model.b1.T); Xs.append(X)
    cost += ((X @ model.E) * X).sum(1) + 2 * X @ model.G
    if not need_grad: return cost, None
    grads = [np.zeros_like(p) for p in pol.params()]
    lamX = (2 * X @ model.E + 2 * model.G) / n_paths; lamh = np.zeros((n_paths, D))   # dJ̄/dX_{i+1}, dJ̄/dδ̃_{i+1}
    for i in range(n - 1, -1, -1):
        X, u = Xs[i], Us[i]
        gu = dt * (2 * u @ R[i] + 2 * rho[i]) / n_paths + _mtv(Ph[i], lamX) @ model.b1      # dJ̄/du_i
        lamh = pol.cell_backward(i, np.concatenate([gu, lamh], axis=1), caches[i], grads)
        lamX = dt * (2 * X @ Q[i] + 2 * q[i]) / n_paths + _mtv(E[i], lamX)              # the state is not a network input
    return cost, grads

def train_frnn_policy(pol, pool, n_epochs=75, batch=256, lr=1e-2, lr_decay=lambda k: 1 / (1 + k / 1000), clip=10.0,
                      seed=0, val=None, val_every_epoch=1, verbose=True):
    """Adam on mini-batches drawn without replacement from the fixed training increments `pool` (n_paths, n, D),
    reshuffled every epoch.  val: increments of separate validation paths, evaluated every `val_every_epoch` epochs.
    Returns dict(batch_cost, val_epoch, val_time, val_cost, val_paths, train_time); times count training work only."""
    rng = np.random.default_rng(seed); ps = pol.params()
    m1 = [np.zeros_like(p) for p in ps]; m2 = [np.zeros_like(p) for p in ps]; b1, b2, eps = 0.9, 0.999, 1e-8
    out = dict(batch_cost=[], val_epoch=[], val_time=[], val_cost=[], val_paths=[]); t_train = 0.0; k = 0
    def validate(ep):
        cv, _ = frnn_cost_and_grad(pol, val, need_grad=False)
        out['val_epoch'].append(ep); out['val_time'].append(t_train); out['val_cost'].append(cv.mean()); out['val_paths'].append(cv)
        return cv.mean()
    if val is not None: validate(0)
    n_batches = pool.shape[0] // batch
    for ep in range(1, n_epochs + 1):
        t0 = time.perf_counter(); order = rng.permutation(pool.shape[0]); costs = []
        for bi in range(n_batches):
            k += 1; dB = pool[order[bi * batch:(bi + 1) * batch]]
            c, g = frnn_cost_and_grad(pol, dB)
            gn = np.sqrt(sum((x ** 2).sum() for x in g))
            if gn > clip: g = [x * clip / gn for x in g]
            a = lr * lr_decay(k)
            for j in range(len(ps)):
                m1[j] = b1 * m1[j] + (1 - b1) * g[j]; m2[j] = b2 * m2[j] + (1 - b2) * g[j] ** 2
                ps[j] -= a * (m1[j] / (1 - b1 ** k)) / (np.sqrt(m2[j] / (1 - b2 ** k)) + eps)
            pol.set_params(ps); costs.append(c.mean())
        out['batch_cost'].append(np.mean(costs)); t_train += time.perf_counter() - t0
        if val is not None and (ep % val_every_epoch == 0 or ep == n_epochs):
            cv = validate(ep)
            if verbose and (ep % 5 == 0 or ep == 1 or ep == n_epochs):
                print(f"epoch {ep:3d} ({k:5d} iterations)  mean batch cost {np.mean(costs):9.4f}  "
                      f"training time {t_train:6.0f}s  validation cost {cv:9.4f}", flush=True)
    out['train_time'] = t_train; out['iterations'] = k
    return {kk: np.array(v) if isinstance(v, list) else v for kk, v in out.items()}

def frnn_control(pol, dB):
    """Callable control(i, t, X, sig) for simulate_driver / evaluate_controls on the increments dB.  At step i it uses
    W_{t_i} = Σ_{k<i} ΔB_k (the noise observed so far) and its stored hidden state (reset at i = 0)."""
    state = {}; D = pol.model.D
    def ctrl(i, t, X, sig):
        if i == 0: state['h'] = np.zeros((X.shape[0], D)); state['W'] = np.zeros((X.shape[0], D))
        else: state['W'] = state['W'] + dB[:, i - 1]
        u, state['h'], _ = pol.cell(i, state['W'], state['h'])
        return u
    return ctrl
