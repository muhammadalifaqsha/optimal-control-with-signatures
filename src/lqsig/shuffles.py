"""Shuffle products and shuffle-pairing matrices <v ш w, g> used to assemble the cost functionals."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from .tensors import TensorSpace


_shuffle_cache = {}

def _shuffle_index(p, q, d):
    """For each shuffle (choice of positions of the first word) the output index of (v, w)."""
    key = (p, q, d)
    if key in _shuffle_cache: return _shuffle_cache[key]
    n = p + q
    vdig = np.array(list(itertools.product(range(d), repeat=p)), dtype=np.int64).reshape(d**p, p)
    wdig = np.array(list(itertools.product(range(d), repeat=q)), dtype=np.int64).reshape(d**q, q)
    idx = []
    for pos in itertools.combinations(range(n), p):
        rest = [i for i in range(n) if i not in pos]
        vc = (vdig * (d ** (n - 1 - np.array(pos, dtype=np.int64)))).sum(1) if p else np.zeros(1, np.int64)
        wc = (wdig * (d ** (n - 1 - np.array(rest, dtype=np.int64)))).sum(1) if q else np.zeros(1, np.int64)
        idx.append((vc[:, None] + wc[None, :]).ravel())
    _shuffle_cache[key] = idx
    return idx

def shuffle_levels(xp, yq, p, q, d):
    out = np.zeros(d ** (p + q))
    prod = np.outer(xp, yq).ravel()
    for idx in _shuffle_index(p, q, d):
        np.add.at(out, idx, prod)
    return out

def shuffle_product(x, y, Lout, d):
    """x ш y truncated at Lout; x, y lists of level arrays."""
    out = [np.zeros(d**n) for n in range(Lout + 1)]
    for p, xp in enumerate(x):
        if not np.any(xp): continue
        for q, yq in enumerate(y):
            if p + q > Lout or not np.any(yq): continue
            out[p + q] += shuffle_levels(xp, yq, p, q, d)
    return out

def shuffle_pair_matrix(g, p, q, d):
    """g: (B, d^(p+q)).  Returns M (B, d^p, d^q) with M[b, v, w] = <v ш w, g_b>.
    Recursion  <va ш wb, g> = <v ш wb, g|a> + <va ш w, g|b>,  g|c[u] := g[uc],
    evaluated by dynamic programming over (|v|, |w|), one anti-diagonal at a time."""
    g = np.atleast_2d(g); B0 = g.shape[0]; n = p + q
    # stripped[k] has batch index (b, a1, ..., ak) with a1 the last stripped letter
    stripped = [g]
    for k in range(n):
        x = stripped[-1]
        stripped.append(x.reshape(x.shape[0], d ** (n - k - 1), d).transpose(0, 2, 1)
                         .reshape(x.shape[0] * d, d ** (n - k - 1)))
    prev = {}
    for s in range(0, n + 1):                       # s = pp + qq
        gk = stripped[n - s]; Bk = gk.shape[0]; cur = {}
        for pp in range(max(0, s - q), min(p, s) + 1):
            qq = s - pp
            if pp == 0:   cur[(pp, qq)] = gk.reshape(Bk, 1, d**qq); continue
            if qq == 0:   cur[(pp, qq)] = gk.reshape(Bk, d**pp, 1); continue
            T1 = prev[(pp - 1, qq)].reshape(Bk, d, d**(pp - 1), d**qq).transpose(0, 2, 1, 3).reshape(Bk, d**pp, d**qq)
            T2 = prev[(pp, qq - 1)].reshape(Bk, d, d**pp, d**(qq - 1)).transpose(0, 2, 3, 1).reshape(Bk, d**pp, d**qq)
            cur[(pp, qq)] = T1 + T2
        prev = cur
    return prev[(p, q)]

def pairing_matrix(g_levels, Lr, Lc, d):
    """G[v, w] = <v ш w, g> for all words |v| <= Lr, |w| <= Lc (flat layout)."""
    tr, tc = TensorSpace(d, Lr), TensorSpace(d, Lc)
    G = np.zeros((tr.dim, tc.dim))
    for p in range(Lr + 1):
        for q in range(Lc + 1):
            G[tr.sl(p), tc.sl(q)] = shuffle_pair_matrix(g_levels[p + q][None], p, q, d)[0]
    return G

def running_functional(ES, m, Lmax, d):
    """Γ^(m) with  <u, Γ^(m)> = <(1^{⊗m} ш u) ⊗ 1, E[Ŵ_T]> = E ∫_0^T t^m/m! <u, Ŵ_t> dt   (levels 0..Lmax)."""
    out = []
    for n in range(Lmax + 1):
        h = ES[m + n + 1].reshape(d ** (m + n), d)[:, 0]        # strip final time letter
        out.append(shuffle_pair_matrix(h[None], m, n, d)[0, 0, :])   # row of word 1^{⊗m} (index 0)
    return out
