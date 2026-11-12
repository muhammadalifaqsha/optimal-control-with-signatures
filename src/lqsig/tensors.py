"""Truncated tensor algebra T^L(R^d) in a flat layout, and the expected signature of time-augmented Brownian motion (Fawcett)."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


class TensorSpace:
    """Truncated tensor algebra T^L(R^d), stored flat: level 0, level 1, ..., level L.
    Word (i1..im) at level m has flat index offset[m] + sum_k i_k d^(m-k)  (row-major)."""
    def __init__(self, d, L):
        self.d, self.L = d, L
        self.sizes = [d**m for m in range(L + 1)]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int)
        self.dim = int(self.offsets[-1])
    def sl(self, m):
        return slice(self.offsets[m], self.offsets[m + 1])
    def index(self, word):
        m = len(word); i = 0
        for a in word: i = i * self.d + a
        return int(self.offsets[m] + i)
    def words(self):
        return [w for m in range(self.L + 1) for w in itertools.product(range(self.d), repeat=m)]
    @staticmethod
    def word_str(w):
        return "∅" if len(w) == 0 else "".join(str(a + 1) for a in w)   # paper's letters: 1 = time

def to_levels(x, ts):
    return [x[..., ts.sl(m)] for m in range(ts.L + 1)]

def from_levels(levels):
    return np.concatenate(levels, axis=-1)

def tensor_product_levels(x, y, Lout, d):
    """(x ⊗ y) truncated at Lout; x, y are lists of level arrays (no batch)."""
    out = [np.zeros(d**n) for n in range(Lout + 1)]
    for p, xp in enumerate(x):
        if p > Lout or not np.any(xp): continue
        for q, yq in enumerate(y):
            if p + q > Lout or not np.any(yq): continue
            out[p + q] += np.outer(xp, yq).ravel()
    return out

def expected_signature_bm(D, T, Lmax):
    """E[Ŵ_T] = exp_⊗( T (e_time + 1/2 Σ_d e_d⊗e_d) ), letters: 0 = time, 1..D = W^(1..D)."""
    d = D + 1
    Q = [np.zeros(d**n) for n in range(Lmax + 1)]
    if Lmax >= 1: Q[1][0] = 1.0
    if Lmax >= 2:
        for a in range(1, d): Q[2][a * d + a] = 0.5
    ES = [np.zeros(d**n) for n in range(Lmax + 1)]; ES[0][0] = 1.0
    term = [lv.copy() for lv in ES]
    for k in range(1, Lmax + 1):
        term = tensor_product_levels(term, Q, Lmax, d)
        term = [t * (T / k) for t in term]
        ES = [a + b for a, b in zip(ES, term)]
    return ES
