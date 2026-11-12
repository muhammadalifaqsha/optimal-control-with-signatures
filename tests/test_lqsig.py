"""Quick consistency checks of the library (a few seconds; run with `pytest` from the repository root)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lqsig as lq                                                             # noqa: E402


@pytest.fixture
def model():
    return lq.LQModel(T=1.0, X0=[10.0, -5.0], b0=[2.0, 0.0], b1=[[1.0, 0.0], [0.2, 0.8]], b2=[[-0.3, 0.7], [0.5, -0.5]],
                      sigma0=[[0.9, 0.1], [0.0, 1.0]], sigma2=[[[0.60, 0.0], [0.0, 0.4]], [[0.1, 0.30], [-0.4, 0.2]]],
                      A=[0 * np.eye(2)], B=[0.5 * np.eye(2), 0.5 * np.eye(2)], C=[[0.0, 0.0]], Dlin=[[0.0, 0.0]],
                      E=np.eye(2), G=[0.0, 0.0])


def test_shuffle_pairing_matrix():
    rng = np.random.default_rng(0); d, L = 3, 3; ts = lq.TensorSpace(d, L)
    g = [rng.normal(size=d ** n) for n in range(2 * L + 1)]
    x, y = rng.normal(size=ts.dim), rng.normal(size=ts.dim)
    lhs = sum(a @ b for a, b in zip(lq.shuffle_product(lq.to_levels(x, ts), lq.to_levels(y, ts), 2 * L, d), g))
    assert np.isclose(lhs, x @ lq.pairing_matrix(g, L, L, d) @ y)


def test_expected_signature_moments():
    ES = lq.expected_signature_bm(2, 1.0, 8)
    G0 = lq.running_functional(ES, 0, 3, 3)
    assert np.isclose(G0[1][0], 0.5) and np.isclose(G0[2][4], 0.25)
    assert np.isclose(lq.running_functional(ES, 1, 2, 3)[1][0], 1 / 3)


def test_state_solvers_agree(model):
    L, M = 4, 2; ts = lq.TensorSpace(model.d, L); q = lq.build_q(model); p0, pU = lq.build_p(model, L, M)
    p = np.concatenate([p0[None], pU])
    ref = lq.solve_state_levelwise(p, q, ts)
    for name in ("picard", "resolvent"):
        assert np.abs(lq.SOLVERS[name](p, q, ts) - ref).max() < 1e-10


def test_exact_quadratic_matches_blackbox(model):
    L, M = 3, 1
    r = lq.solve_signature_lq(model, L, M)
    ES = lq.expected_signature_bm(model.D, model.T, 2 * L + model.m_max + 1)
    c0, h, H = lq.polarisation_coefficients(lambda th: lq.blackbox_cost(model, th, L, M, ES, solver='picard'), len(r['theta']))
    assert abs(c0 - r['c0']) < 1e-8 * max(1, abs(c0))
    assert np.abs(h - r['h']).max() < 1e-7 * max(1, np.abs(h).max())
    assert np.abs(H - r['H']).max() < 1e-7 * max(1, np.abs(H).max())


def test_frnn_gradient_and_consistency(model):
    H, n = 0.3, 20; chol = lq.fbm_cholesky(H, n, model.T)
    dB = lq.fbm_increments(H, 32, n, model.D, model.T, np.random.default_rng(40), chol=chol)
    pol = lq.FRNNPolicy(model, n, width=5, seed=41); pol.W3 *= 3
    c, g = lq.frnn_cost_and_grad(pol, dB); ps = pol.params(); rng = np.random.default_rng(42); err = 0.0
    for j in range(len(ps)):
        for _ in range(3):
            idx = tuple(rng.integers(s) for s in ps[j].shape); old = ps[j][idx]
            ps[j][idx] = old + 1e-6; cp = lq.frnn_cost_and_grad(pol, dB, need_grad=False)[0].mean()
            ps[j][idx] = old - 1e-6; cm = lq.frnn_cost_and_grad(pol, dB, need_grad=False)[0].mean(); ps[j][idx] = old
            fd = (cp - cm) / 2e-6; err = max(err, abs(fd - g[j][idx]) / (abs(fd) + 1e-6))
    assert err < 1e-3
    cs = lq.simulate_driver(model, dB, model.T / n, lq.frnn_control(pol, dB))[0]
    assert np.abs(cs - c).max() / np.abs(c).max() < 1e-12


def test_result_cache_roundtrip(tmp_path):
    path = str(tmp_path / "c.pkl")
    a = lq.ResultCache("compute", path); a.save("sec", x=np.arange(3), y={"k": 1.5})
    b = lq.ResultCache("load", path); v = b.load("sec")
    assert np.array_equal(v["x"], np.arange(3)) and v["y"] == {"k": 1.5}
    with pytest.raises(KeyError):
        b.load("missing")
