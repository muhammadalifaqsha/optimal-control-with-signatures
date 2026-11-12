# Signature methods for linear-quadratic stochastic control

Code and notebooks for solving linear-quadratic (LQ) stochastic control problems with path signatures. They extend the 1-dimensional method of *Aqsha, Bank & Sánchez-Betancourt, "Solving Linear-Quadratic Stochastic Control Problems with Signatures"* (arXiv:2602.23473) in three ways:
- to $N$-dimensional states, $D$-dimensional noise and $K$-dimensional controls;
- to fractional Brownian motion (fBM) drivers;
- with neural-network benchmarks.

## Layout

```
.
├── src/lqsig/                  shared library (installable package)
│   ├── tensors.py              truncated tensor algebra, expected signature of Brownian motion
│   ├── shuffles.py             shuffle products and shuffle-pairing matrices
│   ├── model.py                the controlled system and cost (LQModel)
│   ├── state.py                state-tensor solvers (Picard, level-wise, resolvent)
│   ├── cost.py                 the cost as an exact quadratic in θ, closed-form minimiser
│   ├── blackbox.py             the paper's explicit J^L(u) and polarisation (consistency checks)
│   ├── riccati.py              Riccati benchmark and optimal Brownian feedback
│   ├── montecarlo.py           signatures along paths, simulation, paired Monte Carlo statistics, tables
│   ├── plots.py                (L, M)-grid figures and sample-path comparisons
│   ├── fbm.py                  fBM simulation, Duncan–Pasik-Duncan benchmark, Wong–Zakai simulation,
│   │                           empirical cost functionals, paired out-of-sample evaluation
│   ├── frnn.py                 fully recurrent network of Horvath, Teichmann & Žurič (2021), adapted to control
│   ├── style.py                figure style, LaTeX switch, figure saving
│   └── cache.py                result cache for the compute-then-plot workflow
├── notebooks/
│   ├── 01_signature_lq_control.ipynb   N-dimensional method: Brownian 2-D example, additive and multiplicative fBM
│   └── 02_frnn_benchmark.ipynb         multiplicative fBM: signature control (L, M) = (7, 3) vs the fRNN, same data
├── figures/                    PDFs written by the notebooks (one subfolder per notebook)
├── cache/                      cached results (pickle files) of the compute runs
├── tests/test_lqsig.py         quick consistency checks
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .            # optional: makes `import lqsig` work everywhere
```

The notebooks don't need the install step. They find the repository root by looking for `src/lqsig` above the notebook's folder, and add `src/` to the import path.

For LaTeX-rendered figures (`USE_TEX = True`) you need a LaTeX installation with `amsmath`, `amsfonts` and `cm-super`.

## Running the notebooks: compute on one machine, plot on another

Each notebook's first code cell has two settings:

```python
MODE = "compute"      # "compute": run everything and save results to cache/;  "load": read cache/ and only print and plot
USE_TEX = True        # False on machines without LaTeX
```

1. **Fast machine (no LaTeX):** set `MODE = "compute"` and `USE_TEX = False`, then run the notebook. The results of each expensive section are written to `cache/<notebook>.pkl` as soon as that section finishes, so an interrupted run keeps the sections it has completed.
2. **LaTeX machine:** copy the `.pkl` files into `cache/`, set `MODE = "load"` and `USE_TEX = True`, then run the notebook. The expensive computations are skipped. The tables are printed from the cache, and the figures are drawn with LaTeX into `figures/`.

Things to know about the cache:
- **Cheap steps run in both modes.** These are the test noise (regenerated from fixed seeds), the benchmarks, the gradient checks and the sample-path figures.
- **Settings must match.** A load run checks that its settings match those of the cached run.
- **Timings come from the compute machine.** All reported times were measured there.
- **Library versions.** The cache files are Python pickles of NumPy arrays and dictionaries, so use the same major NumPy version (1.x or 2.x) on both machines.

Run times on a 2-core machine:

| notebook | compute run | load run |
|---|---|---|
| `01` | about 30–40 min | about 1 min |
| `02` | about 25 min | under a minute |

In `01`, most of the compute run is the empirical fBM functionals at L = 7 and the Wong–Zakai evaluations. In `02`, it is the functionals plus 75 training epochs of the fRNN.

## Tests

```bash
pytest
```

The tests take a few seconds and check five things:
- the shuffle-pairing identity and the expected-signature moments;
- that the three state solvers agree;
- that the exact quadratic assembly matches the paper's black box with polarisation;
- the fRNN's hand-written gradient against finite differences, and its consistency with `simulate_driver`;
- the cache round trip.

## Notes

- All randomness uses `numpy.random.default_rng(SEED + offset)` with `SEED = 0`, so runs are reproducible.
- The SDEs driven by fBM are read in the rough-path (Wong–Zakai) sense. With multiplicative noise they are simulated with the exact piecewise-linear step (`simulate_driver`, `simulate_many_driver`).
- The networks are implemented in NumPy with hand-written backward passes, so no deep-learning framework is needed.
