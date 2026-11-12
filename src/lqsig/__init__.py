"""Signature methods for linear-quadratic stochastic control: library shared by the notebooks.

Modules
-------
tensors, shuffles   truncated tensor algebra, expected Brownian signature, shuffle-pairing matrices
model, state, cost  controlled system, state tensors, the cost as an exact quadratic and its minimiser
blackbox            the paper's explicit J^L(u) and polarisation (checks)
riccati             Riccati benchmark and feedback
montecarlo, plots   simulation, paired Monte Carlo statistics, tables and figures
fbm                 fractional Brownian motion: simulation, DPD benchmark, Wong-Zakai step, empirical functionals
frnn                fully recurrent network benchmark (Horvath, Teichmann & Zuric, 2021)
style, cache        figure style / LaTeX switch, result cache for the compute-then-plot workflow
"""
__version__ = "0.1.0"

from .style import setup_matplotlib, tex, tex_text, set_figure_dir, save_figure
from .cache import ResultCache
from .tensors import TensorSpace, to_levels, from_levels, tensor_product_levels, expected_signature_bm
from .shuffles import _shuffle_cache, _shuffle_index, shuffle_levels, shuffle_product, shuffle_pair_matrix, pairing_matrix, running_functional
from .model import LQModel, build_q, build_p
from .state import apply_q, truncate, solve_state_picard, solve_state_levelwise, resolvent, solve_state_resolvent, SOLVERS
from .cost import CostFunctionals, _quad, assemble_quadratic, control_basis, state_affine, solve_signature_lq, theta_to_u, cost_of_theta
from .blackbox import concat_time, word_tensor, J_tensor_explicit, blackbox_cost, polarisation_coefficients
from .riccati import ito_coefficients, riccati_benchmark, riccati_feedback
from .montecarlo import signature_stream, simulate, mean_ci, fmt_mc, l2_ci, fmt_l2, signature_control, simulate_many, mc_grid, mc_mean, rel_excess_ci, rel_l2_ci, _triple, print_grid
from .plots import plot_band, grid_figure, plot_case, _is_benchmark, _path_style, compare_paths
from .fbm import fbm_cholesky, fbm_increments, dpd_benchmark, _wz_step_matrices, simulate_driver, simulate_many_driver, EmpiricalCostFunctionals, evaluate_controls
from .frnn import _mv, _mtv, _expm_batched, step_matrices, FRNNPolicy, frnn_cost_and_grad, train_frnn_policy, frnn_control
