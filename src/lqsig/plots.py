"""Figures over the (L, M) grid and sample-path comparisons."""
import itertools, time, gc
from math import factorial, comb
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from .style import save_figure
from .montecarlo import _triple, l2_ci, mc_mean, rel_excess_ci, rel_l2_ci


def plot_band(ax, x, mean, lo, hi, label=None, fmt='o-', **kw):
    line, = ax.plot(x, mean, fmt, label=label, **kw)
    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.2, lw=0)
    return line

def grid_figure(Ls, Ms, value, ylabel, what, prefix, fname, hlines=(), log=False, Ls_vs_M=None, lev='L', font1 = 25, font2 = 14):
    """One figure with two panels side by side: value(L, M) vs M (one line per L) | vs L (one line per M).
    value returns a float (exact) or a (mean, lower, upper) triple (Monte Carlo, drawn with its 95% band);
    hlines: [((mean, lower, upper) or (value, None, None), label, line kwargs)] drawn in both panels.
    log='auto' uses a log axis when every plotted value is positive.  lev: symbol of the state level, printed as
    \\mathrm{lev}.  Saved as FIG_DIR/fname.pdf."""
    Lsym = r"\mathrm{" + lev + "}"
    Ls_vs_M = Ls if Ls_vs_M is None else Ls_vs_M
    if log == 'auto':
        log = all(np.ravel(value(L, M))[0] > 0 for L in Ls for M in Ms)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    def series(a, x, vals, label):
        if isinstance(vals[0], tuple):
            v = np.array(vals); plot_band(a, x, v[:, 0], v[:, 1], v[:, 2], label)
        else:
            a.plot(x, vals, 'o-', label=label)
    for L in Ls_vs_M: series(ax[0], Ms, [value(L, M) for M in Ms], rf"${Lsym}={L}$")
    for M in Ms: series(ax[1], Ls, [value(L, M) for L in Ls], rf"$M={M}$")
    for a in ax:
        for (m, lo, hi), lab, kw in hlines:
            a.axhline(m, label=lab, **kw)
            if lo is not None: a.axhspan(lo, hi, color=kw.get('color', 'k'), alpha=0.12, lw=0)
        a.set_ylabel(ylabel, fontsize = font1)
        a.tick_params(axis='both', which='major', labelsize=font1)
        if log: a.set_yscale('log')
        a.legend(ncol=2)
    ax[0].set_xlabel(r"$\mathrm{control\ level}\ M$", fontsize = font1); ax[0].set_xticks(Ms); ax[0].set_xlim(min(Ms), max(Ms))
    ax[1].set_xlabel(r"$\mathrm{state\ level}\ " + Lsym + "$", fontsize = font1); ax[1].set_xticks(Ls); ax[1].set_xlim(min(Ls), max(Ls))
    ax[0].legend(fontsize=font2, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax[1].legend(fontsize=font2, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    #fig.suptitle("$" + prefix + r"{:}\ " + what + "$", fontsize='large')
    fig.tight_layout(); save_figure(fig, fname); plt.show()

def plot_case(mc, Ls, Ms, ref_cost, ref_label, fname, prefix, refs=(), ref_sym="u^*", cost_log=False, Ls_vs_M=None,
              zero_line=False, lev='L'):
    """All (L, M)-grid figures of one case, each saved as FIG_DIR/{fname}_<kind>_vs_M_L.pdf:
      Monte Carlo (shaded: 95% confidence bands):  cost, distance to the benchmark, relative excess cost [%], relative L2 error [%];
    refs: [(per-path costs, label)] of further reference controls simulated on the benchmark's paths (dotted grey).
    lev: symbol of the state level, printed as \\mathrm{lev} (default 'L')."""
    kw = dict(Ls_vs_M=Ls_vs_M, lev=lev)
    uLM = r"u^{\mathrm{" + lev + r"},M}"
    bench = dict(color='k', ls='--', lw=1.2); other = dict(color='0.45', ls=':', lw=1.5)
    dist_y = r"\|" + uLM + "-" + ref_sym + r"\|_{L^2(d\mathbb{P}\otimes dt)}"
    relx_y = r"(J(" + uLM + r")-J(" + ref_sym + r"))/J(" + ref_sym + r")\ [\%]"
    reld_y = r"\|" + uLM + "-" + ref_sym + r"\|/\|" + ref_sym + r"\|\ [\%]"
    ci = "$"
    grid_figure(Ls, Ms, lambda L, M: mc_mean(mc[(L, M)]['cost']), r"$J(" + uLM + ")" + ci, r"\mathrm{cost\ (Monte-Carlo)}", prefix,
                f"{fname}_cost_vs_M_L", [(mc_mean(ref_cost), ref_label, bench)] + [(mc_mean(c), lab, other) for c, lab in refs],
                cost_log, **kw)
    grid_figure(Ls, Ms, lambda L, M: l2_ci(mc[(L, M)]['d2']), "$" + dist_y + ci, r"\mathrm{distance\ to}\ " + ref_sym + r"\ \mathrm{(Monte-Carlo)}",
                prefix, f"{fname}_distance_vs_M_L", log=True, **kw)
    zero = [((0.0, None, None), r"$\mathrm{zero\ (no\ improvement)}$", dict(color='k', ls='-', lw=1.0))] if zero_line else []
    grid_figure(Ls, Ms, lambda L, M: _triple(rel_excess_ci(mc[(L, M)]['diff'], mc[(L, M)]['cref'])), "$" + relx_y + ci,
                r"\mathrm{relative\ excess\ cost\ (Monte-Carlo)}", prefix, f"{fname}_rel_excess_vs_M_L",
                [(_triple(rel_excess_ci(c - ref_cost, ref_cost)), lab, other) for c, lab in refs] + zero,
                False if zero_line else 'auto', **kw)
    grid_figure(Ls, Ms, lambda L, M: _triple(rel_l2_ci(mc[(L, M)]['d2'], mc[(L, M)]['q2'])), "$" + reld_y + ci,
                r"\mathrm{relative}\ L^2\ \mathrm{error\ (Monte-Carlo)}", prefix, f"{fname}_rel_distance_vs_M_L", log=True, **kw)

def _is_benchmark(name):
    return "u^*" in name or "Duncan-Pasik-Duncan" in name

def _path_style(name, k):
    """Benchmark in black; every other control the k-th standard colour (reference feedbacks dashed);
    slightly transparent so that overlapping paths stay distinguishable."""
    if _is_benchmark(name):
        return dict(color='black', lw=2.2, ls='-', alpha=0.7, zorder=0)
    dashed = "HJB-obtained" in name or "of" in name
    return dict(color=f"C{k % 10}", lw=1.6, ls='--' if dashed else '-', alpha=0.6, zorder=4 if dashed else 3)

def compare_paths(model, dW, dt, controls, simulator, path=0, title="", fname=None, font1 = 25, font2 = 14, font3 = 20, **sim_kw,):
    """Drive every control with the SAME noise and plot one sample path of each state component (top row) and each
    control component (bottom row); with a single state and a single control the two panels sit side by side.
    controls: name -> (callable(i, t, X, sig) -> u, signature level it needs); names and title are LaTeX strings.
    simulator: `simulate` (Itô Euler, Brownian) or `simulate_driver` (general driver, e.g. fBM).
    fname saves the figure to FIG_DIR/fname.pdf.
    Returns name -> (cost, X, U) for all simulated paths."""
    lev = max(l for _, l in controls.values())
    sims = {name: simulator(model, dW, dt, ctrl, sig_level=lev, **sim_kw) for name, (ctrl, _) in controls.items()}
    steps = dW.shape[1]; tg = np.linspace(0, model.T, steps + 1)
    if model.N == 1 and model.K == 1:                     # only two panels: arrange them in columns
        fig_h, leg_cols = 5.0, 2
        fig, axs = plt.subplots(1, 2, figsize=(14, fig_h), squeeze=False, sharex=True)
        ax_x, ax_u, ax_t = [axs[0, 0]], [axs[0, 1]], [axs[0, 0], axs[0, 1]]
    else:
        ncol = max(model.N, model.K); fig_h, leg_cols = 7.5, 5.5
        fig, axs = plt.subplots(2, ncol, figsize=(max(6 * ncol, 14), fig_h), squeeze=False, sharex=True)
        ax_x, ax_u, ax_t = list(axs[0, :model.N]), list(axs[1, :model.K]), list(axs[1, :model.K])
        for j in range(model.N, ncol): axs[0, j].axis('off')
        for j in range(model.K, ncol): axs[1, j].axis('off')
    k_col = 0; handles = []
    for name, (cost, X, U) in sims.items():
        st = _path_style(name, k_col)
        if not _is_benchmark(name): k_col += 1
        for n, a in enumerate(ax_x):
            h, = a.plot(tg, X[:, path, n], **st)
            a.tick_params(axis='both', which='major', labelsize=font3)
        for k, a in enumerate(ax_u):
            a.plot(tg[:-1], U[:, path, k], drawstyle='steps-post', **st)
            a.tick_params(axis='both', which='major', labelsize=font3)
        h.set_label(name); handles.append(h)
    for n, a in enumerate(ax_x): a.set_title(rf"$\mathrm{{state}}\ X^{{({n + 1})}}_t$", fontsize = font1)
    for k, a in enumerate(ax_u): a.set_title(rf"$\mathrm{{control}}\ u^{{({k + 1})}}_t$", fontsize = font1)
    for a in ax_t: 
        a.set_xlabel(rf'$\mathrm{{Time}}$ $(t)$',  fontsize = font3)
    fig.suptitle(title, fontsize = font1)
    fig.tight_layout()
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=min(leg_cols, len(handles)),
               fontsize=font2, frameon=False)
    if fname is not None: save_figure(fig, fname)
    plt.show()
    return sims
