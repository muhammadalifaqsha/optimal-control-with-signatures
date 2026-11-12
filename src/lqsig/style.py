"""Matplotlib style, the LaTeX switch, label helpers and figure saving."""
import os

import matplotlib.pyplot as plt

_STATE = {"use_tex": False, "fig_dir": "figures"}

RC_PARAMS = {'font.size': 14,
             'legend.fontsize': 'medium',
             'figure.figsize': (15, 5),
             'axes.labelsize': 'x-large',
             'axes.titlesize': 'x-large',
             'figure.titlesize': 'x-large',
             'xtick.labelsize': 'large',
             'ytick.labelsize': 'large'}


def setup_matplotlib(use_tex=True):
    """Apply the figure style of the notebooks.  use_tex=True renders all text with LaTeX (needs a LaTeX
    installation with amsmath, amsfonts and cm-super); use_tex=False uses matplotlib's built-in mathtext."""
    plt.rcParams.update(RC_PARAMS)
    plt.rc('text', usetex=use_tex)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsfonts}')
    _STATE["use_tex"] = bool(use_tex)


def tex(s):
    """Label strings for both renderers: without LaTeX, replace the LaTeX-only \\text{...} dashes that
    matplotlib's mathtext cannot render (everything else used in the figures works in both)."""
    return s if _STATE["use_tex"] else s.replace(r"\text{--}", "–").replace(r"\text{-}", "-")


def tex_text(s):
    """Plain text -> upright LaTeX text, $\\mathrm{...}$ (spaces, underscores, colons and % escaped)."""
    for a, b in [('_', r'\_'), ('%', r'\%'), (':', '{:}'), (' ', r'\ ')]:
        s = s.replace(a, b)
    return r"$\mathrm{" + s + "}$"


def set_figure_dir(path):
    """Folder in which save_figure writes the PDFs (created if needed)."""
    os.makedirs(path, exist_ok=True)
    _STATE["fig_dir"] = path
    return path


def save_figure(fig, name):
    """Save a whole figure (all of its subplots) as a single PDF in the figure folder."""
    fig.savefig(os.path.join(_STATE["fig_dir"], name + ".pdf"), bbox_inches='tight')
