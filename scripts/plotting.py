"""Plotting utilities for the NT confinement-scaling analysis.

Every function is pure: it builds and returns matplotlib objects (so figures
can be composed further) and only touches the filesystem when a
``save_path`` is given, in which case the figure is also written there at
300 dpi. Points are grouped and styled by the ``tok`` label column
(:data:`scripts.reference.GROUP_STYLE`).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from .fitting import FitResult
from .reference import group_style
from .uncertainty import BootstrapResult

#: (color, alpha) sequence for overplotted bootstrap fractions, largest
#: fraction first.
FRAC_STYLES = [('black', 1.0), ('blue', 0.66), ('red', 0.5)]


def _split_xy(fit_or_X, y=None):
    """Accept (FitResult) or (X, y) and return (X, y_actual, y_pred).

    The (X, y) form takes a frame X with a 'tok' column and a two-column
    frame y whose first column is measured and second predicted.
    """
    if isinstance(fit_or_X, FitResult):
        X, y = fit_or_X.X, fit_or_X.y
    else:
        X = fit_or_X
    return X, y.iloc[:, 0].to_numpy(), y.iloc[:, 1].to_numpy()


def plot_scaling(fit_or_X, y=None, axis_labels=('Predicted', 'Measured'), *,
                 is_log=True, upper_bd=0.2, add_residuals=True,
                 add_legend=True, annotate_shots=False, extra=None,
                 save_path=None):
    """Measured-vs-predicted confinement time with the y=x line.

    Accepts a :class:`FitResult` directly (log-space fit; leave
    ``is_log=True``) or explicit ``(X, y)`` frames where the first y column
    is measured and the second predicted (e.g. a reference-law comparison
    from linear-space columns, with ``is_log=False``). ``extra=(X2, y2)``
    overlays additional points that were not part of the fit (e.g. the
    f_GW > 1 phases via ``FitResult.predict``), styled by their own ``tok``
    label. ``add_residuals`` adds a residual panel above the main axes.

    Example
    -------
    >>> from scripts import plotting
    >>> fig, ax = plotting.plot_scaling(
    ...     fit, axis_labels=(r'Predicted $\\tau_{E,th}$ [s]',
    ...                       r'Measured $\\tau_{E,th}$ [s]'),
    ...     extra=fit.predict(fgw_df), save_path='figures/kde_tauth.pdf')

    Reference-law comparison from linear-space columns:

    >>> plotting.plot_scaling(df, df[['tauth_cps', 'tauth_98']], is_log=False)
    """
    X, y_actual, y_pred = _split_xy(fit_or_X, y)
    if extra is not None:
        X_extra, y_extra = extra
        X = pd.concat([X, X_extra], ignore_index=True)
        y_actual = np.concatenate([y_actual, y_extra.iloc[:, 0].to_numpy()])
        y_pred = np.concatenate([y_pred, y_extra.iloc[:, 1].to_numpy()])

    if is_log:
        y_actual, y_pred = np.exp(y_actual), np.exp(y_pred)
    residuals = y_actual - y_pred

    if add_residuals:
        fig, (a1, a0) = plt.subplots(
            2, 1, figsize=(5, 6.1), gridspec_kw={'height_ratios': [1, 5]},
            sharex=True, constrained_layout=True)
    else:
        fig = plt.figure(figsize=(5, 5), constrained_layout=True)
        a0 = fig.add_subplot(111)
    a0.set_aspect('equal')

    for label in X['tok'].unique():
        mask = (X['tok'] == label).to_numpy()
        color, marker = group_style(label)
        a0.scatter(x=y_pred[mask], y=y_actual[mask], marker=marker,
                   facecolors=color, edgecolors=color, alpha=0.5, label=label)
        if add_residuals:
            a1.scatter(x=y_pred[mask], y=residuals[mask], marker=marker,
                       color=color, alpha=0.5, label=label)
        if annotate_shots:
            for shot, xp, ya in zip(X['shotnum'][mask], y_pred[mask],
                                    y_actual[mask]):
                a0.annotate(str(shot), (xp, ya), fontsize=3)

    a0.set_xlabel(axis_labels[0])
    a0.set_ylabel(axis_labels[1])

    line = np.array([0, 100])
    a0.plot(line, line, color='k', linestyle='--', linewidth=0.75,
            scalex=False, scaley=False)
    a0.set(xlim=[0, upper_bd], ylim=[0, upper_bd],
           xticks=np.linspace(0, upper_bd, 5),
           yticks=np.linspace(0, upper_bd, 5))
    a0.tick_params(direction='in', which='both', bottom=True, left=True,
                   right=True, top=True)

    if add_residuals:
        a1.hlines(y=0, xmin=0, xmax=100, linestyle='--', color='k',
                  linewidth=0.75)
        if residuals.min() > -0.001:
            a1.set_ylim([-0.05, residuals.max() * 2])
        else:
            a1.set_ylim([residuals.min() * 2, residuals.max() * 2])
        a1.set_ylabel('Residuals')
        a1.tick_params(direction='in', which='both', bottom=True, left=True,
                       right=True, top=True)

    if add_legend:
        legend = a0.legend(loc='best', fancybox=True)
        legend.get_frame().set_alpha(0.5)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, a0


def plot_param_dist(df, var, n_bins=10, xlabel='', *, overlay_kde=False,
                    ax=None, save_path=None):
    """Per-group overlaid histograms of one parameter (+ optional KDE).

    The dashed KDE curve (Scott bandwidth, scaled to the histogram peak) is
    the 1-D analogue of the density used for the KDE fit weighting.

    Example
    -------
    >>> fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    >>> plotting.plot_param_dist(fit_df, 'ip', 15, r'$I_p$ [MA]',
    ...                          overlay_kde=True, ax=axes[0, 0])
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    edges = np.linspace(0.95 * df[var].min(), 1.05 * df[var].max(),
                        n_bins + 1)
    for label in df['tok'].unique():
        color, _ = group_style(label)
        values = df.loc[df['tok'] == label, var].dropna()
        counts = np.histogram(values, bins=edges)[0]
        ax.bar(x=edges[:-1], height=counts, width=np.diff(edges),
               align='edge', color=color, alpha=0.5, label=label)

    if overlay_kde:
        values = df[var].dropna()
        kernel = gaussian_kde(values.T, bw_method='scott')
        counts = np.histogram(values, bins=edges)[0]
        grid = np.linspace(values.min() * 0.95, values.max() * 1.05, 100)
        density = kernel.evaluate(grid.T)
        density = 0.99 * counts.max() * density / density.max()
        ax.plot(grid, density, 'k--', linewidth=1, label='KDE')

    ax.set_xlabel(xlabel or var)
    legend = ax.legend(loc='best', fancybox=True)
    legend.get_frame().set_alpha(0.5)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_correlation_matrix(df, columns, labels=None, *, size=6,
                            save_path=None):
    """|Correlation| matrix of the given columns (regressor collinearity)."""
    corr = abs(df[columns].corr())
    labels = labels if labels is not None else columns

    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap=plt.cm.YlOrRd, vmin=0, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(labels)
    for (i, j), value in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=9)
    fig.colorbar(cax, ax=ax, shrink=0.8)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_scatter_matrix(df, columns, *, bins=15, s=75, figsize=(10, 10),
                        save_path=None):
    """Pairwise scatter matrix of the given columns, colored by group."""
    slim = df[['tok'] + list(columns)].dropna()
    colors = [group_style(label)[0] for label in slim['tok']]
    axes = pd.plotting.scatter_matrix(
        slim[columns], figsize=figsize, color=colors, alpha=0.25, s=s,
        hist_kwds={'edgecolor': 'black', 'linewidth': 0.5, 'bins': bins})
    fig = axes[0, 0].figure
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def _overlay_histograms(ax, samples_by_label: Dict[str, np.ndarray],
                        n_bins: int, reference=None):
    """Overlay histograms on shared bins spanning the pooled samples.
    ``reference`` may be a scalar or a list of dashed-line positions."""
    pooled = np.concatenate([np.ravel(s) for s in samples_by_label.values()])
    edges = np.linspace(pooled.min(), pooled.max(), n_bins)
    for (label, samples), (color, alpha) in zip(samples_by_label.items(),
                                                FRAC_STYLES):
        ax.hist(samples, bins=edges, alpha=alpha, color=color, label=label)
    if reference is not None:
        for ref in np.atleast_1d(reference):
            if ref is not None:
                ax.axvline(ref, color='k', linestyle='dashed', linewidth=1)


def plot_histograms(samples_by_label: Dict[str, np.ndarray], xlabel, *,
                    reference=None, n_bins=13, xlim=None, ax=None,
                    save_path=None):
    """Single-panel overlay histogram (e.g. bootstrap R^2 scores).

    ``samples_by_label`` maps a legend label (e.g. '75%') to a sample array;
    ``reference`` draws dashed vertical lines (scalar or list).

    Example
    -------
    >>> plotting.plot_histograms(
    ...     {label: b.r2_samples for label, b in boots.items()}, r'$R^2$')
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    _overlay_histograms(ax, samples_by_label, n_bins, reference)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set(xlim=xlim)
    legend = ax.legend(loc='best', fancybox=True)
    legend.get_frame().set_alpha(0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_exponent_histograms(results: Dict[str, BootstrapResult],
                             names: List[str], labels: List[str], *,
                             reference: Optional[Sequence[float]] = None,
                             n_bins=13, ncols=2, save_path=None):
    """Grid of bootstrap exponent PDFs, one panel per free exponent.

    ``results`` maps a fraction label ('75%', '50%') to its
    :class:`~scripts.uncertainty.BootstrapResult`; ``names`` selects
    exponents by ln_ regressor name; ``reference`` gives one dashed-line
    value per panel (None entries skip the line).

    Example
    -------
    >>> plotting.plot_exponent_histograms(
    ...     boots, FREE_REGRESSORS, EXPONENT_LABELS,
    ...     reference=[0.93, 0.15, 0.41, -0.69])
    """
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(names):]:
        ax.set_visible(False)

    for i, (name, ax) in enumerate(zip(names, axes)):
        samples = {label: res.exponent_samples[name]
                   for label, res in results.items()}
        ref = reference[i] if reference is not None else None
        _overlay_histograms(ax, samples, n_bins, ref)
        ax.set_xlabel(labels[i])

    legend = axes[0].legend(loc='best', fancybox=True)
    legend.get_frame().set_alpha(0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes
