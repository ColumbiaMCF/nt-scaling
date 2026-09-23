"""Log-linear power-law regression of confinement times.

The scaling laws are power laws, tau = C * M^m * prod_i(x_i^a_i), fit in
log space: ln tau = ln C + m*ln M + sum_i(a_i * ln x_i). Any subset of
exponents can be frozen (the geometry exponents rsurf/kappa_a/eps are frozen
at their IPB98(y,2) values in the NT analysis).

Two weighting schemes:

* ``'ols'`` — ordinary (unweighted) least squares.
* ``'kde'`` — kernel-density-estimate weighting: a Gaussian KDE (Scott
  bandwidth) over regressor space assigns each point ``sigma =
  sqrt(pdf/pdf.max())``, so densely-sampled regions of parameter space get
  larger sigma and thus less weight. The sigma = sqrt(w) convention is
  intentional — do not "fix" it to 1/weights.

Three choices make the KDE weighting robust:

* the KDE is built over the **free** regressors only (default), not the
  near-constant frozen-geometry columns, which inflate the dimensionality
  of the density estimate without carrying sampling information;
* the density is **floored at its 5th percentile** before inverting, so a
  handful of isolated points cannot take unbounded leverage;
* every weighted fit reports its **effective sample size**,
  ESS = (sum w)^2 / sum(w^2) with w the least-squares weights.

Both the KDE columns (``kde_vars``) and the floor (``kde_cap_percentile``)
are exposed as options of :func:`fit_power_law`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

from .reference import MASS_DEUTERIUM

#: Identity columns carried through prepare_xy for labeling/styling (those
#: present in the dataset; 'tok' drives per-group plotting).
ID_COLUMNS = ('tok', 'shotnum', 'tstart', 'tend')


def build_log_linear_model(regressors: List[str],
                           fixed_exponents: Optional[Dict[str, float]] = None,
                           ln_offset: float = 0.0):
    """Build a log-space power-law model for ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    regressors : ln_* column names, in the order of the design-matrix columns.
    fixed_exponents : subset of ``regressors`` whose exponents are frozen.
    ln_offset : constant additive term (carries the isotope factor,
        e.g. ``0.19*np.log(2)``).

    Returns
    -------
    (model, free_regressors) where ``model(x, *free_exponents, c)`` returns
    ln tau, with free exponents in ``free_regressors`` order and the linear
    coefficient ``c`` last.

    Example
    -------
    >>> model, free = build_log_linear_model(
    ...     ['ln_ip', 'ln_rsurf'], fixed_exponents={'ln_rsurf': 1.97})
    >>> free
    ['ln_ip']
    >>> model(np.log([[1.0, 1.7]]), 0.93, 0.0562)   # ln tau at Ip=1, R0=1.7
    """
    fixed = fixed_exponents or {}
    unknown = set(fixed) - set(regressors)
    if unknown:
        raise ValueError(f'fixed exponents not among regressors: {unknown}')
    template = np.array([fixed.get(var, 0.0) for var in regressors])
    free_idx = np.array([i for i, var in enumerate(regressors)
                         if var not in fixed], dtype=int)
    free_regressors = [regressors[i] for i in free_idx]

    def model(x, *params):
        exponents = template.copy()
        exponents[free_idx] = params[:-1]
        with np.errstate(invalid='ignore'):  # curve_fit may probe c <= 0
            return (np.log(params[-1]) + ln_offset
                    + np.sum(np.multiply(x, exponents), axis=1))

    return model, free_regressors


def prepare_xy(df: pd.DataFrame, target: str, regressors: List[str]):
    """Drop rows with NaN in target/regressors; split into X and y frames.

    X keeps the :data:`ID_COLUMNS` present in ``df`` for labeling; y is a
    single-column frame.
    """
    id_columns = [col for col in ID_COLUMNS if col in df.columns]
    slim = df[id_columns + [target] + regressors].dropna()
    X = slim[id_columns + regressors].reset_index(drop=True)
    y = slim[[target]].reset_index(drop=True)
    return X, y


def kde_weights(X_fit: np.ndarray,
                cap_percentile: Optional[float] = None) -> np.ndarray:
    """Normalized KDE density of each sample in regressor space (0..1].

    ``cap_percentile`` floors the density at that percentile of itself
    before normalizing, bounding the maximum least-squares weight (which is
    proportional to 1/density) — without it, isolated points get unbounded
    leverage. ``None`` applies no floor.

    Example
    -------
    >>> X = df[['ln_ip', 'ln_bcentr', 'ln_density', 'ln_ptot_cps']].dropna().to_numpy()
    >>> w = kde_weights(X, cap_percentile=5.0)   # 1 = densest region
    """
    kernel = gaussian_kde(X_fit.T, bw_method='scott')
    pdf = kernel.evaluate(X_fit.T)
    if cap_percentile is not None:
        pdf = np.maximum(pdf, np.percentile(pdf, cap_percentile))
    return pdf / pdf.max()


@dataclass
class FitResult:
    """A fitted power-law scaling, returned by :func:`fit_power_law`.

    Example
    -------
    >>> fit = fit_power_law(df, 'ln_tauth_ptot', ENGINEERING_REGRESSORS,
    ...                     fixed_exponents=FIXED_GEOMETRY_EXPONENTS,
    ...                     isotope_exponent=0.19, weighting='kde')
    >>> print(fit.summary())          # exponents +/- 1 sigma, C, R^2, ESS
    >>> fit.exponents['ln_ip']        # free exponents by regressor name
    >>> fit.plain_exponents['ip']     # same, without the ln_ prefix
    >>> X, y = fit.predict(df[df.f_gw > 1])   # evaluate on other data
    """
    target: str
    regressors: List[str]
    fixed_exponents: Dict[str, float]
    ln_offset: float
    weighting: str
    exponents: Dict[str, float]      # free exponents by regressor name
    stdevs: Dict[str, float]         # 1-sigma from the fit covariance
    coeff: float                     # linear coefficient C
    coeff_std: float
    r2: float                        # unweighted, in log space
    X: pd.DataFrame                  # id columns + regressors (fit sample)
    y: pd.DataFrame                  # target + 'pred' columns (log space)
    params: np.ndarray = field(repr=False)  # raw curve_fit params
    model: Callable = field(repr=False)
    ess: Optional[float] = None      # effective sample size (weighted fits)
    kde_vars: Optional[List[str]] = None  # columns the KDE was built on

    @property
    def plain_exponents(self) -> Dict[str, float]:
        """Free exponents keyed by plain column name (ln_ prefix stripped)."""
        return {key[3:] if key.startswith('ln_') else key: value
                for key, value in self.exponents.items()}

    def summary(self) -> str:
        head = f'{self.target} ~ {self.weighting.upper()} power-law fit '
        if self.ess is not None:
            head += (f'({len(self.y)} points, effective N = {self.ess:.0f}; '
                     f'KDE over {len(self.kde_vars)} vars)')
        else:
            head += f'({len(self.y)} points)'
        lines = [head]
        for var in self.regressors:
            if var in self.exponents:
                lines.append(f'  a_{var}: {self.exponents[var]:.2f} '
                             f'+/- {self.stdevs[var]:.2f}')
            else:
                lines.append(f'  a_{var}: {self.fixed_exponents[var]:.2f} '
                             f'(fixed)')
        lines.append(f'  C: {self.coeff:.2e} +/- {self.coeff_std:.2e}')
        lines.append(f'  R^2: {self.r2:.3f}')
        return '\n'.join(lines)

    def predict(self, df: pd.DataFrame):
        """Evaluate this fit on another DataFrame (e.g. the excluded phases).

        Returns (X, y) frames in the same layout as ``self.X``/``self.y``
        (y holds the measured target and the 'pred' column, both in log
        space), ready for the plotting functions.
        """
        X, y = prepare_xy(df, self.target, self.regressors)
        y = y.copy()
        y['pred'] = self.model(X[self.regressors].to_numpy(), *self.params)
        return X, y


def fit_power_law(df: pd.DataFrame, target: str, regressors: List[str], *,
                  fixed_exponents: Optional[Dict[str, float]] = None,
                  isotope_exponent: float = 0.0,
                  mass: float = MASS_DEUTERIUM,
                  weighting: str = 'ols',
                  kde_vars: Optional[List[str]] = None,
                  kde_cap_percentile: Optional[float] = 5.0,
                  verbose: bool = True) -> FitResult:
    """Fit ln(target) = ln C + isotope_exponent*ln(mass) + sum a_i*ln x_i.

    The single entry point for all scaling fits. ``target`` and
    ``regressors`` are ``ln_*`` column names; any regressor listed in
    ``fixed_exponents`` is held at the given exponent instead of being fit.
    ``weighting`` is ``'ols'`` or ``'kde'`` (see module docstring).

    KDE options (used only when ``weighting='kde'``):

    * ``kde_vars`` — columns the density estimate is built on. Default
      ``None`` = the free (fitted) regressors, excluding frozen-geometry
      columns. Pass the full regressor list to include the frozen columns
      in the density estimate.
    * ``kde_cap_percentile`` — density floor percentile bounding the maximum
      point weight (default 5.0). ``None`` = no floor.

    Example
    -------
    The headline NT thermal scaling — geometry frozen at IPB98(y,2) values,
    deuterium isotope factor 2^0.19, KDE weighting:

    >>> from scripts import (load_nt_shotlist, fit_power_law,
    ...                      ENGINEERING_REGRESSORS, FIXED_GEOMETRY_EXPONENTS)
    >>> df = load_nt_shotlist()
    >>> fit = fit_power_law(df[df.f_gw < 1], 'ln_tauth_ptot',
    ...                     ENGINEERING_REGRESSORS,
    ...                     fixed_exponents=FIXED_GEOMETRY_EXPONENTS,
    ...                     isotope_exponent=0.19, weighting='kde')
    ln_tauth_ptot ~ KDE power-law fit (260 points, effective N = 156; ...)
      a_ln_ip: 1.06 +/- 0.06
      ...

    Any other combination plugs in the same way, e.g. an unweighted fit of
    the total confinement time against current and power only:

    >>> fit = fit_power_law(df, 'ln_taue_ptot', ['ln_ip', 'ln_ptot_cps'])
    """
    if weighting not in ('ols', 'kde'):
        raise ValueError(f"weighting must be 'ols' or 'kde', got {weighting!r}")

    ln_offset = isotope_exponent * np.log(mass)
    model, free_regressors = build_log_linear_model(
        regressors, fixed_exponents, ln_offset)

    X, y = prepare_xy(df, target, regressors)
    X_fit = X[regressors].to_numpy()
    y_fit = y[target].to_numpy()

    p0 = np.zeros(len(free_regressors) + 1)
    p0[-1] = 1

    sigma, ess, kde_cols = None, None, None
    if weighting == 'kde':
        kde_cols = list(free_regressors) if kde_vars is None else list(kde_vars)
        missing = set(kde_cols) - set(regressors)
        if missing:
            raise ValueError(f'kde_vars not among regressors: {missing}')
        col_idx = [regressors.index(v) for v in kde_cols]
        w_kde = kde_weights(X_fit[:, col_idx], cap_percentile=kde_cap_percentile)
        sigma = np.sqrt(w_kde)
        ls_weights = 1.0 / w_kde          # curve_fit weight ~ 1/sigma^2
        ess = ls_weights.sum() ** 2 / (ls_weights ** 2).sum()

    params, cov = curve_fit(f=model, xdata=X_fit, ydata=y_fit, p0=p0,
                            sigma=sigma, bounds=(-np.inf, np.inf))
    stdevs = np.sqrt(np.diag(cov))

    y = y.copy()
    y['pred'] = model(X_fit, *params)
    r2 = r2_score(y_fit, y['pred'])

    result = FitResult(
        target=target, regressors=list(regressors),
        fixed_exponents=dict(fixed_exponents or {}), ln_offset=ln_offset,
        weighting=weighting,
        exponents=dict(zip(free_regressors, params[:-1])),
        stdevs=dict(zip(free_regressors, stdevs[:-1])),
        coeff=params[-1], coeff_std=stdevs[-1], r2=r2,
        X=X, y=y, params=params, model=model,
        ess=ess, kde_vars=kde_cols,
    )
    if verbose:
        print(result.summary())
    return result
