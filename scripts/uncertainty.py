"""Uncertainty quantification by subset bootstrapping.

The statistical goodness-of-fit of a single regression understates the real
uncertainty of the scaling exponents, which is dominated by the completeness
of the underlying database. Uncertainty is therefore characterized by
refitting on random subsets (75% and 50% of the data, 1000 repeats each in
the NT analysis) and presenting the exponents as distributions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .fitting import fit_power_law
from .reference import ISOTOPE_EXPONENT_THERMAL


@dataclass
class BootstrapResult:
    """Distributions from repeated fits on random subsets of the database.

    Returned by :func:`subset_bootstrap`. All sample arrays have length
    ``num_repeats``; ``exponent_samples`` is keyed by ln_ regressor name.

    Example
    -------
    >>> boot = subset_bootstrap(df, 'ln_tauth_ptot', ENGINEERING_REGRESSORS,
    ...                         fixed_exponents=FIXED_GEOMETRY_EXPONENTS)
    >>> boot.exponent_samples['ln_ip'].mean()      # bootstrap-mean exponent
    >>> boot.exponent_mean_std()                   # {name: (mean, std)}
    """
    frac: float
    num_repeats: int
    weighting: str
    free_regressors: List[str]
    exponent_samples: Dict[str, np.ndarray]   # keyed by ln_ regressor name
    coeff_samples: np.ndarray
    r2_samples: np.ndarray
    n_retries: int = 0

    def exponent_mean_std(self):
        return {name: (samples.mean(), samples.std())
                for name, samples in self.exponent_samples.items()}


def subset_bootstrap(df: pd.DataFrame, target: str,
                     regressors: List[str], *,
                     fixed_exponents: Dict[str, float],
                     isotope_exponent: float = ISOTOPE_EXPONENT_THERMAL,
                     frac: float = 0.75, num_repeats: int = 1000,
                     weighting: str = 'kde',
                     kde_vars: Optional[List[str]] = None,
                     kde_cap_percentile: Optional[float] = 5.0,
                     rng: Optional[np.random.Generator] = None,
                     max_retries: int = 50) -> BootstrapResult:
    """Refit the scaling on ``num_repeats`` random ``frac`` subsets.

    Each repeat records the free exponents, coefficient, and R^2 of a
    :func:`scripts.fitting.fit_power_law` call with the same options
    (including the KDE options). Singular KDE subsets are re-drawn (counted
    in ``n_retries``). Pass a seeded ``rng`` for reproducibility.

    Example
    -------
    >>> boot = subset_bootstrap(
    ...     fit_df, 'ln_tauth_ptot', ENGINEERING_REGRESSORS,
    ...     fixed_exponents=FIXED_GEOMETRY_EXPONENTS,
    ...     isotope_exponent=0.19, frac=0.5, num_repeats=1000,
    ...     weighting='kde', rng=np.random.default_rng(2023))
    >>> boot.exponent_mean_std()['ln_ip']          # ~(1.03, 0.07)
    """
    rng = rng or np.random.default_rng()
    samples: List[np.ndarray] = []
    r2s = np.zeros(num_repeats)
    free_regressors: List[str] = []
    n_retries = 0

    for i in range(num_repeats):
        for _ in range(max_retries):
            try:
                sub = df.sample(frac=frac, random_state=rng)
                fit = fit_power_law(
                    sub, target, regressors,
                    fixed_exponents=fixed_exponents,
                    isotope_exponent=isotope_exponent,
                    weighting=weighting, kde_vars=kde_vars,
                    kde_cap_percentile=kde_cap_percentile, verbose=False)
                break
            except (np.linalg.LinAlgError, RuntimeError):
                n_retries += 1
        else:
            raise RuntimeError(f'fit failed {max_retries} times at repeat {i}')

        free_regressors = list(fit.exponents)
        samples.append(fit.params)
        r2s[i] = fit.r2

    all_params = np.array(samples).T  # (n_free + 1, num_repeats)
    exponent_samples = {name: all_params[j]
                        for j, name in enumerate(free_regressors)}
    return BootstrapResult(
        frac=frac, num_repeats=num_repeats, weighting=weighting,
        free_regressors=free_regressors, exponent_samples=exponent_samples,
        coeff_samples=all_params[-1], r2_samples=r2s, n_retries=n_retries)
