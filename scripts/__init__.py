"""Energy-confinement-time scaling for negative-triangularity plasmas.

Data: the stationary NT discharges of Paz-Soldan et al., Nucl. Fusion 64
094002 (2024), doi:10.1088/1741-4326/ad69a4.

The layers, in dependency order:

    reference    IPB98(y,2) / ITER89-P laws, frozen conventions, styling
    data         shot-list loading, unit conversion, derived columns
    fitting      log-linear power-law regression (OLS / KDE-weighted)
    uncertainty  subset bootstraps
    plotting     figures for all of the above

Quickstart (from a notebook in runs/):

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))

    import numpy as np
    from scripts import (load_nt_shotlist, fit_power_law, subset_bootstrap,
                         ENGINEERING_REGRESSORS, FIXED_GEOMETRY_EXPONENTS,
                         ISOTOPE_EXPONENT_THERMAL, plotting)

    df = load_nt_shotlist()                     # 305 stationary phases
    fit_df = df[df.f_gw < 1]                    # 260 phases below Greenwald
    fit = fit_power_law(fit_df, 'ln_tauth_ptot', ENGINEERING_REGRESSORS,
                        fixed_exponents=FIXED_GEOMETRY_EXPONENTS,
                        isotope_exponent=ISOTOPE_EXPONENT_THERMAL,
                        weighting='kde')
    boot = subset_bootstrap(fit_df, 'ln_tauth_ptot', ENGINEERING_REGRESSORS,
                            fixed_exponents=FIXED_GEOMETRY_EXPONENTS,
                            frac=0.75, rng=np.random.default_rng(2023))
    plotting.plot_scaling(fit)

To use another dataset, build a DataFrame with the columns documented in
scripts.data (or any ln_* columns of your own) and call fit_power_law on it.
"""
from . import plotting
from .data import (DEFAULT_CSV, REPO_ROOT, add_log_params,
                   apply_quality_filter, load_nt_shotlist)
from .fitting import (FitResult, build_log_linear_model, fit_power_law,
                      kde_weights, prepare_xy)
from .reference import (ENGINEERING_REGRESSORS, EXPONENT_LABELS,
                        FIXED_GEOMETRY_EXPONENTS, FREE_REGRESSORS,
                        GROUP_STYLE, IPB98Y2, ISOTOPE_EXPONENT_THERMAL,
                        ITER89P, MASS_DEUTERIUM, ScalingLaw, group_style)
from .uncertainty import BootstrapResult, subset_bootstrap

__all__ = [
    'plotting',
    'DEFAULT_CSV', 'REPO_ROOT', 'add_log_params', 'apply_quality_filter',
    'load_nt_shotlist',
    'FitResult', 'build_log_linear_model', 'fit_power_law', 'kde_weights',
    'prepare_xy',
    'ENGINEERING_REGRESSORS', 'EXPONENT_LABELS', 'FIXED_GEOMETRY_EXPONENTS',
    'FREE_REGRESSORS', 'GROUP_STYLE', 'IPB98Y2', 'ISOTOPE_EXPONENT_THERMAL',
    'ITER89P', 'MASS_DEUTERIUM', 'ScalingLaw', 'group_style',
    'BootstrapResult', 'subset_bootstrap',
]
