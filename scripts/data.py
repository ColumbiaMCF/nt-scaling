"""Loading and preparation of the negative-triangularity shot list.

The shot list (``data/nt_scaling_shotlist.csv``) has one row per stationary
phase of an NT discharge from the campaign of Paz-Soldan et al., Nucl.
Fusion 64 094002 (2024), with the source database's column names in raw
(mostly SI) units. :func:`load_nt_shotlist` converts it into the
regression-ready frame used by the rest of the package.

Columns and units AFTER loading (raw CSV units in parentheses):

    shotnum, tstart, tend   shot number and phase window     [ms]
    tok                     group label, 'NT' for every row
    ip                      plasma current                   [MA]  (A)
    bcentr                  |toroidal field|                 [T]   (signed T)
    density                 line-average density             [1e19 m^-3] (cm^-3)
    ptot                    total heating power, database    [MW]  (W)
    pinj, pnbi, echpwrc     NBI (two sources) and ECH power  [MW]  (kW, W, W)
    wmhd, wfast             stored energy, fast-ion energy   [MJ]  (J)
    wdot                    dW/dt                            [MW]  (W)
    vsurf                   surface loop voltage             [V]
    rsurf, aminor           major / minor radius             [m]
    kappa, area             elongation, cross-section area   [-, m^2]
    tinj                    injected torque                  [N m]
    n2rms                   n=2 RMS magnetic fluctuation     [G]
    tste_core, tsne_core    core T_e, n_e (Thomson)          [keV, 1e19 m^-3]
    cerqrott6, cerarott6    |core rotation| (CER)            [km/s]

Derived columns:

    eps        = aminor / rsurf
    kappa_a    = area / (pi * aminor^2)          area-based elongation
    pohm_cps   = vsurf * ip                      ohmic power [MW]
    ptot_cps   = pinj + pohm_cps + echpwrc       summed heating power [MW]
    tauth_ptot = (wmhd - wfast) / (ptot - wdot)  thermal tau_E from database ptot [s]
    taue_ptot  = wmhd / (ptot - wdot)
    tauth_cps, taue_cps                          same, from ptot_cps
    tauth_98, taue_89                            IPB98(y,2) / ITER89-P predictions
    f_gw       = n20 / (ip / (pi a^2))           Greenwald fraction
    h98, h89                                     tauth_cps/tauth_98, taue_cps/taue_89
    ln_<name>                                    natural logs for the regressions

Conventions:

* ``pinj`` is stored in kW in the CSV (all other powers in W); rows with
  ``pinj == 0`` fall back to ``pnbi``.
* The regression *target* ``tauth_ptot`` uses the database ``ptot``, while
  the power *regressor* is ``ln_ptot_cps``. ptot_cps/ptot has median 1.06
  (range 0.99-1.25) on this database.
* The reference laws are evaluated with ``ptot_cps``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .reference import IPB98Y2, ITER89P

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The NT shot list (Paz-Soldan et al. 2024): one row per stationary phase.
DEFAULT_CSV = REPO_ROOT / 'data' / 'nt_scaling_shotlist.csv'

#: Raw-column -> multiplicative factor to reach the standard units above.
#: Applied only to columns present in the file.
_UNIT_SCALES = {
    'ip': 1e-6,          # A   -> MA
    'density': 1e-13,    # cm^-3 -> 1e19 m^-3
    'ptot': 1e-6,        # W   -> MW
    'pnbi': 1e-6,        # W   -> MW
    'echpwrc': 1e-6,     # W   -> MW
    'pinj': 1e-3,        # kW  -> MW   (source-database quirk: pinj is in kW)
    'wmhd': 1e-6,        # J   -> MJ
    'wfast': 1e-6,       # J   -> MJ
    'wdot': 1e-6,        # W   -> MW
    'tsne_core': 1e-19,  # m^-3 -> 1e19 m^-3
    'cerqnzt6': 1e-19,   # m^-3 -> 1e19 m^-3
    'tste_core': 1e-3,   # eV  -> keV
    'cerqtit6': 1e-3,    # eV  -> keV
    'ceratit6': 1e-3,    # eV  -> keV
}

#: Signed quantities used as magnitudes (field direction, rotation sign).
_ABS_COLUMNS = ('bt', 'bcentr', 'cerqrott6', 'cerarott6')

#: ln_<name> columns added by :func:`add_log_params` (plain column -> ln name).
_LOG_COLUMN_MAP = {
    'ln_ip': 'ip', 'ln_bcentr': 'bcentr', 'ln_density': 'density',
    'ln_ptot': 'ptot', 'ln_ptot_cps': 'ptot_cps',
    'ln_rsurf': 'rsurf', 'ln_aminor': 'aminor', 'ln_eps': 'eps',
    'ln_kappa_a': 'kappa_a',
    'ln_cerqrott6': 'cerqrott6', 'ln_cerarott6': 'cerarott6',
    'ln_tauth_ptot': 'tauth_ptot', 'ln_tauth_cps': 'tauth_cps',
    'ln_taue_ptot': 'taue_ptot', 'ln_taue_cps': 'taue_cps',
}


def load_nt_shotlist(path=DEFAULT_CSV, *, quality_filter: bool = True,
                     add_logs: bool = True) -> pd.DataFrame:
    """Load the NT shot list, ready for regression.

    Steps: read the CSV, rescale to the standard units (module docstring),
    derive the geometry / power / confinement-time columns, optionally apply
    the stationarity quality filter (:func:`apply_quality_filter`), and add
    the ``ln_*`` regression columns (:func:`add_log_params`).

    Example
    -------
    >>> from scripts import load_nt_shotlist, fit_power_law
    >>> from scripts import ENGINEERING_REGRESSORS, FIXED_GEOMETRY_EXPONENTS
    >>> df = load_nt_shotlist()                 # 305 phases, 154 shots
    >>> fit_df = df[df.f_gw < 1]
    >>> fit = fit_power_law(fit_df, 'ln_tauth_ptot', ENGINEERING_REGRESSORS,
    ...                     fixed_exponents=FIXED_GEOMETRY_EXPONENTS,
    ...                     isotope_exponent=0.19, weighting='kde')
    """
    df = pd.read_csv(path)
    df['tok'] = 'NT'

    for col, scale in _UNIT_SCALES.items():
        if col in df.columns:
            df[col] = df[col] * scale
    for col in _ABS_COLUMNS:
        if col in df.columns:
            df[col] = np.abs(df[col])

    # geometry
    df['eps'] = df.aminor / df.rsurf
    df['kappa_a'] = df.area / (np.pi * df.aminor**2)

    # heating power: summed sources vs the database total
    df['pinj'] = np.where(df['pinj'] == 0.0, df['pnbi'], df['pinj'])
    df['pohm_cps'] = df.vsurf * df.ip
    df['ptot_cps'] = df.pinj + df.pohm_cps + df.echpwrc

    # confinement times from either power convention
    df['tauth_ptot'] = (df.wmhd - df.wfast) / (df.ptot - df.wdot)
    df['taue_ptot'] = df.wmhd / (df.ptot - df.wdot)
    df['tauth_cps'] = (df.wmhd - df.wfast) / (df.ptot_cps - df.wdot)
    df['taue_cps'] = df.wmhd / (df.ptot_cps - df.wdot)

    # reference laws, Greenwald fraction, H-factors
    df['tauth_98'] = IPB98Y2.tau(df)
    df['taue_89'] = ITER89P.tau(df)
    df['f_gw'] = (df.density / 10) / (df.ip / (np.pi * df.aminor**2))
    df['h98'] = df.tauth_cps / df.tauth_98
    df['h89'] = df.taue_cps / df.taue_89

    if quality_filter:
        df = apply_quality_filter(df)
    if add_logs:
        df = add_log_params(df)
    return df.reset_index(drop=True)


def apply_quality_filter(df: pd.DataFrame, *, wdot_max: float = 0.15,
                         tinj_min: float = 0.1,
                         n2rms_max: float = 5.0) -> pd.DataFrame:
    """Keep stationary, NBI-heated, MHD-quiet phases.

    * ``|wdot| <= wdot_max`` [MW] — stationary stored energy;
    * ``tinj >= tinj_min`` [N m] — injected torque present (NBI on);
    * ``n2rms <= n2rms_max`` [G] — no large n=2 MHD activity.

    Example
    -------
    >>> df = load_nt_shotlist(quality_filter=False)     # 329 phases
    >>> df = apply_quality_filter(df, wdot_max=0.10)     # stricter stationarity
    """
    keep = ((df['wdot'].abs() <= wdot_max)
            & (df['tinj'] >= tinj_min)
            & (df['n2rms'] <= n2rms_max))
    return df[keep].copy()


def add_log_params(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``ln_<name>`` columns used by the log-linear regressions.

    Columns absent from ``df`` are skipped. Non-positive values give NaN
    (with a suppressed warning) and are dropped by the fitting NaN-drop.
    """
    df = df.copy()
    with np.errstate(invalid='ignore', divide='ignore'):
        for ln_name, name in _LOG_COLUMN_MAP.items():
            if name in df.columns:
                df[ln_name] = np.log(df[name])
    return df
