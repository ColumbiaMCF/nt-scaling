"""Reference scaling laws and regression conventions for the NT analysis.

Single source of truth for the hard-coded physics inputs: the IPB98(y,2)
and ITER89-P laws, the geometry exponents frozen during the engineering
regressions, the isotope-mass convention, the standard regressor sets, and
the per-group plot styling. Everything downstream (data, fitting,
bootstrapping, plotting) imports these names rather than re-declaring
numbers.

Column names follow the conventions of the source shot list
(``bcentr`` for B_t, ``rsurf`` for R_0, ``kappa_a`` for the area-based
elongation, ``ptot_cps`` for the summed heating power) — see
:mod:`scripts.data` for the full column/unit table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# --- regression conventions --------------------------------------------------

#: Geometry exponents frozen at their IPB98(y,2) values during the
#: engineering regressions — a single-device database has too little size
#: variation to fit R_0/kappa/eps. Keys match the ln_ regressor columns.
FIXED_GEOMETRY_EXPONENTS: Dict[str, float] = {
    'ln_rsurf': 1.97, 'ln_kappa_a': 0.78, 'ln_eps': 0.58,
}

#: Isotope-mass exponent of the thermal (IPB98(y,2)-like) scaling and the
#: effective mass of the deuterium database plasmas.
ISOTOPE_EXPONENT_THERMAL = 0.19
MASS_DEUTERIUM = 2.0

#: Regressor set of the engineering fit: four free exponents followed by the
#: three frozen-geometry columns. ``ln_ptot_cps`` is the summed heating power
#: P_inj + V_surf*I_p + P_ECH (see scripts.data for the ptot conventions).
ENGINEERING_REGRESSORS: List[str] = ['ln_ip', 'ln_bcentr', 'ln_density',
                                     'ln_ptot_cps', 'ln_rsurf', 'ln_kappa_a',
                                     'ln_eps']
FREE_REGRESSORS: List[str] = ENGINEERING_REGRESSORS[:4]
EXPONENT_LABELS: List[str] = [r'$\alpha_I$', r'$\alpha_B$', r'$\alpha_n$',
                              r'$\alpha_P$']


# --- reference scaling laws --------------------------------------------------

@dataclass(frozen=True)
class ScalingLaw:
    """A published power-law confinement scaling, evaluated by column name.

        tau = coeff * mass^isotope_exponent * prod_i(x_i ^ exponents[x_i])

    ``exponents`` keys are plain column names in the standard database units
    (MA, T, 1e19 m^-3, MW, m). ``density_scale`` rescales the density column
    before exponentiation: ITER89-P is defined in n20 units while the
    database stores n19, so it carries ``density_scale=0.1``.

    Example
    -------
    >>> from scripts import IPB98Y2, load_nt_shotlist
    >>> df = load_nt_shotlist()
    >>> IPB98Y2.tau(df)                       # per-row prediction [s]
    >>> IPB98Y2.tau({'ip': 1.0, 'bcentr': 2.0, 'density': 4.0, 'ptot_cps': 5.0,
    ...              'rsurf': 1.7, 'kappa_a': 1.3, 'eps': 0.33})

    Define your own law by instantiation:

    >>> MY_LAW = ScalingLaw(name='my-law', coeff=0.05, isotope_exponent=0.2,
    ...                     exponents={'ip': 1.0, 'ptot_cps': -0.5})
    """
    name: str
    coeff: float
    isotope_exponent: float
    exponents: Dict[str, float]
    density_scale: float = 1.0

    def tau(self, params, mass: float = MASS_DEUTERIUM):
        """Confinement time [s] for a DataFrame or any name->value mapping."""
        out = self.coeff * mass**self.isotope_exponent
        for var, exponent in self.exponents.items():
            value = np.asarray(params[var], dtype=float)
            if var == 'density':
                value = value * self.density_scale
            out = out * value**exponent
        return out


#: ITER Physics Basis Editors 1999 Nucl. Fusion 39 2175 (thermal tau_E).
IPB98Y2 = ScalingLaw(
    name='IPB98(y,2)', coeff=0.0562, isotope_exponent=0.19,
    exponents={'ip': 0.93, 'bcentr': 0.15, 'density': 0.41, 'ptot_cps': -0.69,
               'rsurf': 1.97, 'kappa_a': 0.78, 'eps': 0.58},
)

#: Yushmanov et al. 1990 Nucl. Fusion 30 1999 (total tau_E, n20 density).
ITER89P = ScalingLaw(
    name='ITER89-P', coeff=0.048, isotope_exponent=0.5,
    exponents={'ip': 0.85, 'rsurf': 1.2, 'aminor': 0.3, 'kappa_a': 0.5,
               'density': 0.1, 'bcentr': 0.2, 'ptot_cps': -0.5},
    density_scale=0.1,
)


# --- plot styling -------------------------------------------------------------

#: (color, marker) per group label in the ``tok`` column. The loader labels
#: every row 'NT'; the notebook relabels the fit sample and the excluded
#: f_GW > 1 phases so they are styled separately. Extend when adding groups.
GROUP_STYLE: Dict[str, tuple] = {
    'NT': ('black', 'o'),
    r'$f_{GW} < 1$': ('black', 'o'),
    r'$f_{GW} > 1$': ('red', 'o'),
}


def group_style(label: str) -> tuple:
    """(color, marker) for a group label, with a neutral fallback."""
    return GROUP_STYLE.get(label, ('gray', 'x'))
