# Energy Confinement Time Scaling — Negative Triangularity

Regression analysis of the thermal energy confinement time of the stationary
negative-triangularity (NT) discharges reported in Paz-Soldan *et al.*,
Nucl. Fusion **64** 094002 (2024)
([doi:10.1088/1741-4326/ad69a4](https://doi.org/10.1088/1741-4326/ad69a4)).
A log-linear power law in the
engineering parameters ($I_p$, $B_t$, $\langle n_e\rangle$, $P_{tot}$) is fit
with the geometry exponents frozen at their IPB98(y,2) values, using a
kernel-density-estimate (KDE) weighting against non-uniform sampling of
parameter space, and the exponent uncertainty is characterized by subset
bootstrapping.

## Quickstart

Requires Python ≥ 3.8 with numpy, scipy, pandas, scikit-learn, matplotlib
(and jupyter to run the notebook).

```python
import sys; sys.path.insert(0, '/path/to/nt_scaling')

import numpy as np
from scripts import (load_nt_shotlist, fit_power_law, subset_bootstrap,
                     ENGINEERING_REGRESSORS, FIXED_GEOMETRY_EXPONENTS,
                     ISOTOPE_EXPONENT_THERMAL, plotting)

df = load_nt_shotlist()                       # 305 stationary phases, 154 shots
fit_df = df[df.f_gw < 1]                      # 260 phases below the Greenwald limit

fit = fit_power_law(fit_df, 'ln_tauth_ptot', ENGINEERING_REGRESSORS,
                    fixed_exponents=FIXED_GEOMETRY_EXPONENTS,
                    isotope_exponent=ISOTOPE_EXPONENT_THERMAL, weighting='kde')
plotting.plot_scaling(fit)

boot = subset_bootstrap(fit_df, 'ln_tauth_ptot', ENGINEERING_REGRESSORS,
                        fixed_exponents=FIXED_GEOMETRY_EXPONENTS,
                        frac=0.75, num_repeats=1000,
                        rng=np.random.default_rng(2023))
boot.exponent_mean_std()
```

The full analysis is `runs/01_nt_confinement_scaling.ipynb` (executed in
place; figures in `runs/figures/`). Headless execution (~2 minutes):

```bash
cd runs && python3 -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 01_nt_confinement_scaling.ipynb
```

## Package layout (`scripts/`)

| Module | Contents |
|---|---|
| `reference.py` | IPB98(y,2) and ITER89-P laws (`ScalingLaw`), frozen geometry / isotope conventions, regressor sets, plot styling |
| `data.py` | `load_nt_shotlist()`: CSV → standard units → derived columns → `ln_*` regressors; `apply_quality_filter()` |
| `fitting.py` | `fit_power_law()` → `FitResult`: log-linear power-law regression with frozen exponents and OLS or KDE weighting |
| `uncertainty.py` | `subset_bootstrap()` → `BootstrapResult` |
| `plotting.py` | Measured-vs-predicted, parameter distributions, correlation / scatter matrices, bootstrap histograms (pure functions; optional `save_path`) |

Every public function carries a docstring with example usage.

## Data

`data/nt_scaling_shotlist.csv`: one row per stationary phase (329 rows, 161
shots) from the NT campaign of Paz-Soldan *et al.* (2024); column names
follow the source shot list and values are in raw units. After loading:

| Column | Quantity | Unit |
|---|---|---|
| `shotnum`, `tstart`, `tend` | shot, phase window | –, ms, ms |
| `ip` | plasma current | MA |
| `bcentr` | \|toroidal field\| | T |
| `density` | line-average density | 10¹⁹ m⁻³ |
| `ptot` | total heating power (database) | MW |
| `pinj`, `pnbi`, `echpwrc` | NBI, NBI (alt.), ECH power | MW |
| `wmhd`, `wfast`, `wdot` | stored energy, fast-ion energy, dW/dt | MJ, MJ, MW |
| `vsurf` | surface loop voltage | V |
| `rsurf`, `aminor`, `kappa`, `area` | R₀, a, κ, cross-section area | m, m, –, m² |
| `tinj` | injected torque | N m |
| `n2rms` | n = 2 RMS magnetic fluctuation | G |

Derived: `eps`, `kappa_a = area/(πa²)`, `pohm_cps = vsurf·ip`,
`ptot_cps = pinj + pohm_cps + echpwrc`, `tauth_ptot`/`taue_ptot` (from the
database `ptot`), `tauth_cps`/`taue_cps` (from `ptot_cps`), `tauth_98`,
`taue_89`, `f_gw`, `h98`, `h89`, and `ln_<name>` for each regression column.

Quality filter (default on): |`wdot`| ≤ 0.15 MW, `tinj` ≥ 0.1 N m,
`n2rms` ≤ 5 G → 305 phases / 154 shots. The fit uses the `f_gw` < 1 subset
(260 phases / 132 shots); the 23 phases above the Greenwald limit are only
overlaid.

**Conventions:** the regression target
`ln_tauth_ptot` uses the database `ptot`, while the power regressor is
`ln_ptot_cps` (median ratio ptot_cps/ptot = 1.06, range 0.99–1.25); the
reference laws are evaluated with `ptot_cps`; `pinj` is stored in kW in the
CSV and rows with `pinj == 0` fall back to `pnbi`.

## Weighting

`fit_power_law(..., weighting='kde')` gives each point
σ = √(pdf/pdf_max) from a Gaussian KDE (Scott bandwidth), i.e. least-squares
weight ∝ 1/pdf, so densely sampled regions of parameter space count less.
The KDE is built on the four free regressors only (the frozen-geometry
columns carry no sampling information), the density is floored at its 5th
percentile (bounding the leverage of isolated points), and the effective
sample size (Σw)²/Σw² is reported with every weighted fit. Both choices are
exposed as `kde_vars` and `kde_cap_percentile`.

## Results (headline fit)

| exponent | KDE fit | OLS fit | bootstrap 50% (±2σ) | IPB98(y,2) |
|---|---|---|---|---|
| α_I | 1.06 ± 0.06 | 1.00 ± 0.05 | 1.05 ± 0.13 | 0.93 |
| α_B | 0.04 ± 0.06 | 0.22 ± 0.08 | 0.06 ± 0.15 | 0.15 |
| α_n | 0.54 ± 0.04 | 0.43 ± 0.05 | 0.53 ± 0.13 | 0.41 |
| α_P | −0.92 ± 0.02 | −0.88 ± 0.02 | −0.92 ± 0.05 | −0.69 |
| R² (log) | 0.874 | 0.887 | — | — |
| effective N | 156 of 260 | 260 | — | — |

## Reference

The discharges analyzed here are those of

> C. Paz-Soldan *et al.*, "Simultaneous access to high normalized density,
> current, pressure, and confinement in strongly-shaped diverted negative
> triangularity plasmas," *Nuclear Fusion* **64**, 094002 (2024).
> https://doi.org/10.1088/1741-4326/ad69a4

Please cite that paper when using the shot list or the fits.

## Repository map

```
├── README.md
├── data/                nt_scaling_shotlist.csv
├── scripts/             the analysis package (table above)
└── runs/                executed notebook + figures/
```
