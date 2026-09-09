"""
SRMI imputation-quality diagnostics - a DIFFERENT question from
convergence_diagnostics.py's "did the chain stabilize": here it's "do
the imputed values look plausible next to the observed ones." Matches
mice's own densityplot()/stripplot()/bwplot() (R/densityplot.mids.R,
R/stripplot.mids.R, R/bwplot.mids.R) convention as closely as possible,
verified against the actual R source: one group for the truly observed
values (pooled once, identical across implicates - it never changes),
plus one group per implicate holding ONLY that implicate's own newly
imputed values (rows that were originally missing for that variable) -
never the whole column. mice's densityplot literally nulls out the
observed rows in each imputed copy before taking the density, for
exactly this reason (R/densityplot.mids.R's `cd[select, xvar] <- NA`
step).

Not built here (deliberately deferred - a materially bigger lift,
needing a fitted propensity/detrending model rather than just a
reshape): mice's propensity-score xyplot() and its own "worm plot"
(detrended Q-Q by covariate).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def observed_vs_imputed_long_table(
    data: dict[str, dict[str, list]],
    sample_k: int | None = None,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Reshape {variable: {"observed": [...], "implicates": [[...], ...]}}
    into a long (group, vrb, value) table - group is "Observed" (once,
    pooled - identical across implicates) or "Implicate {i}" (that
    implicate's own newly-imputed values only).

    Parameters
    ----------
    data : dict[str, dict[str, list]]
        {variable: {"observed": list of values, "implicates": list of
        lists, one per implicate, of that implicate's own imputed
        values for this variable}}.
    sample_k : int | None, optional
        If given, randomly sample at most this many points per (group,
        variable) - avoids overplotting a stripplot on a large dataset,
        same rationale as mice's own "stripplot is best for small
        datasets, use bwplot for large ones" guidance. Not meaningful
        for density (a KDE should use every point) or a box summary (a
        box plot's five-number summary should too) - only apply this
        for a "strip" (every-point) plot. By default None (no sampling).
    seed : int | None, optional
        Seed for the random sample. By default None.

    Returns
    -------
    pl.DataFrame
        Columns: "group", "vrb", "value".
    """
    rng = np.random.default_rng(seed)

    def _maybe_sample(values: list) -> list:
        if sample_k is None or len(values) <= sample_k:
            return values
        idx = rng.choice(len(values), size=sample_k, replace=False)
        return [values[i] for i in idx]

    rows = {"group": [], "vrb": [], "value": []}
    for vrb, groups in data.items():
        observed = _maybe_sample(list(groups.get("observed", [])))
        rows["group"].extend(["Observed"] * len(observed))
        rows["vrb"].extend([vrb] * len(observed))
        rows["value"].extend(observed)

        for i, imputed_i in enumerate(groups.get("implicates", []), start=1):
            imputed_i = _maybe_sample(list(imputed_i))
            label = f"Implicate {i}"
            rows["group"].extend([label] * len(imputed_i))
            rows["vrb"].extend([vrb] * len(imputed_i))
            rows["value"].extend(imputed_i)

    return pl.DataFrame(rows)


def density_long_table(
    data: dict[str, dict[str, list]],
    n_grid: int = 200,
) -> pl.DataFrame:
    """
    Kernel density estimate per (group, variable) - mice's own
    densityplot() convention (see module docstring): one curve for
    "Observed", one per implicate, each evaluated only over ITS OWN
    values (never the whole column). Numeric-only (a KDE needs numeric
    values) - skips a variable/group with fewer than 2 distinct finite
    values (a KDE needs at least 2 points to pick a bandwidth - mice
    hits the identical wall and has no workaround either, per
    densityplot.mids.R's own docs: "use the more robust bwplot or
    stripplot as a replacement").

    Returns
    -------
    pl.DataFrame
        Columns: "group", "vrb", "x" (grid point), "density".
    """
    from scipy.stats import gaussian_kde

    rows = {"group": [], "vrb": [], "x": [], "density": []}
    for vrb, groups in data.items():
        all_groups = [("Observed", groups.get("observed", []))] + [
            (f"Implicate {i}", imputed_i)
            for i, imputed_i in enumerate(groups.get("implicates", []), start=1)
        ]
        for label, values in all_groups:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 2 or np.std(arr) == 0:
                continue
            kde = gaussian_kde(arr)
            lo, hi = arr.min(), arr.max()
            pad = (hi - lo) * 0.1 if hi > lo else 1.0
            grid = np.linspace(lo - pad, hi + pad, n_grid)
            dens = kde(grid)
            rows["group"].extend([label] * n_grid)
            rows["vrb"].extend([vrb] * n_grid)
            rows["x"].extend(grid.tolist())
            rows["density"].extend(dens.tolist())

    return pl.DataFrame(rows)
