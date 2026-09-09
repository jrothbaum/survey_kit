"""
SRMI chain convergence diagnostics - matches mice's convergence()
function (amices/mice R/convergence.R) as closely as possible, down to
the exact algorithm mice itself delegates to for the potential scale
reduction factor: rstan::Rhat(), i.e. the rank-normalized, folded,
split-Rhat of Vehtari, Gelman, Simpson, Carpenter & Burkner (2021),
"Rank-Normalization, Folding, and Localization: An Improved R-hat for
Assessing Convergence of MCMC" (Bayesian Analysis). Verified against the
actual R source (stan-dev/posterior's R/convergence.R and
R/split_chains.R, and amices/mice's R/convergence.R) rather than
documentation summaries.

Every function here operates on a plain (n_iterations, n_chains) numpy
array - "chains" are SRMI implicates, "draws" are chainMean/chainStd (the
mean/std of a variable's own newly-imputed values each iteration, not
the whole column) rather than posterior samples, but the diagnostic math
is identical.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import rankdata, norm


def split_chains(x: np.ndarray) -> np.ndarray:
    """
    Split each chain (column) into two halves, doubling the number of
    chains - drops the middle row if n_iterations is odd. Matches
    posterior's .split_chains() exactly (including which half a middle
    row is excluded from).
    """
    niter = x.shape[0]
    if niter == 1:
        return x.copy()
    half = niter / 2.0
    first = x[: int(np.floor(half))]
    second = x[int(np.ceil(half + 1)) - 1 :]
    return np.concatenate([first, second], axis=1)


def z_scale(x: np.ndarray) -> np.ndarray:
    """
    Rank-normalize: replace every value (across the WHOLE array, not
    per-chain - matches R's rank() on a matrix, which flattens) by its
    rank (average rank for ties), then apply Blom's (1958) fractional-
    offset (c=3/8) inverse-normal-CDF transform. NaN values keep their
    position (as NaN) and don't participate in the ranking.
    """
    flat = x.ravel()
    valid = ~np.isnan(flat)
    out = np.full_like(flat, np.nan, dtype=float)
    if valid.sum() > 0:
        r = rankdata(flat[valid], method="average")
        c = 3.0 / 8.0
        s = valid.sum()
        u = (r - c) / (s - 2 * c + 1)
        out[valid] = norm.ppf(u)
    return out.reshape(x.shape)


def fold_draws(x: np.ndarray) -> np.ndarray:
    """Fold around the grand median (all chains combined) - |x - median(x)|."""
    return np.abs(x - np.nanmedian(x))


def _rhat_basic(x: np.ndarray) -> float:
    """
    Classic Gelman-Rubin Rhat on already-transformed (split, rank-
    normalized) draws. NaN if there's not enough data (fewer than 2
    chains or 2 iterations) or within-chain variance is exactly 0.
    """
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return float("nan")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return float("nan")
    niterations, nchains = x.shape
    chain_mean = x.mean(axis=0)
    chain_var = x.var(axis=0, ddof=1)
    var_between = niterations * chain_mean.var(ddof=1)
    var_within = chain_var.mean()
    if var_within == 0:
        return float("nan")
    return float(np.sqrt((var_between / var_within + niterations - 1) / niterations))


def rhat(x: np.ndarray) -> float:
    """
    Rhat = max(bulk-Rhat, tail-Rhat) - the improved R-hat of Vehtari et
    al. (2021), matching posterior::rhat()/rstan::Rhat() exactly. x is
    (n_iterations, n_chains) of raw chain-mean or chain-sd values for
    ONE variable. NaN propagates (matches R's max(NA, x) == NA - no
    na.rm) if either half is undefined.
    """
    rhat_bulk = _rhat_basic(z_scale(split_chains(x)))
    rhat_tail = _rhat_basic(z_scale(split_chains(fold_draws(x))))
    if np.isnan(rhat_bulk) or np.isnan(rhat_tail):
        return float("nan")
    return max(rhat_bulk, rhat_tail)


def autocorrelation_mice(param: np.ndarray) -> np.ndarray:
    """
    mice's own "ac" diagnostic - NOT the FFT-based within-chain
    autocorrelation posterior::ess_bulk() etc. use. At each iteration t
    (1-indexed), the CUMULATIVE MEAN of lag-1 Pearson correlations
    computed ACROSS CHAINS between consecutive iterations' chain-mean
    (or chain-sd) vectors - i.e. cor(param[t-2, :], param[t-1, :]) for
    each t from 2 to T, averaged (running) up through t. Undefined
    correlations (fewer than 2 pairwise-complete chains, or a constant
    vector) are treated as 0 before averaging, matching mice's
    coalesce(cor(...), 0). Index 0 (iteration 1) is always NaN - no
    lag-1 correlation is defined yet.

    param : (n_iterations, n_chains)
    Returns an (n_iterations,) array.
    """
    t = param.shape[0]
    ac = np.full(t, np.nan)
    if t < 2:
        return ac

    raw = np.zeros(t - 1)
    for itr in range(1, t):
        a = param[itr - 1, :]
        b = param[itr, :]
        mask = ~np.isnan(a) & ~np.isnan(b)
        corr = np.nan
        if mask.sum() >= 2:
            a_m, b_m = a[mask], b[mask]
            if a_m.std() > 0 and b_m.std() > 0:
                corr = float(np.corrcoef(a_m, b_m)[0, 1])
        raw[itr - 1] = 0.0 if np.isnan(corr) else corr

    ac[1:] = np.cumsum(raw) / np.arange(1, t)
    return ac


def convergence_table(
    chain_mean: dict[str, np.ndarray],
    chain_std: dict[str, np.ndarray],
    diagnostic: str = "all",
    parameter: str = "mean",
) -> pl.DataFrame:
    """
    Assemble the final long-format diagnostics table, one row per
    (iteration, variable) - same shape/column names as mice's own
    convergence(): ".it", "vrb", and "ac" and/or "psrf".

    Parameters
    ----------
    chain_mean, chain_std : dict[str, np.ndarray]
        {variable: (n_iterations, n_chains) array}, already assembled
        (e.g. from each SRMI implicate's own Impute.chain_mean/
        chain_std accumulation) - values may be NaN/None-derived NaN
        for a variable/iteration/chain that has nothing to report
        (e.g. a non-numeric target, or nothing to impute that round).
    diagnostic : str
        "all" (both), "ac", "psrf", or "gr" (alias for "psrf").
    parameter : str
        "mean" or "sd" - which of chain_mean/chain_std to diagnose.

    Returns
    -------
    pl.DataFrame
        Columns: ".it", "vrb", plus "ac" and/or "psrf" per `diagnostic`.
    """
    if diagnostic not in ("all", "ac", "psrf", "gr"):
        raise ValueError(
            f"diagnostic={diagnostic!r} not recognized - use 'all', 'ac', "
            f"'psrf', or 'gr'"
        )
    if parameter not in ("mean", "sd"):
        raise ValueError(f"parameter={parameter!r} not recognized - use 'mean' or 'sd'")

    source = chain_mean if parameter == "mean" else chain_std
    want_ac = diagnostic in ("all", "ac")
    want_psrf = diagnostic in ("all", "psrf", "gr")

    rows = []
    for vrb, param in source.items():
        t = param.shape[0]
        ac_vals = autocorrelation_mice(param) if want_ac else None
        for itr in range(1, t + 1):
            rowi = {".it": itr, "vrb": vrb}
            if ac_vals is not None:
                rowi["ac"] = ac_vals[itr - 1]
            if want_psrf:
                rowi["psrf"] = rhat(param[:itr, :])
            rows.append(rowi)

    if not rows:
        cols = {".it": [], "vrb": []}
        if want_ac:
            cols["ac"] = []
        if want_psrf:
            cols["psrf"] = []
        return pl.DataFrame(cols)

    return pl.DataFrame(rows)


def convergence_long_table(
    chain_mean: dict[str, np.ndarray],
    chain_std: dict[str, np.ndarray],
) -> pl.DataFrame:
    """
    Long/tidy-format chain trace data for plotting - the same
    chainMean/chainVar mice's own plot.mids() reads, reshaped fully
    long (one row per iteration x implicate x variable x parameter)
    rather than mice's wide-columns-plus-lattice-facets shape, which is
    what a tidy plotting library (e.g. plotly express) wants directly.

    Parameters
    ----------
    chain_mean, chain_std : dict[str, np.ndarray]
        {variable: (n_iterations, n_chains) array} - same shape
        convergence_table() takes.

    Returns
    -------
    pl.DataFrame
        Columns: ".it" (1-indexed iteration), "implicate" (1-indexed),
        "vrb" (variable name), "parameter" ("mean" or "sd"), "value".
    """
    rows = []
    for parameter, source in (("mean", chain_mean), ("sd", chain_std)):
        for vrb, param in source.items():
            t, m = param.shape
            for itr in range(1, t + 1):
                for chain in range(1, m + 1):
                    rows.append(
                        {
                            ".it": itr,
                            "implicate": chain,
                            "vrb": vrb,
                            "parameter": parameter,
                            "value": param[itr - 1, chain - 1],
                        }
                    )

    if not rows:
        return pl.DataFrame(
            {".it": [], "implicate": [], "vrb": [], "parameter": [], "value": []}
        )

    return pl.DataFrame(rows)
