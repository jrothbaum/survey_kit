"""
Shared synthetic multiple-imputation data for the adapter tutorials
(basic_r.py, basic_stata.py, basic_python_adapters.py, ...) - kept out of
those files so they can focus on the adapter calls themselves rather than
on how the sample data got built.

Run this file directly to see exactly what it generates:
    python sample_data.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from survey_kit import logger
from survey_kit.statistics.bootstrap import bayes_bootstrap_weights


def make_implicates(n_implicates: int = 5, n: int = 300, seed: int = 0) -> list[pl.DataFrame]:
    """
    n_implicates independent draws of a simple linear DGP:
    y = 1 + 2*x1 - 1.5*x2 + noise, x1/x2 ~ N(0, 1) - standing in for what
    would normally be n_implicates completed datasets from an imputation
    model (e.g. SRMI.df_implicates), just enough variation between
    implicates to make Rubin's-rules combination meaningful.

    Parameters
    ----------
    n_implicates : number of implicates to generate. Default is 5.
    n : rows per implicate. Default is 300.
    seed : seed for the first implicate - later implicates use seed+1,
        seed+2, ... so they're independent draws, not copies. Default is 0.
    """

    def _one(s: int) -> pl.DataFrame:
        rng = np.random.default_rng(s)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 1 + 2 * x1 - 1.5 * x2 + rng.normal(size=n) * 0.4
        return pl.DataFrame({"x1": x1, "x2": x2, "y": y})

    return [_one(seed + i) for i in range(n_implicates)]


def with_bootstrap_weights(
    df_implicates: list[pl.DataFrame], n_replicates: int = 20, seed: int = 1000
) -> list[pl.DataFrame]:
    """
    Adds replicate_0..replicate_<n_replicates> bootstrap weight columns to
    each implicate, via survey_kit's own
    [`bayes_bootstrap_weights`][survey_kit.statistics.bootstrap.bayes_bootstrap_weights]
    - replicate_0 is the full-sample weight (all 1s), replicate_1.. are a
    genuine Bayesian bootstrap draw (strictly positive by construction, so
    they work with every adapter, including ones like linearmodels that
    reject a zero weight outright). Pair with
    `Replicates(weight_stub="replicate_", n_replicates=n_replicates,
    bootstrap=True)` and a `command`/`formula` with a "{weight}"
    placeholder, e.g. "y ~ x1 + x2 [pw={weight}]".

    Parameters
    ----------
    df_implicates : implicates to add weight columns to (e.g. from
        `make_implicates()`).
    n_replicates : number of replicate weight columns. Default is 20.
    seed : seed for the first implicate's weights - later implicates use
        seed+1, seed+2, ... Default is 1000.
    """

    def _one(df: pl.DataFrame, s: int) -> pl.DataFrame:
        df = df.with_columns(pl.lit(1.0).alias("replicate_0"))
        #   bayes_bootstrap_weights returns lazy (survey_kit's narwhals
        #   convention) - collect back to a plain DataFrame to match what
        #   this function documents/returns.
        result = bayes_bootstrap_weights(
            df, prefix="replicate_", n_replicates=n_replicates, seed=s
        )
        return result.collect() if isinstance(result, pl.LazyFrame) else result

    return [_one(df, seed + i) for i, df in enumerate(df_implicates)]


if __name__ == "__main__":
    implicates = make_implicates()
    logger.info(f"Generated {len(implicates)} implicates, {implicates[0].height} rows each:")
    logger.info(implicates[0].head())
