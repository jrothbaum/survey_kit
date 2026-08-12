#   Weights for bootrap

from __future__ import annotations
import polars as pl

from ..utilities.random import RandomData
from ..utilities.dataframe import safe_height, concat_wrapper


def bayes_bootstrap_weights(
    df: pl.LazyFrame | pl.DataFrame,
    weight: str = "",
    prefix: str = "",
    sum_to: int | None = None,
    n_replicates: int = 100,
    seed: int = 0,
) -> pl.LazyFrame | pl.DataFrame:
    if prefix == "":
        if weight == "":
            prefix = "__bb_weight_"
        else:
            prefix = weight

    n_rows = safe_height(df)

    df_weights = bayes_bootstrap(
        n_rows=n_rows, n_draws=n_replicates, seed=seed, prefix=prefix
    )

    df = concat_wrapper([df, df_weights], how="horizontal")

    if weight != "":
        c_weight_original = pl.col(weight)

    with_columns = []
    for i_boot in range(n_replicates):
        coli = f"{prefix}{i_boot + 1}"
        b_add = False
        c_weighti = pl.col(coli)

        if weight != "":
            b_add = True
            c_weighti = c_weighti * c_weight_original

        if sum_to is not None:
            b_add = True
            c_weighti = c_weighti / c_weighti.sum() * sum_to

        if b_add:
            with_columns.append(c_weighti.alias(coli))

    if len(with_columns):
        df = df.with_columns(with_columns)

    return df


def bayes_bootstrap(
    n_rows: int,
    n_draws: int = 1,
    seed: int = 0,
    prefix: str = "__bb_weight_",
    initial_weight_index: int = 1,
) -> pl.DataFrame:
    rd = RandomData(seed=seed, n_rows=n_rows)

    for i in range(n_draws):
        rd.np_distribution(
            name=f"{prefix}{i + initial_weight_index}",
            distribution="gamma",
            shape=1,
            scale=1,
        )

    return rd.to_df()
