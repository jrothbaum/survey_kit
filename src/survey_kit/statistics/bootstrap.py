#   Weights for bootrap

from __future__ import annotations
import polars as pl
import narwhals as nw
from narwhals.typing import IntoFrameT

from ..utilities.random import RandomData
from ..utilities.dataframe import (
    safe_height,
    NarwhalsType,
    concat_wrapper,
    lazy_backend,
)


def bayes_bootstrap_weights(
    df: IntoFrameT,
    weight: str = "",
    prefix: str = "",
    sum_to: int | None = None,
    n_replicates: int = 100,
    seed: int = 0,
    cluster: str | list[str] | None = None,
) -> IntoFrameT:
    """
    Attach n_replicates Bayesian bootstrap replicate weight columns to df.

    Parameters
    ----------
    df : IntoFrameT
        Data to attach replicate weights to.
    weight : str, optional
        An existing weight column - if set, each replicate multiplies it
        in (a weighted Bayesian bootstrap) rather than treating every row
        as equally likely to begin with. By default "" (every row starts
        equally weighted).
    prefix : str, optional
        Column name prefix for the replicate weights, by default "" (uses
        `weight` if set, else "__bb_weight_").
    sum_to : int | None, optional
        Rescale each replicate to sum to this total, by default None (no
        rescaling).
    n_replicates : int, optional
        Number of replicate weight columns to draw, by default 100.
    seed : int, optional
        Random seed, by default 0 (random).
    cluster : str | list[str] | None, optional
        Column(s) identifying a cluster/PSU (e.g. household id) to
        resample AT, rather than resampling individual rows - one Gamma
        draw per cluster per replicate, broadcast to every row in that
        cluster, instead of one independent draw per row. This matters
        for clustered/nested data: drawing independently per row (the
        default, cluster=None) manufactures spurious within-cluster
        variation in the replicate weights that doesn't correspond to any
        real resampling process - a real multistage survey design
        resamples PSUs, not individuals within them, so rows sharing a
        cluster should always move together across replicates. By default
        None (resample rows independently, unchanged from before this was
        added).

    Returns
    -------
    IntoFrameT
        df with n_replicates new columns "{prefix}{1..n_replicates}".
    """
    nw_type = NarwhalsType(df)

    if prefix == "":
        if weight == "":
            prefix = "__bb_weight_"
        else:
            prefix = weight

    n_rows = safe_height(df)

    if cluster is None:
        df_weights = nw_type.from_polars(
            bayes_bootstrap(n_rows=n_rows, n_draws=n_replicates, seed=seed, prefix=prefix)
        )
        df = concat_wrapper([df, df_weights], how="horizontal")
    else:
        cluster_list = [cluster] if isinstance(cluster, str) else list(cluster)
        df_pl = nw_type.to_polars().lazy().collect()

        #   One row per distinct cluster, sorted for a deterministic draw
        #       order given a fixed seed.
        unique_clusters = df_pl.select(cluster_list).unique().sort(cluster_list)
        n_clusters = safe_height(unique_clusters)

        #   One draw per cluster (not per row), then a left join broadcasts
        #       each cluster's draw to every row in it - the "resample at
        #       the PSU, apply to every member" step this parameter exists
        #       for.
        df_cluster_weights = pl.concat(
            [
                unique_clusters,
                bayes_bootstrap(
                    n_rows=n_clusters, n_draws=n_replicates, seed=seed, prefix=prefix
                ),
            ],
            how="horizontal",
        )
        df_pl = df_pl.join(df_cluster_weights, on=cluster_list, how="left")
        df = nw_type.from_polars(df_pl)

    if weight != "":
        c_weight_original = nw.col(weight)

    with_columns = []
    for i_boot in range(n_replicates):
        coli = f"{prefix}{i_boot + 1}"
        b_add = False
        c_weighti = nw.col(coli)

        if weight != "":
            b_add = True
            c_weighti = c_weighti * c_weight_original

        if sum_to is not None:
            b_add = True
            c_weighti = c_weighti / c_weighti.sum() * sum_to

        if b_add:
            with_columns.append(c_weighti.alias(coli))

    if len(with_columns):
        df = nw.from_native(df).with_columns(with_columns).to_native()

    return lazy_backend(nw.from_native(df), nw_type).to_native()


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
