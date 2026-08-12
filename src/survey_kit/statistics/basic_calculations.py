from __future__ import annotations
from typing import Callable


import polars as pl
import polars.selectors as cs

from ..utilities.inputs import list_input

from ..utilities.dataframe import (
    _columns_original_order,
    safe_sum_cast,
    join_wrapper,
    concat_wrapper,
    fill_missing,
    columns_from_list,
)

from ..utilities.compress import compress_df
from .. import logger


def calculate_by(
    df: pl.LazyFrame | pl.DataFrame,
    column_stats: dict[str, list[str]],
    by: dict[str, list[str] | str] | list[list[str] | str] | None = None,
    weight: str = "",
    no_suffix: bool = False,
    quantile_interpolated: bool = False,
    quantile_interpolated_interval: int | float = 2_500,
    always_return_as_collection: bool = False,
) -> dict[str, pl.LazyFrame | pl.DataFrame] | list[pl.LazyFrame | pl.DataFrame] | pl.LazyFrame | pl.DataFrame:
    if by is None:
        by = []

    df = df.lazy()

    columns_to_keep = []
    for coli in column_stats.keys():
        (coli, _, coli_original) = _check_special_modifiers(coli)
        # print(f"{coli}->{coli_original}")

        columns_to_keep.append(coli_original)

    if weight != "":
        columns_to_keep.append(weight)

    if by is None:
        by = dict(all=[])

    if type(by) is list:
        if all(type(itemi) is str for itemi in by):
            by = [by]

        by_for_loop = {i: byi for i, byi in enumerate(by)}

    else:
        by_for_loop = by

    for namei, byi in by_for_loop.items():
        if type(byi) is list:
            columns_to_keep.extend(byi)
        else:
            columns_to_keep.append(byi)

    #   De-dedup and set in original order from df
    columns_to_keep = _columns_original_order(
        list(set(columns_to_keep)), df.collect_schema().names()
    )

    #   Summary stats on booleans don't really work in polars or R
    df = df.select(columns_to_keep).with_columns(cs.boolean().cast(pl.Int8))

    df_out = {}
    if weight != "":
        df = safe_sum_cast(df=df, columns=weight).lazy()

    #   Construct the list of stats
    stats = []
    names_already = []

    share_stats = []

    for coli, stats_to_calculate in column_stats.items():
        stats_to_calculate = list_input(stats_to_calculate)
        for stati in stats_to_calculate:
            col_stat_info = _summary_by_column_stat(
                column=coli, statistic=stati, weight=weight
            )

            if col_stat_info.modifier == "share":
                share_stats.append(col_stat_info)

            if col_stat_info.need_sum_cast:
                df = safe_sum_cast(
                    df=df, columns=col_stat_info.column_name
                ).lazy()

            if col_stat_info.stat_expr is not None:
                if col_stat_info.output_name not in names_already:
                    stats.append(col_stat_info.stat_expr)
                    names_already.append(col_stat_info.output_name)

    if len(share_stats):
        df_shares = df.select([stati.stat_expr for stati in share_stats]).collect()
        columns_share = df_shares.columns

        d_shares = {}
        row0 = df_shares.row(0)
        for j, colj in enumerate(columns_share):
            d_shares[colj] = row0[j]
        # print(d_shares)
        # d_shares = df_shares.to_polars().to_dicts()[0]
        # print(d_shares)

    df_out = {}
    for bynamei, byi in by_for_loop.items():
        if len(byi):
            df_byi = df.group_by(byi).agg(stats).sort(byi)
        else:
            df_byi = df.select(stats)

        if len(share_stats):
            with_share = []

            for col_stat_infoi in share_stats:
                namei = col_stat_infoi.output_name
                with_share.append(
                    (pl.col(namei) / pl.lit(d_shares[namei])).alias(namei)
                )

            df_byi = df_byi.with_columns(with_share)

        df_join = []

        df_q = _quantiles(
            df,
            column_stats=column_stats,
            by=byi,
            weight=weight,
            interpolated=quantile_interpolated,
            interpolated_interval=quantile_interpolated_interval,
        )

        if df_q is not None:
            df_join.append(df_q)

        df_gini = _custom_stat_by(
            df,
            stat_name="gini",
            delegate=_gini,
            column_stats=column_stats,
            by=byi,
            weight=weight,
        )

        if df_gini is not None:
            df_join.append(df_gini)

        if len(df_join):
            if len(byi):
                for df_joini in df_join:
                    df_byi = join_wrapper(df_byi, df_joini, on=byi, how="full").sort(
                        byi
                    )

            else:
                #   No by, just one row
                df_byi = concat_wrapper(
                    [df_byi.collect()] + [dfi.collect() for dfi in df_join],
                    how="horizontal",
                ).lazy()

        df_out[bynamei] = compress_df(
            fill_missing(df_byi, value=None).lazy().collect(),
            no_boolean=True,
        )

    if no_suffix:
        for bynamei, byi in by_for_loop.items():
            for coli in list(column_stats.keys()):
                cols_thesevars = columns_from_list(df_out[bynamei], f"{coli}_*")

                if len(cols_thesevars) == 1:
                    df_out[bynamei] = df_out[bynamei].rename(
                        {cols_thesevars[0]: coli}
                    )
    if len(df_out) == 1 and not always_return_as_collection:
        return next(iter(df_out.values()))
    else:
        if type(by) is list:
            return list(df_out.values())
        else:
            return df_out


def _check_special_modifiers(column: str) -> tuple[str, str, str]:
    #   Special modifiers - pipe separated
    special_modifiers = ["missing", "notmissing", "not0", "is0", "share"]
    modifier = ""
    for modi in special_modifiers:
        if column.endswith(f"|{modi}"):
            column = column[0 : (len(column) - len(f"|{modi}"))]
            modifier = modi

    #   Aliases
    column_original = column
    if column == "n":
        column = "rawcount"
    elif column == "weight":
        column = "count"

    return (column, modifier, column_original)


class _ColumnStatInformation:
    def __init__(
        self,
        stat_expr: pl.Expr,
        need_sum_cast: bool,
        column_name: str,
        modifier: str,
        output_name: str,
    ):
        self.stat_expr = stat_expr
        self.need_sum_cast = need_sum_cast
        self.column_name = column_name
        self.modifier = modifier
        self.output_name = output_name


def _summary_by_column_stat(
    column: str = "", statistic: str = "", weight: str = ""
) -> _ColumnStatInformation:
    original_column = column

    #   Aliases
    if statistic == "median":
        statistic = "q50"
    elif statistic == "n":
        statistic = "rawcount"
    elif statistic == "weight":
        statistic = "count"

    (column, modifier, column_original) = _check_special_modifiers(column)

    if (
        statistic.startswith("count_")
        or statistic.startswith("rawcount_")
        or statistic.startswith("share_")
    ):
        [statistic, modifier] = statistic.split("_")

    suffix = stat_suffix(statistic=statistic, modifier=modifier)

    c_filter = _summary_by_modifier_filter(column_original, modifier, weight)

    arguments = {
        "column": column_original,
        "c_filter": c_filter,
        "weight": weight,
        "suffix": suffix,
    }

    statout = None
    b_safe_sum_cast = False
    if statistic == "mean":
        b_safe_sum_cast = True
        statout = _mean(**arguments)
    elif statistic == "sum":
        b_safe_sum_cast = True
        statout = _sum(**arguments)
    elif statistic == "count":
        b_safe_sum_cast = (weight != "") or (c_filter is not None)
        statout = _count(**arguments)
    elif statistic == "rawcount":
        statout = _rawcount(**arguments)
    elif statistic == "share":
        statout = _share(**arguments)
    elif statistic == "rawshare":
        statout = _rawshare(**arguments)
    elif statistic == "var":
        statout = _var(**arguments)
    elif statistic == "std":
        statout = _std(**arguments)
    elif statistic == "max":
        statout = _max(**arguments)
    elif statistic == "min":
        statout = _min(**arguments)
    elif statistic == "first":
        statout = pl.col(column).first()

    #   For anything else, do nothing

    #   Rename
    output_name = f"{column}_{suffix}"
    if statout is not None:
        statout = statout.alias(f"{column}_{suffix}")

    return _ColumnStatInformation(
        stat_expr=statout,
        need_sum_cast=b_safe_sum_cast,
        column_name=column_original,
        modifier=modifier,
        output_name=output_name,
    )


def stat_suffix(statistic: str = "", modifier: str = "") -> str:
    if modifier != "":
        modifier_suffix = f"_{modifier}"
    else:
        modifier_suffix = ""

    if statistic in ["mean", "sum", "var", "std", "max", "min", "first", "gini"]:
        suffix = statistic + modifier_suffix
    elif statistic == "median":
        suffix = "q0_5" + modifier_suffix
    elif statistic.startswith("q") or statistic.startswith("p"):
        quantile = float(statistic.replace("q", "").replace("p", "")) / 100
        suffix = f"q{str(quantile).replace('.', '_')}" + modifier_suffix
    elif (
        statistic.startswith("count")
        or statistic.startswith("rawcount")
        or statistic.startswith("share")
        or statistic.startswith("rawshare")
        or statistic == "n"
        or statistic == "weight"
    ):
        if statistic.startswith("count") or statistic == "weight":
            count_prefix = "n"
        elif statistic.startswith("rawcount") or statistic == "n":
            count_prefix = "rawn"
        elif statistic.startswith("share"):
            count_prefix = "share"
        elif statistic.startswith("rawshare"):
            count_prefix = "rawshare"

        count_suffix = ""
        suffixes = ["_not0", "_is0", "_notmissing", "_missing", "_share"]
        for si in suffixes:
            if statistic.endswith(si):
                count_suffix = si

        suffix = f"{count_prefix}{count_suffix}{modifier_suffix}"

    try:
        return suffix
    except:
        message = f"{statistic} is not a valid statistic"
        logger.error(message)
        raise Exception(message)


def _summary_by_modifier_filter(
    column: str, modifier: str, weight: str = ""
) -> pl.Expr:
    c_filter = None
    if modifier == "not0":
        c_filter = pl.col(column) != 0
    elif modifier == "notmissing":
        c_filter = ~pl.col(column).is_null()
    elif modifier == "missing":
        c_filter = pl.col(column).is_null()
    elif modifier == "is0":
        c_filter = pl.col(column) == 0

    if weight != "":
        c_weight = (pl.col(weight) != 0) & ~(pl.col(weight).is_null())

        if c_filter is None:
            c_filter = c_weight
        else:
            c_filter = c_filter & c_weight

    if c_filter is not None:
        c_filter = c_filter.cast(pl.Int8)

    return c_filter


def _mean(
    column: str, c_filter: pl.Expr | None, weight: str = "", suffix: str = ""
) -> pl.Expr:
    c_col = pl.col(column)
    if weight != "":
        c_weight = pl.col(weight)
        statout = (c_filter * c_col * c_weight).sum() / (
            c_filter * (~c_col.is_null()).cast(pl.Int8) * c_weight
        ).sum()
    else:
        if c_filter is not None:
            statout = (c_col * c_filter).sum() / c_filter.sum()
        else:
            statout = (c_col).mean()

    return statout


def _sum(
    column: str, c_filter: pl.Expr | None, weight: str = "", suffix: str = ""
) -> pl.Expr:
    c_col = pl.col(column)
    if weight != "":
        c_weight = pl.col(weight)
        statout = (c_filter * c_col * c_weight).sum()
    else:
        if c_filter is not None:
            statout = (c_col * c_filter).sum()
        else:
            statout = (c_col).sum()

    return statout


def _count(
    column: str, c_filter: pl.Expr | None, weight: str = "", suffix: str = ""
) -> pl.Expr:
    if weight != "":
        c_weight = pl.col(weight)
        statout = (c_filter * c_weight).sum()
    else:
        if c_filter is not None:
            statout = c_filter.sum()
        else:
            statout = pl.len()

    return statout


def _rawcount(
    column: str, c_filter: pl.Expr | None, weight: str = "", suffix: str = ""
) -> pl.Expr:
    return _count(column=column, c_filter=c_filter, suffix=suffix)


def _share(
    column: str, c_filter: pl.Expr | None, weight: str = "", suffix: str = ""
) -> pl.Expr:
    if weight != "":
        c_weight = pl.col(weight)
        statout = (c_filter * c_weight).sum() / c_weight.sum()
    else:
        if c_filter is not None:
            statout = c_filter.sum() / pl.len()
        else:
            statout = pl.len()

    return statout


def _rawshare(
    column: str, c_filter: pl.Expr | None, weight: str = "", suffix: str = ""
) -> pl.Expr:
    return _share(column=column, c_filter=c_filter, suffix=suffix)


def _var(
    column: str, c_filter: pl.Expr | None, weight: str = "", suffix: str = ""
) -> pl.Expr:
    c_col = pl.col(column)
    if weight != "":
        c_weight = pl.col(weight)
        c_mean = _mean(column=column, c_filter=c_filter, weight=weight, suffix=suffix)
        c_n = _rawcount(column=column, c_filter=c_filter, weight=weight, suffix=suffix)
        num = (c_weight * c_filter * ((c_col - c_mean) ** 2)).sum()
        denom = ((c_n - 1) / c_n) * (c_filter * c_weight).sum()

        statout = num / denom
    else:
        if c_filter is not None:
            statout = (
                pl.when(c_filter.cast(pl.Boolean)).then(c_col).otherwise(pl.lit(None))
            ).var()
        else:
            statout = c_col.var()

    return statout


def _std(
    column: str, c_filter: pl.Expr | None, weight: str = "", suffix: str = ""
) -> pl.Expr:
    statout = (
        _var(column=column, c_filter=c_filter, weight=weight, suffix=suffix) ** 0.5
    )

    return statout


def _max(
    column: str, c_filter: pl.Expr | None, weight: str = "", suffix: str = ""
) -> pl.Expr:
    c_col = pl.col(column)
    if c_filter is not None:
        statout = (
            pl.when(c_filter.cast(pl.Boolean)).then(c_col).otherwise(pl.lit(None))
        ).max()
    else:
        statout = c_col.max()

    return statout


def _min(
    column: str, c_filter: pl.Expr | None, weight: str = "", suffix: str = ""
) -> pl.Expr:
    c_col = pl.col(column)
    if c_filter is not None:
        statout = (
            pl.when(c_filter.cast(pl.Boolean)).then(c_col).otherwise(pl.lit(None))
        ).min()
    else:
        statout = c_col.min()

    return statout


def _quantiles(
    df: pl.LazyFrame,
    column_stats: dict[str, list[str]],
    by: str | list | None = None,
    weight: str = "",
    interpolated: bool = True,
    interpolated_interval: int = 2500,
):
    if weight != "":
        df = df.with_columns((pl.col(weight) / pl.col(weight).sum()).alias(weight))

    #   Get all the quantile stats we're trying to generate
    column_stats_quantiles_only = {}
    for vari, statsi in column_stats.items():
        q_statsi = []

        for subi in statsi:
            if subi == "median":
                q_statsi.append(0.5)
            elif subi.startswith("q") or subi.startswith("p"):
                q_statsi.append(float(subi.replace("p", "q").replace("q", "")) / 100)

        if len(q_statsi):
            q_statsi = sorted(list(set(q_statsi)))
            column_stats_quantiles_only[vari] = q_statsi

    if len(column_stats_quantiles_only) == 0:
        return None

    if interpolated:
        pass
        #   logger.info("Interpolated")
        df_out = _quantiles_interpolated(
            df=df,
            column_stats=column_stats_quantiles_only,
            by=by,
            weight=weight,
            interpolated_interval=interpolated_interval,
        )
    else:
        #   logger.info("Not interpolated")
        df_out = _quantiles_actual(
            df=df, column_stats=column_stats_quantiles_only, by=by, weight=weight
        )
    return df_out


def _quantiles_actual(
    df: pl.LazyFrame,
    column_stats: dict[str, list[str]],
    by: str | list | None = None,
    weight: str = "",
) -> pl.LazyFrame:
    if type(by) is str:
        by = [by]
    if by is None or not len(by):
        by = []

    columns = list(column_stats.keys())
    if weight == "":
        n = df.select(pl.len()).collect().item(0, 0)
        weight = "___weight___"
        df = df.with_columns(pl.lit(1 / n).alias(weight))

    columns_nameonly = []
    for coli in columns:
        columns_nameonly.append(coli.split("|")[0])

    keep_list = list(set(columns_nameonly)) + [weight] + by

    df = df.select(keep_list)

    df_out = None
    df_by = None
    for coli, qlisti in column_stats.items():
        (coli_final, modifier, coli) = _check_special_modifiers(coli)

        #   logger.info(f"{coli}: {qlisti}")
        sorted_coli = by + [coli]

        keep_condition = ~pl.col(coli).is_null()
        if modifier == "not0":
            keep_condition = keep_condition & (pl.col(coli) != 0)
        else:
            modifier = ""

        if len(by):
            calc_over = (
                pl.col(weight).cum_sum().over(by, order_by=coli)
                / pl.col(weight).sum().over(by)
            ).alias(weight)
        else:
            calc_over = (
                pl.col(weight).cum_sum().over(order_by=[coli]) / pl.col(weight).sum()
            ).alias(weight)

        df_coli = (
            df.filter(keep_condition)
            .group_by(sorted_coli)
            .agg(pl.col(weight).sum())
            .sort(sorted_coli)
            .with_columns(calc_over)
        )

        if df_by is None:
            if len(by):
                df_by = df_coli.select(by).unique()

        #   Create rows with the quantiles we want to calculate
        if df_by is None:
            df_byi = pl.DataFrame({weight: qlisti}).lazy()
        else:
            df_q = pl.DataFrame({weight: qlisti}).lazy()

            df_byi = join_wrapper(
                df_by.with_columns(pl.lit(1).alias("___stats_join_full___")),
                df_q.with_columns(pl.lit(1).alias("___stats_join_full___")),
                how="full",
                on=["___stats_join_full___"],
            ).drop("___stats_join_full___")

        #   Add those rows to the table, with empty values
        #       for the income item
        #   Get the range to interpolate over (with_columns, shift)
        #   Filter to the relevant quantiles (missing coli)
        #   Get the percentile value
        df_coli = (
            concat_wrapper([df_coli, df_byi], how="diagonal")
            .sort(by + [weight])
            .with_columns(
                pl.col(coli)
                .shift(-1)
                .over(order_by=by + [weight])
                .alias("___y_above___")
            )
            .filter(pl.col(coli).is_null())
            .drop(coli)
            .rename({"___y_above___": coli})
        )

        #   Reshape into an output table
        rename = {
            str(
                qi
            ): f"{coli_final}_{stat_suffix(statistic='q' + str(qi * 100), modifier=modifier)}"
            for qi in qlisti
        }

        temp_by = False
        if len(by) == 0:
            by = ["___by___"]
            df_coli = df_coli.with_columns(pl.lit(1).alias(by[0]))

            temp_by = True

        if df_by is None:
            df_coli = df_coli.with_columns(
                pl.col(coli).fill_null(strategy="backward").over(order_by=by + [weight])
            )
        else:
            df_coli = df_coli.with_columns(
                pl.col(coli)
                .fill_null(strategy="backward")
                .over(by, order_by=by + [weight])
            )

        df_coli = df_coli.collect().pivot(index=by, on=weight, values=coli)
        df_coli = df_coli.rename(rename).sort(by)

        if temp_by:
            df_coli = df_coli.drop(by)
            by = []

        if df_out is None:
            df_out = df_coli
        else:
            if len(by):
                df_out = join_wrapper(df_out, df_coli, how="full", on=by)
            else:
                df_out = concat_wrapper([df_out, df_coli], how="horizontal")

    return df_out.lazy()


def _quantile_interpolated_bin_keys(
    df_binned,
    sorted_coli: list,
    by: list,
    coli: str,
    drb_safe_n_bin: int,
):
    """
    Replicate-independent step of interpolated-quantile binning: given one
    row per (by + [coli]) bin with its row count, adds the zero-weight
    lower-boundary anchor row and merges small bins (row count <
    drb_safe_n_bin) forward into the next bin within the same by-group, for
    disclosure safety. Weight never enters this decision (only row counts
    do), and dropping a row from a table that already carries a cumulative
    value doesn't change that value at the surviving rows - so the surviving
    bin keys computed here can be joined against ANY replicate's cumulative
    share afterward instead of being recomputed per replicate.
    """
    var_n_in_bin = "___n_in_bin"
    var_n_cum_sum = "___n_to_bin"

    df_coli = df_binned

    if len(by):
        df_first = df_coli.group_by(by).agg(pl.all().first())
    else:
        df_first = df_coli.head(1)
    df_first = df_first.with_columns(
        [
            pl.col(coli) - 1,
            pl.lit(drb_safe_n_bin).cast(pl.UInt32).alias(var_n_in_bin),
        ]
    )

    if drb_safe_n_bin:
        smallest_bin = df_coli.select(pl.min(var_n_in_bin))[0, 0]

        while smallest_bin < drb_safe_n_bin:
            c_too_small = pl.col(var_n_in_bin) < drb_safe_n_bin
            if len(by):
                c_prior_too_small = (
                    pl.col(var_n_in_bin)
                    .shift(n=1)
                    .over(by, order_by=sorted_coli)
                    .fill_null(pl.lit(drb_safe_n_bin + 1))
                    < drb_safe_n_bin
                )
                cum_sum_expr = (
                    pl.col(var_n_in_bin).cum_sum().over(by, order_by=sorted_coli)
                )
                prior_cum_sum_expr = (
                    pl.col(var_n_cum_sum)
                    .shift(n=1)
                    .over(by, order_by=sorted_coli)
                    .fill_null(pl.lit(0))
                )
            else:
                c_prior_too_small = (
                    pl.col(var_n_in_bin)
                    .shift(n=1)
                    .fill_null(pl.lit(drb_safe_n_bin + 1))
                    < drb_safe_n_bin
                )
                cum_sum_expr = pl.col(var_n_in_bin).cum_sum().over(order_by=sorted_coli)
                prior_cum_sum_expr = (
                    pl.col(var_n_cum_sum).shift(n=1).fill_null(pl.lit(0))
                )

            df_coli = (
                df_coli.with_columns([cum_sum_expr.alias(var_n_cum_sum)])
                .filter(~(c_too_small & ~c_prior_too_small))
                .with_columns(
                    [(pl.col(var_n_cum_sum) - prior_cum_sum_expr).alias(var_n_in_bin)]
                )
                .drop(var_n_cum_sum)
            )
            smallest_bin = df_coli.select(pl.min(var_n_in_bin))[0, 0]

    df_coli = concat_wrapper([df_first, df_coli], how="diagonal")
    return df_coli.select(sorted_coli)


def _quantiles_interpolated(
    df: pl.LazyFrame,
    column_stats: dict[str, list[str]],
    by: str | list | None = None,
    weight: str = "",
    interpolated_interval: int = 2500,
    drb_safe_n_bin: int = 10,
):
    df = df.lazy()

    if by is None or not len(by):
        by = []

    columns = list(column_stats.keys())
    if weight == "":
        n = df.select(pl.len()).collect().item(0, 0)
        weight = "___weight___"
        df = df.with_columns(pl.lit(1 / n).alias(weight))

    columns_nameonly = []
    for coli in columns:
        columns_nameonly.append(coli.split("|")[0])

    columns_nameonly = list(set(columns_nameonly))
    keep_list = columns_nameonly + [weight] + by

    #   Bin the income
    with_intervals = []
    for coli in columns:
        col_split = coli.split("|")

        if len(col_split) == 2:
            col_name = col_split[0]
            modifier = col_split[1]
        else:
            col_name = coli
            modifier = ""

        c_col = pl.col(col_name)

        #   Truncating (not floor) division to match the exact binning
        #   behavior this always had - c_col (income) can be negative,
        #   where floor vs. truncating division differ.
        with_floor = 1 + (c_col / interpolated_interval).cast(pl.Int64)
        if modifier == "not0":
            c_cleared = pl.when(c_col != 0).then(with_floor).otherwise(pl.lit(None))
            with_intervals.append(c_cleared.alias(coli))
        else:
            with_intervals.append(with_floor.alias(col_name))

    df = df.select(keep_list).with_columns(with_intervals)

    df_out = None
    df_by = None

    for coli, qlisti in column_stats.items():
        (coli_nameonly, modifier, coli_original) = _check_special_modifiers(coli)

        sorted_coli = by + [coli]

        #   keep_condition = ~(pl.col(coli).is_null() | pl.col(coli).is_nan()) & ~(pl.col(weight).is_null() | pl.col(weight).is_nan())
        keep_condition = ~(pl.col(coli).is_null()) & ~(
            pl.col(weight).is_null() | pl.col(weight).is_nan()
        )

        if len(by):
            by_over = pl.col(weight) / pl.col(weight).sum().over(by)
            calc_over = (
                pl.col(weight).cum_sum().over(by, order_by=coli)
                / pl.col(weight).sum().over(by)
            ).alias(weight)
        else:
            by_over = pl.col(weight) / pl.col(weight).sum()
            calc_over = (
                pl.col(weight).cum_sum().over(order_by=coli) / pl.col(weight).sum()
            ).alias(weight)

        var_n_in_bin = "___n_in_bin"
        df_grouped = (
            df.filter(keep_condition)
            .with_columns(by_over)
            .with_columns(cs.by_dtype(pl.Int8).cast(pl.Int32))
            .group_by(sorted_coli)
            .agg(
                [
                    pl.col(weight).sum(),
                    pl.len().alias(var_n_in_bin),
                ]
            )
            .sort(sorted_coli)
            .with_columns([calc_over])
            .collect()
        )

        #   The merge decision (which small bins get folded forward into
        #   the next bin, for disclosure safety) depends only on row
        #   counts, never on weight - so it's delegated to a shared helper
        #   that the batched replicate path also calls, rather than being
        #   recomputed per replicate. The synthetic anchor row it returns
        #   has no weight of its own (nothing merges below the lowest real
        #   bin), so it gets a cumulative share of 0 after the join below.
        survivor_keys = _quantile_interpolated_bin_keys(
            df_binned=df_grouped.select(sorted_coli + [var_n_in_bin]),
            sorted_coli=sorted_coli,
            by=by,
            coli=coli,
            drb_safe_n_bin=drb_safe_n_bin,
        )
        df_coli = survivor_keys.join(
            df_grouped.select(sorted_coli + [weight]), on=sorted_coli, how="left"
        ).with_columns(pl.col(weight).fill_null(pl.lit(0.0)))

        if df_by is None:
            if len(by):
                df_by = df_coli.select(by).unique()

        #   Create rows with the quantiles we want to calculate
        if df_by is None:
            df_byi = pl.DataFrame({weight: qlisti}).lazy()
        else:
            df_q = pl.DataFrame({weight: qlisti}).lazy()

            df_byi = join_wrapper(
                df_by.with_columns(pl.lit(1).alias("___stats_join_full___")),
                df_q.with_columns(pl.lit(1).alias("___stats_join_full___")),
                how="full",
                on=["___stats_join_full___"],
            ).drop("___stats_join_full___")

        #   Column references for convenience
        list_w_below = []
        list_w_above = []
        list_y_below = []
        list_y_above = []

        with_shift = []
        c_w = pl.col(weight)
        c_var = pl.col(coli)

        #   This runs on the table sorted by by + [weight] - the shifts
        #   below must stay within each by-group (partitioned via .over),
        #   otherwise the lowest quantile of one group could interpolate
        #   using a bin belonging to a neighboring group that happens to
        #   sort adjacently by cumulative share.
        def _shift_over(expr, n):
            if len(by):
                return expr.shift(n).over(by, order_by=weight)
            return expr.shift(n)

        for shifti in range(1, len(qlisti) + 1):
            with_shift.extend(
                [
                    (
                        pl.when(_shift_over(c_var, shifti).is_not_null())
                        .then(_shift_over(c_w, shifti))
                        .otherwise(pl.lit(None))
                        .alias(f"__w_below___{shifti}")
                    ),
                    (
                        pl.when(_shift_over(c_var, -shifti).is_not_null())
                        .then(_shift_over(c_w, -shifti))
                        .otherwise(pl.lit(None))
                        .alias(f"__w_above___{shifti}")
                    ),
                    _shift_over(c_var, shifti).alias(f"__y_below___{shifti}"),
                    _shift_over(c_var, -shifti).alias(f"__y_above___{shifti}"),
                ]
            )

            list_w_below.append(f"__w_below___{shifti}")
            list_w_above.append(f"__w_above___{shifti}")
            list_y_below.append(f"__y_below___{shifti}")
            list_y_above.append(f"__y_above___{shifti}")

        w_gap = pl.col(weight) - pl.col("__w_below")
        y_below = pl.col("__y_below")
        y_above = pl.col("__y_above")
        y_interval = y_above - y_below

        w_below = pl.col("__w_below")
        w_above = pl.col("__w_above")
        w_interval = w_above - w_below

        val_interpolated = (
            y_below + (w_gap / w_interval) * y_interval
        ) * interpolated_interval

        df_coli = (
            concat_wrapper(
                [df_coli.lazy(), df_byi.lazy()],
                how="diagonal",
            )
            .sort(by + [weight])
            .collect()
            .with_columns(with_shift)
            .lazy()
            .filter(c_var.is_null())
            .select(
                by
                + [coli, weight]
                + list_w_below
                + list_w_above
                + list_y_below
                + list_y_above
            )
            .with_columns(
                [
                    pl.coalesce(list_w_below).alias("__w_below"),
                    pl.coalesce(list_y_below).alias("__y_below"),
                    pl.coalesce(list_w_above).alias("__w_above"),
                    pl.coalesce(list_y_above).alias("__y_above"),
                ]
            )
            .with_columns(
                [
                    val_interpolated.alias(coli),
                    # y_below.alias("test_below"),
                    # w_gap.alias("test_w_gap"),
                    # w_interval.alias("test_w_interval")
                ]
            )
            .drop(
                list_w_below
                + list_w_above
                + list_y_below
                + list_y_above
                + ["__w_below", "__y_below", "__w_above", "__y_above"]
            )
        )

        # #   Add those rows to the table, with empty values
        # #       for the income item
        # #   Get the range to interpolate over (with_columns, shift)
        # #   Filter to the relevant quantiles (missing coli)
        # #   Calculate the interpolated value
        # df_coli = AppendList([df_coli,
        #                       df_byi],
        #                      quietly=True)\
        #             .sort(by + [weight])\
        #             .with_columns(
        #                     [pl.col(weight).shift(1).alias("___w_below___"),
        #                      pl.col(weight).shift(-1).alias("___w_above___"),
        #                      pl.col(coli).shift(1).alias("___y_below___")]
        #                 )\
        #             .filter(pl.col(coli).is_null())\
        #             .with_columns(
        #                     (y_below + interpolated_interval*(qi-w_below)/w_interval).alias(coli)
        #                 )\
        #             .drop(["___w_below___",
        #                    "___w_above___",
        #                    "___y_below___"])
        # #   `lower_bound' + `thisinterval'*((`qi'-`wgt_in_prev_intervals')/`interval_wgt')

        #   Reshape into an output table
        rename = {
            str(
                qi
            ): f"{coli_nameonly}_{stat_suffix(statistic='q' + str(qi * 100), modifier=modifier)}"
            for qi in qlisti
        }

        temp_by = False
        if len(by) == 0:
            by = ["___by___"]
            df_coli = df_coli.with_columns(pl.lit(1).alias(by[0]))

            temp_by = True

        df_coli = df_coli.collect().pivot(index=by, on=weight, values=coli)
        df_coli = df_coli.lazy()

        for coli in df_coli.collect_schema().names():
            rename_to_str = {}
            if type(coli) is not str:
                rename_to_str[coli] = str(coli)

            if len(rename_to_str):
                df_coli = df_coli.rename(rename_to_str)

        df_coli = df_coli.rename(rename).sort(by)

        if temp_by:
            df_coli = df_coli.drop(by)
            by = []

        if df_out is None:
            df_out = df_coli
        else:
            if len(by):
                df_out = join_wrapper(df_out, df_coli, on=by, how="full")

            else:
                df_out = concat_wrapper(
                    [df_out.collect(), df_coli.collect()], how="horizontal"
                ).lazy()

    return df_out


def _custom_stat_by(
    df: pl.LazyFrame,
    stat_name: str,
    delegate: Callable,
    column_stats: dict[str, list[str]],
    by: str | list | None = None,
    weight: str = "",
    **kwargs,
):
    vars_custom = []
    for vari, statsi in column_stats.items():
        if stat_name in statsi:
            vars_custom.append(vari)

    if len(vars_custom) == 0:
        return None

    if type(by) is str:
        by = [by]
    if by is None or not len(by):
        by = []

    columns_nameonly = []
    for coli in vars_custom:
        columns_nameonly.append(coli.split("|")[0])

    columns_nameonly = list(set(columns_nameonly))

    keep_list = list(set(columns_nameonly)) + by
    if weight != "":
        keep_list.append(weight)

    df = df.select(keep_list)

    by_filters = []
    if len(by):
        df_by_groups = df.select(by).unique().collect()

        for i, rowi in enumerate(df_by_groups.rows()):
            wherei = None

            for j, coli in enumerate(by):
                condi = pl.col(coli) == rowi[j]

                if wherei is None:
                    wherei = condi
                else:
                    wherei = wherei & condi

            by_filters.append(wherei)
    else:
        by_filters = [None]

    df_out = None
    for coli in vars_custom:
        (coli_final, modifier, coli) = _check_special_modifiers(coli)

        keep_condition = pl.col(coli).is_not_null()
        if modifier == "not0":
            keep_condition = keep_condition & pl.col(coli).ne(0)
        else:
            modifier = ""

        df_outi = None
        for filteri in by_filters:
            if filteri is None:
                df_byi = df
            else:
                df_byi = df.filter(filteri)

            df_byi = df_byi.filter(keep_condition)

            df_coli = delegate(df=df_byi, weight=weight, variable=coli, **kwargs)

            rename_to = f"{coli_final}_{stat_name}"
            if modifier != "":
                rename_to += f"_{modifier}"
            df_coli = df_coli.rename({stat_name: rename_to})

            if len(by):
                df_coli = concat_wrapper(
                    [
                        df_byi.select(by).head(1).collect(),
                        df_coli.collect(),
                    ],
                    how="horizontal",
                ).lazy()

            if df_outi is None:
                df_outi = df_coli
            else:
                df_outi = concat_wrapper([df_outi, df_coli], how="diagonal")

        if df_out is None:
            df_out = df_outi
        else:
            if len(by):
                df_out = join_wrapper(df_out, df_outi, how="full", on=by)
            else:
                df_out = concat_wrapper([df_out, df_outi], how="horizontal")

    return df_out


def _gini(
    df: pl.LazyFrame | pl.DataFrame,
    variable: str,
    weight: str = "",
    censor_at_zero: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    if weight == "":
        weight = "__gini_weight__"
        df = df.select(variable).with_columns(pl.lit(1).alias(weight))

    else:
        df = df.select([variable, weight])

    c_weight = pl.col(weight)
    df = df.with_columns((c_weight / c_weight.sum()).alias(weight))

    schema = df.collect_schema()
    if censor_at_zero:
        c_income = pl.col(variable) * pl.col(variable).gt(0).cast(schema[variable])
    else:
        c_income = pl.col(variable)

    df = df.sort(variable).filter(c_weight.is_not_null())

    #   Modified from SAS code
    #  retain swt swtey swt2ey swteycw 0;
    #  swt     = sum(swt,&pwgt); /* cumwgt */
    #   if (&income > 0) then do;
    #       swtey   = sum(swtey,(&pwgt*&income)); /* sum of gini area */
    # 	    swt2ey  = sum(swt2ey, (&pwgt*&pwgt*&income));
    # 		swteycw = sum(swteycw, (swt*&pwgt*&income));
    # 	end;
    #  End of file
    #  gini = ((2*swteycw-swt2ey)/(swt * swtey)-1);
    #   swt = pl.col(weight).fill_null(0).cum_sum().alias("swt")
    swt = pl.col(weight).cum_sum().over(order_by=variable).alias("swt")
    swtey = (c_weight * c_income).sum().alias("swtey")
    swt2ey = (c_weight.pow(2) * c_income).sum().alias("swt2ey")
    swteycw = (swt * c_weight * c_income).sum().alias("swteycw")

    df_gini = df.select([swt, swtey, swt2ey, swteycw]).tail(1).collect()

    columns_gini = df_gini.columns

    d = {}
    row0 = df_gini.row(0)
    for j, colj in enumerate(columns_gini):
        d[colj] = row0[j]
    # d = df_gini.to_dicts()[0]

    if d["swt"] is not None:
        try:
            gini = (2 * d["swteycw"] - d["swt2ey"]) / (d["swt"] * d["swtey"]) - 1
        except:
            gini = None
    else:
        gini = None

    df_out = pl.DataFrame({"gini": [gini]}).lazy()

    return df_out


##########################################################
##########################################################
#   Batched replicate-weight computation - START
##########################################################
##########################################################
#   Fast path for StatCalculator's replicate/bootstrap SE computation:
#   instead of one calculate_by() pass per replicate weight (repeating
#   group-membership computation R+1 times even though it only depends on
#   `by`, never on the weight column), build every replicate's expressions
#   up front and run them in a handful of batched group_by/agg passes.
#   Only covers the stats _is_batchable_stat classifies as batchable -
#   anything else (custom delegates, |share's global denominator, multiple
#   Statistics objects, multi-key by) falls back to the existing
#   per-replicate loop in replicates.py, unchanged.


def _is_batchable_stat(stati: str) -> str | None:
    """
    Classify a raw Statistics.stats-style stat string ("mean", "n|not0",
    "median", "q25", "gini", ...) for the batched replicate-SE path.

    Returns "simple", "quantile", "gini", or None (not batchable). The
    |not0/|is0/|missing/|notmissing filter modifiers are just extra filter
    expressions baked into the same stat_expr _summary_by_column_stat
    already builds, so they're batchable for every stat kind. |share (the
    global, cross-group share-of-total denominator) is only batched for
    "simple" stats - _batched_simple_stats computes the extra ungrouped
    denominator reduction calculate_by's own share_stats/d_shares mechanism
    computes for a single weight. Quantile/gini don't build their
    denominator through _summary_by_column_stat at all, so |share on those
    stays unbatched (falls back) rather than being silently ignored.
    """
    stat_mod = stati.split("|")
    stati_raw = stat_mod[0]
    modifier = stat_mod[1] if len(stat_mod) == 2 else ""

    #   Aliases mirrored from _summary_by_column_stat
    resolved = stati_raw
    if resolved == "median":
        resolved = "q50"
    elif resolved == "n":
        resolved = "rawcount"
    elif resolved == "weight":
        resolved = "count"

    if resolved == "gini":
        return None if modifier == "share" else "gini"

    if resolved.startswith("q") or resolved.startswith("p"):
        try:
            float(resolved.replace("q", "").replace("p", ""))
            return None if modifier == "share" else "quantile"
        except ValueError:
            return None

    if resolved in {
        "mean",
        "sum",
        "count",
        "rawcount",
        "var",
        "std",
        "max",
        "min",
        "first",
        "share",
        "rawshare",
    }:
        return "simple"

    return None


def _split_batchable_column_stats(
    column_stats: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], bool]:
    """
    Split a calculate_by-style column_stats dict into (simple, quantile,
    gini) sub-dicts by stat classification. The 4th return value is False
    if every stat was classified (fully batchable), True if anything is
    unrecognized - the caller should fall back entirely in that case.
    """
    simple: dict[str, list[str]] = {}
    quantile: dict[str, list[str]] = {}
    gini: dict[str, list[str]] = {}

    for coli, stats_list in column_stats.items():
        for stati in stats_list:
            kind = _is_batchable_stat(stati)
            if kind == "simple":
                simple.setdefault(coli, []).append(stati)
            elif kind == "quantile":
                quantile.setdefault(coli, []).append(stati)
            elif kind == "gini":
                gini.setdefault(coli, []).append(stati)
            else:
                return (simple, quantile, gini, True)

    return (simple, quantile, gini, False)


def _batched_simple_stats(
    df,
    column_stats: dict[str, list[str]],
    by_cols: list[str],
    weight_list: list[str],
    batch_size: int,
):
    """
    Batched replacement for calculate_by's plain-reduction stats loop.
    column_stats here must only contain stats _is_batchable_stat classifies
    as "simple" - reuses _summary_by_column_stat directly for every
    (column, stat, replicate weight) combination, so naming/aliasing/filter
    behavior is guaranteed identical to the existing single-weight path.

    A |share-modified stat (e.g. "mean|share") additionally needs the same
    stat_expr evaluated once over the WHOLE table (not grouped by by_cols)
    as its denominator - mirroring calculate_by's own
    share_stats/d_shares mechanism, just batched across every replicate in
    the batch as one extra ungrouped reduction pass instead of one pass per
    replicate.

    Returns {replicate_index: single-replicate output table}, each shaped
    like a single weight's calculate_by output (by_cols + "{col}_{suffix}"
    columns).
    """
    if not len(column_stats):
        return {}

    df_polars = df

    cast_cols = set()
    for coli, stats_list in column_stats.items():
        (_, _, coli_original) = _check_special_modifiers(coli)
        for stati in stats_list:
            info = _summary_by_column_stat(column=coli, statistic=stati, weight="")
            if info.need_sum_cast:
                cast_cols.add(coli_original)
    if len(cast_cols):
        df_polars = safe_sum_cast(df=df_polars, columns=list(cast_cols))

    ndf = df_polars.lazy()
    results = {}

    for batch_start in range(0, len(weight_list), batch_size):
        batch_weights = weight_list[batch_start : batch_start + batch_size]

        exprs = []
        share_exprs = []
        share_pairs = []
        for offset, weight_col in enumerate(batch_weights):
            r = batch_start + offset
            for coli, stats_list in column_stats.items():
                for stati in stats_list:
                    info = _summary_by_column_stat(
                        column=coli, statistic=stati, weight=weight_col
                    )
                    if info.stat_expr is not None:
                        out_col = f"{info.output_name}___rep{r}"
                        exprs.append(info.stat_expr.alias(out_col))
                        if info.modifier == "share":
                            denom_col = f"__share_denom__{out_col}"
                            share_exprs.append(info.stat_expr.alias(denom_col))
                            share_pairs.append((out_col, denom_col))

        if len(by_cols):
            wide = ndf.group_by(by_cols).agg(exprs).collect()
        else:
            wide = ndf.select(exprs).collect()

        if len(share_exprs):
            denoms = ndf.select(share_exprs).collect()
            wide = wide.with_columns(
                [
                    (pl.col(out_col) / pl.lit(denoms[0, denom_col])).alias(out_col)
                    for out_col, denom_col in share_pairs
                ]
            )

        #   calculate_by's own output goes through fill_missing(..., value=
        #   None), which (since value=None skips fill_null but still runs
        #   fill_nan) turns 0/0-style NaN artifacts (e.g. mean|missing on a
        #   column with no missing values) into proper nulls - _reshape_
        #   summary_tables doesn't repeat that cleanup, so without doing it
        #   here the batched path would leave raw NaN where the sequential
        #   path produces null, which can silently poison the downstream
        #   replicate-variance sum.
        wide = wide.with_columns(cs.numeric().fill_nan(None))

        for offset, weight_col in enumerate(batch_weights):
            r = batch_start + offset
            suffix = f"___rep{r}"
            rep_cols = [c for c in wide.columns if c.endswith(suffix)]
            rename = {c: c[: -len(suffix)] for c in rep_cols}
            results[r] = wide.select(by_cols + rep_cols).rename(rename)

    return results


def _batched_gini(
    df,
    column_stats: dict[str, list[str]],
    by_cols: list[str],
    weight_list: list[str],
    batch_size: int,
    censor_at_zero: bool = True,
):
    """
    Batched replacement for calculate_by's gini path (_custom_stat_by +
    _gini). Today's per-column/per-by-group/per-replicate implementation
    stacks two Python loops (one over every unique by-group value, one over
    replicates); this vectorizes both away via one shared
    .over(by, order_by=variable) pass per batch, matching the pattern
    _quantiles_actual already uses for its own cumulative-weight step.

    column_stats here must only contain "gini". Returns
    {replicate_index: single-replicate output table} shaped like
    _custom_stat_by's output (by_cols + "{col}_gini" columns).
    """
    gini_columns = list(column_stats.keys())
    if not len(gini_columns):
        return {}

    df_polars = df
    ndf_base = df_polars.lazy()

    results = {}

    for coli in gini_columns:
        (_, modifier, coli_original) = _check_special_modifiers(coli)

        c_keep_condition = pl.col(coli_original).is_not_null()
        if modifier == "not0":
            c_keep_condition = c_keep_condition & pl.col(coli_original).ne(0)

        if censor_at_zero:
            c_income = pl.col(coli_original) * pl.col(coli_original).gt(0).cast(
                pl.Float64
            )
        else:
            c_income = pl.col(coli_original)

        for batch_start in range(0, len(weight_list), batch_size):
            batch_weights = weight_list[batch_start : batch_start + batch_size]

            ndf = ndf_base.filter(c_keep_condition)

            with_cols = []
            for offset, weight_col in enumerate(batch_weights):
                r = batch_start + offset
                c_weight = pl.col(weight_col)
                if len(by_cols):
                    normalized = (c_weight / c_weight.sum().over(by_cols)).alias(
                        f"___cw{r}"
                    )
                else:
                    normalized = (c_weight / c_weight.sum()).alias(f"___cw{r}")
                with_cols.append(normalized)
            ndf = ndf.with_columns(with_cols)

            swt_cols = []
            for offset, weight_col in enumerate(batch_weights):
                r = batch_start + offset
                cw = pl.col(f"___cw{r}")
                if len(by_cols):
                    swt = cw.cum_sum().over(by_cols, order_by=coli_original).alias(
                        f"___swt{r}"
                    )
                else:
                    swt = cw.cum_sum().over(order_by=coli_original).alias(f"___swt{r}")
                swt_cols.append(swt)
            ndf = ndf.with_columns(swt_cols)

            agg_exprs = []
            for offset, weight_col in enumerate(batch_weights):
                r = batch_start + offset
                cw = pl.col(f"___cw{r}")
                swt = pl.col(f"___swt{r}")
                agg_exprs.append((cw * c_income).sum().alias(f"swtey___rep{r}"))
                agg_exprs.append((cw.pow(2) * c_income).sum().alias(f"swt2ey___rep{r}"))
                agg_exprs.append((swt * cw * c_income).sum().alias(f"swteycw___rep{r}"))
                agg_exprs.append(swt.max().alias(f"swt___rep{r}"))

            if len(by_cols):
                wide = ndf.group_by(by_cols).agg(agg_exprs).collect()
            else:
                wide = ndf.select(agg_exprs).collect()

            for offset, weight_col in enumerate(batch_weights):
                r = batch_start + offset
                gini_expr = (
                    (
                        2 * pl.col(f"swteycw___rep{r}")
                        - pl.col(f"swt2ey___rep{r}")
                    )
                    / (pl.col(f"swt___rep{r}") * pl.col(f"swtey___rep{r}"))
                    - 1
                ).alias(f"{coli_original}_gini")

                #   Matches calculate_by's fill_missing(..., value=None)
                #   cleanup - a degenerate all-zero (post-censoring) income
                #   by-group divides by zero and would otherwise leave raw
                #   NaN instead of null, which can poison the downstream
                #   replicate-variance sum.
                out = wide.select(by_cols + [gini_expr]).with_columns(
                    cs.numeric().fill_nan(None)
                )
                if r not in results:
                    results[r] = out
                else:
                    results[r] = join_wrapper(
                        results[r], out, on=by_cols, how="full"
                    ) if len(by_cols) else concat_wrapper(
                        [results[r], out], how="horizontal"
                    )

    return results


def _batched_quantiles(
    df,
    column_stats: dict[str, list[str]],
    by_cols: list[str],
    weight_list: list[str],
    batch_size: int,
):
    """
    Batched replacement for calculate_by's quantile path (_quantiles /
    _quantiles_actual). The original sorts by cumulative share (which is
    replicate-dependent) to interleave target-quantile "query rows" via a
    shift-based lookup. That second sort/shift/pivot isn't needed: the
    table is already sorted by the value column within each by-group, and
    cumulative share is monotonic non-decreasing in that same order for any
    non-negative weight, so the smallest value at or above a target share q
    is just `value.filter(share_r >= q).first()` (expressed here as a
    min-of-masked-value to satisfy polars' order-dependence rules) -
    verified as an exact numeric match against _quantiles_actual.

    column_stats here must only contain quantile/median stats. Returns
    {replicate_index: single-replicate output table} shaped like
    _quantiles_actual's output (by_cols + "{col}_{suffix}" columns).
    """
    df_polars = df
    ndf_base = df_polars.lazy()

    results = {}

    for coli, stats_list in column_stats.items():
        (_, modifier, coli_original) = _check_special_modifiers(coli)

        quantiles = []
        for stati in stats_list:
            resolved = stati
            if resolved == "median":
                resolved = "q50"
            q = float(resolved.replace("q", "").replace("p", "")) / 100
            quantiles.append((stati, q))

        c_keep_condition = pl.col(coli_original).is_not_null()
        if modifier == "not0":
            c_keep_condition = c_keep_condition & pl.col(coli_original).ne(0)

        sort_cols = by_cols + [coli_original]

        for batch_start in range(0, len(weight_list), batch_size):
            batch_weights = weight_list[batch_start : batch_start + batch_size]

            sum_exprs = [
                pl.col(weight_col).sum().alias(f"__wsum{offset}")
                for offset, weight_col in enumerate(batch_weights)
            ]
            grouped = (
                ndf_base.filter(c_keep_condition)
                .group_by(sort_cols)
                .agg(sum_exprs)
                .sort(sort_cols)
            )

            share_exprs = []
            for offset in range(len(batch_weights)):
                s = pl.col(f"__wsum{offset}")
                if len(by_cols):
                    share = (
                        s.cum_sum().over(by_cols, order_by=coli_original)
                        / s.sum().over(by_cols)
                    ).alias(f"__share{offset}")
                else:
                    share = (
                        s.cum_sum().over(order_by=coli_original) / s.sum()
                    ).alias(f"__share{offset}")
                share_exprs.append(share)
            grouped = grouped.with_columns(share_exprs)

            agg_exprs = []
            for offset, weight_col in enumerate(batch_weights):
                r = batch_start + offset
                share_col = pl.col(f"__share{offset}")
                for stati, q in quantiles:
                    suffix = stat_suffix(
                        statistic=("median" if stati == "median" else stati),
                        modifier=modifier,
                    )
                    masked = (
                        pl.when(share_col >= q)
                        .then(pl.col(coli_original))
                        .otherwise(None)
                        .min()
                    )
                    agg_exprs.append(
                        masked.alias(f"{coli_original}_{suffix}___rep{r}")
                    )

            if len(by_cols):
                wide = grouped.group_by(by_cols).agg(agg_exprs).collect()
            else:
                wide = grouped.select(agg_exprs).collect()

            for offset, weight_col in enumerate(batch_weights):
                r = batch_start + offset
                suffix_tag = f"___rep{r}"
                rep_cols = [c for c in wide.columns if c.endswith(suffix_tag)]
                rename = {c: c[: -len(suffix_tag)] for c in rep_cols}
                out = wide.select(by_cols + rep_cols).rename(rename)

                if r not in results:
                    results[r] = out
                else:
                    results[r] = join_wrapper(
                        results[r], out, on=by_cols, how="full"
                    ) if len(by_cols) else concat_wrapper(
                        [results[r], out], how="horizontal"
                    )

    return results


def _batched_quantiles_interpolated(
    df,
    column_stats: dict[str, list[str]],
    by_cols: list[str],
    weight_list: list[str],
    batch_size: int,
    interpolated_interval: int = 2500,
    drb_safe_n_bin: int = 10,
):
    """
    Batched replacement for calculate_by's Census-style interpolated
    quantile path (_quantiles / _quantiles_interpolated,
    quantile_interpolated=True).

    The bin-merge decision (which small bins get folded forward for
    disclosure safety) depends only on row counts, never on weight, so it's
    computed once via _quantile_interpolated_bin_keys and shared across
    every replicate in every batch - each batch only needs to (re)compute
    per-bin weight sums for its own replicates, then roll those up to the
    shared surviving bins via a join against the small bin-keys table
    (never against the full microdata a second time). From there the
    cumulative share and interpolation follow the same filter+min/max
    pattern already validated for _batched_quantiles, generalized to the
    two bracketing bins/shares the linear-interpolation formula needs.

    column_stats here must only contain quantile/median stats. Returns
    {replicate_index: single-replicate output table} shaped like
    _quantiles_interpolated's output (by_cols + "{col}_{suffix}" columns).
    """
    if not len(column_stats):
        return {}

    df_polars = df
    ndf_base = df_polars.lazy()

    results = {}

    for coli, stats_list in column_stats.items():
        (_, modifier, coli_original) = _check_special_modifiers(coli)

        quantiles = []
        for stati in stats_list:
            resolved = stati
            if resolved == "median":
                resolved = "q50"
            q = float(resolved.replace("q", "").replace("p", "")) / 100
            quantiles.append((stati, q))

        bin_col = f"__{coli}_bin"
        c_col = pl.col(coli_original)
        #   Truncating (not floor) division to match the exact binning
        #   behavior this always had - c_col (income) can be negative,
        #   where floor vs. truncating division differ.
        with_floor = 1 + (c_col / interpolated_interval).cast(pl.Int64)
        if modifier == "not0":
            binned_expr = pl.when(c_col != 0).then(with_floor).otherwise(pl.lit(None))
        else:
            binned_expr = with_floor

        sorted_bin = by_cols + [bin_col]
        df_binned_full = ndf_base.with_columns(binned_expr.alias(bin_col)).filter(
            pl.col(bin_col).is_not_null()
        )

        #   Replicate-independent: row counts per original bin determine
        #   the surviving (merged) bin keys, shared across every batch.
        df_counts = (
            df_binned_full.group_by(sorted_bin)
            .agg(pl.len().alias("___n_in_bin"))
            .sort(sorted_bin)
            .collect()
        )
        survivor_keys = _quantile_interpolated_bin_keys(
            df_binned=df_counts,
            sorted_coli=sorted_bin,
            by=by_cols,
            coli=bin_col,
            drb_safe_n_bin=drb_safe_n_bin,
        )

        for batch_start in range(0, len(weight_list), batch_size):
            batch_weights = weight_list[batch_start : batch_start + batch_size]

            sum_exprs = [
                pl.col(weight_col).sum().alias(f"__wsum{offset}")
                for offset, weight_col in enumerate(batch_weights)
            ]
            df_grouped = df_binned_full.group_by(sorted_bin).agg(sum_exprs).sort(
                sorted_bin
            )

            #   Cumulative share must be computed on the FULL per-original-
            #   bin table before rolling up to the merged survivor bins -
            #   a bin that gets merged away still contributed its weight to
            #   every cumulative value at or after it, and (as with the row
            #   counts in _quantile_interpolated_bin_keys) dropping it
            #   afterward doesn't change the cumulative value already
            #   recorded at surviving rows. Summing per bin and joining to
            #   survivor keys FIRST, then cumsum-ing only the survivors,
            #   would silently discard every merged-away bin's weight.
            share_exprs = []
            for offset in range(len(batch_weights)):
                s = pl.col(f"__wsum{offset}")
                if len(by_cols):
                    share = (
                        s.cum_sum().over(by_cols, order_by=bin_col)
                        / s.sum().over(by_cols)
                    ).alias(f"__share{offset}")
                else:
                    share = (
                        s.cum_sum().over(order_by=bin_col) / s.sum()
                    ).alias(f"__share{offset}")
                share_exprs.append(share)
            df_grouped = df_grouped.with_columns(share_exprs).collect()

            #   Roll the already-cumulative shares up to the shared
            #   surviving bins - a join against the small bin-keys table,
            #   never against the full microdata.
            share_cols = [f"__share{offset}" for offset in range(len(batch_weights))]
            df_coli_batch = survivor_keys.join(
                df_grouped.select(sorted_bin + share_cols),
                on=sorted_bin,
                how="left",
            ).with_columns(
                [pl.col(c).fill_null(pl.lit(0.0)) for c in share_cols]
            )
            ndf_batch = df_coli_batch.lazy()

            agg_exprs = []
            for offset, weight_col in enumerate(batch_weights):
                r = batch_start + offset
                share_col = pl.col(f"__share{offset}")
                bin_col_expr = pl.col(bin_col)
                for stati, q in quantiles:
                    suffix = stat_suffix(
                        statistic=("median" if stati == "median" else stati),
                        modifier=modifier,
                    )
                    w_above = (
                        pl.when(share_col >= q).then(share_col).otherwise(None).min()
                    )
                    y_above = (
                        pl.when(share_col >= q)
                        .then(bin_col_expr)
                        .otherwise(None)
                        .min()
                    )
                    w_below = (
                        pl.when(share_col < q).then(share_col).otherwise(None).max()
                    )
                    y_below = (
                        pl.when(share_col < q)
                        .then(bin_col_expr)
                        .otherwise(None)
                        .max()
                    )

                    w_gap = pl.lit(q) - w_below
                    y_interval = y_above - y_below
                    w_interval = w_above - w_below
                    val = (
                        y_below + (w_gap / w_interval) * y_interval
                    ) * interpolated_interval

                    agg_exprs.append(
                        val.alias(f"{coli_original}_{suffix}___rep{r}")
                    )

            if len(by_cols):
                wide = ndf_batch.group_by(by_cols).agg(agg_exprs).collect()
            else:
                wide = ndf_batch.select(agg_exprs).collect()

            for offset, weight_col in enumerate(batch_weights):
                r = batch_start + offset
                suffix_tag = f"___rep{r}"
                rep_cols = [c for c in wide.columns if c.endswith(suffix_tag)]
                rename = {c: c[: -len(suffix_tag)] for c in rep_cols}
                out = wide.select(by_cols + rep_cols).rename(rename)

                if r not in results:
                    results[r] = out
                else:
                    results[r] = join_wrapper(
                        results[r], out, on=by_cols, how="full"
                    ) if len(by_cols) else concat_wrapper(
                        [results[r], out], how="horizontal"
                    )

    return results


##########################################################
##########################################################
#   Batched replicate-weight computation - END
##########################################################
##########################################################


