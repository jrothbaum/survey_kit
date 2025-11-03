from __future__ import annotations
from typing import Callable


import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT

from ..utilities.inputs import list_input

from ..utilities.dataframe import (
    _columns_original_order,
    safe_sum_cast,
    join_wrapper,
    concat_wrapper,
    fill_missing,
    columns_from_list,
    NarwhalsType,
    backend_eager,
)

from ..utilities.compress import compress_df
from .. import logger


def calculate_by(
    df: IntoFrameT,
    column_stats: dict[str, list[str]],
    by: dict[str, list[str] | str] | list[list[str] | str] | None = None,
    weight: str = "",
    no_suffix: bool = False,
    quantile_interpolated: bool = False,
    quantile_interpolated_interval: int | float = 2_500,
    always_return_as_collection: bool = False,
    allow_slow_pandas: bool = False,
) -> dict[str, IntoFrameT] | list[IntoFrameT] | IntoFrameT:
    if by is None:
        by = []

    nw_type = NarwhalsType(df)
    nw_type_original = nw_type

    #   Convert pandas->polars, run calculations, -> pandas
    #       because the group-by aggregations are problematic for pandas
    if nw_type.backend == "pandas" and not allow_slow_pandas:
        #   logger.info("pandas->polars for calculation")
        df = nw_type.to_polars()
        nw_type = NarwhalsType(df)

    df = nw.from_native(df).lazy_backend(nw_type)

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
    df = df.select(columns_to_keep).with_columns(cs.boolean().cast(nw.Int8))

    df_out = {}
    if weight != "":
        df = nw.from_native(
            safe_sum_cast(df=df.to_native(), columns=weight)
        ).lazy_backend(nw_type)

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
                df = nw.from_native(
                    safe_sum_cast(df=df.to_native(), columns=col_stat_info.column_name)
                ).lazy_backend(nw_type)

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
                    (nw.col(namei) / nw.lit(d_shares[namei])).alias(namei)
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
                ).lazy_backend(nw_type)

        df_out[bynamei] = compress_df(
            nw.from_native(fill_missing(df_byi, value=None))
            .lazy_backend(nw_type)
            .collect()
            .to_native(),
            no_boolean=True,
        )

        if nw_type_original.backend == "pandas" and not allow_slow_pandas:
            # logger.info("  return from polars->pandas")
            df_out[bynamei] = nw_type_original.from_polars(df_out[bynamei])

    if no_suffix:
        for bynamei, byi in by_for_loop.items():
            for coli in list(column_stats.keys()):
                cols_thesevars = columns_from_list(df_out[bynamei], f"{coli}_*")

                if len(cols_thesevars) == 1:
                    df_out[bynamei] = (
                        nw.from_native(df_out[bynamei])
                        .rename({cols_thesevars[0]: coli})
                        .to_native()
                    )
    if len(df_out) == 1 and not always_return_as_collection:
        return next(iter(df_out.values()))
    else:
        if type(by) is list:
            return list(df_out.values())
        else:
            return df_out


def _check_special_modifiers(column: str) -> tuple[str, str]:
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
        stat_expr: nw.Expr,
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
        statout = nw.col(column).first()

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
) -> nw.Expr:
    c_filter = None
    if modifier == "not0":
        c_filter = nw.col(column) != 0
    elif modifier == "notmissing":
        c_filter = ~nw.col(column).is_null()
    elif modifier == "missing":
        c_filter = nw.col(column).is_null()
    elif modifier == "is0":
        c_filter = nw.col(column) == 0

    if weight != "":
        c_weight = (nw.col(weight) != 0) & ~(nw.col(weight).is_null())

        if c_filter is None:
            c_filter = c_weight
        else:
            c_filter = c_filter & c_weight

    if c_filter is not None:
        c_filter = c_filter.cast(nw.Int8)

    return c_filter


def _mean(
    column: str, c_filter: nw.Expr | None, weight: str = "", suffix: str = ""
) -> nw.Expr:
    c_col = nw.col(column)
    if weight != "":
        c_weight = nw.col(weight)
        statout = (c_filter * c_col * c_weight).sum() / (
            c_filter * (~c_col.is_null()).cast(nw.Int8) * c_weight
        ).sum()
    else:
        if c_filter is not None:
            statout = (c_col * c_filter).sum() / c_filter.sum()
        else:
            statout = (c_col).mean()

    return statout


def _sum(
    column: str, c_filter: nw.Expr | None, weight: str = "", suffix: str = ""
) -> nw.Expr:
    c_col = nw.col(column)
    if weight != "":
        c_weight = nw.col(weight)
        statout = (c_filter * c_col * c_weight).sum()
    else:
        if c_filter is not None:
            statout = (c_col * c_filter).sum()
        else:
            statout = (c_col).sum()

    return statout


def _count(
    column: str, c_filter: nw.Expr | None, weight: str = "", suffix: str = ""
) -> nw.Expr:
    if weight != "":
        c_weight = nw.col(weight)
        statout = (c_filter * c_weight).sum()
    else:
        if c_filter is not None:
            statout = c_filter.sum()
        else:
            statout = nw.len()

    return statout


def _rawcount(
    column: str, c_filter: nw.Expr | None, weight: str = "", suffix: str = ""
) -> nw.Expr:
    return _count(column=column, c_filter=c_filter, suffix=suffix)


def _share(
    column: str, c_filter: nw.Expr | None, weight: str = "", suffix: str = ""
) -> nw.Expr:
    if weight != "":
        c_weight = nw.col(weight)
        statout = (c_filter * c_weight).sum() / c_weight.sum()
    else:
        if c_filter is not None:
            statout = c_filter.sum() / nw.len()
        else:
            statout = nw.len()

    return statout


def _rawshare(
    column: str, c_filter: nw.Expr | None, weight: str = "", suffix: str = ""
) -> nw.Expr:
    return _share(column=column, c_filter=c_filter, suffix=suffix)


def _var(
    column: str, c_filter: nw.Expr | None, weight: str = "", suffix: str = ""
) -> nw.Expr:
    c_col = nw.col(column)
    if weight != "":
        c_weight = nw.col(weight)
        c_mean = _mean(column=column, c_filter=c_filter, weight=weight, suffix=suffix)
        c_n = _rawcount(column=column, c_filter=c_filter, weight=weight, suffix=suffix)
        num = (c_weight * c_filter * ((c_col - c_mean) ** 2)).sum()
        denom = ((c_n - 1) / c_n) * (c_filter * c_weight).sum()

        statout = num / denom
    else:
        if c_filter is not None:
            statout = (
                nw.when(c_filter.cast(nw.Boolean)).then(c_col).otherwise(nw.lit(None))
            ).var()
        else:
            statout = c_col.var()

    return statout


def _std(
    column: str, c_filter: nw.Expr | None, weight: str = "", suffix: str = ""
) -> nw.Expr:
    statout = (
        _var(column=column, c_filter=c_filter, weight=weight, suffix=suffix) ** 0.5
    )

    return statout


def _max(
    column: str, c_filter: nw.Expr | None, weight: str = "", suffix: str = ""
) -> nw.Expr:
    c_col = nw.col(column)
    if c_filter is not None:
        statout = (
            nw.when(c_filter.cast(nw.Boolean)).then(c_col).otherwise(nw.lit(None))
        ).max()
    else:
        statout = c_col.max()

    return statout


def _min(
    column: str, c_filter: nw.Expr | None, weight: str = "", suffix: str = ""
) -> nw.Expr:
    c_col = nw.col(column)
    if c_filter is not None:
        statout = (
            nw.when(c_filter.cast(nw.Boolean)).then(c_col).otherwise(nw.lit(None))
        ).min()
    else:
        statout = c_col.min()

    return statout


def _quantiles(
    df: nw.LazyFrame,
    column_stats: dict[str, list[str]],
    by: str | list | None = None,
    weight: str = "",
    interpolated: bool = True,
    interpolated_interval: int = 2500,
):
    if weight != "":
        df = df.with_columns((nw.col(weight) / nw.col(weight).sum()).alias(weight))

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
    df: nw.LazyFrame,
    column_stats: dict[str, list[str]],
    by: str | list | None = None,
    weight: str = "",
) -> nw.LazyFrame:
    nw_type = NarwhalsType(df)
    if type(by) is str:
        by = [by]
    if by is None or not len(by):
        by = []

    columns = list(column_stats.keys())
    if weight == "":
        n = df.select(nw.len()).collect().item(0, 0)
        weight = "___weight___"
        df = df.with_columns(nw.lit(1 / n).alias(weight))

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

        keep_condition = ~nw.col(coli).is_null()
        if modifier == "not0":
            keep_condition = keep_condition & (nw.col(coli) != 0)
        else:
            modifier = ""

        if len(by):
            calc_over = (
                nw.col(weight).cum_sum().over(by, order_by=coli)
                / nw.col(weight).sum().over(by)
            ).alias(weight)
        else:
            calc_over = (
                nw.col(weight).cum_sum().over(order_by=[coli]) / nw.col(weight).sum()
            ).alias(weight)

        df_coli = (
            df.filter(keep_condition)
            .group_by(sorted_coli)
            .agg(nw.col(weight).sum())
            .sort(sorted_coli)
            .with_columns(calc_over)
        )

        if df_by is None:
            if len(by):
                df_by = df_coli.select(by).unique()

        #   Create rows with the quantiles we want to calculate
        if df_by is None:
            df_byi = nw.from_dict(
                {weight: qlisti}, backend=backend_eager(nw_type.backend)
            ).lazy_backend(nw_type)
        else:
            df_q = nw.from_dict(
                {weight: qlisti}, backend=backend_eager(nw_type.backend)
            ).lazy_backend(nw_type)

            df_byi = join_wrapper(
                df_by.with_columns(nw.lit(1).alias("___stats_join_full___")),
                df_q.with_columns(nw.lit(1).alias("___stats_join_full___")),
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
                nw.col(coli)
                .shift(-1)
                .over(order_by=by + [weight])
                .alias("___y_above___")
            )
            .filter(nw.col(coli).is_null())
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
            df_coli = df_coli.with_columns(nw.lit(1).alias(by[0]))

            temp_by = True

        if df_by is None:
            df_coli = df_coli.with_columns(
                nw.col(coli).fill_null(strategy="backward").over(order_by=by + [weight])
            )
        else:
            df_coli = df_coli.with_columns(
                nw.col(coli)
                .fill_null(strategy="backward")
                .over(by, order_by=by + [weight])
            )

        #   No pivot in pyarrow
        if backend_eager(nw_type.backend) in ["pyarrow", "pandas"]:
            df_coli = nw.from_native(df_coli.collect().to_polars().lazy())
        df_coli = df_coli.collect().pivot(index=by, on=weight, values=coli)
        if backend_eager(nw_type.backend) == "pyarrow":
            df_coli = nw.from_native(df_coli.to_arrow())

        df_coli = df_coli.rename(rename).sort(by)

        if nw_type.backend == "pandas":
            df_coli = nw.from_native(df_coli.to_pandas())

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

    return df_out.lazy_backend(nw_type)


def _quantiles_interpolated(
    df: nw.LazyFrame,
    column_stats: dict[str, list[str]],
    by: str | list | None = None,
    weight: str = "",
    interpolated_interval: int = 2500,
    drb_safe_n_bin: int = 10,
):
    nw_type = NarwhalsType(df)
    if type(df) is nw.DataFrame:
        df = df.lazy_backend(nw_type)

    if by is None or not len(by):
        by = []

    columns = list(column_stats.keys())
    if weight == "":
        n = df.select(nw.len()).collect().item(0, 0)
        weight = "___weight___"
        df = df.with_columns(nw.lit(1 / n).alias(weight))

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

        c_col = nw.col(col_name)

        with_floor = 1 + (c_col.floordiv(interpolated_interval))
        if modifier == "not0":
            c_cleared = nw.when(c_col != 0).then(with_floor).otherwise(nw.lit(None))
            with_intervals.append(c_cleared.alias(coli))
        else:
            with_intervals.append(with_floor.alias(col_name))

        with_intervals.append(nw.col(col_name).alias(f"__{coli}_original_value"))

    df = df.select(keep_list).with_columns(with_intervals)

    df_out = None
    df_by = None

    for coli, qlisti in column_stats.items():
        col_original_value = f"__{coli}_original_value"
        col_min = f"__{coli}_original_value_min"
        col_max = f"__{coli}_original_value_max"
        (coli_nameonly, modifier, coli_original) = _check_special_modifiers(coli)

        sorted_coli = by + [coli]

        #   keep_condition = ~(nw.col(coli).is_null() | nw.col(coli).is_nan()) & ~(nw.col(weight).is_null() | nw.col(weight).is_nan())
        keep_condition = ~(nw.col(coli).is_null()) & ~(
            nw.col(weight).is_null() | nw.col(weight).is_nan()
        )

        if len(by):
            by_over = nw.col(weight) / nw.col(weight).sum().over(by)
            calc_over = (
                nw.col(weight).cum_sum().over(by, order_by=coli)
                / nw.col(weight).sum().over(by)
            ).alias(weight)
        else:
            by_over = nw.col(weight) / nw.col(weight).sum()
            calc_over = (
                nw.col(weight).cum_sum().over(order_by=coli) / nw.col(weight).sum()
            ).alias(weight)

        var_n_in_bin = "___n_in_bin"
        var_n_cum_sum = "___n_to_bin"
        df_coli = (
            df.filter(keep_condition)
            .with_columns(by_over)
            .with_columns(cs.by_dtype(nw.Int8).cast(nw.Int32))
            .group_by(sorted_coli)
            .agg(
                [
                    nw.col(weight).sum(),
                    nw.len().alias(var_n_in_bin),
                    nw.col(col_original_value).min().alias(col_min),
                    nw.col(col_original_value).max().alias(col_max),
                ]
            )
            .sort(sorted_coli)
            .with_columns([calc_over])
            .collect()
        )

        #   Check if need to insert a "fake" bottom category as the current bottom has more mass than the min quantile
        df_first = df_coli.head(1)
        if (df_first[col_min].item() == df_first[col_max].item()) and (
            df_first[var_n_in_bin].item() > drb_safe_n_bin
        ):
            df_first = df_first.with_columns(nw.lit(0).alias(weight))
        else:
            #   Create a "first" bin that is at the lower bound of the real first bin
            df_first = df_first.with_columns(
                [
                    nw.col(coli) - 1,
                    nw.lit(drb_safe_n_bin).cast(nw.UInt32).alias(var_n_in_bin),
                    nw.lit(0.0).alias(weight),
                ]
            )
        if drb_safe_n_bin:
            #   logger.info(f"Checking n in bin >= {drb_safe_n_bin}")

            smallest_bin = df_coli.select(nw.min(var_n_in_bin))[0, 0]
            #   logger.info(df_coli.height)

            while smallest_bin < drb_safe_n_bin:
                c_too_small = nw.col(var_n_in_bin) < drb_safe_n_bin
                c_prior_too_small = (
                    nw.col(var_n_in_bin)
                    .shift(n=1)
                    .fill_null(nw.lit(drb_safe_n_bin + 1))
                    < drb_safe_n_bin
                )

                df_coli = (
                    df_coli.with_columns(
                        [
                            nw.col(var_n_in_bin)
                            .cum_sum()
                            .over(order_by=sorted_coli)
                            .alias(var_n_cum_sum)
                        ]
                    )
                    .filter(~(c_too_small & ~c_prior_too_small))
                    .with_columns(
                        [
                            (
                                nw.col(var_n_cum_sum)
                                - nw.col(var_n_cum_sum).shift(n=1).fill_null(nw.lit(0))
                            ).alias(var_n_in_bin),
                        ]
                    )
                    .drop(var_n_cum_sum)
                )

                smallest_bin = df_coli.select(nw.min(var_n_in_bin))[0, 0]

        df_coli = concat_wrapper([df_first, df_coli], how="diagonal")
        if df_by is None:
            if len(by):
                df_by = df_coli.select(by).unique()

        #   Create rows with the quantiles we want to calculate
        if df_by is None:
            backend = df.__native_namespace__()
            df_byi = nw.from_dict(
                {weight: qlisti}, backend=backend_eager(nw_type.backend)
            ).lazy_backend(nw_type)
        else:
            df_q = nw.from_dict(
                {weight: qlisti}, backend=backend_eager(nw_type.backend)
            ).lazy_backend(nw_type)

            df_byi = join_wrapper(
                df_by.with_columns(nw.lit(1).alias("___stats_join_full___")),
                df_q.with_columns(nw.lit(1).alias("___stats_join_full___")),
                how="full",
                on=["___stats_join_full___"],
            ).drop("___stats_join_full___")

        #   Column references for convenience
        list_w_below = []
        list_w_above = []
        list_y_below = []
        list_y_above = []

        with_shift = []
        c_w = nw.col(weight)
        c_var = nw.col(coli)
        for shifti in range(1, len(qlisti) + 1):
            with_shift.extend(
                [
                    (
                        nw.when(c_var.shift(shifti).is_not_missing())
                        .then(c_w.shift(shifti))
                        .otherwise(nw.lit(None))
                        .alias(f"__w_below___{shifti}")
                    ),
                    (
                        nw.when(c_var.shift(-shifti).is_not_missing())
                        .then(c_w.shift(-shifti))
                        .otherwise(nw.lit(None))
                        .alias(f"__w_above___{shifti}")
                    ),
                    c_var.shift(shifti).alias(f"__y_below___{shifti}"),
                    c_var.shift(-shifti).alias(f"__y_above___{shifti}"),
                ]
            )

            list_w_below.append(f"__w_below___{shifti}")
            list_w_above.append(f"__w_above___{shifti}")
            list_y_below.append(f"__y_below___{shifti}")
            list_y_above.append(f"__y_above___{shifti}")

        w_gap = nw.col(weight) - nw.col("__w_below")
        y_below = nw.col("__y_below")
        y_above = nw.col("__y_above")
        y_interval = y_above - y_below

        w_below = nw.col("__w_below")
        w_above = nw.col("__w_above")
        w_interval = w_above - w_below

        val_interpolated = (
            y_below + (w_gap / w_interval) * y_interval
        ) * interpolated_interval

        df_coli = (
            concat_wrapper(
                [df_coli.lazy_backend(nw_type), df_byi.lazy_backend(nw_type)],
                how="diagonal",
            )
            .sort(by + [weight])
            .collect()
            .with_columns(with_shift)
            .lazy_backend(nw_type)
            .filter(c_var.is_missing())
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
                    nw.coalesce(list_w_below).alias("__w_below"),
                    nw.coalesce(list_y_below).alias("__y_below"),
                    nw.coalesce(list_w_above).alias("__w_above"),
                    nw.coalesce(list_y_above).alias("__y_above"),
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
        #                     [nw.col(weight).shift(1).alias("___w_below___"),
        #                      nw.col(weight).shift(-1).alias("___w_above___"),
        #                      nw.col(coli).shift(1).alias("___y_below___")]
        #                 )\
        #             .filter(nw.col(coli).is_missing())\
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
            df_coli = df_coli.with_columns(nw.lit(1).alias(by[0]))

            temp_by = True

        #   No pivot in pyarrow
        if backend_eager(nw_type.backend) in ["pyarrow", "pandas"]:
            df_coli = nw.from_native(df_coli.collect().to_polars().lazy())
        df_coli = df_coli.collect().pivot(index=by, on=weight, values=coli)
        if backend_eager(nw_type.backend) == "pyarrow":
            df_coli = nw.from_native(df_coli.to_arrow())

        df_coli = df_coli.lazy_backend(nw_type)

        for coli in df_coli.collect_schema().names():
            rename_to_str = {}
            if type(coli) is not str:
                rename_to_str[coli] = str(coli)

            if len(rename_to_str):
                df_coli = df_coli.rename(rename_to_str)

        df_coli = df_coli.rename(rename).sort(by)

        if nw_type.backend == "pandas":
            df_coli = nw.from_native(df_coli.to_pandas())

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
                ).lazy_backend(nw_type)

    return df_out


def _custom_stat_by(
    df: nw.LazyFrame,
    stat_name: str,
    delegate: Callable,
    column_stats: dict[str, list[str]],
    by: str | list | None = None,
    weight: str = "",
    **kwargs,
):
    nw_type = NarwhalsType(df)
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
                condi = nw.col(coli) == rowi[j]

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

        keep_condition = nw.col(coli).is_not_missing()
        if modifier == "not0":
            keep_condition = keep_condition & nw.col(coli).ne(0)
        else:
            modifier = ""

        df_outi = None
        for filteri in by_filters:
            if filteri is None:
                df_byi = df
            else:
                df_byi = df.filter(filteri)

            df_coli = delegate(df=df_byi, weight=weight, variable=coli, **kwargs)

            rename_to = f"{coli_final}_{stat_name}"
            if modifier != "":
                rename_to += f"_{modifier}"
            df_coli = df_coli.rename({stat_name: rename_to})

            if len(by):
                df_coli = concat_wrapper(
                    [
                        nw.maybe_reset_index(df_byi.select(by).head(1).collect()),
                        nw.maybe_reset_index(df_coli.collect()),
                    ],
                    how="horizontal",
                ).lazy_backend(nw_type)

            if df_outi is None:
                df_outi = df_coli
            else:
                df_outi = concat_wrapper([df_outi, df_coli], how="diagonal")

        if df_out is None:
            df_out = df_outi
        else:
            if len(by):
                df_out = join_wrapper(df_out, df_outi, how="full", by=by)
            else:
                df_out = concat_wrapper([df_out, df_outi], how="horizontal")

    return df_out


def _gini(
    df: nw.LazyFrame | nw.DataFrame,
    variable: str,
    weight: str = "",
    censor_at_zero: bool = True,
) -> nw.LazyFrame | nw.DataFrame:
    nw_type = NarwhalsType(df)
    if weight == "":
        weight = "__gini_weight__"
        df = df.select(variable).with_columns(nw.lit(1).alias(weight))

    else:
        df = df.select([variable, weight])

    c_weight = nw.col(weight)
    df = df.with_columns((c_weight / c_weight.sum()).alias(weight))

    schema = df.collect_schema()
    if censor_at_zero:
        c_income = nw.col(variable) * nw.col(variable).gt(0).cast(schema[variable])
    else:
        c_income = nw.col(variable)

    df = df.sort(variable).filter(c_weight.is_not_missing())

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
    #   swt = nw.col(weight).fill_null(0).cum_sum().alias("swt")
    swt = nw.col(weight).cum_sum().over(order_by=variable).alias("swt")
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

    df_out = nw.from_dict(
        {"gini": [gini]}, backend=backend_eager(nw_type.backend)
    ).lazy_backend(nw_type)
    # df_out = nw.DataFrame({"gini":[gini]})

    return df_out


# def _shares(df:nw.LazyFrame | nw.DataFrame,
#             income:str,
#             shares:list[tuple(float,float)],
#             weight:str="",
#             censor_at_zero:bool=True) -> nw.LazyFrame | nw.DataFrame:

#     from NEWS.CodeUtilities.Python.Recodes import ColumnSumToValue
#     from NEWS.CodeUtilities.Python.Statistics.StatCalculator import StatCalculator

#     return_lazy = type(df) is nw.LazyFrame
#     if weight == "":
#         weight = "__shares_weight__"
#         df = (df.select(income)
#                 .with_columns(nw.lit(weight)))

#     else:
#         df = df.select([income,
#                         weight])

#     c_weight = nw.col(weight)


#     df = ColumnSumToValue(df=df,
#                           Columns=[weight],
#                           to_value=1)
#     if censor_at_zero:
#         c_income = nw.col(income)*nw.col(income).gt(0).cast(df.schema[income])
#     else:
#         c_income = nw.col(income)

#     df = df.sort(income)

#     #   Get a list of all the unique bounds
#     all_share_bounds = []
#     for sharei in shares:
#         all_share_bounds.append(sharei[0])
#         all_share_bounds.append(sharei[1])


#     all_share_bounds = list(set(all_share_bounds))
#     all_share_bounds.sort()

#     c_ptile = nw.cum_sum(weight)
#     df = df.with_columns([c_ptile,
#                           c_income
#                           #,nw.col(weight).alias(f"{weight}_original")])
#                           ])


#     #   Add the bounds to the dataset with income missing
#     c_income = nw.col(income)
#     c_weight = nw.col(weight)
#     df_bounds = nw.DataFrame({weight:all_share_bounds},
#                              schema={weight:nw.Float64})
#     df = AppendList([df,
#                      df_bounds.filter(c_weight.gt(0) & c_weight.lt(1))],
#                     quietly=True).sort(weight)

#     #   Fill income from above and then re-calculate the weights
#     #       This will create an observation that splits the weight at the bound
#     #       to properly get the share of income within each bound
#     df = (df.with_columns([c_income.fill_null(strategy="backward").alias(f"{income}_above")])
#             .with_columns([(nw.when(c_income.is_missing()).then(f"{income}_above")
#                               .otherwise(c_income).alias(income)),
#                            c_weight.alias(f"{weight}_cum"),
#                            (c_weight - c_weight.shift(n=1,
#                                                       fill_value=0)).alias(weight)])
#             .drop(f"{income}_above"))


#     #   Calculate the total income
#     statistics = Statistics(stats=["sum"],
#                             columns=[income])
#     sc = StatCalculator(df=df,
#                         statistics=statistics,
#                         weight=weight,
#                         display=False,
#                         round_output=False)
#     y_full = sc.df_estimates.drop("Variable")[0,0]


#     #   Calculate the income in each bound
#     c_weight_cum = nw.col(f"{weight}_cum")
#     share_values = {}
#     for sharei in shares:

#         c_filter = None
#         if sharei[0] > 0:
#             c_filter = c_weight_cum.gt(sharei[0])
#         if sharei[1] < 1:
#             if c_filter is None:
#                 c_filter = c_weight_cum.le(sharei[1])
#             else:
#                 c_filter = c_filter & c_weight_cum.le(sharei[1])

#         sc = StatCalculator(df=df.filter(c_filter),
#                             statistics=statistics,
#                             weight=weight,
#                             display=False,
#                             round_output=False)
#         yi = sc.df_estimates.drop("Variable")[0,0]
#         #   logger.info(yi)

#         share_values[f"share_{str(sharei[0]*100).replace('.','_')}__{str(sharei[1]*100).replace('.','_')}"] = yi/y_full

#     #   Assign the shares to a dataframe and return
#     df_out = nw.DataFrame(share_values)

#     if return_lazy:
#         df_out = df_out.lazy_backend(nw_type)

#     return df_out
