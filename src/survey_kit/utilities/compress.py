from __future__ import annotations
from narwhals.typing import IntoFrameT

import polars as pl
from .dataframe import NarwhalsType, safe_height

from .. import logger


def compress_df(
    df: IntoFrameT,
    cols: list[str] | str | None = None,
    check_string: bool = False,
    check_string_only: bool = False,
    cast_all_null_to_int8: bool = True,
    check_date_time: bool = True,
    no_boolean: bool = False,
) -> IntoFrameT:
    """
    Optimize DataFrame by downcasting numeric types to smallest possible representation.

    Analyzes numeric columns and casts them to the smallest data type that can
    accommodate all values, reducing memory usage and file sizes.
    Has some optional parameters to handle figuring out the optimal compress
    for stata

    Parameters
    ----------
    df : IntoFrameT
        Input data
    cols : list[str], optional
        Specific columns to compress (default: all)
    check_string : bool
        Attempt to convert string columns to numeric
    check_string_only : bool
        Only check string conversions
    cast_all_null_to_int8 : bool
        Cast all-null columns to Int8 (rather than leaving them at a wider
        default type). Int8 rather than Boolean since polars doesn't
        auto-promote Boolean to numeric on relaxed appends/concats.
    check_date_time : bool
        Optimize datetime columns
    no_boolean : bool
        Skip boolean type casting (and leave as int8)

    Returns
    -------
    IntoFrameT
        Compressed DataFrame with optimized data types

    Examples
    --------
    Basic compression:

    >>> compressed_df = compress_df(df)

    String to numeric conversion:

    >>> compressed_df = compress_df(df, check_string=True)

    Notes
    -----
    Automatically detects the smallest integer type that can hold all values
    in each column, considering ranges like Int8 (-128 to 127), Int16, etc.
    """

    nw_type = NarwhalsType(df)
    df = nw_type.to_polars()

    if check_date_time:
        df = _compress_datetime(df)

    #   Convert numerics
    intlist = {}
    if not no_boolean:
        intlist[pl.Boolean] = [0, 1]
    intlist[pl.Int8] = [-(2**7), 2**7 - 1]
    intlist[pl.Int16] = [-(2**15), 2**15 - 1]
    intlist[pl.Int32] = [-(2**31), 2**31 - 1]
    intlist[pl.Int64] = [-(2**63), 2**63 - 1]

    schema = df.lazy().collect_schema()

    if cols is None:
        cols = schema.names()

    #   String -> numeric conversion, batched across every eligible column:
    #   a non-strict cast plus a before/after null-count comparison gives the
    #   same all-or-nothing per-column semantics as the original's per-column
    #   try/except(strict=True), without a separate collect() per column.
    if check_string:
        string_cols = [
            coli for coli in cols if schema[coli] in (pl.Utf8, pl.String)
        ]
        if len(string_cols):
            checks = (
                df.lazy()
                .select(
                    [pl.col(coli).is_null().sum().alias(f"{coli}___before") for coli in string_cols]
                    + [
                        pl.col(coli)
                        .str.strip_chars()
                        .cast(pl.Float64, strict=False)
                        .is_null()
                        .sum()
                        .alias(f"{coli}___after")
                        for coli in string_cols
                    ]
                )
                .collect()
            )

            numeric_string_cols = [
                coli
                for coli in string_cols
                if checks[0, f"{coli}___after"] == checks[0, f"{coli}___before"]
            ]

            if len(numeric_string_cols):
                df = df.with_columns(
                    [
                        pl.col(coli).str.strip_chars().cast(pl.Float64)
                        for coli in numeric_string_cols
                    ]
                )
                schema = df.lazy().collect_schema()

    if check_string_only:
        df = nw_type.from_polars(df)
        return NarwhalsType.return_df(df, nw_type)

    numeric_intsize = {
        pl.Float64: 65,
        pl.Float32: 65,
        pl.Int64: 64,
        pl.UInt64: 64,
        pl.Int32: 32,
        pl.UInt32: 32,
        pl.Int16: 16,
        pl.UInt16: 16,
        pl.Int8: 8,
        pl.UInt8: 8,
    }

    eligible = [coli for coli in cols if schema[coli] in numeric_intsize]

    if len(eligible):
        #   Gather every column's n_notnull/min/max (and, for float columns,
        #   whether every non-null value is integer-valued) in ONE pass -
        #   the original did this per column with its own collect() each.
        stat_exprs = [pl.len().alias("___height")]
        for coli in eligible:
            stat_exprs.append(pl.col(coli).is_not_null().sum().alias(f"{coli}___n"))
            stat_exprs.append(pl.col(coli).min().alias(f"{coli}___min"))
            stat_exprs.append(pl.col(coli).max().alias(f"{coli}___max"))
            if schema[coli] in (pl.Float32, pl.Float64):
                stat_exprs.append(
                    (pl.col(coli).drop_nulls().mod(1) == 0)
                    .all()
                    .alias(f"{coli}___allint")
                )

        stats = df.lazy().select(stat_exprs).collect()
        height = stats[0, "___height"]

        casts = {}
        for coli in eligible:
            plType = schema[coli]
            plType_intsize = numeric_intsize[plType]
            n_notnull = stats[0, f"{coli}___n"]

            if n_notnull == 0:
                if height != 0 and cast_all_null_to_int8:
                    casts[coli] = pl.Int8
                continue

            if plType in (pl.Float32, pl.Float64) and not stats[0, f"{coli}___allint"]:
                #   check_float32 is unreachable in the original (always
                #   False), so non-integer-valued floats are left as-is here
                #   too - preserves existing behavior rather than changing it.
                continue

            minValue = stats[0, f"{coli}___min"]
            maxValue = stats[0, f"{coli}___max"]

            for inti, (lowerbound, upperbound) in intlist.items():
                intSize = 1 if inti == pl.Boolean else int(str(inti).replace("Int", ""))
                if (
                    plType_intsize > intSize
                    and maxValue <= upperbound
                    and minValue >= lowerbound
                ):
                    casts[coli] = inti
                    break

        if len(casts):
            try:
                df = df.with_columns(
                    [
                        pl.col(coli).cast(target, strict=True).alias(coli)
                        for coli, target in casts.items()
                    ]
                )
            except Exception:
                #   Fall back to casting one column at a time so a single
                #   unexpected failure (e.g. an edge case the min/max check
                #   didn't catch) only drops that column, matching the
                #   original's per-column try/except behavior.
                for coli, target in casts.items():
                    try:
                        df = df.with_columns(
                            pl.col(coli).cast(target, strict=True).alias(coli)
                        )
                    except Exception:
                        logger.warning(
                            "     Cannot cast " + coli + " as " + str(target)
                        )

    df = nw_type.from_polars(df)

    return NarwhalsType.return_df(df, nw_type)


def _compress_datetime(df: pl.LazyFrame | pl.LazyFrame) -> pl.LazyFrame | pl.DataFrame:
    schema = df.lazy().collect_schema()
    cols_date = {
        coli: typei for coli, typei in schema.items() if type(typei) is pl.Datetime
    }

    for coli, typei in cols_date.items():
        cast_complete = False
        dfcast = None

        c_d = pl.col(coli)
        df_time = df.filter(
            (
                c_d.dt.nanosecond()
                + c_d.dt.microsecond()
                + c_d.dt.millisecond()
                + c_d.dt.second()
                + c_d.dt.minute()
                + c_d.dt.hour()
            ).ne(0)
        )
        convert_to_date = safe_height(df_time) == 0

        if convert_to_date:
            try:
                dfcast = df.select(c_d.cast(pl.Date, strict=True))
                cast_complete = True
            except:
                logger.warning(f"     Cannot cast {coli} as {pl.Date}")

            if cast_complete:
                df = pl.concat([df.drop(coli), dfcast], how="horizontal")

    return df
