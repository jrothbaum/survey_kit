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
    cast_all_null_to_boolean: bool = True,
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
    cast_all_null_to_boolean : bool
        Cast all-null columns to boolean
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

    if cols is None:
        cols = df.lazy().collect_schema().names()

    for columni in cols:
        cast_complete = False
        plType = df.lazy().collect_schema()[columni]

        df_col = df.select(pl.col(columni))

        check_integers = False
        check_float32 = False

        maxValue = None
        minValue = None

        if check_string and (plType == pl.Utf8 or plType == pl.String):
            numeric_string = False
            try:
                df_col = (
                    df_col.select(
                        pl.col(columni).str.strip_chars().cast(pl.Float64, strict=True)
                    )
                    .lazy()
                    .collect()
                )
                numeric_string = True
            except:
                pass

            if numeric_string:
                plType = pl.Float64
                df = df.with_columns(df_col[columni].cast(plType).alias(columni))

        if plType == pl.Float64:
            check_integers = True
            check_float32 = False

            plType_intsize = 65
        elif plType == pl.Float32:
            check_integers = True
            check_float32 = False

            plType_intsize = 65

        elif plType == pl.Int64 or plType == pl.UInt64:
            check_integers = True

            plType_intsize = 64
        elif plType == pl.Int32 or plType == pl.UInt32:
            check_integers = True

            plType_intsize = 32
        elif plType == pl.Int16 or plType == pl.UInt16:
            check_integers = True

            plType_intsize = 16
        elif plType == pl.Int8 or plType == pl.UInt8:
            check_integers = True

            plType_intsize = 8

        #   First check integers
        if check_integers and not check_string_only:
            df_col = df_col.lazy().collect()

            dfCastCheck = df_col.filter(pl.col(columni).is_not_null())

            if dfCastCheck.height == 0 and df_col.height != 0:
                #   All missing, cast to bool (for later combinations to ignore)?
                if cast_all_null_to_boolean:
                    try:
                        #   Try casting on the non-null values
                        dfCastCheck = dfCastCheck.select(
                            pl.col(columni).cast(pl.Boolean, strict=True)
                        )
                        dfCastCheck = None
                        #   Worked - then we're good to do on all of them
                        dfcast = df_col.select(
                            pl.col(columni).cast(pl.Boolean, strict=True)
                        )
                        # logger.info("     Cast " + columni + " as " + str(inti))
                        cast_complete = True
                    except:
                        pass
                        #   logger.warning("     Cannot cast " + columni + " as " + str(pl.Boolean))
            else:
                #   All integers?
                if plType == pl.Float32 or plType == pl.Float64:
                    bAllIntegers = (
                        dfCastCheck.with_columns(pl.col(columni).mod(1) == 0).sum()[
                            0, 0
                        ]
                        == dfCastCheck.height
                    )
                else:
                    bAllIntegers = True

                #   Only downcast to an integer if all are integers
                if bAllIntegers:
                    maxValue = dfCastCheck.max().row(0)[0]
                    minValue = dfCastCheck.min().row(0)[0]

                    if minValue is not None:
                        for inti in intlist:
                            if inti == pl.Boolean:
                                intSize = 1
                            else:
                                intSize = int(str(inti).replace("Int", ""))

                            if plType_intsize > intSize:
                                #   logger.info(intlist[inti])
                                lowerbound = intlist[inti][0]
                                upperbound = intlist[inti][1]
                                #   logger.info(lowerbound)
                                #   logger.info(upperbound)

                                if maxValue <= upperbound and minValue >= lowerbound:
                                    #   logger.info("in range for " + str(inti))

                                    try:
                                        #   Try casting on the non-null values
                                        dfCastCheck = dfCastCheck.select(
                                            pl.col(columni).cast(inti, strict=True)
                                        )
                                        dfCastCheck = None
                                        #   Worked - then we're good to do on all of them
                                        dfcast = df_col.select(
                                            pl.col(columni).cast(inti, strict=True)
                                        )
                                        # logger.info("     Cast " + columni + " as " + str(inti))
                                        cast_complete = True

                                    except:
                                        logger.warning(
                                            "     Cannot cast "
                                            + columni
                                            + " as "
                                            + str(inti)
                                        )

                                    if cast_complete:
                                        break

        if not cast_complete and check_float32 and minValue is not None:
            df_col = df_col.lazy().collect()
            try:
                dfcast = df_col.select(pl.col(columni).cast(pl.Float32, strict=True))
                # logger.info("     Cast " + columni + " as " + str(pl.Float32))
                cast_complete = True
            except:
                logger.warning("     Cannot cast " + columni + " as " + str(pl.Float32))

        if cast_complete:
            df = df.with_columns(dfcast[columni].alias(columni))

    df = nw_type.from_polars(df)

    return NarwhalsType.return_df(df, nw_type)


def _compress_datetime(df: pl.LazyFrame | pl.LazyFrame) -> pl.LazyFrame | pl.DataFrame:
    cols_date = {
        coli: df.schema[coli]
        for coli in df.lazy().collect_schema().names()
        if type(df.lazy().collect_schema()[coli]) is pl.Datetime
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
