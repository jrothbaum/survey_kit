import narwhals as nw
import polars as pl
import polars.selectors as cs
import math
from narwhals.typing import IntoFrameT


from .dataframe import (
    columns_from_list,
    NarwhalsType,
)
from .compress import compress_df
from .inputs import list_input


def drb_round_table(
    df: IntoFrameT | nw.LazyFrame | nw.DataFrame,
    columns: list | str | None = None,
    columns_n: list | str | None = None,
    columns_exclude: list | str | None = None,
    round_all: bool = True,
    digits: int = 4,
    compress: bool = False,
    display_only: bool = False,
) -> IntoFrameT | nw.LazyFrame | nw.DataFrame:
    """

    Round a table according to Census DRB disclosure rules

    Parameters
    ----------
    df : pl.LazyFrame | pl.DataFrame
        Table to rount
    columns : list|str|None, optional
        Columns to round according to 4 digit rounding rules.
        The default is None.
    columns_n : list|str|None, optional
        Columns to round according to number of observations rounding rules. The default is None.
    columns_exclude : list|str|None, optional
        Columns to NOT round  (such as location codes). The default is None.
    round_all : bool, optional
        Round all columns. The default is True.
    digits : int, optional
        Number of significant digits. The default is 4.
    compress : bool, optional
        Compress the file to the smallest type after (float->int mostly). The default is False.
    display_only : bool, optional
        Skip some steps if true as the table is only being rounded for printing. The default is False.

    Returns
    -------
    df : pl.LazyFrame | pl.DataFrame

    """

    nw_type = NarwhalsType(df)
    df = nw_type.to_polars()

    columns = list_input(columns)
    columns_n = list_input(columns_n)
    columns_exclude = list_input(columns_exclude)
    if len(columns):
        columns = columns_from_list(df=df, columns=columns)

    if len(columns_n) is not None:
        columns_n = columns_from_list(df=df, columns=columns_n)

    if len(columns_exclude) is not None:
        columns_exclude = columns_from_list(df=df, columns=columns_exclude)
    else:
        columns_exclude = []

    #   Ignore round_all if you passed something for columns
    if len(columns) and round_all:
        #   logging.warning(f"Not rounding all variables as already set to round {columns}")
        pass
    elif len(columns) == 0 and round_all:
        #   Didn't pass anything, round the whole table (except columns_n)
        columns = df.select(cs.numeric()).lazy().collect_schema().names()

        if len(columns_n):
            columns = list(set(columns).difference(columns_n))

        if len(columns_exclude):
            columns = list(set(columns).difference(columns_exclude))

    #   Have to be numeric
    columns = df.lazy().select(columns).select(cs.numeric()).collect_schema().names()
    columns_n = (
        df.lazy().select(columns_n).select(cs.numeric()).collect_schema().names()
    )

    schema = df.lazy().collect_schema()
    #   Build up the list of round operations
    with_round = []
    #   Regular rounding
    for coli in columns:
        typei = schema[coli]
        if display_only:
            #   Cast to string
            typei = pl.String

        if coli not in columns_exclude:
            with_round.append(
                pl.col(coli).round_sig_figs(digits).cast(typei).alias(coli)
            )
        elif display_only:
            with_round.append(pl.col(coli).cast(typei).alias(coli))

    #   n rounding
    for coli in columns_n:
        typei = df.lazy().collect_schema()[coli]

        if display_only:
            #   Cast to string
            typei = pl.String

        if coli not in columns_exclude:
            with_round.append(_drb_round_table_n(coli).cast(typei).alias(coli))
        elif display_only:
            with_round.append(pl.col(coli).cast(typei).alias(coli))

    #   Do the rounding
    if len(with_round):
        df = df.with_columns(with_round)

        if display_only:
            with_null_replace = []
            for coli in df.columns:
                with_null_replace.append(
                    pl.when(pl.col(coli).is_null())
                    .then(pl.lit(""))
                    .otherwise(pl.col(coli))
                    .alias(coli)
                )
            df = df.with_columns(with_null_replace)

    if compress and not display_only:
        df = compress_df(df)

    df = nw_type.from_polars(df)

    return NarwhalsType.return_df(df, nw_type)


def _drb_round_table_n(column: str):
    #   Modified from primitive function to permit negatives (for comparisons)
    c_col = pl.col(column)
    c_abs = c_col.abs()
    c_sign = c_col.sign()

    def __round_n(value):
        return _drb_round_table_place(column, value)

    return (
        pl.when(c_abs < 15)
        .then(c_sign * 15)
        .when(c_abs >= 15, c_abs <= 99)
        .then(__round_n(10))
        .when(c_abs >= 100, c_abs <= 999)
        .then(__round_n(50))
        .when(c_abs >= 1000, c_abs <= 9999)
        .then(__round_n(100))
        .when(c_abs >= 10000, c_abs <= 99999)
        .then(__round_n(500))
        .when(c_abs >= 100000, c_abs <= 999999)
        .then(__round_n(1000))
        .otherwise(c_col.round_sig_figs(4))
        .alias(column)
    )


def _drb_round_table_place(column: str = "", value: int = 1):
    c_col = pl.col(column)
    c_abs = c_col.abs()
    c_sign = c_col.sign()

    return c_sign * (c_abs / value + 0.5).floor() * value


def first_digit_position(value: float):
    return math.floor(math.log10(abs(value)))
