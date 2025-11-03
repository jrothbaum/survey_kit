from __future__ import annotations
from typing import Callable

import logging
import math
import narwhals as nw
import narwhals.selectors as cs
import polars as pl

from narwhals.typing import IntoFrameT
from scipy.stats import norm

from ..utilities.inputs import list_input
from ..utilities.dataframe import (
    join_wrapper,
    concat_wrapper,
    NarwhalsType,
    fill_missing,
    columns_from_list,
)
from ..utilities.rounding import drb_round_table
from ..serializable import Serializable
from .. import logger


class Replicates(Serializable):
    _save_suffix = "replicates"

    def __init__(
        self,
        weight_stub: str,
        df: IntoFrameT | None = None,
        n_replicates: int | None = None,
        bootstrap: bool = False,
    ):
        if n_replicates is None and df is None:
            message = "You must pass either df or n_replicates to Replicates"
            logger.error(message)
            raise Exception(message)

        if n_replicates is None:
            cols_replicates = columns_from_list(df=df, columns=f"{weight_stub}*")
            n_replicates = len(set(cols_replicates).difference([f"{weight_stub}0"]))
        elif df is not None:
            logger.info("Passed both df and n_replicates to Replicates, ignoring df")

        self.weight_stub = weight_stub
        self.n_replicates = n_replicates
        self.bootstrap = bootstrap

        self.rep_list = [f"{weight_stub}{repi}" for repi in range(0, n_replicates + 1)]


class ReplicateStats:
    def __init__(
        self,
        df_estimates: IntoFrameT | None = None,
        df_ses: IntoFrameT | None = None,
        df_replicates: IntoFrameT | None = None,
        bootstrap: bool = False,
    ):
        self.df_estimates = df_estimates
        self.df_ses = df_ses
        self.df_replicates = df_replicates
        self.bootstrap = bootstrap

    def copy(self) -> ReplicateStats:
        return ReplicateStats(
            df_estimates=self.df_estimates,
            df_ses=self.df_ses,
            df_replicates=self.df_replicates,
            bootstrap=self.bootstrap,
        )

    def _df_ci(self, join_on: list[str], ci_level: float = 0.95):
        ci_multiple = norm.ppf(1 - (1 - ci_level) / 2, loc=0, scale=1)

        nw_type = NarwhalsType(self.df_estimates)
        cols_stats = (
            nw.from_native(self.df_estimates)
            .lazy()
            .drop(join_on)
            .select(cs.nw.numeric())
            .collect_schema()
            .names()
        )

        with_cis = [nw.col(coli) * ci_multiple for coli in cols_stats]
        return self.df_ses.with_columns(with_cis)

    def filter(self, filter_expr: nw.Expr) -> ReplicateStats:
        #   Don't edit the underlying object
        self = self.copy()

        for dfi_name in self._df_attributes:
            dfi = getattr(self, dfi_name)

            if dfi is not None:
                setattr(self, dfi_name, dfi.filter(filter_expr))

        return self

    def select(
        self, select_expr: nw.Expr | str | list[str] | list[nw.Expr]
    ) -> ReplicateStats:
        #   Don't edit the underlying object
        self = self.copy()

        cols_keep = (
            nw.from_native(self.df_estimates)
            .select(select_expr)
            .lazy()
            .collect_schema()
            .names()
        )

        for dfi_name in self._df_attributes:
            if dfi_name == "df_replicates":
                replicate_col = ["___replicate___"]
            else:
                replicate_col = []

            dfi = getattr(self, dfi_name)

            if dfi is not None:
                setattr(self, dfi_name, dfi.select(cols_keep + replicate_col))
        return self

    def with_columns(self, with_expr: nw.Expr | list[nw.Expr]) -> ReplicateStats:
        #   Don't edit the underlying object
        self = self.copy()

        for dfi_name in self._df_attributes:
            dfi = getattr(self, dfi_name)

            if dfi is not None:
                setattr(self, dfi_name, dfi.with_columns(with_expr))

        return self

    def sort(
        self, sort_expr: nw.Expr | list[nw.Expr] | str | list[str]
    ) -> ReplicateStats:
        #   Don't edit the underlying object
        self = self.copy()

        for dfi_name in self._df_attributes:
            dfi = getattr(self, dfi_name)

            if dfi is not None:
                setattr(self, dfi_name, dfi.sort(sort_expr))

        return self

    def drop(
        self, drop_expr: nw.Expr | list[nw.Expr] | str | list[str]
    ) -> ReplicateStats:
        #   Don't edit the underlying object
        self = self.copy()

        for dfi_name in self._df_attributes:
            dfi = getattr(self, dfi_name)

            if dfi is not None:
                setattr(self, dfi_name, dfi.drop(drop_expr))

        return self

    def rename(self, d_rename: dict[str, str]) -> ReplicateStats:
        #   Don't edit the underlying object
        self = self.copy()

        for dfi_name in self._df_attributes:
            dfi = getattr(self, dfi_name)

            if dfi is not None:
                setattr(self, dfi_name, dfi.rename(d_rename))

        return self

    def pipe(self, function: Callable, *args, **kwargs) -> None:
        """
        Pipe a function to df_estimates, df_ses, and df_replicates (as necessary)

        Parameters
        ----------
        function : Callable
            Function to pipe.
        *args : TYPE
            arguments to function
        **kwargs : TYPE
            keyword arguments to function

        Returns
        -------
        None.

        """
        #   Don't edit the underlying object
        self = self.copy()

        for dfi_name in self._df_attributes:
            dfi = getattr(self, dfi_name)
            if dfi is not None:
                setattr(self, dfi_name, function(dfi, *args, **kwargs))

        return self

    def concat_with(
        self,
        rs_concat: ReplicateStats,
        join_on_self: list[str] | None = None,
        join_on_concat: list[str] | None = None,
        how: str = "horizontal",
    ) -> ReplicateStats:
        #   Don't edit the underlying object
        self = self.copy()

        def _concat_df(df: IntoFrameT, df_join: IntoFrameT) -> IntoFrameT:
            nw_type = NarwhalsType(df)

            if how == "horizontal":
                columns = nw.from_native(df).lazy().collect_schema().names()
                replicate_col_name = "___replicate___"
                if replicate_col_name in columns:
                    replicate_col = [replicate_col_name]
                else:
                    replicate_col = []

                join_list = join_on_concat + replicate_col
                df_return = join_wrapper(
                    df, df_join, on=join_list, how="left"
                ).lazy_backend(nw_type)

                return df_return
            elif how == "vertical":
                return concat_wrapper([df, df_join], how="diagonal")

        for dfi in self._df_attributes:
            setattr(
                self,
                dfi,
                _concat_df(df=getattr(self, dfi), df_join=getattr(rs_concat, dfi)),
            )
        return self

    @property
    def _df_attributes(self) -> list[str]:
        return ["df_estimates", "df_ses", "df_replicates"]


def print_se_table(
    df_estimates: IntoFrameT,
    df_ses: IntoFrameT,
    display_all_vars: bool = True,
    display_max_vars: int = 20,
    round_output: bool | int = True,
    sort_vars: list | None = None,
    cols_round: list | None = None,
    cols_n: list | None = None,
    round_all: bool = True,
    estimates_per_page: int = 0,
    sub_log: logging = None,
):
    if sub_log is None:
        sub_log = logger

    #   f_print = print
    #   f_print_args = dict(flush=True)
    f_print_args = {}
    f_print = sub_log.info

    sort_vars = list_input(sort_vars)
    cols_n = list_input(cols_n)
    cols_round = list_input(cols_round)

    nw_estimates = NarwhalsType(df_estimates)
    nw_ses = NarwhalsType(df_ses)

    df_estimates = nw_estimates.to_polars().lazy().collect()
    df_ses = nw_ses.to_polars().lazy().collect()

    stat_vars = df_estimates.drop(sort_vars).collect_schema().names()
    if round_output:
        if type(round_output) is bool:
            round_digits = 4
        else:
            round_digits = round_output

        if round_all and len(cols_n) == 0 and len(cols_round) == 0:
            cols_round = stat_vars

        df_estimates = drb_round_table(
            df=df_estimates,
            columns=cols_round,
            columns_n=cols_n,
            round_all=False,
            digits=round_digits,
        )
        df_ses = drb_round_table(
            df=df_ses,
            columns=cols_round,
            columns_n=cols_n,
            round_all=False,
            digits=round_digits,
        )

    col_row_index = "___estimate_row_count___"
    col_estimate_type = "___estimate_type___"

    df_display = pl.concat(
        [
            df_estimates.with_columns(
                pl.lit("Estimate").alias(col_estimate_type)
            ).with_row_index(col_row_index),
            df_ses.with_columns(pl.lit("SE").alias(col_estimate_type)),
        ],
        how="diagonal_relaxed",
    ).lazy()

    df_display = df_display.sort(sort_vars + [col_estimate_type]).with_columns(
        pl.col(col_row_index).forward_fill()
    )
    df_display = df_display.sort([col_row_index] + [col_estimate_type])

    #   Clear extraneous information
    with_clear = []
    c_est = pl.col(col_estimate_type)
    for coli in sort_vars:
        c_col = pl.col(coli)
        with_clear.append(
            pl.when(c_est == "SE")
            .then(pl.lit(""))
            .otherwise(c_col.cast(pl.String))
            .alias(coli)
        )

    df_display = (
        df_display.with_columns(with_clear).drop(col_estimate_type).drop(col_row_index)
    )

    with pl.Config(fmt_str_lengths=50) as cfg:
        #   Basic formatting
        cfg.set_tbl_cell_alignment("RIGHT")
        cfg.set_tbl_hide_column_data_types(True)
        cfg.set_tbl_hide_dataframe_shape(True)
        cfg.set_thousands_separator(True)
        cfg.set_tbl_width_chars(600)
        cfg.set_tbl_cols(len(df_display.lazy().collect_schema()))
        cfg.set_fmt_float("mixed")

        df_display = df_display.lazy().collect()
        n_rows = df_display.height
        cfg.set_tbl_rows(n_rows)

        if estimates_per_page > 0 and n_rows > estimates_per_page:
            slices = math.ceil(n_rows / (estimates_per_page * 2))

            for slicei in range(slices):
                f_print(
                    df_display.slice(
                        offset=estimates_per_page * 2 * slicei,
                        length=estimates_per_page * 2,
                    ),
                    **f_print_args,
                )
        else:
            f_print(df_display, **f_print_args)

    return nw_estimates.from_polars(df_display)


class _ReplicateSEReturn:
    def __init__(
        self,
        df_estimates: IntoFrameT,
        df_ses: IntoFrameT,
        df_replicates: IntoFrameT,
        bootstrap: bool,
    ):
        self.df_estimates = df_estimates
        self.df_ses = df_ses
        self.df_replicates = df_replicates
        self.bootstrap = bootstrap


def replicates_ses_from_function(
    delegate: Callable,
    join_on: list[str],
    arguments: dict | None = None,
    weight_argument_name: str = "weight",
    weights: list | None = None,
    weight_stub: str = "",
    weight_count: int = 0,
    replicate_name: str = "___replicate___",
    df: pl.LazyFrame | pl.DataFrame | None = None,
    df_argument_name: str = "df",
    bootstrap: bool = False,
) -> _ReplicateSEReturn:
    """
    Take an arbitrary estimation function (delegate)
        with some set of kwargs arguments and repeat the estimation
        for replicate weight or bootstraps

    Parameters
    ----------
    delegate
        A method to call to esimate some statistics
        To be returned as a polars dataframe
    arguments : dict, optional
        kwargs arguments for the delegate method.  Default is none (besides weight)

    join_on : list
        The column names to "join" the estimates together by across replicates
        The identifiers for each estimate.  i.e. if the table is ["Variable","Estimate"]
            then join_on = ["Variable"] because the variable name is
            the identifier for the statistics to be compared across replicates
        If the table is a set of estimates by state, i.e. ["Variable", "st", "Estimate"]
            Then join_on = ["Variable", "st"] as the comparisons across replicates
            should be made by variable-state combination
    weights:list | None, optional
        Pass the replicate/bootstrap weights as a list. The default is None.
    weight_stub:str, optional
        Pass the replicate/bootstrap weights as a stub and a count from 0-weight_count
        As an exmaple if  marsupwt_ is the stub and 2 is the count,
        weights would be set to ["marsupwt_0","marsupwt_1","marsupwt_2"]
    replicate_name:str, optional
        Name of variable that holds the replicate #.  Defaults to "___replicate___"
    bootstrap : bool, optional
        Is this a bootstrap calculation rather than replicate factors. The default is False.
        This affects the SE calculation
    Returns
    -------
    tuple of 3 polars DataFrames and a boolean
        (df_estimates,
         df_ses,
         df_replicates,
         bootstrap)
    """

    if arguments is None:
        arguments = {}

    if weights is None:
        weights = [f"{weight_stub}{i}" for i in range(0, weight_count + 1)]

    n_replicates = len(weights) - 1

    arguments = locals().copy()
    del arguments["bootstrap"]
    del arguments["n_replicates"]
    df_replicates = _replicates_ses_from_function_sequential(**arguments)

    (df_estimates, df_ses) = ses_from_replicates(
        df_replicates=df_replicates,
        join_on=join_on,
        n_replicates=n_replicates,
        bootstrap=bootstrap,
        replicate_name=replicate_name,
    )

    return _ReplicateSEReturn(
        df_estimates=df_estimates,
        df_ses=df_ses,
        df_replicates=df_replicates,
        bootstrap=bootstrap,
    )


def _replicates_ses_from_function_sequential(
    delegate: Callable,
    join_on: list[str],
    arguments: dict | None = None,
    weight_argument_name: str = "weight",
    weights: list | None = None,
    weight_stub: str = "",
    weight_count: int = 0,
    replicate_name: str = "___replicate___",
    df: IntoFrameT | None = None,
    df_argument_name: str = "df",
    replicate_start: int = 0,
    replicate_end: int = 0,
) -> IntoFrameT:
    if arguments is None:
        arguments = {}

    if weights is None:
        weights = [f"{weight_stub}{i}" for i in range(0, weight_count + 1)]

    df_replicates = None

    for replicate_number, weighti in enumerate(weights):
        run_replicate = (replicate_start == 0 and replicate_end == 0) or (
            replicate_number >= replicate_start and replicate_number <= replicate_end
        )

        if run_replicate:
            df_replicatesi = _replicates_ses_from_function_one_replicate(
                delegate=delegate,
                weight=weighti,
                replicate_number=replicate_number,
                arguments=arguments.copy(),
                weight_argument_name=weight_argument_name,
                replicate_name=replicate_name,
                df=df,
                df_argument_name=df_argument_name,
            )

            if df_replicates is None:
                df_replicates = df_replicatesi
            else:
                df_replicates = concat_wrapper(
                    [df_replicates, df_replicatesi], how="vertical"
                )

            if replicate_number % 5 == 0:
                logger.info(f"{replicate_number}[!n]")
            else:
                logger.info(".[!n]")
            if replicate_number % 50 == 0 and replicate_number > 0:
                logger.info("")

    print("")

    return df_replicates


def _replicates_ses_from_function_one_replicate(
    delegate: Callable,
    weight: str,
    replicate_number: int,
    arguments: dict | None = None,
    weight_argument_name: str = "weight",
    replicate_name: str = "___replicate___",
    df: IntoFrameT | None = None,
    df_argument_name: str = "df",
) -> IntoFrameT:
    arguments[weight_argument_name] = weight
    if df is not None:
        arguments[df_argument_name] = df

    df_replicatesi = (
        nw.from_native(delegate(**arguments)).with_columns(
            nw.lit(replicate_number).cast(nw.Int16).alias(replicate_name)
        )
    ).to_native()

    return df_replicatesi


def ses_from_replicates(
    df_replicates: IntoFrameT,
    join_on: list[str],
    n_replicates: int,
    bootstrap: bool = False,
    replicate_name: str = "___replicate___",
) -> tuple[IntoFrameT, IntoFrameT]:
    """
    Take a dataframe of replicate (or bootstrap) weight estimates
        and calculate the SEs
    Parameters
    ----------
    df_replicates : IntoFrameT
        The table f replicate estimates
    join_on : list[str]
        The column names to "join" the estimates together by across replicates
        The identifiers for each estimate.  i.e. if the table is ["Variable","Estimate"]
            then join_on = ["Variable"] because the variable name is
            the identifier for the statistics to be compared across replicates
        If the table is a set of estimates by state, i.e. ["Variable", "st", "Estimate"]
            Then join_on = ["Variable", "st"] as the comparisons across replicates
            should be made by variable-state combination
    n_replicates:int
        The number of replicates (or bootstrap) estimates
    bootstrap : bool, optional
        Is this a bootstrap calculation rather than replicate factors. The default is False.
        This affects the SE calculation
    replicate_name : str
        The name of the column with the replicate number.  The default is "___replicate___"

    Returns
    -------
    tuple of Narwhals DataFrames
        (df_estimates,
         df_ses)
    """

    nw_type = NarwhalsType(df_replicates)
    df_replicates = nw_type.safe_to_narwhals()
    c_replicate = nw.col(replicate_name)
    stat_variables = (
        df_replicates.drop(join_on).drop(replicate_name).collect_schema().names()
    )

    sort_index = "__replicate_calc_sort_index___"
    df_replicates = fill_missing(df_replicates, value=float("nan"))
    df_replicates = nw.from_native(df_replicates)

    if bootstrap:
        #   Replicate factor standard errors
        agg_ses = []
        with_mus = []
        with_squared_errors = []
        with_var_to_se = []
        drops = []
        for coli in stat_variables:
            c_col = nw.col(coli)

            c_hat = c_col.mean().over(join_on).alias(f"___mu{coli}")
            with_mus.append(c_hat)
            c_var = (1 / n_replicates * (c_col - c_hat) ** 2).alias(f"___se{coli}")
            with_squared_errors.append(c_var)
            with_var_to_se.append(c_col**0.5)

            c_standarderror = nw.col(f"___se{coli}").sum().alias(coli)
            agg_ses.append(c_standarderror)

            drops.extend([f"___mu{coli}", f"___se{coli}"])

        df_ses = (
            df_replicates.filter(c_replicate != 0)
            .with_columns(with_mus)
            .with_columns(with_squared_errors)
            .group_by(join_on)
            .agg(agg_ses)
            .with_columns(with_var_to_se)
        )
        cols_ses = df_ses.lazy().collect_schema().names()
        drops = [coli for coli in drops if coli in cols_ses]
        df_ses = df_ses.drop(drops)
    else:
        #   Replicate factor standard errors
        agg_ses = []
        with_mus = []
        with_squared_errors = []
        with_var_to_se = []
        drops = []
        for coli in stat_variables:
            c_col = nw.col(coli)

            c_hat = (
                c_col.first()
                .over(join_on, order_by=[replicate_name])
                .alias(f"___mu{coli}")
            )
            with_mus.append(c_hat)
            c_var = (4 / n_replicates * (c_col - c_hat) ** 2).alias(f"___se{coli}")
            with_squared_errors.append(c_var)
            with_var_to_se.append(c_col**0.5)

            c_standarderror = nw.col(f"___se{coli}").sum().alias(coli)
            agg_ses.append(c_standarderror)

            drops.extend([f"___mu{coli}", f"___se{coli}"])

        df_ses = (
            df_replicates.with_columns(cs.boolean().cast(nw.Int8))
            .with_columns(with_mus)
            .with_columns(with_squared_errors)
            .filter(c_replicate != 0)
            .group_by(join_on)
            .agg(agg_ses)
            .with_columns(with_var_to_se)
        )

    df_estimates = df_replicates.filter(c_replicate == 0).drop(replicate_name)

    df_estimates = fill_missing(df_estimates, None)
    df_ses = fill_missing(df_ses, None)
    return df_estimates, df_ses
