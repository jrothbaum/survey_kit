from __future__ import annotations
from typing import Callable


import logging
from copy import deepcopy, copy
import math
import scipy
import narwhals as nw
from narwhals.typing import IntoFrameT

import polars as pl

from .rounding import Rounding
from .statistics import Statistics
from .replicates import (
    Replicates,
    ReplicateStats,
    print_se_table,
    replicates_ses_from_function,
)

from .comparisons import ComparisonItem, statistical_comparison_item, compare

from ..utilities.dataframe import (
    join_wrapper,
    join_list,
    concat_wrapper,
    NarwhalsType,
    safe_sum_cast,
    safe_columns,
)
from ..utilities.rounding import drb_round_table
from ..serializable import Serializable
from .. import logger


class StatCalculator(Serializable):
    """
    A comprehensive class for calculating statistical estimates with optional replicate weights.

    StatCalculator provides a unified interface for computing various statistics on datasets,
    with support for weighted calculations, replicate weight standard errors, grouping,
    and comparison operations. It handles both simple estimates and complex bootstrap
    or replicate weight variance estimation.

    The class supports:
    - Multiple statistics calculated simultaneously
    - Weighted and unweighted estimates
    - Replicate weight and bootstrap standard errors
    - Grouping/stratification via by
    - Automatic disclosure avoidance rounding (which can be disabled)
    - Comparison operations between sets of estimate

    Parameters
    ----------
    df : IntoFrameT
        A narwhals-compliant dataframe
    statistics : list[Statistics]|Statistics|None, optional
        Statistics object(s) defining what columns and statistics to calculate.
        Each Statistics object specifies variables and statistical measures.
        Default is None.
    weight : str, optional
        Column name for survey weights if weighted estimates are desired.
        Default is "" (unweighted).
    scale_wgts_to : float, optional
        Value to scale weights to sum to (proportional adjustment).
        Default is 0.0 (no scaling).
    replicates : Replicates|None, optional
        Replicates object for calculating replicate weight standard errors.
        Generates weight lists from stub names and counts. Default is None.
    by : dict[str,list[str]]|list|None, optional
        Dictionary defining grouping variables for stratified estimates.
        Keys are group names, values are lists of grouping variables.
        Example: {"state":["st"], "county":["st","cty"]}. Default is None.
    display : bool, optional
        Whether to print results to log automatically. Default is True.
    display_all_vars : bool, optional
        Print all variables or truncate display. Default is True.
    display_max_vars : int, optional
        Maximum variables to display when display_all_vars=False. Default is 20.
    round_output : bool|int, optional
        Apply rounding to output (True for DRB rules, int for sig digits).
        Default is True.
    calculate : bool, optional
        Internal parameter - whether to run calculations immediately.
        Default is True.

    Attributes
    ----------
    df_estimates : IntoFrameT, narwhals compliant dataframe
        Main estimates dataframe with calculated statistics.
    df_ses : IntoFrameT, narwhals compliant dataframe
        Standard errors dataframe (populated when replicates are used).
    df_replicates : IntoFrameT, narwhals compliant dataframe
        Full replicate estimates for additional analysis.
    variable_ids : list[str]
        Column names that identify unique estimates/variables.
    summarize_vars : list[str]
        All grouping variables from by (flattened).
    bootstrap : bool
        Whether bootstrap standard errors are to be
        calculated, as opposed to replicate weights.

    Examples
    --------
    Basic usage with simple statistics:

    >>> from NEWS.CodeUtilities.Python.SummaryStats import Statistics
    >>> stats = Statistics(columns=["income", "age"], statistics=["mean", "median"])
    >>> sc = StatCalculator(df=my_data, statistics=stats, weight="survey_wgt")
    >>> sc.print()

    With replicate weights for standard errors:

    >>> from NEWS.CodeUtilities.Python.SummaryStats import Replicates
    >>> reps = Replicates(weight_stub="rep_wgt_", n_replicates=80)
    >>> sc = StatCalculator(df=my_data, statistics=stats, replicates=reps)
    >>> sc.print()  # Will show standard errors

    Grouped analysis:

    >>> calc = StatCalculator(
    ...     df=my_data,
    ...     statistics=stats,
    ...     by={"state": ["state_code"], "region": ["region_code"]}
    ... )

    Comparison between two sets of estimates:

    >>> sc_1 = StatCalculator(df=data1, statistics=stats)
    >>> sc_2 = StatCalculator(df=data2, statistics=stats)
    >>> comparison = sc_1.compare(sc_2)
    >>> comparison["difference"].print()
    """

    _save_suffix = "stats_calc"

    def __init__(
        self,
        df: IntoFrameT | None = None,
        statistics: list[Statistics] | Statistics | None = None,
        weight: str = "",
        scale_wgts_to: float = 0.0,
        replicates: Replicates | None = None,
        by: dict[str, list[str]] | list | None = None,
        display: bool = True,
        display_all_vars: bool = True,
        display_max_vars: int = 20,
        round_output: bool | int = False,
        allow_slow_pandas: bool = False,
        calculate: bool = True,
    ):
        if statistics is None:
            self.statistics = []
            calculate = False
        elif type(statistics) is Statistics:
            statistics = [statistics]

        self.df = df
        if statistics is not None:
            if type(statistics) is not list:
                statistics = [statistics]
        self.statistics = statistics
        self.weight = weight
        if by is not None:
            if len(by) == 0:
                by = None

        self.by = by
        self.display = display
        self.display_all_vars = display_all_vars
        self.display_max_vars = display_max_vars
        self.round_output = round_output
        self.summarize_vars = self._by_vars()
        self.rounding = Rounding(round_output=round_output, round_all=False)

        #   Default columns for variable name id
        self.variable_ids = ["Variable"]

        self.replicates = replicates
        self.scale_wgts_to = scale_wgts_to

        self.allow_slow_pandas = allow_slow_pandas

        self.replicate_stats = ReplicateStats()
        if replicates is not None:
            self.replicate_stats.bootstrap = replicates.bootstrap

        self.scale_weights()

        if calculate:
            if self.replicates is None:
                self._calculate()
            else:
                self._calculate_replicates()

            self.df = None

    def copy(self):
        sc_copy = StatCalculator(
            df=self.df,
            statistics=copy(self.statistics),
            weight=self.weight,
            scale_wgts_to=0,
            replicates=copy(self.replicates),
            by=copy(self.by),
            display=self.display,
            display_all_vars=self.display_all_vars,
            display_max_vars=self.display_max_vars,
            round_output=False,
            allow_slow_pandas=allow_slow_pandas,
            calculate=False,
        )

        sc_copy.scale_wgts_to = self.scale_wgts_to
        sc_copy.rounding = copy(self.rounding)
        sc_copy.replicate_stats = self.replicate_stats.copy()
        sc_copy.variable_ids = self.variable_ids

        return sc_copy

    @property
    def df_estimates(self):
        """
        IntoFrameT : Main estimates dataframe containing calculated statistics.

        This property provides access to the primary results table with all
        calculated statistics. Includes variable identifiers, grouping variables,
        and statistical estimates as columns.
        """
        return self.replicate_stats.df_estimates

    @df_estimates.setter
    def df_estimates(self, value):
        self.replicate_stats.df_estimates = value

    @property
    def df_ses(self):
        """
        IntoFrameT : Standard errors dataframe (when replicate weights are used).

        Contains standard error estimates for all statistics calculated using
        replicate weight methods. Has the same structure as df_estimates but
        with standard errors instead of point estimates. Only populated when
        replicates parameter is provided.
        """
        return self.replicate_stats.df_ses

    @df_ses.setter
    def df_ses(self, value):
        self.replicate_stats.df_ses = value

    @property
    def df_replicates(self):
        """
        IntoFrameT : Full replicate estimates dataframe.

        Contains individual estimates for each replicate weight, allowing for
        custom variance calculations or additional analysis. Includes all
        columns from df_estimates plus a replicate identifier column.
        Only populated when replicates parameter is provided.
        """
        return self.replicate_stats.df_replicates

    @df_replicates.setter
    def df_replicates(self, value):
        self.replicate_stats.df_replicates = value

    @property
    def bootstrap(self):
        return self.replicate_stats.bootstrap

    def _by_vars(self=None, by: dict | None = None) -> list[str]:
        """
        Just get a list of the variables to be used as indexes for
            summary stats (for a select statement)
        Returns
        -------
        list[str]
            A full list of all the indexes (with no duplicates)

        """

        if by is None:
            by = self.by

        summarize_list = []

        if by is not None:
            if type(by) is dict:
                for listi in by.values():
                    summarize_list.extend(listi)
            elif type(by) is list:
                for itemi in by:
                    if type(itemi) is list:
                        summarize_list.extend(itemi)
                    else:
                        summarize_list.append(itemi)

            summarize_list = list(dict.fromkeys(summarize_list))

        return summarize_list

    def _calculate(self, weight: str | None = None, display: bool | None = None):
        """
        Parameters
        ----------
        weight : str|None, optional
            Programmer option, do not use. The default is None.
        display : bool|None, optional
            Display the results?. The default is None.

        Returns
        -------
        Calculate the estimates (no SEs).  Mostly should only be called internally
        Populates df_estimates

        """

        df_collected = None

        if weight is None:
            weight = self.weight

        for statsi in self.statistics:
            dfi = statsi.calculate(
                df=self.df,
                weight=weight,
                by=self.by,
                summarize_vars=self.summarize_vars,
                rounding=self.rounding,
                allow_slow_pandas=self.allow_slow_pandas,
            )

            if df_collected is None:
                df_collected = dfi
            else:
                cols_prior = df_collected.drop(
                    self.variable_ids + self.summarize_vars
                ).columns
                cols_now = dfi.drop(self.variable_ids + self.summarize_vars).columns
                cols_match = list(set(cols_prior).intersection(cols_now))

                n_rows = df_collected.select(nw.len()).collect().item()
                df_collected = (
                    join_wrapper(
                        df=df_collected.with_row_index("__summary_index__"),
                        df_to=dfi.with_row_index("__summary_index2__", offset=n_rows),
                        on=self.variable_ids + self.summarize_vars,
                        how="full",
                    )
                    # df_collected = (
                    #                     JoinFileList_Simple(
                    #                         dflist=[
                    #                                     df_collected.with_row_index("__summary_index__"),
                    #                                     dfi.with_row_index("__summary_index2__",
                    #                                                        offset=dfRowCount(df_collected)),
                    #                                 ],
                    #                         Join="outer",
                    #                         JoinOn=self.variable_ids + self.summarize_vars,
                    #                         join_nulls=True,
                    #                         quietly=True
                    #                     )
                    .with_columns(
                        nw.coalesce(
                            nw.col(["__summary_index__", "__summary_index2__"]).alias(
                                "__summary_index__"
                            )
                        )
                    )
                    .sort(self.summarize_vars + ["__summary_index__"])
                    .drop(["__summary_index__", "__summary_index2__"])
                )

                if len(cols_match):
                    cols_new = list(set(cols_now).difference(cols_prior))
                    cols_select = cols_prior + cols_new
                    with_coalesce = [
                        nw.coalesce(nw.col(coli, f"{coli}_right"))
                        for coli in cols_match
                    ]

                    df_collected = df_collected.with_columns(with_coalesce).select(
                        self.variable_ids + self.summarize_vars + cols_select
                    )

        self.df_estimates = df_collected

        if self.rounding.round_output:
            self.df_estimates = self.round_results()

        if display is None:
            display = self.display

        if display:
            self.print()

        return self.df_estimates

    def _calculate_replicates(self):
        """
        Calculate the estimates for each replicate weight

        Returns
        -------
        Populates df_estimates, df_ses and df_replicates

        """

        replicate_se_return = replicates_ses_from_function(
            delegate=self._calculate,
            arguments={"display": False},
            join_on=self.variable_ids + self.summarize_vars,
            weights=self.replicates.rep_list,
            bootstrap=self.replicates.bootstrap,
        )

        self.df_estimates = replicate_se_return.df_estimates
        self.df_ses = replicate_se_return.df_ses
        self.df_replicates = replicate_se_return.df_replicates

        if self.rounding.round_output:
            self.df_ses = self.round_results(df=self.df_ses)

        if self.display:
            self.print()

    def round_results(
        self,
        df: IntoFrameT | None = None,
        rounding: Rounding | None = None,
        display_only: bool = False,
    ) -> IntoFrameT:
        """
        Parameters
        ----------
        df : IntoFrameT, optional
            Table of estimates. The default is the estimates in df_estimates
        rounding : Rounding|None, optional
            Rounding (True for DRB rules) and an integer for specific number of significant digits. The default is self's rounding.
        display_only : bool, optional
            If True, affects the display of numbers (casts to strings). The default is False.

        Returns
        -------
        df : IntoFrameT
            The rounded estimates.

        """

        if df is None:
            df = self.df_estimates

        if rounding is None:
            rounding = self.rounding

        if df is not None:
            df = drb_round_table(
                df=df,
                columns=rounding.cols_round,
                columns_n=rounding.cols_n,
                columns_exclude=rounding.cols_exclude,
                round_all=rounding.round_all,
                digits=rounding.round_digits,
                display_only=display_only,
            )

        return df

    def scale_weights(self=None):
        if self.df is None:
            return None

        if self.scale_wgts_to > 0:
            if self.weight != "":
                self.df = self.df.with_columns(
                    (
                        nw.col(self.weight) / nw.sum(self.weight) * self.scale_wgts_to
                    ).alias(self.weight)
                )

            if self.replicates is not None:
                for weighti in self.replicates.rep_list:
                    self.df = self.df.with_columns(
                        (nw.col(weighti) / nw.sum(weighti) * self.scale_wgts_to).alias(
                            weighti
                        )
                    )

    def print(
        self,
        df: IntoFrameT | None = None,
        round_output: bool | int | None = None,
        estimates_per_page: int = 0,
        sub_log: logging = None,
    ):
        """
        Print the estimates (with SEs if applicable) to the log.

        Parameters
        ----------
        df : IntoFrameT, optional
            The estimates to display. Default is the estimates in self.
        round_output : bool|int|None, optional
            Rounding rule (True for DRB, integer for number of significant digits).
            Default is self's rounding rule.
        estimates_per_page : int, optional
            Repeat the header every k estimates. Defaults to 0 (don't repeat).
        sub_log : logging, optional
            Override logger. Default is None (no override).

        Returns
        -------
        None
        """
        if self.df_replicates is not None:
            self._print_replicates(
                round_output=round_output,
                estimates_per_page=estimates_per_page,
                sub_log=sub_log,
            )
        else:
            self._print_estimates(
                df=df,
                round_output=round_output,
                estimates_per_page=estimates_per_page,
                sub_log=sub_log,
            )

    def _print_estimates(
        self,
        df: IntoFrameT | None = None,
        round_output: bool | int | None = None,
        estimates_per_page: int = 0,
        sub_log: logging = None,
    ):
        """
        Prints the estimates (when there are no SEs) to the log

        Parameters
        ----------
        df : IntoFrameT, optional
            The estimates to show The default is the estimates in self.
        round_output : bool|int|None, optional
            Rounding rule (True for DRB, integer for number of significant digits). The default is self's rounding rule.
        estimates_per_page : int, optional
            Repeat the header every k estimates.  Defaults to 0 (don't)
        sub_log : logging , optional
            Override logger?  Default is None (no override)
        Returns
        -------
        None.

        """

        if df is None:
            df = self.df_estimates

        nw_type = NarwhalsType(df)
        df = nw_type.to_polars().lazy().collect()
        if sub_log is None:
            sub_log = logger

        #   f_print = print
        f_print = sub_log.info
        #   Round?
        if round_output:
            rounding = deepcopy(self.rounding)
            rounding.set_round_digits(round_output)

            df = self.round_results(df=df, rounding=rounding, display_only=True)

        if self.display_all_vars:
            n_rows = df.height
        else:
            n_rows = min(self.display_max_vars, df.height)

        with pl.Config(fmt_str_lengths=50) as cfg:
            #   Basic formatting
            cfg.set_tbl_cell_alignment("RIGHT")
            cfg.set_tbl_hide_column_data_types(True)
            cfg.set_tbl_hide_dataframe_shape(True)
            cfg.set_thousands_separator(True)
            cfg.set_tbl_width_chars(600)
            cfg.set_tbl_cols(len(df.lazy().collect_schema()))

            cfg.set_tbl_rows(n_rows)

            if estimates_per_page > 0 and n_rows > estimates_per_page:
                slices = math.ceil(n_rows / estimates_per_page)

                for slicei in range(slices):
                    f_print(
                        df.slice(
                            offset=estimates_per_page * slicei,
                            length=estimates_per_page,
                        )
                    )
            else:
                f_print(df)

    def _print_replicates(
        self,
        round_output: bool | int | None = None,
        estimates_per_page: int = 0,
        sub_log: logging = None,
    ):
        """
        Prints the estimates (when there are SEs) to the log

        Parameters
        ----------
        round_output : bool|int|None, optional
            Rounding rule (True for DRB, integer for number of significant digits). The default is self's rounding rule.
        estimates_per_page : int, optional
            Repeat the header every k estimates.  Defaults to 0 (don't)
        sub_log : logging , optional
            Override logger?  Default is None (no override)

        Returns
        -------
        None.

        """
        if sub_log is None:
            sub_log = logger

        #   Round?
        if round_output:
            rounding = deepcopy(self.rounding)
            rounding.set_round_digits(round_output)

            df_estimates = self.round_results(
                df=self.df_estimates, rounding=rounding, display_only=True
            )
            df_ses = self.round_results(
                df=self.df_ses, rounding=rounding, display_only=True
            )
        else:
            df_estimates = self.df_estimates
            df_ses = self.df_ses

        print_se_table(
            df_estimates=df_estimates,
            df_ses=df_ses,
            display_all_vars=self.display_all_vars,
            display_max_vars=self.display_max_vars,
            sort_vars=self.variable_ids + self.summarize_vars,
            round_output=False,
            sub_log=sub_log,
        )

    def table_of_estimates(
        self,
        round_output: bool | int | None = None,
        estimates_to_show: list[str] | None = None,
        variable_prefix: str = "",
        estimate_type_variable_name: str = "Statistic",
        ci_level: float = 0.95,
    ) -> IntoFrameT:
        """
        Create a formatted table of estimates with option of statistics to report.

        Parameters
        ----------
        round_output : bool|int|None, optional
            Rounding rule for display.
        estimates_to_show : list[str] | None, optional
            List of estimate types to include. Options: "estimate", "se", "t", "p", "ci".
            Default is ["estimate", "se"].
        variable_prefix : str, optional
            Prefix to add to variable column names. Default is "".
        estimate_type_variable_name : str, optional
            Name for the column indicating statistic type. Default is "Statistic".
        ci_level : float, optional
            Confidence interval level for "ci" estimates. Default is 0.95.

        Returns
        -------
        IntoFrameT
            Formatted table with estimates arranged by statistic type.
        """
        if estimates_to_show is None:
            estimates_to_show = ["estimate", "se"]

        df_ordered = []
        nw_ordered = []
        col_sort = "__order_output_table__"
        for index, esti in enumerate(estimates_to_show):
            dfi = None
            if esti.lower() == "estimate":
                dfi = self.df_estimates

            elif esti.lower() == "se" and self.df_replicates is not None:
                dfi = self.df_ses
            elif esti.lower() == "t" and self.df_replicates is not None:
                dfi = self._df_t()
            elif esti.lower() == "p" and self.df_replicates is not None:
                dfi = self._df_p()
            elif esti.lower() == "ci" and self.df_replicates is not None:
                dfi = self._df_ci(ci_level=ci_level)
            else:
                message = f"{esti} not allowed for estimates_to_show"
                logger.error(message)
                raise Exception(message)

            if dfi is not None:
                nwi = NarwhalsType(dfi)
                nw_ordered.append(nwi)
                dfi = nwi.to_polars()
                df_ordered.append(
                    dfi.with_columns(
                        [
                            pl.lit(index).alias(col_sort),
                            pl.lit(esti.lower()).alias(estimate_type_variable_name),
                        ]
                    )
                )

        col_row_index = "___estimate_row_count___"
        df_ordered[0] = df_ordered[0].with_row_index(col_row_index)
        df_display = concat_wrapper(df_ordered, how="diagonal").lazy()

        sort_vars = self.variable_ids + self.summarize_vars
        df_display = df_display.sort(sort_vars + [col_sort]).with_columns(
            pl.col(col_row_index).forward_fill()
        )
        df_display = df_display.sort([col_row_index] + [col_sort]).drop(col_row_index)

        #   Clear extraneous information
        with_clear = []
        for coli in sort_vars:
            c_col = pl.col(coli)
            with_clear.append(
                pl.when(pl.col(col_sort) != 0)
                .then(pl.lit(""))
                .otherwise(c_col.cast(pl.String))
                .alias(coli)
            )

        select_order = sort_vars + [estimate_type_variable_name]
        remaining = []
        rename = {}
        for coli in df_display.lazy.collect_schema().names():
            if coli not in select_order and coli != col_sort:
                if variable_prefix != "":
                    rename[coli] = f"{variable_prefix}{coli}"

                remaining.append(coli)

        df_display = df_display.with_columns(with_clear).select(
            select_order + remaining
        )
        #   Round?
        if round_output:
            rounding = deepcopy(self.rounding)
            rounding.set_round_digits(round_output)

            if (
                len(rounding.cols_n) == 0
                and len(rounding.cols_round) == 0
                and not rounding.round_all
            ):
                #   Nothing set to round, round all
                rounding.cols_round = remaining

            df_display = self.round_results(
                df=df_display, rounding=rounding, display_only=True
            )

        if len(rename):
            df_display = df_display.rename(rename)
        return nw_ordered[0].lazy().from_polars(df_display)

    def join_tables_of_estimates(
        self, df_list: list[IntoFrameT], estimate_type_variable_name: str = "Statistic"
    ) -> IntoFrameT:
        df_list = [nw.from_native(dfi) for dfi in df_list]
        sort_vars = (
            self.variable_ids + self.summarize_vars + [estimate_type_variable_name]
        )

        variable_filled = []
        for coli in self.variable_ids:
            if df_list[0].schema[coli] == nw.String:
                c_missing = nw.col(coli).is_null() | (pl.col(coli) == "")
            else:
                c_missing = nw.col(coli).is_missing()

            variable_filled.append(
                (
                    nw.when(c_missing)
                    .then(nw.lit(None))
                    .otherwise(nw.col(coli))
                    .alias(coli)
                    .fill_null(strategy="forward")
                )
            )

        row_indices = []
        for i in range(0, len(df_list)):
            row_indices.append(f"__row_index_{i}")
            df_list[i] = (
                df_list[i]
                .with_row_index(f"__row_index_{i}")
                .with_columns(variable_filled)
            )

        df_out = join_list(df_list, on=sort_vars, how="full").sort(row_indices)

        with_clear = []
        c_index = nw.col("__row_index_0")
        for coli in self.variable_ids:
            c_col = nw.col(coli)

            with_clear.append(
                nw.when(c_index == nw.min("__row_index_0").over(self.variable_ids))
                .then(c_col)
                .otherwise(nw.lit(""))
                .alias(coli)
            )

        df_out = df_out.with_columns(with_clear).news.drop_if_exists(["__row_index_*"])
        return df_out.to_native()

    def _df_t(self) -> IntoFrameT:
        join_on = self.variable_ids + self.summarize_vars
        nw_type = NarwhalsType(self.df_estimates)
        cols_stats = (
            nw.from_native(self.df_estimates)
            .lazy()
            .drop(join_on)
            .collect_schema()
            .names()
        )

        with_t = []
        for coli in cols_stats:
            c_est = nw.col(coli)
            c_se = nw.col(f"{coli}_se")

            with_t.append((c_est / c_se).abs().alias(coli))

        df_out = join_list(
            [self.df_estimates, self.df_ses],
            how="left",
            on=join_on,
            suffixes=["", "_se"],
        ).select(join_on + with_t)
        return NarwhalsType.return_df(df_out, nw_type)

    def _df_p(self) -> pl.DataFrame | pl.LazyFrame:
        nw_p = NarwhalsType(self._df_t())
        df_p = nw_p.to_polars()

        join_on = self.variable_ids + self.summarize_vars
        cols_stats = self.df_estimates.drop(join_on).columns

        def p_value(t):
            if t == float("inf") or t == float("nan"):
                return 0
            df = 1_000_000

            return scipy.stats.t.sf(t, df) * 2

        def p_value_lambda(t, col_t):
            try:
                return p_value(t[col_t])
            except:
                return None

        for coli in cols_stats:
            index_t = df_p.columns.index(f"{coli}")

            df_p = (
                df_p.with_columns(df_p.map_rows(lambda t: p_value_lambda(t, index_t)))
                .drop(coli)
                .rename({"map": coli})
            )

        return nw_p.from_polars(df_p.select(join_on + cols_stats))

    def _df_ci(self, ci_level: float = 0.95):
        return self.replicate_stats._df_ci(
            ci_level=ci_level, join_on=self.variable_ids + self.summarize_vars
        )

    def compare(
        self,
        compare_to,
        difference: bool = True,
        ratio: bool = True,
        display: bool = True,
        ratio_minus_1: bool = True,
        compare_list_variables: list[ComparisonItem.Variable] | None = None,
        compare_list_columns: list[ComparisonItem.Column] | None = None,
        quietly: bool = False,
    ):
        """
        Compare this set of estimates to another set of estimates,
        including MultipleImputation estimates.

        Parameters
        ----------
        compare_to : StatCalculator | MultipleImputation
            The other object to compare to.
        difference : bool, optional
            Calculate and return the difference (with key "difference"). Default is True.
        ratio : bool, optional
            Calculate and return the ratio (with key "ratio"). Default is True.
        ratio_minus_1 : bool, optional
            Rescale ratio by subtracting 1 from it. Default is True.
        display : bool, optional
            Print the difference/ratio to the log. Default is True.
        compare_list_variables : list[ComparisonItem.Variable] | None, optional
            List of variables to compare (i.e. compare rows from prior calculations)
        compare_list_columns : list[ComparisonItem.Column], optional
            List of columns to compare
            For example if compare_list_variables = [ComparisonItem.Column("mean","median")]
                then compare the mean of 1 to the median of 2
        quietly : bool, optional
            Suppress informational messages. Default is False.

        Returns
        -------
        dict[str, StatCalculator]
            Dictionary with keys ["difference","ratio"] containing
            StatCalculator objects with the comparison estimates
            (with SEs if applicable).
        """

        outputs = {}
        if statistical_comparison_item(self) and statistical_comparison_item(
            compare_to
        ):
            if not quietly:
                if self.bootstrap:
                    logger.info("Comparing estimates using bootstrap weights")
                else:
                    logger.info("Comparing estimates using replicate weights")

            outputs = compare(
                stats1=self,
                stats2=compare_to,
                join_on=self.variable_ids + self.summarize_vars,
                rounding=self.rounding,
                difference=difference,
                ratio=ratio,
                ratio_minus_1=ratio_minus_1,
                compare_list_variables=compare_list_variables,
                compare_list_columns=compare_list_columns,
            )

            if display:
                if difference:
                    logger.info("  Difference")
                    outputs["difference"].print(round_output=self.round_output)
                    logger.info("\n")
                if ratio:
                    logger.info("  Ratio")
                    outputs["ratio"].print(round_output=self.round_output)
                    logger.info("\n")

        else:
            if not quietly:
                logger.info("Comparing estimates")

            df1 = self.df_estimates
            df2 = compare_to.df_estimates

            (df1, df2) = StatComp.process_compare_lists(
                df1=df1,
                df2=df2,
                join_on=self._by_vars() + self.variable_ids,
                compare_list_variables=compare_list_variables,
                compare_list_columns=compare_list_columns,
            )

            sm_compare = StatCalculator(
                df=None, statistics=self.statistics, by=self.by, calculate=False
            )

            cols_index = self.variable_ids + self.summarize_vars
            cols_nonindex = df1.drop(cols_index).columns

            df1 = SafeCollect(df1)
            df2 = SafeCollect(df2)

            #   logger.info(df1.schema)
            #   Upcast any columns that need to be
            [df1, df2] = _polars_safe_upcast(
                df1.with_columns(pl.col(pl.Boolean).cast(pl.Int8)),
                df2.with_columns(pl.col(pl.Boolean).cast(pl.Int8)),
                cols1=cols_nonindex,
                cols2=cols_nonindex,
            )

            df_difference = SafeCollect(df2.select(cols_nonindex)) - df1.select(
                cols_nonindex
            )
            df_ratio = (df_difference) / df1.select(cols_nonindex)

            if difference:
                sm_diff = sm_compare

                if ratio:
                    sm_compare = deepcopy(sm_diff)

                sm_diff.df_estimates = pl.concat(
                    [df1.select(cols_index), df_difference], how="horizontal"
                )

                outputs["difference"] = sm_diff

                if display:
                    logger.info("  Difference")
                    sm_diff.print(round_output=sm_diff.round_output)
                    logger.info("\n")

            if ratio:
                sm_ratio = sm_compare

                sm_ratio.df_estimates = pl.concat(
                    [df1.select(cols_index), df_ratio], how="horizontal"
                )

                outputs["ratio"] = sm_ratio

                if display:
                    logger.info("  Ratio")
                    sm_ratio.print(round_output=sm_ratio.round_output)
                    logger.info("\n")

        return outputs

    def from_function(
        delegate: Callable,
        estimate_ids: list | str,
        df: IntoFrameT | None = None,
        df_argument: str = "df",
        arguments: dict | None = None,
        weight: str = "",
        replicates: Replicates | None = None,
        scale_wgts_to: float = 0.0,
        weight_argument_name: str = "weight",
        by: dict[str, list[str]] | None = None,
        display: bool = True,
        display_all_vars: bool = True,
        display_max_vars: int = 20,
        round_output: bool | int = True,
    ) -> StatCalculator:
        """
        Create a StatCalculator from a custom function that returns estimates.

        This static method allows wrapping any function that returns estimates
        in a StatCalculator object for easy display and comparison.

        Parameters
        ----------
        delegate : callable
            Function that returns a table of estimates. Must accept weight
            parameter if replicates are used.
        estimate_ids : list | str
            Column names that identify each unique estimate.
        df : pl.LazyFrame | pl.DataFrame, optional
            Dataframe passed as "df" argument to delegate. Allows dynamic
            subsetting with by. Default is None.
        df_argument : str, optional
            Name of argument with data. Defaults is "df".
        arguments : dict, optional
            Static arguments (other than weight) passed to delegate. Default is None.
        weight : str, optional
            Weight column name for weighted statistics. Default is "".
        replicates : Replicates|None, optional
            Replicates object for replicate weight standard errors. Default is None.
        scale_wgts_to : float, optional
            Scale weights to sum to this value. Default is 0.0 (no scaling).
        weight_argument_name : str, optional
            Keyword argument name for passing weight to delegate. Default is "weight".
        by : dict[str,list[str]]|None, optional
            Dictionary defining grouping variables for summary statistics.
        display : bool, optional
            Print results to log. Default is True.
        display_all_vars : bool, optional
            Print all variables rather than truncated summary. Default is True.
        display_max_vars : int, optional
            Maximum variables to print if display_all_vars=False. Default is 20.
        round_output : bool|int, optional
            Round the output. Default is True.

        Returns
        -------
        StatCalculator
            StatCalculator object containing the function results with
            estimates, SEs, and replicates as applicable.
        """

        #   Input parsing
        if arguments is None:
            arguments = {}

        if type(estimate_ids) is str:
            estimate_ids = [estimate_ids]

        if df is None:
            by = None
        else:
            if scale_wgts_to > 0:
                if weight != "":
                    weights_to_cast = [weight]
                    if replicates is not None:
                        weights_to_cast.extend(replicates.rep_list)
                    df = safe_sum_cast(df, weights_to_cast)

                    with_scale = [
                        (nw.col(weighti) / nw.col(weighti).sum() * scale_wgts_to).alias(
                            nw.col(weighti)
                        )
                        for weighti in weights_to_cast
                    ]
                    df = nw.from_native(df).with_columns(with_scale).to_native()

        if by is None:
            by = {"All": []}

        replicate_name = "___replicate___"

        df_estimates = []
        df_ses = []
        df_replicates = []
        by_vars = StatCalculator._by_vars(by=by)

        nw_type = NarwhalsType(df)
        for keyi, valuei in by.items():
            if keyi == "All":
                logger.info(f"Running {delegate.__name__}")
            else:
                logger.info(f"Running {delegate.__name__} for {keyi}")

            df_list = []
            if df is not None:
                if len(valuei):
                    df_partitioned = (
                        nw_type.to_polars()
                        .lazy()
                        .collect()
                        .partition_by(by=valuei, maintain_order=True, include_key=True)
                    )

                    df_partitioned = [
                        nw_type.from_polars(dfi) for dfi in df_partitioned
                    ]

                    df_list.extend(df_partitioned)
                else:
                    df_list.append(df)

            if len(df_list) == 0:
                df_list = [None]

            for dfi in df_list:
                append_by = []
                append_values = []

                if dfi is not None:
                    arguments[df_argument] = dfi

                    if len(valuei):
                        append_values = dfi.select(valuei).unique().to_dicts()
                        append_by = [
                            nw.lit(valuei).alias(keyi)
                            for keyi, valuei in append_values[0].items()
                        ]

                if replicates is None:
                    df_esti = delegate(**arguments)
                    if len(append_by):
                        df_esti = (
                            nw.from_native(df_esti).with_columns(append_by).to_native()
                        )

                    df_estimates.append(df_esti)
                else:
                    if len(append_values):
                        logger.info(append_values)

                    rep_return = replicates_ses_from_function(
                        delegate=delegate,
                        arguments=arguments,
                        join_on=estimate_ids,
                        weight_argument_name=weight_argument_name,
                        weights=replicates.rep_list,
                        replicate_name=replicate_name,
                    )

                    df_esti = rep_return.df_estimates
                    df_sei = rep_return.df_ses
                    df_repi = rep_return.df_replicates

                    if len(append_by):
                        df_esti = (
                            nw.from_native(df_esti).with_columns(append_by).to_native()
                        )
                        df_sei = (
                            nw.from_native(df_sei).with_columns(append_by).to_native()
                        )

                        df_repi = (
                            nw.from_native(df_repi).with_columns(append_by).to_native()
                        )

                    df_estimates.append(df_esti)
                    df_ses.append(df_sei)
                    df_replicates.append(df_repi)

            del df_list

        #   Set up the output
        ss_out = StatCalculator(
            df=None,
            weight=weight,
            replicates=replicates,
            by=by,
            display=display,
            display_all_vars=display_all_vars,
            display_max_vars=display_max_vars,
            round_output=round_output,
            calculate=False,
        )

        ss_out.variable_ids = estimate_ids

        if len(df_estimates):
            df_estimates = concat_wrapper(df_estimates, how="diagonal")
            #   Final variable order
            if len(by_vars):
                select_order = estimate_ids + by_vars
                select_order.extend(
                    [
                        coli
                        for coli in safe_columns(df_estimates)
                        if coli not in select_order
                    ]
                )
            else:
                select_order = safe_columns(df_estimates)
            ss_out.df_estimates = df_estimates.select(select_order)
        if len(df_ses):
            ss_out.df_ses = concat_wrapper(df_ses, how="diagonal").select(select_order)

        if len(df_ses):
            ss_out.df_replicates = concat_wrapper(df_replicates, how="diagonal").select(
                select_order + [replicate_name]
            )

        ss_out.df_estimates = ss_out.round_results(df=ss_out.df_estimates)
        ss_out.df_ses = ss_out.round_results(df=ss_out.df_ses)

        if display:
            ss_out.print()

        return ss_out

    def filter(self, filter_expr: nw.Expr) -> StatCalculator:
        self = self.copy()
        self.replicate_stats = self.replicate_stats.filter(filter_expr)

        return self

    def select(
        self, select_expr: nw.Expr | str | list[str] | list[nw.Expr]
    ) -> StatCalculator:
        cols_keep = (
            nw.from_native(self.df_estimates)
            .lazy()
            .select(select_expr)
            .collect_schema()
            .names()
        )
        add_join_on = list(set(self.variable_ids).difference(cols_keep))
        cols_keep = add_join_on + cols_keep

        self = self.copy()
        self.replicate_stats = self.replicate_stats.select(cols_keep)

        return self

    def with_columns(self, with_expr: nw.Expr | list[nw.Expr]) -> StatCalculator:
        self = self.copy()
        self.replicate_stats = self.replicate_stats.with_columns(with_expr)

        return self

    def sort(
        self, sort_expr: nw.Expr | list[nw.Expr] | str | list[str]
    ) -> StatCalculator:
        self = self.copy()
        self.replicate_stats = self.replicate_stats.sort(sort_expr)

        return self

    def drop(
        self, drop_expr: nw.Expr | list[nw.Expr] | str | list[str]
    ) -> ReplicateStats:
        self = self.copy()
        self.replicate_stats = self.replicate_stats.drop(drop_expr)

        return self

    def rename(self, d_rename: dict[str, str]) -> StatCalculator:
        self = self.copy()
        self.replicate_stats = self.replicate_stats.rename(d_rename)

        return self

    def scale_by(
        self, factor: float, columns: list[str] | str | None = None
    ) -> StatCalculator:
        if columns is None:
            #   Any columns that aren't the join_on ones
            columns = (
                nw.from_native(self.df_estimates.columns)
                .lazy()
                .collect_schema()
                .names()
            )
            columns = list(set(columns).difference(self.join_on))

        return self.with_columns(with_expr=nw.col(columns) * factor)

    def pipe(self, function: Callable, *args, **kwargs) -> StatCalculator:
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
        StatCalculator

        """

        self = self.copy()
        self.replicate_stats = self.replicate_stats.pipe(
            function=function, *args, **kwargs
        )

        return self

    # def reshape_groups_wide_long(self,
    #                              copy:bool=False,
    #                              group_first:bool=True,
    #                              group_col:str="Group",
    #                              invert_group:bool=False) -> StatCalculator:
    #     if copy:
    #         self = self.copy()

    #     def _reshape(df:pl.DataFrame | pl.LazyFrame,
    #                  join_on:list[str]) -> pl.DataFrame | pl.LazyFrame:
    #         if "___replicate___" in df.columns :
    #             join_on = join_on + ["___replicate___"]

    #         df_concat = []
    #         for coli in df.columns:
    #             if coli not in join_on:
    #                 coli_group = coli.split(":")[0]
    #                 coli_value = coli.split(":")[1]
    #                 c_name = pl.col(join_on[0])
    #                 rename = {coli:coli_value}

    #                 if group_first:
    #                     with_name = pl.concat_str([pl.lit(coli_group),
    #                                                pl.lit(":"),
    #                                                c_name]).alias(c_name.meta.output_name())
    #                 else:
    #                     with_name = pl.concat_str([c_name,
    #                                                pl.lit(":"),
    #                                                pl.lit(coli_group)]).alias(c_name.meta.output_name())

    #                 with_name = with_name.alias(c_name.meta.output_name())
    #                 if group_col != "":
    #                     if invert_group:
    #                         with_name = [with_name,
    #                                      coli_value.alias(group_col)]
    #                     else:
    #                         with_name = [with_name,
    #                                      pl.lit(coli_group).alias(group_col)]

    #                 df_concat.append((df.select(join_on + [coli])
    #                                     .rename(rename)
    #                                     .with_columns(with_name)))

    #         return pl.concat(df_concat,
    #                          how="vertical_relaxed")

    #     self = self.pipe(_reshape,
    #                      join_on=self.variable_ids)

    #     return self
    def concat_with(
        self, sc_concat: StatCalculator, how: str = "horizontal"
    ) -> StatCalculator:
        """
        Concatenate this with another StatCalculator object

        Parameters
        ----------
        sc_concat : StatCalculator
            Other mi object to concatenate with.
        how : str, optional
            horizontal or vertical?
            Horizontal will actually do a join and vertical will just stack them
            The default is "horizontal".

        Returns
        -------
        StatCalculator

        """

        self = self.copy()

        self.replicate_stats.concat_with(
            rs_concat=sc_concat.replicate_stats,
            join_on_self=self.variable_ids,
            join_on_concat=sc_concat.variable_ids,
        )

        return self

    def drb_round_table(
        self,
        columns: list | str | None = None,
        columns_n: list | str | None = None,
        columns_exclude: list | str | None = None,
        round_all: bool = True,
        digits: int = 4,
        compress: bool = False,
    ) -> StatCalculator:
        """
        Apply DRB (Disclosure Review Board) rounding rules to the estimates.

        Parameters
        ----------
        columns : list|str|None, optional
            Specific columns to round. Default is None.
        columns_n : list|str|None, optional
            Columns to treat as counts for rounding. Default is None.
        columns_exclude : list|str|None, optional
            Columns to exclude from rounding. Default is None.
        round_all : bool, optional
            Apply rounding to all numeric columns. Default is True.
        digits : int, optional
            Number of significant digits for rounding. Default is 4.
        compress : bool, optional
            Use compressed rounding format. Default is False.

        Returns
        -------
        StatCalculator
            StatCalculator with DRB rounding applied.
        """
        kwargs = copy(locals())
        del kwargs["self"]
        self.replicate_stats = self.replicate_stats.pipe(
            function=drb_round_table, **kwargs
        )

        return self

    #####################################################
    #   Serializable - BEGIN
    #####################################################
    @classmethod
    def _init_from_dict(cls, data: dict):
        return super()._init_from_dict(data, calculate=False)

    #####################################################
    #   Serializable - END
    #####################################################
