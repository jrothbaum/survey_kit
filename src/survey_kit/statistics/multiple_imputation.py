from __future__ import annotations
from typing import Optional, Callable

import os
import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT
import polars as pl
import polars.selectors as pl_cs
from pathlib import Path
import scipy
from copy import deepcopy, copy


import tempfile
from scipy import stats as scipy_stats


from ..utilities.inputs import list_input
from ..utilities.dataframe import (
    concat_wrapper,
    join_list,
    join_wrapper,
    NarwhalsType,
    fill_missing,
    columns_from_list,
    _columns_original_order,
    safe_columns,
)

from ..utilities.rounding import drb_round_table
from .rounding import Rounding
from .calculator import StatCalculator, print_se_table
from .replicates import ReplicateStats, apply_as_attribute
from .comparisons import ComparisonItem
import survey_kit.statistics.comparisons as kit_comparisons
from ..imputation.srmi import SRMI
from ..serializable import Serializable, SerializableDictionary
from ..utilities.dataframe_list import DataFrameList

from ..orchestration.utilities import CallInputs, CallTypes
from ..orchestration.from_python import FunctionFromPython
from ..orchestration.callers import run_function_list

from .. import logger, config


class MultipleImputation(Serializable):
    """
    A class for managing multiple imputation standard errors.  You should generally estimate
    the statistics with [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function] and then use this class to
    do comparisons, print, or gather results from the dataframes in it.

    MultipleImputation combines estimates from multiple imputed datasets to produce
    final estimates with proper standard errors that account for both within-imputation
    and between-imputation variance. It implements Rubin's rules for multiple imputation
    inference, including degrees of freedom calculations and missing information rates.

    The class supports:
    - Combining estimates from multiple imputed datasets
    - Calculating proper MI standard errors using Rubin's rules
    - Computing degrees of freedom for t-tests with MI data
    - Calculating rates of missing information
    - Statistical comparisons between MI estimate sets
    - Confidence intervals and p-values adjusted for MI uncertainty
    - Integration with replicate weight variance estimation

    Parameters
    ----------
    implicate_stats : list[ReplicateStats] | None, optional
        List of ReplicateStats objects, one for each imputed dataset.
        Each contains estimates and potentially replicate weight SEs. Default is None.
    join_on : list | None, optional
        Column names that identify unique estimates across implicates.
        Used for combining estimates. Default is None (empty list).
    df_estimates : IntoFrameT | None, optional
        Combined estimates dataframe (calculated automatically). Default is None.
    df_ses : IntoFrameT | None, optional
        MI standard errors dataframe (calculated automatically). Default is None.
    df_df : IntoFrameT | None, optional
        Degrees of freedom for each estimate (calculated automatically). Default is None.
    df_t : IntoFrameT | None, optional
        T-statistics dataframe (calculated automatically). Default is None.
    df_p : IntoFrameT | None, optional
        P-values dataframe (calculated automatically). Default is None.
    df_rate_of_missing_information : IntoFrameT | None, optional
        Rate of missing information for each estimate. Default is None.
    rounding : Rounding | None, optional
        Rounding configuration for display and output. Default is None.

    Attributes
    ----------
    n_implicates : int
        Number of imputed datasets used in the analysis.
    summarize_vars : list
        List of all estimate column names (excluding join variables).

    Examples
    --------
    Creating MI estimates from a function applied to multiple implicates:

    >>> from survey_kit.statistics.multiple_imputation import mi_ses_from_function
    >>> from survey_kit.statistics.calculator import StatCalculator
    >>> from survey_kit.statistics.statistics import Statistics
    >>> from survey_kit.imputation.srmi import SRMI
    >>>
    >>> # Define statistics to calculate
    >>> stats = Statistics(stats=["mean"], columns=["var_*"])
    >>> replicates = Replicates(weight_stub="weight_", n_replicates=20)
    >>>
    >>> # Arguments for the function to run on each implicate
    >>> arguments = {
    ...     "statistics": stats,
    ...     "replicates": replicates,
    ...     "round_output": False,
    ...     "display": False
    ... }
    >>> # Load SRMI data
    >>> srmi = SRMI.load(path_model="/projects/data/NEWS/Test/py_srmi_test.srmi/",
    ...                  LazyLoad=False)
    >>>
    >>> # Calculate MI estimates
    >>> mi_results = mi_ses_from_function(
    ...     delegate=StatCalculator,
    ...     df_implicates=srmi.df_implicates,
    ...     df_noimputes=weights_data,
    ...     arguments=arguments,
    ...     join_on=["Variable"]
    ... )
    >>> mi_results.print(round_output=True)

    Comparing two sets of MI estimates:

    >>> mi_means = mi_ses_from_function(...)  # Calculate means
    >>> mi_medians = mi_ses_from_function(...)  # Calculate medians
    >>>
    >>> comparison = mi_means.compare(
    ...     mi_medians,
    ...     compare_list_columns=[("mean", "median")]
    ... )
    >>> comparison["difference"].print()
    >>> comparison["ratio"].print()

    Examining MI-specific diagnostics:

    >>> print("Degrees of freedom:")
    >>> print(mi_results.df_df)
    >>> print("Rate of missing information:")
    >>> print(mi_results.df_rate_of_missing_information)

    Notes
    -----
    The class implements Rubin's (1987) rules for multiple imputation:
    - Combined estimate is the average across implicates
    - Total variance = within-imputation variance + between-imputation variance
    - Degrees of freedom account for finite number of implicates
    - Missing information rate quantifies uncertainty due to imputation

    """

    _save_suffix = "mi"

    def __init__(
        self,
        implicate_stats: list[ReplicateStats] | None = None,
        join_on: list | None = None,
        df_estimates: IntoFrameT | None = None,
        df_ses: IntoFrameT | None = None,
        df_df: IntoFrameT | None = None,
        df_t: IntoFrameT | None = None,
        df_p: IntoFrameT | None = None,
        df_rate_of_missing_information: IntoFrameT | None = None,
        rounding: Rounding | None = None,
    ):
        self.implicate_stats = implicate_stats

        if join_on is None:
            join_on = []
        self.join_on = join_on
        self.df_estimates = df_estimates
        self.df_ses = df_ses
        self.df_df = df_df
        self.df_t = df_t
        self.df_p = df_p
        self.df_rate_of_missing_information = df_rate_of_missing_information
        self.rounding = rounding

    def copy(self) -> MultipleImputation:
        return MultipleImputation(
            implicate_stats=[impi.copy() for impi in self.implicate_stats],
            join_on=copy(self.join_on),
            df_estimates=self.df_estimates,
            df_ses=self.df_ses,
            df_df=self.df_df,
            df_t=self.df_t,
            df_p=self.df_p,
            df_rate_of_missing_information=self.df_rate_of_missing_information,
            rounding=copy(self.rounding),
        )

    def calculate(self, implicate_name="___implicate___"):
        """
        Calculate multiple imputation estimates using Rubin's rules.

        Combines estimates from individual implicates to produce final MI estimates,
        standard errors, degrees of freedom, t-statistics, p-values, and rates of
        missing information. Implements the standard MI combining formulas.

        Parameters
        ----------
        implicate_name : str, optional
            Column name identifier for imputation number. Default is "___implicate___".

        Notes
        -----
        Implements Rubin's combining rules:
        - Q̄ = (1/m) Σ Q̂ᵢ  (combined estimate)
        - U = (1/m) Σ Uᵢ    (within-imputation variance)
        - B = (1/(m-1)) Σ (Q̂ᵢ - Q̄)²  (between-imputation variance)
        - T = U + (1 + 1/m)B  (total variance)
        - ν = (m-1)[1 + mU/((m+1)B)]²  (degrees of freedom)

        where m is the number of implicates.
        """

        df_estimates_stacked = []
        df_ses_stacked = []

        col_sort = "___mi_sort___"
        for indexi, imp_statsi in enumerate(self.implicate_stats):
            if indexi == 0:
                df_sort = (
                    nw.from_native(imp_statsi.df_estimates)
                    .select(self.join_on)
                    .lazy()
                    .collect()
                    .with_row_index(col_sort)
                    .lazy()
                    .to_native()
                )
            df_estimates_stacked.append(imp_statsi.df_estimates)
            df_ses_stacked.append(imp_statsi.df_ses)

        df_estimates_stacked = (
            nw.from_native(
                fill_missing(
                    concat_wrapper(df_estimates_stacked, how="diagonal"),
                    value=float("nan"),
                )
            )
            .lazy()
            .collect()
            .to_native()
        )
        df_ses_stacked = (
            nw.from_native(concat_wrapper(df_ses_stacked, how="diagonal"))
            .with_columns(cs.boolean().cast(nw.Int8))
            .lazy()
            .collect()
            .to_native()
        )

        cols_non_stats = self.join_on + [implicate_name, col_sort]
        cols_stats = safe_columns(df_estimates_stacked)
        cols_stats = _columns_original_order(
            columns_unordered=list(set(cols_stats).difference(cols_non_stats)),
            columns_ordered=cols_stats,
        )

        #   Notation from Joe Schaffer MI FAQ page (downloaded from the Wayback Machine)
        #       NEWS/Documentation/Background/MultipleImputation/MI_FAQ.htm
        df_estimates = (
            nw.from_native(df_estimates_stacked)
            .group_by(self.join_on)
            .agg(nw.all().mean())
            .sort(self.join_on)
        )

        c_stats = nw.col(cols_stats)
        with_between = []
        for coli in cols_stats:
            c_col = nw.col(coli)
            c_bar = nw.col(f"{coli}_bar")
            with_between.append(
                (1 / (self.n_implicates - 1) * (c_col - c_bar) ** 2).alias(coli)
            )

        df_B = (
            nw.from_native(
                join_list(
                    [
                        df_estimates_stacked,
                        (
                            nw.from_native(df_estimates)
                            .rename({coli: f"{coli}_bar" for coli in cols_stats})
                            .to_native()
                        ),
                    ],
                    on=self.join_on,
                    how="left",
                )
            )
            .with_columns(with_between)
            .group_by(self.join_on)
            .agg(c_stats.sum())
            .to_native()
        )

        df_U = (
            nw.from_native(df_ses_stacked)
            .with_columns(c_stats**2)
            .group_by(self.join_on)
            .agg(nw.all().mean())
            .to_native()
        )

        df_variance = (
            nw.from_native(
                concat_wrapper(
                    [
                        nw.from_native(df_B)
                        .with_columns(c_stats * (1 + 1 / self.n_implicates))
                        .to_native(),
                        df_U,
                    ],
                    how="diagonal",
                )
            )
            .group_by(self.join_on)
            .agg(nw.all().sum())
            .to_native()
        )

        self.df_ses = nw.from_native(df_variance).with_columns(c_stats**0.5).to_native()

        self.df_estimates = (
            nw.from_native(df_estimates_stacked)
            .group_by(self.join_on)
            .agg(nw.all().mean())
            .sort(self.join_on)
            .to_native()
        )

        df_B_U = join_list(
            [
                (
                    nw.from_native(df_B)
                    .rename({coli: f"{coli}_B" for coli in cols_stats})
                    .to_native()
                ),
                (
                    nw.from_native(df_U)
                    .rename({coli: f"{coli}_U" for coli in cols_stats})
                    .to_native()
                ),
            ],
            on=self.join_on,
            how="left",
        )

        m = self.n_implicates
        with_df = []
        for coli in cols_stats:
            c_U = nw.col(f"{coli}_U")
            c_B = nw.col(f"{coli}_B")

            with_df.append(
                ((m - 1) * (1 + (m * c_U) / ((m + 1) * c_B)) ** 2).alias(coli)
            )

        self.df_df = nw.from_native(df_B_U).select(self.join_on + with_df).to_native()

        with_r = []
        with_gamma = []
        for coli in cols_stats:
            c_U = nw.col(f"{coli}_U")
            c_B = nw.col(f"{coli}_B")
            c_r = nw.col(f"{coli}_r")
            c_df = nw.col(f"{coli}_df")
            with_r.append(((1 + 1 / m) * c_B / c_U).alias(f"{coli}_r"))

            with_gamma.append(((c_r + (2 / (c_df + 3))) / (c_r + 1)).alias(coli))

        self.df_rate_of_missing_information = (
            nw.from_native(
                join_list(
                    [
                        df_B_U,
                        (
                            nw.from_native(self.df_df)
                            .rename({coli: f"{coli}_df" for coli in cols_stats})
                            .to_native()
                        ),
                    ],
                    how="left",
                    on=self.join_on,
                )
            )
            .with_columns(with_r)
            .select(self.join_on + with_gamma)
            .to_native()
        )

        with_t = []
        for coli in cols_stats:
            c_est = nw.col(coli)
            c_se = nw.col(f"{coli}_se")

            with_t.append((c_est / c_se).abs().alias(coli))

        self.df_t = (
            nw.from_native(
                join_list(
                    [
                        self.df_estimates,
                        (
                            nw.from_native(self.df_ses)
                            .rename({coli: f"{coli}_se" for coli in cols_stats})
                            .to_native()
                        ),
                    ],
                    on=self.join_on,
                    how="left",
                )
            )
            .select(self.join_on + with_t)
            .to_native()
        )

        df_p = join_list(
            [
                (
                    nw.from_native(self.df_df)
                    .rename({coli: f"{coli}_df" for coli in cols_stats})
                    .to_native()
                ),
                (
                    nw.from_native(self.df_t)
                    .rename({coli: f"{coli}_t" for coli in cols_stats})
                    .to_native()
                ),
            ],
            how="left",
            on=self.join_on,
        )

        def p_value(t, df):
            if t == float("inf") or t == float("nan"):
                return 0
            if df == float("inf") or df == float("nan"):
                df = 1_000_000

            return scipy.stats.t.sf(t, df) * 2

        def p_value_lambda(t, col_t, col_df):
            try:
                return p_value(t[col_t], t[col_df])
            except:
                return None

        nw_type = NarwhalsType(df_p)
        df_p = nw_type.to_polars().lazy().collect()
        for coli in cols_stats:
            index_t = df_p.columns.index(f"{coli}_t")
            index_df = df_p.columns.index(f"{coli}_df")

            df_p = df_p.with_columns(
                df_p.map_rows(
                    lambda t: p_value_lambda(t, index_t, index_df),
                    return_dtype=pl.Float64,
                )
            ).rename({"map": coli})

        self.df_p = nw_type.from_polars(df_p.select(self.join_on + cols_stats))

        def _sort_and_fill_null(df: IntoFrameT) -> IntoFrameT:
            nw_type = NarwhalsType(
                join_list([df, df_sort], how="left", on=self.join_on)
            )
            return nw_type.from_polars(
                nw_type.to_polars()
                .sort([col_sort], maintain_order=True)
                .drop(col_sort)
                .fill_nan(None)
            )

        self.df_estimates = _sort_and_fill_null(self.df_estimates)
        self.df_ses = _sort_and_fill_null(self.df_ses)
        self.df_df = _sort_and_fill_null(self.df_df)
        self.df_t = _sort_and_fill_null(self.df_t)
        self.df_p = _sort_and_fill_null(df_p)
        self.df_rate_of_missing_information = _sort_and_fill_null(
            self.df_rate_of_missing_information
        )

    def compare(
        self,
        other: ReplicateStats | MultipleImputation | StatCalculator,
        difference: bool = True,
        ratio: bool = True,
        ratio_minus_1: bool = True,
        replicate_name: str = "___replicate___",
        compare_list_variables: list[ComparisonItem.Variable] | None = None,
        compare_list_columns: list[ComparisonItem.Column] | None = None,
    ) -> dict[str, MultipleImputation]:
        """
        Compare this set of MI estimates to another set of estimates.

        Parameters
        ----------
        other : MultipleImputation | ReplicateStats | StatCalculator
            The other object to compare against.
        difference : bool, optional
            Calculate and return differences. Default is True.
        ratio : bool, optional
            Calculate and return ratios. Default is True.
        ratio_minus_1 : bool, optional
            Subtract 1 from ratios (for percentage change interpretation).
            Default is True.
        replicate_name : str, optional
            Column name for replicate identifier. Default is "___replicate___".
        compare_list_variables : list[ComparisonItem.Variable] | None, optional
            List of variable info to compare.
        compare_list_columns : list[ComparisonItem.Columns] | None, optional
            List of column info to compare.

        Returns
        -------
        dict[str, MultipleImputation]
            Dictionary with keys ["difference", "ratio"] containing
            MultipleImputation objects with comparison results and proper
            MI standard errors for the comparisons.

        Examples
        --------
        Compare mi_news to sc_survey (StatCalculator)
        >>> comparison = mi_news.compare(
        ...     sc_survey
        ... )

        Compare means vs medians:
        >>> comparison = mi_means.compare(
        ...     mi_medians,
        ...     compare_list_columns=compare_list_variables=[
        ...     ComparisonItem.Variable(
        ...         "mean",
        ...         "median",
        ...         name="median_mean"
        ...     )]
        ... )
        >>> comparison["difference"].print()

        Compare specific variables:

        >>> comparison = mi_results1.compare(
        ...     mi_results2,
        ...     compare_list_variables=[ComparisonItem.Variable(
        ...         value1="income",
        ...         value2="income_2",
        ...         name="income_comp"
        ...     )]
        ... )
        """

        return kit_comparisons.compare(
            stats1=self,
            stats2=other,
            join_on=self.join_on,
            rounding=self.rounding,
            difference=difference,
            ratio=ratio,
            ratio_minus_1=ratio_minus_1,
            replicate_name=replicate_name,
            compare_list_variables=compare_list_variables,
            compare_list_columns=compare_list_columns,
        )

    def round_results(
        self,
        df: IntoFrameT = None,
        rounding: Rounding | None = None,
        display_only: bool = False,
    ) -> IntoFrameT:
        """
        Apply rounding rules to MI estimates.

        Parameters
        ----------
        df : IntoFrameT, optional
            Table of estimates to round. Default is df_estimates.
        rounding : Rounding|None, optional
            Rounding configuration (True for DRB rules, int for significant digits).
            Default is self.rounding.
        display_only : bool, optional
            If True, converts numbers to strings for display only. Default is False.

        Returns
        -------
        IntoFrameT
            The dataframe with rounding applied.
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

    def print(self, round_output: bool | int | None = None, sub_log: logging = None):
        """
        Print the MI estimates and standard errors to the log.

        Parameters
        ----------
        round_output : bool|int|None, optional
            Rounding rule (True for DRB, int for significant digits).
            Default is self.rounding configuration.
        sub_log : logging, optional
            Alternative logger to use. Default is None (use standard logging).

        Examples
        --------
        >>> mi_results.print(round_output=True)  # Use DRB rounding
        >>> mi_results.print(round_output=3)     # 3 significant digits
        """

        if sub_log is None:
            sub_log = logger

        #   Round?
        if round_output:
            if self.rounding is None:
                rounding = Rounding()
            else:
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
            # display_all_vars=self.display_all_vars,
            # display_max_vars=self.display_max_vars,
            sort_vars=self.join_on,
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
        Create a formatted table combining different types of estimates.

        Reshapes estimates into a long format table with different statistic
        types (estimates, SEs, t-stats, p-values, etc.) as separate rows.

        Parameters
        ----------
        round_output : bool|int|None, optional
            Rounding rule for display.
        estimates_to_show : list[str] | None, optional
            List of estimate types to include. Options:
            - "estimate": Point estimates
            - "se": Standard errors
            - "t": T-statistics
            - "p": P-values
            - "ci": Confidence intervals
            - "df": Degrees of freedom
            Default is ["estimate", "se"].
        variable_prefix : str, optional
            Prefix to add to variable column names. Default is "".
        estimate_type_variable_name : str, optional
            Name for column indicating statistic type. Default is "Statistic".
        ci_level : float, optional
            Confidence level for confidence intervals. Default is 0.95.

        Returns
        -------
        IntoFrameT
            Formatted table with estimates arranged by statistic type.

        Examples
        --------
        >>> table = mi_results.table_of_estimates(
        ...     estimates_to_show=["estimate", "se", "p", "ci"],
        ...     round_output=True
        ... )
        """
        if estimates_to_show is None:
            estimates_to_show = ["estimate", "se"]

        df_ordered = []
        col_sort = "__order_output_table__"
        for index, esti in enumerate(estimates_to_show):
            if esti.lower() == "estimate":
                df_ordered.append(
                    nw.from_native(self.df_estimates)
                    .with_columns(
                        [
                            nw.lit(index).alias(col_sort),
                            nw.lit(esti.lower()).alias(estimate_type_variable_name),
                        ]
                    )
                    .to_native()
                )
            elif esti.lower() == "se":
                df_ordered.append(
                    nw.from_native(self.df_ses)
                    .with_columns(
                        [
                            nw.lit(index).alias(col_sort),
                            nw.lit(esti.lower()).alias(estimate_type_variable_name),
                        ]
                    )
                    .to_native()
                )
            elif esti.lower() == "t":
                df_ordered.append(
                    nw.from_native(self.df_t)
                    .with_columns(
                        [
                            nw.lit(index).alias(col_sort),
                            nw.lit(esti.lower()).alias(estimate_type_variable_name),
                        ]
                    )
                    .to_native()
                )
            elif esti.lower() == "p":
                cols_to_keep = safe_columns(
                    nw.from_native(self.df_estimates)
                    .lazy()
                    .with_columns(
                        [
                            nw.lit(index).alias(col_sort),
                            nw.lit(esti.lower()).alias(estimate_type_variable_name),
                        ]
                    )
                    .to_native()
                )

                df_p = (
                    nw.from_native(self.df_p)
                    .with_columns(
                        [
                            nw.lit(index).alias(col_sort),
                            nw.lit(esti.lower()).alias(estimate_type_variable_name),
                        ]
                    )
                    .select(cols_to_keep)
                    .to_native()
                )

                when_then_censor_absurd_values = []
                absurd_value_threshold = 1e-10
                #   I don't want to see p-values below a ridiculously low number
                for coli in self.df_estimates.columns:
                    if coli not in self.join_on:
                        ci = nw.col(coli)
                        when_then_censor_absurd_values.append(
                            (
                                (
                                    nw.when(ci.lt(absurd_value_threshold))
                                    .then(nw.lit(0.0))
                                    .otherwise(ci)
                                ).alias(coli)
                            )
                        )

                df_ordered.append(
                    nw.from_native(df_p)
                    .with_columns(when_then_censor_absurd_values)
                    .to_native()
                )

            elif esti.lower() == "ci":
                df_ordered.append(
                    nw.from_native(self._df_ci(ci_level=ci_level))
                    .with_columns(
                        [
                            nw.lit(index).alias(col_sort),
                            nw.lit(esti.lower()).alias(estimate_type_variable_name),
                        ]
                    )
                    .to_native()
                )
            elif esti.lower() == "df":
                with_infinites = []
                for coli in self.summarize_vars:
                    c_coli = nw.col(coli)

                    upper_limit = 10**7
                    with_infinites.append(
                        (
                            nw.when(c_coli.gt(upper_limit))
                            .then(nw.lit(upper_limit))
                            .otherwise(c_coli)
                            .alias(coli)
                        )
                    )

                df_ordered.append(
                    nw.from_native(self.df_df)
                    .with_columns(with_infinites)
                    .with_columns(
                        [
                            nw.lit(index).alias(col_sort),
                            nw.lit(esti.lower()).alias(estimate_type_variable_name),
                        ]
                    )
                )
            else:
                message = f"{esti} not allowed for estimates_to_show"
                logger.error(message)
                raise Exception(message)

        col_row_index = "___estimate_row_count___"
        df_ordered[0] = (
            nw.from_native(df_ordered[0]).with_row_index(col_row_index).to_native()
        )

        df_display = concat_wrapper(df_ordered, how="diagonal")

        nw_type = NarwhalsType(df_display)
        df_display = nw_type.to_polars()

        sort_vars = self.join_on
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
        for coli in df_display.columns:
            if coli not in select_order and coli != col_sort:
                if variable_prefix != "":
                    rename[coli] = f"{variable_prefix}{coli}"

                remaining.append(coli)

        df_display = df_display.with_columns(with_clear).select(
            select_order + remaining
        )
        #   Round?
        if round_output:
            if self.rounding is not None:
                rounding = deepcopy(self.rounding)
            else:
                rounding = Rounding()
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
        return nw_type.from_polars(df_display)

    def _df_ci(self, ci_level: float = 0.95):
        #   Use scipy to get the t-stat ci multiple
        def to_ci_multiple(value) -> float:
            return scipy_stats.t.ppf(1 - (1 - ci_level) / 2, value)

        nw_type = NarwhalsType(self.df_df)
        dof = (
            nw_type.to_polars.lazy()
            .collect()
            .with_columns(
                pl_cs.numeric().map_elements(to_ci_multiple, return_dtype=pl.Float64)
            )
        )

        #   Use numpy to multiply the se by the t-stat ci multiple
        se_np = (
            nw.from_native(self.df_ses).select(cs.numeric()).lazy().collect().to_numpy()
        )
        dof_np = nw.from_native(dof).select(cs.numeric()).lazy().collect().to_numpy()

        #   return the data
        dof = pl.concat(
            [
                dof.select(self.join_on),
                pl.from_numpy(
                    se_np * dof_np,
                    schema={
                        coli: pl.Float64
                        for coli in safe_columns(dof.select(cs.numeric()))
                    },
                ),
            ],
            how="horizontal",
        )

        return nw_type.from_polars(dof)

    def filter(self, filter_expr: nw.Expr) -> MultipleImputation:
        #   Don't edit the underlying object
        self = self.copy()

        for dfi_name in self._df_attributes:
            apply_as_attribute(
                obj=self, df_name=dfi_name, nw_expr=filter_expr, nw_method="filter"
            )

        for repi in range(0, len(self.implicate_stats)):
            self.implicate_stats[repi].filter(filter_expr)

        return self

    def select(
        self, select_expr: nw.Expr | str | list[str] | list[nw.Expr]
    ) -> MultipleImputation:
        self = self.copy()
        select_expr = list_input(select_expr)
        cols_keep = columns_from_list(self.df_estimates, columns=select_expr)

        select_df_p = []
        for coli in cols_keep:
            cols_check = [f"{coli}_df", f"{coli}_t"]

            for checki in cols_check:
                if checki in safe_columns(self.df_p):
                    select_df_p.append(checki)

        add_join_on = list(set(self.join_on).difference(cols_keep))
        cols_keep = add_join_on + cols_keep

        for dfi in self._df_attributes:
            if dfi == "df_p":
                apply_as_attribute(
                    obj=self,
                    df_name=dfi,
                    nw_expr=cols_keep + select_df_p,
                    nw_method="select",
                )
            else:
                apply_as_attribute(
                    obj=self, df_name=dfi, nw_expr=cols_keep, nw_method="select"
                )

            for repi in range(0, len(self.implicate_stats)):
                self.implicate_stats[repi].select(cols_keep)
        return self

    def with_columns(self, with_expr: nw.Expr | list[nw.Expr]) -> MultipleImputation:
        self = self.copy()

        for dfi in self._df_attributes:
            apply_as_attribute(
                obj=self, df_name=dfi, nw_expr=with_expr, nw_method="with_columns"
            )

        for repi in range(0, len(self.implicate_stats)):
            self.implicate_stats[repi].with_columns(with_expr)
        return self

    def rename(self, d_rename: dict[str, str]) -> MultipleImputation:
        self = self.copy()

        for dfi in self._df_attributes:
            apply_as_attribute(
                obj=self, df_name=dfi, nw_expr=d_rename, nw_method="rename"
            )

        for repi in range(0, len(self.implicate_stats)):
            self.implicate_stats[repi].rename(d_rename)

        return self

    def scale_by(
        self, factor: float, columns: list[str] | str | None = None
    ) -> MultipleImputation:
        if columns is None:
            #   Any columns that aren't the join_on ones
            columns = list(
                set(safe_columns(self.df_estimates)).difference(self.join_on)
            )

        return self.with_columns(with_expr=nw.col(columns) * factor)

    def pipe(self, function: Callable, *args, **kwargs) -> MultipleImputation:
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

        self = self.copy()

        for dfi_name in self._df_attributes:
            dfi = getattr(self, dfi_name)
            if dfi is not None:
                setattr(
                    self,
                    dfi_name,
                    nw.to_native(function(nw.from_native(dfi), *args, **kwargs)),
                )

        for repi in range(0, len(self.implicate_stats)):
            self.implicate_stats[repi].pipe(function, *args, **kwargs)

        return self

    def concat_with(
        self, mi_concat: MultipleImputation, how: str = "horizontal"
    ) -> MultipleImputation:
        """
        Concatenate this with another mi object

        Parameters
        ----------
        mi_concat : MultipleImputation
            Other mi object to concatenate with.
        how : str, optional
            horizontal or vertical?
            Horizontal will actually do a join and vertical will just stack them
            The default is "horizontal".

        Returns
        -------
        MultipleImputation

        """

        self = self.copy()

        def _concat_df(df: IntoFrameT, df_join: IntoFrameT) -> IntoFrameT:
            if how == "horizontal":
                return join_wrapper(
                    df=df,
                    df_to=df_join,
                    how="left",
                    left_on=self.join_on,
                    right_on=mi_concat.join_on,
                )
            elif how == "vertical":
                return concat_wrapper([df, df_join], how="horizontal")

        for dfi in self._df_attributes:
            setattr(
                self,
                dfi,
                _concat_df(df=getattr(self, dfi), df_join=getattr(mi_concat, dfi)),
            )

        for repi in range(0, len(self.implicate_stats)):
            self.implicate_stats[repi].concat_with(
                rs_concat=mi_concat.implicate_stats[repi],
                join_on_self=self.join_on,
                join_on_concat=mi_concat.join_on,
            )

        return self

    def sort(
        self, sort_expr: nw.Expr | list[nw.Expr] | str | list[str]
    ) -> MultipleImputation:
        self = self.copy()
        for dfi in self._df_attributes:
            apply_as_attribute(
                obj=self, df_name=dfi, nw_expr=sort_expr, nw_method="sort"
            )

        for repi in range(0, len(self.implicate_stats)):
            self.implicate_stats[repi].sort(sort_expr)

        return self

    def drop(
        self, drop_expr: nw.Expr | list[nw.Expr] | str | list[str]
    ) -> MultipleImputation:
        self = self.copy()
        for dfi in self._df_attributes:
            apply_as_attribute(
                obj=self, df_name=dfi, nw_expr=drop_expr, nw_method="drop"
            )

        for repi in range(0, len(self.implicate_stats)):
            self.implicate_stats[repi].drop(drop_expr)

        return self

    # def reshape_groups_wide_long(self,
    #                              copy:bool=False,
    #                              group_first:bool=True,
    #                              group_col:str="Group",
    #                              invert_group:bool=False) -> MultipleImputation:
    #     if copy:
    #         self = self.copy()

    #     def _reshape(df:IntoFrameT,
    #                  join_on:list[str]) -> IntoFrameT:
    #         if "___replicate___" in df.columns:
    #             join_on = join_on + ["___replicate___"]

    #         df_concat_by_groups = {}
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
    #                                      c_name.alias(group_col)]
    #                     else:
    #                         with_name = [with_name,
    #                                      pl.lit(coli_group).alias(group_col)]

    #                 dfi = (df.select(join_on + [coli])
    #                          .rename(rename)
    #                          .with_columns(with_name))
    #                 if coli_group in df_concat_by_groups.keys():
    #                     df_concat_by_groups[coli_group].append(dfi)
    #                 else:
    #                     df_concat_by_groups[coli_group] = [dfi]
    #                 # df_concat.append((df.select(join_on + [coli])
    #                 #                     .rename(rename)
    #                 #                     .with_columns(with_name)))

    #         df_concat = []
    #         for keyi, itemi in df_concat_by_groups.items():
    #             if len(itemi) == 1:
    #                 df_concat.append(itemi)
    #             else:
    #                 if group_col != "":
    #                     join_on_group = join_on + [group_col]
    #                 else:
    #                     join_on_group = join_on

    #                 df_concat.append(JoinFileList_Simple(itemi,
    #                                                      Join="outer",
    #                                                      JoinOn=join_on_group))

    #         df_concat = pl.concat(df_concat,
    #                               how="diagonal_relaxed")

    #         return df_concat

    #     self = self.pipe(_reshape,
    #                      join_on=self.join_on)

    #     return self

    def drb_round_table(
        self,
        columns: list | str | None = None,
        columns_n: list | str | None = None,
        columns_exclude: list | str | None = None,
        round_all: bool = True,
        digits: int = 4,
        compress: bool = False,
    ) -> MultipleImputation:
        kwargs = copy(locals())
        del kwargs["self"]
        self.pipe(function=drb_round_table, **kwargs)

        return self

    @property
    def _df_attributes(self) -> list[str]:
        return [
            "df_estimates",
            "df_ses",
            "df_df",
            "df_t",
            "df_p",
            "df_rate_of_missing_information",
        ]

    @property
    def n_implicates(self) -> int:
        return len(self.implicate_stats)

    @property
    def summarize_vars(self) -> list:
        return self.implicate_stats[0].df_estimates.drop(self.join_on).columns

    @classmethod
    def concat(cls, mi_list: list[MultipleImputation]) -> MultipleImputation:
        from .comparisons import _append_to_mi

        mi_out = mi_list[0]

        for i in range(1, len(mi_list)):
            mi_out = _append_to_mi(
                name="",
                stat_item=mi_out,
                stats=mi_list[i],
                vertical=True,
                n_implicates=mi_out.n_implicates,
                vertical_drop_var_name=False,
                separator="",
            )

        return mi_out


def mi_ses_from_function(
    delegate: Callable,
    join_on: list,
    path_srmi: str = "",
    df_implicates: list[IntoFrameT] | DataFrameList | None = None,
    index: list | None = None,
    df_noimputes: IntoFrameT | None = None,
    arguments: dict | None = None,
    df_argument_name: str = "df",
    implicate_name: str = "___implicate___",
    parallel: bool = False,
    parallel_inputs: CallInputs | None = None,
    rounding: Rounding | None = None,
    round_output: bool = True,
) -> MultipleImputation:
    """
    Calculate multiple imputation standard errors by applying a function to each implicate.

    This is the main function for computing MI estimates. It applies a specified function
    (delegate) to each imputed dataset, then combines the results using Rubin's rules
    to produce final estimates with proper MI standard errors.

    Parameters
    ----------
    delegate : callable
        Function to apply to each imputed dataset. Should return estimates,
        standard errors, and optionally replicate estimates. Common choices:
        - StatCalculator class constructor
        - Custom analysis function
    path_srmi : str, optional
        The path to a saved SRMI serialized object
        The default is "".
        Must pass either path_srmi or df_implicates
    df_implicates : list[IntoFrameT] | DataFrameList, optional
        List of imputed datasets to analyze.
        The default is None
        Must pass either path_srmi or df_implicates
    join_on : list
        Column names that identify unique estimates across implicates.
        Used for combining results.
    index : list|None, optional
        Index columns for merging df_noimputes with implicates.
        If None, assumes row-wise concatenation (dangerous...). Default is None.
    df_noimputes : IntoFrameT | None, optional
        Non-imputed data (e.g., weights, design variables) to merge with
        each implicate. Default is None.
    arguments : dict|None, optional
        Additional arguments passed to the delegate function. Default is None.
    df_argument_name : str, optional
        Name of the dataframe argument in the delegate function. Default is "df".
    parallel : bool, optional
        Run analysis on implicates in parallel. Default is False.
    parallel_inputs : CallInputs | None, optional
        Configuration for parallel execution. Default is None.
    rounding : Rounding | None, optional
        Rounding configuration for results. Default is None.
    round_output : bool, optional
        Apply rounding to final output. Default is True.

    Returns
    -------
    MultipleImputation
        MultipleImputation object containing combined estimates, MI standard errors,
        degrees of freedom, and other MI diagnostics.

    Examples
    --------
    Basic usage with StatCalculator:

    >>> from NEWS.CodeUtilities.Python.Statistics.StatCalculator import StatCalculator
    >>> from NEWS.CodeUtilities.Python.SummaryStats import Statistics, Replicates
    >>> from NEWS.CodeUtilities.Python.SRMI.SRMI import SRMI
    >>>
    >>> # Load SRMI data
    >>> srmi = SRMI.load(path_model="/projects/data/NEWS/Test/py_srmi_test.srmi/",
    ...                  LazyLoad=False)
    >>>
    >>>
    >>> # Define what to calculate
    >>> stats = Statistics(stats=["mean", "median"], columns=["income", "age"])
    >>> replicates = Replicates(weight_stub="replicate_", n_replicates=80)
    >>>
    >>> # Arguments for StatCalculator
    >>> arguments = {
    ...     "statistics": stats,
    ...     "replicates": replicates,
    ...     "weight": "survey_weight",
    ...     "display": False
    ... }
    >>>
    >>> # Calculate MI estimates
    >>> mi_results = mi_ses_from_function(
    ...     delegate=StatCalculator,
    ...     df_implicates=srmi.df_implicates,
    ...     df_noimputes=weight_data,
    ...     arguments=arguments,
    ...     join_on=["Variable"]
    ... )
    >>>
    >>> # View results
    >>> mi_results.print(round_output=True)
    >>> print("Degrees of freedom:", mi_results.df_df)
    >>> print("Missing information rate:", mi_results.df_rate_of_missing_information)

    With a custom analysis function:

    >>> def custom_analysis(df, weight="", var="income"):
    ...     '''Custom function returning estimates'''
    ...     import polars as pl
    ...
    ...     if weight:
    ...         mean_est = (df[var] * df[weight]).sum() / df[weight].sum()
    ...     else:
    ...         mean_est = df[var].mean()
    ...
    ...     return pl.DataFrame({
    ...         "Variable": [var],
    ...         "estimate": [mean_est]
    ...     })
    >>>
    >>> mi_custom = mi_ses_from_function(
    ...     delegate=custom_analysis,
    ...     df_implicates=srmi.df_implicates,
    ...     arguments={"weight": "survey_weight", "var": "income"},
    ...     join_on=["Variable"]
    ... )

    Parallel processing for faster computation:

    >>> from NEWS.CodeUtilities.Python.Function.Utilities import CallInputs, CallTypes
    >>>
    >>> parallel_config = CallInputs(CallType=CallTypes.shell)
    >>> mi_results = mi_ses_from_function(
    ...     delegate=StatCalculator,
    ...     df_implicates=srmi.df_implicates,
    ...     arguments=arguments,
    ...     join_on=["Variable"],
    ...     parallel=True,
    ...     parallel_inputs=parallel_config
    ... )

    Notes
    -----
    The function implements the standard MI workflow:
    1. Apply delegate function to each imputed dataset
    2. Collect estimates and SEs from each implicate
    3. Combine using Rubin's rules for MI inference
    4. Calculate degrees of freedom and missing information rates

    For delegate functions that return tuples/lists, expects:
    - Item 0: estimates dataframe
    - Item 1: standard errors dataframe
    - Item 2: replicate estimates (optional)
    - Item 3: bootstrap flag (optional)

    See Also
    --------
    [`StatCalculator`][survey_kit.statistics.calculator.StatCalculator] : Main class for statistical calculations
    [`MultipleImputation`][survey_kit.statistics.multiple_imputation.MultipleImputation] : Class for MI results
    """

    #   Don't edit the arguments dictionary
    arguments = copy(arguments)
    if parallel:
        if path_srmi != "":
            n_implicates = SRMI.load(path_srmi).n_implicates
        else:
            n_implicates = len(df_implicates)
        if parallel_inputs is None:
            logger.info("Defaulting to shell for parallel estimation")
            parallel_inputs = CallInputs(
                call_type=CallTypes.shell, n_cpu=int(config.cpus / n_implicates)
            )

        del n_implicates
        arguments = locals().copy()

        implicate_stats = _mi_ses_from_function_parallel(**arguments)

    else:
        arguments = locals().copy()

        del arguments["parallel"]
        del arguments["parallel_inputs"]
        implicate_stats = _mi_ses_from_function_sequential(**arguments)

    if len(implicate_stats):
        mi_stats = MultipleImputation(
            implicate_stats=implicate_stats, join_on=join_on, rounding=rounding
        )
        mi_stats.calculate()

        return mi_stats

    else:
        return None


def _mi_ses_from_function_sequential(
    delegate,
    join_on: list,
    path_srmi: str = "",
    df_implicates: list[IntoFrameT] | DataFrameList | None = None,
    index: list | None = None,
    df_noimputes: IntoFrameT | None = None,
    arguments: dict | None = None,
    df_argument_name: str = "df",
    implicate_name: str = "___implicate___",
    rounding: Rounding | None = None,
    round_output: bool = True,
) -> list[ReplicateStats | StatCalculator]:
    if rounding is None:
        rounding = Rounding(round_output)

    arguments = locals().copy()

    implicate_stats = []

    if path_srmi != "":
        df_implicates = SRMI.load(path_srmi).df_implicates

    for implicate_number in range(0, len(df_implicates)):
        imp_statsi = _mi_ses_from_function_one_implicate(
            implicate_number=implicate_number, **arguments
        )

        if imp_statsi is not None:
            implicate_stats.append(imp_statsi)

    return implicate_stats


def _mi_ses_from_function_parallel(
    delegate,
    join_on: list,
    df_implicates: list[IntoFrameT] | DataFrameList | None = None,
    path_srmi: str = "",
    index: list | None = None,
    df_noimputes: IntoFrameT | None = None,
    arguments: dict | None = None,
    df_argument_name: str = "df",
    implicate_name: str = "___implicate___",
    rounding: Rounding | None = None,
    round_output: bool = True,
    parallel: bool = False,
    parallel_inputs: CallInputs | None = None,
) -> list[ReplicateStats | StatCalculator]:
    if rounding is None:
        rounding = Rounding(round_output)

    if path_srmi != "":
        df_implicates = None

    #   Convert dataframelist to list of dataframes for saving
    if type(df_implicates) is DataFrameList:
        df_implicates = df_implicates._df_list
    arguments = locals().copy()
    del arguments["parallel"]
    del arguments["parallel_inputs"]

    implicate_stats = []
    random_file_save = next(tempfile._get_candidate_names())

    path_temp_dict = Path(f"{config.path_temp_files}/{random_file_save}").as_posix()

    SerializableDictionary(arguments).save(path=path_temp_dict)

    function_list = []
    path_implicates = []

    if path_srmi != "":
        n_implicates = SRMI.load(path_srmi).n_implicates
    else:
        n_implicates = len(df_implicates)

    for implicate_number in range(n_implicates):
        path_temp_implicate = Path(
            f"{config.path_temp_files}/{random_file_save}_implicate_{implicate_number}"
        ).as_posix()

        f = FunctionFromPython(
            function=_mi_ses_from_function_one_implicate,
            parameters=dict(
                implicate_number=implicate_number, path_save=path_temp_implicate
            ),
        )

        f.parameters_positional_post = ["**d_implicate"]
        f.add_pre_function("from survey_kit.serializable import SerializableDictionary")
        f.add_pre_function(
            f"d_implicate = SerializableDictionary.load(path='{path_temp_dict}')"
        )

        path_implicates.append(path_temp_implicate)
        function_list.append(f)

    log = run_function_list(
        function_list, call_input=parallel_inputs, run_all=True, testing=False
    )

    #   Collect the results
    if path_srmi != "":
        n_implicates = SRMI.load(path_srmi).n_implicates
    else:
        n_implicates = len(df_implicates)

    for implicate_number in range(n_implicates):
        path_implicate = path_implicates[implicate_number]

        imp_statsi = Serializable.load(path_implicate, delete=True)

        if imp_statsi is not None:
            implicate_stats.append(imp_statsi)

    #   Clean up
    SerializableDictionary.delete(path=path_temp_dict)

    return implicate_stats


def _mi_ses_from_function_one_implicate(
    implicate_number: int,
    delegate,
    join_on: list,
    df_implicates: list[IntoFrameT] | DataFrameList | None = None,
    path_srmi: str = "",
    index: list | None = None,
    df_noimputes: IntoFrameT | None = None,
    arguments: dict | None = None,
    df_argument_name: str = "df",
    implicate_name: str = "___implicate___",
    rounding: Rounding | None = None,
    round_output: bool = True,
    path_save: str = "",
):
    if path_srmi != "":
        dfi = SRMI.load(path_srmi).df_implicates_by_index(implicate_number)
    else:
        dfi = df_implicates[implicate_number]

    logger.info(f"Implicate #{implicate_number + 1}")
    if df_noimputes is not None:
        if index is None:
            #   No merge index, assume merge by row
            dfi = concat_wrapper([dfi, df_noimputes], how="horizontal")
        else:
            dfi = join_list([dfi, df_noimputes], how="left", on=index)

    arguments[df_argument_name] = dfi
    out = delegate(**arguments)

    imp_statsi = None
    if type(out) is StatCalculator:
        if len(rounding.cols_n) == 0 and len(out.rounding.cols_n):
            rounding.cols_n = out.rounding.cols_n

        if len(rounding.cols_round) == 0 and len(out.rounding.cols_round):
            rounding.cols_round = out.rounding.cols_round
        if out.replicates is None:
            bootstrap = out.bootstrap
        else:
            bootstrap = out.replicates.bootstrap
        imp_statsi = ReplicateStats(
            df_estimates=out.df_estimates,
            df_ses=out.df_ses,
            df_replicates=out.df_replicates,
            bootstrap=bootstrap,
        )

    elif type(out) is list or type(out) is tuple:
        #   Expects item 0 to be estimates
        #           item 1 to be ses
        #           item 2 to be replicates (if there)
        #           item 4 to be bootstrap dummy
        df_estimates = out[0]
        df_ses = out[1]

        if len(out) >= 3:
            df_replicates = out[2]
        else:
            df_replicates = None

        if len(out) >= 4:
            bootstrap = out[3]
        else:
            bootstrap = False

        imp_statsi = ReplicateStats(
            df_estimates=df_estimates,
            df_ses=df_ses,
            df_replicates=df_replicates,
            bootstrap=bootstrap,
        )

    if path_save != "":
        imp_statsi.save(path_save)
    return imp_statsi


# if __name__ == "__main__":
#     def test():
#         from NEWS.CodeUtilities.Python.SRMI.SRMI import SRMI
#         from NEWS.CodeUtilities.Python.Random import RandomNumberGenerator,\
#                                                      SetSeed

#         # Some convenience function for calculating summary stats


#         SetSeed(98482224)
#         rng = RandomNumberGenerator()


#         srmi = SRMI.load(path_model="/projects/data/NEWS/Test/py_srmi_test.srmi/",
#                          LazyLoad=False)


#         df_implicates = srmi.df_implicates

#         n_rows = df_implicates[0].height
#         #   Make the "weights" (base + 10 replicates)
#         n_replicates = 2
#         df_weights = pl.DataFrame(
#                 {"weights":rng.uniform(low=0.5,high=1.5,size=(n_replicates + 1)*n_rows)}
#             )

#         rename = {f"field_{i}":f"weight_{i}" for i in range(0,n_replicates+1)}
#         df_weights = (df_weights.select(pl.col("weights").reshape((n_rows,n_replicates+1))
#                                                          .arr.to_struct())
#                                  .unnest("weights")
#                                  .rename(rename))

#         #   Multiply weight_0 by each other weight to get something more rep weight like
#         df_weights = df_weights.with_columns(
#                 [(pl.col(f"weight_{i}")*pl.col("weight_0")).alias(f"weight_{i}") for i in range(1,n_replicates+1)]
#             )


#         #   Calculate these statistics (stats) for these variables (columns)
#         stats = Statistics(stats=["mean"],
#                             columns=["var_*"])

#         # stats = Statistics(stats=["mean","median","q25|not0"],
#         #                    columns=["var_hd1"])
#         #   Tell it what the weights are
#         replicates = Replicates(weight_stub="weight_",
#                                 n_replicates=n_replicates,
#                                 bootstrap=False)


#         delegate = StatCalculator
#         arguments = {"statistics":stats,
#                      "replicates":replicates,
#                      "round_output":False,
#                      "display":False}


#         mi_mean = mi_ses_from_function(delegate,
#                                        df_implicates=df_implicates,
#                                        df_noimputes=df_weights,
#                                        arguments=arguments,
#                                        join_on=["Variable"],
#                                        parallel=True)


#         mi_mean_seq = mi_ses_from_function(delegate,
#                                            df_implicates=df_implicates,
#                                            df_noimputes=df_weights,
#                                            arguments=arguments,
#                                            join_on=["Variable"],
#                                            parallel=False)

#         mi_comp = mi_mean.compare(mi_mean_seq)
#         mi_comp["difference"].print()
#         mi_comp["ratio"].print()

#     def table_test():
#         from NEWS.CodeUtilities.Python.Serializable import Serializable
#         results = Serializable.load_any("/projects/data/NEWS/Test/cid_comparison")
#         df = SafeCollect(results.table_of_estimates(estimates_to_show=["estimate",
#                                                                                "se",
#                                                                                "ci"]))

#         return df

#     def filter_select_test():
#         mi = MultipleImputation.load("/projects/data/NEWS/V2/Estimates/CPS/2019/Income/NEWS/InKind.mi",
#                                           LazyLoad=False)

#         mi.print()
#         mi.filter(pl.col("ShortName") == "allhh")
#         mi.select(["q10",
#                    "q25",
#                    "q50",
#                    "q75",
#                    "q90"])
#         mi.with_columns(pl.col("q90")/1_000_000)
#         mi.rename({coli:coli.replace("q","") for coli in mi.df_estimates.columns if coli.startswith("q")})

#         mi.print(round_output=True)
#         mi.print(round_output=False)

#         mi.drb_round_table()
#         mi.print()

#     # out = table_test()
#     # dfp = out.to_pandas()

#     #   filter_select_test()
