from __future__ import annotations

from copy import deepcopy

import narwhals as nw
import narwhals.selectors as cs
import polars as pl
from narwhals.typing import IntoFrameT

from ..utilities.inputs import list_input
from .replicates import ReplicateStats, ses_from_replicates
from ..utilities.dataframe import (
    lazy_backend,
    concat_wrapper,
    join_list,
    safe_height,
    NarwhalsType,
    safe_columns,
    upcast_uint_to_int,
)
from .. import logger


def statistical_comparison_item(stat_item) -> bool:
    if hasattr(stat_item, "implicate_stats"):
        return True
    elif hasattr(stat_item, "df_replicates"):
        return stat_item.df_replicates is not None
    else:
        return False


def _comparison_item(stat_item, implicate: int):
    if hasattr(stat_item, "implicate_stats"):
        return stat_item.implicate_stats[implicate]
    else:
        return stat_item


class ComparisonItem:
    """
    Helpers for specifying comparisons in [StatCalculator][survey_kit.statistics.calculator.StatCalculator]
    and [MultipleImputation][survey_kit.statistics.multiple_imputation.MultipleImputation].

    Provides two types of comparisons:

    - Variable: Compare different variables (e.g., income vs income_2)
    - Column: Compare different statistics (e.g., mean vs median)

    """

    class Variable:
        """
        Specify a comparison between two variables.

        Used to compare different variables within the same dataset, such as
        comparing income from two sources or comparing outcomes across groups.

        Parameters
        ----------
        value1 : str
            First variable value to compare (e.g., "income").
        value2 : str
            Second variable value to compare (e.g., "income_2").
        column : str, optional
            Name of the column containing variable names in the estimates dataframe.
            Default is "Variable".
        name : str, optional
            Name for the comparison result. If empty, uses f"{value1}_vs_{value2}".
            Default is "".

        Examples
        --------
        >>> from survey_kit.statistics.calculator import ComparisonItem
        >>>
        >>> # Compare two income variables
        >>> comp = ComparisonItem.Variable(
        ...     value1="wage_income",
        ...     value2="self_employment_income",
        ...     name="wage_vs_self_employment"
        ... )
        >>>
        >>> comparison = sc.compare(
        ...     sc,
        ...     difference=False,
        ...     compare_list_variables=[comp]
        ... )["ratio"]
        """

        def __init__(
            self, value1: str, value2: str, column: str = "Variable", name: str = ""
        ):
            self.column = column
            self.value1 = value1
            self.value2 = value2
            self.name = name

    class Column:
        """
        Specify a comparison between two statistics/columns.

        Used to compare different statistics for the same variable, such as
        comparing mean vs median or comparing different quantiles.

        Parameters
        ----------
        column1 : str
            First statistic column to compare (e.g., "mean").
        column2 : str
            Second statistic column to compare (e.g., "median").
        name : str, optional
            Name for the comparison result. If empty, uses f"c({column1},{column2})".
            Default is "".

        Examples
        --------
        >>> from survey_kit.statistics.calculator import ComparisonItem
        >>>
        >>> # Compare mean vs median
        >>> comp = ComparisonItem.Column(
        ...     column1="mean",
        ...     column2="median (not 0)",
        ...     name="median_mean_diff"
        ... )
        >>>
        >>> comparison = sc.compare(
        ...     sc,
        ...     ratio=False,
        ...     compare_list_columns=[comp]
        ... )["difference"]
        """

        def __init__(self, column1: str, column2: str, name: str = ""):
            self.column1 = column1
            self.column2 = column2

            if name == "":
                name = f"c({column1},{column2})"
            self.name = name


def compare(
    stats1,
    stats2,
    join_on: list[str],
    rounding,
    difference: bool = True,
    ratio: bool = True,
    ratio_minus_1: bool = True,
    replicate_name: str = "___replicate___",
    compare_list_variables: list[ComparisonItem.Variable] | None = None,
    compare_list_columns: list[ComparisonItem.Column] | None = None,
):
    """
    Compare statistics for multiple imputation (and replicate weight)-based
        statistics

    Parameters
    ----------
    stats1 : MultipleImputation | ReplicateStats | StatCalculator
        First set of statistics.
    stats2 : MultipleImputation | ReplicateStats | StatCalculator
        Another set of statistics.  Result will be stats2 - stats1 or stats2/stats1
    rounding : Rounding
        Parameters for rounding of results
    difference : bool, optional
        Calculate and return the difference (with key "difference"). The default is True.
    ratio : bool, optional
        Calculate and return the ratio (with key "ratio"). The default is True.
    ratio_minus_1 : bool, optional
        Rescale ratio by subtracting 1 from it
    compare_list_variables : list[ComparisonItem.Variable] | None, optional
        List of ComparisonItem.Variable of variables to compare.
    compare_list_columns : list[ComparisonItem.Column], optional
        List of columns to compare
        For example if compare_list_columns = [("mean","median")]
            then compare the mean of 1 to the median of 2
    Returns
    -------
    MultipleImputation
        The output statistics.
    """

    from .multiple_imputation import MultipleImputation

    n_implicates1 = 0
    n_implicates2 = 0
    if hasattr(stats1, "implicate_stats"):
        n_implicates1 = len(stats1.implicate_stats)

    if hasattr(stats2, "implicate_stats"):
        n_implicates2 = len(stats2.implicate_stats)

    if (n_implicates1 != n_implicates2) and n_implicates1 != 0 and n_implicates2 != 0:
        message = f"Must have the same number of implicates in stats1 and stats2 ({n_implicates1} != {n_implicates2})"
        logger.error(message)
        raise Exception(message)

    n_implicates = max(max(n_implicates1, n_implicates2), 1)

    comparisons = []
    for i in range(0, n_implicates):
        statsi_1 = _comparison_item(stat_item=stats1, implicate=i)

        statsi_2 = _comparison_item(stat_item=stats2, implicate=i)

        has_replicates1 = getattr(statsi_1, "df_replicates", None) is not None
        has_replicates2 = getattr(statsi_2, "df_replicates", None) is not None

        if has_replicates1 != has_replicates2:
            message = (
                "Cannot compare results where one has replicate weights "
                "(df_replicates) and the other doesn't - both sides must use "
                "the same variance estimation method."
            )
            logger.error(message)
            raise Exception(message)

        if has_replicates1:
            comparisons.append(
                _compare_one_implicate(
                    replicate1=statsi_1,
                    replicate2=statsi_2,
                    join_on=join_on,
                    difference=difference,
                    ratio=ratio,
                    ratio_minus_1=ratio_minus_1,
                    replicate_name=replicate_name,
                    compare_list_variables=compare_list_variables,
                    compare_list_columns=compare_list_columns,
                )
            )
        else:
            comparisons.append(
                _compare_one_implicate_generic(
                    replicate1=statsi_1,
                    replicate2=statsi_2,
                    join_on=join_on,
                    difference=difference,
                    ratio=ratio,
                    ratio_minus_1=ratio_minus_1,
                    compare_list_variables=compare_list_variables,
                    compare_list_columns=compare_list_columns,
                )
            )
    output = {}
    if len(comparisons):
        implicate_stats_difference = []
        implicate_stats_ratio = []

        for comparisoni in comparisons:
            if difference:
                #   StatCalculator?
                if hasattr(stats1, "by") or hasattr(stats2, "by"):
                    if hasattr(stats1, "by"):
                        diffi = deepcopy(stats1)
                    else:
                        diffi = deepcopy(stats2)

                    diffi.df_estimates = comparisoni["difference_estimates"]
                    diffi.df_ses = comparisoni["difference_ses"]
                    diffi.df_replicates = comparisoni["difference_replicates"]
                else:
                    diffi = ReplicateStats(
                        df_estimates=comparisoni["difference_estimates"],
                        df_ses=comparisoni["difference_ses"],
                        df_replicates=comparisoni["difference_replicates"],
                        bootstrap=comparisoni["bootstrap"],
                    )

                implicate_stats_difference.append(diffi)
            if ratio:
                if hasattr(stats1, "by") or hasattr(stats2, "by"):
                    if hasattr(stats1, "by"):
                        ratioi = deepcopy(stats1)
                    else:
                        ratioi = deepcopy(stats2)

                    ratioi.df_estimates = comparisoni["ratio_estimates"]
                    ratioi.df_ses = comparisoni["ratio_ses"]
                    ratioi.df_replicates = comparisoni["ratio_replicates"]
                else:
                    ratioi = ReplicateStats(
                        df_estimates=comparisoni["ratio_estimates"],
                        df_ses=comparisoni["ratio_ses"],
                        df_replicates=comparisoni["ratio_replicates"],
                        bootstrap=comparisoni["bootstrap"],
                    )
                implicate_stats_ratio.append(ratioi)

        if difference:
            if len(comparisons) == 1:
                output["difference"] = implicate_stats_difference[0]
            else:
                mi_stats_diff = MultipleImputation(
                    implicate_stats=implicate_stats_difference,
                    join_on=join_on,
                    rounding=rounding,
                )
                mi_stats_diff.calculate()

                output["difference"] = mi_stats_diff

        if ratio:
            if len(comparisons) == 1:
                output["ratio"] = implicate_stats_ratio[0]
            else:
                mi_stats_ratio = MultipleImputation(
                    implicate_stats=implicate_stats_ratio,
                    join_on=join_on,
                    rounding=rounding,
                )
                mi_stats_ratio.calculate()

                output["ratio"] = mi_stats_ratio

    return output


def _compare_one_implicate(
    replicate1,
    replicate2,
    join_on: list[str],
    difference: bool = True,
    ratio: bool = True,
    ratio_minus_1: bool = True,
    replicate_name: str = "___replicate___",
    compare_list_variables: list[ComparisonItem.Variable] | None = None,
    compare_list_columns: list[ComparisonItem.Column] | None = None,
) -> dict[str, IntoFrameT]:
    df1 = replicate1.df_replicates
    df2 = replicate2.df_replicates

    nw_type = NarwhalsType(df1)
    if hasattr(replicate1, "bootstrap"):
        bootstrap1 = replicate1.bootstrap
    elif hasattr(replicate1, "replicates"):
        bootstrap1 = replicate1.replicates.bootstrap
    if hasattr(replicate2, "bootstrap"):
        bootstrap2 = replicate2.bootstrap
    elif hasattr(replicate2, "replicates"):
        bootstrap2 = replicate2.replicates.bootstrap

    n_replicates1 = (
        lazy_backend(nw.from_native(df1).select(nw.col(replicate_name).max()), nw_type)
        .collect()
        .item(0, 0)
    )
    n_replicates2 = (
        lazy_backend(nw.from_native(df2).select(nw.col(replicate_name).max()), nw_type)
        .collect()
        .item(0, 0)
    )

    if bootstrap1 != bootstrap2:
        message = f"Must have the same type of bootstrap/replicate weights in stats1 and stats2 (bootstrap1 = {bootstrap1}, bootstrap2 = {bootstrap2})"
        logger.error(message)
        raise Exception(message)

    if n_replicates1 != n_replicates2:
        message = f"Must have the same number of replicates in stats1 and stats2 ({n_replicates1} != {n_replicates2})"
        logger.error(message)
        raise Exception(message)

    (df1, df2) = process_compare_lists(
        df1=df1,
        df2=df2,
        join_on=join_on,
        replicate_name=replicate_name,
        compare_list_variables=compare_list_variables,
        compare_list_columns=compare_list_columns,
    )

    comparison = replicate_comparison(
        df_replicates1=df1,
        df_replicates2=df2,
        n_replicates=n_replicates1,
        difference=difference,
        bootstrap=bootstrap1,
        ratio=ratio,
        ratio_minus_1=ratio_minus_1,
        join_on1=join_on,
    )

    return comparison


def _single_stat_column(df: pl.DataFrame, join_on: list[str]) -> str:
    cols = [c for c in df.columns if c not in join_on]
    if len(cols) != 1:
        message = (
            "Comparing results that have no replicate weights (df_replicates) "
            f"only supports a single stat column, got {cols}."
        )
        logger.error(message)
        raise Exception(message)
    return cols[0]


def _matching_row(df: pl.DataFrame, column: str, value) -> dict:
    matched = df.filter(pl.col(column) == value)
    if matched.height == 0:
        message = f"No row found where {column} == {value!r}"
        logger.error(message)
        raise Exception(message)
    return matched.row(0, named=True)


def _vcov_lookup(
    df_vcov: IntoFrameT | None,
    join_on: list[str],
    column: str,
    value1,
    value2,
) -> float | None:
    """
    Look up Cov(value1, value2) from a df_vcov table (long/pairwise form,
    see MultipleImputation.df_vcov). Only meaningful when the comparison is
    keyed by the object's single join_on column, since df_vcov's rows are
    keyed by join_on, not by an arbitrary ComparisonItem.Variable.column.
    Returns None (not calculable) rather than guessing in every other case.
    """
    if df_vcov is None or len(join_on) != 1 or column != join_on[0]:
        return None

    df_vcov = NarwhalsType(df_vcov).to_polars().lazy().collect()
    col_1, col_2 = f"{join_on[0]}_1", f"{join_on[0]}_2"
    value_col = [c for c in df_vcov.columns if c not in (col_1, col_2)][0]

    matched = df_vcov.filter((pl.col(col_1) == value1) & (pl.col(col_2) == value2))
    if matched.height == 0:
        return None
    return matched.row(0, named=True)[value_col]


def _compare_one_implicate_generic(
    replicate1,
    replicate2,
    join_on: list[str],
    difference: bool = True,
    ratio: bool = True,
    ratio_minus_1: bool = True,
    compare_list_variables: list[ComparisonItem.Variable] | None = None,
    compare_list_columns: list[ComparisonItem.Column] | None = None,
) -> dict[str, IntoFrameT]:
    """
    Compare two results that have no replicate weights (df_replicates) - e.g.
    generic delegate output combined via Rubin's rules only. Standard errors
    for the difference/ratio are computed by classical propagation
    (Var(diff) = Var(a) + Var(b) - 2 Cov(a,b), and the analogous delta-method
    formula for a ratio) instead of by resampling.

    Cov(a,b) is only nonzero when both values come from the SAME underlying
    result (replicate1 is replicate2 - e.g. `mi.compare(mi, ...)` comparing
    two rows/coefficients of one fit) and that result has a df_vcov to look
    the covariance up in; otherwise the two values are treated as
    independent, which is the standard (and only available) assumption when
    comparing across two separately-estimated results. Raises when
    independence can't be safely assumed (same source, different rows) and
    no covariance information is available.
    """
    if compare_list_columns:
        message = (
            "compare_list_columns is not supported when comparing results "
            "that have no replicate weights (df_replicates) - only a plain "
            "row-by-row compare() or compare_list_variables is supported here."
        )
        logger.error(message)
        raise Exception(message)

    df1_est = NarwhalsType(replicate1.df_estimates).to_polars().lazy().collect()
    df2_est = NarwhalsType(replicate2.df_estimates).to_polars().lazy().collect()
    df1_ses = NarwhalsType(replicate1.df_ses).to_polars().lazy().collect()
    df2_ses = NarwhalsType(replicate2.df_ses).to_polars().lazy().collect()

    value_col1 = _single_stat_column(df1_est, join_on)
    value_col2 = _single_stat_column(df2_est, join_on)

    same_source = replicate1 is replicate2
    compare_list_variables = list_input(compare_list_variables)

    if compare_list_variables:
        if len(join_on) != 1:
            message = (
                "compare_list_variables for results without replicate weights "
                "only supports a single join_on column."
            )
            logger.error(message)
            raise Exception(message)

        rows = []
        for comparei in compare_list_variables:
            col = comparei.column
            name = comparei.name if comparei.name else f"{comparei.value1}_vs_{comparei.value2}"

            est1 = _matching_row(df1_est, col, comparei.value1)[value_col1]
            est2 = _matching_row(df2_est, col, comparei.value2)[value_col2]
            se1 = _matching_row(df1_ses, col, comparei.value1)[value_col1]
            se2 = _matching_row(df2_ses, col, comparei.value2)[value_col2]

            cov = 0.0
            if same_source:
                if comparei.value1 == comparei.value2:
                    #   Same row compared to itself - perfectly correlated.
                    cov = se1**2
                else:
                    looked_up = _vcov_lookup(
                        replicate1.df_vcov,
                        join_on,
                        col,
                        comparei.value1,
                        comparei.value2,
                    )
                    if looked_up is None:
                        message = (
                            f"Cannot compute a joint standard error between "
                            f"'{comparei.value1}' and '{comparei.value2}' from the "
                            "same result without a variance-covariance matrix "
                            "(df_vcov) or replicate weights (df_replicates) - "
                            "independence can't be assumed within the same fit."
                        )
                        logger.error(message)
                        raise Exception(message)
                    cov = looked_up

            rows.append(
                {
                    join_on[0]: name,
                    "___est1___": est1,
                    "___est2___": est2,
                    "___se1___": se1,
                    "___se2___": se2,
                    "___cov___": cov,
                }
            )
        merged = pl.DataFrame(rows)
    else:
        if same_source:
            message = (
                "A plain row-by-row compare() of a result against itself "
                "would compare correlated rows without a covariance lookup - "
                "use compare_list_variables instead, which can use df_vcov."
            )
            logger.error(message)
            raise Exception(message)

        merged = (
            df1_est.rename({value_col1: "___est1___"})
            .join(
                df2_est.rename({value_col2: "___est2___"}), on=join_on, how="inner"
            )
            .join(
                df1_ses.rename({value_col1: "___se1___"}), on=join_on, how="inner"
            )
            .join(
                df2_ses.rename({value_col2: "___se2___"}), on=join_on, how="inner"
            )
            .with_columns(pl.lit(0.0).alias("___cov___"))
        )

    outputs = {}
    c_est1 = pl.col("___est1___")
    c_est2 = pl.col("___est2___")
    c_se1 = pl.col("___se1___")
    c_se2 = pl.col("___se2___")
    c_cov = pl.col("___cov___")

    if difference:
        df_diff = merged.with_columns(
            [
                (c_est2 - c_est1).alias(value_col1),
                (c_se1**2 + c_se2**2 - 2 * c_cov).sqrt().alias("___diff_se___"),
            ]
        )
        outputs["difference_estimates"] = df_diff.select(join_on + [value_col1])
        outputs["difference_ses"] = df_diff.select(
            join_on + [pl.col("___diff_se___").alias(value_col1)]
        )
        outputs["difference_replicates"] = None

    if ratio:
        ratio_expr = c_est2 / c_est1
        if ratio_minus_1:
            ratio_expr = ratio_expr - 1
        #   Delta method: Var(b/a) = (b/a^2)^2 Var(a) + (1/a)^2 Var(b) - 2(b/a^3)Cov(a,b)
        var_ratio = (
            (c_est2 / c_est1**2) ** 2 * c_se1**2
            + (1 / c_est1) ** 2 * c_se2**2
            - 2 * (c_est2 / c_est1**3) * c_cov
        )
        df_ratio = merged.with_columns(
            [
                ratio_expr.alias(value_col1),
                var_ratio.sqrt().alias("___ratio_se___"),
            ]
        )
        outputs["ratio_estimates"] = df_ratio.select(join_on + [value_col1])
        outputs["ratio_ses"] = df_ratio.select(
            join_on + [pl.col("___ratio_se___").alias(value_col1)]
        )
        outputs["ratio_replicates"] = None

    outputs["bootstrap"] = False
    return outputs


def process_compare_lists(
    df1: IntoFrameT,
    df2: IntoFrameT,
    join_on: list[str],
    replicate_name: str = "",
    compare_list_variables: list[ComparisonItem.Variable] | None = None,
    compare_list_columns: list[ComparisonItem.Column] | None = None,
) -> tuple[IntoFrameT, IntoFrameT]:
    nw_type1 = NarwhalsType(df1)
    nw_type2 = NarwhalsType(df2)

    df1 = nw_type1.safe_to_narwhals()
    df2 = nw_type2.safe_to_narwhals()

    compare_list_variables = list_input(compare_list_variables)
    compare_list_columns = list_input(compare_list_columns)

    if len(compare_list_variables):
        df1_list = []
        df2_list = []

        for comparei in compare_list_variables:
            comparei_col = comparei.column
            comparei_1 = comparei.value1
            comparei_2 = comparei.value2
            rename_to = comparei.name

            c_v = nw.col(comparei_col)

            df1i = df1.filter(c_v == comparei_1).with_columns(
                nw.when(c_v == comparei_1)
                .then(nw.lit(rename_to))
                .otherwise(c_v)
                .alias(comparei_col)
            )

            df2i = df2.filter(c_v == comparei_2).with_columns(
                nw.when(c_v == comparei_2)
                .then(nw.lit(rename_to))
                .otherwise(c_v)
                .alias(comparei_col)
            )

            if safe_height(df1i) and safe_height(df2i):
                df1_list.append(df1i)
                df2_list.append(df2i)

            del df1i
            del df2i

        df1 = concat_wrapper(df1_list, how="diagonal")
        df2 = concat_wrapper(df2_list, how="diagonal")

    if len(compare_list_columns):
        cols_keep = join_on.copy()
        if replicate_name != "":
            cols_keep.append(replicate_name)

        for comparei in compare_list_columns:
            comparei_1 = comparei.column1
            comparei_2 = comparei.column2

            if comparei.name == "":
                if comparei_1 != comparei_2:
                    comp_col = f"c({comparei_1},{comparei_2})"
                else:
                    comp_col = comparei_1
            else:
                comp_col = comparei.name
            df1 = df1.with_columns(nw.col(comparei_1).alias(comp_col))
            df2 = df2.with_columns(nw.col(comparei_2).alias(comp_col))
            cols_keep.append(comp_col)

        df1 = df1.select(cols_keep)
        df2 = df2.select(cols_keep)

    return NarwhalsType.return_df(df1, nw_type1), NarwhalsType.return_df(df2, nw_type2)


def replicate_comparison(
    df_replicates1: IntoFrameT,
    df_replicates2: IntoFrameT,
    n_replicates: int,
    join_on1: list | None = None,
    join_on2: list | None = None,
    map1_to_2: dict[str, list[str] | str] | None = None,
    bootstrap: bool = False,
    difference: bool = True,
    ratio: bool = True,
    ratio_minus_1: bool = True,
    replicate_name1: str = "___replicate___",
    replicate_name2: str = "___replicate___",
) -> dict:
    nw_type1 = NarwhalsType(df_replicates1)
    nw_type2 = NarwhalsType(df_replicates2)

    #   Upcast UInt columns to signed Int - otherwise the difference/ratio
    #   computations below can silently wrap around instead of going negative.
    df_replicates1 = upcast_uint_to_int(
        lazy_backend(nw_type1.safe_to_narwhals(), nw_type1)
    )
    df_replicates2 = upcast_uint_to_int(
        lazy_backend(nw_type2.safe_to_narwhals(), nw_type2)
    )

    join_on1 = list_input(join_on1)

    if join_on2 is None:
        #   Intentional - join2 = join1 if not passed
        join_on2 = join_on1

    if map1_to_2 is None:
        #   Simple just merge 1 to 2 and compare
        compare_vars1 = (
            df_replicates1.drop(join_on1 + [replicate_name1]).collect_schema().names()
        )

        rename1 = {}
        rename2 = {coli: f"{coli}_2" for coli in compare_vars1}

        compare_vars2 = [f"{coli}_2" for coli in compare_vars1]

    else:
        pass

    if len(rename1):
        df_replicates1 = df_replicates1.rename(rename1)

    if len(rename2):
        df_replicates2 = df_replicates2.rename(rename2)

    df_combined = nw.from_native(
        join_list(
            [df_replicates1, df_replicates2],
            how="inner",
            on=[join_on1 + [replicate_name1], join_on2 + [replicate_name2]],
        )
    )

    outputs = {}
    if difference:
        with_differences = []
        drops = []
        for i in range(0, len(compare_vars1)):
            col1 = compare_vars1[i]
            col2 = compare_vars2[i]
            c_1 = nw.col(col1)
            c_2 = nw.col(col2)
            with_differences.append((c_2 - c_1).alias(col1))
            drops.append(col2)
        df_difference_replicates = (
            df_combined.with_columns(cs.boolean().cast(nw.Int8))
            .with_columns(with_differences)
            .drop(drops)
        )

        (df_difference_estimates, df_difference_ses) = ses_from_replicates(
            df_replicates=df_difference_replicates,
            join_on=join_on1,
            n_replicates=n_replicates,
            bootstrap=bootstrap,
            replicate_name=replicate_name1,
        )

        outputs["difference_replicates"] = lazy_backend(
            lazy_backend(nw.from_native(df_difference_replicates), nw_type1).collect(),
            nw_type1,
        ).to_native()
        outputs["difference_estimates"] = lazy_backend(
            lazy_backend(nw.from_native(df_difference_estimates), nw_type1).collect(),
            nw_type1,
        ).to_native()
        outputs["difference_ses"] = lazy_backend(
            lazy_backend(nw.from_native(df_difference_ses), nw_type1).collect(),
            nw_type1,
        ).to_native()

    if ratio:
        with_ratios = []
        drops = []
        for i in range(0, len(compare_vars1)):
            col1 = compare_vars1[i]
            col2 = compare_vars2[i]
            c_1 = nw.col(col1)
            c_2 = nw.col(col2)
            if ratio_minus_1:
                with_ratios.append(((c_2 / c_1) - 1).alias(col1))
            else:
                with_ratios.append((c_2 / c_1).alias(col1))
            drops.append(col2)
        df_ratio_replicates = df_combined.with_columns(with_ratios).drop(drops)

        (df_ratio_estimates, df_ratio_ses) = ses_from_replicates(
            df_replicates=df_ratio_replicates,
            join_on=join_on1,
            n_replicates=n_replicates,
            bootstrap=bootstrap,
            replicate_name=replicate_name1,
        )

        outputs["ratio_replicates"] = lazy_backend(
            lazy_backend(nw.from_native(df_ratio_replicates), nw_type1).collect(),
            nw_type1,
        ).to_native()
        outputs["ratio_estimates"] = lazy_backend(
            lazy_backend(nw.from_native(df_ratio_estimates), nw_type1).collect(),
            nw_type1,
        ).to_native()
        outputs["ratio_ses"] = lazy_backend(
            lazy_backend(nw.from_native(df_ratio_ses), nw_type1).collect(), nw_type1
        ).to_native()

    outputs["bootstrap"] = bootstrap
    return outputs


def _append_to_mi(
    name: str,
    stat_item,
    stats,
    vertical: bool,
    n_implicates: int,
    vertical_drop_var_name: bool,
    separator: str = ":",
):
    from .replicates import ReplicateStats
    from .multiple_imputation import MultipleImputation
    from .calculator import StatCalculator

    variable_ids = _join_variables(stat_item)

    if stats.join_on is None:
        stats.join_on = variable_ids
    elif len(stats.join_on) == 0:
        stats.join_on = variable_ids
    elif stats.join_on != variable_ids:
        message = f"ID variables for stats must match across data ({stats.join_on} != {variable_ids})"
        logger.error(message)
        raise Exception(message)
    if type(stat_item) is not MultipleImputation:
        implicate_stats = []
        for i in range(0, n_implicates):
            if type(stat_item) is ReplicateStats:
                rep_stats = stat_item
            elif isinstance(stat_item, StatCalculator):
                rep_stats = stat_item.replicate_stats

            implicate_stats.append(rep_stats)

        stat_item = MultipleImputation(
            implicate_stats=implicate_stats, join_on=variable_ids
        )
        stat_item.calculate()
    if vertical:
        if vertical_drop_var_name:
            rename = nw.lit(name).alias(variable_ids[0])
        else:
            rename = nw.concat_str(
                nw.lit(name), nw.col(variable_ids[0]), separator=separator
            ).alias(variable_ids[0])
    else:
        rename = {}
        rename_df_p = {}
        for coli in stat_item.df_estimates.columns:
            if coli not in variable_ids:
                rename[coli] = f"{name}{separator}{coli}"
                rename_df_p[f"{coli}_df"] = f"{name}{separator}{coli}_df"
                rename_df_p[f"{coli}_t"] = f"{name}{separator}{coli}_t"

    df_add_names = [
        "df_estimates",
        "df_ses",
        "df_df",
        "df_t",
        "df_p",
        "df_rate_of_missing_information",
    ]
    dfs_to_add = {}

    for df_name in df_add_names:
        if vertical:
            dfs_to_add[df_name] = (
                nw.from_native(getattr(stat_item, df_name))
                .with_columns(rename)
                .to_native()
            )
        else:
            if df_name == "df_p":
                dfs_to_add[df_name] = (
                    nw.from_native(getattr(stat_item, df_name))
                    .rename(rename)
                    .rename(rename_df_p)
                    .to_native()
                )
            else:
                dfs_to_add[df_name] = (
                    nw.from_native(getattr(stat_item, df_name))
                    .rename(rename)
                    .to_native()
                )

    implicate_stats = []
    for impi in stat_item.implicate_stats:
        if vertical:
            repi = ReplicateStats(
                df_estimates=(
                    nw.from_native(impi.df_estimates).with_columns(rename).to_native()
                ),
                df_ses=(nw.from_native(impi.df_ses).with_columns(rename).to_native()),
                df_replicates=(
                    nw.from_native(impi.df_replicates).with_columns(rename).to_native()
                ),
                bootstrap=impi.bootstrap,
            )
        else:
            repi = ReplicateStats(
                df_estimates=(
                    nw.from_native(impi.df_estimates).rename(rename).to_native()
                ),
                df_ses=(nw.from_native(impi.df_ses).rename(rename).to_native()),
                df_replicates=(
                    nw.from_native(impi.df_replicates).rename(rename).to_native()
                ),
                bootstrap=impi.bootstrap,
            )

        implicate_stats.append(repi)

    stats = _add_to_df_list(stats, dfs_to_add, vertical, variable_ids)

    for n_impi, impi in enumerate(implicate_stats):
        dfs_to_add = {
            "df_estimates": impi.df_estimates,
            "df_ses": impi.df_ses,
            "df_replicates": impi.df_replicates,
        }

        if len(stats.implicate_stats) > n_impi:
            stats.implicate_stats[n_impi] = _add_to_df_list(
                stats.implicate_stats[n_impi], dfs_to_add, vertical, variable_ids
            )
        else:
            stats.implicate_stats.append(impi)

    return stats


def _append_to_sc(
    name: str,
    stat_item,
    stats,
    vertical: bool,
    vertical_drop_var_name: bool,
    separator: str = ":",
):
    from .replicates import ReplicateStats
    from .calculator import StatCalculator

    variable_ids = _join_variables(stat_item)

    if len(stats.variable_ids) == 0:
        stats.variable_ids = variable_ids
        stats.replicates = stat_item.replicates

    elif stats.variable_ids != variable_ids:
        message = f"ID variables for stats must match across data ({stats.variable_ids} != {variable_ids})"
        logger.error(message)
        raise Exception(message)

    if vertical:
        if vertical_drop_var_name:
            rename = nw.lit(name).alias(variable_ids[0])
        else:
            rename = nw.concat_str(
                nw.lit(name), nw.col(variable_ids[0]), separator=separator
            ).alias(variable_ids[0])

    else:
        rename = {}
        for coli in safe_columns(stat_item.df_estimates):
            if coli not in variable_ids:
                rename[coli] = f"{name}{separator}{coli}"

    df_add_names = ["df_estimates", "df_ses", "df_replicates"]
    dfs_to_add = {}

    for df_name in df_add_names:
        if vertical:
            if getattr(stat_item, df_name) is not None:
                dfs_to_add[df_name] = (
                    nw.from_native(getattr(stat_item, df_name))
                    .with_columns(rename)
                    .to_native()
                )
        else:
            if getattr(stat_item, df_name) is not None:
                dfs_to_add[df_name] = (
                    nw.from_native(getattr(stat_item, df_name))
                    .rename(rename)
                    .to_native()
                )

    stats = _add_to_df_list(stats, dfs_to_add, vertical, variable_ids)

    return stats


def _add_to_df_list(stats, dfs_to_add: dict, vertical: bool, variable_ids: list[str]):
    for df_name, dfi in dfs_to_add.items():
        df_add_to = getattr(stats, df_name)
        if df_add_to is None:
            setattr(stats, df_name, dfi)
        else:
            if vertical:
                setattr(
                    stats, df_name, concat_wrapper([df_add_to, dfi], how="diagonal")
                )
            else:
                if df_name == "df_replicates":
                    join_on_replicates = ["___replicate___"]
                else:
                    join_on_replicates = []

                setattr(
                    stats,
                    df_name,
                    join_list(
                        [df_add_to, dfi],
                        how="outer",
                        on=variable_ids + join_on_replicates,
                    ),
                )
    return stats


def _join_variables(stat_item) -> list[str]:
    variable_ids = None
    if hasattr(stat_item, "variable_ids"):
        variable_ids = stat_item.variable_ids

    if variable_ids is None:
        variable_ids = []

    if len(variable_ids) == 0:
        variable_ids = [stat_item.df_estimates.columns[0]]

    return variable_ids
