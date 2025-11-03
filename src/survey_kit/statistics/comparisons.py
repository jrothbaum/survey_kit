from copy import deepcopy

import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT

from ..utilities.inputs import list_input
from .replicates import ReplicateStats, ses_from_replicates
from ..utilities.dataframe import concat_wrapper, join_list, safe_height, NarwhalsType
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
    class Variable:
        def __init__(self, column: str, value1: str, value2: str, name: str = ""):
            self.column = column
            self.value1 = value1
            self.value2 = value2
            self.name = name

    class Column:
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
                logger.info("MI COMPARISONS PENDING")
                # mi_stats_diff = MultipleImputation(implicate_stats=implicate_stats_difference,
                #                                         join_on=join_on,
                #                                         rounding=rounding)
                # mi_stats_diff.calculate()

                # output["difference"] = mi_stats_diff

        if ratio:
            if len(comparisons) == 1:
                output["ratio"] = implicate_stats_ratio[0]
            else:
                logger.info("MI COMPARISONS PENDING")
                # mi_stats_ratio = MultipleImputation(implicate_stats=implicate_stats_ratio,
                #                                           join_on=join_on,
                #                                           rounding=rounding)
                # mi_stats_ratio.calculate()

                # output["ratio"] = mi_stats_ratio

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
    elif hasattr(replicate1, "replicates"):
        bootstrap2 = replicate2.replicates.bootstrap

    n_replicates1 = (
        nw.from_native(df1)
        .select(nw.col(replicate_name).max())
        .lazy_backend(nw_type)
        .collect()
        .item(0, 0)
    )
    n_replicates2 = (
        nw.from_native(df2)
        .select(nw.col(replicate_name).max())
        .lazy_backend(nw_type)
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
            comparei_1 = comparei[0]
            comparei_2 = comparei[1]

            if comparei_1 != comparei_2:
                comp_col = f"c({comparei_1},{comparei_2})"
            else:
                comp_col = comparei_1
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

    df_replicates1 = nw_type1.safe_to_narwhals().lazy_backend(nw_type1)
    df_replicates2 = nw_type2.safe_to_narwhals().lazy_backend(nw_type2)

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

        outputs["difference_replicates"] = (
            nw.from_native(df_difference_replicates)
            .lazy_backend(nw_type1)
            .collect()
            .lazy_backend(nw_type1)
            .to_native()
        )
        outputs["difference_estimates"] = (
            nw.from_native(df_difference_estimates)
            .lazy_backend(nw_type1)
            .collect()
            .lazy_backend(nw_type1)
            .to_native()
        )
        outputs["difference_ses"] = (
            nw.from_native(df_difference_ses)
            .lazy_backend(nw_type1)
            .collect()
            .lazy_backend(nw_type1)
            .to_native()
        )

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

        outputs["ratio_replicates"] = (
            nw.from_native(df_ratio_replicates)
            .lazy_backend(nw_type1)
            .collect()
            .lazy_backend(nw_type1)
            .to_native()
        )
        outputs["ratio_estimates"] = (
            nw.from_native(df_ratio_estimates)
            .lazy_backend(nw_type1)
            .collect()
            .lazy_backend(nw_type1)
            .to_native()
        )
        outputs["ratio_ses"] = (
            nw.from_native(df_ratio_ses)
            .lazy_backend(nw_type1)
            .collect()
            .lazy_backend(nw_type1)
            .to_native()
        )

    outputs["bootstrap"] = bootstrap
    return outputs
