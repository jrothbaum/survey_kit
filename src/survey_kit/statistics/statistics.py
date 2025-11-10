import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT

from formulaic import Formula
from .rounding import Rounding
from ..utilities.inputs import list_input
from ..utilities.dataframe import (
    concat_wrapper,
    columns_from_list,
    NarwhalsType,
    drop_if_exists,
    safe_columns,
)
from .basic_calculations import (
    calculate_by,
    _check_special_modifiers,
    _columns_original_order,
)

from .. import logger


class Statistics:
    """
    Parameters
    ----------
    stats : list[str],
        List of statistics to calculate (mean, median, etc.)
        Call Statistics.available_stats() for options
    formula : str, optional
        formulaic (or R)-style formula for defining statistics to be calculated.
        The default is "".  This takes precedence over columns
    columns : list[str]|str|None, optional
        List of columns to calculate statistics over. The default is None.
    quantile_interpolated : bool, optional
        Use linear interpolation (census-style) for quantiles. The default is False.
    quantile_interpolated_interval : int, optional
        If quantile_interpolated, what is the bin interval? The default is 2500.

    """

    def __init__(
        self,
        stats: list[str],
        formula: str = "",
        columns: list[str] | str | None = None,
        quantile_interpolated: bool = False,
        quantile_interpolated_interval: int = 2500,
    ):
        #   Input parsing/set defaults
        if columns is None:
            columns = []
        elif type(columns) is str:
            columns = [columns]

        self.formula = formula
        self.columns = columns
        self.quantile_interpolated = quantile_interpolated
        self.quantile_interpolated_interval = quantile_interpolated_interval
        self.stats = stats

    @nw.narwhalify
    def calculate(
        self,
        df: IntoFrameT,
        weight: str = "",
        by: dict[str, list[str]] | None = None,
        summarize_vars: list | None = None,
        rounding: Rounding | None = None,
        allow_slow_pandas: bool = False,
    ):
        nw_type = NarwhalsType(df)

        if summarize_vars is None:
            summarize_vars = []
        if rounding is None:
            rounding = Rounding(round_output=False)

        if by is None:
            by = {"All": []}

        if type(by) is list:
            by = {f"{i}": itemi for i, itemi in enumerate(by)}

        if self.formula != "":
            #   It's a formula, process accordingly
            df_summary = nw.from_native(
                Formula(self.formula).get_model_matrix(df)
            ).lazy_backend(nw_type)
            cols_summary = df_summary.collect_schema().names()
        else:
            #   It's a variable list
            if len(self.columns):
                cols = []
                for coli in self.columns:
                    cols.extend(columns_from_list(df=df, columns=[coli]))
                df_summary = df.select(cols)
                cols_summary = cols
            else:
                cols_summary = df.lazy_backend(nw_type).collect_schema().names()

        df_summary = df_summary.select(cs.numeric(), cs.boolean())
        cols_summary = safe_columns(df_summary)
        #   Keep the weights
        if (
            weight != ""
            and weight not in df_summary.lazy_backend(nw_type).collect_schema().names()
        ):
            df_summary = concat_wrapper(
                [df_summary, df.select(weight)], how="horizontal"
            )

        if len(summarize_vars):
            df_summary = concat_wrapper(
                [drop_if_exists(df_summary, summarize_vars), df.select(summarize_vars)],
                how="horizontal",
            )

        #   Rename the stats for more useful table headers
        stats_rename = {
            "count": "n, weighted",
            "rawcount": "n",
            "weight": "n, weighted",
        }

        #   Same with the "modifiers"
        modifiers_output = {
            "not0": " (not 0)",
            "is0": " (== 0)",
            "notmissing": " (not null)",
            "share": " (share)",
            "missing": " (missing)",
        }

        #   Process the stats
        stats_headers = {}
        stats_dict = {}
        for stati in self.stats:
            stat_mod = stati.split("|")

            stati_raw = stat_mod[0]

            modifier = ""
            mod_header = ""
            if len(stat_mod) == 2:
                modifier = stat_mod[1]

                if modifier in modifiers_output.keys():
                    mod_header = modifiers_output[modifier]

            if stati_raw in stats_rename.keys():
                stat_headeri = stats_rename[stati_raw]
            else:
                stat_headeri = stati_raw
            stats_headers[stati] = f"{stat_headeri}{mod_header}"

            if modifier != "":
                cols_include = [f"{coli}|{modifier}" for coli in cols_summary]
            else:
                cols_include = cols_summary.copy()
            stats_dict = column_stats_builder(
                column_stats=stats_dict,
                cols_include=cols_include,
                df=df_summary,
                stat=[stati_raw],
            )

        summary_tables = calculate_by(
            df=df_summary,
            column_stats=stats_dict,
            by=by,
            always_return_as_collection=True,
            weight=weight,
            quantile_interpolated=self.quantile_interpolated,
            quantile_interpolated_interval=self.quantile_interpolated_interval,
            allow_slow_pandas=allow_slow_pandas,
        )

        suffixes = {}

        for stati in self.stats:
            stat_mod = stati.split("|")
            stat_onlyi = stat_mod[0]
            if len(stat_mod) == 2:
                modifier = stat_mod[1]
            else:
                modifier = ""

            suffixes[stati] = self.stat_suffix(stat_onlyi, modifier)

        stat_cols_final = None
        default_index = "___index___"

        for keyi, valuei in summary_tables.items():
            valuei = nw.from_native(valuei)
            b_default_index = False

            if keyi in by.keys():
                index = list_input(by[keyi])

                if index is None:
                    index = default_index
                elif len(index) == 0:
                    index = default_index
            else:
                index = default_index

            if index == default_index:
                #   Just use the row number as the index
                valuei = (
                    valuei.lazy()
                    .collect()
                    .with_row_index(name=index)
                    .lazy_backend(nw_type)
                )
                index = [default_index]
                b_default_index = True

            summaries_by_var = []

            #   Reshape to wide
            for coli in cols_summary:
                #   Catch if it's been renamed, like n->rawcount
                (coli, _, coli_original) = _check_special_modifiers(coli)

                cols_stats = [f"{coli}_{suffixi}" for suffixi in set(suffixes.values())]
                keep_list = index + cols_stats

                summaryi = valuei.select(keep_list)

                if stat_cols_final is None:
                    stat_cols_final = [
                        f"{stats_headers[keyi]}" for keyi in stats_headers.keys()
                    ]

                summaryi = summaryi.rename(
                    {
                        f"{coli}_{suffixes[keyi]}": stats_headers[keyi]
                        for keyi in stats_headers.keys()
                    }
                )
                if b_default_index:
                    summaryi = summaryi.drop(index)
                    keep_list_final = ["Variable"] + list(
                        dict.fromkeys(stats_headers.values())
                    )
                else:
                    keep_list_final = (
                        ["Variable"]
                        + index
                        + list(dict.fromkeys(stats_headers.values()))
                    )

                summaryi = summaryi.with_columns(
                    nw.lit(coli_original).alias("Variable")
                ).select(keep_list_final)
                summaries_by_var.append(summaryi)

            valuei = concat_wrapper(summaries_by_var, how="vertical")
            # summary_tables[keyi] = Compress(valuei,
            #                                 no_boolean=True)
            summary_tables[keyi] = valuei

        cols_by = []
        for keyi, valuei in summary_tables.items():
            cols_by.extend(
                list(
                    set(
                        summary_tables[keyi].lazy().collect_schema().names()
                    ).difference(stat_cols_final + cols_by)
                )
            )
            cols_by.remove("Variable")

        cols_dedupped = list(set(stat_cols_final))
        if len(cols_dedupped) != len(stat_cols_final):
            stat_cols_final = _columns_original_order(
                cols_unordered=cols_dedupped, cols_ordered=stat_cols_final
            )
        keep_order = ["Variable"] + cols_by + stat_cols_final
        output_table = concat_wrapper(
            list(summary_tables.values()), how="diagonal"
        ).select(keep_order)

        if len(cols_by):
            output_table = output_table.sort(cols_by)

        #   Get the information for rounding
        (cols_round, cols_n) = self.rounding_columns(
            output_table.drop(["Variable"] + summarize_vars)
        )

        rounding.cols_round = list(set(rounding.cols_round + cols_round))
        rounding.cols_n = list(set(rounding.cols_n + cols_n))

        return output_table.lazy_backend(nw_type)

    def stat_suffix(
        self=None,
        Statistic: str = "",
        #   not0, missing, nonmissing
        modifier: str = "",
    ) -> str:
        if modifier != "":
            modifier_suffix = f"_{modifier}"
        else:
            modifier_suffix = ""

        if Statistic in ["mean", "sum", "var", "std", "max", "min", "first", "gini"]:
            suffix = Statistic + modifier_suffix
        elif Statistic == "median":
            suffix = "q0_5" + modifier_suffix
        elif Statistic.startswith("q") or Statistic.startswith("p"):
            quantile = float(Statistic.replace("q", "").replace("p", "")) / 100
            suffix = f"q{str(quantile).replace('.', '_')}" + modifier_suffix
        elif (
            Statistic.startswith("count")
            or Statistic.startswith("rawcount")
            or Statistic.startswith("share")
            or Statistic.startswith("rawshare")
            or Statistic == "n"
            or Statistic == "weight"
        ):
            if Statistic.startswith("count") or Statistic == "weight":
                count_prefix = "n"
            elif Statistic.startswith("rawcount") or Statistic == "n":
                count_prefix = "rawn"
            elif Statistic.startswith("share"):
                count_prefix = "share"
            elif Statistic.startswith("rawshare"):
                count_prefix = "rawshare"

            count_suffix = ""
            suffixes = ["_not0", "_is0", "_notmissing", "_missing", "_share"]
            for si in suffixes:
                if Statistic.endswith(si):
                    count_suffix = si

            suffix = f"{count_prefix}{count_suffix}{modifier_suffix}"

        try:
            return suffix
        except:
            message = f"{Statistic} is not a valid statistic"
            logger.error(message)
            raise Exception(message)

    @nw.narwhalify
    def rounding_columns(self, df: IntoFrameT) -> tuple[list[str], list[str]]:
        cols_n = [
            "n",
            "n (missing)",
            "n (not null)",
            "n (not 0)",  # ,
            # "n, weighted",
            # "n missing, weighted",
            # "n (not null), weighted",
            # "n (not 0), weighted"
        ]
        columns = df.lazy().collect_schema().names()
        cols_n = list(set(cols_n).intersection(columns))
        cols_round = list(set(columns).difference(cols_n))

        return (cols_round, cols_n)

    def available_stats():
        examples = [
            "mean",
            "sum",
            "median",
            "q10",
            "q97.5",
            "std",
            "var",
            "max",
            "min",
            "weight",
            "n",
            "gini",
        ]
        logger.info("")
        logger.info(f"Some examples: {examples}")
        logger.info("")
        logger.info(
            "Stats can also have 'modifiers' appended to them separated by a pipe ('|'), including"
        )
        modifiers = ["not0", "missing", "notmissing", "is0", "share"]
        logger.info(modifiers)

        logger.info("")
        logger.info("For quantiles, pass q{number} where number in (0,100)")
        logger.info("")
        logger.info("n is the unweighted count and weight is the weighted count")
        logger.info(f"     for n/weight: {modifiers}")

        modifiers = ["not0"]
        logger.info(f"     for all other stats: {modifiers}")

        examples = [
            "mean|not0",
            "sum|not0",
            "median",
            "min|not0",
            "count|missing",
            "n|notmissing",
            "n|share",
        ]
        logger.info("")
        logger.info(f"""Some examples: {examples}""")


@nw.narwhalify
def column_stats_builder(
    stat: str | list[str],
    column_stats: dict[str, list[str]] | None = None,
    cols_include: list[str] | None = None,
    cols_exclude: list[str] | None = None,
    df: IntoFrameT | None = None,
):
    if column_stats is None:
        column_stats = {}

    cols_include = list_input(cols_include)

    if len(cols_include) == 0 and df is not None:
        cols_include = df.collect_schema().names()
    else:
        cols_include = columns_from_list(df, cols_include)

    if cols_exclude is not None:
        if df is not None:
            cols_exclude = columns_from_list(df=df, columns=cols_exclude)
    else:
        cols_exclude = []

    if df is not None:
        df = df.lazy()

        #   Pass a df?  Then we can check for *
        final_cols = []
        for coli in cols_include:
            (coli, modifier, coli_original) = _check_special_modifiers(coli)

            collisti = columns_from_list(df=df, columns=coli_original)
            if cols_exclude is not None:
                collisti = list(set(collisti).difference(cols_exclude))

            if modifier != "":
                collisti = [f"{coli}|{modifier}" for coli in collisti]

            final_cols.extend(collisti)
        cols = final_cols
    else:
        if len(cols_exclude):
            cols = list(set(cols_include).difference(cols_exclude))
        else:
            cols = cols_include

    for coli in cols:
        if type(stat) is str:
            stat = [stat]

        for stati in stat:
            (stati, modifier, coli_original) = _check_special_modifiers(stati)

            if modifier == "":
                coli_add = coli
            else:
                coli_add = f"{coli}|{modifier}"

            if coli_add in column_stats:
                if stati not in column_stats[coli_add]:
                    column_stats[coli].append(stati)
            else:
                column_stats[coli_add] = [stati]

    return column_stats
