from __future__ import annotations
from typing import TYPE_CHECKING

import os
import polars as pl
import polars.selectors as cs
import numpy as np
from copy import deepcopy
from glob import glob

from ..utilities.logging import set_logging
from ..utilities.inputs import create_folders_if_needed

from ..utilities.dataframe import (
    safe_height,
    join_list,
    concat_wrapper,
    drop_if_exists,
    asterisk_matched_substring,
    columns_from_list,
    safe_columns,
)

from ..statistics.bootstrap import bayes_bootstrap_weights
from ..utilities.random import set_seed, generate_seed
from ..utilities.rounding import drb_round_table
from ..statistics.statistics import Statistics
from ..statistics.calculator import StatCalculator

#   SRMI modules
from .variable import Variable
from .impute import Impute

from ..serializable import Serializable
from .. import logger


if TYPE_CHECKING:
    from .srmi import SRMI


class Implicate(Serializable):
    """
    Manages imputation for a single implicate in the SRMI process.

    This class handles the execution of imputation for one independent run
    of the model, including iteration management and variable processing.

    Parameters
    ----------
    parent : SRMI | None, optional
        Pointer to parent SRMI instance, by default None
    number : int, optional
        Implicate number identifier, by default 0
    seed : int, optional
        Random seed for this implicate, by default 0
    load : bool, optional
        Whether to load existing data, by default True
    """

    _save_suffix = "srmi.implicate"
    _save_exclude_items = ["parent"]

    def __init__(
        self,
        parent: SRMI | None = None,
        number: int = 0,
        seed: int = 0,
        load: bool = True,
        **df_kwargs,
    ):
        """
        This class handles the pieces need to impute a single
        implicate (separate independent run of the model).
        This calls the necessary sub-classes (Implicate, Selection)
        to impute each variable

        Parameters
        ----------
        parent : SRMI
            Pointer to the parent SRMI class to get any information from there
        number : int, optional
            Implicate number

        """

        self.parent = parent
        self.number = number
        self.seed = seed

        #   Dataframe for only the imputed variables (and linkage keys)
        self.df = None

        #   Statistics for each iteration
        self.df_summary_stats = {}

        #   Status variables - current iteration, variable, and variable_by
        #       subgroup models
        self.status_iteration = 0
        self.status_variable = 0
        self.status_iteration_complete = False
        self.complete = False
        self.in_progress = False

        if load:
            self.load(**df_kwargs)

    def __del__(self):
        #   Remove circular reference to parent
        self.parent = None

    #####################################################
    #   Serializable - BEGIN
    #####################################################
    def save(self):
        create_folders_if_needed(self.path_logs, quietly=True)

        super().save(self.path_implicate)

    def load(self, **df_kwargs):
        path_load = f"{self.path_implicate}"
        full_path = f"{self.path_implicate}.{self._save_suffix}"

        if os.path.isdir(full_path):
            obj = super().load(
                path_load, init_kwargs=dict(parent=self.parent, load=False), **df_kwargs
            )

            items = vars(obj)

            for key, value in items.items():
                setattr(self, key, value)

    #####################################################
    #   Serializable - END
    #####################################################

    #   What get's saved/loaded along with df
    def _save_items(self) -> list[str]:
        return [
            "status_iteration",
            "status_variable",
            "status_iteration_complete",
            "complete",
            "in_progress",
            "seed",
        ]

    def run(
        self, iteration_number: int = 0, variable_start: int = 0, variable_end: int = 0
    ) -> None:
        """
        Run the implicate from the last completed step.

        Parameters
        ----------
        iteration_number : int, optional
            Run only a specific iteration (0 = run all), by default 0.
                This is mostly for handling run continuations and parallel runs in chunks
        variable_start : int, optional
            Starting variable index for subset runs, by default 0.
                When running a chunked run (i.e. impute in this thread from variable 5 to 10)
        variable_end : int, optional
            Ending variable index for subset runs, by default 0.
                When running a chunked run (i.e. impute in this thread from variable 5 to 10)

        Notes
        -----
        - Manages iteration and variable loops
        - Handles Bayesian bootstrap if enabled
        - Saves progress after each iteration/variable as configured
        - Generates summary statistics upon completion
        """

        #   Make sure the log for this variable is not also writing to prior implicates
        if "SRMI_Implicate" in logger.root.manager.loggerDict.keys():
            del logger.root.manager.loggerDict["SRMI_Implicate"]
        self.logging = set_logging(
            path_log=os.path.normpath(f"{self.path_logs}/000.parent.log"),
            to_console=True,
            name="SRMI_Implicate",
        )

        set_seed(self.seed)
        #   self.logging.info(f"Current Random Number: {generate_seed()}")

        did_anything = False
        if self.complete:
            self.logging.info(
                f"Imputation already complete for implicate #{self.number}"
            )
        else:
            self.logging.info(f"Running implicate #{self.number}")
            self.df = self.df_full()

            self.in_progress = True

            if iteration_number > 0:
                start_at = iteration_number
                end_at = iteration_number

                #   Check that the prior iteration is complete
                if iteration_number > 1:
                    prior_complete = (
                        self.status_iteration == iteration_number - 1
                        and self.status_iteration_complete
                    ) or (self.status_iteration >= iteration_number)

                    if not prior_complete:
                        self.logging.info(
                            f"     PRIOR ITERATION NOT COMPLETE FOR ITERATION #{iteration_number}, STOPPING"
                        )
                        return None
            else:
                start_at = 1
                end_at = self.parent.n_iterations

            for iterationi in range(start_at, end_at + 1):
                if self.status_iteration <= iterationi:
                    did_something = self._run_one_iteration(
                        iterationi=iterationi,
                        variable_start=variable_start,
                        variable_end=variable_end,
                    )

                    did_anything = did_anything or did_something

                    if (
                        iterationi == self.parent.n_iterations
                        and self.status_iteration_complete
                        and self.status_iteration == self.parent.n_iterations
                    ):
                        self.complete = True

            if self.complete:
                self.df_full_summary_stats(print_table=True)

            if did_anything:
                self.save()

    def _run_one_iteration(
        self, iterationi: int, variable_start: int = 0, variable_end: int = 0
    ):
        self.status_iteration_complete = False

        #   New iteration?
        if self.status_iteration != iterationi:
            self.status_variable = 1

            #   Statistics for each iteration
            self.df_summary_stats[iterationi] = None

        did_something = False

        variable_index = 0
        for vari in self.parent.variables:
            variable_index += 1

            to_run = (variable_start == 0 or variable_end == 0) or (
                variable_index >= variable_start and variable_index <= variable_end
            )

            if self.status_variable <= variable_index and to_run:
                if variable_index > 1:
                    #   Not the first variable, make sure we're ready for it
                    if self.status_variable != (variable_index - 1):
                        self.logging.info(
                            f"     PRIOR VARIABLE NOT COMPLETE FOR VARIABLE #{variable_index}, STOPPING BECAUSE PRIOR = {self.status_variable}"
                        )
                        return False

                self.status_iteration = iterationi
                self.status_variable = variable_index
                np.random.seed(generate_seed())

                if variable_index == 1:
                    self.logging.info(
                        "*********************************************************"
                    )
                    self.logging.info(
                        "*********************************************************"
                    )
                    self.logging.info(f"Iteration #{self.status_iteration}")
                    self.logging.info(
                        "*********************************************************"
                    )
                    self.logging.info(
                        "*********************************************************"
                    )
                    self.logging.info("\n\n")

                self.seed = generate_seed()
                np.random.seed(generate_seed())

                self.logging.info(
                    "#########################################################"
                )
                self.logging.info(
                    f"     Variable: {self.status_iteration}.{variable_index}"
                )
                self.logging.info(f"     Name:     {vari.impute_var}")
                if vari.header != "":
                    self.logging.info(f"     {vari.header}")

                if iterationi == 1 and len(vari.preFunctions_initialize_implicate):
                    self._call_pre_post_functions(
                        vari.preFunctions_initialize_implicate, initialize=True
                    )

                self._call_pre_post_functions(vari.preFunctions)

                #   If first, iteration, exclude all subsequent variables from this imputation
                if iterationi == 1:
                    if len(vari.predictors_exclude_first_iteration):
                        self.logging.info(
                            f"        Overriding default exclusion list on first pass: {vari.predictors_exclude_first_iteration}"
                        )
                        exclude_list = vari.predictors_exclude_first_iteration
                    else:
                        exclude_list = self._subsequent_variables(
                            variable_index=variable_index - 1
                        )

                        self.logging.info(
                            f"        Excluding subsequent variables from first pass: {exclude_list}"
                        )

                    if len(exclude_list) > 0:
                        vari = deepcopy(vari)
                        vari.exclude_variables_from_models(
                            df=self.parent.df, additional_exclude=exclude_list
                        )
                self._impute_variable(
                    variable=vari,
                    iteration=iterationi,
                    variable_index=variable_index,
                    path_diagnostics=f"{self.path_logs}/{self.status_iteration:03d}.{variable_index:04d}.{vari.impute_var}.log",
                )

                self._call_pre_post_functions(vari.postFunctions)

                self.logging.info(f"     END:      {vari.impute_var}")
                self.logging.info(
                    "#########################################################"
                )
                self.logging.info("\n\n\n\n\n\n\n\n\n\n")

                if self.parent.save_every_variable and not (
                    self.parent.save_every_iteration
                    and variable_index != (len(self.parent.variables) - 1)
                ):
                    self.status_variable = variable_index + 1

                    self.status_iteration_complete = (
                        len(self.parent.variables) == variable_index
                    )
                    self.save()

                did_something = True

        if did_something:
            if (
                "By"
                in self.df_summary_stats[iterationi].lazy().collect_schema().names()
            ):
                col_reorder = (
                    self.df_summary_stats[iterationi].lazy().collect_schema().names()
                )
                col_reorder.remove("By")
                col_reorder.insert(2, "By")
                self.df_summary_stats[iterationi] = self.df_summary_stats[
                    iterationi
                ].select(col_reorder)

            if self.status_variable == len(self.parent.variables):
                self._iteration_summary_stats(iterationi)

                self.logging.info(
                    "*********************************************************"
                )
                self.logging.info(
                    "*********************************************************"
                )
                self.logging.info(f"END: Iteration #{self.status_iteration}")
                self.logging.info(
                    "*********************************************************"
                )
                self.logging.info(
                    "*********************************************************"
                )

                self.status_iteration_complete = True

            if (
                self.parent.save_every_iteration
                and iterationi != self.parent.n_iterations
                and self.status_iteration_complete
            ):
                #   Save (if not finished, since then it'll save later)
                self.save()

        return did_something

    def _impute_variable(
        self,
        variable: Variable,
        iteration: int,
        variable_index: int,
        path_diagnostics: str = "",
    ) -> None:
        """
        Run the imputation model for a single variable.

        Parameters
        ----------
        variable : Variable
            The variable (SRMI.Variable) to be imputed
        path_diagnostics : str, optional
            Where to save the imputation output log. The default is "".

        Returns
        -------
        None

        """
        if self.parent.bayesian_bootstrap:
            self.df = bayes_bootstrap_weights(
                df=self.df,
                weight=self.parent.weight,
                prefix="bbweight__",
                n_replicates=1,
                sum_to=safe_height(self.df),
            )
            weight = "bbweight__1"
        else:
            weight = self.parent.weight

        impute = Impute(
            df=self.df,
            parent=self.parent,
            variable=variable,
            index=self.parent.index,
            variable_number=variable_index,
            implicate_number=self.number,
            weight=weight,
            path_diagnostics=path_diagnostics,
        )

        self.df = impute.run()

        df_post_impute_statistics = impute.df_post_impute_statistics

        if df_post_impute_statistics is not None:
            cols_stats = (
                df_post_impute_statistics.lazy().collect_schema().names()
            )

            clear_name = []
            clear_name.append(
                pl.when(pl.col("#") == 0)
                .then(pl.col("Variable"))
                .otherwise(pl.lit(None))
                .alias("Variable")
            )

            if "By" in cols_stats:
                clear_name.append(
                    pl.when(pl.col("#").mod(4) == 0)
                    .then(pl.col("By"))
                    .otherwise(pl.lit(None))
                    .alias("By")
                )

            with_ignore = [
                pl.lit(variable.impute_var).alias("ignore_Variable"),
                pl.lit(variable_index).alias("ignore_#"),
            ]
            cols_stats = cols_stats + ["ignore_Variable", "ignore_#"]
            df_post_impute_statistics = (
                df_post_impute_statistics.lazy()
                .collect()
                .with_row_index(name="#")
                .with_columns(with_ignore)
                .with_columns(clear_name)
                .with_columns(
                    pl.when(pl.col("#") == 0)
                    .then(pl.lit(variable_index))
                    .otherwise(pl.lit(None))
                    .alias("#")
                )
                .select(["#"] + cols_stats)
            )

            if iteration not in self.df_summary_stats.keys():
                self.df_summary_stats[iteration] = df_post_impute_statistics
            else:
                self.df_summary_stats[iteration] = concat_wrapper(
                    [self.df_summary_stats[iteration], df_post_impute_statistics],
                    how="diagonal",
                )

        if self.parent.bayesian_bootstrap:
            self.df = self.df.drop(weight)

        del impute

    def _call_pre_post_functions(
        self,
        function_list: list[
            Variable.PrePost.Function | Variable.PrePost.PolarsExpression
        ]
        | None = None,
        initialize: bool = False,
    ):
        if len(function_list):
            for fi in function_list:
                if type(fi) is Variable.PrePost.Function:
                    name = fi.delegate.__name__
                    logger.info(f"\n\nCalling {name}")
                    if initialize:
                        fi.call(self)
                    else:
                        self.df = fi.call(self.df)
                elif type(fi) is Variable.PrePost.PolarsExpression:
                    logger.info(
                        f"Updating data according to polars expression: {fi.expression}"
                    )
                    self.df = fi.call(self.df)

    def df_full(
        self, drop_flags: bool = False, with_appended_cols: bool = False
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Combine imputed variables with the full dataset.

        Combine the information being imputed for this implicate
        with the full dataset.  Convenience function to make
        easier to work with this implicate's data without storing
        duplicate information

        Parameters
        ----------
        drop_flags : bool, optional
            Whether to drop imputation flags, by default False
        with_appended_cols : bool, optional
            Whether to include additionally saved columns, by default False

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            Complete dataframe with imputed and non-imputed variables
        """

        df = self.df
        columns = safe_columns(df)
        replace_list = list(set(columns).difference(self.parent.index))

        drop_list = []
        # Also drop ___imp_missing variables
        var_index = 0
        for vari in self.parent.variables:
            var_index += 1
            if drop_flags:
                drop_list.append(f"___imp_missing_{vari.impute_var}_{var_index}")

        if len(drop_list) > 0:
            df = drop_if_exists(df=df, columns=drop_list)

        appended_data = []
        appended_prefixes = []
        if with_appended_cols:
            path_glob = f"{self.path_appended_stat}/*.parquet"

            file_list = glob(path_glob)
            if len(file_list):
                #   logger.info(f"Load files and merge on {self.parent.index}")
                d_file_prefix = asterisk_matched_substring(
                    pattern=path_glob, input_list=file_list
                )

                logger.info("Loading the following additional file(s):")
                for filei in file_list:
                    logger.info(f"     {d_file_prefix[filei]}")
                    appended_prefixes.append(f"{d_file_prefix[filei]}_")

                appended_data = file_list

        return join_list(
            [drop_if_exists(df=self.parent.df, columns=replace_list), df]
            + appended_data,
            on=self.parent.index,
            how="left",
            prefixes=["", ""] + appended_prefixes,
        )

    def save_appended_cols_to_implicate(
        self, df: pl.LazyFrame | pl.DataFrame, columns: list[str], name: str
    ):
        """
        Save additional columns to be merged with implicate results.
            I.e., we want to calculate the SPM from the re-imputed data
            and save the results to be loaded with the implicate later

        Parameters
        ----------
        df : pl.LazyFrame | pl.DataFrame
            Dataframe containing the columns to save
        columns : list[str]
            Column names to save
        name : str
            Name identifier for the saved column set
            This will be the file name and the prefix of the variables
            when loaded with the implicate in with_appended_cols
        """

        #   Run through wildcard check mapping to columns
        #   Remove the prefix if it's already there
        rename = {}
        append_cols = []
        for coli in columns:
            if coli.startswith(f"{name}_"):
                fromi = coli
                toi = coli[len(f"{name}_") :]

                rename[fromi] = toi
                append_cols.append(toi)
            else:
                append_cols.append(coli)

        if len(rename):
            df = df.rename(rename)

        save_cols = columns_from_list(df=df, columns=self.parent.index + append_cols)
        create_folders_if_needed([self.path_appended_stat], quietly=True)
        save_path = f"{self.path_appended_stat}/{name}.parquet"

        df.select(save_cols).lazy().sink_parquet(save_path)

    def _subsequent_variables(self, variable_index: int = 0) -> list[str]:
        """
        Get a list of all variables after the current one in the imputation model

        Parameters
        ----------
        variable_index : int, optional
            What number variable is this. The default is 0.

        Returns
        -------
        list[str]
            List of variables after this.  Used by SRMI to drop the subsequent
            variables on the first iteration.

        """
        out_list = []
        for ivar in range(variable_index + 1, len(self.parent.variables)):
            vari = self.parent.variables[ivar]
            out_list.append(vari.impute_var)

            if "donate_list" in list(vari.parameters.keys()):
                if len(vari.parameters["donate_list"]):
                    out_list.extend(vari.parameters["donate_list"])

        return out_list

    def _iteration_summary_stats(self, iterationi: int):
        #   Get the final summary stats for each variable
        var_list = (
            self.df_summary_stats[iterationi]
            .lazy()
            .collect()
            .with_columns(
                pl.col("ignore_#").min().over("ignore_Variable").alias("ignore_min_#")
            )
            .filter(
                pl.col("Variable").is_not_null()
                & (pl.col("ignore_min_#") == pl.col("ignore_#"))
            )
            .select("ignore_min_#", "Variable")
            .unique()
            .sort("ignore_min_#")["Variable"]
            .to_list()
        )

        if self.parent.imputation_stats is not None:
            stats_to_calculate = self.parent.imputation_stats
        else:
            stats_to_calculate = Impute._post_impute_statistics_items()

        statistics = Statistics(stats=stats_to_calculate, columns=var_list)
        stats_final_iteration = StatCalculator(
            df=self.df, statistics=statistics, display=False, round_output=False
        )

        n_ignorej = (
            self.df_summary_stats[iterationi]
            .select(pl.col("ignore_#").max())
            .lazy()
            .collect()
            .item(0, 0)
        )
        df_stats_final = (
            stats_final_iteration.df_estimates.lazy()
            .collect()
            .with_row_index(name="ignore_#")
            .with_columns(pl.col("ignore_#") + n_ignorej)
            .with_columns(
                [
                    pl.lit(True).alias("ignore_final"),
                    pl.col("Variable").alias("ignore_Variable"),
                ]
            )
        )

        self.df_summary_stats[iterationi] = concat_wrapper(
            [self.df_summary_stats[iterationi], df_stats_final], how="diagonal"
        )

        #   For displaying, put it in the StatCalculator (for rounding and printing)
        stats_calc = StatCalculator(round_output=True)

        drop_cols = [
            coli
            for coli in self.df_summary_stats[iterationi]
            .lazy()
            .collect_schema()
            .names()
            if coli.startswith("ignore_")
        ]
        if len(drop_cols):
            stats_calc.df_estimates = self.df_summary_stats[iterationi].drop(
                drop_cols
            )

        self.logging.info(f"Imputation statistics after iteration #{iterationi}")
        stats_calc.rounding.round_digits = 4
        stats_calc.rounding.round_all = True
        stats_calc.print(round_output=True, estimates_per_page=40, sub_log=self.logging)

    @property
    def path_implicate(self):
        """
        Folder where the implicate data saved

        Returns
        -------
        str
            Path.

        """
        return f"{self.parent.path_model}/{self.number}"

    @property
    def path_logs(self):
        """
        Folder where the implicate logs

        Returns
        -------
        str
            Path.

        """
        return f"{self.parent.path_model}/logs/{self.number}"

    @property
    def path_appended_stat(self) -> str:
        """
        Path for appended stats

        Parameters
        ----------
        name : str
            Name of appended stats group

        Returns
        -------
        str

        """

        return f"{self.parent.path_model}/appended/{self.number}"

    def df_full_summary_stats(
        self, print_table: bool = False, print_by_variable: bool = True
    ):
        df_processed = []

        for keyi, valuei in self.df_summary_stats.items():
            df_processed.append(
                valuei.with_columns(cs.boolean().cast(pl.Int8))
                .with_columns(pl.lit(keyi).alias("ignore_Iteration"))
                .lazy()
                .collect()
                .with_row_index(name="row_number")
            )

        df_out = (
            concat_wrapper(df_processed, how="diagonal")
            .with_columns(
                pl.col("ignore_#").min().over("ignore_Variable").alias("ignore_min_#")
            )
            .sort(["ignore_min_#", "ignore_Iteration", "ignore_#", "row_number"])
            .filter(
                pl.col("mean").is_not_null()
                | (
                    pl.col("row_number")
                    == pl.col("row_number").max().over("ignore_Variable")
                )
            )
            .drop(["row_number"])
            .with_columns(
                pl.when(pl.col("#").is_not_null() | (pl.col("ignore_final") == 1))
                .then(pl.col("ignore_Iteration"))
                .otherwise(pl.lit(None))
                .alias("Iteration")
            )
        )

        df_out = df_out.lazy().collect()

        df_final = df_out.filter(pl.col("ignore_final") == 1)

        df_out = df_out.filter(pl.col("ignore_final").is_null())

        col_ordered = columns_from_list(df=df_out, columns="*", exclude=["ignore_*"])
        col_ordered.remove("Iteration")
        col_ordered = ["Iteration"] + col_ordered

        if print_table:
            stats_calc = StatCalculator()
            stats_calc.rounding.round_digits = 4
            stats_calc.rounding.round_all = True

            if print_by_variable:
                variable_list = (
                    df_out.filter(
                        pl.col("Variable").is_not_null()
                        & (pl.col("ignore_min_#") == pl.col("ignore_#"))
                    )
                    .select("ignore_min_#", "Variable")
                    .unique()
                    .sort("ignore_min_#")["Variable"]
                    .to_list()
                )

                for vari in variable_list:
                    logger.info(f"\n\n{vari}")
                    stats_calc.df_estimates = (
                        df_out.filter(pl.col("ignore_Variable") == vari)
                        .select(col_ordered)
                        .drop("Variable")
                        .filter(pl.col("mean").is_not_null())
                    )
                    stats_calc.print(round_output=True, sub_log=self.logging)
            else:
                stats_calc.df_estimates = df_out.select(col_ordered)
                stats_calc.print(round_output=True, sub_log=self.logging)

            logger.info("\n\nFinal Estimates by Iteration")
            drop_list = ["#", "Imputed", "n"]
            drop_list = list(
                set(drop_list).intersection(
                    df_final.lazy().collect_schema().names()
                )
            )

            stats_calc.df_estimates = df_final.select(col_ordered)
            if len(drop_list):
                stats_calc.df_estimates = stats_calc.df_estimates.drop(drop_list)
            stats_calc.print(round_output=True, sub_log=self.logging)

        df_out = df_out.select(col_ordered)
        df_final = df_final.select(col_ordered)

        drb_round_table(df_final.drop("Imputed")).lazy().collect().write_csv(
            f"{self.parent.path_model}/{self.number}.final_summary_stats.csv"
        )

        df_out = concat_wrapper([df_out, df_final], how="diagonal")

        return df_out

    @property
    def paths_full(self) -> list[str]:
        paths = [f"{self.path_implicate}.srmi.implicate"]

        path_glob = f"{self.path_appended_stat}/*.parquet"
        paths.extend(glob(path_glob))

        return paths
