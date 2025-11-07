from __future__ import annotations
from typing import TYPE_CHECKING

import os
import logging
import gc
import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT
import polars as pl
import numpy as np
from copy import deepcopy
import formulaic

#   Nearest neighbor search using sklearn
from sklearn.neighbors import KDTree
from scipy.special import expit
from scipy.stats import norm

from ..utilities.logging import set_logging
from ..utilities.inputs import create_folders_if_needed


from ..utilities.dataframe import (
    safe_height,
    print_longer_table,
    join_list,
    concat_wrapper,
    NarwhalsType,
    drop_if_exists,
    safe_upcast_list,
    columns_from_list,
    safe_columns,
)
from ..utilities.compress import compress_df
from ..utilities.formula_builder import FormulaBuilder

from ..statistics.basic_calculations import calculate_by
from ..statistics.statistics import Statistics
from ..statistics.calculator import StatCalculator


from ..utilities.dataframe import NarwhalsType, join_list, concat_wrapper

from ..utilities.random import RandomNumberGenerator, generate_seed

from ..utilities.rounding import drb_round_table, first_digit_position

from .utilities.draw_from_quantiles import DrawFromQuantileVectors
from .utilities.lightgbm_wrapper import Survey_kit_Lightgbm as kit_lightgbm
from .variable import Variable
from .parameters import Parameters
from .selection import Selection

if TYPE_CHECKING:
    from .srmi import SRMI

from .. import logger


class Impute:
    """
    Manages the imputation process for a single variable.

    This class coordinates the specific imputation method (regression, LightGBM,
    hot deck, etc.) for one variable in one implicate.

    Parameters
    ----------
    df : IntoFrameT
        The data needed to run the imputation model
    parent : SRMI
        Reference to the parent SRMI instance
    variable : Variable
        SRMI.Variable object with imputation specifications
    index : list
        Unique identifier columns for observations
    variable_number : int
        Position of this variable in the imputation sequence
    implicate_number : int
        Current implicate number
    weight : str, optional
        Weight variable name, by default ""
    path_diagnostics : str, optional
        Path for saving diagnostic logs, by default ""
    """

    def __init__(
        self,
        df: IntoFrameT,
        parent: SRMI,
        variable: Variable,
        index: list,
        variable_number: int,
        implicate_number: int,
        weight: str = "",
        path_diagnostics: str = "",
    ):
        self.df = df
        self.parent = parent
        #   Put a copy of the variable here, but
        #       only a copy, since it might get edited and
        #       I don't want to affect the original variable object
        self.variable = deepcopy(variable)
        self.index = index
        self.weight = weight

        self.original_variable = variable

        if path_diagnostics != "":
            #   Make sure the log for this variable is not also writing to prior variables
            if "SRMI_Impute_Variable" in logging.root.manager.loggerDict.keys():
                del logging.root.manager.loggerDict["SRMI_Impute_Variable"]
            self.logging = set_logging(
                path_log=os.path.normpath(path_diagnostics),
                to_console=not self.parent.parallel,
                force=True,
                name="SRMI_Impute_Variable",
                level=logging.INFO,
            )
        else:
            self.logging = logger

        self.df_post_impute_statistics = None
        self.current_by = {}

        self.variable_number = variable_number
        self.implicate_number = implicate_number

    def __del__(self):
        #   Remove circular reference to SRMI
        self.parent = None

    def run(self) -> IntoFrameT:
        """
        Execute the imputation for this variable.

        Routes to the appropriate imputation method based on variable.modeltype
        and handles by-group processing if specified.

        Returns
        -------
        IntoFrameT
            Updated dataframe with imputed values
        """

        #   Don't do things separately for each by group in these
        if self.variable.modeltype == Variable.ModelType.StatMatch:
            df = self.statmatch()

        elif self.variable.modeltype == Variable.ModelType.HotDeck:
            df = self.hotdeck()

        else:
            if not self.variable.selection.select_within_by:
                self._run_selection()

            #   Do the imputation separately for each by group (if applicable)
            if len(self.variable.By) > 0:
                df_by = self.df.lazy().collect().partition_by(self.variable.By)
            else:
                df_by = [self.df]

            for idf in range(len(df_by)):
                if len(self.variable.By) > 0:
                    self.current_by = (
                        NarwhalsType(df_by[idf])
                        .to_polars()
                        .select(self.variable.By)
                        .head(1)
                        .row(0, named=True)
                    )
                    self.logging.info(f"     By: {self.current_by}")
                if self.variable.selection.select_within_by:
                    self._run_selection(df=df_by[idf])

                if self.variable.modelfunction is not None:
                    df_by[idf] = self.variable.modelfunction(self, df=df_by[idf])
                elif self.variable.modeltype == Variable.ModelType.Regression:
                    df_by[idf] = self.regression(df=df_by[idf])
                elif self.variable.modeltype == Variable.ModelType.pmm:
                    df_by[idf] = self.pmm(df=df_by[idf])
                elif self.variable.modeltype == Variable.ModelType.LightGBM:
                    df_by[idf] = self.lightgbm(df=df_by[idf])
                elif self.variable.modeltype == Variable.ModelType.NearestNeighbor:
                    df_by[idf] = self.nearestneighbor(df=df_by[idf])
                # elif self.variable.modeltype == Variable.ModelType.TwoSampleRegression:
                #     df_by[idf] = self.two_sample_regression(df=df_by[idf])
                self.logging.info("\n\n\n\n")
            if len(df_by) == 1:
                df = df_by[0]
            else:
                self.logging.info(
                    f"     Putting the partitioned file back together for {self.variable.By}"
                )
                df = concat_wrapper(df_by, how="diagonal")

        return df

    def _run_selection(self, df: IntoFrameT | None = None):
        if df is None:
            df = self.df

        if self.variable.selection.method != Selection.Method.No:
            self.logging.info(
                f"     Running variable selection: {self.variable.selection.method}"
            )

            [fb, _, _] = self.variable.process_model(df=self.df, NoConstant=True)
            selected_model = self.variable.selection.run(
                df=self.df, y=self.variable.impute_var, formula=fb.formula
            )

            self.variable = deepcopy(self.original_variable)
            self.variable.model = selected_model

    ##########################################################
    ##########################################################
    #   Imputation functions - Start
    ##########################################################
    ##########################################################
    def statmatch(self, df: IntoFrameT | None = None) -> IntoFrameT:
        """
        Perform statistical matching imputation.

        Randomly matches donors and recipients within cells defined
        by matching variables.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Input dataframe, uses self.df if None

        Returns
        -------
        IntoFrameT
            Dataframe with statistically matched values
        """
        if df is None:
            df = self.df

        self.logging.info("     Imputation using statistical matching")
        #   Vars to keep
        keep_vars = []
        #   Vars to donate
        donate_vars = [self.variable.impute_var]

        #   Keep the merge keys
        keep_vars.extend(self.index)

        #   Keep the variable to be imputed
        keep_vars.append(self.variable.impute_var)

        #   Keep any additional variables to be imputed
        if len(self.variable.parameters["donate_list"]) > 0:
            keep_vars.extend(self.variable.parameters["donate_list"])
            donate_vars.extend(self.variable.parameters["donate_list"])

        #   Model variables
        for modeli in self.variable.parameters["model_list"]:
            keep_vars.extend(modeli)

        if len(self.variable.By) > 0:
            keep_vars.extend(self.variable.By)

        donate_vars = list(set(donate_vars))
        keep_vars = list(set(keep_vars))

        df_donors = self.df_model(df=df, keep_vars=keep_vars, drop_imputed=True)
        df_recipients = self.df_impute(df=df, keep_vars=keep_vars)

        if safe_height(df_recipients) == 0:
            self.logging.info("No rows to impute")
            return df

        nToMatch = safe_height(df_recipients)

        #   Is there a by?, if so, fall back to it lastby
        all_models_pre = self.variable.parameters["model_list"].copy()

        #   Remove any duplicates
        all_models = []
        for modi in all_models_pre:
            modi.sort()

            if modi not in all_models:
                all_models.append(modi)

        if len(self.variable.By) > 0:
            all_models.append([])

        self.logging.info(all_models)
        for modeli in all_models:
            modeli = modeli.copy()

            #   Add the by group to the model, if needed
            if len(self.variable.By) > 0:
                modeli.extend(self.variable.By)

            self.current_by = modeli

            if safe_height(df_recipients) > 0:
                (df_matched, df_recipients) = self._statmatch_merge(
                    df_donors=df_donors,
                    df_recipients=df_recipients,
                    donate_vars=donate_vars,
                    model=modeli,
                )

                #   Share matched
                nMatched = safe_height(df_matched)
                shareMatched = nMatched / nToMatch
                self.logging.info("     Matches")
                self.logging.info(f"          obs =   {nMatched:,.0f}")
                self.logging.info(f"          share = {shareMatched:.4f}")

                if nMatched > 0:
                    #   Stats on the donors and recipients
                    self._post_impute_statistics(
                        df_model=df_donors,
                        df_impute=df_matched,
                        donate_vars=donate_vars,
                    )

                    #   Merge results onto main file
                    df = self._merge_imputes_to_df(
                        df_imputed=df_matched, df=df, merge_list=donate_vars
                    )

                    #   Most common matches
                    self.logging.info("     Most common matches: ")
                    index_renamed = [f"donor_{vari}" for vari in self.index]
                    df_matchcount = nw.from_native(
                        calculate_by(
                            df=(
                                nw.from_native(df_matched).with_columns(
                                    nw.lit(1).alias("nDonors")
                                )
                            ),
                            column_stats={"nDonors": ["count"]},
                            by=index_renamed,
                            no_suffix=True,
                        )
                    ).sort(["nDonors"], descending=True)

                    self.logging.info(nw.from_native(df_matchcount).head(5).to_native())
                self.logging.info("\n\n")

        #   Done - return the dataframe
        return df

    def lightgbm(self, df: IntoFrameT | None = None) -> IntoFrameT:
        """
        Perform LightGBM-based imputation.

        Uses gradient boosting for prediction, with options for quantile
        regression and PMM for final value assignment.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Input dataframe, uses self.df if None

        Returns
        -------
        IntoFrameT
            Dataframe with LightGBM-imputed values
        """
        if df is None:
            df = self.df

        self.logging.info("     Imputation using LightGBM")

        df_impute = self.df_impute(df=df)
        df_model = self.df_model(df=df)

        if safe_height(df_impute) == 0:
            self.logging.info("No rows to impute")
            return df

        parameters = self.variable.parameters["parameters"]
        tune_hyperparameter_path = self.variable.parameters["tune_hyperparameter_path"]

        lgbm_model = kit_lightgbm(
            df=df_model,
            y=self.variable.impute_var,
            formula=self.variable.model,
            weight=self.weight,
            parameters=parameters,
        )

        if tune_hyperparameter_path != "":
            lgbm_model.load_tuned_parameters(
                path=f"{tune_hyperparameter_path}/{self.variable.impute_var}.pickle"
            )

        #   Run the LightGBM model
        if len(self.variable.parameters["quantiles"]) > 0:
            #   Run len(quantiles) models for each percentile
            df = self._lightgbm_quantiles(
                lgbm_model=lgbm_model, df=df, df_model=df_model, df_impute=df_impute
            )
        else:
            df = self._lightgbm_simple(
                lgbm_model=lgbm_model, df=df, df_model=df_model, df_impute=df_impute
            )
        return df

    def pmm(self, df: IntoFrameT | None = None) -> IntoFrameT:
        """
        Perform predictive mean matching imputation.

        Fits regression model, finds nearest neighbors based on predicted values,
        and randomly selects donor values.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Input dataframe, uses self.df if None

        Returns
        -------
        IntoFrameT
            Dataframe with PMM-imputed values
        """
        if df is None:
            df = self.df

        regression_type = self.variable.parameters["model"]
        self.logging.info(
            f"     Imputation using {regression_type.name} regression with PMM matching"
        )

        [fb, _, model_vars] = self.variable.process_model(df)
        # fb = FormulaBuilder(df=df)
        # fb.formula = f"{self.variable.impute_var}{self.variable.model}"
        # model_vars = fb.columns
        donate_vars = [self.variable.impute_var]

        keep_vars = model_vars + self.index

        if self.weight != "":
            keep_vars.append(self.weight)
            model_vars.append(self.weight)

        df_impute = self.df_impute(df=df, keep_vars=keep_vars)

        if safe_height(df_impute) == 0:
            self.logging.info("No rows to impute")
            return df

        #   Any other variables donated?
        if len(self.variable.parameters["donate_list"]) > 0:
            keep_vars.extend(self.variable.parameters["donate_list"])
            donate_vars.extend(self.variable.parameters["donate_list"])

        df_model = self.df_model(df=df, keep_vars=keep_vars)

        regmodel = self.variable.parameters["model"]

        (df_pmm_model, df_pmm_leave_out) = self._pmm_leave_out(df_model=df_model)

        [df_pmm_model, df_impute, _, df_pmm_leave_out] = self._run_regression(
            df_model=df_pmm_model,
            df_impute=df_impute,
            model_vars=model_vars,
            formula=fb.formula,
            regmodel=regmodel,
            df_pmm_leave_out=df_pmm_leave_out,
        )

        if df_pmm_leave_out is not None:
            df_impute = df_impute.sort(self.index)

            # self._pmm_adjust_leave_out(df_impute=df_impute,
            #                            p_pmm_model=df_impute.select("___prediction"),
            #                            p_pmm_leave_out=df_pmm_leave_out.select("___prediction"),
            #                            p_col="___prediction")

            df_model = concat_wrapper([df_pmm_model, df_pmm_leave_out], how="diagonal")
        else:
            df_model = df_pmm_model

        df_impute = self._find_nearest_neighbor_by(
            df_model=df_model,
            df_impute=df_impute,
            knearest=self.variable.parameters["knearest"],
            match_on=["___prediction"],
            donate_vars=donate_vars,
            donate_by=self.variable.parameters["donate_by"],
        )

        self._post_impute_statistics(
            df_model=df_model, df_impute=df_impute, donate_vars=donate_vars
        )
        df = self._merge_imputes_to_df(
            df_imputed=df_impute, df=df, merge_list=donate_vars
        )
        return df

    def regression(self, df: IntoFrameT | None = None) -> IntoFrameT:
        """
        Perform regression-based imputation.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Input dataframe, uses self.df if None

        Returns
        -------
        IntoFrameT
            Dataframe with regression-based imputed values
        """

        if df is None:
            df = self.df

        regression_type = self.variable.parameters["model"]
        self.logging.info(f"     Imputation using {regression_type.name} regression")

        [fb, _, model_vars] = self.variable.process_model(df)
        # fb = FormulaBuilder(df=df)
        # fb.formula = f"{self.variable.impute_var}{self.variable.model}"
        # model_vars = fb.columns
        keep_vars = model_vars + self.index

        if self.weight != "":
            keep_vars.append(self.weight)
            model_vars.append(self.weight)

        df_model = self.df_model(df=df, keep_vars=keep_vars)

        df_impute = self.df_impute(df=df, keep_vars=keep_vars)

        if safe_height(df_impute) == 0:
            self.logging.info("No rows to impute")
            return df

        regmodel = self.variable.parameters["model"]
        errordraw = self.variable.parameters["error"]

        if errordraw == Parameters.ErrorDraw.pmm:
            (df_pmm_model, df_pmm_leave_out) = self._pmm_leave_out(df_model=df_model)
        else:
            df_pmm_model = df_model
            df_pmm_leave_out = None

        [df_pmm_model, df_impute, _, df_pmm_leave_out] = self._run_regression(
            df_model=df_pmm_model,
            df_impute=df_impute,
            model_vars=model_vars,
            formula=fb.formula,
            regmodel=regmodel,
            df_pmm_leave_out=df_pmm_leave_out,
        )

        if df_pmm_leave_out is not None:
            df_impute = df_impute.sort(self.index)

            # self._pmm_adjust_leave_out(df_impute=df_impute,
            #                            p_pmm_model=df_impute.select("___prediction"),
            #                            p_pmm_leave_out=df_pmm_leave_out.select("___prediction"),
            #                            p_col="___prediction")
            df_model = concat_wrapper([df_pmm_model, df_pmm_leave_out], how="diagonal")
        else:
            df_model = df_pmm_model

        df_impute = self._regression_draw_errors(
            df_model=df_model,
            df_impute=df_impute,
            regmodel=regmodel,
            errordraw=errordraw,
        )

        donate_vars = None
        if errordraw == Parameters.ErrorDraw.pmm:
            donate_vars = [self.variable.impute_var]
            if "donate_list" in self.variable.parameters:
                if len(self.variable.parameters["donate_list"]) > 0:
                    donate_vars.extend(self.variable.parameters["donate_list"])

        self._post_impute_statistics(
            df_model=df_model, df_impute=df_impute, donate_vars=donate_vars
        )
        df = self._merge_imputes_to_df(
            df_imputed=df_impute, df=df, merge_list=self.variable.impute_var
        )

        return df

    def hotdeck(self, df: IntoFrameT | None = None) -> IntoFrameT:
        """
        Perform hot deck imputation.

        Manages arrays of donor values and sequentially assigns them
            to recipients within matching cells.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Input dataframe, uses self.df if None

        Returns
        -------
        IntoFrameT
            Dataframe with hot deck imputed values
        """
        if df is None:
            df = self.df

        self.logging.info("     Imputation using hot deck")
        #   Vars to keep
        keep_vars = []
        #   Vars to donate
        donate_vars = [self.variable.impute_var]

        #   Keep the merge keys
        keep_vars.extend(self.index)

        #   Keep the variable to be imputed
        keep_vars.append(self.variable.impute_var)

        #   Keep any additional variables to be imputed

        if len(self.variable.parameters["donate_list"]) > 0:
            keep_vars.extend(self.variable.parameters["donate_list"])
            donate_vars.extend(self.variable.parameters["donate_list"])

        #   Model variables
        for modeli in self.variable.parameters["model_list"]:
            keep_vars.extend(modeli)

        if len(self.variable.By) > 0:
            keep_vars.extend(self.variable.By)

        donate_vars = list(set(donate_vars))
        keep_vars = list(set(keep_vars))

        df_donors = self.df_model(df=df, keep_vars=keep_vars, drop_imputed=True)
        df_recipients = self.df_impute(df=df, keep_vars=keep_vars)

        if safe_height(df_recipients) == 0:
            self.logging.info("No rows to impute")
            return df
        nToMatch = safe_height(df_recipients)

        #   Is there a by?, if so, fall back to it lastby
        all_models_pre = self.variable.parameters["model_list"].copy()

        #   Remove any duplicates
        all_models = []
        for modi in all_models_pre:
            modi.sort()

            if modi not in all_models:
                all_models.append(modi)

        if len(self.variable.By) > 0:
            all_models.append([])

        for modeli in all_models:
            modeli = modeli.copy()
            if safe_height(df_recipients) > 0:
                #   Add the by group to the model, if needed
                if len(self.variable.By) > 0:
                    modeli.extend(self.variable.By)

                self.current_by = modeli
                self.logging.info(f"     Matching on: {modeli}")

                if self.variable.parameters["sort_by"] is None:
                    [df_matched, df_recipients] = self._hotdeck_random(
                        df_donors=df_donors,
                        df_recipients=df_recipients,
                        donate_vars=donate_vars,
                        model=modeli,
                    )
                else:
                    self.logging.error("*********************************************")
                    self.logging.error("*********************************************")
                    self.logging.error("*****     DETERMINISTIC SORTED          *****")
                    self.logging.error("*****     (ACS STYLE) HOT DECK          *****")
                    self.logging.error("*****     NOT IMPLEMENTED (MAYBE NEVER) *****")
                    self.logging.error("*********************************************")
                    self.logging.error("*********************************************")

                #   Share matched
                nMatched = safe_height(df_matched)
                shareMatched = nMatched / nToMatch
                self.logging.info("     Matches")
                self.logging.info(f"          obs =   {nMatched:,.0f}")
                self.logging.info(f"          share = {shareMatched:.4f}")

                #   Merge results onto main file
                if nMatched > 0:
                    self._post_impute_statistics(
                        df_model=df_donors,
                        df_impute=df_matched,
                        donate_vars=donate_vars,
                    )

                    df = self._merge_imputes_to_df(
                        df_imputed=df_matched, df=df, merge_list=donate_vars
                    )

                    #   Most common matches
                    self.logging.info("     Most common matches: ")
                    index_renamed = [f"donor_{vari}" for vari in self.index]
                    df_matchcount = (
                        nw.from_native(
                            calculate_by(
                                df=(
                                    nw.from_native(df_matched)
                                    .with_columns(nw.lit(1).alias("nDonors"))
                                    .to_native()
                                ),
                                column_stats={"nDonors": ["count"]},
                                by=index_renamed,
                                no_suffix=True,
                            )
                        )
                        .sort(["nDonors"], descending=True)
                        .to_native()
                    )
                    self.logging.info(
                        nw.from_native(df_matchcount)
                        .head(5)
                        .lazy()
                        .collect()
                        .to_native()
                    )
                self.logging.info("\n\n")

        #   Done - return the dataframe
        return df

    def nearestneighbor(self, df: IntoFrameT | None = None) -> IntoFrameT:
        """
        Perform nearest neighbor imputation.

        Directly matches on specified variables using k-nearest neighbors
        without intermediate regression step.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Input dataframe, uses self.df if None

        Returns
        -------
        IntoFrameT
            Dataframe with nearest neighbor imputed values
        """

        if df is None:
            df = self.df

        self.logging.info("     Imputation by  matching directly on X")

        df_impute = self.df_impute(df=df)
        df_model = self.df_model(df=df)

        if safe_height(df_impute) == 0:
            self.logging.info("No rows to impute")
            return df

        match_to = self.variable.parameters["match_to"]
        knearest = self.variable.parameters["knearest"]
        donate_vars = [self.variable.impute_var]

        if "donate_list" in self.variable.parameters:
            if len(self.variable.parameters["donate_list"]) > 0:
                donate_vars.extend(self.variable.parameters["donate_list"])

        df_impute = self._find_nearest_neighbor_by(
            df_model=df_model,
            df_impute=df_impute,
            knearest=knearest,
            match_on=match_to,
            donate_vars=donate_vars,
            donate_by=self.variable.parameters["donate_by"],
        )

        self._post_impute_statistics(
            df_model=df_model, df_impute=df_impute, donate_vars=donate_vars
        )
        df = self._merge_imputes_to_df(
            df_imputed=df_impute, df=df, merge_list=donate_vars
        )

        return df

    # def two_sample_regression(self,
    #                           df:IntoFrameT | None=None) -> IntoFrameT:
    #     """
    #     Perform two-sample regression imputation.

    #     Fits regression in one sample, applies to another, and draws
    #         from empirical distributions of first-sample.
    #         This can be run separately on one sample to save the coefficients
    #         and parameters, which can be used later on another sample
    #         (with no access to the first sample) to draw values.

    #     Parameters
    #     ----------
    #     df : IntoFrameT | None, optional
    #         Input dataframe, uses self.df if None

    #     Returns
    #     -------
    #     IntoFrameT
    #         Dataframe with two-sample regression imputed values
    #     """
    #     self.logging.info(self.variable.parameters)

    #     if self.variable.parameters["path_save"] != "":
    #         create_folders_if_needed(
    #             [self.variable.parameters["path_save"]],
    #             quietly=True
    #         )

    #         by_clause = ""
    #         for keyi, valuei in self.current_by.items():
    #             by_clause = f"{by_clause}_{keyi}_{int(valuei)}"

    #         file_name = f"{self.variable.impute_var}_{self.implicate_number}_{self.variable_number}{by_clause}"

    #         if self.variable.parameters["path_load"] != "" and self.variable.parameters["load_from_save"]:
    #             file_name_1_fallback = f"{self.variable.impute_var}_1_{self.variable_number}{by_clause}"
    #             path_save = os.path.normpath(f"{self.variable.parameters['path_load']}/{file_name}.xlsx")

    #             file_exists = os.path.isfile(path_save)
    #             if not file_exists:
    #                 #   Doesn't exist, check the fallback file
    #                 path_save = os.path.normpath(f"{self.variable.parameters['path_load']}/{file_name_1_fallback}.xlsx")
    #                 self.logging.info(f"  Falling back to {path_save} for two sample parameters")

    #         else:
    #             path_save = os.path.normpath(f"{self.variable.parameters['path_save']}/{file_name}.xlsx")
    #             path_support = os.path.normpath(f"{self.variable.parameters['path_save']}/{file_name}_support.xlsx")
    #     else:
    #         path_save = ""
    #         path_support = ""

    #     [fb,_,model_vars] = self.variable.process_model(df)

    #     bin_by = self.variable.parameters["bin_by"]

    #     donate_vars = [self.variable.impute_var]
    #     keep_vars = model_vars + self.index + bin_by

    #     if self.weight != "":
    #         keep_vars.append(self.weight)
    #         model_vars.append(self.weight)

    #     df_model = self.df_model(df=df,
    #                              keep_vars=keep_vars)
    #     df_impute = self.df_impute(df=df,
    #                                keep_vars=keep_vars)

    #     if safe_height(df_impute) == 0:
    #         self.logging.info("No rows to impute")
    #         return df

    #     draw_error = self.variable.parameters["draw_error"]
    #     is_boolean = self.variable.parameters["is_boolean"]
    #     regression_type = self.variable.parameters["model"]
    #     save_percentile_cuts = self.variable.parameters["save_percentile_cuts"]

    #     if self.variable.parameters["load_from_save"]:
    #         #   No regression, just load
    #         #   Get the y/phats
    #         df_betas = pl.from_pandas(pd.read_excel(path_save,
    #                                                 sheet_name="b"))

    #         var_list = []

    #         #   Build up the x*beta calculation
    #         beta = df_betas.filter(pl.col("term") == "(Intercept)").select("estimate")[0,0]
    #         xb = pl.lit(beta)
    #         #   self.logging.info(f"    {beta}")

    #         vars_beta = df_betas["term"].to_list()
    #         for coli in vars_beta:
    #            if coli != "(Intercept)":
    #                 var_list.append(coli)

    #                 beta = df_betas.filter(pl.col("term") == coli).select("estimate")[0,0]
    #                 xb = xb + pl.col(coli)*beta
    #                 #   self.logging.info(f"    +{beta}*{coli}")

    #         del df_betas

    #         fb.formula = self.variable.impute_var + FormulaBuilder.columns_to_formula(vars_beta,
    #                                                                                   Constant=True)

    #         [_,_,df_prediction] = model_matrix(df=df_impute,
    #                                            formula=fb.formula,
    #                                            keep_na=True,
    #                                            keep_y=False)

    #         x_orphans = set(vars_beta).difference(df_prediction.columns)
    #         for coli in x_orphans:
    #             if coli != "(Intercept)":
    #                 if ":" in coli:
    #                     self.logging.info(f"     Recalculating {coli} with model.matrix")
    #                     cols_interaction = coli.split(":")

    #                     cols_prediction = []
    #                     for colpredi in cols_interaction:
    #                         if colpredi.startswith("factor("):
    #                             colpredi = colpredi.split(")")
    #                             colpredi = colpredi[0:len(colpredi)-1]
    #                             colpredi = "".join(colpredi) + ")"

    #                         cols_prediction.append(colpredi)

    #                     formula_int = "~" + ":".join(cols_prediction)

    #                     [_,_,df_missing_pred] = model_matrix(df=df_impute,
    #                                                        formula=formula_int,
    #                                                        keep_na=True,
    #                                                        keep_y=False)
    #                     df_prediction = pl.concat([df_prediction,
    #                                                df_missing_pred.select(coli)],
    #                                               how="horizontal")
    #                     del df_missing_pred

    #                 else:
    #                     orphan_stub = coli.split(")")
    #                     orphan_stub = orphan_stub[0:len(orphan_stub)-1]
    #                     orphan_stub = "".join(orphan_stub) + ")"
    #                     cols_with_stub = [checki for checki in df_prediction.columns if checki.startswith(orphan_stub)]

    #                     self.logging.info(f"     Setting {coli} from {cols_with_stub}")
    #                     df_prediction = df_prediction.with_columns(pl.all_horizontal(pl.col(cols_with_stub).not_()).alias(coli))

    #         df_prediction = (pl.concat([df_impute.select(self.index + [self.weight]),
    #                                     (df_prediction.select(var_list)
    #                                         .with_columns(xb.alias("___prediction")))],
    #                                    how="horizontal")
    #                              .select(["___prediction"]
    #                                      + self.index
    #                                      + [self.weight]))

    #         if regression_type == Parameters.RegressionModel.Logit:
    #             df_prediction = df_prediction.with_columns((1/(1+(-1*pl.col("___prediction")).exp())).alias("___prediction"))
    #         elif regression_type == Parameters.RegressionModel.Probit:
    #             pass
    #             #   p_hat = norm.cdf(df_prediction.select("___prediction").to_numpy())

    #         #   Get the percentile cutoffs
    #         if save_percentile_cuts:
    #             df_prediction = drb_round_table(df=df_prediction,
    #                                             columns=["___prediction"],
    #                                             round_all=False)

    #             df_hat_cuts = pl.from_pandas(pd.read_excel(path_save,
    #                                                        sheet_name="cutoffs"))

    #             cut_points = df_hat_cuts["points"].to_list()
    #         else:
    #             percentile_cuts = self._two_sample_percentiles()

    #             sc = StatCalculator(df=df_prediction,
    #                                 statistics=Statistics(stats=percentile_cuts,
    #                                                       columns=["___prediction"]),
    #                                 weight=self.weight,
    #                                 round_output=True,
    #                                 display=False)
    #             cut_points = list(sc.df_estimates.row(0)[1:])

    #         #   Assign the groups to the predictions
    #         df_prediction = ToCategorical(df=df_prediction,
    #                                       Columns=["___prediction"],
    #                                       Cutoffs=cut_points,
    #                                       ZeroGroup=False,
    #                                       MissingGroup=False,
    #                                       AsDummies=False)

    #         df_distribution = pl.from_pandas(pd.read_excel(path_save,
    #                                                        sheet_name="y_empirical"))
    #     else:

    #         self.logging.info(f"     Imputation using {regression_type.name} regression")

    #         (df_prediction,
    #          hat_cuts,
    #          df_distribution) = self._two_sample_get_predictions(df_model=df_model,
    #                                                              df_impute=df_impute,
    #                                                              model_vars=model_vars,
    #                                                              fb=fb,
    #                                                              path_save=path_save,
    #                                                              path_support=path_support,
    #                                                              regression_type=regression_type,
    #                                                              is_boolean=is_boolean,
    #                                                              save_percentile_cuts=save_percentile_cuts,
    #                                                              draw_error=draw_error)

    #         df_prediction = df_prediction.filter(pl.col(self.variable.impute_var).is_null())

    #     #   Draw values
    #     rng = RandomNumberGenerator()
    #     if is_boolean:
    #         self.logging.info("Prepare draws")
    #         df_prediction = (
    #             join_list(
    #                 [
    #                     df_prediction,
    #                     (
    #                         nw.from_native(df_distribution)
    #                         .select(
    #                             bin_by +
    #                             ["___prediction_cat",
    #                              "mean"]
    #                         )
    #                         .rename({"mean":"___p_hat"})
    #                     )
    #                 ],
    #                 how="left",
    #                 on=bin_by + ["___prediction_cat"]
    #             )
    #         )

    #         nw_type = NarwhalsType(df_prediction)
    #         df_prediction = concat_wrapper(
    #             [
    #                 df_prediction,
    #                 nw_type.from_polars(
    #                     pl.DataFrame({"___imputed_value":rng.uniform(low=0,high=1,size=safe_height(df_prediction))})
    #                 )
    #             ]
    #         )

    #         c_draw = nw.col("___imputed_value")
    #         c_threshold = nw.col("___p_hat")
    #         df_prediction = (
    #             nw.from_native(df_prediction)
    #             .with_columns((c_draw <= c_threshold).alias(self.variable.impute_var))
    #         )
    #     else:
    #         cols_qtiles = df_distribution.news.columns("q*")

    #         self.logging.info("Prepare draws")
    #         df_prediction = join_list(
    #             [
    #                 df_prediction,
    #                 drop_if_exists(df_distribution,
    #                                columns=["n"])
    #             ],
    #             how="left",
    #             on=bin_by + ["___prediction_cat"]
    #         )
    #         #   cols_qtiles = [f"q{i*100}" for i in self.variable.parameters["continuous_qtiles_y_cuts"]]

    #         df_prediction = draw_values(df=df_prediction,
    #                                     index=self.index,
    #                                     cols_qtiles=cols_qtiles,
    #                                     n_draws=1,
    #                                     seed=rng.integers(1,2**31-1,1)[0],
    #                                     var_stub="___imputed_value")

    #         if draw_error:
    #             c_pred = pl.col("___prediction")
    #             c_draw = pl.col("___imputed_value")
    #             df_prediction = df_prediction.with_columns((c_pred + c_draw).alias(self.variable.impute_var))
    #         else:
    #             df_prediction = (df_prediction.news.drop_if_exists(self.variable.impute_var)
    #                                           .rename({"___imputed_value":self.variable.impute_var}))

    #     self._post_impute_statistics(df_model=df_model,
    #                                  df_impute=df_prediction,
    #                                  donate_vars=donate_vars)
    #     df = self._merge_imputes_to_df(df_imputed=df_prediction,
    #                                    df=df,
    #                                    merge_list=donate_vars)
    #     return df

    ##########################################################
    ##########################################################
    #   Imputation functions - END
    ##########################################################
    ##########################################################

    ##########################################################
    ##########################################################
    #   HELPERS - LightGBM - START
    ##########################################################
    ##########################################################

    def _lightgbm_simple(
        self,
        lgbm_model: kit_lightgbm,
        df: IntoFrameT,
        df_model: IntoFrameT,
        df_impute: IntoFrameT,
    ):
        donate_vars = [self.variable.impute_var]
        if "donate_list" in self.variable.parameters:
            if len(self.variable.parameters["donate_list"]) > 0:
                donate_vars.extend(self.variable.parameters["donate_list"])

        lgbm_model.parameters["seed"] = generate_seed()

        lgbm_model.train(show_eval=False)

        df_importance = lgbm_model.importance()
        cols_feature = (
            nw.from_native(df_importance).lazy().collect()["Feature"].to_list()
        )

        statistics = Statistics(
            stats=[
                "share|missing",
                "mean",
            ],
            columns=cols_feature,
        )

        if safe_height(df_importance) == 0:
            self.logging.info(
                "**********************************************************"
            )
            self.logging.info(
                "**********************************************************"
            )
            self.logging.info(
                "**********************************************************"
            )
            self.logging.info(f"Lightgbm failed for {self.variable.impute_var}")
            self.logging.info("    Falling back to a random draw")
            self.logging.info(
                "**********************************************************"
            )
            self.logging.info(
                "**********************************************************"
            )
            self.logging.info(
                "**********************************************************"
            )

            stat_match_var = "___stat_match_var___"
            df_model = nw.from_native(df_model).with_columns(
                nw.lit(1).alias(stat_match_var)
            )
            df_impute = nw.from_native(df_impute).with_columns(
                nw.lit(1).alias(stat_match_var)
            )

            (df_impute, _) = self._statmatch_merge(
                df_donors=df_model,
                df_recipients=df_impute,
                donate_vars=donate_vars,
                model=[stat_match_var],
            )

            self._post_impute_statistics(
                df_model=df_model, df_impute=df_impute, donate_vars=donate_vars
            )
            df = self._merge_imputes_to_df(
                df_imputed=df_impute, df=df, merge_list=donate_vars
            )

            return df

        stats_model = StatCalculator(
            df=df_model, statistics=statistics, display=False, round_output=True
        )
        stats_impute = StatCalculator(
            df=df_impute, statistics=statistics, display=False, round_output=True
        )
        print_longer_table(
            df=(
                join_list(
                    [
                        df_importance,
                        (
                            nw.from_native(stats_model.df_estimates)
                            .rename(
                                {
                                    vari: f"Model\n{vari}"
                                    for vari in safe_columns(stats_model.df_estimates)
                                }
                            )
                            .rename({"Model\nVariable": "Feature"})
                            .to_native()
                        ),
                        (
                            nw.from_native(stats_impute.df_estimates)
                            .rename(
                                {
                                    vari: f"Impute\n{vari}"
                                    for vari in safe_columns(stats_model.df_estimates)
                                }
                            )
                            .rename({"Impute\nVariable": "Feature"})
                        ),
                    ],
                    how="left",
                    on=["Feature"],
                )
            ),
            logging=self.logging,
        )

        del stats_model
        del stats_impute
        del df_importance

        #   Get the predictions
        predict_impute = lgbm_model.predict(df_predict=df_impute, name="___prediction")

        predict_model = lgbm_model.predict(name="___prediction")

        df_model = concat_wrapper([df_model, predict_model], how="horizontal")

        df_impute = concat_wrapper([df_impute, predict_impute], how="horizontal")

        predict_model = (
            nw.from_native(self.variable.df_impute_original_where(df=df_model))
            .select([self.variable.impute_var, "___prediction"])
            .lazy()
            .collect()
            .to_native()
        )

        predict_impute = (
            nw.from_native(self.variable.df_impute_original_where(df=df_impute))
            .select([self.variable.impute_var, "___prediction"])
            .lazy()
            .collect()
            .to_native()
        )

        if predict_model is not None:
            if safe_height(predict_model) == 0:
                predict_model = None

        if predict_model is None:
            predict_model = (
                nw.from_native(self.variable.df_where(df=df_model))
                .select([self.variable.impute_var, "___prediction"])
                .lazy()
                .collect()
            )

        self.logging.info("Predictions")

        self.logging.info(
            pl.concat(
                [
                    predict_model.rename({"___prediction": "Model (yhat)"}).describe(),
                    predict_impute.rename({"___prediction": "Imputed (yhat)"})
                    .describe()
                    .select("Imputed (yhat)"),
                ],
                how="horizontal",
            )
        )
        self.logging.info(
            pl.concat(
                [
                    (
                        NarwhalsType(predict_model)
                        .to_polars()
                        .lazy()
                        .collect()
                        .rename({"___prediction": "Model (yhat)"})
                        .describe()
                    ),
                    (
                        NarwhalsType(predict_impute)
                        .to_polars()
                        .lazy()
                        .collect()
                        .rename({"___prediction": "Imputed (yhat)"})
                        .describe()
                        .select("Imputed (yhat)")
                    ),
                ],
                how="horizontal",
            )
        )

        #   Draw as if binary or continuous (does nothing if draw is pmm)
        if (
            type(
                nw.from_native(df_model)
                .lazy()
                .collect_schema()[self.variable.impute_var]
            )
            == nw.Boolean
        ):
            regmodel = Parameters.RegressionModel.Logit
        else:
            regmodel = Parameters.RegressionModel.OLS

        df_impute = self._regression_draw_errors(
            df_model=df_model, df_impute=df_impute, regmodel=regmodel
        )

        self._post_impute_statistics(
            df_model=df_model, df_impute=df_impute, donate_vars=donate_vars
        )
        df = self._merge_imputes_to_df(
            df_imputed=df_impute, df=df, merge_list=donate_vars
        )

        del lgbm_model
        gc.collect()
        return df

    def _lightgbm_quantiles(
        self,
        lgbm_model: kit_lightgbm,
        df: IntoFrameT,
        df_model: IntoFrameT,
        df_impute: IntoFrameT,
    ):
        if lgbm_model.parameters["objective"] != "quantile":
            self.logging.warning("RESETTING LightGBM OBJECTIVE TO QUANTILE")
            lgbm_model.parameters["objective"] = "quantile"

        predict_impute = None
        predict_model = None

        df_impute = nw.from_native(df_impute).sort(self.index).to_native()
        #   Run LightGBM for each quantile
        quantiles = self.variable.parameters["quantiles"]
        for qi in quantiles:
            self.logging.info(f"Running LightGBM for q={qi}")
            lgbm_model.parameters["alpha"] = qi

            lgbm_model.parameters["seed"] = generate_seed()
            lgbm_model.train(show_eval=False)

            #   Get the predictions
            p_impute = lgbm_model.predict(df_predict=df_impute, name=f"___p{qi}")

            p_model = lgbm_model.predict(name=f"___p{qi}")

            #   if qi == 0.5:
            #   Basic correlation
            corr = (
                NarwhalsType(
                    concat_wrapper(
                        [
                            p_model,
                            nw.from_native(df_model)
                            .select(self.variable.impute_var)
                            .to_native(),
                        ],
                        how="horizontal",
                    )
                )
                .to_polars()
                .select(pl.corr(self.variable.impute_var, f"___p{qi}"))
                .item(0, 0)
            )

            self.logging.info(
                f"     Correlation between {self.variable.impute_var} and q={qi} prediction: {corr:,.3f}"
            )
            del corr
            #   Collect the predictions
            if predict_impute is None:
                predict_impute = p_impute
                predict_model = p_model
            else:
                predict_impute = concat_wrapper(
                    [predict_impute, p_impute], how="horizontal"
                )

                predict_model = concat_wrapper(
                    [predict_model, p_model], how="horizontal"
                )
            del p_impute, p_model

            self.logging.info("\n\n")

        statistics = Statistics(
            stats=["n", "n|missing", "mean", "std", "q25", "q50", "q75"],
            columns=safe_columns(predict_impute),
        )

        predict_both = concat_wrapper(
            [
                (
                    nw.from_native(predict_model).with_columns(
                        [nw.lit("Model").alias("Sample"), nw.lit(0).alias("___sort___")]
                    )
                ),
                (
                    nw.from_native(predict_impute).with_columns(
                        [
                            nw.lit("Imputed").alias("Sample"),
                            nw.lit(1).alias("___sort___"),
                        ]
                    )
                ),
            ],
            how="diagonal",
        )

        stats_both = StatCalculator(
            df=predict_both,
            by={"Sample": ["Sample", "___sort___"]},
            statistics=statistics,
            display=False,
            round_output=True,
        )

        stats_both.df_estimates = (
            nw.from_native(stats_both.df_estimates)
            .with_columns(nw.col("Variable").str.replace("___", ""))
            .sort(["Variable", "___sort___"])
            .with_columns(
                nw.when(nw.col("___sort___") == 0)
                .then(nw.col("Variable"))
                .otherwise(nw.lit(""))
                .alias("Variable")
            )
            .drop("___sort___")
            .to_native()
        )

        stats_both.print(sub_log=self.logging)

        del predict_both
        del stats_both

        errordraw = self.variable.parameters["error"]
        if errordraw == Parameters.ErrorDraw.pmm:
            #   Get the marginal by imputing with pmm
            #       If so, use pmm on the mean
            self.logging.info(
                "Running LightGBM for the mean for estimating the marginal distribution"
            )

            (df_pmm_model, df_pmm_leave_out) = self._pmm_leave_out(df_model=df_model)

            if df_pmm_leave_out is not None:
                lgbm_model_pmm = kit_lightgbm(
                    df=df_pmm_model,
                    y=self.variable.impute_var,
                    formula=self.variable.model,
                    weight=self.weight,
                    parameters=self.variable.parameters["parameters"],
                )
            else:
                lgbm_model_pmm = lgbm_model

            if "alpha" in lgbm_model_pmm.parameters.keys():
                del lgbm_model_pmm.parameters["alpha"]
            lgbm_model_pmm.parameters["seed"] = generate_seed()
            lgbm_model_pmm.parameters["objective"] = "regression"

            donate_vars = [self.variable.impute_var]
            if "donate_list" in self.variable.parameters:
                if len(self.variable.parameters["donate_list"]) > 0:
                    donate_vars.extend(self.variable.parameters["donate_list"])

            #   Run the model
            lgbm_model_pmm.train(show_eval=False)

            #   Get the predictions
            p_impute = lgbm_model_pmm.predict(df_predict=df_impute, name="___yhat")

            df_impute = concat_wrapper([df_impute, p_impute], how="horizontal")

            # print_longer_table(df=lgbm_model_pmm.importance(),
            #                    logger=self.logging)

            df_importance = lgbm_model_pmm.importance()
            cols_feature = (
                nw.from_native(df_importance).lazy().collect()["Feature"].to_list()
            )
            statistics = Statistics(
                stats=[
                    "share|missing",
                    "mean",
                ],
                columns=cols_feature,
            )
            stats_model = StatCalculator(
                df=df_model, statistics=statistics, display=False, round_output=True
            )
            stats_impute = StatCalculator(
                df=df_impute, statistics=statistics, display=False, round_output=True
            )

            print_longer_table(
                df=(
                    join_list(
                        [
                            df_importance,
                            (
                                nw.from_native(stats_model.df_estimates)
                                .rename(
                                    {
                                        vari: f"Model\n{vari}"
                                        for vari in safe_columns(
                                            stats_model.df_estimates
                                        )
                                    }
                                )
                                .rename({"Model\nVariable": "Feature"})
                                .to_native()
                            ),
                            (
                                nw.from_native(stats_impute.df_estimates)
                                .rename(
                                    {
                                        vari: f"Impute\n{vari}"
                                        for vari in safe_columns(
                                            stats_model.df_estimates
                                        )
                                    }
                                )
                                .rename({"Impute\nVariable": "Feature"})
                                .to_native()
                            ),
                        ],
                        how="left",
                        on=["Feature"],
                    )
                ),
                logging=self.logging,
            )

            del stats_model
            del stats_impute
            del df_importance

            if df_pmm_leave_out is not None:
                p_model = lgbm_model_pmm.predict(df_predict=df_model, name="___yhat")

                p_pmm_model = lgbm_model_pmm.predict(
                    df_predict=df_pmm_model, name="___yhat"
                )
                p_pmm_leave_out = lgbm_model_pmm.predict(
                    df_predict=df_pmm_leave_out, name="___yhat"
                )

                # df_impute = self._pmm_adjust_leave_out(
                #     df_impute=df_impute,
                #     p_pmm_model=p_pmm_model,
                #     p_pmm_leave_out=p_pmm_leave_out,
                #     p_col="___yhat"
                # )
            else:
                p_model = lgbm_model_pmm.predict(name="___yhat")

            self.logging.info("Predictions")
            self.logging.info(
                pl.concat(
                    [
                        NarwhalsType(p_model)
                        .to_polars()
                        .lazy()
                        .collect()
                        .rename({"___yhat": "Model (yhat)"})
                        .describe(),
                        NarwhalsType(p_impute)
                        .to_polars()
                        .lazy()
                        .collect()
                        .rename({"___yhat": "Imputed (yhat)"})
                        .describe()
                        .select("Imputed (yhat)"),
                    ],
                    how="horizontal",
                )
            )
            df_model = concat_wrapper([df_model, p_model], how="horizontal")

            #   Basic correlation
            corr = (
                NarwhalsType(df_model)
                .to_polars()
                .lazy()
                .collect()
                .select(pl.corr(self.variable.impute_var, "___yhat"))
                .item(0, 0)
            )

            self.logging.info(
                f"     Correlation between {self.variable.impute_var} and prediction: {corr:,.3f}"
            )
            knearest = self.variable.parameters["knearest"]

            #   Get the quantile regression interpolated imputes
            #       Which are used for assigning ranks
            [_, predict_impute_values] = self._draw_interpolated_percentiles(
                df=predict_impute, percentiles=quantiles
            )

            col_0 = (
                nw.from_native(predict_impute_values).lazy().collect_schema().names()[0]
            )
            predict_impute_values = (
                nw.from_native(predict_impute_values)
                .rename({col_0: "___y_draw"})
                .to_native()
            )

            #   Append the quantile imputes to the df_impute file
            df_impute = concat_wrapper(
                [df_impute, predict_impute_values], how="horizontal"
            )

            df_impute_missing = (
                nw.from_native(df_impute)
                .filter(nw.col("___y_draw").is_missing())
                .to_native()
            )

            df_impute = (
                nw.from_native(df_impute)
                .filter(nw.col("___y_draw").is_not_missing())
                .to_native()
            )

            #   Get the pmm imputes
            #       Which determine the marginal distribution and are the
            #       actual values imputed based on the quantile ranks assigned below
            df_marginal = self._find_nearest_neighbor_by(
                df_model=df_model,
                df_impute=df_impute,
                knearest=knearest,
                match_on=["___yhat"],
                donate_vars=donate_vars,
                donate_by=self.variable.parameters["donate_by"],
            )

            df_impute = (
                nw.from_native(
                    concat_wrapper(
                        [
                            (
                                nw.from_native(df_impute)
                                .drop(donate_vars)
                                .sort("___y_draw")
                                .to_native()
                            ),
                            (
                                nw.from_native(df_marginal)
                                .drop(self.index)
                                .sort(self.variable.impute_var)
                            ),
                        ],
                        how="horizontal",
                    )
                )
                .drop(["___yhat", "___y_draw"])
                .to_native()
            )

            if safe_height(df_impute_missing) > 0:
                df_impute = concat_wrapper(
                    [df_impute, df_impute_missing], how="diagonal"
                )

            del lgbm_model_pmm
            gc.collect()
        elif errordraw == Parameters.ErrorDraw.Random:
            #   Interpolate values from the quantile predictions
            [_, predict_impute_values] = self._draw_interpolated_percentiles(
                df=predict_impute, percentiles=quantiles
            )

            col_rename = (
                nw.from_native(predict_impute_values).lazy().collect_schema().names()[0]
            )
            predict_impute_values = predict_impute_values.rename(
                {col_rename: self.variable.impute_var}
            )

            donate_vars = [self.variable.impute_var]

            df_impute = concat_wrapper(
                [
                    (nw.from_native(df_impute).drop(donate_vars).to_native()),
                    predict_impute_values,
                ],
                how="horizontal",
            )

        #   Anyone fall through?
        #   df_impute= df_impute.with_columns(pl.when(pl.col('_row_index_') <= 20).then(pl.lit(None)).otherwise(pl.col("var1")).alias("var1"))
        n_missing = safe_height(
            nw.from_native(df_impute)
            .filter(nw.col(self.variable.impute_var).is_missing())
            .to_native()
        )

        if n_missing > 0:
            df_impute_mean = (
                nw.from_native(df_impute)
                .filter(nw.col(self.variable.impute_var).is_missing())
                .to_native()
            )

            #       If so, use pmm on the mean
            self.logging.info(
                f"Running LightGBM for the mean for missing imputes for {n_missing} observations"
            )

            if lgbm_model.parameters["objective"] != "regression":
                #   Run the model
                del lgbm_model.parameters["alpha"]
                lgbm_model.parameters["objective"] = "regression"

                lgbm_model.train(show_eval=False)

            #   Get the predictions
            b_have_for_recipients = "___yhat" in safe_columns(df_impute_mean)
            if b_have_for_recipients:
                b_have_for_recipients = (
                    safe_height(
                        nw.from_native(df_impute_mean)
                        .filter(pl.col("___yhat").is_missing())
                        .to_native()
                    )
                    == 0
                )

            b_have_for_donors = (
                "___yhat" in nw.from_native(df_model).lazy().collect_schema().names()
            )
            if b_have_for_donors:
                b_have_for_donors = (
                    safe_height(
                        nw.from_native(df_model)
                        .filter(pl.col("___yhat").is_missing())
                        .to_native()
                    )
                    == 0
                )

            if not b_have_for_recipients:
                p_impute = lgbm_model.predict(df_predict=df_impute_mean, name="___yhat")

                df_impute_mean = concat_wrapper(
                    [df_impute_mean, p_impute], how="horizontal"
                )
            if not b_have_for_donors:
                p_model = lgbm_model.predict(name="___yhat")
                df_model = concat_wrapper([df_model, p_model], how="horizontal")

            knearest = self.variable.parameters["knearest"]
            df_impute_mean = self._find_nearest_neighbor_by(
                df_model=df_model,
                df_impute=df_impute_mean,
                knearest=knearest,
                match_on=["___yhat"],
                donate_vars=donate_vars,
                donate_by=self.variable.parameters["donate_by"],
            )

            df_impute = concat_wrapper(
                [
                    (
                        nw.from_native(df_impute)
                        .filter(nw.col(self.variable.impute_var).is_not_missing())
                        .to_native()
                    ),
                    (nw.from_native(df_impute_mean).drop("___yhat").to_native()),
                ],
                how="diagonal",
            )

        self._post_impute_statistics(
            df_model=df_model, df_impute=df_impute, donate_vars=donate_vars
        )
        df = self._merge_imputes_to_df(
            df_imputed=df_impute, df=df, merge_list=donate_vars
        )

        del lgbm_model
        gc.collect()
        return df

    ##########################################################
    ##########################################################
    #   HELPERS - LightGBM - START
    ##########################################################
    ##########################################################

    ##########################################################
    ##########################################################
    #   HELPERS - Regression - START
    ##########################################################
    ##########################################################
    def _regression_draw_errors(
        self,
        df_model: IntoFrameT,
        df_impute: IntoFrameT,
        regmodel: Parameters.RegressionModel,
        errordraw: Parameters.ErrorDraw | None = None,
    ):
        if errordraw is None:
            if "error" in self.variable.parameters:
                errordraw = self.variable.parameters["error"]

        if errordraw == Parameters.ErrorDraw.Random:
            rng = RandomNumberGenerator()

            if (
                regmodel == Parameters.RegressionModel.Logit
                or regmodel == Parameters.RegressionModel.Probit
            ):
                #   Draw the values for df_impute from the uniform where 1 if <= prediction

                nw_type = NarwhalsType(df_impute)
                df_impute = nw.from_native(
                    concat_wrapper(
                        [
                            df_impute,
                            nw_type.from_polars(
                                pl.from_numpy(
                                    rng.uniform(size=safe_height(df_impute)),
                                    schema={"___phat": pl.Float64},
                                )
                            ),
                        ],
                        how="horizontal",
                    ).with_columns(
                        (nw.col("___phat") <= nw.col("___prediction"))
                        .cast(nw.Boolean)
                        .alias(self.variable.impute_var)
                    )
                )
            elif regmodel == Parameters.RegressionModel.OLS:
                #   Get the sd of the errors in the model data set
                df_std = calculate_by(
                    df=(
                        nw.from_native(df_model)
                        .with_columns(
                            (
                                nw.col(self.variable.impute_var)
                                - nw.col("___prediction")
                            ).alias("___ehat")
                        )
                        .to_native()
                    ),
                    column_stats={"___ehat": ["std"]},
                    weight=self.weight,
                )

                std = nw.from_native(df_std).item(0, 0)

                #   Draw the values for df_impute
                nw_type = NarwhalsType(df_impute)
                df_impute = concat_wrapper(
                    [
                        df_impute,
                        nw_type.from_polars(
                            pl.from_numpy(
                                rng.normal(scale=std, size=safe_height(df_impute)),
                                schema={"___ehat": pl.Float64},
                            )
                        ),
                    ],
                    how="horizontal",
                )

                df_impute = (
                    nw.from_native(df_impute)
                    .with_columns(
                        (pl.col("___prediction") + nw.col("___ehat")).alias(
                            self.variable.impute_var
                        )
                    )
                    .to_native()
                )

        elif errordraw == Parameters.ErrorDraw.pmm:
            knearest = self.variable.parameters["knearest"]

            donate_vars = [self.variable.impute_var]
            if "donate_list" in self.variable.parameters:
                if len(self.variable.parameters["donate_list"]) > 0:
                    donate_vars.extend(self.variable.parameters["donate_list"])

            df_impute = self._find_nearest_neighbor_by(
                df_model=df_model,
                df_impute=df_impute,
                knearest=knearest,
                match_on=["___prediction"],
                donate_vars=donate_vars,
                donate_by=self.variable.parameters["donate_by"],
            )

        return df_impute

    def _run_regression(
        self,
        df_model: IntoFrameT,
        df_impute: IntoFrameT,
        model_vars: list,
        formula: str,
        regmodel: Parameters.RegressionModel | None = None,
        df_pmm_leave_out: IntoFrameT | None = None,
        min_n_x_var: int = 0,
    ) -> tuple[IntoFrameT, IntoFrameT, IntoFrameT | None]:
        nw_model = NarwhalsType(df_model)
        nw_impute = NarwhalsType(df_impute)

        df_model = nw_model.to_polars().lazy().collect()
        df_impute = nw_impute.to_polars().lazy().collect()

        if df_pmm_leave_out is not None:
            nw_leave_out = NarwhalsType(df_pmm_leave_out)
            df_pmm_leave_out = nw_leave_out.to_polars().lazy().collect()

        if regmodel is None:
            regmodel = self.variable.parameters["model"]

        if "random_share" in self.variable.parameters.keys():
            random_share = self.variable.parameters["random_share"]
        else:
            random_share = 1
        if random_share < 1:
            self.logging.info(f"     Using a {random_share} subsample")
            df_model = df_model.sample(fraction=random_share, seed=generate_seed())

        if type(self.variable.model) is list:
            fb = FormulaBuilder(df=df_model, formula=formula)
            vars_rhs = fb.columns_rhs
            df_model_mm = df_model.select(vars_rhs + [self.variable.impute_var])
            df_impute_mm = df_impute.select(vars_rhs + [self.variable.impute_var])

            if df_pmm_leave_out is not None:
                df_pmm_leave_out_mm = df_pmm_leave_out.select(
                    vars_rhs + [self.variable.impute_var]
                )
        else:
            f = FormulaBuilder(formula=formula)
            f.remove_constant()
            formula_obj = formulaic.Formula(f.rhs())
            df_model_mm = formula_obj.get_model_matrix(df_model, na_action="ignore")
            df_impute_mm = df_model_mm.model_spec.get_model_matrix(
                df_impute, na_action="ignore"
            )

            if df_pmm_leave_out is not None:
                df_pmm_leave_out_mm = df_model_mm.model_spec.get_model_matrix(
                    df_pmm_leave_out, na_action="ignore"
                )
                df_pmm_leave_out_mm = pl.DataFrame(
                    df_pmm_leave_out_mm.to_arrow()
                ).fill_nan(None)

            df_model_mm = pl.DataFrame(df_model_mm.to_arrow()).fill_nan(None)
            df_impute_mm = pl.DataFrame(df_impute_mm.to_arrow()).fill_nan(None)

        if min_n_x_var:
            self.logging.info(
                f"        Restricting to X variables with more than {min_n_x_var} observations != 0"
            )
            sc = StatCalculator(
                df=df_model_mm,
                statistics=Statistics(stats=["n|not0"], columns=vars_rhs),
                display=False,
                round_output=False,
            )

            vars_rhs = sc.df_estimates.filter(pl.col("n (not 0)") >= min_n_x_var)[
                "Variable"
            ].to_list()
            self.logging.info(
                f"            Dropping {sc.df_estimates.filter(pl.col('n (not 0)') < min_n_x_var)['Variable'].to_list()}"
            )

            df_model_mm = df_model_mm.select(vars_rhs)
            df_impute_mm = df_impute_mm.select(vars_rhs)

        if regmodel == Parameters.RegressionModel.OLS:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
        elif regmodel == Parameters.RegressionModel.Logit:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression()
        # elif regmodel == Parameters.RegressionModel.Probit:

        d_extra_model_args = {}
        if self.weight != "":
            d_extra_model_args["sample_weight"] = df_model[self.weight]

        model.fit(
            X=df_model_mm,
            y=df_model.select(self.variable.impute_var),
            **d_extra_model_args,
        )

        df_betas = pl.DataFrame(
            dict(
                Variable=safe_columns(df_model_mm) + ["_Intercept_"],
                Beta=[
                    float(vali)
                    for vali in list(model.coef_[0]) + [float(model.intercept_)]
                ],
            )
        )

        predict_model = pl.DataFrame(
            model.predict(df_model_mm), schema=dict(___prediction=pl.Float64)
        )
        predict_impute = pl.DataFrame(
            model.predict(df_impute_mm), schema=dict(___prediction=pl.Float64)
        )

        del df_model_mm
        del df_impute_mm

        df_model = pl.concat([df_model, predict_model], how="horizontal")
        df_impute = pl.concat([df_impute, predict_impute], how="horizontal")

        r_2 = (
            df_model.select(pl.corr(self.variable.impute_var, "___prediction")).item(
                0, 0
            )
            ** 2
        )

        self.logging.info(f"R2 = {r_2:0.4f}")
        print_longer_table(drb_round_table(df_betas), logging=self.logging)

        if df_pmm_leave_out is not None:
            predict_leave_out = pl.DataFrame(
                model.predict(df_pmm_leave_out_mm),
                schema=dict(___prediction=pl.Float64),
            )

            df_pmm_leave_out = pl.concat(
                [df_pmm_leave_out, predict_leave_out], how="horizontal"
            )

            df_pmm_leave_out = nw_leave_out.from_polars(df_pmm_leave_out)

        df_model = nw_model.from_polars(df_model)
        df_impute = nw_impute.from_polars(df_impute)
        df_betas = nw_model.from_polars(df_betas)

        return (df_model, df_impute, df_betas, df_pmm_leave_out)

    ##########################################################
    ##########################################################
    #   HELPERS - Regression - END
    ##########################################################
    ##########################################################

    ##########################################################
    ##########################################################
    #   HELPERS - Hot Deck/Stat Match - START
    ##########################################################
    ##########################################################
    def _hotdeck_random(
        self,
        df_donors: IntoFrameT,
        df_recipients: IntoFrameT,
        donate_vars: list,
        model: list,
    ) -> tuple[IntoFrameT, IntoFrameT]:
        nw_donors = NarwhalsType(df_donors)
        nw_recipients = NarwhalsType(df_recipients)
        df_donors = nw_donors.to_polars().lazy().collect()
        df_recipients = nw_recipients.to_polars().lazy().collect()

        rng = RandomNumberGenerator()
        sort_by = model.copy() + ["___random_sort"]

        #   Find the recipients that match the donors
        #       From the donors, get a dataset with 1 observation
        #           per observed set of values for the model list
        #           When linked to df_recipients, will identify the ones
        #           that have a match
        df_donor_matches = (
            df_donors.select(model)
            .with_columns(pl.lit(1).alias("___bMatched"))
            .group_by(model)
            .head(1)
        )

        #       Split the matched table into those that have a match
        #           and those that don't
        [df_recipients, df_donor_matches] = safe_upcast_list(
            [df_recipients, df_donor_matches]
        )
        df_matched = df_recipients.join(
            df_donor_matches, on=model, how="left", nulls_equal=True
        )

        #           Those that don't have a match
        df_unmatched = df_matched.filter(pl.col("___bMatched").is_null()).drop(
            "___bMatched"
        )

        #           Those that do
        df_matched = (
            df_matched.filter(pl.col("___bMatched") == True)
            .drop("___bMatched")
            .drop(list(set(donate_vars).difference(self.index)))
        )
        del df_donor_matches

        #   Create a variable to randomize the sort with hot deck cells
        df_matched = drop_if_exists(
            df=(
                pl.concat(
                    [
                        df_matched,
                        pl.from_numpy(
                            rng.uniform(size=safe_height(df_matched)),
                            schema={"___random_sort": pl.Float64},
                        ),
                    ],
                    how="horizontal",
                )
            ),
            columns=donate_vars,
        )

        #   Create {variable.parameters["n_hotdeck_array"]} frames to merge to
        #       recipients for hot deck
        #       This allows us to have an "array" of that many donor values carried
        #       around at all times

        #   The regular hot deck donors
        df_donors_hd = []
        #   A "warmed" set of donors to start off with for each model cell
        df_donors_warm = []

        #   Temporary columns to be created with the donor values
        fill_columns = []

        #   Everything is indexed from 0:n_hotdeck_array-1 in temporary columns
        #       In which the donor values are stored
        for hdi in range(self.variable.parameters["n_hotdeck_array"]):
            #   Polars expressions for the creation of the temp columns below
            donor_vars = [
                pl.col(vari).alias(f"donor_{vari}_{hdi}")
                for vari in (self.index + donate_vars)
            ]

            #   List of the temp donor columns
            fill_columns.extend(
                [f"donor_{vari}_{hdi}" for vari in (self.index + donate_vars)]
            )
            #   Create a dataset with variable.parameters["n_hotdeck_array"]
            #       For each group in the model to be the "warmed" hot deck values
            #       They are sorted randomly first (to pick a random "warmed" observation)
            #       Then, they're assigned a random sort of -1 (to be first in the final file)
            df_donors_warm.append(
                pl.concat(
                    [
                        df_donors,
                        pl.from_numpy(
                            rng.uniform(size=safe_height(df_donors)),
                            schema={"___random_sort": pl.Float64},
                        ),
                    ],
                    how="horizontal",
                )
                .sort(sort_by)
                .group_by(model, maintain_order=True)
                .head(1)
                .with_columns([pl.lit(-1).alias("___random_sort")] + donor_vars)
            )

            #   Create the non-warmed hot deck donor table
            df_donors_hd.append(
                pl.concat(
                    [
                        df_donors,
                        pl.from_numpy(
                            rng.uniform(size=safe_height(df_donors)),
                            schema={"___random_sort": pl.Float64},
                        ),
                    ],
                    how="horizontal",
                ).with_columns(donor_vars)
            )

        #   Lists for with columns to:
        #       Convert the possible donated values to arrays to draw from
        to_array = []
        #       Select donor expression (to pull the donor value)
        select_donor = []

        #   List of variables to be dropped at the end
        drop_list = []
        drop_list_donor_arrays = []

        #   For each variable to be donated
        for vari in self.index + donate_vars:
            array_vars = []

            #   For each donor in the donor "array"
            for hdi in range(self.variable.parameters["n_hotdeck_array"]):
                this_var = f"donor_{vari}_{hdi}"
                array_vars.append(this_var)
                drop_list.append(this_var)

            #   Convert the donor values into a polars array datatype
            to_array.append(
                pl.concat_list(array_vars)
                .list.to_array(self.variable.parameters["n_hotdeck_array"])
                .alias(f"donor_{vari}")
            )

            #   The final list of array variables the donation comes from
            if vari in donate_vars:
                #   Actual variable to be donated
                drop_list_donor_arrays.append(f"donor_{vari}")
                rename_to = vari
            else:
                #   Index variable (i.e. donor's key) - keep as donor_{vari}
                rename_to = f"donor_{vari}"

            #   The polars expression to draw the nth value (___donor_index)
            #       For each donated var (and the index)
            select_donor.append(
                pl.col(f"donor_{vari}")
                .arr.get(pl.col("___donor_index"))
                .alias(rename_to)
            )

        #   Merge the files
        #       sort by model then random sort
        #       So that warmed are first (so there are enough obs to fill all
        #       the matched recipients)
        #   Then fill_null fills the potential donor values into the
        #       empty recipient rows
        #   Then keep only the recipients
        #   df_matched has a row for each matched recipient
        #       with n_hotdeck_array temp columns (filled) for each
        #       variable to be donated (and also for the donor index)
        df_matched = (
            pl.concat(
                df_donors_warm
                + df_donors_hd
                + [df_matched.with_columns(pl.lit(1).alias("___recipients"))],
                how="diagonal_relaxed",
            )
            .sort(sort_by)
            .with_columns(pl.col(fill_columns).fill_null(strategy="forward"))
            .filter(pl.col("___recipients") == True)
            .drop("___recipients")
        )

        #   Get selected column as ___donor_index
        #       and convert possible donors to array using expression
        #           in to_array
        #       then drop the individual variable in drop_list
        #       then select the donor using the donor_index from to_array
        #       then drop the no-longer-needed variables
        #   The final data set has a row for each recipient
        #       with the selected donated values for each variable
        #       in donate_vars and for the donor index
        df_matched = (
            df_matched.with_row_index(name="___donor_index")
            .with_columns(
                pl.col("___donor_index").mod(
                    self.variable.parameters["n_hotdeck_array"]
                )
            )
            .with_columns(to_array)
            .drop(drop_list)
            .with_columns(select_donor)
            .drop(["___donor_index", "___random_sort"])
            .drop(drop_list_donor_arrays)
        )

        return (nw_donors.from_polars(df_matched), nw_donors.from_polars(df_unmatched))

    def _statmatch_merge(
        self,
        df_donors: IntoFrameT,
        df_recipients: IntoFrameT,
        donate_vars: list[str],
        model: list[str],
    ) -> tuple[IntoFrameT, IntoFrameT]:
        """
        Do the actual stat match for this model

        Parameters
        ----------
        df_donors : IntoFrameT
            Potential donor observations.
        df_recipients : IntoFrameT
            Potential recipient observations.
        donate_vars : list[str]
            List of variables to donate.
        model : list[str]
            List of variables to match on.

        Returns
        -------
        df_matched : IntoFrameT
            Matched recipients that found a donor (impute complete)
        df_unmatched : IntoFrameT
            Unmatched recipients with no donor (impute pending)

        """

        nw_donors = NarwhalsType(df_donors)
        nw_recipients = NarwhalsType(df_recipients)
        df_donors = nw_donors.to_polars().lazy().collect()
        df_recipients = nw_recipients.to_polars().lazy().collect()

        rng = RandomNumberGenerator()

        self.logging.info(f"     Matching on: {model}")

        sort_donors = model.copy()
        sort_donors.extend(self.index)

        #   For each donor, assign them an index in their
        #       stat match cell
        #       i.e. if there are 8 donors with these characteristics
        #       They will have ___donornumber in { 1,2,3,4...,8}
        df_donors = (
            df_donors.sort(sort_donors)
            .with_columns(pl.lit(1).alias("___donornumber"))
            .with_columns(
                pl.cum_sum("___donornumber")
                .over(model)
                .cast(pl.Int64)
                .alias("___donornumber")
            )
        )

        #   By stat match cell, get a data set with the count
        #       of the number of donors
        #       For the chars above, there would be one row with
        #       ___nInGroup = 8
        countcol = model[len(model) - 1]
        df_donorsby = calculate_by(
            df=df_donors, column_stats={countcol: ["count"]}, by=model
        ).rename({f"{countcol}_n": "___nInGroup"})

        #   Merge the counts onto the recipient table
        [df_recipients, df_donorsby] = safe_upcast_list([df_recipients, df_donorsby])

        df_recipients = df_recipients.join(
            df_donorsby, on=model, how="left", nulls_equal=True
        )

        #   Add a random number to recipients and then round (ceiling)
        #       to integer
        #   By matching this to df_donors on ___donornumber
        #       We will be randomly drawing a donor for each recipient
        df_recipients = pl.concat(
            [
                df_recipients,
                pl.from_numpy(
                    rng.uniform(size=safe_height(df_recipients)),
                    schema={"___donornumber": pl.Float64},
                ),
            ],
            how="horizontal",
        ).with_columns(
            (pl.col("___donornumber") * pl.col("___nInGroup"))
            .ceil()
            .cast(pl.Int64)
            .alias("___donornumber")
        )

        #   Split the table into those recipients that found a match
        df_matched = df_recipients.filter(pl.col("___donornumber").is_not_null())

        #   and those that didn't
        #       These potential recipients will go through to the
        #       next model to try to find a donor
        df_unmatched = df_recipients.filter(pl.col("___donornumber").is_null()).drop(
            ["___donornumber", "___nInGroup"]
        )

        #   For those that have a match
        #       Merge recipients to donors to donate the values

        #   Match recipients to donors by the stat match variables (model)
        #       and the random donor index (___donornumber)
        JoinOn = model.copy()
        JoinOn.append("___donornumber")

        #   List of variables to keep on the donor file when matching
        #       Model variables
        donor_keep = JoinOn.copy()
        #       Donation variables
        donor_keep.extend(donate_vars)
        #       Index (to identify donors)
        donor_keep.extend(self.index)
        d_index_rename = {f"{vari}_right": f"donor_{vari}" for vari in self.index}

        #   Remove any duplicates, just in case
        donor_keep = list(set(donor_keep))

        [df_matched, df_donors] = safe_upcast_list([df_matched, df_donors])
        df_matched = (
            df_matched.drop(donate_vars)
            .join(df_donors, on=JoinOn, how="left", nulls_equal=True)
            .rename(d_index_rename)
            .drop(["___donornumber", "___nInGroup"])
        )

        return (nw_donors.from_polars(df_matched), nw_donors.from_polars(df_unmatched))

    ##########################################################
    ##########################################################
    #   HELPERS - Hot Deck/Stat Match - END
    ##########################################################
    ##########################################################

    ##########################################################
    ##########################################################
    #   HELPERS - pmm - START
    ##########################################################
    ##########################################################

    def _pmm_leave_out(self, df_model: IntoFrameT) -> (IntoFrameT, IntoFrameT):
        if self.variable.parameters["share_leave_out"] > 0:
            share_leave_out = self.variable.parameters["share_leave_out"]
            col_leave_out = "___pmm_leave_out___"
            rng = RandomNumberGenerator()

            nw_type = NarwhalsType(df_model)
            df_model = concat_wrapper(
                [
                    df_model,
                    nw_type.from_polars(
                        pl.DataFrame(
                            {
                                col_leave_out: rng.uniform(
                                    low=0, high=1, size=safe_height(df_model)
                                )
                            }
                        )
                    ),
                ],
                how="horizontal",
            )

            df_model = (
                nw.from_native(df_model)
                .with_columns(nw.col(col_leave_out) <= share_leave_out)
                .to_native()
            )

            nw_type = NarwhalsType(df_model)
            df_model_partitioned = nw_type.to_polars().partition_by(col_leave_out)

            if df_model_partitioned[0].select(col_leave_out)[0, 0]:
                df_model = df_model_partitioned[1].drop(col_leave_out)
                df_pmm_leave_out = df_model_partitioned[0].drop(col_leave_out)
            else:
                df_model = df_model_partitioned[0].drop(col_leave_out)
                df_pmm_leave_out = df_model_partitioned[1].drop(col_leave_out)

            return (
                nw_type.from_polars(df_model),
                nw_type.from_polars(df_pmm_leave_out),
            )
        else:
            #   No subsetting, just return the data
            return (df_model, None)

    # def _pmm_adjust_leave_out(self,
    #                           df_impute:IntoFrameT,
    #                           p_pmm_model:IntoFrameT,
    #                           p_pmm_leave_out:IntoFrameT,
    #                           p_col:str):
    #     qlist = [f"q{qi*5}" for qi in range(1,20)]
    #     qs_pmm_model = StatCalculator(df=p_pmm_model,
    #                                   statistics=Statistics(stats=qlist,
    #                                                         columns=p_col),
    #                                   display=False)
    #     qs_pmm_leave_out = StatCalculator(df=p_pmm_leave_out,
    #                                       statistics=Statistics(stats=qlist,
    #                                                             columns=p_col),
    #                                       display=False)

    #     qs_adjustment = qs_pmm_leave_out.compare(qs_pmm_model,
    #                                              display=False)

    #     df_adjustment = qs_adjustment["ratio"].df_estimates
    #     df_adjustment = df_adjustment.rename({coli:coli[1:] for coli in df_adjustment.columns})

    #     col_ptile = "___pmm_ptile___"
    #     col_adjustment = "___pmm_adjustment___"
    #     df_adjustment = (df_adjustment.transpose(include_header=True).filter(pl.col("column") != "ariable")
    #                                   .with_columns([(pl.col("column").cast(pl.Float32)/100).alias(col_ptile),
    #                                                  (1+pl.col("column_0").cast(pl.Float64)).alias(col_adjustment)])
    #                                   .drop(cs.starts_with("column")))

    #     self.logging.info("PMM leave-out adjustment")
    #     with pl.Config(fmt_str_lengths=50) as cfg:
    #         #   Basic formatting
    #         cfg.set_tbl_cell_alignment("RIGHT")
    #         cfg.set_tbl_hide_column_data_types(True)
    #         cfg.set_tbl_hide_dataframe_shape(True)
    #         cfg.set_thousands_separator(True)
    #         cfg.set_tbl_width_chars(600)
    #         cfg.set_tbl_cols(len(df_adjustment.columns))
    #         cfg.set_fmt_float("mixed")
    #         cfg.set_tbl_rows(safe_height(df_adjustment))

    #         self.logging.info(drb_round_table(df_adjustment))

    #     df_impute = (df_impute.sort(p_col).with_columns((pl.lit(1)/safe_height(df_impute)).alias(col_ptile))
    #                                       .with_columns(pl.cum_sum(col_ptile)))

    #     df_impute = AppendList([df_impute, df_adjustment],
    #                            quietly=True)
    #     df_impute = (df_impute.sort(col_ptile)
    #                           .with_columns(pl.col(col_adjustment).interpolate("linear"))
    #                           .with_columns(pl.col(col_adjustment).fill_null(strategy="forward").fill_null(strategy="backward"))
    #                           .with_columns((pl.col(p_col)*pl.col("___pmm_adjustment___")).alias(p_col))
    #                           .filter(pl.col(p_col).is_not_missing())
    #                           .drop(cs.starts_with("___pmm_"))
    #                           .sort(self.index))

    #     return df_impute

    ##########################################################
    ##########################################################
    #   HELPERS - pmm - END
    ##########################################################
    ##########################################################

    # ##########################################################
    # ##########################################################
    # #   HELPERS - two_sample_regression - START
    # ##########################################################
    # ##########################################################

    # def _two_sample_percentiles(self,
    #                             q_prefix:bool=True) -> list[str|int|float]:
    #     params = self.variable.parameters

    #     cuts = []
    #     if "percentile_cuts" in params.keys():
    #         cuts = params["percentile_cuts"]

    #     if len(cuts) == 0:
    #         cuts = [i*100/params['bins'] for i in range(1, params["bins"])]

    #     if max(cuts) < 1:
    #         adjustment = 100
    #     else:
    #         adjustment = 1

    #     if q_prefix:
    #         return [f"q{i*adjustment}" for i in cuts]
    #     else:
    #         return [{i*adjustment} for i in cuts]

    # def _two_sample_get_predictions(self,
    #                                 df_model:IntoFrameT,
    #                                 df_impute:IntoFrameT,
    #                                 model_vars:list[str],
    #                                 fb:FormulaBuilder,
    #                                 regression_type:Parameters.RegressionModel,
    #                                 path_save:str,
    #                                 path_support:str,
    #                                 is_boolean:bool,
    #                                 save_percentile_cuts:bool,
    #                                 draw_error:bool) -> tuple[IntoFrameT,
    #                                                           IntoFrameT,
    #                                                           IntoFrameT]:

    #     min_n_x_var = self.variable.parameters["min_n_x_var"]
    #     bin_by = self.variable.parameters["bin_by"]

    #     [df_model,
    #      df_impute,
    #      df_betas,
    #      _] = self._run_regression(df_model=df_model,
    #                                           df_impute=df_impute,
    #                                           model_vars=model_vars,
    #                                           formula=fb.formula,
    #                                           regmodel=regression_type,
    #                                           min_n_x_var=min_n_x_var)

    #     keep_vars = self.index + ["___prediction"] + bin_by
    #     if self.weight != "":
    #         keep_vars.append(self.weight)

    #     df_prediction = concat_wrapper(
    #         [
    #             (
    #                 nw.from_native(df_model)
    #                 .select(keep_vars + [self.variable.impute_var])
    #                 .to_native()
    #             ),
    #             (
    #                 nw.from_native(df_impute)
    #                 .select(keep_vars)
    #                 .to_native()
    #             )
    #         ],
    #         how="diagonal"
    #     )

    #     df_prediction = concat_wrapper(
    #         [
    #             drb_round_table(
    #                 nw.from_native(df_prediction)
    #                 .select("___prediction")
    #                 .to_native()
    #             ),
    #             (
    #                 nw.from_native(df_prediction)
    #                 .drop("___prediction")
    #                 .to_native()
    #             )
    #         ],
    #         how="horizontal"
    #     )

    #     quantiles = self._two_sample_percentiles()

    #     #   Cut into bins
    #     sc = StatCalculator(df=df_model,
    #                         statistics=Statistics(stats=quantiles,
    #                                               columns=["___prediction"]),
    #                         weight=self.weight,
    #                         round_output=self.variable.parameters["round_impute_var_digits"],
    #                         display=False)

    #     cuts = list(set(sc.df_estimates.row(0)[1:]))
    #     cuts.sort()
    #     if save_percentile_cuts:
    #         df_hat_cuts = self._two_sample_n_at_points(df=df_prediction.select("___prediction"),
    #                                                    points=cuts,
    #                                                    var_to_check="___prediction",
    #                                                    round_digits=self.variable.parameters["round_impute_var_digits"])

    #     del sc
    #     df_prediction = ToCategorical(df=df_prediction,
    #                                   Columns=["___prediction"],
    #                                   Cutoffs=cuts,
    #                                   ZeroGroup=False,
    #                                   MissingGroup=False,
    #                                   AsDummies=False)

    #     if is_boolean:
    #         sc = StatCalculator(df=df_prediction.filter(pl.col(self.variable.impute_var).is_not_missing()),
    #                             statistics=Statistics(stats=["mean",
    #                                                          "n"],
    #                                                   columns=[self.variable.impute_var]),
    #                             by={"q":bin_by + ["___prediction_cat"]},
    #                             weight=self.weight,
    #                             round_output=False,
    #                             display=False)

    #         df_distribution = (sc.df_estimates.drop("Variable")
    #                              .with_columns(pl.col("mean").round(self.variable.parameters["round_impute_var_digits"])))

    #     else:
    #         if self.variable.parameters["round_impute_var_digits"] > 0:
    #             df_prediction = drb_round_table(df=df_prediction,
    #                                             columns=[self.variable.impute_var],
    #                                             round_all=False,
    #                                             digits=self.variable.parameters["round_impute_var_digits"])

    #         y_quantiles = [f"q{i*100}" for i in self.variable.parameters["continuous_qtiles_y_cuts"]]

    #         if draw_error:
    #             summary_col = "___ehat"
    #             df_prediction = df_prediction.with_columns((pl.col("___prediction") - pl.col(self.variable.impute_var)).alias("___ehat"))
    #         else:
    #             summary_col = self.variable.impute_var

    #         interpolate_by_bin = self.variable.parameters["continuous_qtiles_interpolate_by_bin"]

    #         if interpolate_by_bin:
    #             df_distribution_items = []
    #             for bini in range(0,len(cuts)+1):
    #                 df_distribution_items.append(self._two_sample_interpolation_points(df=df_prediction.filter(pl.col("___prediction_cat") == bini),
    #                                                                         summary_col=summary_col,
    #                                                                         y_quantiles=y_quantiles))

    #             df_distribution = AppendList(df_distribution_items)
    #             del df_distribution_items
    #         else:
    #             df_distribution = self._two_sample_interpolation_points(df=df_prediction,
    #                                                                     summary_col=summary_col,
    #                                                                     y_quantiles=y_quantiles)

    #         self.logging.info(df_distribution)

    #     if path_save != "":
    #         #   Load up what we need for regressions
    #         [_,_,df_model_processed] = model_matrix(df=df_model,
    #                                                 formula=f"~{fb.rhs()}",
    #                                                 with_constant=False)

    #         var_list = []
    #         for coli in df_betas["term"].to_list():
    #             if coli != "(Intercept)":
    #                 var_list.append(coli)
    #         #   In case some variables got dropped
    #         var_list = df_model_processed.news.columns(var_list)
    #         sc = StatCalculator(df=df_model_processed,
    #                             statistics=Statistics(stats=["n","n|not0"],
    #                                                   columns=var_list),
    #                             round_output=False,
    #                             display=False)

    #         df_ns_per_beta = sc.df_estimates

    #         #   Save the regression coefficients
    #         (df_betas.select(["term",
    #                           "estimate"])
    #                  .to_pandas()
    #                  .to_excel(excel_writer=path_save,
    #                            sheet_name="b",
    #                            index=False))

    #         #       Save the count != 0 for each regression variable
    #         (df_ns_per_beta.to_pandas()
    #                        .to_excel(excel_writer=path_support,
    #                                  sheet_name="b",
    #                                  index=False))

    #         if save_percentile_cuts:
    #             #   Save the ptile cutoffs
    #             with pd.ExcelWriter(path_save,
    #                                 mode="a") as e_writer:
    #                 (df_hat_cuts.select(["points"])
    #                          .to_pandas()
    #                          .to_excel(excel_writer=e_writer,
    #                                    sheet_name="cutoffs",
    #                                    index=False))

    #             #   Save the count EXACTLY at each cutoff
    #             with pd.ExcelWriter(path_support,
    #                                 mode="a") as e_writer:
    #                 (df_hat_cuts.select(["n"])
    #                          .to_pandas()
    #                          .to_excel(excel_writer=e_writer,
    #                                    sheet_name="cutoffs",
    #                                    index=False))

    #         if is_boolean:
    #             with pd.ExcelWriter(path_save,
    #                                 mode="a") as e_writer:
    #                 (df_distribution.drop("n")
    #                          .to_pandas()
    #                          .to_excel(excel_writer=e_writer,
    #                                    sheet_name="y_empirical",
    #                                    index=False))

    #             #   Save the count in each group
    #             with pd.ExcelWriter(path_support,
    #                                 mode="a") as e_writer:
    #                 (df_distribution.drop(df_distribution.news.columns("mean"))
    #                          .to_pandas()
    #                          .to_excel(excel_writer=e_writer,
    #                                    sheet_name="y_empirical",
    #                                    index=False))
    #         else:
    #             with pd.ExcelWriter(path_save,
    #                                 mode="a") as e_writer:
    #                 (df_distribution.drop("n")
    #                          .to_pandas()
    #                          .to_excel(excel_writer=e_writer,
    #                                    sheet_name="y_empirical",
    #                                    index=False))

    #             #   Save the count EXACTLY at each cutoff
    #             with pd.ExcelWriter(path_support,
    #                                 mode="a") as e_writer:
    #                 (df_distribution.drop(df_distribution.news.columns("q*"))
    #                          .to_pandas()
    #                          .to_excel(excel_writer=e_writer,
    #                                    sheet_name="y_empirical",
    #                                    index=False))

    #     return (df_prediction,
    #             cuts,
    #             df_distribution)

    # def _two_sample_n_at_points(self,
    #                             df:pl.LazyFrame | pl.DataFrame,
    #                             points:list[float|int],
    #                             var_to_check:str,
    #                             round_digits:int=0) -> pl.DataFrame:

    #     with_at_cuts = []

    #     cut_index = 0
    #     for cuti in points:
    #         cut_index += 1
    #         with_at_cuts.append(
    #                 (pl.col(var_to_check) == cuti).sum().alias(f"n_at_{cut_index}")
    #             )

    #     if round_digits > 0:
    #         df = drb_round_table(df=df.select(var_to_check),
    #                              digits=round_digits)
    #     n_at_points = list(df.select(with_at_cuts)
    #                     .row(0))

    #     return pl.DataFrame(
    #                     {"points":points,
    #                      "n":n_at_points}
    #                 )

    # def _two_sample_interpolation_points(self,
    #                                      df:pl.LazyFrame | pl.DataFrame,
    #                                      summary_col:str,
    #                                      y_quantiles:list[str],
    #                                      interval_adjustment:float=1.0) -> pl.LazyFrame | pl.DataFrame:
    #     bin_by = self.variable.parameters["bin_by"]
    #     c_y = pl.col(self.variable.impute_var)

    #     interpolation_interval_std = df.select(c_y.std())[0,0]/2
    #     interpolation_interval_max_min = df.select(c_y.max() - c_y.min())[0,0]/500
    #     interval = interval_adjustment*min(interpolation_interval_std,
    #                                        interpolation_interval_max_min)
    #     sc = StatCalculator(df=df.filter(pl.col(self.variable.impute_var).is_not_missing()),
    #                         statistics=Statistics(stats=y_quantiles + ["n"],
    #                                               columns=[summary_col],
    #                                               quantile_interpolated=True,
    #                                               quantile_interpolated_interval=interval),
    #                         by={"q":bin_by + ["___prediction_cat"]},
    #                         weight=self.weight,
    #                         round_output=False,
    #                         display=False)

    #     #   Any null?  Halve the interval, then
    #     any_missing = sc.df_estimates.select(pl.any_horizontal(pl.col(y_quantiles).news.is_missing().any())).item()
    #     if any_missing:
    #         self.logging.info(f"  _two_sample_interpolation_points: Quantile missing, adjusting quantile bin steps by {interval_adjustment/2}")
    #         return self._two_sample_interpolation_points(df=df,
    #                                                      summary_col=summary_col,
    #                                                      y_quantiles=y_quantiles,
    #                                                      interval_adjustment=interval_adjustment/2)

    #     else:
    #         return drb_round_table(df=sc.df_estimates.drop("Variable"),
    #                                columns=sc.df_estimates.news.columns("q*"),
    #                                columns_exclude=bin_by + ["___prediction_cat"])

    ##########################################################
    ##########################################################
    #   HELPERS - two_sample_regression - END
    ##########################################################
    ##########################################################

    ##########################################################
    ##########################################################
    #   HELPERS - General - START
    ##########################################################
    ##########################################################

    def _draw_interpolated_percentiles(
        self, df: IntoFrameT, percentiles: list
    ) -> tuple[IntoFrameT, IntoFrameT]:
        tail = "gaussian"

        draw = DrawFromQuantileVectors(
            df_quantiles=df, alphas=percentiles, tails=tail, seed=generate_seed()
        )
        df_results = draw.draw_random_values()

        return (
            (nw.from_native(df_results).select("p").to_native()),
            (nw.from_native(df_results).select("values").to_native()),
        )

    def _find_nearest_neighbor_by(
        self,
        df_model: IntoFrameT,
        df_impute: IntoFrameT,
        knearest: int,
        match_on: list[str],
        donate_vars: list[str],
        donate_by: list[str] | None = None,
    ) -> IntoFrameT:
        if donate_by is None or len(donate_by) == 0:
            return self._find_nearest_neighbor(
                df_model=df_model,
                df_impute=df_impute,
                knearest=knearest,
                match_on=match_on,
                donate_vars=donate_vars,
            )
        else:
            nw_model = NarwhalsType(df_model)
            nw_impute = NarwhalsType(df_impute)
            d_model = nw_model.from_polars(
                nw_model.to_polars().partition_by(
                    donate_by, as_dict=True, include_key=True
                )
            )

            d_impute = nw_impute.from_polars(
                nw_impute.to_polars().partition_by(
                    donate_by, as_dict=True, include_key=True
                )
            )

            df_matched = []
            for keyi in d_impute.keys():
                self.logging.info(f"Matching on {donate_vars}:{keyi} - BEGIN")
                df_matched.append(
                    self._find_nearest_neighbor(
                        df_model=d_model[keyi],
                        df_impute=d_impute[keyi],
                        knearest=knearest,
                        match_on=match_on,
                        donate_vars=donate_vars,
                    )
                )
                self.logging.info(f"Matching on {donate_vars}:{keyi} - END")

            return concat_wrapper(df_matched, how="diagonal")

    def _find_nearest_neighbor(
        self,
        df_model: IntoFrameT,
        df_impute: IntoFrameT,
        knearest: int,
        match_on: list[str],
        donate_vars: list[str],
    ) -> IntoFrameT:
        self.logging.info(f"     Finding {knearest} nearest neighbors on {match_on}")

        nw_model = NarwhalsType(df_model)
        nw_impute = NarwhalsType(df_impute)

        df_model = nw_model.to_polars()
        df_impute = nw_impute.to_polars()

        #   Add random jitter to the points to make matching random in ties
        c_match = pl.col(match_on)
        min_model = (
            df_model.select(match_on)
            .filter(c_match.ne(0))
            .select(pl.col(match_on).abs().min())
            .lazy()
            .collect()
            .item()
        )
        min_impute = (
            df_model.select(match_on)
            .filter(c_match.ne(0))
            .select(pl.col(match_on).abs().min())
            .lazy()
            .collect()
            .item()
        )
        jitter_base = 10 ** (first_digit_position(min(min_model, min_impute)) - 8)

        rng = RandomNumberGenerator()
        jitter_range_multiple = 500
        jitter_model = rng.uniform(
            low=-jitter_base * jitter_range_multiple,
            high=jitter_base * jitter_range_multiple,
            size=(safe_height(df_model), 1),
        )
        jitter_impute = rng.uniform(
            low=-jitter_base * jitter_range_multiple,
            high=jitter_base * jitter_range_multiple,
            size=(safe_height(df_impute), 1),
        )

        #   Set up the nearest neighbor object for the donors (df_model)
        kdt = KDTree(
            df_model.select(match_on).lazy().collect().to_numpy() + jitter_model
        )

        #   Find the knearest from df_model to each observation in df_impute
        df_matches = pl.from_numpy(
            kdt.query(
                df_impute.select(match_on).to_numpy() + jitter_impute,
                k=knearest,
                return_distance=False,
            )
        )

        self.logging.info(f"     Randomly picking one and donating {donate_vars}")
        #   Get an integer from 0-knearest-1 to pick the match randomly

        rng = RandomNumberGenerator()
        df_randoms = pl.DataFrame(
            {
                "___matched": np.floor(
                    rng.uniform(low=0, high=knearest, size=safe_height(df_impute))
                )
            },
            schema={"___matched": pl.Int32},
        )

        #   Convert the nearest neighbors to an array of knearest values
        df_matches = drop_if_exists(
            df_matches.with_columns(
                pl.concat_list(pl.all())
                .list.to_array(knearest)
                .alias("___possiblematches")
            ),
            columns="column_*",
        )

        #   Merge the neighbors to the random draw
        df_randoms = pl.concat([df_randoms, df_matches], how="horizontal")
        #   Get the index of the selected neighbor
        df_randoms = df_randoms.with_columns(
            pl.col("___possiblematches")
            .arr.get(pl.col("___matched"))
            .alias("___selectedmatch")
        ).select("___selectedmatch")

        #   Get the donate variable values from df_model
        df_matched = join_list(
            [
                df_randoms,
                (
                    df_model.lazy()
                    .collect()
                    .select(donate_vars + self.index)
                    .with_row_index(name="___selectedmatch")
                ),
            ],
            on=["___selectedmatch"],
            how="left",
        ).drop("___selectedmatch")

        #   Most common matches
        self.logging.info("     Most common matches: ")
        df_matchcount = (
            nw.from_native(
                calculate_by(
                    df=(
                        nw.from_native(df_matched)
                        .with_columns(nw.lit(1).alias("nDonors"))
                        .to_native()
                    ),
                    column_stats={"nDonors": ["count"]},
                    by=self.index,
                    no_suffix=True,
                )
            )
            .sort(["nDonors"], descending=True)
            .to_native()
        )

        self.logging.info(df_matchcount.head(5).lazy().collect())

        self.logging.info("\n\n")
        df_matched = concat_wrapper(
            [df_impute.select(self.index), df_matched.select(donate_vars)],
            how="horizontal",
        )

        return nw_model.from_polars(df_matched)

    def _merge_imputes_to_df(
        self, df_imputed: IntoFrameT, df: IntoFrameT, merge_list: list | str
    ) -> IntoFrameT:
        if type(merge_list) is str:
            merge_list = [merge_list]
        #   Merge results onto main file
        replace_list = []
        for vari in merge_list:
            replace_list.append(
                nw.when(nw.col(f"{vari}_right").is_not_missing())
                .then(nw.col(f"{vari}_right"))
                .otherwise(nw.col(vari))
                .alias(vari)
            )

        imputed_keep = self.index.copy()
        imputed_keep.extend(merge_list)

        df = (
            nw.from_native(
                join_list(
                    [df, (nw.from_native(df_imputed).select(imputed_keep).to_native())],
                    on=self.index,
                    how="left",
                )
            )
            .with_columns(replace_list)
            .drop([f"{vari}_right" for vari in merge_list])
            .to_native()
        )

        df = compress_df(df=df, cols=merge_list)

        return df

    def _post_impute_statistics_items(self=None) -> list[str]:
        return [
            "n",
            "n|notmissing",
            "mean",
            "mean|not0",
            "std|not0",
            "q10|not0",
            "q25|not0",
            "q50|not0",
            "q75|not0",
            "q90|not0",
            "min|not0",
            "max|not0",
        ]

    def _post_impute_statistics(
        self,
        df_model: IntoFrameT,
        df_impute: IntoFrameT,
        donate_vars: list = None,
        append: bool = True,
        show_by: bool = True,
    ):
        #   TODO? Hot dec/stat match stats on samples weighted by
        #       share of recipients in each cell?
        if donate_vars is None:
            donate_vars = [self.variable.impute_var]

        keep_list = donate_vars
        if self.original_variable.weight != "":
            keep_list.append(self.original_variable.weight)

        df_summary = concat_wrapper(
            [
                (
                    nw.from_native(df_model)
                    .select(keep_list)
                    .with_columns(nw.lit(0).alias("Imputed"))
                    .to_native()
                ),
                (
                    nw.from_native(df_impute)
                    .select(keep_list)
                    .with_columns(nw.lit(1).alias("Imputed"))
                    .to_native()
                ),
            ],
            how="diagonal",
        )

        if self.parent.imputation_stats is not None:
            stats_to_calculate = self.parent.imputation_stats
        else:
            stats_to_calculate = Impute._post_impute_statistics_items()
        statistics = Statistics(stats=stats_to_calculate, columns=donate_vars)
        summarize_by = {"All": [], "impute": ["Imputed"]}

        self.logging.info(f"Post-imputation statistics for {donate_vars}")
        self.logging.info(f"    Where:          {self.variable.Where}")
        self.logging.info(f"    Where (impute): {self.variable.Where_impute}")

        stats_post = StatCalculator(
            df=df_summary,
            statistics=statistics,
            by=summarize_by,
            display=False,
            round_output=False,
        )
        stats_post.rounding.cols_exclude = stats_post.rounding.cols_n
        stats_post.print(round_output=True, sub_log=self.logging)

        if append and stats_post.df_estimates is not None:
            cols_stats = (
                nw.from_native(stats_post.df_estimates).lazy().collect_schema().names()
            )
            df_empty = (
                nw.from_native(stats_post.df_estimates)
                .head(0)
                .with_columns([nw.lit(None).alias(coli) for coli in cols_stats])
                .to_native()
            )

            df_post_impute = stats_post.df_estimates
            if len(self.variable.By) and show_by:
                df_post_impute = (
                    nw.from_native(df_post_impute)
                    .with_columns(nw.lit(str(self.current_by)).alias("By"))
                    .to_native()
                )

            df_post_impute = concat_wrapper([df_post_impute, df_empty], how="diagonal")

            if self.df_post_impute_statistics is None:
                self.df_post_impute_statistics = df_post_impute
            else:
                self.df_post_impute_statistics = concat_wrapper(
                    [self.df_post_impute_statistics, df_post_impute], how="diagonal"
                )

    def df_impute(self, df: IntoFrameT, keep_vars: list | None = None) -> IntoFrameT:
        if keep_vars is None:
            keep_vars = safe_columns(df)

        return (
            nw.from_native(self.variable.df_impute_where(df=df))
            .select(keep_vars)
            .lazy()
            .collect()
            .to_native()
        )

    def df_model(
        self, df: IntoFrameT, keep_vars: list | None = None, drop_imputed: bool = False
    ) -> IntoFrameT:
        if keep_vars is None:
            keep_vars = nw.from_native(df).lazy().collect_schema().names()

        df = (
            nw.from_native(self.variable.df_predict_where(df=df))
            .filter(nw.col(self.variable.impute_var).is_not_missing())
            .select(keep_vars)
            .lazy()
            .collect()
            .to_native()
        )

        return df

    ##########################################################
    ##########################################################
    #   HELPERS - General - END
    ##########################################################
    ##########################################################
