from __future__ import annotations
from typing import TYPE_CHECKING

import os
import logging
import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT
import polars as pl
import numpy as np
from copy import deepcopy
from survey_kit_formula import ModelSpec

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
    winsorize_by_percentiles,
)
from ..utilities.compress import compress_df
from ..utilities.formula_builder import FormulaBuilder

from ..statistics.basic_calculations import calculate_by
from ..statistics.statistics import Statistics
from ..statistics.calculator import StatCalculator

from ..utilities.random import RandomNumberGenerator, generate_seed

from ..utilities.rounding import drb_round_table, first_digit_position

from .utilities.draw_from_quantiles import DrawFromQuantileVectors
from .utilities.lightgbm_wrapper import Survey_kit_Lightgbm as kit_lightgbm
from .utilities.leaf_donor_matching import leaf_cooccurrence_match, extract_leaf_indices
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
                to_console=not self.parent.parallel.enabled,
                force=True,
                name="SRMI_Impute_Variable",
                level=logging.INFO,
            )
        else:
            self.logging = logger

        self.df_post_impute_statistics = None
        #   SRMI.convergence()'s chainMean/chainVar equivalent - only
        #       actually set inside _post_impute_statistics, which isn't
        #       always reached (e.g. "No rows to impute" short-circuits
        #       before it) - default None here so Implicate._impute_variable
        #       can always read these attributes safely either way.
        self.chain_mean = None
        self.chain_std = None
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
                elif self.variable.modeltype in (
                    Variable.ModelType.Regression,
                    Variable.ModelType.RandomForest,
                    Variable.ModelType.XGBoost,
                    Variable.ModelType.CatBoost,
                    Variable.ModelType.SklearnModel,
                    #   pmm is, mechanically, just regression with a fixed
                    #       model/error choice baked into
                    #       Parameters.pmm()'s own convenience builder -
                    #       see its docstring. No separate impute.py
                    #       method any more - routes here, same as
                    #       RandomForest()/etc.
                    Variable.ModelType.pmm,
                ):
                    #   RandomForest/XGBoost/CatBoost/SklearnModel are all
                    #       "regression with a different model plugged in"
                    #       - see Parameters.RandomForest() etc. and
                    #       _run_regression's estimator handling.
                    df_by[idf] = self.regression(df=df_by[idf])
                elif self.variable.modeltype == Variable.ModelType.LightGBM:
                    df_by[idf] = self.lightgbm(df=df_by[idf])
                elif self.variable.modeltype == Variable.ModelType.Multinomial:
                    df_by[idf] = self.multinomial(df=df_by[idf])
                elif self.variable.modeltype == Variable.ModelType.OrderedCategorical:
                    df_by[idf] = self.ordered_categorical(df=df_by[idf])
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

            [fb, _, _] = self.variable.process_model(df=df, NoConstant=True)
            selected_model = self.variable.selection.run(
                df=df,
                y=self.variable.impute_var,
                formula=fb.formula,
                weight=self.weight,
            )

            self.variable = deepcopy(self.original_variable)
            self.variable.model = self.variable.union_required_predictors(
                selected_model
            )

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

        #   "model" (the RegressionModel enum) is only meaningful for
        #       plain OLS/Logit - RandomForest()/XGBoost()/CatBoost()/
        #       SklearnModel() don't set it at all (they set "estimator"
        #       instead), so log/derive the model choice from
        #       self.variable.modeltype, which is accurate either way.
        regmodel = self.variable.parameters.get("model")
        self.logging.info(f"     Imputation using {self.variable.modeltype.name}")

        [fb, _, model_vars] = self.variable.process_model(df)
        # fb = FormulaBuilder(df=df)
        # fb.formula = f"{self.variable.impute_var}{self.variable.model}"
        # model_vars = fb.columns
        keep_vars = model_vars + self.index

        #   categorical_feature columns are deliberately left out of a
        #       formula-string model= (see
        #       Variable._validate_estimator_available /
        #       _run_regression's raw-column concat), so process_model's
        #       model_vars won't include them - add them here or they'd
        #       get dropped by the df_model/df_impute select below before
        #       _run_regression ever sees them. No-op for the list-model
        #       branch, where they're already part of model_vars.
        categorical_feature = self.variable.parameters.get("categorical_feature")
        if categorical_feature:
            for vari in categorical_feature:
                if vari not in keep_vars:
                    keep_vars.append(vari)

        if self.weight != "":
            keep_vars.append(self.weight)
            model_vars.append(self.weight)

        #   self.weight (used for sample_weight, above) and
        #       self.original_variable.weight (the Variable's own declared
        #       weight, read only by _post_impute_statistics's descriptive
        #       display below) are DIFFERENT columns whenever bootstrap is
        #       enabled - self.weight becomes the bootstrap replicate
        #       weight then, overriding whatever the Variable declared.
        #       Both need to already be present in df_model/df_impute, or
        #       _post_impute_statistics fails looking for a column that
        #       was never fetched.
        if (
            self.original_variable.weight != ""
            and self.original_variable.weight not in keep_vars
        ):
            keep_vars.append(self.original_variable.weight)

        #   group_levels (Parameters.Regression()/RandomForest()/etc.) -
        #       the nested random-intercept-heuristic columns (e.g. state,
        #       county, hhid) need to be present for _run_regression's
        #       shrinkage step, and the PRIOR iteration's persisted
        #       intercept estimate (if this isn't the first iteration)
        #       needs to ride along too so _run_regression can residualize
        #       against it before fitting - see _nested_group_shrinkage.
        group_levels = self.variable.parameters.get("group_levels", [])
        group_intercept_col = f"___group_intercept_{self.variable.impute_var}___"
        if group_levels:
            for vari in group_levels:
                if vari not in keep_vars:
                    keep_vars.append(vari)
            if (
                group_intercept_col
                in nw.from_native(df).lazy().collect_schema().names()
                and group_intercept_col not in keep_vars
            ):
                keep_vars.append(group_intercept_col)

        #   .get() with a pmm default, not a bare key lookup - Parameters.pmm()'s
        #       own dict (used directly by Variable.ModelType.pmm, which
        #       routes here too, not to its own now-removed pmm() method)
        #       has no "error" key at all, since standalone pmm-modeltype
        #       imputation always was, unconditionally, pmm-style matching.
        errordraw = self.variable.parameters.get("error", Parameters.ErrorDraw.pmm)

        #   Any other variables donated? Relevant for any donation-based
        #   error draw (assigns values via matching, not a direct draw) -
        #   PMM (knearest on scalar yhat) and leaf (tree leaf
        #   co-occurrence, RandomForest()/XGBoost()/CatBoost()/
        #   SklearnModel() only) both donate; Random computes the imputed
        #   value directly and never donates.
        if errordraw in (Parameters.ErrorDraw.pmm, Parameters.ErrorDraw.leaf):
            if len(self.variable.parameters["donate_list"]) > 0:
                keep_vars.extend(self.variable.parameters["donate_list"])

            #   donate_by groups are partitioned out of both df_model and
            #   df_impute in _find_nearest_neighbor_by/
            #   _leaf_match_donor_positions, so both need it kept. Only
            #   add names not already present - keep_vars isn't
            #   deduplicated here and .select() errors on a repeated name.
            for vari in self.variable.parameters["donate_by"]:
                if vari not in keep_vars:
                    keep_vars.append(vari)

        df_model = self.df_model(df=df, keep_vars=keep_vars)

        df_impute = self.df_impute(df=df, keep_vars=keep_vars)

        if safe_height(df_impute) == 0:
            self.logging.info("No rows to impute")
            return df

        b_winsorized = False
        if errordraw in (Parameters.ErrorDraw.pmm, Parameters.ErrorDraw.leaf):
            (df_model, b_winsorized) = self._pmm_winsorize_for_fit(df_model)

        [df_model, df_impute, _] = self._run_regression(
            df_model=df_model,
            df_impute=df_impute,
            model_vars=model_vars,
            formula=fb.formula,
            regmodel=regmodel,
        )

        if group_levels:
            #   Persist this iteration's freshly re-estimated group
            #       intercept back into the working df - "just another
            #       variable to keep and append", the same way
            #       donate_list values ride along, so next iteration's
            #       call to this same variable reads it back in via
            #       keep_vars above instead of starting over from 0.
            #       Every row _run_regression touched (donors AND
            #       recipients) gets a fresh value; anything outside this
            #       variable's Where-restricted sample this iteration
            #       (never in df_model/df_impute) simply keeps whatever
            #       it already had.
            df_group_intercepts = concat_wrapper(
                [
                    nw.from_native(df_model)
                    .select(self.index + [group_intercept_col])
                    .to_native(),
                    nw.from_native(df_impute)
                    .select(self.index + [group_intercept_col])
                    .to_native(),
                ],
                how="vertical",
            )
            #   df isn't necessarily polars-native here (any narwhals-
            #       supported backend), so drop the prior iteration's copy
            #       (if any) via narwhals directly rather than
            #       drop_if_exists, which expects a native .lazy() to
            #       already exist on whatever's passed in.
            nw_df = nw.from_native(df)
            if group_intercept_col in nw_df.lazy().collect_schema().names():
                df = nw_df.drop(group_intercept_col).to_native()
            df = join_list([df, df_group_intercepts], on=self.index, how="left")

        #   ___prediction is computed now - safe to swap back to the true
        #   (un-winsorized) value before _regression_draw_errors's pmm/leaf
        #   branches use df_model as the donor pool.
        df_model = self._pmm_restore_true_value(df_model, b_winsorized)

        df_impute = self._regression_draw_errors(
            df_model=df_model,
            df_impute=df_impute,
            regmodel=regmodel,
            errordraw=errordraw,
        )

        donate_vars = None
        if errordraw in (Parameters.ErrorDraw.pmm, Parameters.ErrorDraw.leaf):
            donate_vars = [self.variable.impute_var]
            if "donate_list" in self.variable.parameters:
                if len(self.variable.parameters["donate_list"]) > 0:
                    donate_vars.extend(self.variable.parameters["donate_list"])

        self._post_impute_statistics(
            df_model=df_model, df_impute=df_impute, donate_vars=donate_vars
        )
        df = self._merge_imputes_to_df(
            df_imputed=df_impute,
            df=df,
            merge_list=(
                donate_vars if donate_vars is not None else self.variable.impute_var
            ),
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

    def multinomial(self, df: IntoFrameT | None = None) -> IntoFrameT:
        """
        Perform multinomial (unordered categorical, 3+ levels) imputation.

        Fits a RandomForestClassifier, then imputes by donor matching on
        leaf co-occurrence: for each recipient, pool the donors sharing a
        leaf with it across every tree, draw one uniformly at random -
        the same donor-selection mechanism mice's rf method uses (pool
        with mice.impute.rf's `unlist(...)` then `sample(..., 1)`; here
        via a streaming weighted-reservoir sample instead of materializing
        the pool - see utilities/leaf_donor_matching.py). Genuinely
        different machinery from regression()/RandomForest() etc. - this
        is classification with donor selection driven by tree structure,
        not a scalar yhat with PMM/knearest matching.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Input dataframe, uses self.df if None

        Returns
        -------
        IntoFrameT
            Dataframe with multinomial-imputed values
        """
        if df is None:
            df = self.df

        self.logging.info("     Imputation using Multinomial")

        #   Predictors can be either form of model= - a plain column list
        #       (used raw/untouched, so any categorical predictor needs to
        #       already be numeric-coded/one-hot encoded, same restriction
        #       as RandomForest()) or an R-style formula string (its
        #       C(...)/factor-dtype terms already get one-hot-encoded into
        #       numeric dummy columns by _build_model_matrix's ModelSpec
        #       branch, same as every other model that reuses it). Nothing
        #       about RandomForestClassifier needs the list form
        #       specifically - it just needs a numeric matrix, exactly
        #       like RandomForestRegressor already gets via either form.
        [fb, _, model_vars] = self.variable.process_model(df)
        keep_vars = list(model_vars) + self.index

        donate_vars = [self.variable.impute_var]
        donate_list = self.variable.parameters.get("donate_list", [])
        if len(donate_list) > 0:
            donate_vars.extend(donate_list)

        donate_by = self.variable.parameters.get("donate_by", [])

        for vari in donate_vars + donate_by:
            if vari not in keep_vars:
                keep_vars.append(vari)

        if self.weight != "" and self.weight not in keep_vars:
            keep_vars.append(self.weight)

        #   self.weight (used for sample_weight, above) and
        #       self.original_variable.weight (the Variable's own
        #       declared weight, read only by _post_impute_statistics's
        #       descriptive display below) are DIFFERENT columns whenever
        #       bootstrap is enabled - see regression()'s identical
        #       distinction. Both need to already be present, or
        #       _post_impute_statistics fails looking for a column that
        #       was never fetched.
        if (
            self.original_variable.weight != ""
            and self.original_variable.weight not in keep_vars
        ):
            keep_vars.append(self.original_variable.weight)

        df_model = self.df_model(df=df, keep_vars=keep_vars)
        df_impute = self.df_impute(df=df, keep_vars=keep_vars)

        if safe_height(df_impute) == 0:
            self.logging.info("No rows to impute")
            return df

        nw_model_type = NarwhalsType(df_model)
        nw_impute_type = NarwhalsType(df_impute)
        df_model_pl = nw_model_type.to_polars().lazy().collect()
        df_impute_pl = nw_impute_type.to_polars().lazy().collect()

        random_share = self.variable.parameters.get("random_share", 1.0)
        if random_share < 1:
            self.logging.info(f"     Using a {random_share} subsample")
            df_model_pl = df_model_pl.sample(fraction=random_share, seed=generate_seed())

        df_model_mm, df_impute_mm, vars_rhs = self._build_model_matrix(
            df_model=df_model_pl,
            df_impute=df_impute_pl,
            formula=fb.formula,
        )

        #   Row position 0..n-1, assigned AFTER any random_share subsample
        #       above (not before) so it stays aligned with X_model/
        #       donor_leaves below, which are built from this same,
        #       possibly-subsampled frame. df_model_mm/df_impute_mm's rows
        #       stay in the same order as df_model_pl/df_impute_pl's -
        #       _build_model_matrix only selects/transforms columns.
        df_model_pl = df_model_pl.with_row_index(name="___row_pos___")
        df_impute_pl = df_impute_pl.with_row_index(name="___row_pos___")

        X_model = df_model_mm.to_numpy()
        y_model = df_model_pl[self.variable.impute_var].to_numpy()
        X_impute = df_impute_mm.to_numpy()

        parameters = self.variable.parameters.get("parameters") or {}

        #   Imported here, not at module level - sklearn's ensemble import
        #       costs real time and only Multinomial/RandomForest/etc.
        #       need it.
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(**parameters)

        fit_kwargs = {}
        if self.weight != "":
            fit_kwargs["sample_weight"] = df_model_pl[self.weight].to_numpy()

        model.fit(X_model, y_model, **fit_kwargs)

        self.logging.info(
            f"     Extracting leaf indices ({model.get_params()['n_estimators']} trees)"
        )
        donor_leaves = model.apply(X_model).astype(np.int32)
        recipient_leaves = model.apply(X_impute).astype(np.int32)

        matched_donor_idx = self._leaf_match_donor_positions(
            donor_leaves=donor_leaves,
            recipient_leaves=recipient_leaves,
            df_model_pl=df_model_pl,
            df_impute_pl=df_impute_pl,
            donate_by=donate_by,
        )

        df_donated = self._leaf_gather_donations(
            df_model_pl=df_model_pl,
            matched_donor_idx=matched_donor_idx,
            donate_vars=donate_vars,
        )

        #   _post_impute_statistics (below) mutates its own donate_vars
        #       argument in place to append self.original_variable.weight
        #       (the Variable's own declared weight, untouched by
        #       bootstrap - see regression()'s identical distinction from
        #       self.weight, the bootstrap-or-declared column used for
        #       sample_weight during fitting, which diverges from
        #       original_variable.weight whenever bootstrap is enabled)
        #       when a weight variable is set, then .select()s that same
        #       (now-mutated) list from BOTH df_model and df_impute - so
        #       both need the weight column already present, and
        #       merge_list further down (which reuses this same
        #       donate_vars list, now also carrying weight) needs
        #       df_impute_matched to already carry a legitimate value for
        #       it too. Carry the RECIPIENT's own weight through (not the
        #       donor's) - _merge_imputes_to_df would otherwise overwrite
        #       each recipient's weight with whatever ends up in
        #       df_impute_matched for it.
        impute_extra_cols = (
            [self.original_variable.weight]
            if self.original_variable.weight != ""
            else []
        )
        df_impute_matched = pl.concat(
            [df_impute_pl.select(self.index + impute_extra_cols), df_donated],
            how="horizontal",
        )

        stats_model_cols = list(donate_vars) + self.index
        if (
            self.original_variable.weight != ""
            and self.original_variable.weight not in stats_model_cols
        ):
            stats_model_cols.append(self.original_variable.weight)

        self._post_impute_statistics(
            df_model=nw_model_type.from_polars(df_model_pl.select(stats_model_cols)),
            df_impute=nw_impute_type.from_polars(df_impute_matched),
            donate_vars=donate_vars,
        )

        df = self._merge_imputes_to_df(
            df_imputed=nw_impute_type.from_polars(df_impute_matched),
            df=df,
            merge_list=donate_vars,
        )

        return df

    def ordered_categorical(self, df: IntoFrameT | None = None) -> IntoFrameT:
        """
        Perform ordered-categorical imputation.

        Unlike multinomial() (unordered, classification), this fits a
        mean-regression estimator (default RandomForestRegressor, or any
        factory supplied via Parameters.OrderedCategorical()) against an
        integer rank encoding of the declared category order, then
        donates the REAL observed category from a matched donor - never
        the numeric rank, and never a category that wasn't actually
        observed (same guarantee multinomial() gives). Donor matching is
        either pmm (knearest on the predicted rank, via
        _find_nearest_neighbor_by - the same mechanism plain
        Regression()/RandomForest() use) or leaf (tree leaf
        co-occurrence, via the same _leaf_match_donor_positions/
        _leaf_gather_donations helpers multinomial() and
        _regression_draw_errors's leaf branch use).

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Input dataframe, uses self.df if None

        Returns
        -------
        IntoFrameT
            Dataframe with ordered-categorical-imputed values
        """
        if df is None:
            df = self.df

        self.logging.info("     Imputation using OrderedCategorical")

        categories = self.variable.parameters["categories"]
        category_rank = {cati: float(i) for i, cati in enumerate(categories)}

        [fb, _, model_vars] = self.variable.process_model(df)
        keep_vars = list(model_vars) + self.index

        donate_vars = [self.variable.impute_var]
        donate_list = self.variable.parameters.get("donate_list", [])
        if len(donate_list) > 0:
            donate_vars.extend(donate_list)

        donate_by = self.variable.parameters.get("donate_by", [])

        for vari in donate_vars + donate_by:
            if vari not in keep_vars:
                keep_vars.append(vari)

        categorical_feature = self.variable.parameters.get("categorical_feature")
        if categorical_feature:
            for vari in categorical_feature:
                if vari not in keep_vars:
                    keep_vars.append(vari)

        if self.weight != "" and self.weight not in keep_vars:
            keep_vars.append(self.weight)

        #   self.weight (used for sample_weight, below) and
        #       self.original_variable.weight (the Variable's own
        #       declared weight, read only by _post_impute_statistics's
        #       descriptive display) are DIFFERENT columns whenever
        #       bootstrap is enabled - see regression()'s identical
        #       distinction.
        if (
            self.original_variable.weight != ""
            and self.original_variable.weight not in keep_vars
        ):
            keep_vars.append(self.original_variable.weight)

        df_model = self.df_model(df=df, keep_vars=keep_vars)
        df_impute = self.df_impute(df=df, keep_vars=keep_vars)

        if safe_height(df_impute) == 0:
            self.logging.info("No rows to impute")
            return df

        nw_model_type = NarwhalsType(df_model)
        nw_impute_type = NarwhalsType(df_impute)
        df_model_pl = nw_model_type.to_polars().lazy().collect()
        df_impute_pl = nw_impute_type.to_polars().lazy().collect()

        observed = set(
            df_model_pl[self.variable.impute_var].drop_nulls().unique().to_list()
        )
        unknown = observed - set(categories)
        if unknown:
            message = (
                f"{self.variable.impute_var}: OrderedCategorical's "
                f"categories={categories} doesn't cover observed "
                f"value(s) {sorted(unknown, key=str)} - every observed "
                f"category must appear in categories (lowest/coarsest to "
                f"highest/finest)."
            )
            self.logging.error(message)
            raise ValueError(message)

        random_share = self.variable.parameters.get("random_share", 1.0)
        if random_share < 1:
            self.logging.info(f"     Using a {random_share} subsample")
            df_model_pl = df_model_pl.sample(fraction=random_share, seed=generate_seed())

        #   The numeric rank proxy is what the regression estimator
        #       actually fits against/predicts - the real category
        #       column (still present, untouched) is what donation
        #       copies below, so the imputed value is always a genuinely
        #       observed category, never an interpolated rank.
        rank_col = "___ordinal_rank___"
        df_model_pl = df_model_pl.with_columns(
            pl.col(self.variable.impute_var)
            .replace_strict(category_rank, return_dtype=pl.Float64)
            .alias(rank_col)
        )

        df_model_mm, df_impute_mm, vars_rhs = self._build_model_matrix(
            df_model=df_model_pl,
            df_impute=df_impute_pl,
            formula=fb.formula,
        )

        prepare_data = self.variable.parameters.get("estimator_prepare_data")
        if prepare_data is not None:
            df_model_mm, df_impute_mm = prepare_data(df_model_mm, df_impute_mm)

        model_factory = self.variable.parameters["estimator"]
        model = model_factory()

        fit_kwargs = {}
        if self.weight != "":
            fit_kwargs["sample_weight"] = df_model_pl[self.weight]

        model.fit(X=df_model_mm, y=df_model_pl.select(rank_col), **fit_kwargs)

        predict_rank_model = model.predict(df_model_mm)
        predict_rank_impute = model.predict(df_impute_mm)

        errordraw = self.variable.parameters.get("error", Parameters.ErrorDraw.pmm)

        if errordraw == Parameters.ErrorDraw.leaf:
            self.logging.info(
                "     Extracting leaf indices for leaf-based donor matching"
            )
            donor_leaves = extract_leaf_indices(model, df_model_mm)
            recipient_leaves = extract_leaf_indices(model, df_impute_mm)

            matched_donor_idx = self._leaf_match_donor_positions(
                donor_leaves=donor_leaves,
                recipient_leaves=recipient_leaves,
                df_model_pl=df_model_pl,
                df_impute_pl=df_impute_pl,
                donate_by=donate_by,
            )
            df_donated = self._leaf_gather_donations(
                df_model_pl=df_model_pl,
                matched_donor_idx=matched_donor_idx,
                donate_vars=donate_vars,
            )
            impute_extra_cols = (
                [self.original_variable.weight]
                if self.original_variable.weight != ""
                else []
            )
            df_impute_matched = pl.concat(
                [df_impute_pl.select(self.index + impute_extra_cols), df_donated],
                how="horizontal",
            )
        else:
            #   pmm - knearest on the predicted rank, the same mechanism
            #       plain Regression()/RandomForest() use, just matched
            #       on the rank proxy instead of impute_var directly.
            knearest = self.variable.parameters.get("knearest", 10)
            df_model_pl = df_model_pl.with_columns(
                pl.Series("___prediction", predict_rank_model)
            )
            df_impute_pl = df_impute_pl.with_columns(
                pl.Series("___prediction", predict_rank_impute)
            )
            df_impute_matched = self._find_nearest_neighbor_by(
                df_model=nw_model_type.from_polars(df_model_pl),
                df_impute=nw_impute_type.from_polars(df_impute_pl),
                knearest=knearest,
                match_on=["___prediction"],
                donate_vars=donate_vars,
                donate_by=donate_by,
            )
            df_impute_matched = (
                NarwhalsType(df_impute_matched).to_polars().lazy().collect()
            )

        stats_model_cols = list(donate_vars) + self.index
        if (
            self.original_variable.weight != ""
            and self.original_variable.weight not in stats_model_cols
        ):
            stats_model_cols.append(self.original_variable.weight)

        #   _post_impute_statistics's default stats (mean/std/quantiles)
        #       assume a numeric column - impute_var here is a category
        #       label (often a string), which breaks that machinery (a
        #       pre-existing gap in the generic stats path, not specific
        #       to OrderedCategorical - it's just the first modeltype
        #       that reliably hits it, since Multinomial()'s own tests
        #       happen to use an Int64-coded category). Substitute the
        #       already-known integer rank for STATS DISPLAY ONLY, under
        #       the same column name - the actual donated/merged value
        #       below is untouched, still the real observed category.
        rank_expr = pl.col(self.variable.impute_var).replace_strict(
            category_rank, return_dtype=pl.Float64
        )
        df_model_stats = df_model_pl.select(stats_model_cols).with_columns(
            rank_expr.alias(self.variable.impute_var)
        )
        df_impute_stats = df_impute_matched.with_columns(
            rank_expr.alias(self.variable.impute_var)
        )

        self._post_impute_statistics(
            df_model=nw_model_type.from_polars(df_model_stats),
            df_impute=nw_impute_type.from_polars(df_impute_stats),
            donate_vars=donate_vars,
        )

        df = self._merge_imputes_to_df(
            df_imputed=nw_impute_type.from_polars(df_impute_matched),
            df=df,
            merge_list=donate_vars,
        )

        return df

    def _leaf_match_donor_positions(
        self,
        donor_leaves: np.ndarray,
        recipient_leaves: np.ndarray,
        df_model_pl: pl.DataFrame,
        df_impute_pl: pl.DataFrame,
        donate_by: list[str],
    ) -> np.ndarray:
        """
        Shared leaf-co-occurrence donor-position matching, used by both
        multinomial() (RandomForestClassifier) and
        _regression_draw_errors's leaf branch (RandomForest()/XGBoost()/
        CatBoost()/SklearnModel() mean-regression) - the model choice
        differs, but "pool donors sharing a leaf with each recipient
        across every tree, optionally restricted within donate_by groups"
        is identical either way. donor_leaves/recipient_leaves must
        already be row-aligned with df_model_pl/df_impute_pl (same order,
        same row count).

        Returns an int64 array of matched donor row-positions into
        donor_leaves/df_model_pl, one per recipient row, or -1 for a
        recipient that shared no leaf with any donor (in its donate_by
        group, if set).
        """
        n_impute = recipient_leaves.shape[0]
        matched_donor_idx = np.full(n_impute, -1, dtype=np.int64)
        rng = RandomNumberGenerator()

        if donate_by:
            self.logging.info(f"     Matching donors within groups: {donate_by}")
            model_pos_df = df_model_pl.select(donate_by).with_row_index(
                name="___row_pos___"
            )
            impute_pos_df = df_impute_pl.select(donate_by).with_row_index(
                name="___row_pos___"
            )
            model_groups = model_pos_df.partition_by(donate_by, as_dict=True)
            impute_groups = impute_pos_df.partition_by(donate_by, as_dict=True)
            for keyi, impute_part in impute_groups.items():
                impute_pos = impute_part["___row_pos___"].to_numpy()
                if keyi not in model_groups:
                    #   No donors at all in this recipient's group - left
                    #       as -1 (unmatched), same as leaf_cooccurrence_match's
                    #       own convention for "no shared leaf anywhere".
                    continue
                model_part = model_groups[keyi]
                model_pos = model_part["___row_pos___"].to_numpy()
                group_result = leaf_cooccurrence_match(
                    donor_leaves[model_pos], recipient_leaves[impute_pos], rng
                )
                valid = group_result != -1
                matched_donor_idx[impute_pos[valid]] = model_pos[group_result[valid]]
        else:
            matched_donor_idx = leaf_cooccurrence_match(
                donor_leaves, recipient_leaves, rng
            )

        n_unmatched = int((matched_donor_idx == -1).sum())
        if n_unmatched > 0:
            self.logging.info(
                f"     {n_unmatched} recipient(s) shared no leaf with any "
                f"donor (in their donate_by group, if set) - left null."
            )

        return matched_donor_idx

    def _leaf_gather_donations(
        self,
        df_model_pl: pl.DataFrame,
        matched_donor_idx: np.ndarray,
        donate_vars: list[str],
    ) -> pl.DataFrame:
        """
        Gather donate_vars from each recipient's matched donor row (see
        _leaf_match_donor_positions) - shared by multinomial() and
        _regression_draw_errors's leaf branch. Unmatched recipients
        (matched_donor_idx == -1) come back null in every donate_vars
        column rather than an arbitrary donor's value.
        """
        n_unmatched = int((matched_donor_idx == -1).sum())

        #   Placeholder 0 for unmatched rows (gather needs a valid
        #       position) - nulled out below via the mask.
        gather_pos = np.where(matched_donor_idx == -1, 0, matched_donor_idx)
        df_donated = df_model_pl.select(donate_vars).__getitem__(gather_pos)
        if n_unmatched > 0:
            unmatched_mask = pl.Series("___unmatched___", matched_donor_idx == -1)
            df_donated = df_donated.with_columns(
                [
                    pl.when(unmatched_mask).then(None).otherwise(pl.col(c)).alias(c)
                    for c in donate_vars
                ]
            )

        return df_donated

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

        #   df_impute, the formula, and the other inputs to formula processing
        #       are the same for every quantile below (only alpha/the trained
        #       model change) - process it once here rather than re-parsing
        #       the formula/rebuilding the model matrix on every iteration.
        df_impute_processed = lgbm_model.process_predict_frame(df_impute)

        #   Run LightGBM for each quantile
        quantiles = self.variable.parameters["quantiles"]
        for qi in quantiles:
            self.logging.info(f"Running LightGBM for q={qi}")
            lgbm_model.parameters["alpha"] = qi

            lgbm_model.parameters["seed"] = generate_seed()
            lgbm_model.train(show_eval=False)

            #   Get the predictions
            p_impute = lgbm_model.predict(
                df_predict_processed=df_impute_processed, name=f"___p{qi}"
            )

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

            cv_folds = self.variable.parameters.get("cv_folds", 0)
            if cv_folds and cv_folds > 1:
                #   Donor pool prediction via cv_folds-way cross-validation
                #       instead of the in-sample fit - see
                #       Parameters._tabular_ml_params's cv_folds
                #       docstring. p_impute (above) still comes
                #       from lgbm_model_pmm fit on all of df_model -
                #       recipients are already genuinely out-of-sample, so
                #       they don't need CV treatment.
                df_model_cv = NarwhalsType(df_model).to_polars().lazy().collect()

                def _fit_predict_fold(is_holdout):
                    train_mask = ~is_holdout
                    fold_lgbm = kit_lightgbm(
                        df=df_model_cv.filter(train_mask),
                        y=self.variable.impute_var,
                        formula=self.variable.model,
                        weight=self.weight,
                        parameters=lgbm_model_pmm.parameters,
                    )
                    fold_lgbm.parameters["seed"] = generate_seed()
                    fold_lgbm.train(show_eval=False)
                    pred = fold_lgbm.predict(
                        df_predict=df_model_cv.filter(is_holdout), name="___yhat"
                    )
                    return (
                        nw.from_native(pred).lazy().collect()["___yhat"].to_numpy()
                    )

                p_model = pl.DataFrame(
                    {
                        "___yhat": self._pmm_cv_out_of_fold_predictions(
                            n_rows=safe_height(df_model_cv),
                            cv_folds=cv_folds,
                            fit_predict_fold=_fit_predict_fold,
                        )
                    }
                )
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
                .filter(nw.col("___y_draw").is_null())
                .to_native()
            )

            df_impute = (
                nw.from_native(df_impute)
                .filter(~nw.col("___y_draw").is_null())
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

            #   Rank-align recipients (sorted by their quantile-draw rank,
            #       ___y_draw) against the PMM-matched donors (sorted by the
            #       donated value) by pairing them positionally - this is
            #       what actually assigns which real donor value each
            #       recipient gets. donate_by groups must stay separate
            #       through this step: doing it globally would pair a
            #       recipient's rank against a donor value that was matched
            #       within a DIFFERENT donate_by group, silently mixing
            #       donor pools across strata even though
            #       _find_nearest_neighbor_by matched correctly within-group
            #       moments earlier.
            donate_by = self.variable.parameters["donate_by"]

            def _rank_align_donors(df_impute_part, df_marginal_part):
                marginal_drop = self.index + (list(donate_by) if donate_by else [])
                return (
                    nw.from_native(
                        concat_wrapper(
                            [
                                (
                                    nw.from_native(df_impute_part)
                                    .drop(donate_vars)
                                    .sort("___y_draw")
                                    .to_native()
                                ),
                                (
                                    nw.from_native(df_marginal_part)
                                    .drop(marginal_drop)
                                    .sort(self.variable.impute_var)
                                    .to_native()
                                ),
                            ],
                            how="horizontal",
                        )
                    )
                    .drop(["___yhat", "___y_draw"])
                    .to_native()
                )

            if donate_by:
                df_impute_collected = nw.from_native(df_impute).lazy().collect().to_native()
                df_marginal_collected = (
                    nw.from_native(df_marginal).lazy().collect().to_native()
                )
                d_impute_by = NarwhalsType(df_impute_collected).to_polars().partition_by(
                    donate_by, as_dict=True, include_key=True
                )
                d_marginal_by = NarwhalsType(
                    df_marginal_collected
                ).to_polars().partition_by(donate_by, as_dict=True, include_key=True)

                df_impute = concat_wrapper(
                    [
                        _rank_align_donors(d_impute_by[keyi], d_marginal_by[keyi])
                        for keyi in d_impute_by.keys()
                    ],
                    how="diagonal",
                )
            else:
                df_impute = _rank_align_donors(df_impute, df_marginal)

            if safe_height(df_impute_missing) > 0:
                df_impute = concat_wrapper(
                    [df_impute, df_impute_missing], how="diagonal"
                )

            del lgbm_model_pmm
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
            .filter(nw.col(self.variable.impute_var).is_null())
            .to_native()
        )

        if n_missing > 0:
            df_impute_mean = (
                nw.from_native(df_impute)
                .filter(nw.col(self.variable.impute_var).is_null())
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
                        .filter(nw.col("___yhat").is_null())
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
                        .filter(nw.col("___yhat").is_null())
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
                        .filter(~nw.col(self.variable.impute_var).is_null())
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

            #   Probit isn't an implemented RegressionModel option (see
            #   parameters.py) - only Logit needs the Bernoulli-style draw below.
            if regmodel == Parameters.RegressionModel.Logit:
                #   Draw the values for df_impute from the uniform where 1 if <= prediction

                nw_type = NarwhalsType(df_impute)
                df_impute = (
                    nw.from_native(
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
                        )
                    )
                    .with_columns(
                        (nw.col("___phat") <= nw.col("___prediction"))
                        .cast(nw.Boolean)
                        .alias(self.variable.impute_var)
                    )
                    .to_native()
                )
            elif regmodel in (Parameters.RegressionModel.OLS, None):
                #   None means a tree/ensemble estimator (RandomForest()/
                #       XGBoost()/CatBoost()/SklearnModel()) - regmodel is
                #       only ever set at all for the plain OLS/Logit
                #       RegressionModel enum (see _run_regression's own
                #       regmodel=None fallback comment). None of what
                #       follows reads anything OLS-specific though - just
                #       ___prediction and impute_var, both already generic
                #       to any model that reaches here - so it works
                #       identically well for a tree/ensemble estimator's
                #       continuous ___prediction.
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
                        (nw.col("___prediction") + nw.col("___ehat")).alias(
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

        elif errordraw == Parameters.ErrorDraw.leaf:
            #   Donor matching by tree leaf co-occurrence instead of
            #       PMM's knearest-on-scalar-yhat - RandomForest()/
            #       XGBoost()/CatBoost()/SklearnModel() only (see
            #       ErrorDraw.leaf's docstring). ___leaf_ids___ was
            #       persisted by _run_regression right after fitting -
            #       see its comment there for why it has to happen
            #       there, not here.
            donate_vars = [self.variable.impute_var]
            if "donate_list" in self.variable.parameters:
                if len(self.variable.parameters["donate_list"]) > 0:
                    donate_vars.extend(self.variable.parameters["donate_list"])

            leaf_col = "___leaf_ids___"
            nw_model_type = NarwhalsType(df_model)
            nw_impute_type = NarwhalsType(df_impute)
            df_model_pl = nw_model_type.to_polars().lazy().collect()
            df_impute_pl = nw_impute_type.to_polars().lazy().collect()

            if leaf_col not in df_model_pl.columns or leaf_col not in df_impute_pl.columns:
                message = (
                    f"error=ErrorDraw.leaf needs _run_regression to have "
                    f"persisted '{leaf_col}', but it's missing here - the "
                    f"fitted estimator likely doesn't expose leaf indices "
                    f"(see utilities/leaf_donor_matching.extract_leaf_indices)."
                )
                self.logging.error(message)
                raise RuntimeError(message)

            donor_leaves = df_model_pl[leaf_col].to_numpy()
            recipient_leaves = df_impute_pl[leaf_col].to_numpy()

            matched_donor_idx = self._leaf_match_donor_positions(
                donor_leaves=donor_leaves,
                recipient_leaves=recipient_leaves,
                df_model_pl=df_model_pl,
                df_impute_pl=df_impute_pl,
                donate_by=self.variable.parameters["donate_by"],
            )

            df_donated = self._leaf_gather_donations(
                df_model_pl=df_model_pl,
                matched_donor_idx=matched_donor_idx,
                donate_vars=donate_vars,
            )

            df_impute_pl = df_impute_pl.with_columns(
                [df_donated[c].alias(c) for c in donate_vars]
            )
            df_impute = nw_impute_type.from_polars(df_impute_pl)

        return df_impute

    def _build_model_matrix(
        self,
        df_model: IntoFrameT,
        df_impute: IntoFrameT,
        formula: str,
        min_n_x_var: int = 0,
    ) -> tuple[IntoFrameT, IntoFrameT, list[str]]:
        """
        Build the numeric predictor model matrix (df_model_mm/df_impute_mm)
        from either form of model= - a plain column list (raw columns,
        untouched) or an R-style formula string (via ModelSpec, which
        one-hot-encodes any factor/categorical predictor term into
        numeric dummy columns regardless of which model ultimately
        consumes the result). Also applies min_n_x_var (drop sparse
        predictors) and, when set, categorical_feature (add those columns
        in raw, for whichever model choice's own estimator_prepare_data
        hook - see Parameters._categorical_enum_prepare_data - to cast to
        a native categorical dtype afterward).

        Shared by _run_regression() (OLS/Logit/RandomForest/XGBoost/
        CatBoost/SklearnModel) and multinomial()
        (RandomForestClassifier) - nothing about which model consumes
        df_model_mm/df_impute_mm affects how this is built; every one of
        them just needs a numeric matrix, whether from formula/C(...)
        encoding or the list form's untouched raw columns.

        df_model/df_impute must already be eager polars DataFrames (both
        callers collect via NarwhalsType before this).
        """
        if type(self.variable.model) is list:
            fb = FormulaBuilder(df=df_model, formula=formula)
            vars_rhs = fb.columns_rhs
            df_model_mm = df_model.select(vars_rhs)
            df_impute_mm = df_impute.select(vars_rhs)
        else:
            f = FormulaBuilder(formula=formula)
            f.remove_constant()
            #   Named to match the list-formula branch's vars_rhs above -
            #   the min_n_x_var block below references vars_rhs regardless
            #   of which branch built the model frame.
            vars_rhs = FormulaBuilder.columns_from_formula(formula=f.rhs())

            #   Fit the spec against the union of every frame it'll be
            #   reapplied to (with null_dummy=True) so a companion
            #   null-indicator column gets allocated for any predictor
            #   that's null in df_impute even if it has no nulls in
            #   df_model - otherwise a null showing up only at reapply time
            #   raises (no companion column was allocated for it when the
            #   spec's structure was fixed at fit time).
            frames_to_fit = [df_model.select(vars_rhs), df_impute.select(vars_rhs)]
            df_fit_union = pl.concat(frames_to_fit, how="diagonal")

            #   ModelSpec.from_formula() parses with survey_kit_formula's own
            #   parse_formula(), which (unlike FormulaBuilder's parsing)
            #   requires a "~" - f.rhs() is a bare rhs-only fragment.
            model_spec = ModelSpec.from_formula(
                f"~{f.rhs()}", df_fit_union, null_dummy=True
            )
            df_model_mm = model_spec.get_model_frame(df_model)
            df_impute_mm = model_spec.get_model_frame(df_impute)

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
            #   round_output=False can leave df_estimates as a LazyFrame (see
            #   _post_impute_statistics's use of nw.from_native(...) on the
            #   same round_output=False output) - collect before the polars-
            #   native filter+bracket-index below, which only works on an
            #   eager DataFrame.
            df_n_not0 = nw.from_native(sc.df_estimates).lazy().collect().to_native()

            vars_rhs = df_n_not0.filter(pl.col("n (not 0)") >= min_n_x_var)[
                "Variable"
            ].to_list()
            self.logging.info(
                f"            Dropping {df_n_not0.filter(pl.col('n (not 0)') < min_n_x_var)['Variable'].to_list()}"
            )

            df_model_mm = df_model_mm.select(vars_rhs)
            df_impute_mm = df_impute_mm.select(vars_rhs)

        #   categorical_feature columns not referenced in the formula at
        #       all never reach df_model_mm/df_impute_mm through
        #       model_spec (only in the formula branch - the list branch
        #       above already selected them in as ordinary raw
        #       predictors), so add them in here as plain untouched
        #       columns, ready for estimator_prepare_data (below) to cast
        #       to a native categorical dtype. One that IS referenced
        #       (Variable._validate_estimator_available only allows this
        #       for a numeric-dtype column) already made it through as a
        #       plain passthrough term under its own name - skip it here,
        #       concatenating it again would raise on the duplicate name.
        categorical_feature = self.variable.parameters.get("categorical_feature")
        if categorical_feature and type(self.variable.model) is not list:
            missing_categorical = [
                c for c in categorical_feature if c not in df_model_mm.columns
            ]
            if missing_categorical:
                df_model_mm = pl.concat(
                    [df_model_mm, df_model.select(missing_categorical)],
                    how="horizontal",
                )
                df_impute_mm = pl.concat(
                    [df_impute_mm, df_impute.select(missing_categorical)],
                    how="horizontal",
                )

        return df_model_mm, df_impute_mm, vars_rhs

    def _run_regression(
        self,
        df_model: IntoFrameT,
        df_impute: IntoFrameT,
        model_vars: list,
        formula: str,
        regmodel: Parameters.RegressionModel | None = None,
        min_n_x_var: int = 0,
    ) -> tuple[IntoFrameT, IntoFrameT, IntoFrameT]:
        nw_model = NarwhalsType(df_model)
        nw_impute = NarwhalsType(df_impute)

        df_model = nw_model.to_polars().lazy().collect()
        df_impute = nw_impute.to_polars().lazy().collect()

        if regmodel is None:
            #   .get(), not ["model"] - RandomForest()/XGBoost()/CatBoost()/
            #       SklearnModel() don't set "model" at all (they set
            #       "estimator" instead, checked below), so this can
            #       legitimately stay None here without being a missing-key
            #       error.
            regmodel = self.variable.parameters.get("model")

        if "random_share" in self.variable.parameters.keys():
            random_share = self.variable.parameters["random_share"]
        else:
            random_share = 1
        if random_share < 1:
            self.logging.info(f"     Using a {random_share} subsample")
            df_model = df_model.sample(fraction=random_share, seed=generate_seed())

        df_model_mm, df_impute_mm, vars_rhs = self._build_model_matrix(
            df_model=df_model,
            df_impute=df_impute,
            formula=formula,
            min_n_x_var=min_n_x_var,
        )

        #   Set by Parameters.RandomForest()/XGBoost()/CatBoost()/
        #       SklearnModel() - a hook the model choice itself owns for
        #       whatever data prep IT needs (e.g. XGBoost/CatBoost casting
        #       their declared categorical columns to a fixed-category
        #       dtype) - _run_regression stays agnostic to what, if
        #       anything, actually happens here. Identity/no-op for
        #       OLS/Logit (no adapter at all) and for models that don't
        #       need any special prep.
        prepare_data = self.variable.parameters.get("estimator_prepare_data")
        if prepare_data is not None:
            df_model_mm, df_impute_mm = prepare_data(df_model_mm, df_impute_mm)

        #   Set by Parameters.RandomForest()/XGBoost()/CatBoost()/
        #       SklearnModel() - always a zero-arg factory already, not a
        #       preset name to resolve. Kept as a factory (not just one
        #       instance) so the CV-fold code below can build fresh
        #       instances that keep whatever hyperparameters the factory
        #       bakes in, rather than reconstructing with type(model)()
        #       and silently losing them.
        model_factory = self.variable.parameters.get("estimator")
        if model_factory is None:
            if regmodel == Parameters.RegressionModel.OLS:
                from sklearn.linear_model import LinearRegression

                model_factory = LinearRegression
            elif regmodel == Parameters.RegressionModel.Logit:
                from sklearn.linear_model import LogisticRegression

                model_factory = LogisticRegression
            # elif regmodel == Parameters.RegressionModel.Probit:

        model = model_factory()

        d_extra_model_args = {}
        if self.weight != "":
            d_extra_model_args["sample_weight"] = df_model[self.weight]

        #   Set by Parameters.Regression()/RandomForest()/XGBoost()/
        #       CatBoost()/SklearnModel() - a cheap, shrinkage-heuristic
        #       stand-in for a nested random-intercept term (e.g. state ->
        #       county -> hhid), not a real mixed model - see
        #       _nested_group_shrinkage's docstring. y_for_fit is the
        #       target net of the PRIOR iteration's group-intercept
        #       estimate (0 on the first iteration, before one exists);
        #       the model then only has to explain what that didn't
        #       already, and the group intercept itself gets re-estimated
        #       below from THIS fit's residuals, ready for next iteration.
        errordraw = self.variable.parameters.get("error")
        group_levels = self.variable.parameters.get("group_levels", [])
        prior_intercept_col = f"___group_intercept_{self.variable.impute_var}___"
        if group_levels:
            prior_intercept_model = (
                df_model[prior_intercept_col].to_numpy()
                if prior_intercept_col in df_model.columns
                else np.zeros(safe_height(df_model))
            )
            y_for_fit = pl.DataFrame(
                {
                    self.variable.impute_var: (
                        df_model[self.variable.impute_var].to_numpy()
                        - prior_intercept_model
                    )
                }
            )
        else:
            y_for_fit = df_model.select(self.variable.impute_var)

        model.fit(
            X=df_model_mm,
            y=y_for_fit,
            **d_extra_model_args,
        )

        #   Linear models (OLS/Logit) expose coef_/intercept_ - report
        #       those as before. Tree/ensemble estimators don't have
        #       linear coefficients but usually expose
        #       feature_importances_ instead; fall back to that, or an
        #       empty table if the estimator exposes neither.
        if hasattr(model, "coef_"):
            coef = model.coef_[0] if np.ndim(model.coef_) > 1 else model.coef_
            intercept = (
                model.intercept_[0]
                if np.ndim(model.intercept_) > 0
                else model.intercept_
            )
            df_betas = pl.DataFrame(
                dict(
                    Variable=safe_columns(df_model_mm) + ["_Intercept_"],
                    Beta=[float(vali) for vali in list(coef) + [float(intercept)]],
                )
            )
        elif hasattr(model, "feature_importances_"):
            df_betas = pl.DataFrame(
                dict(
                    Variable=safe_columns(df_model_mm),
                    Beta=[float(vali) for vali in model.feature_importances_],
                )
            )
        else:
            df_betas = pl.DataFrame(dict(Variable=[], Beta=[]))

        #   For Logit, model.predict() returns the hard 0/1 class label, not a
        #   probability - PMM matching and the "Random" error draw both need the
        #   continuous predicted probability (P(y=1|X)), so use predict_proba()
        #   instead. OLS has no predict_proba() and model.predict() is already
        #   the continuous yhat we want.
        if regmodel == Parameters.RegressionModel.Logit:

            def _predict(X):
                return model.predict_proba(X)[:, 1]
        else:

            def _predict(X):
                return model.predict(X)

        cv_folds = self.variable.parameters.get("cv_folds", 0)
        if cv_folds and cv_folds > 1:
            #   Donor pool prediction via cv_folds-way cross-validation
            #       instead of the in-sample fit - see
            #       Parameters._tabular_ml_params's cv_folds docstring.
            #       The final model (fit on all of
            #       df_model, above) still supplies df_betas/df_impute's
            #       prediction - recipients are already genuinely
            #       out-of-sample, so they don't need CV treatment.
            def _fit_predict_fold(is_holdout):
                train_mask = ~is_holdout
                fold_model = model_factory()
                fold_extra_args = {}
                if self.weight != "":
                    fold_extra_args["sample_weight"] = df_model.filter(train_mask)[
                        self.weight
                    ]
                fold_model.fit(
                    X=df_model_mm.filter(train_mask),
                    #   y_for_fit, not df_model.select(impute_var) - stays
                    #       net of the prior group intercept, same as the
                    #       main fit above, so cv_folds and group_levels
                    #       combine consistently.
                    y=y_for_fit.filter(train_mask),
                    **fold_extra_args,
                )
                X_holdout = df_model_mm.filter(is_holdout)
                if regmodel == Parameters.RegressionModel.Logit:
                    return fold_model.predict_proba(X_holdout)[:, 1]
                else:
                    #   y was fit as a 1-column DataFrame (2D), so
                    #       LinearRegression.predict() returns shape
                    #       (n, 1) here - ravel to 1D to match what
                    #       _pmm_cv_out_of_fold_predictions expects.
                    return fold_model.predict(X_holdout).ravel()

            predict_model = pl.DataFrame(
                self._pmm_cv_out_of_fold_predictions(
                    n_rows=safe_height(df_model),
                    cv_folds=cv_folds,
                    fit_predict_fold=_fit_predict_fold,
                ),
                schema=dict(___prediction=pl.Float64),
            )
        else:
            predict_model = pl.DataFrame(
                _predict(df_model_mm), schema=dict(___prediction=pl.Float64)
            )
        predict_impute = pl.DataFrame(
            _predict(df_impute_mm), schema=dict(___prediction=pl.Float64)
        )

        #   error=ErrorDraw.leaf (RandomForest()/XGBoost()/CatBoost()/
        #       SklearnModel() only - see ErrorDraw.leaf's docstring)
        #       donates by tree leaf co-occurrence instead of PMM's
        #       knearest-on-scalar-yhat. The leaf ids only make sense
        #       from THIS fitted model against THIS model matrix, both
        #       still in scope here (and about to be deleted below), so
        #       extract them now and persist as a plain column - the
        #       same "just another variable to carry along" pattern
        #       group_levels' prior_intercept_col already uses -
        #       _regression_draw_errors's leaf branch reads it back out
        #       after _run_regression returns.
        if errordraw == Parameters.ErrorDraw.leaf:
            self.logging.info("     Extracting leaf indices for leaf-based donor matching")
            leaf_model = extract_leaf_indices(model, df_model_mm)
            leaf_impute = extract_leaf_indices(model, df_impute_mm)
            df_model = df_model.with_columns(pl.Series("___leaf_ids___", leaf_model))
            df_impute = df_impute.with_columns(pl.Series("___leaf_ids___", leaf_impute))

        del df_model_mm
        del df_impute_mm

        if group_levels:
            #   Residual of THIS fit (whatever produced predict_model -
            #       in-sample or cv_folds, doesn't matter which) against
            #       the same residualized target the model was fit to -
            #       what's left over is what the group levels get a shot
            #       at explaining.
            fit_residual = (
                y_for_fit[self.variable.impute_var].to_numpy()
                - predict_model["___prediction"].to_numpy()
            )
            intercept_model, intercept_impute = self._nested_group_shrinkage(
                df_model=df_model.with_columns(
                    pl.Series("___group_fit_residual___", fit_residual)
                ),
                df_impute=df_impute,
                group_levels=group_levels,
                residual_col="___group_fit_residual___",
                k=self.variable.parameters.get("group_shrinkage_k", 10.0),
                #   Same weight the model fit itself already used for
                #       sample_weight - see _nested_group_shrinkage's
                #       weight docstring for why this matters for
                #       consistency, not just correctness.
                weight=(
                    df_model[self.weight].to_numpy() if self.weight != "" else None
                ),
            )
            predict_model = predict_model.with_columns(
                (pl.col("___prediction") + pl.Series(intercept_model)).alias(
                    "___prediction"
                )
            )
            predict_impute = predict_impute.with_columns(
                (pl.col("___prediction") + pl.Series(intercept_impute)).alias(
                    "___prediction"
                )
            )
            #   Carried through in df_model/df_impute (drop_if_exists,
            #       not with_columns, in case a prior iteration's copy of
            #       this column is already present from being selected in
            #       upstream) so regression() can persist it back into the
            #       working df for next iteration to read as
            #       prior_intercept_model above.
            df_model = drop_if_exists(df_model, prior_intercept_col).with_columns(
                pl.Series(prior_intercept_col, intercept_model)
            )
            df_impute = drop_if_exists(df_impute, prior_intercept_col).with_columns(
                pl.Series(prior_intercept_col, intercept_impute)
            )

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

        df_model = nw_model.from_polars(df_model)
        df_impute = nw_impute.from_polars(df_impute)
        df_betas = nw_model.from_polars(df_betas)

        return (df_model, df_impute, df_betas)

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

    _PMM_WINSOR_TRUE_VALUE_COL = "___pmm_winsor_true_value___"

    def _pmm_winsorize_for_fit(self, df_model: IntoFrameT) -> tuple[IntoFrameT, bool]:
        """
        Clip the outcome before it's used to fit the regression that produces
        ___prediction, so a handful of extreme values don't distort the fit
        (matching Selection.lasso()'s same winsorize-before-fit pattern).

        Only the fit is affected - the donor's actual value must stay
        un-clipped, since it's what PMM copies onto the recipient. The true
        value is preserved in a temp column here and restored via
        _pmm_restore_true_value() once ___prediction has been computed from
        it, before df_model is used as the donor pool.

        Returns
        -------
        tuple[IntoFrameT, bool]
            The (possibly winsorized) df_model, and whether winsorizing
            actually happened (i.e. whether restoration is needed later).
        """
        #   [0, 1] is the no-winsorization sentinel (Parameters.pmm's default).
        winsor = self.variable.parameters.get("winsor", [0, 1])
        if list(winsor) == [0, 1]:
            return (df_model, False)

        df_model = (
            nw.from_native(df_model)
            .with_columns(
                nw.col(self.variable.impute_var).alias(
                    self._PMM_WINSOR_TRUE_VALUE_COL
                )
            )
            .to_native()
        )
        df_model = winsorize_by_percentiles(
            df=df_model, percentiles=winsor, columns=self.variable.impute_var
        )
        return (df_model, True)

    def _pmm_restore_true_value(
        self, df_model: IntoFrameT, b_winsorized: bool
    ) -> IntoFrameT:
        if not b_winsorized:
            return df_model

        return (
            nw.from_native(df_model)
            .with_columns(
                nw.col(self._PMM_WINSOR_TRUE_VALUE_COL).alias(
                    self.variable.impute_var
                )
            )
            .drop(self._PMM_WINSOR_TRUE_VALUE_COL)
            .to_native()
        )

    def _pmm_cv_out_of_fold_predictions(
        self,
        n_rows: int,
        cv_folds: int,
        fit_predict_fold,
    ) -> np.ndarray:
        """
        Split n_rows into cv_folds random folds and return an array of
        out-of-fold predictions covering every row, in original row order.

        Used so the donor pool's matching prediction comes from a fit that
        never saw that donor's own y, instead of the in-sample fit's
        prediction - see Parameters._tabular_ml_params's cv_folds
        docstring for why that matters.

        Parameters
        ----------
        n_rows : int
            Number of rows in the donor pool to fold over.
        cv_folds : int
            Number of folds (must be > 1 to actually cross-validate).
        fit_predict_fold : Callable[[np.ndarray], np.ndarray]
            Given a boolean mask marking the held-out fold (True = held
            out), fits on the complement and returns predictions for the
            held-out rows, in their relative order within that mask.

        Returns
        -------
        np.ndarray
            Out-of-fold predictions, one per row, in original row order.
        """
        rng = RandomNumberGenerator()
        fold_assignment = rng.integers(0, cv_folds, size=n_rows)

        predictions = np.empty(n_rows, dtype=float)
        for foldi in range(cv_folds):
            is_holdout = fold_assignment == foldi
            predictions[is_holdout] = fit_predict_fold(is_holdout)

        return predictions

    def _nested_group_shrinkage(
        self,
        df_model: pl.DataFrame,
        df_impute: pl.DataFrame,
        group_levels: list[str],
        residual_col: str,
        k: float,
        weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        A cheap, shrinkage-heuristic stand-in for a random-intercept term
        (state/county/hhid-style nested clustering) - not a real mixed
        model (no joint variance-component estimation, no Gibbs sampling),
        just an empirical-Bayes-style shrunk group mean, applied one
        nested level at a time, coarsest first.

        At each level, the group mean of what's left of the residual is
        shrunk toward 0 (residuals are already centered, so "shrink
        toward 0" is "shrink toward no group effect") by
        n_group / (n_group + k) - a small/noisy group gets pulled close
        to 0, a large group keeps close to its own raw mean. That shrunk
        value is subtracted out before moving to the next (finer) level,
        so a two-level (state, county) pass decomposes the residual into
        a state effect plus a county effect *within* what the state
        didn't already explain, rather than double-counting.

        This intentionally never estimates more than one number per row
        per level (no covariance matrix, no per-cluster slopes) - seat
        with the design discussion this implements: get most of what a
        random intercept buys you (shrinkage for small/noisy clusters, a
        cluster-level baseline) without any of a real mixed model's
        estimation cost. Meant to be re-run every SRMI iteration, each
        time on that iteration's fresh residuals (see _run_regression),
        so it improves alongside everything else SRMI already refits
        iteration to iteration rather than needing its own inner
        convergence loop.

        Parameters
        ----------
        df_model : pl.DataFrame
            Donor pool - must contain group_levels and residual_col.
        df_impute : pl.DataFrame
            Recipients - must contain group_levels (not residual_col - a
            recipient's shrunk intercept comes entirely from the donor
            pool's group statistics). A recipient in a group never seen
            in df_model gets 0 at that level (and, for a group seen at a
            coarser level but not this finer one, whatever the coarser
            level(s) already contributed).
        group_levels : list[str]
            Column names, ordered COARSEST to FINEST (e.g.
            ["state", "county", "hhid"]) - order matters, since each
            level only sees what the coarser ones left behind.
        residual_col : str
            Column in df_model holding the fixed-effect model's residual
            (actual - predicted) to decompose.
        k : float
            Shrinkage constant - a group needs roughly this many
            (weighted, if weight is set) observations before its own mean
            starts to dominate over being pulled toward 0.
        weight : np.ndarray | None, optional
            Row weight for df_model's rows (self.weight - the bootstrap or
            declared weight, same one the model fit itself already used
            for sample_weight), by default None (every row counts equally,
            same as before this was added). When set, both the group mean
            and the "how much data do we have" measure become weighted -
            a weighted mean instead of a plain one, and sum-of-weight
            instead of row-count in the n/(n+k) shrinkage factor - so a
            few high-weight rows can outweigh many low-weight ones, the
            same way they already do in the model fit. Rescaled internally
            so weights average to 1 across df_model, regardless of the
            original weight's scale (a declared survey weight often sums
            to a population total, not row count) - keeps k's units
            comparable to "roughly this many observations", the same
            meaning it has when weight is None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (total shrunk intercept for df_model's rows, same for
            df_impute's rows) - the sum of every level's contribution,
            one float per row, in each frame's original row order.
        """
        resid = df_model[residual_col].to_numpy().astype(float).copy()
        total_model = np.zeros(len(df_model))
        total_impute = np.zeros(len(df_impute))

        if weight is not None:
            weight = weight.astype(float)
            weight = weight * (len(weight) / weight.sum())

        for level in group_levels:
            if weight is None:
                stats = (
                    df_model.select(level)
                    .with_columns(pl.Series("___resid___", resid))
                    .group_by(level)
                    .agg(
                        pl.col("___resid___").mean().alias("___mean___"),
                        pl.col("___resid___").len().alias("___n___"),
                    )
                )
            else:
                stats = (
                    df_model.select(level)
                    .with_columns(
                        pl.Series("___resid___", resid),
                        pl.Series("___weight___", weight),
                    )
                    .group_by(level)
                    .agg(
                        (
                            (pl.col("___resid___") * pl.col("___weight___")).sum()
                            / pl.col("___weight___").sum()
                        ).alias("___mean___"),
                        pl.col("___weight___").sum().alias("___n___"),
                    )
                )

            stats = stats.with_columns(
                (
                    pl.col("___mean___")
                    * pl.col("___n___")
                    / (pl.col("___n___") + k)
                ).alias("___shrunk___")
            ).select([level, "___shrunk___"])

            model_shrunk = (
                df_model.select(level)
                .join(stats, on=level, how="left")["___shrunk___"]
                .fill_null(0.0)
                .to_numpy()
            )
            impute_shrunk = (
                df_impute.select(level)
                .join(stats, on=level, how="left")["___shrunk___"]
                .fill_null(0.0)
                .to_numpy()
            )

            total_model += model_shrunk
            total_impute += impute_shrunk
            resid = resid - model_shrunk

        return total_model, total_impute

    ##########################################################
    ##########################################################
    #   HELPERS - pmm - END
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
                        extra_keep_vars=donate_by,
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
        extra_keep_vars: list[str] | None = None,
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
            df_impute.select(match_on)
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

        #   Imported here (not at module level) since sklearn's base import is
        #   ~450ms and this is the only place in the module that needs it -
        #   HotDeck/StatMatch-only runs never call this and shouldn't pay for it.
        from sklearn.neighbors import KDTree

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
        df_randoms = (
            df_randoms.with_columns(
                pl.col("___possiblematches")
                .arr.get(pl.col("___matched"))
                .alias("___selectedmatch")
            )
            .select("___selectedmatch")
            #   Tag the original recipient row order explicitly before the
            #       join below - polars' own docs disclaim relying on a
            #       join's output row order without maintain_order, and the
            #       result here gets paired with df_impute purely by row
            #       position a few lines below (in the caller), so losing
            #       that order would silently pair recipients with the
            #       wrong donated value.
            .with_row_index(name="___orig_row_order")
        )

        #   Get the donate variable values from df_model
        df_matched = (
            join_list(
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
            )
            #   Restore the original recipient row order regardless of what
            #       the join itself did internally.
            .sort("___orig_row_order")
            .drop(["___selectedmatch", "___orig_row_order"])
        )

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
        #   extra_keep_vars carries the donate_by group key(s) through when
        #       called from _find_nearest_neighbor_by, so the group identity
        #       survives into the combined multi-group result and downstream
        #       rank-alignment can stay within-group instead of mixing donor
        #       values across donate_by strata.
        keep_vars = self.index + (extra_keep_vars if extra_keep_vars else [])
        #   self.original_variable.weight (the Variable's own declared
        #       weight - _post_impute_statistics reads it, not self.weight,
        #       for its descriptive display) needs to survive this narrowing
        #       too, or that call fails looking for a column that got
        #       dropped here even though an upstream caller fetched it.
        #       Only added if the caller already fetched it (checked via
        #       schema, not blindly appended) - not every caller of this
        #       method does, and this is a read-only narrowing step, not
        #       the place to go fetch a column nobody asked for.
        if (
            self.original_variable.weight != ""
            and self.original_variable.weight not in keep_vars
            and self.original_variable.weight
            in nw.from_native(df_impute).lazy().collect_schema().names()
        ):
            keep_vars.append(self.original_variable.weight)
        df_matched = concat_wrapper(
            [df_impute.select(keep_vars), df_matched.select(donate_vars)],
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
                nw.when(~nw.col(f"{vari}_right").is_null())
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
            #   Plain (not |not0) - unlike the |not0 variants below, this
            #       is safe to use as-is for a variable where 0 is a
            #       genuine value (e.g. Variable.two_part()'s semicontinuous
            #       output) - SRMI.convergence() relies on this one, not
            #       std|not0, for chainVar (see convergence.py).
            "std",
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
        #   SRMI.convergence()'s chainMean/chainVar equivalent - the
        #       mean/std of JUST this variable's own newly-imputed
        #       values this call (unweighted, matching mice's own
        #       chainMean/chainVar, which are also unweighted).
        #       Independent of donate_vars (which may carry unrelated
        #       donate_list extras) and independent of the descriptive-
        #       stats table built below (which is display-oriented -
        #       "Variable" is blanked past each call's first row for
        #       print-friendliness, so a caller can't reliably re-
        #       identify which row is impute_var's own afterward when
        #       there's more than one donate_var). None for a non-
        #       numeric/boolean target (e.g. Multinomial()/
        #       OrderedCategorical()'s category labels, or a string-
        #       valued HotDeck/StatMatch donate) - no meaningful
        #       mean/std there, and no convergence tracking for it.
        self.chain_mean = None
        self.chain_std = None
        numeric_impute_cols = (
            nw.from_native(df_impute)
            .lazy()
            .select(cs.numeric(), cs.boolean())
            .collect_schema()
            .names()
        )
        if self.variable.impute_var in numeric_impute_cols:
            imputed_values = (
                nw.from_native(df_impute)
                .lazy()
                .select(nw.col(self.variable.impute_var).cast(nw.Float64))
                .collect()[self.variable.impute_var]
                .to_numpy()
            )
            if imputed_values.size > 0:
                self.chain_mean = float(np.nanmean(imputed_values))
                self.chain_std = (
                    float(np.nanstd(imputed_values, ddof=1))
                    if imputed_values.size > 1
                    else float("nan")
                )

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

        self.logging.info(f"Post-imputation statistics for {donate_vars}")
        self.logging.info(f"    Where:          {self.variable.Where}")
        self.logging.info(f"    Where (impute): {self.variable.Where_impute}")

        #   Statistics' default stat list (mean/std/quantiles) is numeric/
        #       boolean only - it silently drops any other-dtype column
        #       from consideration (Statistics._resolve_summary_df's
        #       cs.numeric()/cs.boolean() select), which is fine when
        #       donate_vars is a mix (the non-numeric ones just don't
        #       appear in the table) but crashes downstream
        #       ("No items to concatenate") if EVERY donate_vars column
        #       ends up dropped - e.g. a category label (Multinomial(),
        #       OrderedCategorical(), or a string-valued HotDeck/
        #       StatMatch donate_vars). Skip the stats computation
        #       entirely in that case rather than let it crash - there's
        #       nothing numeric to summarize.
        has_summarizable_dtype = (
            len(
                nw.from_native(df_summary)
                .lazy()
                .select(cs.numeric(), cs.boolean())
                .collect_schema()
                .names()
            )
            > 0
        )
        if not has_summarizable_dtype:
            self.logging.info(
                f"    (descriptive stats skipped - {donate_vars} has no "
                f"numeric/boolean column to summarize)"
            )
            return

        if self.parent.imputation_stats is not None:
            stats_to_calculate = self.parent.imputation_stats
        else:
            stats_to_calculate = Impute._post_impute_statistics_items()
        statistics = Statistics(stats=stats_to_calculate, columns=donate_vars)
        summarize_by = {"All": [], "impute": ["Imputed"]}

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
            nw.from_native(self.variable.df_impute_where(df=df, keep_vars=keep_vars))
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

        #   df_predict_where's result gets filtered on impute_var nullness
        #       right below - make sure impute_var survives the keep_vars
        #       projection inside df_predict_where even if the caller didn't
        #       ask for it, then narrow back to keep_vars via the .select()
        #       below (unchanged from before this projection was added).
        where_keep_vars = keep_vars
        if self.variable.impute_var not in keep_vars:
            where_keep_vars = keep_vars + [self.variable.impute_var]

        df = (
            nw.from_native(
                self.variable.df_predict_where(
                    df=df, drop_imputed=drop_imputed, keep_vars=where_keep_vars
                )
            )
            .filter(~nw.col(self.variable.impute_var).is_null())
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
