from __future__ import annotations
from typing import Optional
import os
import logging
import narwhals as nw
from narwhals.typing import IntoFrameT
from copy import deepcopy
from enum import Enum

import formulaic
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression


from ..utilities.formula_builder import FormulaBuilder

from ..orchestration.config import Config
from .utilities.lasso import Lasso as rep_lasso
from ..utilities.dataframe import safe_height, winsorize_by_percentiles
from ..serializable import Serializable

from .. import logger


class Selection(Serializable):
    """
    Handles variable selection for imputation models.

    Supports various selection methods including LASSO, stepwise selection,
    and custom functions to reduce model dimensionality.

    Parameters
    ----------
    method : Selection.Method, optional
        Selection method to use, by default Method.No
    parameters : dict, optional
        Method-specific parameters, by default None
    select_within_by : bool, optional
        Whether to run selection within each by-group, by default True
    function : callable, optional
        Custom selection function, by default None
    """

    #   List of acceptable methods
    #       Commented out if not yet implemented
    class Method(Enum):
        No = 0
        Stepwise = 1
        LASSO = 2
        #   LightGBM = 3
        Custom = 4

    class Parameters:
        def lasso(
            nfolds: int = 5,
            type_measure: str = "default",
            include_base_with_interaction: bool = True,
            winsorize: tuple[float, float] | None = None,
            continuous: bool = False,
            binomial: bool = False,
            missing_dummies: bool = True,
            optimal_lambda: float = None,
            optimal_lambda_from_pre: bool = True,
            scale_lambda: float = 1.0,
        ) -> dict:
            """
            Parameters for LASSO variable selection method.

            Parameters
            ----------
            nfolds : int, default=5
                Number of folds for cross-validation during LASSO regression. Used to
                determine optimal lambda value through k-fold cross-validation.
            type_measure : str, default="default"
                Type of measure to use for cross-validation error. Determines how model
                performance is evaluated during lambda selection. Options are "default",
                "mse", "deviance", "class", "auc", "mae".
            include_base_with_interaction : bool, default=True
                Whether to include base variables when interaction terms are selected.
                If True, base variables are automatically included when their interactions
                are selected by LASSO.
            winsorize : tuple[float, float] or None, default=None
                Percentile bounds for winsorizing the dependent variable. Tuple of
                (low_percentile, high_percentile) to cap extreme values. None means
                no winsorization.
            continuous : bool, default=False
                Force treatment of dependent variable as continuous, overriding
                automatic detection.
            binomial : bool, default=False
                Force treatment of dependent variable as binomial, overriding
                automatic detection.
            missing_dummies : bool, default=True
                Whether to create dummy variables for missing values in predictor variables.
            optimal_lambda : float or None, default=None
                Pre-specified optimal lambda value for LASSO regularization. If None,
                optimal lambda will be determined through cross-validation.
            optimal_lambda_from_pre : bool, default=True
                Whether to use optimal lambda from a previous preselection step.
            scale_lambda : float, default=1.0
                Scaling factor applied to the optimal lambda value. Values > 1 make
                regularization stronger (fewer variables selected), values < 1 make
                it weaker (more variables selected).

            Returns
            -------
            dict
                Dictionary containing all parameter values for LASSO selection.

            Raises
            ------
            Exception
                When type_measure is not one of the acceptable values.
            """

            arguments = deepcopy(locals())

            type_measure_acceptable = [
                "default",
                "mse",
                "deviance",
                "class",
                "auc",
                "mae",
            ]
            #   Error checking
            if type_measure not in type_measure_acceptable:
                message = f"type_measure options are {type_measure_acceptable} (passed {type_measure})"
                logger.error(message)
                raise Exception(message)

            return arguments

        def lightgbm():
            return {}

        def stepwise(
            nfolds: int = 5,
            scoring: str = "neg_mean_squared_error",
            # Include base variables with interactions
            include_base_with_interaction: bool = True,
            # winsorize dependent on selection [low ptile, high ptile]
            winsorize: tuple[float, float] | None = None,
            # Force use of one or the other, otherwise, defaults based on data
            missing_dummies: bool = True,
            min_features_to_select: int = 10,
        ):
            """
            Parameters for stepwise variable selection method using Recursive Feature
            Elimination with Cross-Validation (RFECV).

            Parameters
            ----------
            nfolds : int, default=5
                Number of folds for cross-validation during stepwise selection. Used
                in RFECV to evaluate feature importance.
            scoring : str, default="neg_mean_squared_error"
                Scoring metric used to evaluate model performance during cross-validation.
                Should be a valid sklearn scoring parameter.
            include_base_with_interaction : bool, default=True
                Whether to include base variables when interaction terms are selected.
                If True, base variables are automatically included when their interactions
                are selected.
            winsorize : tuple[float, float] or None, default=None
                Percentile bounds for winsorizing the dependent variable. Tuple of
                (low_percentile, high_percentile) to cap extreme values. None means
                no winsorization.
            missing_dummies : bool, default=True
                Whether to create dummy variables for missing values in predictor variables.
            min_features_to_select : int, default=10
                Minimum number of features that must be selected by the stepwise procedure.
                Prevents over-reduction of the feature set.

            Returns
            -------
            dict
                Dictionary containing all parameter values for stepwise selection.
            """
            return deepcopy(locals())

    def __init__(
        self,
        method: Selection.Method = Method.No,
        parameters: dict = None,
        select_within_by: bool = True,
        function=None,
    ):
        """
        This class handles any variable selection steps needed to reduce
        the dimensionality of the imputation model

        Parameters
        ----------
        method : Selection.Method, optional
            Enumeration for existing selectio method. The default is Method.No (no selection).
        parameters : dict, optional
            Method-specific dictionary of parameters. The default is None.
        select_within_by : bool, optional
            During imputation, if True, run for each by group,
            if False, run once before all of them
        function: function handle, optional
            Custom selection function - if you want to use your own variable selection approach
                Function arguments need to be:
                df:IntoFrameT,
                variable:Variable,
                parameters:dict,
                preselection:bool

        Returns
        -------
        None.

        """

        if parameters is None:
            if method == Selection.Method.LASSO:
                parameters = Selection.Parameters.lasso()
            elif method == Selection.Method.Stepwise:
                parameters = Selection.Parameters.stepwise()
            #   elif method == Selection.Method.LightGBM:
            #       parameters = Selection.Parameters.lightgbm()
            else:
                parameters = {}

        if function is not None:
            method = Selection.Method.Custom
        self.method = method
        self.parameters = parameters
        self.function = function
        self.select_within_by = select_within_by

    def run(
        self,
        df: IntoFrameT,
        y: str,
        formula: str,
        weight: str = "",
        sub_log: logging | None = None,
    ):
        arguments = {"df": df, "y": y, "formula": formula, "sub_log": sub_log}

        delegate = None
        if self.method == Selection.Method.LASSO:
            delegate = self.lasso
        elif self.method == Selection.Method.Stepwise:
            delegate = self.stepwise
        # elif self.method == Selection.Method.LightGBM:
        #     delegate = self.lightgbm
        elif self.method == Selection.Method.Custom:
            delegate = self.function
            arguments["parameters"] = self.parameters
            arguments["preselection"] = self.preselection

        #   Call the actual selection function
        if delegate is None:
            return ""
        else:
            return delegate(**arguments)

    #   Selection methods - must have this signature/arguments
    def lasso(
        self,
        df: IntoFrameT,
        y: str,
        formula: str,
        weight: str = "",
        sub_log: logging | None = None,
    ):
        #   Save the original formula to compare to later
        formula_in = formula

        if sub_log is None:
            sub_log = logger

        #   Assign the parameter values
        nfolds = self.parameters["nfolds"]
        type_measure = self.parameters["type_measure"]
        winsorize = self.parameters["winsorize"]
        continuous = self.parameters["continuous"]
        binomial = self.parameters["binomial"]
        missing_dummies = self.parameters["missing_dummies"]
        optimal_lambda = self.parameters["optimal_lambda"]
        scale_lambda = self.parameters["scale_lambda"]

        #   collinearity_tolerance = self.parameters["collinearity_tolerance"]
        #   collinearity_tolerance_post = self.parameters["collinearity_tolerance_post"]

        include_base_with_interaction = self.parameters["include_base_with_interaction"]

        df = nw.from_native(df).filter(nw.col(y).is_not_missing()).to_native()

        if missing_dummies:
            [df, formula, missing_dummies] = Selection._add_missing_dummy(
                df=df, y=y, formula=formula
            )

        #   TODO - reimplement
        if winsorize is not None:
            df = winsorize_by_percentiles(df=df, percentiles=winsorize, columns=y)

        # matrix_y = as_matrix(PolarsToR(df_y))
        # del df_y

        #   TODO - Pre screen collinear variables?
        #   if collinearity_tolerance > 0:
        #       pass

        # r_args = {"y":matrix_y,
        #           "x":matrix_x,
        #           "parallel":True,
        #           "type.measure":type_measure,
        #           "alpha":1}

        # if weight != "":
        #     r_args["weights"] = as_matrix(PolarsToR(df.select(weight)))
        # if binomial:
        #     r_args["family"] = "binomial"
        # else:
        #     r_args["family"] = "gaussian"
        # if optimal_lambda is None:
        #     #   Get the lambda value
        #     r_args_crossval = deepcopy(r_args)
        #     r_args_crossval["nfolds"] = nfolds

        #     crossval = do_call(glmnet.cv_glmnet,dict_to_r_list(r_args_crossval))
        #     RInterop.set_item("crossval", crossval)
        #     optimal_lambda = RToPythonVariable(RInterop.get_item("crossval$lambda.min"))

        #     #   Assign it to the Selection object
        #     self.parameters["optimal_lambda"] = optimal_lambda

        #     #   Clean up
        #     RInterop.ReleaseMemory(remove_vars=["crossval"])

        #   With the optimal lambda set, run the lasso

        lm = rep_lasso(
            df=df,
            y=y,
            formula=formula,
            weight=weight,
            nfolds=nfolds,
            optimal_lambda=optimal_lambda,
        )

        if optimal_lambda is None:
            lm.find_optimal_lambda()
            optimal_lambda = lm.optimal_lambda

        if scale_lambda != 1:
            sub_log.info(f"         Scaling lambda by {scale_lambda}")
        lm.optimal_lambda = optimal_lambda * scale_lambda
        #   vars_kept = []
        vars_kept = lm.run()
        # try:

        #     # RInterop.set_item("fit", fit)
        #     # coefficients = RInterop.get_item("coef(fit)")
        #     # coefficients = RToPolars(as_data_frame(as_matrix(coefficients)),
        #     #                          KeepRowNames=True)
        #     # #   df_out = RToPolars(as_data_frame(tidy(fit)))
        #     # coefficients = coefficients.rename({
        #     #         coefficients.columns[0]:"Variable",
        #     #         coefficients.columns[1]:"Coefficient"
        #     #     })

        #     # #   Drop the empty coefficients
        #     # coefficients = coefficients.filter(pl.col("Coefficient") != 0)\
        #     #                            .filter(pl.col("Variable") != "(Intercept)")

        #     # vars_kept = coefficients["Variable"].to_list()
        # except Exception as error:
        #     sub_log.info(f"Error in glmnet call: {error}")

        if len(vars_kept) == 0:
            #   Are there any interactions, try without them
            [formula_nointeractions, any_dropped] = FormulaBuilder.exclude_interactions(
                formula=formula_in
            )

            if any_dropped:
                #   Changed formula, try lasso again
                output = self.lasso(
                    df=df,
                    y=y,
                    formula=formula_nointeractions,
                    weight=weight,
                    sub_log=sub_log,
                )
            else:
                #   Nothing changed, try stepwise
                output = self.stepwise(
                    df=df,
                    y=y,
                    formula=formula_nointeractions,
                    weight=weight,
                    sub_log=sub_log,
                )

            #   Return whatever we get - this runs recursively
            return output

        #   Clear the missingness dummies from the results
        if missing_dummies:
            vars_kept = [
                vari
                for vari in vars_kept
                if not vari.startswith("___missing___dummy___")
            ]

        #   Convert the final results into a formula
        fb = FormulaBuilder(df=df)
        fb.formula = formula_in
        fb.match_formula_to_columns(columns=vars_kept)

        if include_base_with_interaction:
            #   Get any base variables that go with the interaction
            fb.add_base_from_interactions()

        sub_log.info(f"         Selected model: ~{fb.rhs()}")
        return f"~{fb.rhs()}"

    def lightgbm(self, df: IntoFrameT, y: str, formula: str, weight: str = ""):
        pass

    def stepwise(
        self,
        df: IntoFrameT,
        y: str,
        formula: str,
        weight: str = "",
        sub_log: logging | None = None,
    ):
        #   Save the original formula to compare to later
        formula_in = formula

        if sub_log is None:
            sub_log = logger

        nfolds = self.parameters["nfolds"]

        if "scoring" in self.parameters.keys():
            scoring = self.parameters["scoring"]
        else:
            scoring = "neg_mean_squared_error"

        if "min_features_to_select" in self.parameters.keys():
            min_features_to_select = self.parameters["min_features_to_select"]
        else:
            min_features_to_select = 1

        winsorize = self.parameters["winsorize"]
        missing_dummies = self.parameters["missing_dummies"]
        include_base_with_interaction = self.parameters["include_base_with_interaction"]

        df = nw.from_native(df).filter(nw.col(y).is_not_missing()).to_native()
        if missing_dummies:
            [df, formula, missing_dummies] = Selection._add_missing_dummy(
                df=df, y=y, formula=formula
            )

        estimator = LinearRegression()
        selector = RFECV(
            estimator,
            step=1,
            cv=nfolds,
            scoring=scoring,
            min_features_to_select=min_features_to_select,
        )

        df_x = formulaic.Formula(formula).get_model_matrix(df)

        df_y = nw.from_native(df).select(y).to_native()
        # if winsorize is not None:
        #     df_y = winsorize_by_percentiles(df=df_y,
        #                                     percentiles=winsorize,
        #                                     columns=y)
        selector = selector.fit(
            nw.from_native(df_x).to_numpy(), nw.from_native(df_y).to_numpy()
        )

        vars_kept = []
        columns = nw.from_native(df_x).lazy().collect_schema().names()
        for i in range(0, len(columns)):
            if selector.support_[i]:
                vars_kept.append(columns[i])

        #   Clear the missingness dummies from the results
        if missing_dummies:
            vars_kept = [
                vari
                for vari in vars_kept
                if not vari.startswith("___missing___dummy___")
            ]

        #   Convert the final results into a formula
        fb = FormulaBuilder(df=df)
        fb.formula = formula_in
        fb.match_formula_to_columns(columns=vars_kept)

        if include_base_with_interaction:
            #   Get any base variables that go with the interaction
            fb.add_base_from_interactions()

        sub_log.info(f"         Selected model: ~{fb.rhs()}")
        return f"~{fb.rhs()}"

    def _add_missing_dummy(
        df: IntoFrameT, y: str, formula: str
    ) -> tuple[IntoFrameT, str, bool]:
        missing_dummies = []
        missing_recodes = []

        fb = FormulaBuilder(df=df)
        fb.formula = formula
        for coli in fb.columns:
            if coli != y:
                n_missing = safe_height(
                    nw.from_native(df)
                    .select(coli)
                    .filter(nw.col(coli).is_missing())
                    .to_native()
                )

                if n_missing > 0:
                    missing_dummies.append(
                        nw.when(nw.col(coli).is_missing())
                        .then(nw.lit(True))
                        .otherwise(nw.lit(False))
                        .alias(f"___missing___dummy___{coli}")
                    )
                    missing_recodes.append(
                        nw.when(nw.col(coli).is_missing())
                        .then(nw.lit(0))
                        .otherwise(nw.col(coli))
                        .alias(coli)
                    )

        if len(missing_dummies) > 0:
            df = (
                nw.from_native(df)
                .with_columns(missing_dummies + missing_recodes)
                .to_native()
            )

            #   Add these variables to the formula
            #       Update the formula to the new dataframe so it can find the variables
            fb.df = df
            fb.continuous(columns="___missing___dummy___*")

            return (df, formula, True)
        else:
            #   None missing
            return (df, formula, False)
