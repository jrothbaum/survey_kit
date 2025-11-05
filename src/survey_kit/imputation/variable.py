from __future__ import annotations

import os
import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT
from enum import Enum
from copy import deepcopy

from ..utilities.formula_builder import FormulaBuilder
from ..utilities.dataframe import columns_from_list, NarwhalsType, safe_height
from ..utilities.compress import compress_df

#   SRMI modules
from .selection import Selection

from ..serializable import Serializable
from .. import logger


class Variable(Serializable):
    _save_suffix = "srmi.variable"

    """
    Defines a variable to be imputed and its imputation specifications.
    
    This class encapsulates all settings for imputing a single variable,
    including model type, formula, selection methods, and conditions.
    
    Parameters
    ----------
    impute_var : str
        Name of the variable to be imputed
    Where : nw.Expr | None, optional
        General condition to restrict sample for imputation, by default None
            This restricts the sample before anything happens
    Where_impute : nw.Expr | None, optional
        Condition defining observations to be imputed, by default None
            Whose values are getting imputed
    Where_predict : nw.Expr | None, optional
        Condition defining observations for prediction, by default None
            Whose values are used for the prediction (i.e. in a regression)
    Where_predict_only_when_not_imputed : bool, optional
        Only predict for non-imputed observations, by default False
            i.e. in iteration 2, should I include the imputed values from 
            iteration 1 in the prediction model?
    bimpute_if_missing : bool, optional
        Include missingness as imputation condition, by default True
    preFunctions : list | Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | nw.Expr | None, optional
        Operations to run before imputation, by default None
    postFunctions : list | Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | nw.Expr | None, optional
        Operations to run after imputation, by default None
    preFunctions_initialize_implicate : list | Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | nw.Expr | None, optional
        Operations to run once before first iteration, by default None
            This can be useful if different data needs to be merged to each implicate
    predictors_exclude : list, optional
        Variables to exclude from the model, by default None
            i.e. I don't want to include occupation as a predictor
            for the presence of earnings since all earners have an occupation
    predictors_exclude_first_iteration : list, optional
        Variables to exclude only in first iteration, by default None
    predictors_require : list, optional
        Variables to always include in model, by default None
    weight : str, optional
        Weight variable name, by default ""
    joint : dict, optional
        Variables to include together if any is selected, by default None
    header : str, optional
        Header text for logging, by default ""
    model : str, optional
        R formula string or variable list for model, by default ""
    selection : Selection, optional
        Variable selection method, by default None
    preselection : Selection, optional
        Pre-imputation variable selection, by default None
    modeltype : ModelType, optional
        Type of imputation model, by default None
    modelfunction : callable, optional
        Custom imputation function, by default None
    parameters : dict, optional
        Model-specific parameters, by default None
    By : list, optional
        Variables defining by-groups for separate models, by default None
            i.e. if by=["state"] it would run a separate imputation model for each state
    """

    class ModelType(Enum):
        #   Predicted mean matching
        pmm = 0

        #   Predict y with lightgbm, then impute according to passed parameters
        LightGBM = 1

        #   Basic hot deck imputation (carrying arrays from prior observations)
        HotDeck = 2
        #   Stat match is theoretically ~= hot deck, but imputes are
        #       drawn from random sorts of the data and joins
        StatMatch = 3

        #   qreg = 4

        #   Predict y by regression, then impute according to passed parameters
        Regression = 5
        #   TwoSampleRegression = 6
        #   rifreg = 7
        #   quantile_spacing = 8

        #   Find nearest neighbor on x directly without reducing
        #       to a single index (yhat) from regression or lightgbm
        NearestNeighbor = 9

    class PrePost:
        """
        Namespace within Variable class for handling pre and post
            imputation operations.

        Currently that can be:
            1) a Narwhals Expr (NarwhalsExpression)
                which is anything you can put in nw.from_native(df).with_columns()
            2) a python function handle and parameters
                which allows you to call an arbitrary function
        """

        class NarwhalsExpression(Serializable):
            def __init__(self, expression: list[nw.Expr] | nw.Expr):
                """
                Pass the call information to call before or after an imputation step

                Parameters
                ----------
                expression:list[nw.Expr] | nw.Expr
                    A narwhals with_columns expression or list of expressions
                df_variable : str, optional
                    parameters[df_variable] to pass into the function.
                    The assumption is that anything that needs to happen pre/post
                    imputation needs the data.  The default is "df".

                Returns
                -------
                None.

                """

                self.expression = expression

            def call(self, df: IntoFrameT) -> IntoFrameT:
                # """
                # Call the specific pre-post function.

                # Parameters
                # ----------
                # df : IntoFrameT
                #     The current implicate data.

                # Returns
                # -------
                # IntoFrameT (data) to return as updated implicate data

                # """
                return nw.from_native(df).with_columns(self.expression).to_native()

        class Function:
            def __init__(
                self,
                delegate,
                parameters: dict | None = None,
                initialize: bool = False,
                df_variable: str = "",
            ):
                """
                Pass the call information to call before or after an imputation step

                Parameters
                ----------
                delegate : function handle
                    Function to be called
                parameters : dict | None, optional
                    Parameters that don't change with each call. The default is None.
                df_variable : str, optional
                    parameters[df_variable] to pass into the function.
                    The assumption is that anything that needs to happen pre/post
                    imputation needs the data.  The default is "df".

                Returns
                -------
                None.

                """

                if df_variable == "":
                    if initialize:
                        df_variable = "implicate"
                    else:
                        df_variable = "df"

                if parameters is None:
                    parameters = {}

                self.delegate = delegate
                self.parameters = parameters
                self.df_variable = df_variable

            def call(self, df: IntoFrameT) -> IntoFrameT:
                # """
                # Call the specific pre-post function.

                # Parameters
                # ----------
                # df : IntoFrameT
                #     The current implicate data.

                # Returns
                # -------
                # Lazy/DataFrame to return as updated implicate data

                # """
                if self.df_variable != "":
                    self.parameters[self.df_variable] = df
                df = self.delegate(**self.parameters)

                if self.df_variable != "":
                    del self.parameters[self.df_variable]

                return df

    def __init__(
        self,
        impute_var: str = "",
        Where: nw.Expr | None = None,
        Where_impute: nw.Expr | None = None,
        Where_predict: nw.Expr | None = None,
        Where_predict_only_when_not_imputed: bool = False,
        bimpute_if_missing: bool = True,
        preFunctions: list[
            Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | nw.Expr
        ]
        | Variable.PrePost.Function
        | Variable.PrePost.NarwhalsExpression
        | nw.Expr
        | None = None,
        postFunctions: list[
            Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | nw.Expr
        ]
        | Variable.PrePost.Function
        | Variable.PrePost.NarwhalsExpression
        | nw.Expr
        | None = None,
        preFunctions_initialize_implicate: list[
            Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | nw.Expr
        ]
        | Variable.PrePost.Function
        | Variable.PrePost.NarwhalsExpression
        | nw.Expr
        | None = None,
        predictors_exclude: list = None,
        predictors_exclude_first_iteration: list = None,
        predictors_require: list = None,
        weight: str = "",
        joint: dict = None,
        header: str = "",
        model: str = "",
        selection: Selection = None,
        preselection: Selection = None,
        modeltype: ModelType = None,
        modelfunction=None,
        parameters: dict = None,
        By: list = None,
    ):
        """


        Parameters
        ----------
        impute_var : str
            Variable to be imputed
        Where : nw.Expr, optional
            Condition to restrict sample for this imputation. The default is "".
        Where_impute : nw.Expr, optional
            Define the set of observations to be imputed, in addition
            to bimpute_if_missing | Where. The default is "".
        Where_predict : nw.Expr, optional
            Define the set of observations for the prediction | Where. The default is "".
        Where_predict_only_when_not_imputed : bool, optional
            Predict only if not imputed.  The default is False
        bimpute_if_missing : bool, optional
            Make an imputation condition that the variable is initially missing
        preFunctions : list, optional
            Any operations to run before this imputation at each iteration. The default is None.
        postFunctions : list, optional
            Any operations to run after this imputation at each iteration. The default is None.
        preFunctions_initialize_implicate: list, optional
            Any operations to run before this imputation ONLY ONCE before running the first implicate
            If it's a function, it will expect the implicate to be passed in
        predictors_exclude : list, optional
            What to exclude from the model. The default is None.
        predictors_exclude_first_iteration: list, optional
            The first iteration, it will exclude downstream variables
            by default, use this to override the default
        predictors_require : list, optional
            What to include no matter what. The default is None.
        weight : str, optional
            Weight for the imputation modeling. The default is "".
        joint : dict, optional
            dictionary of key (variable name) value lists/pairs where if the
            key is selected for the model, then so most the values. The default is None.
        header : str, optional
            Just a header to write to the log when this variable comes up. The default is "".
        model : str, optional
            R string formula, Override the "parent" SRMI model?. The default is "" (no).
        selection : Selection, optional
            Override the "parent" SRMI selection used?
            If variable selection is used within the imputation, this class
                handles it.  The default is no selection
        preselection : Selection, optional
            Override the "parent" SRMI selection used?
            If variable selection is done before the SRMI starts
                to pre-prune the inputs, this class handles it.
                The default is no selection
        modeltype : ModelType, optional
            Override the "parent" SRMI modeltype?. The default is "" (no).
        modelfunction : function delegate, optional
            Override modeltype completely and just run a custom imputation function, optional
            The function arguments are
                df:IntoFrameT - a dataframe with the full srmi data
                variable:Variable - an SRMI.Variable object ,
                index:list - the merge key of the data,
                weight:str - weight variable?
                sub_log:logging - to write the imputation output to a separate file in the
                    implicate folder
        parameters : dict, optional
            Override the "parent" SRMI model parameters?. The default is "" (no).
        By : list, optional
            Variable list for by groups

        Returns
        -------
        None.

        """

        self.impute_var = impute_var
        self.header = header

        #   Validate that the wheres are all either strings (for SQL filtering)
        #       or narwhals expressions (for .filter) - deprecated as only using expressions
        wheres = [Where, Where_impute, Where_predict]
        #   self.b_where_strings = all(type(wherei) is str or wherei is None for wherei in wheres)
        #   b_where_expressions = all(type(wherei) is nw.Expr or wherei is None for wherei in wheres)

        # if not (self.b_where_strings or b_where_expressions):
        #     where_types = {
        #                 "Where":type(Where),
        #                 "Where_impute":type(Where_impute),
        #                 "Where_predict":type(Where_predict)
        #         }
        #     message = f"Must pass where clauses as all strings or all narwhals expressions, passed as {where_types}"
        #     logger.error(message)
        #     raise Exception(message)

        self.Where = Where
        self.Where_impute_original = Where_impute
        self.Where_impute = Where_impute
        self.Where_predict = Where_predict
        self.Where_predict_only_when_not_imputed = Where_predict_only_when_not_imputed

        self.bimpute_if_missing = bimpute_if_missing

        self.preFunctions = self._parse_pre_post_function_inputs(preFunctions)
        self.postFunctions = self._parse_pre_post_function_inputs(postFunctions)
        self.preFunctions_initialize_implicate = self._parse_pre_post_function_inputs(
            preFunctions_initialize_implicate
        )

        if predictors_exclude is None:
            predictors_exclude = []
        self.predictors_exclude = predictors_exclude

        if predictors_exclude_first_iteration is None:
            predictors_exclude_first_iteration = []
        self.predictors_exclude_first_iteration = predictors_exclude_first_iteration

        self.predictors_require = predictors_require
        self.weight = weight
        self.joint = joint
        self.header = header
        self.model = model

        self.selection = selection

        if preselection is not None:
            preselection.preselection = True

        self.preselection = preselection
        self.modeltype = modeltype
        self.modelfunction = modelfunction

        if parameters is None:
            parameters = {}

        self.parameters = parameters

        if By is None:
            By = []
        if type(By) is str:
            By = [By]
        self.By = By

        #   Set later
        self.imputation_flag = ""

        #   No selection/pre-selection on hot deck or stat match
        if self.modeltype in [Variable.ModelType.HotDeck, Variable.ModelType.StatMatch]:
            #   logger.info(f"      Setting preselection to {Selection.Method.No} for {self.impute_var}, no selection for {self.modeltype}")
            self.preselection = Selection(method=Selection.Method.No)
            #   logger.info(f"      Setting selection to {Selection.Method.No} for {self.impute_var}, no selection for {self.modeltype}")
            self.selection = Selection(method=Selection.Method.No)

    def _parse_pre_post_function_inputs(
        self,
        functions: list[
            Variable.PrePost.Function
            | Variable.PrePost.NarwhalsExpression
            | nw.Expr
            | list[nw.Expr]
        ]
        | Variable.PrePost.Function
        | Variable.PrePost.NarwhalsExpression
        | nw.Expr
        | None = None,
    ) -> list[Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression]:
        if functions is None:
            functions = []
        if type(functions) is not list:
            functions = [functions]

        final_functions = []
        for fi in functions:
            if type(fi) == nw.Expr or type(fi) == list:
                final_functions.append(Variable.PrePost.NarwhalsExpression(fi))
            else:
                final_functions.append(fi)
        return final_functions

    def exclude_variables_from_models(
        self, df: IntoFrameT, additional_exclude: list = None
    ):
        if additional_exclude is None:
            additional_exclude = []

        exclude_list = [self.impute_var] + self.predictors_exclude + additional_exclude

        if "donate_list" in list(self.parameters.keys()):
            if len(self.parameters["donate_list"]):
                exclude_list.extend(self.parameters["donate_list"])

        #   Exclude the variable itself from it's own model
        #       as well as any other items in variable.predictors_exclude

        #   hot deck models - exclude this variable
        if "model_list" in list(self.parameters.keys()):
            if type(self.parameters["model_list"]) is list:
                final_list = []
                for modeli in range(len(self.parameters["model_list"])):
                    if type(self.parameters["model_list"][modeli]) is str:
                        item = FormulaBuilder.exclude_variables(
                            exclude_list=exclude_list,
                            formula=self.parameters["model_list"][modeli],
                            df=df,
                        )

                        if item != "":
                            final_list.append(item)
                    elif type(self.parameters["model_list"][modeli]) is list:
                        item = list(
                            set(self.parameters["model_list"][modeli]).difference(
                                exclude_list
                            )
                        )

                        if len(item) > 0:
                            final_list.append(item)

                self.parameters["model_list"] = final_list

        #   Exclude variables in the model
        if type(self.model) is str:
            self.model = FormulaBuilder.exclude_variables(
                exclude_list=exclude_list, formula=self.model, df=df
            )
        elif type(self.model) is list:
            self.model = columns_from_list(df=df, columns=self.model)
            self.model = [
                vari
                for vari in self.model
                if vari not in exclude_list and vari != self.impute_var
            ]

    def process_model(
        self, df: IntoFrameT, NoConstant: bool = False
    ) -> tuple[FormulaBuilder, FormulaBuilder, list[str]]:
        if type(self.model) is list:
            model_vars = self.model + [self.impute_var]

            if NoConstant:
                constant = "0"
            else:
                constant = "1"
            fb = FormulaBuilder(df=df)
            fb.formula = f"{self.impute_var}~{constant}+{'+'.join(self.model)}"
            fb_rhs = FormulaBuilder(df=df)
            fb_rhs.formula = f"~{constant}+{'+'.join(self.model)}"
        else:
            if self.model == "":
                formula = "~0"
            else:
                if NoConstant:
                    formula = self.model.replace("~1", "~0")
                else:
                    formula = self.model

            fb = FormulaBuilder(df=df)
            fb.formula = f"{self.impute_var}{formula}"

            fb_rhs = FormulaBuilder(df=df)
            fb_rhs.formula = formula
            model_vars = fb.columns
        return (fb, fb_rhs, model_vars)

    def validate_inputs(self, df: IntoFrameT):
        # """
        # Try to catch some variable specification errors upfront rather than
        #     finding out later that things don't work'

        # Parameters
        # ----------
        # df : IntoFrameT
        #     Input imputation data.

        # Returns
        # -------
        # None.  Throws errors if there are issues.

        # """

        #   Check for reserved variable names that will cause an error
        #       down the line and throw an error now to save time
        self._validate_reserved_names()

        if self.modeltype == Variable.ModelType.LightGBM:
            #   If impute_var is a dummy variable, can't run quantile gbm
            self._validate_lightgbm_boolean_quantile(df=df)

        if (
            self.modeltype == Variable.ModelType.HotDeck
            or self.modeltype == Variable.ModelType.StatMatch
        ):
            #   If donate_vars don't have the same missingness pattern
            #       You'll be left with missing values at the end
            self._validate_hot_deck_problematic_donate_missing(df=df)

            #   Donate vars shouldn't be in model
            #       Remove them and note it
            self._validate_hot_deck_remove_donates()

    def _validate_reserved_names(self):
        """
        Check for variable anmes now that would throw an error down the line to
            avoid wasting the time before it shows up

        Raises
        ------
        Exception
            Reserved variable name used.

        Returns
        -------
        None.

        """
        reserved_names = ["Imputed"]

        full_donate = [self.impute_var]

        if "donate_list" in self.parameters.keys():
            if self.parameters["donate_list"] is not None:
                full_donate.extend(self.parameters["donate_list"])

        if len(set(full_donate).intersection(reserved_names)):
            message = f"One of the variables to impute {full_donate} is a reserved variable in the SRMI implementation ({reserved_names})"
            logger.error(message)
            raise Exception(message)

    def _validate_lightgbm_boolean_quantile(self, df: IntoFrameT):
        """
        If impute_var is a dummy variable, can't run quantile gbm

        Raises
        ------
        Exception
            No boolean dependent in quantile regression.

        Returns
        -------
        None.

        """

        impute_type = (
            compress_df(df=nw.from_native(df).select(self.impute_var).to_native())
            .lazy()
            .collect_schema()[self.impute_var]
        )

        #   Check for the passed objective
        params_lgbm = self.parameters["parameters"]

        if type(params_lgbm) is dict:
            if "objective" in params_lgbm.keys():
                if impute_type == nw.Boolean and params_lgbm["objective"] == "quantile":
                    message = f"{self.impute_var} is boolean.  Cannot run quantile regression (objective=quantile) in LightGBM with a boolean dependent variable."
                    logger.error(message)
                    raise Exception(message)

    def _validate_hot_deck_remove_donates(self):
        """
        Donate vars shouldn't be in model
            Remove them and note it

        Returns
        -------
        None.

        """
        full_models = []
        for modi in self.parameters["model_list"]:
            full_models.extend(modi)
        full_models = list(set(full_models))
        full_donate = [self.impute_var]
        if self.parameters["donate_list"] is not None:
            full_donate.extend(self.parameters["donate_list"])

        donates_in_models = set(full_donate).intersection(full_models)

        if len(donates_in_models):
            logger.info(
                f"Dropping donated variables {donates_in_models} from hot deck models"
            )

            models_post = []
            for modi in self.parameters["model_list"]:
                models_post.append(
                    [itemi for itemi in modi if itemi not in donates_in_models]
                )
            logger.info(models_post)

            #   Remove any duplicates
            models_post_deduped = []
            [
                models_post_deduped.append(modi)
                for modi in models_post
                if modi not in models_post_deduped
            ]
            self.parameters["model_list"] = models_post_deduped

    def _validate_hot_deck_problematic_donate_missing(self, df: IntoFrameT):
        """
        If donate_vars don't have the same missingness pattern
            You'll be left with missing values at the end

        Parameters
        ----------
        df : IntoFrameT
            Impute dataframe.

        Raises
        ------
        Exception
            Checks for problematic missingness structure.

        Returns
        -------
        None.

        """
        #   If donate_vars don't have the same missingness pattern
        #       You'll be left with missing values at the end
        if self.parameters["donate_list"] is not None:
            additional_donates = [
                donatei
                for donatei in self.parameters["donate_list"]
                if donatei != self.impute_var
            ]

            if len(additional_donates):
                with_bad_donates = [
                    (
                        nw.col(donatei).is_missing()
                        & nw.col(self.impute_var).is_not_missing()
                    ).alias(f"bad_{donatei}")
                    for donatei in additional_donates
                ]

                df_bad = nw.from_native(df).select(with_bad_donates).to_native()
                if safe_height(
                    nw.from_native(df_bad).filter(nw.any_horizontal(df_bad.columns))
                ):
                    message = f"Cannot have missing values in donate_vars ({additional_donates} with non-missing value in '{self.impute_var}').  It will result in missings at the end of the imputation"
                    logger.error(message)
                    raise Exception(message)

        #   Donate vars shouldn't be in model
        #       Remove them and note it
        self._validate_hot_deck_remove_donates()

    def where_impute_add_flag(self, flag: str):
        # if self.b_where_strings:
        #     if self.Where_impute is None:
        #         self.Where_impute = f"({flag} == 1)"
        #     elif self.Where_impute != "":
        #         self.Where_impute = f"({self.Where_impute}) and ({flag} == 1)"
        #     else:
        #         self.Where_impute = f"({flag} == 1)"
        # else:
        if self.Where_impute is None:
            self.Where_impute = nw.col(flag)
        else:
            self.Where_impute = (self.Where_impute) & nw.col(flag)

    def df_where(self, df: IntoFrameT) -> IntoFrameT:
        return self._df_where_list(df, [self.Where])

    def df_predict_where(
        self, df: IntoFrameT, drop_imputed: bool = False
    ) -> IntoFrameT:
        if drop_imputed:
            where_list = [self.Where, self.Where_predict, self.Where_impute]
            negate_list = [False, False, True]
        else:
            where_list = [self.Where, self.Where_predict]
            negate_list = None

        df = self._df_where_list(df=df, where_list=where_list, negate_list=negate_list)

        if self.Where_predict_only_when_not_imputed and self.imputation_flag != "":
            df = nw.from_native(df).filter(~nw.col(self.imputation_flag)).to_native()
        return df

    def df_impute_where(self, df: IntoFrameT) -> IntoFrameT:
        return self._df_where_list(df, [self.Where, self.Where_impute])

    def df_impute_original_where(self, df: IntoFrameT) -> IntoFrameT:
        return self._df_where_list(df, [self.Where, self.Where_impute_original])

    def _df_where_list(
        self,
        df: IntoFrameT,
        where_list: list[str | None | nw.Expr],
        negate_list: list[bool] | None = None,
    ) -> IntoFrameT:
        # if self.b_where_strings:
        #     #   Each where is a string (or None)
        #     Where = ""

        #     where_index = 0
        #     for wherei in where_list:
        #         if wherei is not None:
        #             if wherei != "":
        #                 if Where != "":
        #                     Where += " and "

        #                 negate = False
        #                 if negate_list is not None:
        #                     negate = negate_list[where_index]

        #                 if negate:
        #                     Where += f"(not ({wherei}))"
        #                 else:
        #                     Where += f"({wherei})"

        #         where_index += 1

        #     if Where != "":
        #         df = SafeCollect(SqlWhereFilter(df=df,
        #                                         Where=Where))
        # else:
        #   Each where is a narwhals expressions (or None)
        nw_type = NarwhalsType(df)
        df = nw.from_native(df).lazy().to_native()

        where_index = 0
        for wherei in where_list:
            if wherei is not None:
                negate = False
                if negate_list is not None:
                    negate = negate_list[where_index]

                if negate:
                    df = nw.from_native(df).filter(~wherei).to_native()
                else:
                    df = nw.from_native(df).filter(wherei).to_native()

            where_index += 1

        return nw.from_native(df).lazy().collect().lazy_backend(nw_type).to_native()

    def split_when_missing(
        variable: Variable, exclude_for_missing: list[str]
    ) -> list[Variable]:
        # """
        # For an impute variable, split it into two impute stages
        #     where the first is when it is not missing and it can use the
        #     model as is and the second handles when it is missing by dropping
        #     any other variables that will be missing with this variable.
        #     An example of this is if ern_yn is imputed to True in the CPS ASEC,
        #     all the downstream earnings variables will be missing simultaneously
        #     and can't be used in the imputation (some may be derived)
        # Parameters
        # ----------
        # variable : Variable
        #     Variable information
        # exclude_for_missing : list[str]
        #     List of model variables to exclude when variable.impute_var is missing

        # Returns
        # -------
        # list[Variable]

        # """

        vars_out = []

        variable_not_missing = deepcopy(variable)

        if variable.b_where_strings:
            if not variable.Where_impute:
                variable.Where_impute = f"{variable.impute_var} is null"
                variable_not_missing.Where_impute = f"{variable.impute_var} is not null"
            else:
                variable.Where_impute = (
                    f"({variable.Where_impute}) and ({variable.impute_var} is null)"
                )
                variable_not_missing.Where_impute = f"({variable_not_missing.Where_impute}) and ({variable.impute_var} is not null)"
        else:
            if variable.Where_impute is None:
                variable.Where_impute = nw.col(variable.impute_var).is_missing()
                variable_not_missing.Where_impute = nw.col(
                    variable.impute_var
                ).is_not_missing()
            else:
                variable.Where_impute = (
                    variable.Where_impute & nw.col(variable.impute_var).is_missing()
                )
                variable_not_missing.Where_impute = (
                    variable_not_missing.Where_impute
                    & nw.col(variable.impute_var).is_not_missing()
                )

        variable.header += " with missing values"
        variable_not_missing.header += " with no missing values"
        variable.predictors_exclude.extend(exclude_for_missing)

        vars_out.append(variable_not_missing)
        vars_out.append(variable)

        return vars_out
