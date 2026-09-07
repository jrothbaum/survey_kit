from __future__ import annotations

import os
import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT
from enum import Enum
from copy import deepcopy

from ..utilities.formula_builder import FormulaBuilder
from ..utilities.dataframe import (
    columns_from_list,
    NarwhalsType,
    safe_height,
    lazy_backend,
)
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
    Construction groups related settings into sub-objects - see __init__
    for the full parameter list:

    impute_var, header, model, weight, modelfunction, modeltype, parameters,
    selection, preselection, By : flat, core settings
    sample : Variable.Sample - which rows this variable's imputation applies to
        (Where / Where_impute / Where_predict / Where_predict_only_when_not_imputed /
        bimpute_if_missing)
    hooks : Variable.Hooks - pre/post-imputation operations
        (pre / post / pre_initialize)
    predictors : Variable.Predictors - predictor inclusion/exclusion control
        (exclude / exclude_first_iteration / require / joint)

    For code still using the pre-refactor flat-kwarg signature
    (Where=..., preFunctions=..., predictors_exclude=..., etc.), use
    Variable.from_legacy(...) instead of Variable(...).
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

    class Sample(Serializable):
        """Which rows this variable's imputation applies to."""

        _save_suffix = "variable.sample"

        def __init__(
            self,
            Where: nw.Expr | None = None,
            Where_impute: nw.Expr | None = None,
            Where_predict: nw.Expr | None = None,
            Where_predict_only_when_not_imputed: bool = False,
            bimpute_if_missing: bool = True,
        ):
            """
            Parameters
            ----------
            Where : nw.Expr, optional
                Condition to restrict sample for this imputation. The default is None.
            Where_impute : nw.Expr, optional
                Define the set of observations to be imputed, in addition
                to bimpute_if_missing | Where. The default is None.
            Where_predict : nw.Expr, optional
                Define the set of observations for the prediction | Where. The default is None.
            Where_predict_only_when_not_imputed : bool, optional
                Predict only if not imputed.  The default is False.
            bimpute_if_missing : bool, optional
                Make an imputation condition that the variable is initially missing.
                The default is True.
            """
            self.Where = Where
            self.Where_impute_original = Where_impute
            self.Where_impute = Where_impute
            self.Where_predict = Where_predict
            self.Where_predict_only_when_not_imputed = (
                Where_predict_only_when_not_imputed
            )
            self.bimpute_if_missing = bimpute_if_missing

        def with_Where(self, value: nw.Expr | None) -> Variable.Sample:
            return self._with(Where=value)

        def with_Where_impute(self, value: nw.Expr | None) -> Variable.Sample:
            #   Where_impute_original tracks the pristine (pre where_impute_add_flag)
            #   value - setting a new Where_impute here is a fresh user-specified
            #   value, so both move together, matching __init__'s behavior.
            return self._with(Where_impute=value, Where_impute_original=value)

        def with_Where_predict(self, value: nw.Expr | None) -> Variable.Sample:
            return self._with(Where_predict=value)

        def with_Where_predict_only_when_not_imputed(self, value: bool) -> Variable.Sample:
            return self._with(Where_predict_only_when_not_imputed=value)

        def with_bimpute_if_missing(self, value: bool) -> Variable.Sample:
            return self._with(bimpute_if_missing=value)

    class Hooks(Serializable):
        """Operations to run before/after this variable's imputation each iteration."""

        _save_suffix = "variable.hooks"

        def __init__(
            self,
            pre: list[
                Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | nw.Expr
            ]
            | Variable.PrePost.Function
            | Variable.PrePost.NarwhalsExpression
            | nw.Expr
            | None = None,
            post: list[
                Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | nw.Expr
            ]
            | Variable.PrePost.Function
            | Variable.PrePost.NarwhalsExpression
            | nw.Expr
            | None = None,
            pre_initialize: list[
                Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | nw.Expr
            ]
            | Variable.PrePost.Function
            | Variable.PrePost.NarwhalsExpression
            | nw.Expr
            | None = None,
        ):
            """
            Parameters
            ----------
            pre : list, optional
                Any operations to run before this imputation at each iteration.
                The default is None.
            post : list, optional
                Any operations to run after this imputation at each iteration.
                The default is None.
            pre_initialize : list, optional
                Any operations to run before this imputation ONLY ONCE before running
                the first implicate. If it's a function, it will expect the implicate
                to be passed in. The default is None.
            """
            self.pre = Variable._parse_pre_post_function_inputs(pre)
            self.post = Variable._parse_pre_post_function_inputs(post)
            self.pre_initialize = Variable._parse_pre_post_function_inputs(
                pre_initialize
            )

        def with_pre(self, value) -> Variable.Hooks:
            return self._with(pre=Variable._parse_pre_post_function_inputs(value))

        def with_post(self, value) -> Variable.Hooks:
            return self._with(post=Variable._parse_pre_post_function_inputs(value))

        def with_pre_initialize(self, value) -> Variable.Hooks:
            return self._with(
                pre_initialize=Variable._parse_pre_post_function_inputs(value)
            )

    class Predictors(Serializable):
        """Which predictor variables enter (or are forced into/out of) this variable's model."""

        _save_suffix = "variable.predictors"

        def __init__(
            self,
            exclude: list = None,
            exclude_first_iteration: list = None,
            require: list = None,
            joint: dict = None,
        ):
            """
            Parameters
            ----------
            exclude : list, optional
                What to exclude from the model. The default is None.
            exclude_first_iteration : list, optional
                The first iteration, it will exclude downstream variables
                by default, use this to override the default. The default is None.
            require : list, optional
                What to include no matter what. The default is None.
            joint : dict, optional
                dictionary of key (variable name) value lists/pairs where if the
                key is selected for the model, then so must the values. The default is None.
            """
            if exclude is None:
                exclude = []
            self.exclude = exclude

            if exclude_first_iteration is None:
                exclude_first_iteration = []
            self.exclude_first_iteration = exclude_first_iteration

            self.require = require
            self.joint = joint

        def with_exclude(self, value: list | None) -> Variable.Predictors:
            return self._with(exclude=value if value is not None else [])

        def with_exclude_first_iteration(self, value: list | None) -> Variable.Predictors:
            return self._with(exclude_first_iteration=value if value is not None else [])

        def with_require(self, value: list | None) -> Variable.Predictors:
            return self._with(require=value)

        def with_joint(self, value: dict | None) -> Variable.Predictors:
            return self._with(joint=value)

    def __init__(
        self,
        impute_var: str = "",
        header: str = "",
        model: str = "",
        weight: str = "",
        modelfunction=None,
        modeltype: ModelType = None,
        parameters: dict = None,
        selection: Selection = None,
        preselection: Selection = None,
        By: list = None,
        sample: Variable.Sample = None,
        hooks: Variable.Hooks = None,
        predictors: Variable.Predictors = None,
    ):
        """
        Parameters
        ----------
        impute_var : str
            Variable to be imputed
        header : str, optional
            Just a header to write to the log when this variable comes up. The default is "".
        model : str, optional
            R string formula, Override the "parent" SRMI model?. The default is "" (no).
        weight : str, optional
            Weight for the imputation modeling. The default is "".
        modelfunction : function delegate, optional
            Override modeltype completely and just run a custom imputation function, optional
            The function arguments are
                df:IntoFrameT - a dataframe with the full srmi data
                variable:Variable - an SRMI.Variable object ,
                index:list - the merge key of the data,
                weight:str - weight variable?
                sub_log:logging - to write the imputation output to a separate file in the
                    implicate folder
        modeltype : ModelType, optional
            Override the "parent" SRMI modeltype?. The default is "" (no).
        parameters : dict, optional
            Override the "parent" SRMI model parameters?. The default is "" (no).
        selection : Selection, optional
            Override the "parent" SRMI selection used?
            If variable selection is used within the imputation, this class
                handles it.  The default is no selection
        preselection : Selection, optional
            Override the "parent" SRMI selection used?
            If variable selection is done before the SRMI starts
                to pre-prune the inputs, this class handles it.
                The default is no selection
        By : list, optional
            Variable list for by groups
        sample : Variable.Sample, optional
            Which rows this variable's imputation applies to. The default is Variable.Sample().
        hooks : Variable.Hooks, optional
            Pre/post-imputation operations. The default is Variable.Hooks().
        predictors : Variable.Predictors, optional
            Predictor inclusion/exclusion control. The default is Variable.Predictors().

        Returns
        -------
        None.

        """

        self.impute_var = impute_var
        self.header = header

        self.sample = sample if sample is not None else Variable.Sample()
        self.hooks = hooks if hooks is not None else Variable.Hooks()
        self.predictors = predictors if predictors is not None else Variable.Predictors()

        self.weight = weight
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

    @classmethod
    def from_legacy(
        cls,
        impute_var: str = "",
        Where: nw.Expr | None = None,
        Where_impute: nw.Expr | None = None,
        Where_predict: nw.Expr | None = None,
        Where_predict_only_when_not_imputed: bool = False,
        bimpute_if_missing: bool = True,
        preFunctions=None,
        postFunctions=None,
        preFunctions_initialize_implicate=None,
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
    ) -> Variable:
        """
        Construct a Variable from the pre-refactor flat-kwarg signature.

        Migration aid only - new code should pass sample=Variable.Sample(...),
        hooks=Variable.Hooks(...), predictors=Variable.Predictors(...) directly
        to Variable() instead.
        """
        return cls(
            impute_var=impute_var,
            header=header,
            model=model,
            weight=weight,
            modelfunction=modelfunction,
            modeltype=modeltype,
            parameters=parameters,
            selection=selection,
            preselection=preselection,
            By=By,
            sample=Variable.Sample(
                Where=Where,
                Where_impute=Where_impute,
                Where_predict=Where_predict,
                Where_predict_only_when_not_imputed=Where_predict_only_when_not_imputed,
                bimpute_if_missing=bimpute_if_missing,
            ),
            hooks=Variable.Hooks(
                pre=preFunctions,
                post=postFunctions,
                pre_initialize=preFunctions_initialize_implicate,
            ),
            predictors=Variable.Predictors(
                exclude=predictors_exclude,
                exclude_first_iteration=predictors_exclude_first_iteration,
                require=predictors_require,
                joint=joint,
            ),
        )

    #####################################################
    #   Flat-attribute compatibility properties - BEGIN
    #       Where/preFunctions/predictors_exclude/etc. are read (and in some
    #       cases mutated) throughout this file plus implicate.py and impute.py.
    #       These properties redirect that existing behavior onto the new
    #       self.sample/self.hooks/self.predictors objects so none of those
    #       call sites needed to change - only construction did.
    #####################################################
    @property
    def Where(self):
        return self.sample.Where

    @Where.setter
    def Where(self, value):
        self.sample.Where = value

    @property
    def Where_impute(self):
        return self.sample.Where_impute

    @Where_impute.setter
    def Where_impute(self, value):
        self.sample.Where_impute = value

    @property
    def Where_impute_original(self):
        return self.sample.Where_impute_original

    @Where_impute_original.setter
    def Where_impute_original(self, value):
        self.sample.Where_impute_original = value

    @property
    def Where_predict(self):
        return self.sample.Where_predict

    @Where_predict.setter
    def Where_predict(self, value):
        self.sample.Where_predict = value

    @property
    def Where_predict_only_when_not_imputed(self):
        return self.sample.Where_predict_only_when_not_imputed

    @Where_predict_only_when_not_imputed.setter
    def Where_predict_only_when_not_imputed(self, value):
        self.sample.Where_predict_only_when_not_imputed = value

    @property
    def bimpute_if_missing(self):
        return self.sample.bimpute_if_missing

    @bimpute_if_missing.setter
    def bimpute_if_missing(self, value):
        self.sample.bimpute_if_missing = value

    @property
    def preFunctions(self):
        return self.hooks.pre

    @preFunctions.setter
    def preFunctions(self, value):
        self.hooks.pre = value

    @property
    def postFunctions(self):
        return self.hooks.post

    @postFunctions.setter
    def postFunctions(self, value):
        self.hooks.post = value

    @property
    def preFunctions_initialize_implicate(self):
        return self.hooks.pre_initialize

    @preFunctions_initialize_implicate.setter
    def preFunctions_initialize_implicate(self, value):
        self.hooks.pre_initialize = value

    @property
    def predictors_exclude(self):
        return self.predictors.exclude

    @predictors_exclude.setter
    def predictors_exclude(self, value):
        self.predictors.exclude = value

    @property
    def predictors_exclude_first_iteration(self):
        return self.predictors.exclude_first_iteration

    @predictors_exclude_first_iteration.setter
    def predictors_exclude_first_iteration(self, value):
        self.predictors.exclude_first_iteration = value

    @property
    def predictors_require(self):
        return self.predictors.require

    @predictors_require.setter
    def predictors_require(self, value):
        self.predictors.require = value

    @property
    def joint(self):
        return self.predictors.joint

    @joint.setter
    def joint(self, value):
        self.predictors.joint = value

    #####################################################
    #   Flat-attribute compatibility properties - END
    #####################################################

    @staticmethod
    def _parse_pre_post_function_inputs(
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
                        nw.col(donatei).is_null() & ~nw.col(self.impute_var).is_null()
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

        return lazy_backend(nw.from_native(df).lazy().collect(), nw_type).to_native()
