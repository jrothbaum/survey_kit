from __future__ import annotations

import polars as pl
import narwhals as nw
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
from .parameters import Parameters

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
    transforms : Variable.Transforms - pre/post-imputation operations
        (pre / post / pre_initialize / post_finalize)
    predictors : Variable.Predictors - predictor inclusion/exclusion control
        (exclude / exclude_first_iteration / require / joint)
    """

    class ModelType(Enum):
        #   Predicted mean matching - Parameters.pmm()'s own fixed
        #       model/error choice. Routes through the same
        #       Impute.regression() every other regression-shaped
        #       modeltype below does (no separate impute.py method of
        #       its own).
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

        #   NearestNeighbor has no modeltype of its own - matching
        #       directly on raw x values (no fitted model) is a
        #       restricted, worse-behaved case of Regression, so
        #       Parameters.NearestNeighbor() builds a Regression parameter
        #       dict instead: fit an OLS on the same predictors and
        #       PMM-match on that prediction rather than raw distance -
        #       see that function's docstring for why that's the better
        #       default.

        #   Predict y with a mean-regression sklearn-compatible estimator
        #       (RandomForestRegressor/XGBRegressor/CatBoostRegressor/your
        #       own), then impute the same way Regression does - see
        #       Parameters.RandomForest()/XGBoost()/CatBoost()/SklearnModel()
        RandomForest = 10
        XGBoost = 11
        CatBoost = 12
        SklearnModel = 13

        #   Predict an unordered categorical y (3+ levels) with a
        #       RandomForestClassifier, then impute by donor matching on
        #       leaf co-occurrence: pool the donors sharing a leaf with
        #       the recipient across every tree, draw one uniformly at
        #       random - the same donor-selection mechanism mice's rf
        #       method uses (see
        #       imputation/utilities/leaf_donor_matching.py). Unlike
        #       RandomForest/XGBoost/CatBoost/SklearnModel above, this is
        #       classification, not mean regression - it doesn't reuse
        #       _run_regression at all. See Parameters.Multinomial().
        Multinomial = 14

        #   Predict an ORDERED categorical y (e.g. education level, a
        #       Likert scale) by fitting a mean-regression estimator
        #       (default RandomForestRegressor, or your own factory)
        #       against an integer rank encoding of the declared
        #       category order, then donating the REAL observed category
        #       from a matched donor - never the numeric rank, and never
        #       a category that wasn't actually observed (same guarantee
        #       Multinomial() gives). Donor matching is PMM (knearest on
        #       the predicted rank) or leaf (tree leaf co-occurrence -
        #       see ErrorDraw.leaf and utilities/leaf_donor_matching.py).
        #       See Parameters.OrderedCategorical().
        OrderedCategorical = 15

    class Class(Enum):
        """
        A variable's statistical TYPE (separate from ModelType, which
        picks - a specific fitting algorithm for that type). Used by
        SRMI.simple_model()/utilities/auto_detect.py to pick a sensible
        default ModelType per variable: binary/continuous can be told
        apart automatically (dtype/0-1-only check); ordered_categorical
        and unordered_categorical can never be inferred from the data
        alone (there's no way to know intended category order, or that
        a numeric-looking column is really a category code, without
        being told) - a variable of either categorical kind always
        needs an explicit Class declaration.
        """

        binary = 1
        continuous = 2
        ordered_categorical = 3
        unordered_categorical = 4

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
        class PolarsExpression(Serializable):
            def __init__(self, expression: list[pl.Expr] | pl.Expr):
                """
                Pass the call information to call before or after an imputation step

                Parameters
                ----------
                expression:list[pl.Expr] | pl.Expr
                    A polars with_columns expression or list of expressions

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

                if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
                    return df.with_columns(self.expression)
                else:
                    nw_type = NarwhalsType(df)
                    df = (
                        nw_type.to_polars()
                        .with_columns(self.expression)
                        .lazy()
                        .collect()
                    )
                    return nw_type.from_polars(df)
                

                return df.with_columns(self.expression)

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

    class Transforms(Serializable):
        """
        Operations to run before/after this variable's imputation each
        iteration (pre/post), plus once-only bookends around the whole
        implicate (pre_initialize before the first iteration,
        post_finalize after the last).
        """

        _save_suffix = "variable.transforms"

        def __init__(
            self,
            pre: list[
                Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | Variable.PrePost.PolarsExpression | nw.Expr | pl.Expr
            ]
            | Variable.PrePost.Function
            | Variable.PrePost.NarwhalsExpression
            | nw.Expr
            | None = None,
            post: list[
                Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | Variable.PrePost.PolarsExpression | nw.Expr | pl.Expr
            ]
            | Variable.PrePost.Function
            | Variable.PrePost.NarwhalsExpression
            | nw.Expr
            | None = None,
            pre_initialize: list[
                Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | Variable.PrePost.PolarsExpression | nw.Expr | pl.Expr
            ]
            | Variable.PrePost.Function
            | Variable.PrePost.NarwhalsExpression
            | nw.Expr
            | None = None,
            post_finalize: list[
                Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | Variable.PrePost.PolarsExpression | nw.Expr | pl.Expr
            ]
            | Variable.PrePost.Function
            | Variable.PrePost.NarwhalsExpression
            | Variable.PrePost.PolarsExpression
            | nw.Expr
            | pl.Expr
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
            post_finalize : list, optional
                Any operations to run ONLY ONCE, after the implicate's LAST
                iteration completes (the mirror image of pre_initialize -
                once at the end instead of once at the start). Unlike
                pre_initialize, these use the same calling convention as
                pre/post: a function receives/returns df, not the
                implicate. Runs once per implicate (not once across the
                whole SRMI run), in variable order, so it's a deterministic
                place to do one-time cleanup a variable's own imputation
                needed but that shouldn't be repeated every iteration or
                show up in the final output - e.g. dropping a scaffolding
                column another variable's pre/pre_initialize created only
                to drive that variable's own donor matching. Not called
                again on a resumed run that finds the implicate already
                complete. The default is None.
            """
            self.pre = Variable._parse_pre_post_function_inputs(pre)
            self.post = Variable._parse_pre_post_function_inputs(post)
            self.post_finalize = Variable._parse_pre_post_function_inputs(post_finalize)
            self.pre_initialize = Variable._parse_pre_post_function_inputs(
                pre_initialize
            )

        def with_pre(self, value) -> Variable.Transforms:
            return self._with(pre=Variable._parse_pre_post_function_inputs(value))

        def with_post(self, value) -> Variable.Transforms:
            return self._with(post=Variable._parse_pre_post_function_inputs(value))

        def with_pre_initialize(self, value) -> Variable.Transforms:
            return self._with(
                pre_initialize=Variable._parse_pre_post_function_inputs(value)
            )

        def with_post_finalize(self, value) -> Variable.Transforms:
            return self._with(
                post_finalize=Variable._parse_pre_post_function_inputs(value)
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
        transforms: Variable.Transforms = None,
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
        transforms : Variable.Transforms, optional
            Pre/post-imputation operations. The default is Variable.Transforms().
        predictors : Variable.Predictors, optional
            Predictor inclusion/exclusion control. The default is Variable.Predictors().

        Returns
        -------
        None.

        """

        self.impute_var = impute_var
        self.header = header

        self.sample = sample if sample is not None else Variable.Sample()
        self.transforms = transforms if transforms is not None else Variable.Transforms()
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
        Construct a Variable from a flat set of keyword arguments, rather
        than the grouped sample=Variable.Sample(...)/
        transforms=Variable.Transforms(...)/
        predictors=Variable.Predictors(...) form Variable() itself takes.
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
            transforms=Variable.Transforms(
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

    @classmethod
    def two_part(
        cls,
        df: IntoFrameT,
        impute_var: str,
        model: list[str] | str,
        modeltype: Variable.ModelType = ModelType.Regression,
        parameters: dict | None = None,
        yn_model: list[str] | str | None = None,
        yn_modeltype: Variable.ModelType | None = None,
        yn_parameters: dict | None = None,
        yn_var: str | None = None,
        yn_missing: nw.Expr | None = None,
        value_if_no: float | None = 0,
        weight: str = "",
        By: list[str] | str | None = None,
    ) -> tuple[IntoFrameT, list[Variable]]:
        """
        Shortcut for semicontinuous (point mass at zero + continuous)
        two-part imputation: a binary y/n ("is impute_var nonzero")
        variable, imputed first, then impute_var itself, restricted to
        the y/n==True population - the standard two-part/hurdle approach
        (as opposed to just PMM/leaf-matching the raw variable directly,
        which reproduces the point mass for free but assumes a single
        model/predictor set explains both the participation and
        intensity margins - see the two_part design discussion this
        wraps up).

        Takes df (needed to derive the y/n column - see below) and
        returns (df, variables): df with the y/n column added, and a
        plain list, always safe to `variables.extend(...)` or loop over
        - normally `[yn_variable, value_variable]` (order matters: yn
        must precede value in your SRMI variables list, so value sees
        this iteration's fresh yn draw, not last iteration's), but
        sometimes shorter - see yn_var and the HotDeck/StatMatch note
        below. Use the RETURNED df (not your original) to construct
        SRMI - SRMI.__init__ needs every impute_var to already exist as
        a real column up front, even the y/n one, so it can't be created
        lazily via a hook the way the per-iteration consistency fixups
        below are.

        Mechanics (mostly via existing Variable.Transforms hooks, no
        per-call special-casing in impute.py):
          - If creating y/n (yn_var not given): the column is derived
            once, right here, straight into the returned df (not via
            pre_initialize - the derivation is a fixed function of df's
            own observed values, identical for every implicate, so
            there's nothing to recompute per-implicate): null wherever
            impute_var is null (or, if yn_missing is given, ALSO
            wherever that expression is true - additive, never
            narrower, since a null impute_var can never tell us whether
            y/n was really True or False), else impute_var != 0.
          - If yn_var IS given but still has missing values of its own:
            value_variable's Where (below) would otherwise silently
            exclude those rows from ever being touched at all (a null
            never satisfies a boolean filter) - so a real yn_variable
            still gets built for it, using the same modeltype/
            parameters derivation as the create-our-own case, just
            reading/imputing the caller's own column in place rather
            than deriving it, and never dropping it at the end (not
            ours to clean up). If yn_var has no missing values at all,
            none of this applies - value_variable alone is returned.
          - Whenever a real yn_variable is built (either case above)
            and its own donation is genuine (pmm/leaf, or HotDeck/
            StatMatch - always donation-based), impute_var itself rides
            along in its donate_list - so a row whose y/n was just
            resolved gets its value from that SAME matched donor,
            rather than from a possibly different one value_variable's
            own later pass would find. A harmless no-op when yn's error
            draw is Random instead, which never donates anything.
          - value_variable.sample.Where restricts value's donor pool AND
            recipients to yn==True rows.
          - value_variable.transforms.pre runs every iteration, before
            value's own fit/donation: wherever yn is currently True but
            value == 0 (stale from a prior iteration when yn was False),
            null it out (so it's a genuine recipient again this
            iteration); wherever yn is currently False, force value to
            value_if_no.
          - yn_variable.transforms.post_finalize drops the scratch y/n
            column once, after the implicate's last iteration - only
            when this function created it (see yn_var).

        Parameters
        ----------
        df : IntoFrameT
            Source data - read to derive the y/n column, and to return
            the augmented copy you should actually build SRMI from.
        impute_var : str
            The semicontinuous target.
        model : list[str] | str
            Predictors for impute_var (value). Also yn's predictors,
            unless yn_model overrides them.
        modeltype : Variable.ModelType, optional
            value's modeltype. Supported: Regression, pmm, LightGBM
            (fit a real Logit/binary-objective variant for yn);
            RandomForest, XGBoost, CatBoost, SklearnModel (yn reuses the
            SAME estimator factory via ModelType.OrderedCategorical
            with categories=[False, True] - no regressor/classifier
            estimator swap, which isn't generically possible: sklearn/
            XGBoost/CatBoost each have model-family-specific
            hyperparameters, like criterion/objective/loss_function,
            that don't transfer across that boundary, and there's no
            way to do it at all for SklearnModel's arbitrary factory);
            HotDeck, StatMatch (donation doesn't care about target
            dtype, no swap needed - see the collapse behavior below).
            Anything else (Multinomial, OrderedCategorical) raises -
            not semicontinuous-shaped as value's own type. By default
            Variable.ModelType.Regression.
        parameters : dict, optional
            value's parameters (e.g. Parameters.RandomForest(...)). By
            default None ({}).
        yn_model : list[str] | str | None, optional
            Predictors for yn, if they should differ from model - the
            participation and intensity margins often don't share
            predictors/mechanism. By default None (same as model).
        yn_modeltype : Variable.ModelType | None, optional
            Explicit override for yn's modeltype - pass together with
            yn_parameters for full manual control. By default None
            (derived from modeltype per the modeltype docstring above).
        yn_parameters : dict | None, optional
            Explicit override for yn's parameters. If given without
            yn_modeltype, yn_modeltype defaults to modeltype (or
            OrderedCategorical, if modeltype is one of the four that
            reuse it) rather than being derived further - if you're
            supplying parameters yourself, supply the modeltype too if
            it's not that default. By default None (derived).
        yn_var : str | None, optional
            Reuse an existing y/n column instead of deriving one from
            impute_var. If it's already fully observed, this function
            builds ONLY value_variable (a 1-item list), restricted to
            yn_var==True. If it still has missing values of its own,
            this function ALSO builds a yn_variable to resolve them
            (using the same modeltype/yn_modeltype/yn_parameters
            derivation as the create-our-own case - see the mechanics
            note above), so the returned list is still
            [yn_variable, value_variable] in that case - it's only
            "your own concern" when there's genuinely nothing left for
            it to do. Never dropped at the end either way - it's your
            column. By default None (create one, named
            f"___{impute_var}_yn___").
        yn_missing : nw.Expr | None, optional
            Only meaningful when yn_var is not given. Rows where this is
            true are ALSO treated as yn-missing, on top of
            impute_var.is_null() (additive, not a replacement - see the
            mechanics note above for why). By default None.
        value_if_no : float | None, optional
            What value becomes on yn==False rows, every iteration. By
            default 0 (pass None to leave it null instead).
        weight : str, optional
            Applied identically to both variables. By default "".
        By : list[str] | str | None, optional
            Applied identically to both variables. By default None.

        Returns
        -------
        tuple[IntoFrameT, list[Variable]]
            (df, variables) - df augmented with the y/n column (only
            when this function created one - unchanged otherwise);
            variables is [yn_variable, value_variable] normally,
            [value_variable] alone only when either (a) yn_var was given
            and is already fully observed (nothing left for a
            yn_variable to do), or (b) modeltype is HotDeck/StatMatch
            and there's no signal at all (no yn_var, yn_missing,
            yn_model, yn_modeltype, or yn_parameters) that the two
            margins should be modeled differently - a single donation
            pass on value, unrestricted, already reproduces the point
            mass for free in that case, so a redundant yn model is
            skipped (with a warning).
        """
        direct_swap = (
            Variable.ModelType.Regression,
            Variable.ModelType.pmm,
            Variable.ModelType.LightGBM,
        )
        ordered_categorical_reuse = (
            Variable.ModelType.RandomForest,
            Variable.ModelType.XGBoost,
            Variable.ModelType.CatBoost,
            Variable.ModelType.SklearnModel,
        )
        trivial_reuse = (Variable.ModelType.HotDeck, Variable.ModelType.StatMatch)
        supported = direct_swap + ordered_categorical_reuse + trivial_reuse

        if modeltype not in supported:
            message = (
                f"Variable.two_part(): modeltype={modeltype} isn't a "
                f"semicontinuous-shaped mean-regression/donation "
                f"modeltype - supported: {[m.name for m in supported]}."
            )
            logger.error(message)
            raise ValueError(message)

        if parameters is None:
            parameters = {}

        no_yn_signal = (
            yn_var is None
            and yn_missing is None
            and yn_model is None
            and yn_modeltype is None
            and yn_parameters is None
        )

        if modeltype in trivial_reuse and no_yn_signal:
            message = (
                f"{impute_var}: two_part() with modeltype={modeltype.name} "
                f"and no yn_var/yn_missing/yn_model/yn_modeltype/"
                f"yn_parameters given - there's no signal that the y/n and "
                f"value margins should be modeled differently, and "
                f"donation-based matching (HotDeck/StatMatch) doesn't care "
                f"about the target's dtype either way, so a separate y/n "
                f"model would be redundant here. Imputing value directly, "
                f"unrestricted, instead."
            )
            logger.warning(message)
            return (
                df,
                [
                    Variable(
                        impute_var=impute_var,
                        model=model,
                        weight=weight,
                        By=By,
                        modeltype=modeltype,
                        parameters=parameters,
                    )
                ],
            )

        #   yn_col/creating: yn_var given means the CALLER'S OWN column -
        #       we never overwrite its values with the derived formula,
        #       and never drop it. But it might still have missing
        #       values of its own (not yet resolved) - Where=nw.col(...)
        #       below excludes a null row entirely (from both value's
        #       donor pool AND recipients), so if we did nothing further
        #       those rows would silently never get EITHER yn or value
        #       filled in. So: if yn_var has its own missingness, we
        #       still need a real yn_variable (using yn_modeltype/
        #       yn_parameters, same derivation as the "create our own"
        #       case below) to resolve it - just without the derive-it-
        #       from-value formula and without post_finalize (it's not
        #       ours to clean up). If yn_var is already fully observed,
        #       none of that is needed - value_variable alone suffices,
        #       same as before.
        creating = yn_var is None
        yn_col = yn_var if yn_var is not None else f"___{impute_var}_yn___"
        yn_predictors = yn_model if yn_model is not None else model

        if not creating:
            yn_still_missing = (
                nw.from_native(df)
                .lazy()
                .select(nw.col(yn_col).is_null().sum())
                .collect()
                .item(0, 0)
                > 0
            )
        else:
            yn_still_missing = True

        value_variable = Variable(
            impute_var=impute_var,
            model=model,
            weight=weight,
            By=By,
            modeltype=modeltype,
            parameters=parameters,
            sample=Variable.Sample(Where=nw.col(yn_col)),
            transforms=Variable.Transforms(
                pre=Variable.PrePost.Function(
                    _two_part_value_consistency,
                    parameters={
                        "yn_var": yn_col,
                        "value_var": impute_var,
                        "value_if_no": value_if_no,
                    },
                )
            ),
        )

        if not yn_still_missing:
            #   yn_var was given and is already fully observed - nothing
            #       for a yn_variable to do, df is unchanged.
            return (df, [value_variable])

        if yn_parameters is not None:
            if yn_modeltype is None:
                yn_modeltype = (
                    Variable.ModelType.OrderedCategorical
                    if modeltype in ordered_categorical_reuse
                    else modeltype
                )
        elif yn_modeltype is not None:
            message = (
                f"{impute_var}: two_part() got yn_modeltype without "
                f"yn_parameters - can't auto-derive parameters for a "
                f"modeltype different from value's own; pass "
                f"yn_parameters too."
            )
            logger.error(message)
            raise ValueError(message)
        elif modeltype in direct_swap:
            yn_modeltype = modeltype
            yn_parameters = deepcopy(parameters)
            #   donate_list means "when you find a donor for THIS
            #       target, also carry these other variables from that
            #       same donor" - copied verbatim from value's own
            #       parameters, it would make yn's own (independent,
            #       generally different-donor) match ALSO donate those
            #       variables, only to have value's own pass silently
            #       overwrite that donation right after (value runs
            #       second) - wasted work at best, a confusing
            #       intermediate state at worst. donate_by (a shared
            #       grouping/restriction, not something donated) is
            #       fine to share and stays untouched. impute_var
            #       itself, though, DOES belong in yn's donate_list (see
            #       this function's own docstring on why) - when yn's
            #       own donation mechanism is real (pmm/leaf), this
            #       pulls value from the SAME matched donor as yn,
            #       right when yn itself is resolved, rather than value
            #       getting a possibly-different donor later from its
            #       own separate match; a no-op, harmlessly ignored, if
            #       yn ends up drawing via error=Random instead (which
            #       never donates anything at all).
            yn_parameters["donate_list"] = [impute_var]
            if modeltype == Variable.ModelType.LightGBM:
                yn_parameters["parameters"] = deepcopy(
                    yn_parameters.get("parameters") or {}
                )
                yn_parameters["parameters"]["objective"] = "binary"
            else:
                #   Regression/pmm both carry a plain "model" key
                #       (RegressionModel enum) regardless of which of
                #       the two built the dict.
                yn_parameters["model"] = Parameters.RegressionModel.Logit
        elif modeltype in ordered_categorical_reuse:
            yn_modeltype = Variable.ModelType.OrderedCategorical
            #   Same estimator factory, unmodified - no regressor/
            #       classifier swap (see this function's own docstring
            #       for why that's not generically possible). error only
            #       carries over if leaf - OrderedCategorical has no
            #       Random branch, and leaf is the only donation
            #       mechanism unaffected by ___prediction being a fitted
            #       rank rather than value's own scale. donate_list is
            #       [impute_var], not copied from value's own - see the
            #       direct_swap branch's comment above for why.
            value_error = parameters.get("error")
            yn_parameters = Parameters.OrderedCategorical(
                categories=[False, True],
                estimator=parameters.get("estimator"),
                categorical_feature=parameters.get("categorical_feature"),
                estimator_prepare_data=parameters.get("estimator_prepare_data"),
                donate_list=[impute_var],
                donate_by=parameters.get("donate_by"),
                knearest=parameters.get("knearest", 10),
                error=(
                    value_error
                    if value_error == Parameters.ErrorDraw.leaf
                    else Parameters.ErrorDraw.pmm
                ),
            )
        else:
            #   trivial_reuse (HotDeck/StatMatch), and at least one
            #       yn_* signal was given, or yn_var's own missingness
            #       needs resolving - donation doesn't care about
            #       target dtype, so parameters copy over unchanged
            #       (EXCEPT donate_list, which becomes [impute_var] -
            #       see the direct_swap branch's comment above for why)
            #       UNLESS yn_model demands a different matching-cell
            #       spec, which lives inside parameters["model_list"]
            #       for these two, not Variable.model - rebuild it via
            #       Parameters.HotDeck() rather than trying to hand-edit
            #       the built dict. (Parameters.StatMatch() is just
            #       Parameters.HotDeck() under its own name, so this is
            #       correct either way.) HotDeck/StatMatch always
            #       genuinely donate (no error=Random escape hatch), so
            #       this is never a no-op here the way it can be for
            #       direct_swap.
            yn_modeltype = modeltype
            if yn_model is not None:
                yn_parameters = Parameters.HotDeck(
                    model_list=yn_model, donate_list=[impute_var]
                )
            else:
                yn_parameters = deepcopy(parameters)
                yn_parameters["donate_list"] = [impute_var]

        if creating:
            missing_expr = nw.col(impute_var).is_null()
            if yn_missing is not None:
                missing_expr = missing_expr | yn_missing

            yn_expr = (
                nw.when(missing_expr)
                .then(None)
                .otherwise(nw.col(impute_var) != 0)
                .alias(yn_col)
            )
            df = nw.from_native(df).with_columns(yn_expr).to_native()

        yn_transforms = (
            Variable.Transforms(
                post_finalize=Variable.PrePost.Function(
                    _two_part_drop_yn, parameters={"yn_var": yn_col}
                ),
            )
            if creating
            else None
        )

        yn_variable = Variable(
            impute_var=yn_col,
            model=yn_predictors,
            weight=weight,
            By=By,
            modeltype=yn_modeltype,
            parameters=yn_parameters,
            transforms=yn_transforms,
        )

        return (df, [yn_variable, value_variable])

    #####################################################
    #   Flat-attribute compatibility properties - BEGIN
    #       Where/preFunctions/predictors_exclude/etc. are read (and in some
    #       cases mutated) throughout this file plus implicate.py and impute.py.
    #       These properties redirect that existing behavior onto the new
    #       self.sample/self.transforms/self.predictors objects so none of those
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
        return self.transforms.pre

    @preFunctions.setter
    def preFunctions(self, value):
        self.transforms.pre = value

    @property
    def postFunctions(self):
        return self.transforms.post

    @postFunctions.setter
    def postFunctions(self, value):
        self.transforms.post = value

    @property
    def preFunctions_initialize_implicate(self):
        return self.transforms.pre_initialize

    @preFunctions_initialize_implicate.setter
    def preFunctions_initialize_implicate(self, value):
        self.transforms.pre_initialize = value

    @property
    def postFunctions_finalize_implicate(self):
        return self.transforms.post_finalize

    @postFunctions_finalize_implicate.setter
    def postFunctions_finalize_implicate(self, value):
        self.transforms.post_finalize = value

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
            | Variable.PrePost.PolarsExpression
            | nw.Expr
            | pl.Expr
            | list[nw.Expr]
        ]
        | Variable.PrePost.Function
        | Variable.PrePost.NarwhalsExpression
        | Variable.PrePost.PolarsExpression
        | nw.Expr
        | pl.Expr
        | None = None,
    ) -> list[Variable.PrePost.Function | Variable.PrePost.NarwhalsExpression | Variable.PrePost.PolarsExpression]:
        if functions is None:
            functions = []
        if type(functions) is not list:
            functions = [functions]

        final_functions = []
        for fi in functions:
            if type(fi) is nw.Expr:
                final_functions.append(Variable.PrePost.NarwhalsExpression(fi))
            elif type(fi) is pl.Expr:
                final_functions.append(Variable.PrePost.PolarsExpression(fi))
            elif type(fi) is list:
                for fi_sub in fi:
                    if type(fi_sub) is nw.Expr:
                        final_functions.append(Variable.PrePost.NarwhalsExpression(fi_sub))
                    elif type(fi_sub) is pl.Expr:
                        final_functions.append(Variable.PrePost.PolarsExpression(fi_sub))
                    else:
                        final_functions.append(fi_sub)
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
                        #   Order-preserving difference (matches this
                        #   model_list entry's own predictor order)
                        #   rather than list(set(...)), which would
                        #   reorder based on Python's per-process
                        #   string hash randomization - this feeds the
                        #   model's own predictor column order.
                        item = [
                            v
                            for v in self.parameters["model_list"][modeli]
                            if v not in exclude_list
                        ]

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

    def validate_inputs(self, df: IntoFrameT, bootstrap_enabled: bool = True):
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

        #   A variable can't be both forced into the model and excluded from it
        self._validate_require_exclude_overlap()

        #   Fail before the run starts (not partway through, after some
        #       variables/iterations already succeeded) if an estimator
        #       preset's package isn't installed.
        self._validate_estimator_available(df=df)

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

        #   HotDeck/StatMatch donor selection is entirely unweighted -
        #       unlike pmm/Regression()/RandomForest()/XGBoost()/
        #       CatBoost()/SklearnModel()/LightGBM()/Multinomial, neither
        #       reads the weight column at all (checked: zero weight-
        #       related references anywhere in statmatch()/hotdeck() or
        #       their helpers), so both a declared survey weight AND the
        #       Bayesian bootstrap's per-implicate weight perturbation -
        #       one of the main sources of proper between-implicate MI
        #       variance - are silently ignored for these two. Warn
        #       once here (not every iteration/implicate, which is where
        #       this would otherwise actually fire) rather than fixing
        #       it: efficient weighted sampling isn't trivial for
        #       StatMatch, and HotDeck's sequential donor-carry design has
        #       no natural notion of weighting at all.
        if self.modeltype in (
            Variable.ModelType.HotDeck,
            Variable.ModelType.StatMatch,
        ) and (self.weight != "" or bootstrap_enabled):
            reason = (
                "HotDeck's sequential donor-carry design has no natural "
                "notion of weighting"
                if self.modeltype == Variable.ModelType.HotDeck
                else "weighted donor sampling isn't implemented for it"
            )
            message = (
                f"{self.impute_var}: {self.modeltype.name} donor selection "
                f"is unweighted - {reason}. "
            )
            if self.weight != "":
                message += f"The declared weight ({self.weight!r}) is ignored here. "
            if bootstrap_enabled:
                message += (
                    "The Bayesian bootstrap's per-implicate weight "
                    "perturbation is also ignored, so between-implicate "
                    "variance for this variable comes only from random "
                    "donor tie-breaking, not from the bootstrap."
                )
            logger.warning(message)

            #   Donate vars shouldn't be in model
            #       Remove them and note it
            self._validate_hot_deck_remove_donates()

    def _validate_require_exclude_overlap(self):
        """
        A variable can't be both forced into the model (predictors_require)
        and excluded from it (predictors_exclude) - that's a contradictory
        configuration, so fail loudly here rather than silently picking one.

        Raises
        ------
        Exception
            predictors_require and predictors_exclude overlap.

        Returns
        -------
        None.

        """
        if not self.predictors_require:
            return

        overlap = set(self.predictors_require).intersection(self.predictors_exclude)
        if len(overlap):
            message = (
                f"{self.impute_var}: predictors_require and predictors_exclude "
                f"overlap on {sorted(overlap)} - a variable can't be both "
                f"required and excluded."
            )
            logger.error(message)
            raise Exception(message)

    def _validate_estimator_available(self, df: IntoFrameT):
        """
        If this variable declares categorical_feature (via
        Parameters.XGBoost()/CatBoost()/SklearnModel()), and model= is an
        R-style formula string, confirm any categorical_feature column
        referenced there is only ever passed through as a plain numeric
        term - catch this now, before the SRMI run starts, rather than
        mid-run.

        categorical_feature works by keeping those columns raw and casting
        them to a fixed-category dtype right before fitting. A bare
        reference to a string/categorical/enum/bool-dtype column still
        gets auto one-hot-encoded by dtype (regardless of C(...) - see
        survey_kit_formula's classify.py), which would leave no raw column
        behind for the categorical-dtype cast to apply to - so that
        combination is rejected. A numeric-dtype column stays untouched by
        the formula either way (it's a plain passthrough term), so it's
        fine for it to appear there too - _run_regression's model matrix
        already carries it through under its own name, ready for the
        categorical-dtype cast. A categorical_feature column can also
        simply be left out of the formula entirely, in which case
        _run_regression adds it into the model matrix as a plain untouched
        column alongside whatever the formula produces.

        Returns
        -------
        None.

        """
        if self.parameters is None:
            return

        categorical_feature = self.parameters.get("categorical_feature")
        if categorical_feature and type(self.model) is not list:
            formula_columns = set(FormulaBuilder.columns_from_formula(formula=self.model))
            referenced = [c for c in categorical_feature if c in formula_columns]
            if not referenced:
                return

            schema = nw.from_native(df).lazy().collect_schema()
            non_numeric_dtypes = (nw.String, nw.Categorical, nw.Enum, nw.Boolean)
            not_continuous = [
                c for c in referenced if schema[c] in non_numeric_dtypes
            ]
            if not_continuous:
                message = (
                    f"{self.impute_var}: categorical_feature columns "
                    f"{not_continuous} are referenced in model= (a formula) "
                    f"with a non-numeric dtype - a bare reference there "
                    f"still gets auto one-hot-encoded by dtype, leaving no "
                    f"raw column for categorical_feature's native-categorical "
                    f"casting to apply to. Either leave these columns out of "
                    f"the formula entirely, or pass a numeric-coded version "
                    f"of the column if it needs to appear there as a plain "
                    f"continuous term too."
                )
                logger.error(message)
                raise Exception(message)

    def union_required_predictors(self, formula: str) -> str:
        """
        Add any predictors_require variables not already present in formula.

        Used after variable selection (LASSO/stepwise) replaces the model
        formula, to guarantee required predictors survive selection - the
        selection methods themselves never see predictors_require (they
        don't receive the Variable object), so this has to happen here,
        in the caller, once selection has returned.

        Parameters
        ----------
        formula : str
            A model formula, typically the one returned by
            Selection.run()/lasso()/stepwise().

        Returns
        -------
        str
            formula, with any missing required predictors added.
        """
        if not self.predictors_require or formula == "":
            return formula

        fb = FormulaBuilder(formula=formula)
        existing = fb.columns_rhs
        for vari in self.predictors_require:
            if vari not in existing:
                fb.add_to_formula(vari)
        return fb.formula

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
            nw.from_native(
                compress_df(df=nw.from_native(df).select(self.impute_var).to_native())
            )
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
        #   Dedupe preserving order - list(set(...)) would reorder based
        #   on Python's per-process string hash randomization, breaking
        #   determinism across separate runs even with a fixed seed.
        full_models = list(dict.fromkeys(full_models))
        full_donate = [self.impute_var]
        if self.parameters["donate_list"] is not None:
            full_donate.extend(self.parameters["donate_list"])

        #   Order-preserving intersection (not set(...).intersection(...)),
        #   which would reorder based on Python's per-process string
        #   hash randomization - only used in a log message below, kept
        #   deterministic for readability/consistency.
        full_models_set = set(full_models)
        donates_in_models = [v for v in full_donate if v in full_models_set]

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
                    nw.from_native(df_bad).filter(
                        #   with_bad_donates' columns are all plain
                        #       booleans (is_null()/~is_null() never
                        #       produce null themselves) - ignore_nulls
                        #       is required by newer narwhals but has no
                        #       actual effect here.
                        nw.any_horizontal(df_bad.columns, ignore_nulls=True)
                    )
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

    def df_where(self, df: IntoFrameT, keep_vars: list | None = None) -> IntoFrameT:
        return self._df_where_list(df, [self.Where], keep_vars=keep_vars)

    def df_predict_where(
        self,
        df: IntoFrameT,
        drop_imputed: bool = False,
        keep_vars: list | None = None,
    ) -> IntoFrameT:
        if drop_imputed:
            where_list = [self.Where, self.Where_predict, self.Where_impute]
            negate_list = [False, False, True]
        else:
            where_list = [self.Where, self.Where_predict]
            negate_list = None

        #   The imputation_flag filter below runs after _df_where_list returns,
        #       so if we're projecting down to keep_vars inside _df_where_list,
        #       imputation_flag needs to survive that projection too.
        where_keep_vars = keep_vars
        if (
            keep_vars is not None
            and self.Where_predict_only_when_not_imputed
            and self.imputation_flag != ""
            and self.imputation_flag not in keep_vars
        ):
            where_keep_vars = keep_vars + [self.imputation_flag]

        df = self._df_where_list(
            df=df,
            where_list=where_list,
            negate_list=negate_list,
            keep_vars=where_keep_vars,
        )

        if self.Where_predict_only_when_not_imputed and self.imputation_flag != "":
            df = nw.from_native(df).filter(~nw.col(self.imputation_flag)).to_native()
            if where_keep_vars is not keep_vars:
                df = nw.from_native(df).select(keep_vars).to_native()
        return df

    def df_impute_where(self, df: IntoFrameT, keep_vars: list | None = None) -> IntoFrameT:
        return self._df_where_list(df, [self.Where, self.Where_impute], keep_vars=keep_vars)

    def df_impute_original_where(self, df: IntoFrameT) -> IntoFrameT:
        return self._df_where_list(df, [self.Where, self.Where_impute_original])

    def _df_where_list(
        self,
        df: IntoFrameT,
        where_list: list[str | None | nw.Expr],
        negate_list: list[bool] | None = None,
        keep_vars: list | None = None,
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

        df_lazy = nw.from_native(df).lazy()
        if keep_vars is not None:
            #   Project down to keep_vars before the single collect() below,
            #       while everything is still lazy - the filters above still
            #       see every column they need (the query optimizer pushes
            #       this projection past them), but the eventual materialized
            #       result only computes/reads keep_vars instead of every
            #       column in the input dataframe.
            df_lazy = df_lazy.select(keep_vars)

        return lazy_backend(df_lazy.collect(), nw_type).to_native()


#   ##########################################################
#   Variable.two_part() hooks - module-level (not nested closures),
#       matching the self-contained-function convention every other
#       pre/post/pre_initialize hook in this codebase already follows
#       (see tests/main/srmi.py's square_var/recalculate_interaction) -
#       column names and other per-call values ride along via
#       Variable.PrePost.Function's own `parameters=`, not a closure.
#       Only two hooks, not three - the y/n column itself is derived
#       once, directly into the returned df inside two_part() (see its
#       docstring for why), not via a pre_initialize hook.
#   ##########################################################
def _two_part_value_consistency(df, yn_var: str, value_var: str, value_if_no):
    import narwhals as nw

    #   yn_var isn't guaranteed to be boolean-dtype - a caller-given
    #       column (or an auto-built one) is just as likely to be an
    #       int-coded 0/1 flag, extremely common in raw survey data.
    #       ~nw.col(yn_var) on a non-boolean column is BITWISE NOT, not
    #       logical negation - ~1 == -2 and ~0 == -1 in two's complement,
    #       both truthy, so the "yn is false" branch below would
    #       silently match EVERY row (not just the false ones),
    #       overwriting every yn=True/value!=0 row with value_if_no.
    #       Cast to boolean explicitly so ~ means what it looks like.
    yn_bool = nw.col(yn_var).cast(nw.Boolean)
    fixed = (
        nw.when(yn_bool & (nw.col(value_var) == 0))
        .then(None)
        .when(~yn_bool)
        .then(value_if_no)
        .otherwise(nw.col(value_var))
        .alias(value_var)
    )
    return nw.from_native(df).with_columns(fixed).to_native()


def _two_part_drop_yn(df, yn_var: str):
    import narwhals as nw

    return nw.from_native(df).drop(yn_var).to_native()
