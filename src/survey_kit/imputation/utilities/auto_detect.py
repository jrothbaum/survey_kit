"""
Builds a ready-to-run list[Variable] (and, when yn_pairs is used, a
possibly-updated df - see Variable.two_part()) from a dataframe and a
handful of light, optional overrides. This is the implementation behind
SRMI.simple_model() - survey_kit's equivalent of mice's `mice(data,
m=5)` one-liner on-ramp: point it at a dataframe, get back something
ready to .run(), inspect/tweak by hand from there.

Nothing here does anything Variable/Parameters/SRMI couldn't already do
by hand - it's purely a defaulting layer that decides, per variable:
which columns need imputing (any nulls, unless you say otherwise),
whether a variable is binary or continuous (dtype/0-1-only check -
NEVER categorical, ordered or unordered; a categorical variable always
needs an explicit Class declaration, since there's no way to infer
intended category order, or that a numeric-looking column is really a
category code, from the data alone), and which ModelType/Parameters to
use by default for each Class (LightGBM for binary/continuous,
RandomForest-backed for both categorical classes). Anything this
doesn't cover - deep per-variable customization - always remains
reachable by hand-building a Variable and passing it via
SRMI(variables=[...]) directly instead of through here.
"""

from __future__ import annotations

import narwhals as nw
from narwhals.typing import IntoFrameT

from ..variable import Variable
from ..parameters import Parameters
from ... import logger

#   Everything here works through plain narwhals (nw.Expr/nw.col/
#       schema dtype checks) rather than assuming a polars-native df -
#       this runs on the RAW df a caller hands SRMI.simple_model()
#       BEFORE SRMI.__init__ has done any of its own backend handling,
#       so it can't assume polars the way code deeper in the SRMI/
#       Implicate/Impute pipeline (which normalizes to polars
#       internally) safely does. Same convention Variable's own
#       _apply_where_filters uses for backend-agnostic nw.Expr work.
_NON_NUMERIC_DTYPES = (nw.String, nw.Categorical, nw.Enum, nw.Boolean)


#   Modeltypes with native categorical-predictor support
#       (categorical_feature=...) - everything else needs a categorical
#       predictor one-hot-encoded via a formula string's C(...) syntax
#       instead (RandomForest()'s own documented workaround).
_NATIVE_CATEGORICAL_MODELTYPES = {
    Variable.ModelType.LightGBM,
    Variable.ModelType.XGBoost,
    Variable.ModelType.CatBoost,
}

#   Modeltypes whose Parameters accept group_levels/group_shrinkage_k
#       (the ones routing through Parameters._sklearn_model_params) -
#       LightGBM/Multinomial/OrderedCategorical do NOT (see
#       _sklearn_model_params's own docstring for LightGBM's case;
#       Multinomial()/OrderedCategorical() simply have no such params).
_GROUP_LEVELS_SUPPORTED_MODELTYPES = {
    Variable.ModelType.Regression,
    Variable.ModelType.RandomForest,
    Variable.ModelType.XGBoost,
    Variable.ModelType.CatBoost,
    Variable.ModelType.SklearnModel,
}

DEFAULT_MODEL_BY_CLASS = {
    Variable.Class.binary: Variable.ModelType.LightGBM,
    Variable.Class.continuous: Variable.ModelType.LightGBM,
    Variable.Class.ordered_categorical: Variable.ModelType.OrderedCategorical,
    Variable.Class.unordered_categorical: Variable.ModelType.Multinomial,
}


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _is_effectively_binary(df: IntoFrameT, schema, col: str) -> bool:
    dtype = schema[col]
    if dtype == nw.Boolean:
        return True
    if dtype in _NON_NUMERIC_DTYPES:
        return False
    distinct = (
        nw.from_native(df)
        .lazy()
        .select(nw.col(col))
        .drop_nulls(subset=[col])
        .unique(subset=[col])
        .collect()[col]
        .to_list()
    )
    return len(distinct) > 0 and set(distinct) <= {0, 1}


def _default_parameters(
    modeltype: Variable.ModelType,
    ordered_categories_for_var: list | None,
    group_levels: list[str],
    var: str,
) -> dict:
    if modeltype == Variable.ModelType.LightGBM:
        parameters = Parameters.LightGBM(error=Parameters.ErrorDraw.pmm)
    elif modeltype == Variable.ModelType.RandomForest:
        parameters = Parameters.RandomForest(
            error=Parameters.ErrorDraw.pmm, group_levels=group_levels or None
        )
    elif modeltype == Variable.ModelType.XGBoost:
        parameters = Parameters.XGBoost(
            error=Parameters.ErrorDraw.pmm, group_levels=group_levels or None
        )
    elif modeltype == Variable.ModelType.CatBoost:
        parameters = Parameters.CatBoost(
            error=Parameters.ErrorDraw.pmm, group_levels=group_levels or None
        )
    elif modeltype == Variable.ModelType.Regression:
        parameters = Parameters.Regression(
            model=Parameters.RegressionModel.OLS,
            error=Parameters.ErrorDraw.pmm,
            group_levels=group_levels or None,
        )
    elif modeltype == Variable.ModelType.Multinomial:
        parameters = Parameters.Multinomial()
    elif modeltype == Variable.ModelType.OrderedCategorical:
        if not ordered_categories_for_var:
            message = (
                f"auto_detect: {var!r} is ordered_categorical but has no "
                f"entry in ordered_categories - the category order can't "
                f"be inferred from the data, it must be given explicitly "
                f"(ordered_categories={{{var!r}: [...]}})."
            )
            logger.error(message)
            raise ValueError(message)
        parameters = Parameters.OrderedCategorical(categories=ordered_categories_for_var)
    else:
        message = (
            f"auto_detect: no built-in default Parameters for "
            f"modeltype={modeltype!r} - pass an explicit "
            f"(modeltype, parameters) tuple in `model=` for {var!r} instead "
            f"of a bare ModelType."
        )
        logger.error(message)
        raise ValueError(message)

    if group_levels and modeltype not in _GROUP_LEVELS_SUPPORTED_MODELTYPES:
        logger.info(
            f"auto_detect: group_levels was given but {modeltype!r} "
            f"(chosen for {var!r}) doesn't support it - ignored for this "
            f"variable. Override model={{class: ModelType.RandomForest}} "
            f"(or Regression/XGBoost/CatBoost/SklearnModel) for {var!r} if "
            f"you want group shrinkage to apply here."
        )

    return parameters


def _apply_categorical_feature(
    modeltype: Variable.ModelType, parameters: dict, cat_feature_list: list[str]
) -> dict:
    if not cat_feature_list:
        return parameters
    parameters = dict(parameters)
    if modeltype == Variable.ModelType.LightGBM:
        #   LightGBM()'s own Parameters nest the raw lgbm kwargs under
        #       "parameters" - see impute.py's lightgbm() method, which
        #       reads self.variable.parameters["parameters"] directly.
        nested = dict(parameters.get("parameters") or {})
        nested["categorical_feature"] = cat_feature_list
        parameters["parameters"] = nested
    else:
        parameters["categorical_feature"] = cat_feature_list
    return parameters


def _predictor_spec(
    predictors: list[str], categorical_predictors: set[str], native_categorical_support: bool
) -> tuple[list[str] | str, list[str]]:
    """
    Returns (model_spec, categorical_feature_list). model_spec is
    either the plain predictor list (native-categorical modeltypes -
    categorical_feature_list tells the caller which of them to declare)
    or an R-style formula string wrapping flagged predictors in C(...)
    (everything else - no separate categorical_feature_list needed,
    hence always []).
    """
    if native_categorical_support:
        cat_feature_list = [p for p in predictors if p in categorical_predictors]
        return list(predictors), cat_feature_list

    if not any(p in categorical_predictors for p in predictors):
        return list(predictors), []

    terms = [f"C({p})" if p in categorical_predictors else p for p in predictors]
    formula = "~1+" + "+".join(terms) if terms else "~1"
    return formula, []


def _resolve_modeltype_and_parameters(
    var: str,
    var_class: Variable.Class,
    model_overrides: dict,
    ordered_categories: dict,
    group_levels: list[str],
) -> tuple[Variable.ModelType, dict]:
    override = model_overrides.get(var_class)
    if override is None:
        modeltype = DEFAULT_MODEL_BY_CLASS[var_class]
        parameters = _default_parameters(
            modeltype, ordered_categories.get(var), group_levels, var
        )
    elif isinstance(override, tuple):
        modeltype, parameters = override
        parameters = dict(parameters)
    else:
        modeltype = override
        parameters = _default_parameters(
            modeltype, ordered_categories.get(var), group_levels, var
        )
    return modeltype, parameters


def build_simple_model_variables(
    df: IntoFrameT,
    index: list[str] | str | None = None,
    variables_to_impute: list[str] | None = None,
    classes: dict[str, Variable.Class] | None = None,
    auto_binary: bool = True,
    ordered_categories: dict[str, list] | None = None,
    model: dict[Variable.Class, Variable.ModelType | tuple] | None = None,
    yn_pairs: dict[str, str] | None = None,
    exclude: dict[str, list[str]] | None = None,
    exclude_global: list[str] | None = None,
    categorical_predictors: list[str] | None = None,
    group_levels: list[str] | str | None = None,
) -> tuple[IntoFrameT, list[Variable]]:
    """
    See this module's own docstring for the overall design. Parameters
    mirror SRMI.simple_model()'s own - that's a thin wrapper around this
    function plus an SRMI(...) construction call.

    Returns
    -------
    tuple[IntoFrameT, list[Variable]]
        The (possibly Variable.two_part()-updated, for any yn_pairs)
        df, and the built variable list, in the order: yn/value pairs
        first (as [yn_variable, value_variable] each, matching
        Variable.two_part()'s own ordering), then every other
        auto-detected/explicitly-listed variable.
    """
    index = _as_list(index)
    classes = dict(classes) if classes else {}
    ordered_categories = dict(ordered_categories) if ordered_categories else {}
    model_overrides = dict(model) if model else {}
    yn_pairs = dict(yn_pairs) if yn_pairs else {}
    exclude = dict(exclude) if exclude else {}
    exclude_global = set(_as_list(exclude_global))
    categorical_predictors = set(_as_list(categorical_predictors))
    #   A variable declared ordered_categorical/unordered_categorical
    #       (in `classes`) is - by definition - categorical whenever it
    #       shows up as a PREDICTOR for some other variable too. We
    #       already know this from `classes` - no reason to also make
    #       the caller repeat it in categorical_predictors.
    categorical_predictors |= {
        var
        for var, var_class in classes.items()
        if var_class
        in (Variable.Class.ordered_categorical, Variable.Class.unordered_categorical)
    }
    group_levels = _as_list(group_levels)

    always_exclude = exclude_global | set(index) | set(group_levels)

    df_native = nw.from_native(df).lazy().collect().to_native()
    df_lazy = nw.from_native(df_native).lazy()
    schema = df_lazy.collect_schema()
    columns = schema.names()

    if variables_to_impute is None:
        null_counts = df_lazy.select(
            [nw.col(c).null_count().alias(c) for c in columns]
        ).collect()
        variables_to_impute = [
            c
            for c in columns
            if c not in set(index) and null_counts[c].item() > 0
        ]
    else:
        variables_to_impute = list(variables_to_impute)
        missing = [c for c in variables_to_impute if c not in columns]
        if missing:
            message = (
                f"auto_detect: variables_to_impute contains column(s) not "
                f"present in df: {missing} - have: {columns}."
            )
            logger.error(message)
            raise ValueError(message)

    value_var_names = set(yn_pairs.keys())
    yn_var_names = set(yn_pairs.values())

    def _predictors_for(var: str) -> list[str]:
        var_exclude = set(exclude.get(var, []))
        return [
            c
            for c in columns
            if c != var and c not in always_exclude and c not in var_exclude
        ]

    def _resolve_class(var: str) -> Variable.Class:
        if var in classes:
            return classes[var]
        if auto_binary and _is_effectively_binary(df_native, schema, var):
            return Variable.Class.binary
        return Variable.Class.continuous

    result_variables: list[Variable] = []

    #   yn/value pairs first - explicit, so processed regardless of
    #       whether variables_to_impute would otherwise have included
    #       value_var/yn_var.
    for value_var, yn_var in yn_pairs.items():
        var_class = _resolve_class(value_var)
        modeltype, parameters = _resolve_modeltype_and_parameters(
            value_var, var_class, model_overrides, ordered_categories, group_levels
        )
        predictors = _predictors_for(value_var)
        native_cat = modeltype in _NATIVE_CATEGORICAL_MODELTYPES
        model_spec, cat_feature_list = _predictor_spec(
            predictors, categorical_predictors, native_cat
        )
        parameters = _apply_categorical_feature(modeltype, parameters, cat_feature_list)

        logger.info(
            f"auto_detect: {value_var!r} (yn={yn_var!r}) -> class={var_class.name}, "
            f"modeltype={modeltype.name}, predictors={predictors}"
        )

        df_native, new_vars = Variable.two_part(
            df=df_native,
            impute_var=value_var,
            model=model_spec,
            modeltype=modeltype,
            parameters=parameters,
            yn_model=model_spec,
            yn_var=yn_var,
        )
        result_variables.extend(new_vars)

    #   Everything else.
    for var in variables_to_impute:
        if var in value_var_names or var in yn_var_names:
            continue

        var_class = _resolve_class(var)
        modeltype, parameters = _resolve_modeltype_and_parameters(
            var, var_class, model_overrides, ordered_categories, group_levels
        )
        predictors = _predictors_for(var)
        native_cat = modeltype in _NATIVE_CATEGORICAL_MODELTYPES
        model_spec, cat_feature_list = _predictor_spec(
            predictors, categorical_predictors, native_cat
        )
        parameters = _apply_categorical_feature(modeltype, parameters, cat_feature_list)

        logger.info(
            f"auto_detect: {var!r} -> class={var_class.name}, "
            f"modeltype={modeltype.name}, predictors={predictors}"
        )

        result_variables.append(
            Variable(
                impute_var=var,
                model=model_spec,
                modeltype=modeltype,
                parameters=parameters,
            )
        )

    return df_native, result_variables
