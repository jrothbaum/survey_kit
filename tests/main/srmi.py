import sys
import os
from pathlib import Path

import numpy as np
import narwhals as nw
import polars as pl
import polars.selectors as cs

from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import summary

from survey_kit.utilities.formula_builder import FormulaBuilder
from survey_kit.imputation.utilities.lasso import Lasso
import survey_kit.imputation.utilities.lightgbm_wrapper as kit_lgbm
from survey_kit.imputation.utilities.lightgbm_wrapper import Tuner, Objective
from survey_kit.imputation.utilities.tuning import HyperparameterSpace, IntRange, FloatRange

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.selection import Selection
from survey_kit.imputation.srmi import SRMI

from survey_kit import logger, config
from survey_kit.utilities.dataframe import summary, columns_from_list

n_rows = 10_000
impute_share = 0.25


path_scratch = config.path_temp_files

df = (
    RandomData(n_rows=n_rows, seed=32565437)
    .index("index")
    .integer("year", 2016, 2020)
    .integer("month", 1, 12)
    .integer("var2", 0, 10)
    .integer("var3", 0, 50)
    .float("var4", 0, 1)
    .integer("var5", 0, 1)
    .float("unrelated_1", 0, 1)
    .float("unrelated_2", 0, 1)
    .float("unrelated_3", 0, 1)
    .float("unrelated_4", 0, 1)
    .float("unrelated_5", 0, 1)
    .np_distribution("epsilon_hd1", "normal", scale=5)
    .np_distribution("epsilon_hd2", "normal", scale=5)
    .np_distribution("epsilon_reg1", "normal", scale=5)
    .np_distribution("epsilon_reg2", "normal", scale=5)
    .np_distribution("epsilon_lgbm1", "normal", scale=5)
    .np_distribution("epsilon_lgbm2", "normal", scale=5)
    .float("missing_hd1", 0, 1)
    .float("missing_hd2", 0, 1)
    .float("missing_reg1", 0, 1)
    .float("missing_reg2", 0, 1)
    .float("missing_lgbm1", 0, 1)
    .float("missing_lgbm2", 0, 1)
    .to_df()
)


#   Convenience references to them for creating dependent variables
c_var2 = pl.col("var2")
c_var3 = pl.col("var3")
c_var4 = pl.col("var4")
c_var5 = pl.col("var5")

c_e_hd1 = pl.col("epsilon_hd1")
c_e_hd2 = pl.col("epsilon_hd2")

c_e_reg1 = pl.col("epsilon_reg1")
c_e_reg2 = pl.col("epsilon_reg2")

c_e_lgbm1 = pl.col("epsilon_lgbm1")
c_e_lgbm2 = pl.col("epsilon_lgbm2")


#   Create a bunch of variables that are functions of the variables created above
df = (
    df.with_columns(
        [
            (c_var2 * 2 - c_var3 * 3 * c_var5 + c_e_hd1).alias("var_hd1"),
            ((c_var2 * 1.5 - c_var3 * 1 * c_var4 + c_e_hd2) > 0).alias("var_hd2"),
            -(c_var2 + c_var3 * 2 * (1 - c_var5) + c_e_reg1).alias("var_reg1"),
            (-(c_var3 + c_var4 * (1 - c_var5) + c_e_reg2) > 0).alias("var_reg2"),
            (c_var2 - 2 * c_var3 * c_var4 * c_var5 + c_e_lgbm1).alias("var_lgbm1"),
            (
                (
                    c_var2
                    - 2 * c_var3 * c_var4
                    - 1.5 * c_var3 * c_var5
                    + c_var4 * c_var5
                    + c_e_lgbm2
                )
                > 0
            ).alias("var_lgbm2"),
        ]
    )
    .drop(columns_from_list(df=df, columns="epsilon*"))
    .with_row_index(name="_row_index_")
)
df_original = df

#   Set variables to missing according to the uniform random variables missing_
clear_missing = []
for prefixi in ["hd", "reg", "lgbm"]:
    for i in range(1, 3):
        vari = f"var_{prefixi}{i}"
        missingi = f"missing_{prefixi}{i}"

        clear_missing.append(
            pl.when(pl.col(missingi) < impute_share)
            .then(pl.lit(None))
            .otherwise(pl.col(vari))
            .alias(vari)
        )
df = df.with_columns(clear_missing).drop(cs.starts_with("missing_"))

#   Make a fully collinear var for testing
df = df.with_columns(pl.col("unrelated_1").alias("repeat_1"))


summary(df)


#   Actually do the imputation

#       The list of variables to impute (eventually)
vars_impute = []


#   1) Impute some variables to impute using stat match/hot deck
modeltype = Variable.ModelType.StatMatch
modeltype_binary = Variable.ModelType.HotDeck

#       Hot deck a continuous variable
#           Each model has a set of possible parameters
#           that determine what happens in the model
parameters_hd1 = Parameters.HotDeck(  #   model_list - a list of variables to match
    #       donors and recipients
    model_list=["var2", "var3", "var5", "var_reg2", "var_lgbm2", "var_hd2"],
    #   Drop the last variable sequentially
    #       until everyone has a match?
    sequential_drop=True,
    #   Donate anything other than the variable
    #       (i.e. donate together)
    #       In this case, it's redundant and does nothing...
    donate_list=["var_hd1"],
)

#           Set up the variable to be imputed
v_hd1 = Variable(  #  Name of the variable to be imputed
    impute_var="var_hd1",
    #  Run the model separately by group
    #      For hot decks, this is akin to just prepending
    #      this to model_list
    #  By=["year","month"],
    #  modeltype - set above as StatMatch or HotDeck
    modeltype=modeltype,
    #  Pass in the parameters set above
    parameters=parameters_hd1,
)
#           Add this variable to the list of variables to be imputed
vars_impute.append(v_hd1)


#       Hot deck a binary variable
#           Just doing the same basic stuff, but as the other type (HotDeck, rather than stat match)
parameters_hd2 = Parameters.HotDeck(
    model_list=["var2", "var3", "var5", "var_reg2", "var_lgbm2", "var_hd1"],
    sequential_drop=True,
    donate_list=["var_hd2"],
)

v_hd2 = Variable(
    impute_var="var_hd2",
    By=["year", "month"],
    modeltype=modeltype_binary,
    parameters=parameters_hd2,
)
vars_impute.append(v_hd2)


f_model = FormulaBuilder(df=df)
f_model.formula_with_varnames_in_brackets(
    "~1+{var_*}+var2+var4+var4*var3*C(var5)+{unrelated_*}+{repeat_*}"
)
logger.info(f_model.formula)


parameters_pmm = Parameters.pmm()
parameters_reg = Parameters.Regression(
    model=Parameters.RegressionModel.OLS,
    error=Parameters.ErrorDraw.pmm,
    parameters_pmm=parameters_pmm,
)


v_reg1 = Variable(
    impute_var="var_reg1",
    #   By=["year"],
    modeltype=Variable.ModelType.Regression,
    model=f_model.formula,
    parameters=parameters_reg,  # ,
    # selection=Selection(method=Selection.Method.LASSO,
    #                     select_within_by=False),
    # preselection=Selection(method=Selection.Method.LASSO,
    #                        parameters=Selection.Parameters.lasso(winsorize=[0.05,0.95],
    #                                                             missing_dummies=True,
    #                                                             scale_lambda=0.1))
)
vars_impute.append(v_reg1)


v_reg2 = Variable(
    impute_var="var_reg2",
    modeltype=Variable.ModelType.pmm,
    model=f_model.formula,
    parameters=parameters_pmm,
    # selection=Selection(method=Selection.Method.LASSO,
    #                     select_within_by=False),
    # preselection=Selection(method=Selection.Method.LASSO,
    #                        parameters=Selection.Parameters.lasso(missing_dummies=True,
    #                                                              scale_lambda=0.1))
)

vars_impute.append(v_reg2)


tuner = Tuner(
    space=HyperparameterSpace(
        num_leaves=IntRange(2, 256),
        max_depth=IntRange(2, 256),
        min_data_in_leaf=IntRange(10, 250),
        num_iterations=IntRange(25, 200),
        bagging_fraction=FloatRange(0.5, 1.0),
        bagging_freq=IntRange(1, 5),
    ),
    objective=Objective.mae,
    n_trials=50,
    path_save_dir=f"{config.path_temp_files}/tuner_outputs",
    overwrite=False,
)


#   Impute a continuous variable with lgbm
#   Set the lightgbm parameters, note that the tuner won't be run if
#       there's already a saved version in the tuner's own path_save_dir +
#       variable name, unless the tuner's own overwrite=True
#   Parameters set here are the defaults, but they are overwritten by
#       tuner parameters if that is passed (as it is here)
#   This is doing series of quantile regressions to determine your predicted
#       rank (effectively) then drawing from an empirical distribution
#       estimated with a pmm draw

parameters_lgbm = Parameters.LightGBM(
    tune=True,
    tuner=tuner,
    quantiles=[0.1, 0.5, 0.9],
    #  quantiles=[0.25,0.5,0.75],
    parameters={
        "objective": "regression",
        "num_leaves": 32,
        "min_data_in_leaf": 20,
        "num_iterations": 100,
        "test_size": 0.2,
        "boosting": "gbdt",
        "categorical_feature": ["Var5"],
        "verbose": -1,  # ,
        # "early_stopping_round":100
    },
    error=Parameters.ErrorDraw.pmm,
    parameters_pmm=Parameters.pmm(),
)


#   Test a simple pre-post function
#       These would get run gets run in each iteration (in each implicate)
#           before (preFunctions) or after (postFunctions) this variable is imputed
#   Notes for these functions:
#       1) No type hints on imported package types (will throw an error)
#           i.e. no df:pl.DataFrame or -> pl.DataFrame
#       2) Must be completely self-contained (i.e. all imports within the function)
#           This has to do with how it gets saved and loaded in async calls
#       3) Effectively, you have to assume it'll be called
#           in an environment with no imports before it
def square_var(df, var_to_square: str, name: str):
    import narwhals as nw

    return (
        nw.from_native(df)
        .with_columns((nw.col(var_to_square) ** 2).alias(name))
        .to_native()
    )


def recalculate_interaction(df, var1: str, var2: str, name: str):
    import narwhals as nw

    return (
        nw.from_native(df)
        .with_columns((nw.col(var1) * nw.col(var2)).alias(name))
        .to_native()
    )


v_lgbm1 = Variable(
    impute_var="var_lgbm1",
    model=["var_*", "var4", "var3", "var5", "unrelated_*", "repeat_*"],
    modeltype=Variable.ModelType.LightGBM,
    selection=Selection(method=Selection.Method.No),
    preselection=Selection(method=Selection.Method.No),
    parameters=parameters_lgbm,
    # postFunctions=Variable.PrePost.Function(square_var,
    #                                         parameters={"var_to_square":"var_lgbm1",
    #                                                     "name":"var_lgbm1_sq"})
)
vars_impute.append(v_lgbm1)


#   Impute a binary variable with lgbm, note that objective != quantile for a binary variable
#       This isn't working well and I wouldn't use it if I got these kinds of results
parameters_lgbm2 = Parameters.LightGBM(
    tune=True,
    tuner=tuner,
    # quantiles=[0.25,0.5,0.75],
    parameters={
        "objective": "binary",
        "num_leaves": 32,
        "min_data_in_leaf": 20,
        "num_iterations": 100,
        "test_size": 0.2,
        "boosting": "gbdt",
        "categorical_feature": ["Var5"],
        "verbose": -1,  # ,
        # "early_stopping_round":100
    },
    error=Parameters.ErrorDraw.pmm,
)


#    Test using an arbitrary function for pre-imputation processing
#       This gets run in each iteration (in each implicate) before this
#       variable is imputed
preFunctions = [
    Variable.PrePost.NarwhalsExpression((nw.col("var_reg1") ** 2).alias("var_reg1_sq")),
    Variable.PrePost.Function(
        recalculate_interaction,
        parameters={"var1": "var_reg1", "var2": "var_reg2", "name": "var_reg12"},
    ),
    Variable.PrePost.Function(
        square_var, parameters={"var_to_square": "var_reg1", "name": "var_reg1_sq"}
    ),
]
v_lgbm2 = Variable(
    impute_var="var_lgbm2",
    model=["var_*", "var4", "var3", "var5", "unrelated_*", "repeat_*"],
    modeltype=Variable.ModelType.LightGBM,
    selection=Selection(method=Selection.Method.No),
    preselection=Selection(method=Selection.Method.No),
    parameters=parameters_lgbm2,
    transforms=Variable.Transforms(pre=preFunctions),
)
vars_impute.append(v_lgbm2)


srmi = SRMI(
    df=df,  #     .lazy().collect().to_pandas(),
    variables=vars_impute,
    replication=SRMI.Replication(n_implicates=2, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False, testing=False),
    # parallel=SRMI.Parallel(enabled=False, testing=False,
    #                        call_inputs=CallInputs(call_type=CallTypes.shell,
    #                                                n_cpu=4, mem_in_mb=5000)),
    bootstrap=SRMI.Bootstrap(enabled=True),
    defaults=SRMI.Defaults(
        selection=Selection(method=Selection.Method.LASSO),
        # preselection=Selection(method=Selection.Method.LASSO,
        #                        parameters=Selection.Parameters.lasso(scale_lambda=0.5)),
        modeltype=modeltype,
        model=f_model.formula,
    ),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test", force_start=True
    ),
)

srmi.run()

if True:
    impute_vars = [vari.impute_var for vari in srmi.variables]
    df_original = nw.from_native(df_original).select(impute_vars).to_native()
    stable_vars = columns_from_list(df_original, columns="var*", exclude=impute_vars)

    summary(df_original)

    srmi_loaded = srmi.load(f"{path_scratch}/py_srmi_test")

    dfs = srmi.df_implicates
    dfs_loaded = srmi_loaded.df_implicates

    for i in range(len(dfs)):
        logger.info(f"{i} (imputed):")
        summary(dfs[i].select(impute_vars))
        summary(dfs_loaded[i].select(impute_vars))

        if not srmi.parallel.enabled:
            logger.info("Assert imputed variables are equal across run/loaded")
            assert dfs[i].collect().equals(dfs_loaded[i].collect())

        if i > 0:
            logger.info(
                "Assert stable variables stay constant across run/loaded and implicates"
            )

            if not srmi.parallel.enabled:
                assert (
                    dfs[i]
                    .select(stable_vars)
                    .collect()
                    .equals(dfs_loaded[i - 1].select(stable_vars).collect())
                )
        logger.info("\n\n")


#   ============================================================
#   Variable.Transforms.post_finalize - runs once per implicate, after the
#       LAST iteration completes (the mirror image of pre_initialize) - in
#       variable order, and NOT again on a resumed run that finds the
#       implicate already complete.
#   ============================================================

post_finalize_call_log = []


def _post_finalize_marker(df, name: str, drop_col: str = ""):
    post_finalize_call_log.append(name)
    if drop_col != "":
        import narwhals as nw

        df = nw.from_native(df).drop(drop_col).to_native()
    return df


n_pf = 400
df_pf = pl.DataFrame(
    dict(
        idx=range(n_pf),
        x1=[float(i % 7) for i in range(n_pf)],
        pf_a=[None if i % 4 == 0 else float(i) for i in range(n_pf)],
        pf_b=[None if i % 5 == 0 else float(i) * 2 for i in range(n_pf)],
    )
)

var_pf_a = Variable(
    impute_var="pf_a",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(error=Parameters.ErrorDraw.Random),
    transforms=Variable.Transforms(
        post_finalize=Variable.PrePost.Function(
            _post_finalize_marker, parameters={"name": "pf_a_finalize"}
        )
    ),
)
#   pf_b's post_finalize also drops idx*10 as a scaffolding-cleanup stand-in
#       (the actual motivating use case: a helper variable another
#       variable's pre/pre_initialize created only to drive its own
#       imputation, dropped once the whole implicate is done with it).
df_pf = df_pf.with_columns((pl.col("idx") * 10).alias("pf_scratch_col"))
var_pf_b = Variable(
    impute_var="pf_b",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(error=Parameters.ErrorDraw.Random),
    transforms=Variable.Transforms(
        post_finalize=Variable.PrePost.Function(
            _post_finalize_marker,
            parameters={"name": "pf_b_finalize", "drop_col": "pf_scratch_col"},
        )
    ),
)

srmi_pf = SRMI(
    df=df_pf,
    variables=[var_pf_a, var_pf_b],
    replication=SRMI.Replication(n_implicates=1, n_iterations=3),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_post_finalize", force_start=True
    ),
)
srmi_pf.run()

logger.info(f"post_finalize call log: {post_finalize_call_log}")
assert post_finalize_call_log == ["pf_a_finalize", "pf_b_finalize"], (
    "expected post_finalize to fire exactly once per variable, in variable "
    f"order, got {post_finalize_call_log}"
)

df_pf_out = nw.from_native(srmi_pf.implicates[0].df).lazy().collect().to_native()
assert "pf_scratch_col" not in df_pf_out.columns, (
    "expected pf_b's post_finalize to have dropped pf_scratch_col"
)
assert df_pf_out["pf_a"].null_count() == 0
assert df_pf_out["pf_b"].null_count() == 0

#   Resuming a run that's already complete must NOT call post_finalize again.
srmi_pf_loaded = SRMI.load(f"{path_scratch}/py_srmi_test_post_finalize")
srmi_pf_loaded.variables = [var_pf_a, var_pf_b]
srmi_pf_loaded.run()
assert post_finalize_call_log == ["pf_a_finalize", "pf_b_finalize"], (
    "expected NO additional post_finalize calls on a resumed already-"
    f"complete run, got {post_finalize_call_log}"
)

logger.info("srmi.py: post_finalize checks passed")


#   ============================================================
#   ModelType.NearestNeighbor removed / ModelType.pmm consolidated -
#       both are now just Regression under the hood (no separate
#       impute.py method, no separate modeltype dispatch beyond routing
#       to Impute.regression()).
#   ============================================================

logger.info("ModelType.NearestNeighbor was removed from the enum")
assert not hasattr(Variable.ModelType, "NearestNeighbor")

logger.info(
    "Parameters.NearestNeighbor() + modeltype=Regression - the "
    "replacement: fits an OLS on the same predictors instead of raw-"
    "distance matching, then PMM-matches on the fitted prediction"
)
n_nn = 400
df_nn = pl.DataFrame(
    dict(
        idx=range(n_nn),
        x1=[float((i * 7) % 23) / 5.0 for i in range(n_nn)],
        y=[None if i % 5 == 0 else float((i * 7) % 23) / 5.0 * 2.0 for i in range(n_nn)],
    )
)
var_nn = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.NearestNeighbor(match_to=["x1"]),
)
srmi_nn = SRMI(
    df=df_nn,
    variables=[var_nn],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_nearestneighbor_via_regression",
        force_start=True,
    ),
)
srmi_nn.run()
df_nn_out = nw.from_native(srmi_nn.implicates[0].df).lazy().collect().to_native()
assert df_nn_out["y"].is_null().sum() == 0
logger.info("ModelType.NearestNeighbor -> Regression consolidation passed")


#   ============================================================
#   Variable.two_part() - semicontinuous (point mass at zero + continuous)
#       two-part imputation shortcut.
#   ============================================================

rng_tp = np.random.default_rng(20260909)
n_tp = 3000
x1_tp = rng_tp.normal(size=n_tp)
x2_tp = rng_tp.normal(size=n_tp)

#   Real semicontinuous DGP: participation (yes/no) depends on x1,
#       amount (if any) depends on x2 - genuinely different predictors
#       per margin, the scenario two_part() exists for.
p_yes_tp = 1 / (1 + np.exp(-(1.0 * x1_tp)))
is_yes_tp = rng_tp.random(n_tp) < p_yes_tp
amount_tp = np.where(
    is_yes_tp,
    np.clip(50 + 20 * x2_tp + rng_tp.normal(scale=5, size=n_tp), 1, None),
    0.0,
)
miss_mask_tp = rng_tp.random(n_tp) < 0.25
earnings_missing = [
    None if miss_mask_tp[i] else float(amount_tp[i]) for i in range(n_tp)
]
df_tp = pl.DataFrame(
    dict(
        idx_tp=np.arange(n_tp),
        x1_tp=x1_tp,
        x2_tp=x2_tp,
        earnings_tp=earnings_missing,
        group_tp=rng_tp.choice(["a", "b"], size=n_tp),
    )
)


def run_two_part_case(name, path_suffix, df_in, two_part_kwargs, extra_vars=None):
    df_out_srmi, vars_tp = Variable.two_part(df=df_in, **two_part_kwargs)
    if extra_vars:
        vars_tp = extra_vars + vars_tp
    srmi_tp = SRMI(
        df=df_out_srmi,
        variables=vars_tp,
        index=["idx_tp"],
        replication=SRMI.Replication(n_implicates=1, n_iterations=3),
        parallel=SRMI.Parallel(enabled=False),
        bootstrap=SRMI.Bootstrap(enabled=True),
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_srmi_test_two_part_{path_suffix}",
            force_start=True,
        ),
    )
    srmi_tp.run()
    df_res = nw.from_native(srmi_tp.implicates[0].df).lazy().collect().to_native()
    assert df_res["earnings_tp"].is_null().sum() == 0, f"{name}: nulls remain"
    scratch_cols = [
        c for c in df_res.columns if c.startswith("___") and c.endswith("___")
    ]
    assert scratch_cols == [], f"{name}: scratch column(s) leaked: {scratch_cols}"
    logger.info(f"two_part: {name} passed ({len(vars_tp)} variable(s))")
    return df_res, vars_tp


logger.info("two_part: Regression bucket (direct swap - model=Logit for yn)")
_, vars_reg_tp = run_two_part_case(
    "Regression",
    "regression",
    df_tp,
    dict(
        impute_var="earnings_tp",
        model=["x1_tp", "x2_tp"],
        modeltype=Variable.ModelType.Regression,
        parameters=Parameters.Regression(
            model=Parameters.RegressionModel.OLS,
            error=Parameters.ErrorDraw.pmm,
            parameters_pmm=Parameters.pmm(donate_list=["x1_tp"]),
        ),
    ),
)
assert len(vars_reg_tp) == 2
assert vars_reg_tp[0].impute_var == "___earnings_tp_yn___"
assert vars_reg_tp[0].parameters["model"] == Parameters.RegressionModel.Logit
#   x1_tp (value's own extra donate_list entry, "also carry these other
#       variables from the SAME matched donor") must NOT be copied onto
#       yn's own, independently-matched donation - yn's own pass would
#       otherwise donate x1_tp from a different donor than value's pass
#       does, only to have value's pass silently overwrite it right
#       after (value runs second). earnings_tp itself (impute_var) SHOULD
#       be there instead - donating value alongside yn from the SAME
#       matched donor, right when yn is resolved, is more coherent than
#       leaving value to a separate, later, possibly-different match.
assert vars_reg_tp[0].parameters["donate_list"] == ["earnings_tp"]
assert vars_reg_tp[1].parameters["donate_list"] == ["x1_tp"]

logger.info("two_part: RandomForest bucket (OrderedCategorical reuse, error=leaf)")
_, vars_rf_tp = run_two_part_case(
    "RandomForest",
    "rf",
    df_tp,
    dict(
        impute_var="earnings_tp",
        model=["x1_tp", "x2_tp"],
        modeltype=Variable.ModelType.RandomForest,
        parameters=Parameters.RandomForest(
            parameters={"n_estimators": 100, "max_depth": 6},
            error=Parameters.ErrorDraw.leaf,
        ),
    ),
)
assert vars_rf_tp[0].modeltype == Variable.ModelType.OrderedCategorical

logger.info("two_part: HotDeck, no yn signal -> collapses to a single value-only variable")
df_hd_collapse, vars_hd_collapse = run_two_part_case(
    "HotDeck collapse",
    "hotdeck_collapse",
    df_tp,
    dict(
        impute_var="earnings_tp",
        model=["group_tp"],
        modeltype=Variable.ModelType.HotDeck,
        parameters=Parameters.HotDeck(model_list=["group_tp"]),
    ),
)
assert len(vars_hd_collapse) == 1
assert df_hd_collapse["earnings_tp"].is_null().sum() == 0

logger.info("two_part: HotDeck with yn_model given -> full two-part split")
_, vars_hd_full = run_two_part_case(
    "HotDeck full",
    "hotdeck_full",
    df_tp,
    dict(
        impute_var="earnings_tp",
        model=["group_tp"],
        modeltype=Variable.ModelType.HotDeck,
        parameters=Parameters.HotDeck(model_list=["group_tp"]),
        yn_model=["group_tp"],
    ),
)
assert len(vars_hd_full) == 2

logger.info(
    "two_part: yn_var given and fully observed -> value-only, no "
    "scratch column, no yn_variable built"
)
#   Fully observed - NOT derived from earnings_tp's own missingness
#       pattern (that would leave it null wherever earnings_tp is null
#       too, which is the DIFFERENT scenario tested right below).
df_existing_yn = df_tp.with_columns(pl.Series("has_earnings_tp", is_yes_tp))
_, vars_existing_tp = run_two_part_case(
    "yn_var given (fully observed)",
    "existing_yn",
    df_existing_yn,
    dict(
        impute_var="earnings_tp",
        model=["x1_tp", "x2_tp"],
        modeltype=Variable.ModelType.Regression,
        parameters=Parameters.Regression(
            model=Parameters.RegressionModel.OLS, error=Parameters.ErrorDraw.pmm
        ),
        yn_var="has_earnings_tp",
    ),
)
assert len(vars_existing_tp) == 1

logger.info(
    "two_part: yn_var given but STILL has missing values -> two_part "
    "builds a real yn_variable for it too, donate_list=[impute_var]"
)
df_existing_yn_missing = df_tp.with_columns(
    (pl.col("earnings_tp") != 0).alias("has_earnings_tp")
)
assert df_existing_yn_missing["has_earnings_tp"].null_count() > 0
_, vars_yn_missing_tp = run_two_part_case(
    "yn_var given (still missing)",
    "existing_yn_missing",
    df_existing_yn_missing,
    dict(
        impute_var="earnings_tp",
        model=["x1_tp", "x2_tp"],
        modeltype=Variable.ModelType.Regression,
        parameters=Parameters.Regression(
            model=Parameters.RegressionModel.OLS, error=Parameters.ErrorDraw.pmm
        ),
        yn_model=["x1_tp"],
        yn_var="has_earnings_tp",
    ),
)
assert len(vars_yn_missing_tp) == 2
assert vars_yn_missing_tp[0].impute_var == "has_earnings_tp"
assert vars_yn_missing_tp[0].parameters["donate_list"] == ["earnings_tp"]

logger.info("two_part: unsupported modeltype raises")
try:
    Variable.two_part(
        df=df_tp,
        impute_var="earnings_tp",
        model=["x1_tp"],
        modeltype=Variable.ModelType.Multinomial,
    )
    raise AssertionError("expected Multinomial to raise")
except AssertionError:
    raise
except ValueError as e:
    assert "semicontinuous" in str(e)
    logger.info(f"Correctly rejected: {e}")

logger.info("srmi.py: two_part checks passed")


#   ============================================================
#   SRMI.convergence() - matches mice's own convergence() as closely as
#       possible (see utilities/convergence_diagnostics.py).
#   ============================================================

rng_conv = np.random.default_rng(20260909)
n_conv = 800
x1_conv = rng_conv.normal(size=n_conv)
y_conv = 2.0 * x1_conv + rng_conv.normal(scale=0.5, size=n_conv)
miss_conv = rng_conv.random(n_conv) < 0.2
y_conv_missing = [None if miss_conv[i] else float(y_conv[i]) for i in range(n_conv)]
df_conv = pl.DataFrame(dict(idx_conv=range(n_conv), x1_conv=x1_conv, y_conv=y_conv_missing))

var_conv = Variable(
    impute_var="y_conv",
    model=["x1_conv"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(
        model=Parameters.RegressionModel.OLS, error=Parameters.ErrorDraw.pmm
    ),
)
srmi_conv = SRMI(
    df=df_conv,
    variables=[var_conv],
    index=["idx_conv"],
    replication=SRMI.Replication(n_implicates=4, n_iterations=8),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_convergence", force_start=True
    ),
)
srmi_conv.run()

logger.info("convergence: diagnostic='all', parameter='mean' (defaults)")
conv_all = srmi_conv.convergence()
conv_all_pl = nw.from_native(conv_all).lazy().collect().to_native()
assert set(conv_all_pl.columns) == {".it", "vrb", "ac", "psrf"}
assert conv_all_pl.height == 8  # one row per iteration, one variable
assert conv_all_pl["vrb"].unique().to_list() == ["y_conv"]
assert conv_all_pl.sort(".it")[".it"].to_list() == list(range(1, 9))
#   iteration 1 has no lag-1 correlation yet (matches mice's leading NA)
assert conv_all_pl.filter(pl.col(".it") == 1)["ac"].to_list()[0] is None or np.isnan(
    conv_all_pl.filter(pl.col(".it") == 1)["ac"].to_list()[0]
)
#   this is a simple, well-behaved OLS+pmm fit - by the last couple
#       iterations psrf should be reasonably close to 1 (loose bound,
#       not a tight statistical claim, just "not obviously diverging").
last_psrf = conv_all_pl.sort(".it")["psrf"].to_list()[-1]
logger.info(f"convergence: final-iteration psrf = {last_psrf}")
assert last_psrf is not None and not np.isnan(last_psrf)
assert last_psrf < 3.0, f"expected a well-behaved fit to have a modest psrf, got {last_psrf}"

logger.info("convergence: diagnostic='ac' only")
conv_ac = nw.from_native(srmi_conv.convergence(diagnostic="ac")).lazy().collect().to_native()
assert set(conv_ac.columns) == {".it", "vrb", "ac"}

logger.info("convergence: diagnostic='psrf'/'gr' only, both equivalent")
conv_psrf = nw.from_native(srmi_conv.convergence(diagnostic="psrf")).lazy().collect().to_native()
conv_gr = nw.from_native(srmi_conv.convergence(diagnostic="gr")).lazy().collect().to_native()
assert set(conv_psrf.columns) == {".it", "vrb", "psrf"}
assert conv_psrf.equals(conv_gr)

logger.info("convergence: parameter='sd'")
conv_sd = nw.from_native(srmi_conv.convergence(parameter="sd")).lazy().collect().to_native()
assert set(conv_sd.columns) == {".it", "vrb", "ac", "psrf"}

logger.info("convergence: bad diagnostic/parameter raise")
try:
    srmi_conv.convergence(diagnostic="bogus")
    raise AssertionError("expected a bad diagnostic to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

try:
    srmi_conv.convergence(parameter="bogus")
    raise AssertionError("expected a bad parameter to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

logger.info("convergence: m < 2 raises")
srmi_conv_1m = SRMI(
    df=df_conv,
    variables=[var_conv],
    index=["idx_conv"],
    replication=SRMI.Replication(n_implicates=1, n_iterations=8),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_convergence_1m", force_start=True
    ),
)
srmi_conv_1m.run()
try:
    srmi_conv_1m.convergence()
    raise AssertionError("expected m=1 to raise")
except AssertionError:
    raise
except ValueError as e:
    assert "implicates" in str(e)
    logger.info(f"Correctly rejected: {e}")

logger.info("convergence: iterations < 3 raises")
srmi_conv_2it = SRMI(
    df=df_conv,
    variables=[var_conv],
    index=["idx_conv"],
    replication=SRMI.Replication(n_implicates=2, n_iterations=2),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_convergence_2it", force_start=True
    ),
)
srmi_conv_2it.run()
try:
    srmi_conv_2it.convergence()
    raise AssertionError("expected 2 iterations to raise")
except AssertionError:
    raise
except ValueError as e:
    assert "iterations" in str(e)
    logger.info(f"Correctly rejected: {e}")

logger.info(
    "convergence: two_part()'s Where-restricted value_variable isn't "
    "contaminated by value_if_no zeros"
)
rng_conv_tp = np.random.default_rng(3)
n_conv_tp = 1500
x1_conv_tp = rng_conv_tp.normal(size=n_conv_tp)
p_yes_conv_tp = 1 / (1 + np.exp(-x1_conv_tp))
is_yes_conv_tp = rng_conv_tp.random(n_conv_tp) < p_yes_conv_tp
amount_conv_tp = np.where(
    is_yes_conv_tp,
    np.clip(30 + rng_conv_tp.normal(scale=5, size=n_conv_tp), 1, None),
    0.0,
)
miss_conv_tp = rng_conv_tp.random(n_conv_tp) < 0.25
earnings_conv_tp = [
    None if miss_conv_tp[i] else float(amount_conv_tp[i]) for i in range(n_conv_tp)
]
df_conv_tp = pl.DataFrame(
    dict(idx_conv_tp=range(n_conv_tp), x1_conv_tp=x1_conv_tp, earnings_conv_tp=earnings_conv_tp)
)
df_conv_tp2, vars_conv_tp = Variable.two_part(
    df=df_conv_tp,
    impute_var="earnings_conv_tp",
    model=["x1_conv_tp"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(
        model=Parameters.RegressionModel.OLS, error=Parameters.ErrorDraw.pmm
    ),
)
srmi_conv_tp = SRMI(
    df=df_conv_tp2,
    variables=vars_conv_tp,
    index=["idx_conv_tp"],
    replication=SRMI.Replication(n_implicates=3, n_iterations=4),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_convergence_two_part", force_start=True
    ),
)
srmi_conv_tp.run()
true_positive_mean = amount_conv_tp[amount_conv_tp > 0].mean()
for impi in srmi_conv_tp.implicates:
    for it, val in impi.chain_mean["earnings_conv_tp"].items():
        if val is not None:
            assert abs(val - true_positive_mean) < 5.0, (
                f"two_part()'s value chain_mean ({val}) should track the "
                f"true positive-amount mean ({true_positive_mean}), not be "
                f"pulled toward 0 by value_if_no"
            )
logger.info("convergence: two_part() Where-conditioning check passed")

logger.info("srmi.py: convergence checks passed")


#   ============================================================
#   Extending a completed SRMI to a higher n_iterations - mice has no
#       automatic "run until converged" mode either (maxit is always
#       fixed; mice.mids() is how you bolt on more iterations after
#       checking convergence by hand) - this is survey_kit's version of
#       that: reload, bump replication.n_iterations, reset complete,
#       run() again. Surfaced three real, pre-existing bugs the first
#       time this was tried (none specific to convergence tracking
#       itself): (1) Serializable.save() deleted its own target folder
#       BEFORE materializing any still-lazy (freshly reloaded) frames
#       that were scanning FROM files in that same folder: a bare
#       SRMI.load(path) + .save() with zero other changes crashed with
#       FileNotFoundError every time. (2) int dict keys (chain_mean/
#       chain_std's iteration keys) come back as strings after a JSON
#       save/load round trip, breaking int comparisons downstream. (3)
#       Implicate._run_one_iteration's per-variable resume guard
#       (status_variable <= variable_index) can't distinguish "this
#       variable just finished" from "still in progress" at the exact
#       boundary, so re-entering an ALREADY-complete iteration (which
#       happens every time you extend, since the outer loop always
#       restarts at iteration 1) silently re-ran that iteration's last
#       variable a second time.
#   ============================================================

rng_ext = np.random.default_rng(20260909)
n_ext = 500
x1_ext = rng_ext.normal(size=n_ext)
y_ext = 2.0 * x1_ext + rng_ext.normal(scale=0.5, size=n_ext)
miss_ext = rng_ext.random(n_ext) < 0.2
y_ext_missing = [None if miss_ext[i] else float(y_ext[i]) for i in range(n_ext)]
df_ext = pl.DataFrame(dict(idx_ext=range(n_ext), x1_ext=x1_ext, y_ext=y_ext_missing))

var_ext = Variable(
    impute_var="y_ext",
    model=["x1_ext"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(
        model=Parameters.RegressionModel.OLS, error=Parameters.ErrorDraw.pmm
    ),
)
path_ext = f"{path_scratch}/py_srmi_test_extend"
srmi_ext = SRMI(
    df=df_ext,
    variables=[var_ext],
    index=["idx_ext"],
    replication=SRMI.Replication(n_implicates=2, n_iterations=3),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(path_model=path_ext, force_start=True),
)
srmi_ext.run()

logger.info("extend: a bare reload + re-save (no changes) doesn't crash")
srmi_ext_reload = SRMI.load(path_ext)
srmi_ext_reload.implicates[0].save()

logger.info("extend: reload, bump n_iterations, reset complete, run() again")
srmi_ext2 = SRMI.load(path_ext)
srmi_ext2.replication.n_iterations = 6
for impi in srmi_ext2.implicates:
    impi.complete = False
srmi_ext2.run()

#   run() calling _initialize_implicates() unconditionally must NOT
#       duplicate implicates that SRMI.load() already populated -
#       confirmed this was a real bug (silently doubled self.implicates,
#       corrupting anything that reads len(self.implicates) directly,
#       like convergence()'s own m).
assert len(srmi_ext2.implicates) == 2, (
    f"expected exactly 2 implicates (no duplication from re-running "
    f"_initialize_implicates on an already-loaded SRMI), got "
    f"{len(srmi_ext2.implicates)}"
)
for impi in srmi_ext2.implicates:
    assert impi.status_iteration == 6
    assert impi.complete
    keys = sorted(int(it) for it in impi.chain_mean["y_ext"].keys())
    assert keys == list(range(1, 7)), (
        f"expected clean 1..6 with no duplicate/redundant re-run, got {keys}"
    )

df_ext_out = nw.from_native(srmi_ext2.implicates[0].df).lazy().collect().to_native()
assert df_ext_out["y_ext"].is_null().sum() == 0

logger.info("extend: convergence() works on the reloaded+extended object")
conv_ext = nw.from_native(srmi_ext2.convergence()).lazy().collect().to_native()
assert conv_ext.height == 6
assert conv_ext.sort(".it")[".it"].to_list() == list(range(1, 7))

logger.info("srmi.py: extend-n_iterations checks passed")


#   ============================================================
#   SRMI.plot_convergence() - mice's plot(imp)/plot.mids() equivalent.
#       No "auto-plot" flag anywhere in run() - purely a standalone,
#       callable-any-time method on top of the same chain_mean/
#       chain_std data convergence() reads, requiring the optional
#       'plotly' package only when actually called.
#   ============================================================

logger.info("plot_convergence: 'both' (default) parameter")
fig_both = srmi_conv.plot_convergence()
import plotly.graph_objects as go

assert isinstance(fig_both, go.Figure)
#   4 implicates x (1 variable x 2 parameters (mean, sd)) = 8 traces
assert len(fig_both.data) == 8, f"expected 8 traces, got {len(fig_both.data)}"

logger.info("plot_convergence: 'mean' only")
fig_mean = srmi_conv.plot_convergence(parameter="mean")
assert len(fig_mean.data) == 4, f"expected 4 traces, got {len(fig_mean.data)}"

logger.info("plot_convergence: 'sd' only")
fig_sd = srmi_conv.plot_convergence(parameter="sd")
assert len(fig_sd.data) == 4, f"expected 4 traces, got {len(fig_sd.data)}"

logger.info("plot_convergence: path= also saves a self-contained HTML file")
path_html = f"{path_scratch}/py_srmi_test_plot_convergence.html"
fig_saved = srmi_conv.plot_convergence(path=path_html)
assert isinstance(fig_saved, go.Figure)
assert os.path.isfile(path_html)
assert os.path.getsize(path_html) > 0

logger.info("plot_convergence: bad parameter raises")
try:
    srmi_conv.plot_convergence(parameter="bogus")
    raise AssertionError("expected a bad parameter to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

logger.info("srmi.py: plot_convergence checks passed")


#   ============================================================
#   SRMI.plot_imputation_quality() - mice's densityplot()/stripplot()/
#       bwplot() equivalent - a DIFFERENT question from convergence
#       ("do imputed values look plausible"), not "did the chain
#       stabilize". No auto-plot flag, same as plot_convergence().
#   ============================================================

rng_qual = np.random.default_rng(20260909)
n_qual = 1000
x1_qual = rng_qual.normal(size=n_qual)
x2_qual = rng_qual.normal(size=n_qual)
y1_qual = 2.0 * x1_qual + rng_qual.normal(scale=0.5, size=n_qual)
y2_qual = -1.0 * x2_qual + rng_qual.normal(scale=1.0, size=n_qual)
miss1_qual = rng_qual.random(n_qual) < 0.25
miss2_qual = rng_qual.random(n_qual) < 0.25
df_qual = pl.DataFrame(
    dict(
        idx_qual=range(n_qual),
        x1_qual=x1_qual,
        x2_qual=x2_qual,
        y1_qual=[None if miss1_qual[i] else float(y1_qual[i]) for i in range(n_qual)],
        y2_qual=[None if miss2_qual[i] else float(y2_qual[i]) for i in range(n_qual)],
    )
)
var1_qual = Variable(
    impute_var="y1_qual",
    model=["x1_qual"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(
        model=Parameters.RegressionModel.OLS, error=Parameters.ErrorDraw.pmm
    ),
)
var2_qual = Variable(
    impute_var="y2_qual",
    model=["x2_qual"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(
        model=Parameters.RegressionModel.OLS, error=Parameters.ErrorDraw.pmm
    ),
)
srmi_qual = SRMI(
    df=df_qual,
    variables=[var1_qual, var2_qual],
    index=["idx_qual"],
    replication=SRMI.Replication(n_implicates=3, n_iterations=3),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_quality", force_start=True
    ),
)
srmi_qual.run()

logger.info("plot_imputation_quality: variable=None (all), kind='density' (default)")
fig_qual_all = srmi_qual.plot_imputation_quality()
#   2 variables x (1 Observed + 3 implicates) = 8 traces
assert len(fig_qual_all.data) == 8, f"expected 8 traces, got {len(fig_qual_all.data)}"

logger.info("plot_imputation_quality: variable by name")
fig_qual_name = srmi_qual.plot_imputation_quality(variable="y2_qual", kind="density")
assert len(fig_qual_name.data) == 4

logger.info("plot_imputation_quality: variable by 0-indexed position")
fig_qual_idx = srmi_qual.plot_imputation_quality(variable=0, kind="box")
assert len(fig_qual_idx.data) == 4

logger.info("plot_imputation_quality: variable as a mixed name/index list")
fig_qual_list = srmi_qual.plot_imputation_quality(variable=["y1_qual", 1], kind="strip")
assert len(fig_qual_list.data) == 8

logger.info("plot_imputation_quality: sample_k limits strip points, ignored for density/box")
fig_qual_sampled = srmi_qual.plot_imputation_quality(
    variable="y1_qual", kind="strip", sample_k=25, seed=42
)
max_pts = max(
    len(tr.x) for tr in fig_qual_sampled.data if getattr(tr, "x", None) is not None
)
assert max_pts <= 25, f"expected sample_k to cap points at 25, got {max_pts}"

logger.info("plot_imputation_quality: path= also saves a self-contained HTML file")
path_qual_html = f"{path_scratch}/py_srmi_test_plot_quality.html"
srmi_qual.plot_imputation_quality(path=path_qual_html)
assert os.path.isfile(path_qual_html)
assert os.path.getsize(path_qual_html) > 0

logger.info("plot_imputation_quality: bad variable name/index/kind raise")
try:
    srmi_qual.plot_imputation_quality(variable="bogus")
    raise AssertionError("expected a bad variable name to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

try:
    srmi_qual.plot_imputation_quality(variable=99)
    raise AssertionError("expected an out-of-range index to raise")
except AssertionError:
    raise
except IndexError as e:
    logger.info(f"Correctly rejected: {e}")

try:
    srmi_qual.plot_imputation_quality(kind="bogus")
    raise AssertionError("expected a bad kind to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

logger.info("srmi.py: plot_imputation_quality checks passed")


#   ============================================================
#   SRMI.plot_propensity() - a DIFFERENT, conditional question from
#       plot_imputation_quality()'s marginal comparison: under MAR,
#       missing rows can legitimately have a different marginal
#       distribution than observed ones, so this compares observed vs.
#       imputed CONDITIONAL on predicted response-propensity (a binary
#       LightGBM model of the imputation_flag on that variable's own
#       predictors) instead of overall. kind="density" (the default)
#       compares the kernel density of residuals from regressing y on
#       the propensity - matching the diagnostic in Raghunathan &
#       Bondarenko (2007)/Bondarenko & Raghunathan (2016), the same one
#       used in the user's own SRMI/CPS-ASEC paper. kind="binned_mean"
#       is a simpler mean-by-propensity-bin alternative. Also covers
#       categorical_feature threading through to the propensity model -
#       a variable (e.g. CatBoost()) can keep a categorical predictor
#       OUT of its own formula and add it only via categorical_feature,
#       which must still reach the propensity model or it silently
#       vanishes.
#   ============================================================

logger.info("plot_propensity: variable=None (all, no categorical predictors), kind='density' default")
fig_prop_all = srmi_qual.plot_propensity()
#   2 variables x (1 Observed + 3 implicates) = 8 traces
assert len(fig_prop_all.data) == 8, f"expected 8 traces, got {len(fig_prop_all.data)}"

logger.info("plot_propensity: variable by name")
fig_prop_name = srmi_qual.plot_propensity(variable="y2_qual")
assert len(fig_prop_name.data) == 4

logger.info("plot_propensity: variable by 0-indexed position, kind='binned_mean'")
fig_prop_idx = srmi_qual.plot_propensity(variable=0, kind="binned_mean", n_bins=5)
assert len(fig_prop_idx.data) == 4

logger.info("plot_propensity: kind='binned_mean' with n_bins < 2 raises")
try:
    srmi_qual.plot_propensity(kind="binned_mean", n_bins=1)
    raise AssertionError("expected n_bins=1 to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

logger.info("plot_propensity: kind='density' with cv_folds < 2 raises")
try:
    srmi_qual.plot_propensity(cv_folds=1)
    raise AssertionError("expected cv_folds=1 to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

logger.info("plot_propensity: kind='density' with a smaller cv_folds still produces traces")
fig_prop_cv = srmi_qual.plot_propensity(cv_folds=3)
assert len(fig_prop_cv.data) == 8, f"expected 8 traces, got {len(fig_prop_cv.data)}"

logger.info("plot_propensity: bad kind raises")
try:
    srmi_qual.plot_propensity(kind="bogus")
    raise AssertionError("expected a bad kind to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

logger.info("plot_propensity: bad variable name/index raise")
try:
    srmi_qual.plot_propensity(variable="bogus")
    raise AssertionError("expected a bad variable name to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

try:
    srmi_qual.plot_propensity(variable=99)
    raise AssertionError("expected an out-of-range index to raise")
except AssertionError:
    raise
except IndexError as e:
    logger.info(f"Correctly rejected: {e}")

logger.info("plot_propensity: path= also saves a self-contained HTML file")
path_prop_html = f"{path_scratch}/py_srmi_test_plot_propensity.html"
srmi_qual.plot_propensity(path=path_prop_html)
assert os.path.isfile(path_prop_html)
assert os.path.getsize(path_prop_html) > 0

#   ------------------------------------------------------------
#   categorical_feature threading: a CatBoost() variable whose
#       categorical predictor is deliberately kept out of the
#       formula (added only via categorical_feature, same convention
#       tests/main/tabular_ml.py uses) must still reach the
#       propensity model as a native categorical, not silently drop
#       out of the predictor list or crash on a raw string column.
#   ------------------------------------------------------------

rng_prop_cat = np.random.default_rng(20260910)
n_prop_cat = 800
x_prop_cat = rng_prop_cat.normal(size=n_prop_cat)
cat_levels_prop = {"a": 0.0, "b": 3.0, "c": -2.0}
cat_prop = rng_prop_cat.choice(list(cat_levels_prop.keys()), size=n_prop_cat)
cat_effect_prop = np.array([cat_levels_prop[ci] for ci in cat_prop])
y_prop_cat = 1.5 * x_prop_cat + cat_effect_prop + rng_prop_cat.normal(
    scale=0.5, size=n_prop_cat
)
miss_prop_cat = rng_prop_cat.random(n_prop_cat) < 0.25
df_prop_cat = pl.DataFrame(
    dict(
        idx_prop_cat=range(n_prop_cat),
        x_prop_cat=x_prop_cat,
        cat_prop_cat=cat_prop,
        y_prop_cat=[
            None if miss_prop_cat[i] else float(y_prop_cat[i])
            for i in range(n_prop_cat)
        ],
    )
)
var_prop_cat = Variable(
    impute_var="y_prop_cat",
    model="~1+x_prop_cat",
    modeltype=Variable.ModelType.CatBoost,
    parameters=Parameters.CatBoost(categorical_feature=["cat_prop_cat"]),
)
srmi_prop_cat = SRMI(
    df=df_prop_cat,
    variables=[var_prop_cat],
    index=["idx_prop_cat"],
    replication=SRMI.Replication(n_implicates=2, n_iterations=2),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_propensity_cat", force_start=True
    ),
)
srmi_prop_cat.run()

logger.info("plot_propensity: categorical_feature (CatBoost, formula-excluded) threads through")
resolved_predictors = srmi_prop_cat._variable_predictors(
    var_prop_cat, srmi_prop_cat.implicates[0].df
)
assert "cat_prop_cat" in resolved_predictors, (
    f"categorical_feature column should be unioned into predictors even "
    f"though it's not in the formula, got {resolved_predictors}"
)
assert srmi_prop_cat._variable_categorical_predictors(var_prop_cat) == ["cat_prop_cat"]

fig_prop_cat = srmi_prop_cat.plot_propensity()
#   1 variable x (1 Observed + 2 implicates) = 3 traces
assert len(fig_prop_cat.data) == 3, f"expected 3 traces, got {len(fig_prop_cat.data)}"

fig_prop_cat_binned = srmi_prop_cat.plot_propensity(kind="binned_mean", n_bins=4)
assert len(fig_prop_cat_binned.data) == 3, f"expected 3 traces, got {len(fig_prop_cat_binned.data)}"

logger.info("srmi.py: plot_propensity checks passed")


#   ============================================================
#   SRMI.simple_model() - survey_kit's mice(data, m=5)-equivalent
#       one-liner on-ramp. Covers: auto-detected continuous/binary,
#       explicitly-declared ordered_categorical/unordered_categorical
#       (with required ordered_categories), categorical_predictors
#       (native categorical_feature for LightGBM, C(...) formula for
#       Multinomial/OrderedCategorical's default RandomForest
#       estimator - neither has native categorical support), auto-
#       folding a declared class into categorical_predictors when used
#       as someone else's predictor, group_levels (applied where
#       supported, logged+ignored where not, auto-excluded as an
#       ordinary predictor either way), per-variable exclude, yn_pairs
#       via Variable.two_part(), and variables_to_impute's "exact list,
#       no auto-scan" semantics. This end-to-end run is also the
#       permanent regression coverage for three real, pre-existing bugs
#       found while building it: _two_part_value_consistency's bitwise-
#       NOT-on-a-non-boolean-column data corruption (variable.py),
#       LightGBM's list-form categorical_feature crash on raw string
#       predictors plus train/predict code-consistency (lightgbm_wrapper.py),
#       and the post-impute-stats has_summarizable_dtype guard checking
#       the wrong scope (impute.py).
#   ============================================================

rng_simple = np.random.default_rng(20260911)
n_simple = 1500

x1_simple = rng_simple.normal(size=n_simple)
x2_simple = rng_simple.normal(size=n_simple)
cat_pred_levels_simple = {"a": 0.0, "b": 2.0, "c": -1.5}
cat_pred_simple = rng_simple.choice(list(cat_pred_levels_simple.keys()), size=n_simple)
cat_pred_effect_simple = np.array([cat_pred_levels_simple[c] for c in cat_pred_simple])
state_simple = rng_simple.choice(["ca", "tx", "ny", "fl"], size=n_simple)
state_effect_simple = {"ca": 1.0, "tx": -0.5, "ny": 0.5, "fl": -1.0}
group_effect_simple = np.array([state_effect_simple[s] for s in state_simple])

y_cont_simple = (
    2.0 * x1_simple
    - 1.0 * x2_simple
    + cat_pred_effect_simple
    + group_effect_simple
    + rng_simple.normal(scale=0.5, size=n_simple)
)
y_bin_latent_simple = (
    1.0 * x1_simple + 0.5 * cat_pred_effect_simple + rng_simple.normal(scale=1.0, size=n_simple)
)
y_bin_simple = (y_bin_latent_simple > np.median(y_bin_latent_simple)).astype(int)

ordinal_levels_simple = ["low", "mid", "high"]
ordinal_score_simple = 0.8 * x2_simple + rng_simple.normal(scale=1.0, size=n_simple)
ordinal_idx_simple = np.clip(
    (ordinal_score_simple - ordinal_score_simple.min()) / np.ptp(ordinal_score_simple) * 3,
    0,
    2.999,
).astype(int)
y_ordinal_simple = [ordinal_levels_simple[i] for i in ordinal_idx_simple]

unordered_levels_simple = ["red", "green", "blue", "yellow"]
y_unordered_simple = rng_simple.choice(unordered_levels_simple, size=n_simple)

#   Int-coded (0/1), not boolean - the exact dtype that triggered the
#       _two_part_value_consistency bitwise-NOT bug.
hours_yn_simple = (rng_simple.random(n_simple) < 0.8).astype(int)
hours_value_simple = np.where(
    hours_yn_simple == 1,
    30 + 5 * x1_simple + rng_simple.normal(scale=3, size=n_simple),
    0.0,
)

df_simple = pl.DataFrame(
    dict(
        idx_simple=range(n_simple),
        x1_simple=x1_simple,
        x2_simple=x2_simple,
        cat_pred_simple=cat_pred_simple,
        state_simple=state_simple,
        y_cont_simple=y_cont_simple,
        y_bin_simple=y_bin_simple,
        y_ordinal_simple=y_ordinal_simple,
        y_unordered_simple=y_unordered_simple,
        hours_yn_simple=hours_yn_simple,
        hours_value_simple=hours_value_simple,
        #   Near-perfect proxy for y_cont_simple - should never show up
        #       as a predictor for it once explicitly excluded below.
        downstream_only_simple=y_cont_simple * 2 + 1,
    )
)

miss_mask_simple = {}
for _col, _share in [
    ("y_cont_simple", 0.2),
    ("y_bin_simple", 0.2),
    ("y_ordinal_simple", 0.2),
    ("y_unordered_simple", 0.2),
    ("hours_value_simple", 0.15),
]:
    miss_mask_simple[_col] = rng_simple.random(n_simple) < _share

df_simple = df_simple.with_columns(
    [
        pl.when(pl.Series(miss_mask_simple[_col]))
        .then(None)
        .otherwise(pl.col(_col))
        .alias(_col)
        for _col in miss_mask_simple
    ]
)

logger.info("simple_model: fully auto plus explicit overrides")
srmi_simple = SRMI.simple_model(
    df=df_simple,
    index="idx_simple",
    classes={
        "y_ordinal_simple": Variable.Class.ordered_categorical,
        "y_unordered_simple": Variable.Class.unordered_categorical,
    },
    ordered_categories={"y_ordinal_simple": ordinal_levels_simple},
    categorical_predictors=["cat_pred_simple"],
    group_levels="state_simple",
    exclude={"y_cont_simple": ["downstream_only_simple"]},
    yn_pairs={"hours_value_simple": "hours_yn_simple"},
    replication=SRMI.Replication(n_implicates=2, n_iterations=2),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_simple_model", force_start=True
    ),
)

impute_vars_built_simple = {v.impute_var for v in srmi_simple.variables}
assert "y_cont_simple" in impute_vars_built_simple
assert "y_bin_simple" in impute_vars_built_simple
assert "y_ordinal_simple" in impute_vars_built_simple
assert "y_unordered_simple" in impute_vars_built_simple
assert "hours_value_simple" in impute_vars_built_simple
#   hours_yn_simple has no missingness by construction - two_part()
#       correctly builds no Variable for it (nothing to impute).
assert "hours_yn_simple" not in impute_vars_built_simple

v_ycont_simple = next(
    v for v in srmi_simple.variables if v.impute_var == "y_cont_simple"
)
assert v_ycont_simple.modeltype == Variable.ModelType.LightGBM
assert "downstream_only_simple" not in v_ycont_simple.model, (
    f"per-variable exclude not applied: {v_ycont_simple.model}"
)
assert "cat_pred_simple" in v_ycont_simple.model
assert "state_simple" not in v_ycont_simple.model, (
    "group_levels var should be excluded as an ordinary predictor"
)
assert set(v_ycont_simple.parameters["parameters"].get("categorical_feature", [])) == {
    "cat_pred_simple",
    "y_ordinal_simple",
    "y_unordered_simple",
}, v_ycont_simple.parameters

v_ybin_simple = next(v for v in srmi_simple.variables if v.impute_var == "y_bin_simple")
assert v_ybin_simple.modeltype == Variable.ModelType.LightGBM

v_ord_simple = next(
    v for v in srmi_simple.variables if v.impute_var == "y_ordinal_simple"
)
assert v_ord_simple.modeltype == Variable.ModelType.OrderedCategorical

v_unord_simple = next(
    v for v in srmi_simple.variables if v.impute_var == "y_unordered_simple"
)
assert v_unord_simple.modeltype == Variable.ModelType.Multinomial
#   Multinomial has no native categorical support -> C(...) formula.
assert "C(cat_pred_simple)" in v_unord_simple.model, v_unord_simple.model

logger.info("simple_model: running end to end (locks in the three bug fixes above)")
srmi_simple.run()
logger.info("simple_model: run() completed OK")

logger.info("simple_model: variables_to_impute gives an exact list, no auto-scan")
srmi_simple2 = SRMI.simple_model(
    df=df_simple,
    index="idx_simple",
    variables_to_impute=["y_cont_simple"],
    categorical_predictors=["cat_pred_simple"],
    replication=SRMI.Replication(n_implicates=2, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_simple_model2", force_start=True
    ),
)
assert [v.impute_var for v in srmi_simple2.variables] == ["y_cont_simple"]

logger.info("simple_model: ordered_categorical without ordered_categories raises")
try:
    SRMI.simple_model(
        df=df_simple,
        index="idx_simple",
        classes={"y_ordinal_simple": Variable.Class.ordered_categorical},
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_srmi_test_simple_model3", force_start=True
        ),
    )
    raise AssertionError("expected missing ordered_categories to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

logger.info("simple_model: bad variables_to_impute entry raises a clear error")
try:
    SRMI.simple_model(
        df=df_simple,
        index="idx_simple",
        variables_to_impute=["bogus_col_simple"],
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_srmi_test_simple_model4", force_start=True
        ),
    )
    raise AssertionError("expected a bogus variables_to_impute entry to raise")
except AssertionError:
    raise
except ValueError as e:
    logger.info(f"Correctly rejected: {e}")

logger.info("simple_model: model= override, bare ModelType - RandomForest for continuous")
srmi_simple_model_override = SRMI.simple_model(
    df=df_simple,
    index="idx_simple",
    variables_to_impute=["y_cont_simple"],
    model={Variable.Class.continuous: Variable.ModelType.RandomForest},
    group_levels="state_simple",
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_simple_model5", force_start=True
    ),
)
v_override = srmi_simple_model_override.variables[0]
assert v_override.modeltype == Variable.ModelType.RandomForest
#   RandomForest supports group_levels (unlike the LightGBM default) -
#       should actually be threaded through this time, not just logged
#       and ignored.
assert v_override.parameters.get("group_levels") == ["state_simple"], v_override.parameters
assert "state_simple" not in v_override.model, (
    "group_levels var should still be excluded as an ordinary predictor"
)

logger.info("simple_model: model= override, (ModelType, parameters) tuple - exact parameters used")
custom_xgb_parameters = Parameters.XGBoost(
    categorical_feature=["cat_pred_simple"], error=Parameters.ErrorDraw.pmm
)
srmi_simple_tuple_override = SRMI.simple_model(
    df=df_simple,
    index="idx_simple",
    variables_to_impute=["y_cont_simple"],
    model={
        Variable.Class.continuous: (Variable.ModelType.XGBoost, custom_xgb_parameters)
    },
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_simple_model6", force_start=True
    ),
)
v_tuple_override = srmi_simple_tuple_override.variables[0]
assert v_tuple_override.modeltype == Variable.ModelType.XGBoost
assert v_tuple_override.parameters["categorical_feature"] == ["cat_pred_simple"]

logger.info("simple_model: exclude_global applies to every built variable, not just one")
srmi_simple_exclude_global = SRMI.simple_model(
    df=df_simple,
    index="idx_simple",
    variables_to_impute=["y_cont_simple", "y_bin_simple"],
    exclude_global=["downstream_only_simple"],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_simple_model7", force_start=True
    ),
)
for v in srmi_simple_exclude_global.variables:
    assert "downstream_only_simple" not in v.model, (
        f"{v.impute_var}: exclude_global not applied, model={v.model}"
    )

logger.info("simple_model: auto_binary=False forces continuous even for a 0/1 column")
srmi_simple_no_auto_binary = SRMI.simple_model(
    df=df_simple,
    index="idx_simple",
    variables_to_impute=["y_bin_simple"],
    auto_binary=False,
    model={Variable.Class.continuous: Variable.ModelType.RandomForest},
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_simple_model8", force_start=True
    ),
)
#   y_bin_simple is 0/1 - with auto_binary defaulted True it would
#       resolve to Class.binary (LightGBM default, untouched by the
#       continuous-only override above); with it off, it must resolve
#       to Class.continuous instead, picking up the override.
assert srmi_simple_no_auto_binary.variables[0].modeltype == Variable.ModelType.RandomForest

logger.info("simple_model: index accepts a multi-column list")
df_simple_multi_index = df_simple.with_columns(
    pl.Series("idx2_simple", [f"r{i}" for i in range(df_simple.height)])
)
srmi_simple_multi_index = SRMI.simple_model(
    df=df_simple_multi_index,
    index=["idx_simple", "idx2_simple"],
    variables_to_impute=["y_cont_simple"],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_srmi_test_simple_model9", force_start=True
    ),
)
assert srmi_simple_multi_index.index == ["idx_simple", "idx2_simple"]
v_multi_index = srmi_simple_multi_index.variables[0]
assert "idx_simple" not in v_multi_index.model
assert "idx2_simple" not in v_multi_index.model

logger.info("srmi.py: simple_model checks passed")


#   ============================================================
#   _lightgbm_simple's cv_folds bug: the plain (non-quantile)
#       LightGBM path computed its PMM donor-pool prediction purely
#       in-sample, never checking self.variable.parameters["cv_folds"]
#       at all - _lightgbm_quantiles's own pmm branch and
#       _run_regression (RandomForest/XGBoost/CatBoost/SklearnModel)
#       both did this correctly; _lightgbm_simple silently ignored
#       cv_folds entirely. A plain "does it crash" test can't catch
#       this (the in-sample path doesn't crash either) - assert the
#       donor-pool prediction (and therefore the imputed values
#       themselves) actually CHANGES between cv_folds=0 and
#       cv_folds=5, using a deliberately overfittable model
#       (unregularized, many leaves relative to n) where in-sample vs.
#       out-of-fold predictions are guaranteed to diverge.
#   ============================================================

rng_cv = np.random.default_rng(20260910)
n_cv = 1500
x1_cv = rng_cv.normal(size=n_cv)
y_cv_data = 2.0 * x1_cv + rng_cv.normal(scale=3.0, size=n_cv)

df_cv = pl.DataFrame(dict(idx_cv=range(n_cv), x1_cv=x1_cv, y_cv=y_cv_data))
miss_cv = rng_cv.random(n_cv) < 0.2
df_cv = df_cv.with_columns(
    pl.when(pl.Series(miss_cv)).then(None).otherwise(pl.col("y_cv")).alias("y_cv")
)

overfit_lgbm_parameters = {
    "num_leaves": 200,
    "num_iterations": 300,
    "learning_rate": 0.3,
    "min_data_in_leaf": 1,
    "verbose": -1,
}


def _build_and_run_cv_folds_srmi(cv_folds, path_suffix):
    v = Variable(
        impute_var="y_cv",
        model=["x1_cv"],
        modeltype=Variable.ModelType.LightGBM,
        parameters=Parameters.LightGBM(
            parameters=dict(overfit_lgbm_parameters),
            error=Parameters.ErrorDraw.pmm,
            cv_folds=cv_folds,
        ),
    )
    srmi_built = SRMI(
        df=df_cv,
        variables=[v],
        index=["idx_cv"],
        replication=SRMI.Replication(n_implicates=1, n_iterations=1),
        parallel=SRMI.Parallel(enabled=False),
        bootstrap=SRMI.Bootstrap(enabled=False),
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_srmi_test_lgbm_cv_folds_{path_suffix}",
            force_start=True,
        ),
    )
    srmi_built.run()
    return srmi_built


logger.info("_lightgbm_simple: cv_folds=0 (in-sample donor pool)")
srmi_cv_off = _build_and_run_cv_folds_srmi(cv_folds=0, path_suffix="off")

logger.info("_lightgbm_simple: cv_folds=5 (out-of-fold donor pool)")
srmi_cv_on = _build_and_run_cv_folds_srmi(cv_folds=5, path_suffix="on")

y_cv_off = nw.from_native(srmi_cv_off.implicates[0].df).lazy().collect()["y_cv"]
y_cv_on = nw.from_native(srmi_cv_on.implicates[0].df).lazy().collect()["y_cv"]

n_diff_cv = (
    pl.DataFrame({"off": y_cv_off, "on": y_cv_on}).filter(pl.col("off") != pl.col("on")).height
)
logger.info(f"Rows where the imputed value differs between cv_folds=0 and cv_folds=5: {n_diff_cv}")
assert n_diff_cv > 0, (
    "cv_folds=5 produced identical imputed values to cv_folds=0 for a plain "
    "LightGBM variable - _lightgbm_simple's donor-pool prediction isn't actually "
    "using cv_folds (regression of the fix)"
)

logger.info("srmi.py: _lightgbm_simple cv_folds checks passed")
