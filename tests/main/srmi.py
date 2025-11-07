import sys
import os
from pathlib import Path

import narwhals as nw
import polars as pl
import polars.selectors as cs

from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import summary

from survey_kit.utilities.formula_builder import FormulaBuilder
from survey_kit.imputation.utilities.lasso import Lasso
import survey_kit.imputation.utilities.lightgbm_wrapper as kit_lgbm
from survey_kit.imputation.utilities.lightgbm_wrapper import Tuner_optuna

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.selection import Selection
from survey_kit.imputation.srmi import SRMI
from survey_kit.orchestration.config import Config

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


parameters_pmm = Parameters.pmm(share_leave_out=0.1)
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


tuner = Tuner_optuna(
    n_trials=50, objective=kit_lgbm.Tuner.Objectives.mae, test_size=0.25
)

#   Set the tuner parameters to the defaults
tuner.parameters()

#   Set the ranges of values as follows
tuner.hyperparameters["num_leaves"] = [2, 256]
tuner.hyperparameters["max_depth"] = [2, 256]
tuner.hyperparameters["min_data_in_leaf"] = [10, 250]
tuner.hyperparameters["num_iterations"] = [25, 200]
tuner.hyperparameters["bagging_fraction"] = [0.5, 1]
tuner.hyperparameters["bagging_freq"] = [1, 5]


#   Impute a continuous variable with lgbm
#   Set the lightgbm parameters, note that the tuner won't be run if
#       there's already a saved version in tune_hyperparameter_path + variable name
#       unless tune_overwrite=True
#   Parameters set here are the defaults, but they are overwritten by
#       tuner parameters if that is passed (as it is here)
#   This is doing series of quantile regressions to determine your predicted
#       rank (effectively) then drawing from an empirical distribution
#       estimated with a pmm draw

parameters_lgbm = Parameters.LightGBM(
    tune=True,
    tune_hyperparameter_path=f"{Config().path_temp_files}/tuner_outputs",
    tuner=tuner,
    tune_overwrite=False,
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
    parameters_pmm=Parameters.pmm(share_leave_out=0.1),
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
#       This tuner is set to run at the beginning (tune_overwrite=True) even if
#           a file already exists in the path
parameters_lgbm2 = Parameters.LightGBM(
    tune=True,
    tune_hyperparameter_path=f"{config.data_root}/tuner_outputs",
    tuner=tuner,
    tune_overwrite=False,
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
    preFunctions=preFunctions,
)
vars_impute.append(v_lgbm2)


srmi = SRMI(
    df=df,  #     .lazy().collect().to_pandas(),
    variables=vars_impute,
    n_implicates=2,
    n_iterations=1,
    parallel=False,
    # parallel_CallInputs=CallInputs(CallType=CallTypes.shell,
    #                                CPUs=4,
    #                                MemInMB=5000),
    selection=Selection(method=Selection.Method.LASSO),
    # preselection=Selection(method=Selection.Method.LASSO,
    #                        parameters=Selection.Parameters.lasso(scale_lambda=0.5)),
    modeltype=modeltype,
    model=f_model.formula,
    bayesian_bootstrap=True,
    parallel_testing=False,
    path_model=f"{path_scratch}/py_srmi_test",
    force_start=True,
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

        if not srmi.parallel:
            logger.info("Assert imputed variables are equal across run/loaded")
            assert dfs[i].collect().equals(dfs_loaded[i].collect())

        if i > 0:
            logger.info(
                "Assert stable variables stay constant across run/loaded and implicates"
            )

            if not srmi.parallel:
                assert (
                    dfs[i]
                    .select(stable_vars)
                    .collect()
                    .equals(dfs_loaded[i - 1].select(stable_vars).collect())
                )
        logger.info("\n\n")
