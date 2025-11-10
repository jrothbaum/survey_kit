import sys
import os
from pathlib import Path

import narwhals as nw
import polars as pl
import polars.selectors as cs

from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import summary

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.srmi import SRMI
from survey_kit.imputation.selection import Selection
import survey_kit.imputation.utilities.lightgbm_wrapper as rep_lgbm
from survey_kit.imputation.utilities.lightgbm_wrapper import Tuner_optuna

from survey_kit import logger, config
from survey_kit.utilities.dataframe import summary, columns_from_list


# %%
# Draw some random data

n_rows = 10_000
impute_share = 0.25


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
    .np_distribution("epsilon_gbm1", "normal", scale=5)
    .np_distribution("epsilon_gbm2", "normal", scale=5)
    .np_distribution("epsilon_gbm3", "normal", scale=5)
    .float("missing_gbm1", 0, 1)
    .float("missing_gbm2", 0, 1)
    .float("missing_gbm3", 0, 1)
    .to_df()
)


#   Convenience references to them for creating dependent variables
c_var2 = pl.col("var2")
c_var3 = pl.col("var3")
c_var4 = pl.col("var4")
c_var5 = pl.col("var5")

c_e_gbm1 = pl.col("epsilon_gbm1")
c_e_gbm2 = pl.col("epsilon_gbm2")


#   Convenience references to them for creating dependent variables
c_var2 = pl.col("var2")
c_var3 = pl.col("var3")
c_var4 = pl.col("var4")
c_var5 = pl.col("var5")


logger.info("var_gbm1 is binary and conditional on other variables")
c_gbm1 = ((c_var2 * 2 - c_var3 * 3 * c_var5 + c_e_gbm1) > 0).alias("var_gbm1")

logger.info("var_gbm2 is != 0 only if var_gbm1 == True")
c_gbm2 = (
    pl.when(pl.col("var_gbm1"))
    .then((c_var2 * 1.5 - c_var3 * 1 * c_var4 + c_e_gbm2))
    .otherwise(pl.lit(0))
    .alias("var_gbm2")
)

c_gbm3 = (
    pl.when(pl.col("var_gbm1"))
    .then((c_var2 * 1.5 - c_var3 * 1 * c_var4 + c_e_gbm2))
    .otherwise(pl.lit(0))
    .alias("var_gbm3")
)
#   Create a bunch of variables that are functions of the variables created above
df = (
    df.with_columns(c_gbm1)
    .with_columns(c_gbm2, c_gbm3)
    .drop(columns_from_list(df=df, columns="epsilon*"))
    .with_row_index(name="_row_index_")
)
df_original = df

#   Set variables to missing according to the uniform random variables missing_
clear_missing = []
for prefixi in ["gbm"]:
    for i in range(1, 4):
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


# %%
logger.info("Define some dummy functions to run after imputation of 2")


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


# %%
logger.info("Set up hyperparameter tuning")
tuner = Tuner_optuna(
    n_trials=50, objective=rep_lgbm.Tuner.Objectives.mae, test_size=0.25
)

logger.info("   Set the tuner parameters to the defaults")
tuner.parameters()

logger.info("   Then specify ranges to check between as follow")
tuner.hyperparameters["num_leaves"] = [2, 256]
tuner.hyperparameters["max_depth"] = [2, 256]
tuner.hyperparameters["min_data_in_leaf"] = [10, 250]
tuner.hyperparameters["num_iterations"] = [25, 200]
tuner.hyperparameters["bagging_fraction"] = [0.5, 1]
tuner.hyperparameters["bagging_freq"] = [1, 5]


vars_impute = []

# %%
logger.info("Impute the boolean variable (var_gbm1)")
logger.info("   to the default setup for predicted mean matching")
logger.info("   using lightgbm")
logger.info("   (you can pass a formula, but you don't need to)")

logger.info("First, set up the lightgbm parameters")
logger.info("   This says, do hyperparameter tuning first (tune)")
logger.info("   Redo it at each run (tune_overwrite)")
logger.info(
    "   And sets the lightgbm parameter defaults (parameters) that the tuning can overwrite"
)
parameters_lgbm1 = Parameters.LightGBM(
    tune=True,
    tune_hyperparameter_path=f"{config.data_root}/tuner_outputs",
    tuner=tuner,
    tune_overwrite=True,
    parameters={
        "objective": "binary",
        "num_leaves": 32,
        "min_data_in_leaf": 20,
        "num_iterations": 100,
        "test_size": 0.2,
        "boosting": "gbdt",
        "categorical_feature": ["var5"],
        "verbose": -1,  # ,
    },
    error=Parameters.ErrorDraw.pmm,
)


logger.info("Actually define the variable and the model")
v_gbm1 = Variable(
    impute_var="var_gbm1",
    model=["var_*", "var4", "var3", "var5", "unrelated_*", "repeat_*"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=parameters_lgbm1,
)
logger.info("Add the variable to the list to be imputed")
vars_impute.append(v_gbm1)


logger.info("Impute the continuous variable (var_gbm2) ")
logger.info("   conditional on var_gbm1, using narwhals (nw.col('var_gbm1'))")
logger.info("   as well as a post-processing edit to set var_gbm2=0 when var_gbm1==0")
logger.info("   and some other random post-processing")
logger.info("Different parameters for the continuous variable")
parameters_lgbm2 = Parameters.LightGBM(
    tune=True,
    tune_hyperparameter_path=f"{config.data_root}/tuner_outputs",
    tuner=tuner,
    tune_overwrite=True,
    parameters={
        "objective": "regression",
        "num_leaves": 32,
        "min_data_in_leaf": 20,
        "num_iterations": 100,
        "test_size": 0.2,
        "boosting": "gbdt",
        "categorical_feature": ["var5"],
        "verbose": -1,  # ,
    },
    error=Parameters.ErrorDraw.pmm,
)

v_gbm2 = Variable(
    impute_var="var_gbm2",
    Where=nw.col("var_gbm1"),
    #   Needed in case var_gbm1 changes between iterations
    Where_predict=(nw.col("var_gbm2") != 0),
    model=["var_*", "var4", "var3", "var5", "unrelated_*", "repeat_*"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=parameters_lgbm2,
    postFunctions=[
        (
            nw.when(nw.col("var_gbm1"))
            .then(nw.col("var_gbm2"))
            .otherwise(nw.lit(0))
            .alias("var_gbm2")
        ),
        Variable.PrePost.Function(
            recalculate_interaction,
            parameters=dict(var1="var_gbm1", var2="var_gbm2", name="var_gbm12"),
        ),
        Variable.PrePost.Function(
            square_var, parameters=dict(var_to_square="var_gbm2", name="var_gbm2_sq")
        ),
    ],
)

vars_impute.append(v_gbm2)


logger.info("Now do one with the quantile-regression lightgbm")
logger.info("   To do this, pass quantiles and set objective='quantile'")
parameters_lgbm3 = Parameters.LightGBM(
    tune=True,
    tune_hyperparameter_path=f"{config.data_root}/tuner_outputs",
    tuner=tuner,
    tune_overwrite=True,
    quantiles=[0.25, 0.5, 0.75],
    parameters={
        "objective": "quantile",
        "num_leaves": 32,
        "min_data_in_leaf": 20,
        "num_iterations": 100,
        "test_size": 0.2,
        "boosting": "gbdt",
        "categorical_feature": ["var5"],
        "verbose": -1,  # ,
    },
    error=Parameters.ErrorDraw.pmm,
)

v_gbm3 = Variable(
    impute_var="var_gbm3",
    Where=nw.col("var_gbm1"),
    #   Needed in case var_gbm1 changes between iterations
    Where_predict=(nw.col("var_gbm3") != 0),
    model=["var_*", "var4", "var3", "var5", "unrelated_*", "repeat_*"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=parameters_lgbm3,
    postFunctions=[
        (
            nw.when(nw.col("var_gbm1"))
            .then(nw.col("var_gbm3"))
            .otherwise(nw.lit(0))
            .alias("var_gbm3")
        )
    ],
)

vars_impute.append(v_gbm3)


# %%
logger.info("Set up the imputation")
srmi = SRMI(
    df=df,
    variables=vars_impute,
    n_implicates=2,
    n_iterations=2,
    parallel=False,
    index=["index"],
    modeltype=Variable.ModelType.pmm,
    bayesian_bootstrap=True,
    path_model=f"{config.path_temp_files}/py_srmi_test_gbm",
    force_start=True,
)

# %%
logger.info("Run it")
srmi.run()

logger.info("It's automatically saved and can be loaded with (see path_model above):")
logger.info("path_model = f'{config.path_temp_files}/py_srmi_test_gbm'")
logger.info("srmi = SRMI.load(path_model)")


# %%
logger.info("Get the results")
_ = df_list = srmi.df_implicates

# %%
logger.info("\n\nLook at the original")
_ = summary(df_original, detailed=True, drb_round=True)

logger.info("\n\nLook at the imputes")
_ = df_list.pipe(summary, detailed=True, drb_round=True)

logger.info("\n\nLook at the imputes | var_gbm1 == 0")
_ = df_list.filter(~nw.col("var_gbm1")).pipe(summary, detailed=True, drb_round=True)

logger.info("\n\nLook at the imputes | var_gbm1 == 1")
_ = df_list.filter(nw.col("var_gbm1")).pipe(summary, detailed=True, drb_round=True)
