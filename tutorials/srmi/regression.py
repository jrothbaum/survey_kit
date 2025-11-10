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

from survey_kit import logger, config
from survey_kit.utilities.dataframe import summary, columns_from_list
from survey_kit.utilities.formula_builder import FormulaBuilder


path = Path(config.code_root)
sys.path.append(os.path.normpath(path.parent.parent / "tests"))
from scratch import path_scratch


config.data_root = path_scratch(temp_file_suffix=False)


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
    .np_distribution("epsilon_reg1", "normal", scale=5)
    .np_distribution("epsilon_reg2", "normal", scale=5)
    .float("missing_reg1", 0, 1)
    .float("missing_reg2", 0, 1)
    .to_df()
)


#   Convenience references to them for creating dependent variables
c_var2 = pl.col("var2")
c_var3 = pl.col("var3")
c_var4 = pl.col("var4")
c_var5 = pl.col("var5")

c_e_reg1 = pl.col("epsilon_reg1")
c_e_reg2 = pl.col("epsilon_reg2")


#   Convenience references to them for creating dependent variables
c_var2 = pl.col("var2")
c_var3 = pl.col("var3")
c_var4 = pl.col("var4")
c_var5 = pl.col("var5")


logger.info("var_reg1 is binary and conditional on other variables")
c_reg1 = ((c_var2 * 2 - c_var3 * 3 * c_var5 + c_e_reg1) > 0).alias("var_reg1")

logger.info("var_reg2 is != 0 only if var_reg1 == True")
c_reg2 = (
    pl.when(pl.col("var_reg1"))
    .then((c_var2 * 1.5 - c_var3 * 1 * c_var4 + c_e_reg2))
    .otherwise(pl.lit(0))
    .alias("var_reg2")
)
#   Create a bunch of variables that are functions of the variables created above
df = (
    df.with_columns(c_reg1)
    .with_columns(c_reg2)
    .drop(columns_from_list(df=df, columns="epsilon*"))
    .with_row_index(name="_row_index_")
)

df_original = df

#   Set variables to missing according to the uniform random variables missing_
clear_missing = []
for prefixi in ["reg"]:
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


# %%
logger.info(
    "Define the regression model (intentionally include some extraneous variables"
)

f_model = FormulaBuilder(df=df)
f_model.formula_with_varnames_in_brackets(
    "~1+{var_*}+var2+var4+var4*var3*C(var5)+{unrelated_*}+{repeat_*}"
)
logger.info(f_model.formula)


# %%
# Set up the variable to be imputed
vars_impute = []

logger.info("Impute the boolean variable (var_reg1)")
logger.info("   to the default setup for predicted mean matching")
logger.info("   using logit regression")
v_reg1 = Variable(
    impute_var="var_reg1",
    modeltype=Variable.ModelType.pmm,
    model=f_model.formula,
    parameters=Parameters.Regression(model=Parameters.RegressionModel.Logit),
)
logger.info("Add the variable to the list to be imputed")
vars_impute.append(v_reg1)

logger.info("Impute the continuous variable (var_reg2) ")
logger.info("   conditional on var_reg1, using narwhals (nw.col('var_reg1'))")
logger.info("   by setting the model type")
logger.info("   and the formula")
logger.info("   as well as a post-processing edit to set var_reg2=0 when var_reg1==0")
v_reg2 = Variable(
    impute_var="var_reg2",
    Where=nw.col("var_reg1"),
    modeltype=Variable.ModelType.pmm,
    model=f_model.formula,
    #   Default parameters
    parameters=Parameters.Regression(),
    postFunctions=(
        nw.when(nw.col("var_reg1"))
        .then(nw.col("var_reg2"))
        .otherwise(nw.lit(0))
        .alias("var_reg2")
    ),
)

vars_impute.append(v_reg2)


# %%
logger.info("Set up the imputation")
logger.info("Add LASSO selection before each imputation")
srmi = SRMI(
    df=df,
    variables=vars_impute,
    n_implicates=2,
    n_iterations=2,
    parallel=False,
    selection=Selection(method=Selection.Method.LASSO),
    modeltype=Variable.ModelType.pmm,
    model=f_model.formula,
    bayesian_bootstrap=True,
    path_model=f"{config.path_temp_files}/py_srmi_test_regression",
    force_start=True,
)

# %%
logger.info("Run it")
srmi.run()


# %%
logger.info("Get the results")
_ = df_list = srmi.df_implicates

# %%
logger.info("\n\nLook at the original")
_ = summary(df_original)

logger.info("\n\nLook at the imputes")
_ = df_list.pipe(summary)

logger.info("\n\nLook at the imputes | var_reg1 == 0")
_ = df_list.filter(~nw.col("var_reg1")).pipe(summary)

logger.info("\n\nLook at the imputes | var_reg1 == 1")
_ = df_list.filter(nw.col("var_reg1")).pipe(summary)
