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
from survey_kit.orchestration.config import Config

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
    .np_distribution("epsilon_hd1", "normal", scale=5)
    .np_distribution("epsilon_hd2", "normal", scale=5)
    .float("missing_hd1", 0, 1)
    .float("missing_hd2", 0, 1)
    .to_df()
)


#   Convenience references to them for creating dependent variables
c_var2 = pl.col("var2")
c_var3 = pl.col("var3")
c_var4 = pl.col("var4")
c_var5 = pl.col("var5")

c_e_hd1 = pl.col("epsilon_hd1")
c_e_hd2 = pl.col("epsilon_hd2")


logger.info("var_hd1 is binary and conditional on other variables")
c_hd1 = ((c_var2 * 2 - c_var3 * 3 * c_var5 + c_e_hd1) > 0).alias("var_hd1")

logger.info("var_hd2 is != 0 only if var_hd1 == True")
c_hd2 = (
    pl.when(pl.col("var_hd1"))
    .then((c_var2 * 1.5 - c_var3 * 1 * c_var4 + c_e_hd2))
    .otherwise(pl.lit(0))
    .alias("var_hd2")
)
#   Create a bunch of variables that are functions of the variables created above
df = (
    df.with_columns(c_hd1)
    .with_columns(c_hd2)
    .drop(columns_from_list(df=df, columns="epsilon*"))
    .with_row_index(name="_row_index_")
)
df_original = df

#   Set variables to missing according to the uniform random variables missing_
clear_missing = []
for prefixi in ["hd"]:
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
parameters_hd1 = Parameters.HotDeck(
    #   model_list - a list of variables to match
    #       donors and recipients
    model_list=["var2", "var3", "var5"],
    #   Donate anything other than the variable
    #       (i.e. donate together)
    #       In this case, it's redundant and does nothing...
    donate_list=["var_hd1"],
)

# %%
# Set up the variable to be imputed

logger.info("Impute the boolean variable (var_hd1)")
logger.info("   by setting the model type (a stat match)")
logger.info("   and the list of match variables")
v_hd1 = Variable(
    impute_var="var_hd1",
    modeltype=Variable.ModelType.StatMatch,
    parameters=Parameters.HotDeck(model_list=["var2", "var3", "var5"]),
)

logger.info("Add the variable to the list to be imputed")
vars_impute.append(v_hd1)


logger.info("Impute the continuous variable (var_hd2) ")
logger.info("   conditional on var_hd1, using narwhals (nw.col('var_hd1'))")
logger.info("   by setting the model type (a hot deck)")
logger.info("   and the list of match variables")
logger.info("   as well as a post-processing edit to set var_hd2=0 when var_hd1==0")

v_hd2 = Variable(
    impute_var="var_hd2",
    sample=Variable.Sample(Where=nw.col("var_hd1")),
    By=["year", "month"],
    modeltype=Variable.ModelType.HotDeck,
    parameters=Parameters.HotDeck(model_list=["var2", "var3", "var5"]),
    transforms=Variable.Transforms(
        post=(
            nw.when(nw.col("var_hd1"))
            .then(nw.col("var_hd2"))
            .otherwise(nw.lit(0))
            .alias("var_hd2")
        )
    ),
)
vars_impute.append(v_hd2)


# %%
logger.info("Set up the imputation")
srmi = SRMI(
    df=df,
    variables=vars_impute,
    replication=SRMI.Replication(n_implicates=2, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False, testing=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/py_srmi_test_hd", force_start=True
    ),
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

logger.info("\n\nLook at the imputes | var_hd1 == 0")
_ = df_list.filter(~nw.col("var_hd1")).pipe(summary)

logger.info("\n\nLook at the imputes | var_hd1 == 1")
_ = df_list.filter(nw.col("var_hd1")).pipe(summary)


# %%
logger.info(
    "sequential_drop and the empty-match-key fallback - what happens when "
    "no donor shares a recipient's match key at all"
)
logger.info(
    "   sequential_drop=True (HotDeck's own default) builds a cascade that "
    "progressively drops the LAST match variable - ['var2','var3','var5'] "
    "-> ['var2','var3'] -> ['var2'] - trying each level in turn until a "
    "recipient finds a donor. Whoever is STILL unmatched after even the "
    "shortest level gets matched fully at random instead, with a logged "
    "warning, rather than being left unmatched forever - this applies "
    "unconditionally (even with sequential_drop=False, or an explicitly "
    "empty model_list=[]), not just at the end of a cascade."
)

v_hd_fallback = Variable(
    impute_var="var_hd1",
    modeltype=Variable.ModelType.HotDeck,
    #   sequential_drop=True is the default - spelled out here for clarity
    parameters=Parameters.HotDeck(model_list=["var2", "var3", "var5"], sequential_drop=True),
)
srmi_fallback = SRMI(
    df=df,
    variables=[v_hd_fallback],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/py_srmi_test_hd_fallback",
        force_start=True,
    ),
)
srmi_fallback.run()
logger.info(
    "   Look for '     Matching on: [] (fully random fallback)' above (if "
    "it appears) - that's the guaranteed-to-succeed last resort actually "
    "being reached for whichever recipients no real match level could "
    "cover."
)
