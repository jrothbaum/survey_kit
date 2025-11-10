import sys
import os
from pathlib import Path
import narwhals as nw
import polars as pl

from survey_kit.utilities.dataframe import summary
from survey_kit.imputation.srmi import SRMI
from survey_kit import logger, config

from survey_kit.statistics.multiple_imputation import (
    MultipleImputation,
    mi_ses_from_function,
)
from survey_kit.statistics.calculator import StatCalculator
from survey_kit.statistics.statistics import Statistics
from survey_kit.statistics.replicates import Replicates
from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import safe_height
from survey_kit.statistics.bootstrap import bayes_bootstrap
from survey_kit.utilities.random import set_seed, generate_seed


# %%
logger.info("Load the SRMI results from that tutorial")
path_model = f"{config.path_temp_files}/py_srmi_test_gbm"
srmi = SRMI.load(path_model)
df_implicates = srmi.df_implicates

# %%
logger.info("Draw bootstrap weights")
set_seed(8345)
n_rows = safe_height(df_implicates[0])
n_replicates = 10

df_weights = bayes_bootstrap(
    n_rows=n_rows,
    n_draws=n_replicates + 1,
    seed=generate_seed(),
    prefix="weight_",
    initial_weight_index=0,
).with_row_index("index")


# %%
logger.info("What's the data look like")
_ = df_implicates.pipe(summary)
_ = summary(df_weights)


# %%
logger.info("What statistics do I want:")
logger.info("   In this case, the mean of all variables that start with var_")
stats = Statistics(
    stats=["mean"],
    columns="var_*",
)

# %%
logger.info(
    "Define the 'replicate' object, which tell is what the weight variables are"
)
replicates = Replicates(weight_stub="weight_", n_replicates=n_replicates)


# %%
logger.info("Arguments that are getting passed to StatCalculator at each run")
arguments = dict(statistics=stats, replicates=replicates)


# %%
logger.info("Get the multiple imputation standard errofs by calling StatCalculator")
logger.info("   for each implicate and each replicate factor")
logger.info("   If you had 100 bootstrap weights and 5 imputation draws (implicates)")
logger.info("   the StatCalculator calculation would run 5*100=500 times")
mi_results_seq = mi_ses_from_function(
    delegate=StatCalculator,
    df_implicates=df_implicates,
    df_noimputes=df_weights,
    index=["index"],
    arguments=arguments,
    join_on=["Variable"],
    parallel=False,
)


# %%
logger.info(
    "Do it again, but run each implicate in it's own process and collect the results"
)
logger.info("   This will split up the job and ")
logger.info(
    "   use all your cpus (or what you set in config.cpus), dividing them up among the jobs"
)
mi_results = mi_ses_from_function(
    delegate=StatCalculator,
    df_implicates=df_implicates,
    df_noimputes=df_weights,
    index=["index"],
    arguments=arguments,
    join_on=["Variable"],
    parallel=True,
)

# %%
logger.info("The sequential (non-parallel) job results")
mi_results.print()

# %%
logger.info("The parallel job results")
mi_results_seq.print()

# %%
logger.info("Save them for later")
mi_results.save(f"{config.path_temp_files}/mi_sequential")
mi_results_seq.save(f"{config.path_temp_files}/mi_parallel")

del mi_results
del mi_results_seq

# %%
logger.info("Load them back up")
mi_results = MultipleImputation.load(f"{config.path_temp_files}/mi_sequential")
mi_results_seq = MultipleImputation.load(f"{config.path_temp_files}/mi_parallel")


# %%
logger.info("Compare them")
d_comparison = mi_results.compare(mi_results_seq)
logger.info("   Print the mi_results_seq - mi_results")
d_comparison["difference"].print()
logger.info("   Print the (mi_results_seq-mi_results)/mi_results")
d_comparison["ratio"].print()
