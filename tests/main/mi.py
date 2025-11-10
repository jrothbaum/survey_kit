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


path_model = f"{config.path_temp_files}/py_srmi_test_gbm"
srmi = SRMI.load(path_model)
df_implicates = srmi.df_implicates

# df_implicates._df_list = [dfi.collect().to_pandas() for dfi in df_implicates._df_list]
# df_implicates._df_list = [nw.from_native(dfi.collect().to_arrow()).lazy(backend="duckdb").to_native() for dfi in df_implicates._df_list]

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

_ = df_implicates.pipe(summary)
_ = summary(df_weights)

df_weights = df_weights.to_pandas()
stats = Statistics(
    stats=["mean"],
    columns="var_*",
)
replicates = Replicates(weight_stub="weight_", n_replicates=n_replicates)

arguments = dict(statistics=stats, replicates=replicates)


mi_results_seq = mi_ses_from_function(
    delegate=StatCalculator,
    # path_srmi=path_model,
    df_implicates=df_implicates,
    df_noimputes=df_weights,
    index=["index"],
    arguments=arguments,
    join_on=["Variable"],
    parallel=False,
)

mi_results = mi_ses_from_function(
    delegate=StatCalculator,
    path_srmi=path_model,
    # df_implicates=df_implicates,
    df_noimputes=df_weights,
    index=["index"],
    arguments=arguments,
    join_on=["Variable"],
    parallel=True,
)

# mi_results.print()


mi_results.print()
mi_results_seq.print()

d_comparison = mi_results.compare(mi_results_seq)
d_comparison["difference"].print()
d_comparison["ratio"].print()
