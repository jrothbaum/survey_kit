from __future__ import annotations

import polars as pl

from survey_kit.utilities.dataframe import summary, join_wrapper
from survey_kit.imputation.srmi import SRMI
from survey_kit import logger, config

from survey_kit.statistics.multiple_imputation import mi_ses_from_function
from survey_kit.statistics.calculator import StatCalculator
from survey_kit.statistics.replicates import Replicates
from survey_kit.utilities.dataframe import safe_height
from survey_kit.statistics.bootstrap import bayes_bootstrap

# %%
logger.info("Load the data from the SRMI tutorial")
path_model = f"{config.path_temp_files}/py_srmi_test_gbm"
srmi = SRMI.load(path_model)
df_implicates = srmi.df_implicates

# %%
logger.info("Make random weights for the bootstrap")
n_rows = safe_height(df_implicates[0])
n_replicates = 10

df_weights = bayes_bootstrap(
    n_rows=n_rows,
    n_draws=n_replicates + 1,
    seed=8345,
    prefix="weight_",
    initial_weight_index=0,
).with_row_index("index")

# %%
logger.info("What's the data look like")
logger.info("   The SRMI data - both implicates")
_ = df_implicates.pipe(summary)
logger.info("   The weights")
_ = summary(df_weights)

# %%
logger.info("To run an arbitrary stat, you need")
logger.info(
    "   any function that takes a dataframe and a weight and returns a dataframe"
)
logger.info("The function will be run for each boostrap/replicate weight")
logger.info("   Note: you may need to import packages within the function")
logger.info("   if executed in parallel (it will get pickled and reloaded)")


def run_regression(
    df: pl.LazyFrame | pl.DataFrame, y: str, X: list[str], weight: str = ""
) -> pl.LazyFrame | pl.DataFrame:
    import polars as pl
    from sklearn.linear_model import LinearRegression

    df = df.lazy().collect()
    model = LinearRegression()
    d_weight = {}
    if weight != "":
        d_weight["sample_weight"] = df[weight]

    model.fit(X=df.select(X), y=df.select(y), **d_weight)

    df_betas = pl.DataFrame(
        dict(
            Variable=df.select(X).lazy().collect_schema().names() + ["_Intercept_"],
            Beta=[
                float(vali)
                for vali in [float(coefi) for coefi in model.coef_[0]]
                + [float(model.intercept_[0])]
            ],
        )
    )

    return df_betas


# %%
logger.info("The betas from the regression (no weights, implicate 0)")
y = "var_gbm2"
X = ["var2", "var3", "var4", "var5"]
df_betas = run_regression(df=df_implicates[0], y=y, X=X)
logger.info(df_betas)


# %%
logger.info("Confirming how to run it on one implicate")
arguments_sc = dict(
    y=y,
    X=X,
)
replicates = Replicates(
    weight_stub="weight_", n_replicates=n_replicates, bootstrap=True
)
sc_compare = StatCalculator.from_function(
    run_regression,
    df=join_wrapper(df_implicates[0], df_weights, on=["index"], how="left"),
    estimate_ids=["Variable"],
    arguments=arguments_sc,
    replicates=replicates,
    display=False,
)
sc_compare.print()


# %%
logger.info("Now the full function")
logger.info("   Note the nested arguments...")
logger.info("       Maybe not the best API, but it does mean")
logger.info("       that you can run ANYTHING that takes in a")
logger.info("       dataframe and a weight and returns a dataframe of estiamtes")
logger.info("        and run it through the bootstrap + MI process")
arguments_sc = dict(
    y=y,
    X=X,
)
arguments_mi = dict(
    delegate=run_regression,
    arguments=arguments_sc,
    estimate_ids=["Variable"],
    replicates=replicates,
)

mi_results_seq = mi_ses_from_function(
    delegate=StatCalculator.from_function,
    df_implicates=df_implicates,
    path_srmi=path_model,
    df_noimputes=df_weights,
    index=["index"],
    arguments=arguments_mi,
    join_on=["Variable"],
    parallel=False,
)

mi_results_seq.print()

# %%
logger.info("For parallel, it's better to pass the path of the model")
logger.info("   To avoid unnecessary I/O")
logger.info("   (For sequential it doesn't really matter)")
mi_results = mi_ses_from_function(
    delegate=StatCalculator.from_function,
    # df_implicates=df_implicates,
    path_srmi=path_model,
    df_noimputes=df_weights,
    index=["index"],
    arguments=arguments_mi,
    join_on=["Variable"],
    parallel=True,
)

# %%
logger.info("The results")
mi_results.print()
mi_results_seq.print()

# %%
logger.info("Compare two sets of results")
d_comparison = mi_results.compare(mi_results_seq)
d_comparison["difference"].print()
d_comparison["ratio"].print()
