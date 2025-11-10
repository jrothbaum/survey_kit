from __future__ import annotations

import polars as pl
from sklearn.linear_model import LinearRegression

from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import summary, columns_from_list
from survey_kit.statistics.calculator import StatCalculator
from survey_kit.statistics.replicates import Replicates
from survey_kit.statistics.bootstrap import bayes_bootstrap
from survey_kit.utilities.random import set_seed, generate_seed

from survey_kit import logger


# %%
# Draw some random data
def gen_random_table(n_rows: int, n_replicates: int, seed: int):
    set_seed(seed)
    df = (
        RandomData(n_rows=n_rows, seed=generate_seed())
        .index("index")
        .integer("v_1", 0, 10)
        .float("v_2", 0, 100)
        .float("v_3", 0, 100)
        .float("v_4", 0, 100)
        .np_distribution("e", "normal", loc=0, scale=100)
    )

    df = df.to_df()

    c_v1 = pl.col("v_1")
    c_v2 = pl.col("v_2")
    c_v3 = pl.col("v_3")
    c_v4 = pl.col("v_4")
    e = pl.col("e")
    df = df.with_columns(y=10 * c_v1 - 5 * c_v2 + 100 * c_v3 + c_v4 + e)

    df = pl.concat(
        [
            df,
            bayes_bootstrap(
                n_rows=n_rows,
                n_draws=n_replicates + 1,
                seed=generate_seed(),
                initial_weight_index=0,
                prefix="weight_",
            ),
        ],
        how="horizontal",
    ).lazy()

    return df


# %%
n_rows = 1_000
n_replicates = 10

logger.info(
    "Create two datasets to compare, with replicate weights (or bootstrap weights)"
)
df = gen_random_table(n_rows, n_replicates, seed=1230)
df_compare = gen_random_table(n_rows, n_replicates, seed=3254)

logger.info("The data")
_ = summary(df)


# %%
logger.info("To run an arbitrary stat, you need")
logger.info(
    "   any function that takes a dataframe and a weight and returns a dataframe"
)
logger.info("The function will be run for each boostrap/replicate weight")


def run_regression(
    df: pl.LazyFrame | pl.DataFrame, y: str, X: list[str], weight: str
) -> pl.LazyFrame | pl.DataFrame:
    df = df.lazy().collect()
    model = LinearRegression()
    model.fit(X=df.select(X), y=df.select(y), sample_weight=df[weight])

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
logger.info("The betas from the regression")
df_betas = run_regression(
    df=df, y="y", X=columns_from_list(df, ["v_*"]), weight="weight_0"
)
logger.info(df_betas)


logger.info("Call StatCalculator.from_function")
logger.info("   Pass in the data")
logger.info("   'estimate_ids': the name of the column(s) that index the estimates")
logger.info("   'arguments':    stable (non df and non-weight) arguments")
logger.info("   'replicates':   weights to loop over")
sc = StatCalculator.from_function(
    run_regression,
    df=df,
    estimate_ids=["Variable"],
    arguments=dict(y="y", X=columns_from_list(df, ["v_*"])),
    replicates=Replicates(
        weight_stub="weight_", n_replicates=n_replicates, bootstrap=True
    ),
    display=False,
)


logger.info("The result")
sc.print()


# %%
logger.info(
    "You can compare it like any other stat, run the same regression on another sample"
)
logger.info("   Run the regression on a different random draw")
sc_compare = StatCalculator.from_function(
    run_regression,
    df=df_compare,
    estimate_ids=["Variable"],
    arguments=dict(y="y", X=columns_from_list(df, ["v_*"])),
    replicates=Replicates(
        weight_stub="weight_", n_replicates=n_replicates, bootstrap=True
    ),
    display=False,
)

# %%
logger.info("   And compare the results")
d_comp = sc.compare(sc_compare)
