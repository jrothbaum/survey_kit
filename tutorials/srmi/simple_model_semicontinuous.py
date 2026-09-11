import numpy as np
import narwhals as nw
import polars as pl

from survey_kit.imputation.srmi import SRMI
from survey_kit.utilities.dataframe import summary
from survey_kit import logger, config


# %%
# A semicontinuous ("two-part"/hurdle) variable: most people have $0 of
# self-employment income (they don't have any), and everyone else has some
# genuinely continuous positive amount. Modeling that as one plain
# continuous variable fights itself - the "is it zero" and "how much, given
# it's not zero" questions are really two different models.

n_rows = 4_000
rng = np.random.default_rng(20260913)

x1 = rng.normal(size=n_rows)

has_self_employment_income = (rng.normal(size=n_rows) + 0.4 * x1 > 0.8).astype(int)
self_employment_income = np.where(
    has_self_employment_income == 1,
    15_000 + 6_000 * x1 + rng.normal(scale=4_000, size=n_rows),
    0.0,
)

df = pl.DataFrame(
    dict(
        person_id=range(n_rows),
        x1=x1,
        has_self_employment_income=has_self_employment_income,
        self_employment_income=self_employment_income,
    )
)

#   Only the dollar amount has missingness here - the yes/no flag is fully
#       observed, a common real-world pattern (people usually answer "do
#       you have this income source" even when they skip the amount)
missing_amount = rng.random(n_rows) < 0.2
df = df.with_columns(
    pl.when(pl.Series(missing_amount))
    .then(None)
    .otherwise(pl.col("self_employment_income"))
    .alias("self_employment_income")
)

# %%
logger.info(
    "yn_pairs={value_var: yn_var} routes self_employment_income through a proper "
    "two-part model: has_self_employment_income gates it, and only the "
    "has_self_employment_income==True population gets a real continuous model fit"
)

srmi = SRMI.simple_model(
    df=df,
    index="person_id",
    yn_pairs={"self_employment_income": "has_self_employment_income"},
    replication=SRMI.Replication(n_implicates=2, n_iterations=2),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_simple_model_semicontinuous",
        force_start=True,
    ),
)

logger.info(f"Built variables: {[v.impute_var for v in srmi.variables]}")

# %%
logger.info("Run it")
srmi.run()

# %%
logger.info(
    "Every imputed value respects the hurdle: 0 whenever "
    "has_self_employment_income is False, a real positive draw otherwise"
)
#   has_self_employment_income is an int-coded (0/1) column, not a real
#       boolean - use == 0 rather than ~, which does bitwise (not logical)
#       negation on a non-boolean column and would match every row
_ = (
    srmi.df_implicates.filter(nw.col("has_self_employment_income") == 0)
    .select("self_employment_income")
    .pipe(summary)
)
