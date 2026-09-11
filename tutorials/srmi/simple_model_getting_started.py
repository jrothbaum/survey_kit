import numpy as np
import polars as pl

from survey_kit.imputation.srmi import SRMI
from survey_kit.utilities.dataframe import summary
from survey_kit import logger, config


# %%
# Draw some random data with a few variables missing

n_rows = 5_000
rng = np.random.default_rng(20260913)

x1 = rng.normal(size=n_rows)
x2 = rng.normal(size=n_rows)

income_latent = 40_000 + 8_000 * x1 - 3_000 * x2 + rng.normal(scale=5_000, size=n_rows)
has_disability = (rng.normal(size=n_rows) + 0.5 * x1 > 1.0).astype(int)

df = pl.DataFrame(
    dict(
        person_id=range(n_rows),
        age=rng.integers(18, 90, size=n_rows),
        education_years=rng.integers(8, 20, size=n_rows),
        income=income_latent,
        has_disability=has_disability,
    )
)

#   Punch some holes in the two variables we want imputed
missing_income = rng.random(n_rows) < 0.2
missing_disability = rng.random(n_rows) < 0.15
df = df.with_columns(
    [
        pl.when(pl.Series(missing_income)).then(None).otherwise(pl.col("income")).alias(
            "income"
        ),
        pl.when(pl.Series(missing_disability))
        .then(None)
        .otherwise(pl.col("has_disability"))
        .alias("has_disability"),
    ]
)

logger.info(f"Missing income:      {missing_income.sum()} of {n_rows} rows")
logger.info(f"Missing disability:  {missing_disability.sum()} of {n_rows} rows")

# %%
logger.info("That's it - point SRMI.simple_model() at the dataframe and an id column")
logger.info(
    "It finds every column with missing values, figures out whether each one is "
    "binary or continuous, and picks a sensible model for each - no Variable objects "
    "to build by hand."
)

srmi = SRMI.simple_model(
    df=df,
    index="person_id",
    replication=SRMI.Replication(n_implicates=3, n_iterations=3),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_simple_model_getting_started",
        force_start=True,
    ),
)

# %%
logger.info("Inspect what it decided before running anything")
for v in srmi.variables:
    logger.info(f"  {v.impute_var}: modeltype={v.modeltype.name}, predictors={v.model}")

# %%
logger.info("Now actually run the imputation")
srmi.run()

# %%
logger.info("Look at one completed implicate")
_ = srmi.df_implicates.pipe(summary)
