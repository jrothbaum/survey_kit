import numpy as np
import polars as pl

from survey_kit.imputation.srmi import SRMI
from survey_kit import logger, config


# %%
# A common real-world trap: "hours worked last week" is downstream of
# "employed" in the survey's own logic (you can't have hours worked if
# you're not employed), so it would be circular to use it - or anything
# derived from it - as a predictor when imputing an earnings flag that's
# really asking the same underlying question.

n_rows = 4_000
rng = np.random.default_rng(20260913)

x1 = rng.normal(size=n_rows)

employed = (rng.normal(size=n_rows) + 0.5 * x1 > 0.3).astype(int)
hours_worked = np.where(employed == 1, 35 + rng.normal(scale=5, size=n_rows), 0.0)
#   Derived from hours_worked - if it leaked in as a predictor for
#       "employed", the model would just be reading the answer off the
#       column it's supposed to help predict
full_time = (hours_worked >= 35).astype(int)

income = 40_000 + 8_000 * x1 + rng.normal(scale=5_000, size=n_rows)

df = pl.DataFrame(
    dict(
        person_id=range(n_rows),
        x1=x1,
        employed=employed,
        hours_worked=hours_worked,
        full_time=full_time,
        income=income,
    )
)
for col, share in [("employed", 0.15), ("income", 0.2)]:
    missing = rng.random(n_rows) < share
    df = df.with_columns(
        pl.when(pl.Series(missing)).then(None).otherwise(pl.col(col)).alias(col)
    )

# %%
logger.info("exclude={var: [...]} keeps a predictor out of just ONE variable's model")
srmi_exclude = SRMI.simple_model(
    df=df,
    index="person_id",
    exclude={"employed": ["hours_worked", "full_time"]},
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_simple_model_predictors_exclude",
        force_start=True,
    ),
)
for v in srmi_exclude.variables:
    logger.info(f"  {v.impute_var} predictors: {v.model}")

# %%
logger.info(
    "exclude_global=[...] keeps a predictor out of EVERY variable's model - useful "
    "for something like a survey weight or a raw text field that's never a real "
    "predictor for anything"
)
srmi_exclude_global = SRMI.simple_model(
    df=df,
    index="person_id",
    exclude_global=["hours_worked", "full_time"],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_simple_model_predictors_exclude_global",
        force_start=True,
    ),
)
for v in srmi_exclude_global.variables:
    logger.info(f"  {v.impute_var} predictors: {v.model}")

# %%
logger.info(
    "variables_to_impute=[...] gives an EXACT list - nothing else gets scanned for "
    "missingness, even if it has some. Handy when you only want to touch one or two "
    "columns in a much wider dataframe"
)
srmi_exact = SRMI.simple_model(
    df=df,
    index="person_id",
    variables_to_impute=["income"],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_simple_model_predictors_exact",
        force_start=True,
    ),
)
logger.info(f"Built variables: {[v.impute_var for v in srmi_exact.variables]}")

# %%
logger.info("Run the exclude= version end to end")
srmi_exclude.run()
