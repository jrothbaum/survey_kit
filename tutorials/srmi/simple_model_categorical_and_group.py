import numpy as np
import polars as pl

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.srmi import SRMI
from survey_kit import logger, config


# %%
# Draw data with a categorical PREDICTOR (not itself being imputed) and a
# grouping variable (state) whose effect we want captured without treating
# it as a plain dummy-coded predictor

n_rows = 4_000
rng = np.random.default_rng(20260913)

x1 = rng.normal(size=n_rows)

industry_levels = ["retail", "healthcare", "manufacturing", "tech"]
industry_effect = {"retail": -2_000.0, "healthcare": 3_000.0, "manufacturing": 0.0, "tech": 6_000.0}
industry = rng.choice(industry_levels, size=n_rows)

state_levels = ["ca", "tx", "ny", "fl"]
state_effect = {"ca": 4_000.0, "tx": -1_000.0, "ny": 2_000.0, "fl": -2_000.0}
state = rng.choice(state_levels, size=n_rows)

income = (
    40_000
    + 8_000 * x1
    + np.array([industry_effect[i] for i in industry])
    + np.array([state_effect[s] for s in state])
    + rng.normal(scale=4_000, size=n_rows)
)

df = pl.DataFrame(
    dict(
        person_id=range(n_rows),
        x1=x1,
        industry=industry,
        state=state,
        income=income,
    )
)
missing_income = rng.random(n_rows) < 0.2
df = df.with_columns(
    pl.when(pl.Series(missing_income)).then(None).otherwise(pl.col("income")).alias("income")
)

# %%
logger.info(
    "'industry' is a string predictor for income, not something being imputed - "
    "declare it via categorical_predictors so the model treats it as a real "
    "category, not a scrambled numeric column"
)
logger.info(
    "'state' is a grouping variable - group_levels lets the model borrow strength "
    "across observations that share a state, without adding 'state' as an ordinary "
    "dummy-coded predictor"
)

srmi = SRMI.simple_model(
    df=df,
    index="person_id",
    categorical_predictors=["industry"],
    group_levels="state",
    #   group_levels only actually applies to models that support it -
    #       the continuous default (LightGBM) doesn't, so switch to
    #       RandomForest here to see it take effect
    model={Variable.Class.continuous: Variable.ModelType.RandomForest},
    replication=SRMI.Replication(n_implicates=2, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_simple_model_categorical_group",
        force_start=True,
    ),
)

v_income = srmi.variables[0]
logger.info(f"income predictors: {v_income.model}")
logger.info(f"income categorical_feature: {v_income.parameters.get('categorical_feature')}")
logger.info(f"income group_levels: {v_income.parameters.get('group_levels')}")

# %%
logger.info("Run it")
srmi.run()
