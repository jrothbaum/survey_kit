import numpy as np
import polars as pl

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.srmi import SRMI
from survey_kit import logger, config


# %%
# Draw data with one of each variable "shape": continuous, binary
# (auto-detected), and both kinds of categorical (which always need to be
# declared - there's no way to guess category order, or that a column of
# text is really a fixed set of categories, just by looking at the data)

n_rows = 4_000
rng = np.random.default_rng(20260913)

x1 = rng.normal(size=n_rows)
x2 = rng.normal(size=n_rows)

income = 40_000 + 8_000 * x1 + rng.normal(scale=5_000, size=n_rows)
employed = (rng.normal(size=n_rows) + 0.5 * x1 > 0.5).astype(int)

education_levels = ["less_than_hs", "hs_grad", "some_college", "bachelors_plus"]
education_score = 0.7 * x1 + rng.normal(scale=1.0, size=n_rows)
education_idx = np.clip(
    (education_score - education_score.min())
    / np.ptp(education_score)
    * len(education_levels),
    0,
    len(education_levels) - 0.001,
).astype(int)
education = [education_levels[i] for i in education_idx]

region_levels = ["northeast", "midwest", "south", "west"]
region = rng.choice(region_levels, size=n_rows)

df = pl.DataFrame(
    dict(
        person_id=range(n_rows),
        x1=x1,
        x2=x2,
        income=income,
        employed=employed,
        education=education,
        region=region,
    )
)

for col, share in [("income", 0.2), ("employed", 0.15), ("education", 0.2), ("region", 0.2)]:
    missing = rng.random(n_rows) < share
    df = df.with_columns(
        pl.when(pl.Series(missing)).then(None).otherwise(pl.col(col)).alias(col)
    )

# %%
logger.info("Declare the two categorical variables explicitly")
logger.info(
    "'education' is ORDERED - category order has to be given, there's no way to "
    "infer it from the data"
)
logger.info("'region' is UNORDERED - just needs to be flagged as categorical")

srmi = SRMI.simple_model(
    df=df,
    index="person_id",
    classes={
        "education": Variable.Class.ordered_categorical,
        "region": Variable.Class.unordered_categorical,
    },
    ordered_categories={"education": education_levels},
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_simple_model_variable_types",
        force_start=True,
    ),
)

for v in srmi.variables:
    logger.info(f"  {v.impute_var}: modeltype={v.modeltype.name}")

# %%
logger.info("income and employed weren't declared - auto-detected as continuous/binary")
logger.info(
    "Both default to LightGBM. Override the model for just one class with model= - "
    "here, use RandomForest for every continuous variable instead"
)

srmi_override = SRMI.simple_model(
    df=df,
    index="person_id",
    classes={
        "education": Variable.Class.ordered_categorical,
        "region": Variable.Class.unordered_categorical,
    },
    ordered_categories={"education": education_levels},
    model={Variable.Class.continuous: Variable.ModelType.RandomForest},
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_simple_model_variable_types_override",
        force_start=True,
    ),
)
for v in srmi_override.variables:
    logger.info(f"  {v.impute_var}: modeltype={v.modeltype.name}")

# %%
logger.info(
    "For full control over a model's own settings, pass (ModelType, parameters) "
    "instead of a bare ModelType - here, XGBoost with a smaller cv_folds setting"
)
custom_parameters = Parameters.XGBoost(cv_folds=3, error=Parameters.ErrorDraw.pmm)

srmi_custom = SRMI.simple_model(
    df=df,
    index="person_id",
    variables_to_impute=["income"],
    model={Variable.Class.continuous: (Variable.ModelType.XGBoost, custom_parameters)},
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_simple_model_variable_types_custom",
        force_start=True,
    ),
)
v_income = srmi_custom.variables[0]
logger.info(f"  income: modeltype={v_income.modeltype.name}, cv_folds={v_income.parameters['cv_folds']}")

# %%
logger.info("Run the first (fully auto + declared categoricals) version end to end")
srmi.run()
