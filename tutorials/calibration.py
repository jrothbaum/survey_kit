from pathlib import Path
from survey_kit.utilities.random import RandomData
from survey_kit.utilities.formula_builder import FormulaBuilder
from survey_kit.calibration.moment import Moment
from survey_kit.calibration.calibration import Calibration
from survey_kit.utilities.dataframe import summary
import narwhals as nw
from survey_kit import logger

# %%
logger.info("Generating data for weighting")
n_rows = 100_000
df_population = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_1", 1, 10)
    .np_distribution("v_f_continuous_0", "normal", loc=10, scale=2)
    .np_distribution("v_f_continuous_1", "normal", loc=10, scale=2)
    .np_distribution("v_f_continuous_2", "normal", loc=10, scale=2)
    .float("v_extra", -1, 2)
    .np_distribution("weight_0", "normal", loc=10, scale=1)
    .np_distribution("weight_1", "normal", loc=10, scale=1)
    .integer("year", 2016, 2021)
    .integer("month", 1, 12)
    .to_df()
    .lazy()
)

df_treatment = (
    RandomData(n_rows=n_rows, seed=894654)
    .index("index")
    .integer("v_1", 1, 10)
    #   Intentionally set the loc/scale as different than above
    .np_distribution("v_f_continuous_0", "normal", loc=11, scale=4)
    .np_distribution("v_f_continuous_1", "normal", loc=11, scale=4)
    .np_distribution("v_f_continuous_2", "normal", loc=11, scale=4)
    .float("v_extra", -1, 2)
    .np_distribution("weight_0", "normal", loc=10, scale=1)
    .np_distribution("weight_1", "normal", loc=10, scale=1)
    .integer("year", 2016, 2021)
    .integer("month", 1, 12)
    .to_df()
    .lazy()
)

# print(df.describe())

# %%
logger.info("Weighting 'function'")
f = FormulaBuilder(df=df_population, constant=False)
f.continuous(columns=["v_1", "v_f_continuous_*", "v_f_p2_*"])
#   f.simple_interaction(columns=["v_1","v_f_continuous_0"])

logger.info("Define the target moments that the weighting will match")
logger.info("   This can be a dataset or a single row of pop controls")
m = Moment(
    df=df_population,
    formula=f.formula,
    weight="weight_0",
    index="index",
    #    by=["year"],
    rescale=True,
)

logger.info("You can save/reload moments if you want")
# m.save("/my/path/moment")
# m_loaded = Moment.load("/my/path/moment")

# %%
#   Calibrate the data in df_treatment to the moment above
c = Calibration(
    df=df_treatment, moments=m, weight="weight_1", final_weight="weight_final"
)

c.run(
    #   Drop a moment if there are too few observations
    min_obs=5,
    # If it fails to converge, set bounds on the weights
    #   final weights = (base*ratio) where the bounds are on the ratio
    #   for "best possible" weights
    bounds=(0.001, 1000),
)

#   Merge the final weights back on the treatment data
df_treatment = c.get_final_weights(df_treatment)

# %%
logger.info("'Population' estimates")
_ = summary(df_population, weight="weight_0")

# %%
logger.info("\n\n'Treatment', original weights")
_ = summary(df_treatment, weight="weight_1")

# %%
logger.info("\n\n'Treatment', calibrated")
_ = summary(df_treatment, weight="weight_final")
