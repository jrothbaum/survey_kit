import os
import sys
from pathlib import Path
from time import perf_counter
from survey_kit.utilities.random import RandomData
from survey_kit.utilities.formula_builder import FormulaBuilder
from survey_kit.calibration.moment import Moment
from survey_kit.calibration.calibration import Calibration

import narwhals as nw

from survey_kit import logger, config

path_scratch = config.path_temp_files


n_rows = 100_000
df_m = (
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

df_c = (
    RandomData(n_rows=n_rows, seed=894654)
    .index("index")
    .integer("v_1", 1, 10)
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

f = FormulaBuilder(df=df_m, constant=False)
f.continuous(columns=["v_1", "v_f_continuous_*", "v_f_p2_*"])
#   f.simple_interaction(columns=["v_1","v_f_continuous_0"])

f_by = FormulaBuilder(df=df_m, constant=False)
f_by.simple_interaction(columns=["year", "month"])
print(f_by.formula)

# df_c = nw.from_native(df_c).lazy().collect().lazy_backend(NarwhalsType(backend="pyarrow")).to_native()
# df_m = nw.from_native(df_m).lazy().collect().lazy_backend(NarwhalsType(backend="pyarrow")).to_native()
start_moment = perf_counter()
m = Moment(
    df=df_m,
    formula=f.formula,
    weight="weight_0",
    index="index",
    #    by=["year"],
    rescale=True,
    #    by=f_by.formula
)
elapsed_moment = perf_counter() - start_moment

start_calibration_setup = perf_counter()
c = Calibration(
    df=df_c,
    moments=m,
    weight="weight_1",
)
elapsed_calibration_setup = perf_counter() - start_calibration_setup

c.save(f"{path_scratch}/calibration")

start_calibration_run = perf_counter()
c.run(min_obs=5, bounds=(0.000001, 1000))
print(type(c.df))
print(type(c.diagnostics_out["diagnostics"]))
elapsed_calibration_run = perf_counter() - start_calibration_run

logger.info(f"Moment:              {elapsed_moment:0.4f}")
logger.info(f"Calibration (setup): {elapsed_calibration_setup:0.4f}")
logger.info(f"Calibration (run):   {elapsed_calibration_run:0.4f}")


logger.info("Validation that Estimates ~= Targets ")
df_check = (
    nw.from_native(c.diagnostics_out["diagnostics"])
    .select(
        ((nw.col("Estimates") - nw.col("Targets")).abs() / nw.col("Targets") <= 0.00001)
    )
    .lazy()
    .collect()
    .to_native()
)
logger.info(df_check)
assert nw.from_native(df_check).select(nw.all().all()).item(0, 0)


c = Calibration.load(f"{path_scratch}/calibration", backend="duckdb")
c.run()
print(type(c.df))
print(type(c.diagnostics_out["diagnostics"]))
