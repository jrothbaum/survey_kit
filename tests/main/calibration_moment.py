import os
import sys
import narwhals as nw
from pathlib import Path

from survey_kit.utilities.random import RandomData
from survey_kit.utilities.formula_builder import FormulaBuilder
from survey_kit.calibration.moment import Moment
from survey_kit import config

path_scratch = config.path_temp_files


n_rows = 1_000
df = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_1", 0, 10)
    .float("v_f_continuous_0", -1, 2)
    .float("v_f_continuous_1", -1, 2)
    .float("v_f_continuous_2", -1, 2)
    .float("v_extra", -1, 2)
    .np_distribution("weight_0", "normal", loc=1, scale=1)
    .np_distribution("weight_1", "normal", loc=1, scale=1)
    .integer("year", 2016, 2021)
    .integer("month", 1, 12)
    .to_df()
    .lazy()
)

# print(df.describe())

f = FormulaBuilder(df=df, constant=False)
f.continuous(columns=["v_1", "v_f_continuous_*", "v_f_p2_*"])
f.simple_interaction(columns=["v_1", "v_f_continuous_0"])

f_by = FormulaBuilder(df=df, constant=False)
f_by.simple_interaction(columns=["year", "month"])
print(f_by.formula)
m = Moment(
    df=df,
    formula=f.formula,
    weight="weight_0",
    index="index",
    by=["year"],
    rescale=False,
    #    by=f_by.formula
)


m_pandas = Moment(
    df=df.collect().to_pandas(),
    formula=f.formula,
    weight="weight_0",
    index="index",
    by=["year"],
    rescale=False,
    #    by=f_by.formula
)

m_arrow = Moment(
    df=df.collect().to_arrow(),
    formula=f.formula,
    weight="weight_0",
    index="index",
    by=["year"],
    rescale=False,
    equalize_by=["year"],
    #    by=f_by.formula
)

# print(m.df.lazy().collect())
# print(m_pandas.df)

# print(m.model_matrix.lazy().collect())
# print(m_pandas.model_matrix)


m.save(f"{path_scratch}/moment")

m_new = Moment.load(
    f"{path_scratch}/moment",
    delete=False,
    # session=DuckDBSession(),
    backend="duckdb",
)

for i in range(len(m.sub_moments)):
    print(nw.from_native(m_new.sub_moments[i].targets).lazy().collect().to_native())
    print(m_pandas.sub_moments[i].targets)

    if m.sub_moments[i].scale is not None:
        print(nw.from_native(m_new.sub_moments[i]).scale.lazy().collect().to_native())
        print(m_pandas.sub_moments[i].scale)
