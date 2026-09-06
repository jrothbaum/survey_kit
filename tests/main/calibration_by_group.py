"""
Checks that Calibration's weight-side sub-moments are actually restricted to
their own by-group.

process_single_moment recurses into each sub_moment by passing down the
parent's full (unfiltered) dataframe, never applying the sub_moment's own
by_where_expressions. That means a year-specific column (and its "_in"
group indicator) gets computed from every row in the whole dataset instead
of just that year's rows -- turning a should-vary-by-row indicator into a
flat constant, which bakes an unsatisfiable constraint into the combined
moment (a constant column's weighted mean can never match a different
target value, no matter how the weights are adjusted).
"""

import narwhals as nw
import polars as pl

from survey_kit.calibration.calibration import Calibration
from survey_kit.calibration.moment import Moment
from survey_kit.utilities.formula_builder import FormulaBuilder

n_per_year = 300
years = [2020, 2021, 2022]

rows = []
idx = 0
for y in years:
    for i in range(n_per_year):
        rows.append({"index": idx, "year": y, "v": 1.0 + (i % 5) * 0.1, "w": 1.0})
        idx += 1

df = pl.DataFrame(rows).lazy()

f = FormulaBuilder(df=df, constant=False)
f.continuous(columns=["v"])

m = Moment(df=df, formula=f.formula, weight="w", index="index", by=["year"], rescale=True)

c = Calibration(df=df, moments=m, weight="w", index=["index"])
c.combine_moments(all=True, sub_moments=True)

mm = (
    nw.from_native(c.moments[0].model_matrix)
    .lazy()
    .collect()
    .to_native()
    .join(df.collect(), on="index")
)

for y in years:
    in_col = f"m0_year=={y}:_in"
    v_col = f"m0_year=={y}:v"

    assert in_col in mm.columns, f"Expected column {in_col} in combined model matrix"

    in_group = mm.filter(pl.col("year") == y)
    out_group = mm.filter(pl.col("year") != y)

    #   Rows in this year's group should carry the real (nonzero) indicator
    #   value; rows outside it should be zero-filled, not the same constant.
    assert (in_group[in_col] != 0).all(), (
        f"{in_col}: expected all in-group rows to be nonzero, "
        f"got {in_group[in_col].to_list()[:5]}"
    )
    assert (out_group[in_col] == 0).all(), (
        f"{in_col}: expected all out-of-group rows to be 0, but got values "
        f"like {out_group[in_col].drop_nulls().to_list()[:5]} -- this is the bug: "
        "the sub-moment's model matrix was built from the whole dataset instead "
        "of being restricted to its own year"
    )

    #   Same check for the year-specific variable column: only that year's
    #   rows should carry real "v" values, everyone else should be 0.
    assert (out_group[v_col] == 0).all(), (
        f"{v_col}: expected all out-of-group rows to be 0, got values like "
        f"{out_group[v_col].drop_nulls().to_list()[:5]}"
    )

print("All Calibration by-group restriction checks passed.")
