"""
Checks that Calibration.run() with aggregation="Sequential" actually works.

That branch of run() checked convergence against `self.Tolerance_Loop`, an
attribute the class never sets (only `self.tolerance` exists), so it should
raise AttributeError as soon as a Sequential run reaches convergence checking.
"""

from survey_kit.utilities.random import RandomData
from survey_kit.utilities.formula_builder import FormulaBuilder
from survey_kit.calibration.moment import Moment
from survey_kit.calibration.calibration import Calibration

n_rows = 1_000
df = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_1", 1, 10)
    .np_distribution("weight_0", "normal", loc=10, scale=1)
    .np_distribution("weight_1", "normal", loc=10, scale=1)
    .to_df()
    .lazy()
)

f = FormulaBuilder(df=df, constant=False)
f.continuous(columns=["v_1"])

m = Moment(df=df, formula=f.formula, weight="weight_0", index="index", rescale=True)

c = Calibration(
    df=df,
    moments=m,
    weight="weight_1",
    aggregation="Sequential",
)

diagnostics = c.run(min_obs=0, bounds=(0.000001, 1000))

assert diagnostics is not None, "Expected diagnostics from a Sequential run"
assert "converged" in diagnostics, "Expected a 'converged' key in diagnostics"

print("Sequential aggregation run completed without error.")
print(f"converged = {diagnostics['converged']}")
