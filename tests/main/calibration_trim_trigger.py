"""
Checks that Calibration.run() actually invokes Trim.trim_in_loop when trim.trim
is True and bounds= is passed (the documented "aebw with bounds" trim case).

The comment above the check says "Don't trim if aebw without separately passed
bounds", i.e. trim unless (aebw AND no bounds) -- by De Morgan's law that's
(method != "aebw") OR ("bounds" in additional_params), but the code used `and`
instead of `or`, and since Calibration.method is always "aebw" today, that made
the check permanently False regardless of the Trim object passed in.
"""

from survey_kit.utilities.random import RandomData
from survey_kit.utilities.formula_builder import FormulaBuilder
from survey_kit.calibration.moment import Moment
from survey_kit.calibration.calibration import Calibration
from survey_kit.calibration.trim import Trim

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

c = Calibration(df=df, moments=m, weight="weight_1")

# Spy on Trim.trim_in_loop to record whether/how often it actually gets called.
trim_calls = []
real_trim_in_loop = Trim.trim_in_loop


def _spy_trim_in_loop(self, c, iLoop, nLoops):
    trim_calls.append(iLoop)
    return real_trim_in_loop(self, c, iLoop, nLoops)


Trim.trim_in_loop = _spy_trim_in_loop

# Deliberately tiny bounds relative to the raw calibration weights, so trimming
# is certain to be needed if it's engaged at all.
trim_params = Trim(trim=True, min_val=0.99, max_val=1.01)

try:
    c.run(min_obs=0, bounds=(0.000001, 1000), trim=trim_params)
finally:
    Trim.trim_in_loop = real_trim_in_loop

assert len(trim_calls) > 0, (
    "Expected Trim.trim_in_loop to be called at least once when trim.trim=True "
    "and bounds= was passed to run() (method is 'aebw' with explicit bounds), "
    "but it was never invoked."
)

print(f"trim_in_loop was called {len(trim_calls)} time(s), as expected.")
