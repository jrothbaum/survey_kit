"""
Checks that Trim.trim_in_loop actually trims weights that are out of bounds,
using the default ignore_n=0.

trim_in_loop only reads/writes c.df and c.final_weight, so a minimal stand-in
for Calibration is used instead of building a real one.
"""

import narwhals as nw
import polars as pl

from survey_kit.calibration.trim import Trim


class _FakeCalibration:
    def __init__(self, df):
        self.df = df
        self.final_weight = "final_weight"


def _make_df(weights):
    return pl.DataFrame({"final_weight": weights}).lazy()


def _weight_max(c):
    return (
        nw.from_native(c.df).select(nw.col(c.final_weight).max()).collect().item(0, 0)
    )


def _weight_min(c):
    return (
        nw.from_native(c.df).select(nw.col(c.final_weight).min()).collect().item(0, 0)
    )


trim = Trim(trim=True, min_val=0.05, max_val=5.0, step=0.02, tolerance_step=0.01)

# --- A weight well above max_val should get trimmed down to <= max_val ---
c_max = _FakeCalibration(_make_df([1.0, 2.0, 10.0]))
b_complete_max = trim.trim_in_loop(c=c_max, iLoop=1, nLoops=5)

assert not b_complete_max, (
    "Expected trimming to be needed (max weight 10.0 exceeds bound 5.0), "
    "but trim_in_loop reported complete=True"
)
assert _weight_max(c_max) <= trim.max_val, (
    f"Expected the over-bound weight to be trimmed to <= {trim.max_val}, "
    f"got max={_weight_max(c_max)}"
)

# --- A weight well below min_val should get trimmed up to >= min_val ---
c_min = _FakeCalibration(_make_df([1.0, 2.0, 0.001]))
b_complete_min = trim.trim_in_loop(c=c_min, iLoop=1, nLoops=5)

assert not b_complete_min, (
    "Expected trimming to be needed (min weight 0.001 is below bound 0.05), "
    "but trim_in_loop reported complete=True"
)
assert _weight_min(c_min) >= trim.min_val, (
    f"Expected the under-bound weight to be trimmed to >= {trim.min_val}, "
    f"got min={_weight_min(c_min)}"
)

# --- All weights within bounds: nothing should be trimmed ---
in_bounds = [1.0, 2.0, 3.0]
c_ok = _FakeCalibration(_make_df(in_bounds))
b_complete_ok = trim.trim_in_loop(c=c_ok, iLoop=1, nLoops=5)

assert b_complete_ok, (
    "Expected no trimming needed for in-bounds weights, but trim_in_loop "
    "reported complete=False"
)
assert _weight_max(c_ok) == max(in_bounds) and _weight_min(c_ok) == min(in_bounds), (
    "Expected in-bounds weights to be left untouched, but they were changed"
)

print("All trim_in_loop checks passed.")
