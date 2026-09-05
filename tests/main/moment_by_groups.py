"""
Sanity/regression checks on Moment's by-group sub_moments: every group is
covered exactly once, weight shares sum to 1, and per-group observation
counts add up to the total. Exercises the same _create_sub_moments path
touched by materializing df_by once before the per-group loop (an
efficiency fix) instead of leaving it lazy across the whole loop.
"""

import polars as pl

from survey_kit.calibration.moment import Moment
from survey_kit.utilities.formula_builder import FormulaBuilder

n_per_group = 500
n_groups = 6

df = pl.DataFrame(
    {
        "index": range(n_per_group * n_groups),
        "grp": [g for g in range(n_groups) for _ in range(n_per_group)],
        "v": [
            float(g) + (i % 7) * 0.1
            for g in range(n_groups)
            for i in range(n_per_group)
        ],
        "w": [1.0 + (i % 5) * 0.2 for _ in range(n_groups) for i in range(n_per_group)],
    }
).lazy()

f = FormulaBuilder(df=df, constant=False)
f.continuous(columns=["v"])

m = Moment(
    df=df, formula=f.formula, weight="w", index="index", by=["grp"], rescale=False
)

assert len(m.sub_moments) == n_groups, (
    f"Expected {n_groups} sub_moments, got {len(m.sub_moments)}"
)

seen_groups = set()
by_share_total = 0.0
n_obs_total = 0
for sub in m.sub_moments:
    grp_value = int(sub.by_where_strings[0].split("==")[1])
    seen_groups.add(grp_value)
    by_share_total += sub.by_share
    n_obs_total += sub.n_observations

assert seen_groups == set(range(n_groups)), (
    f"Expected sub_moments for groups {set(range(n_groups))}, got {seen_groups}"
)
assert abs(by_share_total - 1.0) < 1e-9, (
    f"Expected by_share across groups to sum to 1.0, got {by_share_total}"
)
assert n_obs_total == n_per_group * n_groups, (
    f"Expected total n_observations {n_per_group * n_groups}, got {n_obs_total}"
)

print("All Moment by-group checks passed.")
