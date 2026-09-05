"""
Sanity/regression checks on Moment's by-group sub_moments: every group is
covered exactly once, weight shares sum to 1, and per-group observation
counts add up to the total. Exercises _create_sub_moments_batched, which
computes all groups' weight/obs sums and weighted-mean targets/non-zero
counts via a handful of batched groupby calls instead of one query per
group.

Also checks that a pyarrow-backed Moment (which can't run the batched
groupby -- narwhals' pyarrow backend rejects the compound weighted-mean
aggregation) falls back to _create_sub_moments_looped and still matches
the polars-backed (batched) result.
"""

import narwhals as nw
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

# --- pyarrow (looped fallback) vs polars (batched) cross-check ---
m_arrow = Moment(
    df=df.collect().to_arrow(),
    formula=f.formula,
    weight="w",
    index="index",
    by=["grp"],
    rescale=False,
)

assert len(m_arrow.sub_moments) == n_groups, (
    f"Expected {n_groups} sub_moments (pyarrow), got {len(m_arrow.sub_moments)}"
)

by_polars = {int(s.by_where_strings[0].split("==")[1]): s for s in m.sub_moments}
by_arrow = {
    int(s.by_where_strings[0].split("==")[1]): s for s in m_arrow.sub_moments
}
assert set(by_polars) == set(by_arrow), (by_polars.keys(), by_arrow.keys())

for grp_value, s_polars in by_polars.items():
    s_arrow = by_arrow[grp_value]
    assert abs(s_polars.by_share - s_arrow.by_share) < 1e-9, (
        f"group {grp_value}: by_share differs, "
        f"polars={s_polars.by_share} pyarrow={s_arrow.by_share}"
    )
    assert s_polars.n_observations == s_arrow.n_observations, (
        f"group {grp_value}: n_observations differs, "
        f"polars={s_polars.n_observations} pyarrow={s_arrow.n_observations}"
    )
    target_polars = nw.from_native(s_polars.targets).lazy().collect().item(0, "v")
    target_arrow = nw.from_native(s_arrow.targets).lazy().collect().item(0, "v")
    assert abs(target_polars - target_arrow) < 1e-9, (
        f"group {grp_value}: target differs, "
        f"polars={target_polars} pyarrow={target_arrow}"
    )

print("pyarrow fallback matches polars batched path.")
