"""
Vertical concat_with() and rename_values() across AdapterStats,
StatCalculator and MultipleImputation. Run directly or with pytest.
"""

import narwhals as nw
import polars as pl

from survey_kit.statistics.adapter_stats import AdapterStats
from survey_kit.statistics.calculator import StatCalculator
from survey_kit.statistics.multiple_imputation import MultipleImputation
from survey_kit.statistics.replicates import ReplicateStats, Replicates
from survey_kit.statistics.statistics import Statistics


def _collect(df):
    return df.lazy().collect()


def _ids(df, col="Variable"):
    return _collect(df)[col].to_list()


def _adapter(names, tidy=False, vcov=True):
    e = pl.DataFrame({"Variable": names, "coef": [1.0 + i for i in range(len(names))]})
    s = pl.DataFrame({"Variable": names, "coef": [0.1] * len(names)})
    v = pl.DataFrame(
        [
            {"Variable_1": a, "Variable_2": b, "coef": 0.01}
            for a in names
            for b in names
        ]
    )
    return AdapterStats(
        e,
        s,
        df_vcov=v if vcov else None,
        df_tidy=e if tidy else None,
        display=False,
    )


def _mi(names, vcov=False):
    imps = []
    for k in range(3):
        e = pl.DataFrame(
            {"Variable": names, "coef": [1.0 + i + k * 0.1 for i in range(len(names))]}
        )
        s = pl.DataFrame({"Variable": names, "coef": [0.1] * len(names)})
        v = (
            pl.DataFrame(
                [
                    {"Variable_1": a, "Variable_2": b, "coef": 0.01 if a == b else 0.0}
                    for a in names
                    for b in names
                ]
            )
            if vcov
            else None
        )
        imps.append(ReplicateStats(df_estimates=e, df_ses=s, df_vcov=v))
    mi = MultipleImputation(implicate_stats=imps, join_on=["Variable"])
    mi.calculate()
    return mi


def test_adapter_vertical_concat_with_vcov_and_tidy():
    a = _adapter(["a", "b"], tidy=True)
    b = _adapter(["c", "d"], tidy=True)
    out = a.concat_with(b, how="vertical")

    assert _ids(out.df_estimates) == ["a", "b", "c", "d"]
    assert _ids(out.df_ses) == ["a", "b", "c", "d"]
    assert out.replicate_stats.df_vcov.shape == (8, 3)
    assert out.replicate_stats.df_tidy.shape == (4, 2)
    #   non-mutating
    assert _ids(a.df_estimates) == ["a", "b"]
    assert a.replicate_stats.df_vcov is not None


def test_adapter_vertical_concat_overlap_raises():
    a = _adapter(["a", "b"])
    try:
        a.concat_with(_adapter(["b", "c"]), how="vertical")
    except ValueError:
        return
    raise AssertionError("overlapping terms with df_vcov should raise")


def test_adapter_rename_values_all_tables():
    out = _adapter(["a", "b"], tidy=True).rename_values({"a": "A"})
    rs = out.replicate_stats
    assert _ids(out.df_estimates) == ["A", "b"]
    assert _ids(out.df_ses) == ["A", "b"]
    assert _ids(rs.df_tidy) == ["A", "b"]
    assert set(_collect(rs.df_vcov)["Variable_1"]) == {"A", "b"}
    assert set(_collect(rs.df_vcov)["Variable_2"]) == {"A", "b"}


def test_rename_values_expr_fn():
    out = _adapter(["a", "b"]).rename_values(
        expr_fn=lambda c: nw.when(c == "a").then(nw.lit("z")).otherwise(c)
    )
    assert _ids(out.df_estimates) == ["z", "b"]


def test_stat_calculator_rename_values_includes_replicates():
    import numpy as np

    rng = np.random.default_rng(0)
    n = 100
    df = pl.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "w": np.ones(n),
            **{f"w{i}": rng.uniform(0.5, 1.5, n) for i in range(7)},
        }
    )
    sc = StatCalculator(
        df,
        statistics=Statistics(stats=["mean"], columns=["x", "y"]),
        weight="w",
        replicates=Replicates(weight_stub="w", n_replicates=6),
        display=False,
    )
    out = sc.rename_values({"x": "xx"})
    assert sorted(_ids(out.df_estimates)) == ["xx", "y"]
    assert sorted(_ids(out.df_ses)) == ["xx", "y"]
    assert sorted(set(_ids(out.df_replicates))) == ["xx", "y"]
    assert sorted(_ids(sc.df_estimates)) == ["x", "y"]


def test_mi_vertical_concat_reaches_implicates():
    out = _mi(["a", "b"]).concat_with(_mi(["c", "d"]), how="vertical")
    assert _ids(out.df_estimates) == ["a", "b", "c", "d"]
    for imp in out.implicate_stats:
        assert _ids(imp.df_estimates) == ["a", "b", "c", "d"]


def test_mi_vertical_concat_stacks_vcov_block_diagonally():
    out = _mi(["a", "b"], vcov=True).concat_with(_mi(["c", "d"], vcov=True), how="vertical")
    for imp in out.implicate_stats:
        assert imp.df_vcov is not None
        assert _collect(imp.df_vcov).shape == (8, 3)
    vcov = _collect(out.df_vcov)
    assert set(vcov["Variable_1"]) == {"a", "b", "c", "d"}
    #   no within-implicate covariance between a term in each object
    cross = vcov.filter(
        (pl.col("Variable_1") == "a") & (pl.col("Variable_2") == "c")
    )
    assert cross.shape[0] <= 1


def test_mi_horizontal_concat():
    other = _mi(["a", "b"]).rename({"coef": "coef2"})
    out = _mi(["a", "b"]).concat_with(other, how="horizontal")
    assert _collect(out.df_estimates).columns == ["Variable", "coef", "coef2"]
    for imp in out.implicate_stats:
        assert _collect(imp.df_estimates).columns == ["Variable", "coef", "coef2"]


def test_mi_rename_values_everywhere():
    mi = _mi(["a", "b"])
    out = mi.rename_values({"a": "A"})
    for df in (out.df_estimates, out.df_ses, out.df_p):
        assert _ids(df) == ["A", "b"]
    for imp in out.implicate_stats:
        assert _ids(imp.df_estimates) == ["A", "b"]
    assert _ids(mi.df_estimates) == ["a", "b"]


def test_mi_methods_propagate_to_implicates():
    mi = _mi(["a", "b", "c"])
    out = mi.filter(nw.col("Variable") != "c")
    for imp in out.implicate_stats:
        assert _ids(imp.df_estimates) == ["a", "b"]
    out = mi.rename({"coef": "beta"})
    for imp in out.implicate_stats:
        assert _collect(imp.df_estimates).columns == ["Variable", "beta"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
