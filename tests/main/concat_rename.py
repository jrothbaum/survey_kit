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


def test_adapter_filter_values_keeps_vcov():
    out = _adapter(["a", "b", "c"], tidy=True).filter_values(["a", "c"])
    rs = out.replicate_stats
    assert _ids(out.df_estimates) == ["a", "c"]
    assert _ids(out.df_ses) == ["a", "c"]
    assert _ids(rs.df_tidy) == ["a", "c"]
    assert _collect(rs.df_vcov).shape == (4, 3)
    assert _ids(_adapter(["a", "b"]).filter_values("b").df_estimates) == ["b"]


def test_mi_filter_values_everywhere():
    out = _mi(["a", "b", "c"], vcov=True).filter_values(["a", "b"])
    for df in (out.df_estimates, out.df_ses, out.df_p):
        assert _ids(df) == ["a", "b"]
    for imp in out.implicate_stats:
        assert _ids(imp.df_estimates) == ["a", "b"]
        assert _collect(imp.df_vcov).shape == (4, 3)
    assert set(_collect(out.df_vcov)["Variable_1"]) <= {"a", "b"}


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_polars_expressions_on_adapter_and_mi():
    a = _adapter(["a", "b", "c"])
    assert _ids(a.filter(pl.col("Variable") != "c").df_estimates) == ["a", "b"]
    a_novcov = _adapter(["a", "b", "c"], vcov=False)
    out = a_novcov.with_columns(pl.col("coef") * 2)
    assert _collect(out.df_estimates)["coef"].to_list() == [2.0, 4.0, 6.0]
    out = a_novcov.sort(pl.col("Variable").sort_by(pl.col("coef"), descending=True))
    assert _ids(out.df_estimates) == ["c", "b", "a"]

    mi = _mi(["a", "b", "c"])
    out = mi.filter(pl.col("Variable") != "c")
    assert _ids(out.df_estimates) == ["a", "b"]
    for imp in out.implicate_stats:
        assert _ids(imp.df_estimates) == ["a", "b"]
    out = mi.with_columns(pl.col("coef") + 1)
    assert _collect(out.df_estimates)["coef"][0] > 1.5


def test_adapter_sort_leaves_vcov_and_tidy_unsorted():
    a = _adapter(["a", "b", "c"], tidy=True)
    out = a.sort(pl.col("Variable").sort_by("Variable", descending=True))
    assert _ids(out.df_estimates) == ["c", "b", "a"]
    assert _ids(out.df_ses) == ["c", "b", "a"]
    assert _ids(out.replicate_stats.df_tidy) == ["a", "b", "c"]
    assert _collect(out.replicate_stats.df_vcov).equals(_collect(a.replicate_stats.df_vcov))


def test_polars_expression_mixed_and_non_polars_raise():
    a = _adapter(["a", "b"], vcov=False)
    _raises(
        lambda: a.with_columns([pl.col("coef") * 2, nw.col("coef") * 3]), TypeError
    )

    pd_est = _collect(a.df_estimates).to_pandas()
    pd_ses = _collect(a.df_ses).to_pandas()
    a_pd = AdapterStats(pd_est, pd_ses, display=False)
    #   narwhals expression is fine on a pandas-backed frame...
    assert len(a_pd.filter(nw.col("Variable") != "a").df_estimates) == 1
    #   ...a polars one is not
    _raises(lambda: a_pd.filter(pl.col("Variable") != "a"), TypeError)


def test_select_with_polars_and_narwhals_expressions():
    e = pl.DataFrame({"Variable": ["a", "b"], "x": [1.0, 2.0], "y": [3.0, 4.0]})
    a = AdapterStats(e, e.clone(), display=False)
    for arg in ("x", ["x"], nw.col("x"), pl.col("x"), pl.col("^x$"), pl.exclude("y")):
        assert _collect(a.select(arg).df_estimates).columns == ["Variable", "x"], arg
        assert _collect(a.select(arg).df_ses).columns == ["Variable", "x"], arg

    mi = _mi(["a", "b"])
    out = mi.select(pl.col("coef"))
    assert _collect(out.df_estimates).columns == ["Variable", "coef"]
    for imp in out.implicate_stats:
        assert _collect(imp.df_estimates).columns == ["Variable", "coef"]
    _raises(lambda: a.select([pl.col("x"), nw.col("y")]), TypeError)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
