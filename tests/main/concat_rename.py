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


def test_sort_by_value_column_stays_aligned_and_follows_into_vcov_tidy():
    n = ["a", "b", "c"]
    e = pl.DataFrame({"Variable": n, "coef": [1.0, 2.0, 3.0]})
    s = pl.DataFrame({"Variable": n, "coef": [0.3, 0.2, 0.1]})
    v = pl.DataFrame(
        [{"Variable_1": x, "Variable_2": y, "coef": 0.01} for x in n for y in n]
    )
    a = AdapterStats(e, s, df_vcov=v, df_tidy=e.clone(), display=False)
    out = a.sort(-pl.col("coef"))
    assert _ids(out.df_estimates) == ["c", "b", "a"]
    assert _ids(out.df_ses) == ["c", "b", "a"]
    assert _ids(out.replicate_stats.df_tidy) == ["c", "b", "a"]
    vc = _collect(out.replicate_stats.df_vcov)
    assert vc["Variable_1"].to_list() == ["c"] * 3 + ["b"] * 3 + ["a"] * 3
    assert vc["Variable_2"].to_list() == ["c", "b", "a"] * 3
    #   SEs still pair with the right variable
    assert _collect(out.df_ses)["coef"].to_list() == [0.1, 0.2, 0.3]


def test_sort_descending():
    a = _adapter(["a", "b", "c"], tidy=True)
    for key in ("Variable", pl.col("Variable")):
        out = a.sort(key, descending=True)
        assert _ids(out.df_estimates) == ["c", "b", "a"]
        assert _ids(out.df_ses) == ["c", "b", "a"]
        assert _ids(out.replicate_stats.df_tidy) == ["c", "b", "a"]
        assert _collect(out.replicate_stats.df_vcov)["Variable_1"][0] == "c"
    #   narwhals-style column-name sort too, and a value column
    out = a.sort("coef", descending=True)
    assert _ids(out.df_estimates) == ["c", "b", "a"]
    out = _mi(["a", "b", "c"]).sort("Variable", descending=True)
    assert _ids(out.df_estimates) == ["c", "b", "a"]
    for imp in out.implicate_stats:
        assert _ids(imp.df_ses) == ["c", "b", "a"]


def test_mi_sort_by_value_follows_implicates():
    mi = _mi(["a", "b", "c"], vcov=True)
    out = mi.sort(-pl.col("coef"))
    for df in (out.df_estimates, out.df_ses, out.df_p):
        assert _ids(df) == ["c", "b", "a"]
    for imp in out.implicate_stats:
        assert _ids(imp.df_estimates) == ["c", "b", "a"]
        assert _ids(imp.df_ses) == ["c", "b", "a"]
        assert _collect(imp.df_vcov)["Variable_1"].to_list()[:3] == ["c"] * 3


def _grouped_calculator(columns):
    import numpy as np

    rng = np.random.default_rng(0)
    n = 300
    df = pl.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n) + 1,
            "g": rng.integers(0, 3, n),
            "w": np.ones(n),
            **{f"w{i}": rng.uniform(0.5, 1.5, n) for i in range(7)},
        }
    )
    return StatCalculator(
        df,
        statistics=Statistics(stats=["mean", "median"], columns=columns),
        weight="w",
        replicates=Replicates(weight_stub="w", n_replicates=6),
        by={"Group": ["g"]},
        display=False,
    )


def _paired(sc):
    """estimate/SE pairs keyed by (Variable, g) - order-independent."""
    return _collect(sc.df_estimates).join(
        _collect(sc.df_ses), on=["Variable", "g"], suffix="_se"
    )


def _same_pairs(result, base):
    pr = _paired(result)
    m = pr.join(_paired(base), on=["Variable", "g"], suffix="_o")
    return all(
        (m[c] == m[c + "_o"]).all() for c in pr.columns if c not in ("Variable", "g")
    )


def test_grouped_calculator_operations_keep_groups_and_pairing():
    sc = _grouped_calculator(["x", "y"])
    for out in (
        sc.sort("mean", descending=True),
        sc.sort(["Variable", "g"], descending=[True, False]),
        sc.filter_values("x"),
        sc.filter(nw.col("g") > 0),
        sc.select("mean"),
    ):
        assert "g" in _collect(out.df_estimates).columns
        assert _same_pairs(out, sc)
    #   select must keep the group column in every table
    sel = sc.select("mean")
    assert _collect(sel.df_replicates).columns == ["Variable", "g", "mean", "___replicate___"]

    renamed = sc.rename_values({"x": "xx"})
    assert sorted(set(_ids(renamed.df_replicates))) == ["xx", "y"]
    assert sorted(set(_collect(renamed.df_estimates)["g"])) == [0, 1, 2]

    both = _grouped_calculator(["x"]).concat_with(
        _grouped_calculator(["y"]), how="vertical"
    )
    assert _collect(both.df_estimates).shape == (6, 4)
    assert _collect(both.df_replicates).shape[0] == 42


def test_grouped_mi_operations_follow_implicates():
    def mk(names):
        imps = []
        for k in range(3):
            rows = [(v, g) for v in names for g in (0, 1, 2)]
            e = pl.DataFrame(
                {
                    "Variable": [r[0] for r in rows],
                    "g": [r[1] for r in rows],
                    "mean": [float(i) + k * 0.1 for i in range(len(rows))],
                }
            )
            s = e.with_columns(pl.Series("mean", [0.1 + 0.01 * i for i in range(len(rows))]))
            imps.append(ReplicateStats(df_estimates=e, df_ses=s))
        mi = MultipleImputation(implicate_stats=imps, join_on=["Variable", "g"])
        mi.calculate()
        return mi

    mi = mk(["x", "y"])
    out = mi.sort("mean", descending=True)
    top = _collect(out.df_estimates).row(0, named=True)
    assert (top["Variable"], top["g"]) == ("y", 2)
    pairs = _collect(out.df_estimates).join(
        _collect(out.df_ses), on=["Variable", "g"], suffix="_se"
    )
    base = _collect(mi.df_estimates).join(
        _collect(mi.df_ses), on=["Variable", "g"], suffix="_se"
    )
    m = pairs.join(base, on=["Variable", "g"], suffix="_o")
    assert (m["mean_se"] == m["mean_se_o"]).all()
    for imp in out.implicate_stats:
        assert _collect(imp.df_estimates).row(0, named=True)["g"] == 2
    assert _collect(mi.select("mean").df_estimates).columns == ["Variable", "g", "mean"]
    assert _collect(mi.filter_values("x").df_estimates).shape == (3, 3)
    both = mi.concat_with(mk(["z"]), how="vertical")
    assert _collect(both.df_estimates).shape == (9, 3)


def _regression_data(seed, n=400):
    import numpy as np

    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    x1 = rng.normal(size=n)
    slope = np.where(g == 0, 2.0, -1.0)
    y = 0.5 + slope * x1 + rng.normal(scale=0.3, size=n)
    reps = {f"w{i}": rng.uniform(0.5, 1.5, n) for i in range(8)}
    return pl.DataFrame({"y": y, "x1": x1, "g": g, "w": np.ones(n), **reps})


def test_adapters_by_group_direct():
    from survey_kit.statistics.adapters import polars_ds_adapter, statsmodels_adapter

    df = _regression_data(0)
    for adapter, kwargs in (
        (statsmodels_adapter, {"y": "y", "x": ["x1"]}),
        (polars_ds_adapter, {"y": "y", "x": ["x1"]}),
    ):
        out = adapter(df, by="g", **kwargs)
        est = _collect(out.df_estimates)
        assert est.columns[:3] == ["Variable", "g", "estimate"], est.columns
        assert est.height == 4
        slopes = {
            r["g"]: r["estimate"] for r in est.filter(pl.col("Variable") == "x1").to_dicts()
        }
        assert abs(slopes[0] - 2.0) < 0.2 and abs(slopes[1] + 1.0) < 0.2
        assert _collect(out.df_ses).height == 4
        assert out.summarize_vars == ["g"]
        if adapter is statsmodels_adapter:
            vcov = _collect(out.replicate_stats.df_vcov)
            assert vcov.columns == ["Variable_1", "Variable_2", "g", "estimate"]
            assert vcov.height == 8  # 2 groups x 2x2 terms - no cross-group pairs
        #   same as running each group on its own
        one = adapter(df.filter(pl.col("g") == 1), **kwargs)
        est1 = _collect(one.df_estimates).sort("Variable")["estimate"].to_list()
        got = est.filter(pl.col("g") == 1).sort("Variable")["estimate"].to_list()
        assert est1 == got
        #   downstream operations keep working with the group column
        sorted_out = out.sort("estimate", descending=True)
        assert _collect(sorted_out.df_estimates).row(0, named=True)["g"] == 0


def test_mi_ses_by_group_with_vcov_and_replicates():
    from survey_kit.statistics.adapters import mi_ses_from_statsmodels

    implicates = [_regression_data(seed) for seed in range(3)]
    mi = mi_ses_from_statsmodels(
        df_implicates=implicates, y="y", x=["x1"], by="g", round_output=False
    )
    est = _collect(mi.df_estimates)
    assert est.height == 4 and "g" in est.columns
    ses = _collect(mi.df_ses)
    assert ses.height == 4 and "g" in ses.columns
    vcov = _collect(mi.df_vcov)
    assert vcov.columns == ["Variable_1", "Variable_2", "g", "estimate"]
    assert vcov.height == 8
    #   within-group variances on the diagonal match df_ses squared
    diag = vcov.filter(pl.col("Variable_1") == pl.col("Variable_2")).rename(
        {"Variable_1": "Variable"}
    )
    m = diag.join(ses, on=["Variable", "g"], suffix="_se")
    assert ((m["estimate"] ** 0.5 - m["estimate_se"]).abs() < 1e-9).all()

    from survey_kit.statistics.replicates import Replicates

    mi_rep = mi_ses_from_statsmodels(
        df_implicates=implicates,
        y="y",
        x=["x1"],
        by="g",
        replicates=Replicates(weight_stub="w", n_replicates=7),
        round_output=False,
    )
    est_rep = _collect(mi_rep.df_estimates)
    assert est_rep.height == 4 and "g" in est_rep.columns
    assert _collect(mi_rep.df_ses).height == 4


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
