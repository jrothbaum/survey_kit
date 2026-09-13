"""
Coverage for survey_kit.statistics.adapters - previously untested (this
file didn't exist before). Three things specifically added/claimed this
session and only verified by hand or by assertion, not in the test suite,
that this checks for real:

    1. join_on now optional on every mi_ses_from_<package>/
       mi_ses_from_r_fixest.*/mi_ses_from_pyfixest.* - derived from
       join_on_name instead of needing to be passed.
    2. replicates= (bootstrap SEs from replicate weights instead of the
       package's own vcov/cov_type) on every mi_ses_from_* that has a
       weight argument to substitute a replicate column into - including
       that the implicate is converted to whatever the underlying package
       needs (pandas, an R data.frame, a materialized polars frame) once
       per implicate, not once per replicate weight.
    3. "DataFrame agnostic (Polars, Pandas, Arrow, DuckDB via Narwhals)"
       is a documented key feature (user-guide/statistics.md) but was
       only ever exercised against polars input for these adapters -
       checks pandas and DuckDB implicates work identically and pins
       down the actual output type contract (adapters' own direct-call
       tuples are always pl.DataFrame/None regardless of input backend;
       every mi_ses_from_*'s combined .df_estimates is always a
       pl.LazyFrame).

R/Stata sections are guarded by availability checks and skip (with a
logged reason) rather than fail when the R/Stata install this was written
against isn't present - same pattern r_arbitrary_estimators.py uses for
the optional quantreg package.
"""

import numpy as np
import polars as pl

from survey_kit import logger
from survey_kit.statistics import adapters
from survey_kit.statistics.adapters import (
    mi_ses_from_statsmodels,
    mi_ses_from_linearmodels,
    mi_ses_from_pyfixest,
    mi_ses_from_polars_ds,
    mi_ses_from_r_fixest,
    mi_ses_from_stata,
)
from survey_kit.statistics.replicates import Replicates
from survey_kit.statistics.bootstrap import bayes_bootstrap_weights
from survey_kit.statistics._r_interop import check_r_setup
from survey_kit.statistics._stata_interop import check_stata_setup


#   ---------------------------------------------------------------------
#   Shared synthetic data - y = 1 + 2*x1 - 1.5*x2 + noise, 5 implicates.
#   ---------------------------------------------------------------------
def _make_implicate(seed: int, n: int = 400) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1 + 2 * x1 - 1.5 * x2 + rng.normal(size=n) * 0.4
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


df_implicates = [_make_implicate(seed) for seed in range(5)]

TRUE_PARAMS = {"x1": 2.0, "x2": -1.5}
#   Loose - this just needs to catch broken wiring (wrong column fit, term
#   misalignment, weight not applied, ...), not be a precise stats test.
TOL = 0.3


def _assert_close(df_estimates, name: str, label: str):
    import narwhals as nw

    df_estimates = nw.from_native(df_estimates).lazy().collect().to_polars()
    row = df_estimates.filter(pl.col("Variable") == name)
    assert row.height == 1, f"{label}: '{name}' missing from df_estimates"
    value = row["estimate"][0]
    assert abs(value - TRUE_PARAMS[name]) < TOL, (
        f"{label}: {name} estimate {value} too far from true value "
        f"{TRUE_PARAMS[name]}"
    )


def _assert_mi_estimates(mi_result, label: str):
    import narwhals as nw

    for name in TRUE_PARAMS:
        _assert_close(mi_result.df_estimates, name, label)
    n_terms = nw.from_native(mi_result.df_estimates).lazy().collect().shape[0]
    logger.info(f"{label}: OK ({n_terms} terms)")


#   ---------------------------------------------------------------------
#   join_on now derived from join_on_name - not passed anywhere below.
#   Basic correctness for each mi_ses_from_* while we're at it (zero
#   adapter test coverage existed before this file).
#   ---------------------------------------------------------------------
_assert_mi_estimates(
    mi_ses_from_statsmodels(df_implicates=df_implicates, y="y", x=["x1", "x2"]),
    "mi_ses_from_statsmodels",
)
_assert_mi_estimates(
    mi_ses_from_linearmodels(df_implicates=df_implicates, formula="y ~ 1 + x1 + x2"),
    "mi_ses_from_linearmodels",
)
_assert_mi_estimates(
    mi_ses_from_pyfixest.feols(df_implicates=df_implicates, fml="y ~ x1 + x2"),
    "mi_ses_from_pyfixest.feols",
)
_assert_mi_estimates(
    mi_ses_from_polars_ds(df_implicates=df_implicates, y="y", x=["x1", "x2"]),
    "mi_ses_from_polars_ds",
)


#   ---------------------------------------------------------------------
#   Input-backend agnosticism - "DataFrame agnostic (Polars, Pandas,
#   Arrow, DuckDB via Narwhals)" is a documented key feature
#   (user-guide/statistics.md), but was only ever exercised against
#   polars input for these adapters specifically. Check pandas and
#   DuckDB implicates work identically, and pin down what type actually
#   comes back either way (a real, checkable contract, not just "doesn't
#   crash") - observed empirically first: adapters' own direct-call
#   tuples are always pl.DataFrame (or None for a vcov an adapter
#   doesn't compute) regardless of input backend, and every
#   mi_ses_from_*'s combined .df_estimates is always a pl.LazyFrame.
#   ---------------------------------------------------------------------
import duckdb  # noqa: E402

_duckdb_con = duckdb.connect()


def _as_duckdb(df: pl.DataFrame, name: str):
    _duckdb_con.register(name, df.to_pandas())
    return _duckdb_con.sql(f"SELECT * FROM {name}")


df_implicates_pandas = [d.to_pandas() for d in df_implicates]
df_implicates_duckdb = [
    _as_duckdb(d, f"implicate_{i}") for i, d in enumerate(df_implicates)
]

#   (direct-call adapter, kwargs, expected df_vcov type)
_direct_call_cases = [
    (adapters.statsmodels_adapter, {"y": "y", "x": ["x1", "x2"]}, pl.DataFrame),
    (adapters.linearmodels_adapter, {"formula": "y ~ 1 + x1 + x2"}, pl.DataFrame),
    (adapters.polars_ds_adapter, {"y": "y", "x": ["x1", "x2"]}, type(None)),
]

for backend_name, one_implicate in [
    ("pandas", df_implicates_pandas[0]),
    ("duckdb", df_implicates_duckdb[0]),
]:
    for adapter_fn, kwargs, expected_vcov_type in _direct_call_cases:
        label = f"{adapter_fn.__name__}({backend_name} input)"
        df_estimates, df_ses, df_vcov, df_tidy = adapter_fn(one_implicate, **kwargs)
        assert isinstance(df_estimates, pl.DataFrame), f"{label}: df_estimates is {type(df_estimates)}"
        assert isinstance(df_ses, pl.DataFrame), f"{label}: df_ses is {type(df_ses)}"
        assert isinstance(df_vcov, expected_vcov_type), (
            f"{label}: df_vcov is {type(df_vcov)}, expected {expected_vcov_type}"
        )
        assert isinstance(df_tidy, pl.DataFrame), f"{label}: df_tidy is {type(df_tidy)}"
        _assert_close(df_estimates, "x1", label)
        _assert_close(df_estimates, "x2", label)
        logger.info(f"{label}: OK (types + estimates)")

#   (mi_ses_from_* wrapper, kwargs)
_mi_cases = [
    (mi_ses_from_statsmodels, {"y": "y", "x": ["x1", "x2"]}),
    (mi_ses_from_linearmodels, {"formula": "y ~ 1 + x1 + x2"}),
    (mi_ses_from_pyfixest.feols, {"fml": "y ~ x1 + x2"}),
    (mi_ses_from_polars_ds, {"y": "y", "x": ["x1", "x2"]}),
]

for backend_name, implicates in [
    ("pandas", df_implicates_pandas),
    ("duckdb", df_implicates_duckdb),
]:
    for mi_fn, kwargs in _mi_cases:
        label = f"{mi_fn.__qualname__}({backend_name} implicates)"
        mi_result = mi_fn(df_implicates=implicates, **kwargs)
        assert isinstance(mi_result.df_estimates, pl.LazyFrame), (
            f"{label}: df_estimates is {type(mi_result.df_estimates)}, expected pl.LazyFrame"
        )
        _assert_mi_estimates(mi_result, label)

_duckdb_con.close()


#   ---------------------------------------------------------------------
#   replicates= bootstrapping: correctness, plus "converted once per
#   implicate, not once per replicate weight".
#   ---------------------------------------------------------------------
N_REPLICATES = 15


def _with_replicate_weights(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    df = df.with_columns(pl.lit(1.0).alias("replicate_0"))
    return bayes_bootstrap_weights(
        df, prefix="replicate_", n_replicates=N_REPLICATES, seed=seed
    )


df_implicates_weighted = [
    _with_replicate_weights(df, seed) for seed, df in enumerate(df_implicates)
]
replicates = Replicates(weight_stub="replicate_", n_replicates=N_REPLICATES, bootstrap=True)


class _CallCounter:
    """Wraps a conversion function, counting calls where `is_converted`
    is False (real work) vs True (already-converted passthrough)."""

    def __init__(self, original, is_converted):
        self.original = original
        self.is_converted = is_converted
        self.real = 0
        self.passthrough = 0

    def __call__(self, df):
        if self.is_converted(df):
            self.passthrough += 1
        else:
            self.real += 1
        return self.original(df)


#   statsmodels/linearmodels/pyfixest all share _to_pandas - one
#   instrumented run covers all three.
import pandas as pd  # noqa: E402

_counter = _CallCounter(adapters._to_pandas, lambda df: isinstance(df, pd.DataFrame))
adapters._to_pandas = _counter
try:
    _assert_mi_estimates(
        mi_ses_from_statsmodels(
            df_implicates=df_implicates_weighted, y="y", x=["x1", "x2"], replicates=replicates
        ),
        "mi_ses_from_statsmodels(replicates=)",
    )
    _assert_mi_estimates(
        mi_ses_from_linearmodels(
            df_implicates=df_implicates_weighted,
            formula="y ~ 1 + x1 + x2",
            replicates=replicates,
        ),
        "mi_ses_from_linearmodels(replicates=)",
    )
    _assert_mi_estimates(
        mi_ses_from_pyfixest.feols(
            df_implicates=df_implicates_weighted, fml="y ~ x1 + x2", replicates=replicates
        ),
        "mi_ses_from_pyfixest.feols(replicates=)",
    )
finally:
    adapters._to_pandas = _counter.original

expected_real = 3 * len(df_implicates_weighted)  # 3 packages x 5 implicates
assert _counter.real == expected_real, (
    f"_to_pandas did real conversion {_counter.real} times, expected exactly "
    f"{expected_real} (once per implicate per package) - the replicate loop "
    f"is redoing conversion work it should be caching."
)
logger.info(
    f"_to_pandas caching: {_counter.real} real conversions, "
    f"{_counter.passthrough} cheap passthroughs - OK"
)


#   polars_ds's own conversion (_to_polars) only does real work for a
#   LazyFrame - pass implicates as lazy to actually exercise the cached
#   path (an already-materialized pl.DataFrame is cheap either way).
_counter_pl = _CallCounter(adapters._to_polars, lambda df: isinstance(df, pl.DataFrame))
adapters._to_polars = _counter_pl
try:
    _assert_mi_estimates(
        mi_ses_from_polars_ds(
            df_implicates=[d.lazy() for d in df_implicates_weighted],
            y="y",
            x=["x1", "x2"],
            replicates=replicates,
        ),
        "mi_ses_from_polars_ds(replicates=, lazy implicates)",
    )
finally:
    adapters._to_polars = _counter_pl.original

assert _counter_pl.real == len(df_implicates_weighted), (
    f"_to_polars collected the LazyFrame {_counter_pl.real} times, expected "
    f"exactly {len(df_implicates_weighted)} (once per implicate) - a lazy "
    "implicate's upstream plan is being re-executed per replicate instead "
    "of collected once."
)
logger.info(
    f"_to_polars caching: {_counter_pl.real} real collects, "
    f"{_counter_pl.passthrough} cheap passthroughs - OK"
)


#   ---------------------------------------------------------------------
#   R (fixest) - guarded, skips cleanly if R/rpy2/fixest aren't available.
#   ---------------------------------------------------------------------
_r_setup = check_r_setup(["fixest"])
_r_available = (
    _r_setup.get("rscript_found")
    and _r_setup.get("rpy2_installed")
    and _r_setup.get("rpy2_arrow_installed")
    and _r_setup["r_packages"].get("fixest")
)
if _r_available:
    from survey_kit.statistics import _r_interop

    _assert_mi_estimates(
        mi_ses_from_r_fixest.feols(df_implicates=df_implicates, formula="y ~ x1 + x2"),
        "mi_ses_from_r_fixest.feols",
    )

    import rpy2.rinterface as rinterface  # noqa: E402

    _counter_r = _CallCounter(
        _r_interop.dataframe_to_r, lambda df: isinstance(df, rinterface.Sexp)
    )
    _r_interop.dataframe_to_r = _counter_r
    try:
        _assert_mi_estimates(
            mi_ses_from_r_fixest.feols(
                df_implicates=df_implicates_weighted,
                formula="y ~ x1 + x2",
                replicates=replicates,
            ),
            "mi_ses_from_r_fixest.feols(replicates=)",
        )
    finally:
        _r_interop.dataframe_to_r = _counter_r.original

    assert _counter_r.real == len(df_implicates_weighted), (
        f"dataframe_to_r did real conversion {_counter_r.real} times, "
        f"expected exactly {len(df_implicates_weighted)} (once per "
        "implicate) - the replicate loop is redoing R conversion work."
    )
    logger.info(
        f"dataframe_to_r caching: {_counter_r.real} real conversions, "
        f"{_counter_r.passthrough} cheap passthroughs - OK"
    )
else:
    logger.info(
        "R/rpy2/fixest not fully available - skipping mi_ses_from_r_fixest "
        f"checks ({_r_setup.get('missing')})"
    )


#   ---------------------------------------------------------------------
#   Stata - guarded, skips cleanly if pystata isn't importable (it isn't
#   on PyPI - only ships inside Stata 17+).
#   ---------------------------------------------------------------------
_stata_setup = check_stata_setup()
if _stata_setup.get("pystata_importable") and _stata_setup.get("polars_readstat_importable"):
    _assert_mi_estimates(
        mi_ses_from_stata(df_implicates=df_implicates, command="regress y x1 x2"),
        "mi_ses_from_stata",
    )
    _assert_mi_estimates(
        mi_ses_from_stata(
            df_implicates=df_implicates_weighted,
            command="regress y x1 x2 [pw={weight}]",
            replicates=replicates,
        ),
        "mi_ses_from_stata(replicates=)",
    )
else:
    logger.info(
        "pystata not importable in this environment (needs a "
        "Stata 17+ - see check_stata_setup()) - skipping "
        f"mi_ses_from_stata checks ({_stata_setup.get('missing')})"
    )


logger.info("tests/main/adapters.py: all checks passed")
