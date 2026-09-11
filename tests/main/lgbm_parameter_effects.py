"""
For each LightGBM-relevant Parameters.LightGBM()/Variable option, build two
otherwise-identical SRMI runs that differ in ONE parameter, and confirm BOTH:
  1) a direct log-file signal that the intended code path actually executed
     (per-variable log files SRMI already writes to disk for every run - see
     _imputation_test_utils.read_variable_log), and
  2) the claimed statistical/behavioral effect actually shows up in the data.

A "does it crash" test alone can't catch a parameter being silently ignored
(exactly what happened with _lightgbm_simple's cv_folds bug, fixed earlier -
the in-sample fallback path doesn't crash either). This file is not part of
the fast day-to-day regression loop - run it whenever imputation code
(impute.py/lightgbm_wrapper.py/parameters.py) changes.
"""

import numpy as np
import polars as pl

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.srmi import SRMI
from survey_kit.imputation.utilities.lightgbm_wrapper import Tuner
from survey_kit.imputation.utilities.tuning import HyperparameterSpace, IntRange
from survey_kit import logger, config

from _imputation_test_utils import read_variable_log, capture_global_log

path_scratch = config.path_temp_files


def _build_and_run(df, variable, path_suffix, n_implicates=1, n_iterations=1):
    srmi = SRMI(
        df=df,
        variables=[variable],
        index=["idx"],
        replication=SRMI.Replication(n_implicates=n_implicates, n_iterations=n_iterations),
        parallel=SRMI.Parallel(enabled=False),
        bootstrap=SRMI.Bootstrap(enabled=False),
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_lgbm_param_effects_{path_suffix}",
            force_start=True,
        ),
    )
    srmi.run()
    return srmi


def _collect(srmi, var_name):
    import narwhals as nw

    return nw.from_native(srmi.implicates[0].df).lazy().collect().to_native()


#   ============================================================
#   1) error: Random vs. pmm
#   ============================================================
logger.info("=== error=Random vs error=pmm ===")

rng1 = np.random.default_rng(20260910)
n1 = 1500
x1_e = rng1.normal(size=n1)
y_e = 2.0 * x1_e + rng1.normal(scale=1.0, size=n1)
df_e = pl.DataFrame(dict(idx=range(n1), x1=x1_e, y=y_e))
miss_e = rng1.random(n1) < 0.2
observed_e_values = set(df_e.filter(~pl.Series(miss_e))["y"].to_list())
df_e = df_e.with_columns(
    pl.when(pl.Series(miss_e)).then(None).otherwise(pl.col("y")).alias("y")
)

v_pmm = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(error=Parameters.ErrorDraw.pmm),
)
srmi_pmm = _build_and_run(df_e, v_pmm, "error_pmm")
log_pmm = read_variable_log(srmi_pmm, "y")
assert "error=pmm:" in log_pmm, "expected the error=pmm log marker"

imputed_pmm = (
    _collect(srmi_pmm, "y")
    .filter(pl.col("idx").is_in(pl.Series(miss_e.nonzero()[0].tolist())))["y"]
    .to_list()
)
assert all(v in observed_e_values for v in imputed_pmm), (
    "error=pmm should only ever donate real OBSERVED values - found an imputed "
    "value that was never actually observed"
)

v_random = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(error=Parameters.ErrorDraw.Random),
)
srmi_random = _build_and_run(df_e, v_random, "error_random")
log_random = read_variable_log(srmi_random, "y")
assert "error=Random:" in log_random, "expected the error=Random log marker"

imputed_random = (
    _collect(srmi_random, "y")
    .filter(pl.col("idx").is_in(pl.Series(miss_e.nonzero()[0].tolist())))["y"]
    .to_list()
)
n_exact_matches_random = sum(1 for v in imputed_random if v in observed_e_values)
assert n_exact_matches_random < len(imputed_random) * 0.2, (
    "error=Random should draw novel values (predicted + a residual draw), not "
    "mostly exact matches to observed values - got "
    f"{n_exact_matches_random}/{len(imputed_random)} exact matches"
)
logger.info("error=Random vs error=pmm: PASSED")


#   ============================================================
#   2) quantiles
#   ============================================================
logger.info("=== quantiles ===")

rng2 = np.random.default_rng(20260910)
n2 = 1500
x1_q = rng2.normal(size=n2)
y_q = 2.0 * x1_q + rng2.normal(scale=1.0, size=n2)
df_q = pl.DataFrame(dict(idx=range(n2), x1=x1_q, y=y_q))
miss_q = rng2.random(n2) < 0.2
df_q = df_q.with_columns(
    pl.when(pl.Series(miss_q)).then(None).otherwise(pl.col("y")).alias("y")
)

v_no_quantiles = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(error=Parameters.ErrorDraw.pmm),
)
srmi_no_q = _build_and_run(df_q, v_no_quantiles, "no_quantiles")
log_no_q = read_variable_log(srmi_no_q, "y")
assert "Running LightGBM for q=" not in log_no_q, (
    "no quantiles were requested - the quantile-regression path shouldn't have run"
)

v_quantiles = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(
        quantiles=[0.1, 0.5, 0.9], error=Parameters.ErrorDraw.pmm
    ),
)
srmi_q = _build_and_run(df_q, v_quantiles, "quantiles")
log_q = read_variable_log(srmi_q, "y")
for qi in ["0.1", "0.5", "0.9"]:
    assert f"Running LightGBM for q={qi}" in log_q, f"expected q={qi} to have run"
logger.info("quantiles: PASSED")


#   ============================================================
#   3) donate_list
#   ============================================================
logger.info("=== donate_list ===")

rng3 = np.random.default_rng(20260910)
n3 = 1500
x1_d = rng3.normal(size=n3)
primary = 2.0 * x1_d + rng3.normal(scale=1.0, size=n3)
#   Deterministic function of primary - if companion is donated from the
#       SAME donor as primary, companion == primary * 10 exactly, even
#       after imputation (a different row's original values got donated in)
companion = primary * 10.0

df_d = pl.DataFrame(dict(idx=range(n3), x1=x1_d, primary=primary, companion=companion))
miss_d = rng3.random(n3) < 0.2
df_d = df_d.with_columns(
    pl.when(pl.Series(miss_d)).then(None).otherwise(pl.col("primary")).alias("primary")
)

v_no_donate_list = Variable(
    impute_var="primary",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(error=Parameters.ErrorDraw.pmm),
)
srmi_no_dl = _build_and_run(df_d, v_no_donate_list, "no_donate_list")
result_no_dl = _collect(srmi_no_dl, "primary").filter(pl.Series(miss_d))
n_consistent_no_dl = result_no_dl.filter(
    (pl.col("companion") - pl.col("primary") * 10.0).abs() < 1e-6
).height
assert n_consistent_no_dl < result_no_dl.height, (
    "without donate_list, companion should NOT generally equal the newly "
    "imputed primary * 10 (it's untouched, from a different original row)"
)

v_donate_list = Variable(
    impute_var="primary",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(
        error=Parameters.ErrorDraw.pmm, parameters_pmm=Parameters.pmm(donate_list=["companion"])
    ),
)
srmi_dl = _build_and_run(df_d, v_donate_list, "donate_list")
log_dl = read_variable_log(srmi_dl, "primary")
assert "['primary', 'companion']" in log_dl, (
    f"expected the donate log to show both variables together, got: {log_dl}"
)
result_dl = _collect(srmi_dl, "primary").filter(pl.Series(miss_d))
n_consistent_dl = result_dl.filter(
    (pl.col("companion") - pl.col("primary") * 10.0).abs() < 1e-6
).height
assert n_consistent_dl == result_dl.height, (
    "with donate_list=['companion'], EVERY imputed row's companion should equal "
    f"the SAME donor's primary * 10 exactly - only {n_consistent_dl}/{result_dl.height} did"
)
logger.info("donate_list: PASSED")


#   ============================================================
#   4) donate_by
#   ============================================================
logger.info("=== donate_by ===")

rng4 = np.random.default_rng(20260910)
n4 = 2000
group = rng4.choice(["a", "b"], size=n4)
#   No predictor carries any group signal - without donate_by, PMM's
#       nearest-neighbor match on a near-constant yhat is effectively
#       random across ALL rows, so cross-group contamination is expected
group_offset = np.where(group == "a", 1000.0, -1000.0)
y_g = group_offset + rng4.normal(scale=1.0, size=n4)
x1_g = rng4.normal(size=n4)  # pure noise, no real signal for y

df_g = pl.DataFrame(dict(idx=range(n4), x1=x1_g, group=group, y=y_g))
miss_g = rng4.random(n4) < 0.2
df_g = df_g.with_columns(
    pl.when(pl.Series(miss_g)).then(None).otherwise(pl.col("y")).alias("y")
)

v_no_donate_by = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(error=Parameters.ErrorDraw.pmm),
)
srmi_no_db = _build_and_run(df_g, v_no_donate_by, "no_donate_by")
result_no_db = _collect(srmi_no_db, "y").filter(pl.Series(miss_g))
n_wrong_side_no_db = result_no_db.filter(
    ((pl.col("group") == "a") & (pl.col("y") < 0))
    | ((pl.col("group") == "b") & (pl.col("y") > 0))
).height
assert n_wrong_side_no_db > 0, (
    "without donate_by, some cross-group contamination was expected (predictors "
    "carry no group signal) but every imputed value landed on its own group's side"
)

v_donate_by = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(
        error=Parameters.ErrorDraw.pmm, parameters_pmm=Parameters.pmm(donate_by="group")
    ),
)
srmi_db = _build_and_run(df_g, v_donate_by, "donate_by")
log_db = read_variable_log(srmi_db, "y")
assert "donate_by=['group']" in log_db, f"expected the donate_by log marker, got: {log_db}"
result_db = _collect(srmi_db, "y").filter(pl.Series(miss_g))
n_wrong_side_db = result_db.filter(
    ((pl.col("group") == "a") & (pl.col("y") < 0))
    | ((pl.col("group") == "b") & (pl.col("y") > 0))
).height
assert n_wrong_side_db == 0, (
    f"with donate_by='group', every imputed value should stay on its own group's "
    f"side - {n_wrong_side_db}/{result_db.height} didn't"
)
logger.info("donate_by: PASSED")


#   ============================================================
#   5) raw parameters (e.g. num_leaves) actually reach lgb.train()
#   ============================================================
logger.info("=== raw parameters (num_leaves) pass-through ===")

rng5 = np.random.default_rng(20260910)
n5 = 1500
x1_p = rng5.normal(size=n5)
y_p = 2.0 * x1_p + rng5.normal(scale=1.0, size=n5)
df_p = pl.DataFrame(dict(idx=range(n5), x1=x1_p, y=y_p))
miss_p = rng5.random(n5) < 0.2
df_p = df_p.with_columns(
    pl.when(pl.Series(miss_p)).then(None).otherwise(pl.col("y")).alias("y")
)

v_restricted = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(
        parameters={"num_leaves": 2, "num_iterations": 10}, error=Parameters.ErrorDraw.pmm
    ),
)
with capture_global_log() as get_log:
    srmi_restricted = _build_and_run(df_p, v_restricted, "num_leaves_2")
    log_restricted = get_log()
assert "'num_leaves': 2," in log_restricted, (
    f"expected num_leaves=2 to actually reach lgb.train(), log shows: {log_restricted[-800:]}"
)

v_flexible = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(
        parameters={"num_leaves": 200, "num_iterations": 200}, error=Parameters.ErrorDraw.pmm
    ),
)
with capture_global_log() as get_log:
    srmi_flexible = _build_and_run(df_p, v_flexible, "num_leaves_200")
    log_flexible = get_log()
assert "'num_leaves': 200," in log_flexible, (
    f"expected num_leaves=200 to actually reach lgb.train(), log shows: {log_flexible[-800:]}"
)
logger.info("raw parameters pass-through: PASSED")


#   ============================================================
#   6) tune / tuner
#   ============================================================
logger.info("=== tune/tuner ===")

rng6 = np.random.default_rng(20260910)
n6 = 1500
x1_t = rng6.normal(size=n6)
y_t = 2.0 * x1_t + rng6.normal(scale=1.0, size=n6)
df_t = pl.DataFrame(dict(idx=range(n6), x1=x1_t, y=y_t))
miss_t = rng6.random(n6) < 0.2
df_t = df_t.with_columns(
    pl.when(pl.Series(miss_t)).then(None).otherwise(pl.col("y")).alias("y")
)

#   A narrow range far from any plausible untouched/default num_leaves -
#       if tuning actually ran and its result reached the final fit, the
#       log's final num_leaves must land inside [2, 6]
tuner = Tuner(
    space=HyperparameterSpace(num_leaves=IntRange(2, 6)),
    n_trials=5,
    path_save_dir=f"{path_scratch}/py_lgbm_param_effects_tune/tuner_outputs",
    overwrite=True,
)

v_tuned = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.LightGBM,
    parameters=Parameters.LightGBM(
        tune=True,
        tuner=tuner,
        parameters={"num_leaves": 200, "test_size": 0.25, "verbose": -1},
        error=Parameters.ErrorDraw.pmm,
    ),
)
with capture_global_log() as get_log:
    srmi_tuned = _build_and_run(df_t, v_tuned, "tune")
    log_tuned = get_log()
assert "trials finished" in log_tuned, "expected tuning to have actually run"

import re

matches = re.findall(r"'num_leaves': (\d+)", log_tuned)
assert len(matches) > 0, f"couldn't find a final num_leaves in the log: {log_tuned[:500]}"
final_num_leaves = int(matches[-1])
assert 2 <= final_num_leaves <= 6, (
    f"expected the FINAL fit to use the tuned num_leaves (2-6), not the "
    f"untouched base value (200) - got {final_num_leaves}"
)
logger.info("tune/tuner: PASSED")

logger.info("lgbm_parameter_effects.py: all checks passed")
