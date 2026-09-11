"""
Same convention as lgbm_parameter_effects.py, for Parameters.Regression()/
pmm() (plain OLS/Logit) - see that file's own module docstring for the
overall rationale. Not part of the fast day-to-day regression loop - run
when imputation code changes.
"""

import numpy as np
import polars as pl

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.srmi import SRMI
from survey_kit import logger, config

from _imputation_test_utils import read_variable_log

path_scratch = config.path_temp_files


def _build_and_run(df, variable, path_suffix, index="idx"):
    srmi = SRMI(
        df=df,
        variables=[variable],
        index=[index],
        replication=SRMI.Replication(n_implicates=1, n_iterations=1),
        parallel=SRMI.Parallel(enabled=False),
        bootstrap=SRMI.Bootstrap(enabled=False),
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_regression_param_effects_{path_suffix}",
            force_start=True,
        ),
    )
    srmi.run()
    return srmi


def _collect(srmi):
    import narwhals as nw

    return nw.from_native(srmi.implicates[0].df).lazy().collect().to_native()


#   ============================================================
#   1) error: Random vs. pmm (OLS)
#   ============================================================
logger.info("=== error=Random vs error=pmm (OLS) ===")

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
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(error=Parameters.ErrorDraw.pmm),
)
srmi_pmm = _build_and_run(df_e, v_pmm, "error_pmm")
assert "error=pmm:" in read_variable_log(srmi_pmm, "y")
imputed_pmm = _collect(srmi_pmm).filter(pl.Series(miss_e))["y"].to_list()
assert all(v in observed_e_values for v in imputed_pmm), (
    "error=pmm should only ever donate real observed values"
)

v_random = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(error=Parameters.ErrorDraw.Random),
)
srmi_random = _build_and_run(df_e, v_random, "error_random")
assert "error=Random:" in read_variable_log(srmi_random, "y")
imputed_random = _collect(srmi_random).filter(pl.Series(miss_e))["y"].to_list()
n_exact = sum(1 for v in imputed_random if v in observed_e_values)
assert n_exact < len(imputed_random) * 0.2, (
    f"error=Random should draw novel values, got {n_exact}/{len(imputed_random)} exact matches"
)
logger.info("error=Random vs error=pmm (OLS): PASSED")


#   ============================================================
#   2) model=Logit - imputes only True/False, and Random draws a
#      plausible mix (not degenerate) tracking the predicted probability
#   ============================================================
logger.info("=== model=Logit ===")

rng2 = np.random.default_rng(20260910)
n2 = 3000
x1_l = rng2.normal(size=n2)
p_true = 1 / (1 + np.exp(-2.0 * x1_l))
y_l = (rng2.random(n2) < p_true).astype(bool)
df_l = pl.DataFrame(dict(idx=range(n2), x1=x1_l, y=y_l))
miss_l = rng2.random(n2) < 0.2
df_l = df_l.with_columns(
    pl.when(pl.Series(miss_l)).then(None).otherwise(pl.col("y")).alias("y")
)

v_logit_random = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(
        model=Parameters.RegressionModel.Logit, error=Parameters.ErrorDraw.Random
    ),
)
srmi_logit = _build_and_run(df_l, v_logit_random, "logit_random")
result_logit = _collect(srmi_logit).filter(pl.Series(miss_l))
share_true = result_logit["y"].cast(pl.Int8).mean()
assert 0.05 < share_true < 0.95, (
    f"error=Random on a Logit model should give a plausible mix of True/False, "
    f"not a degenerate always-True/always-False result - got share_true={share_true}"
)
logger.info("model=Logit: PASSED")


#   ============================================================
#   3) group_levels/group_shrinkage_k
#   ============================================================
logger.info("=== group_levels ===")

rng3 = np.random.default_rng(20260910)
n3 = 2000
group = rng3.choice(["a", "b"], size=n3)
group_offset = np.where(group == "a", 50.0, -50.0)
x1_g = rng3.normal(size=n3)  # no real group signal in the predictor
y_g = group_offset + rng3.normal(scale=2.0, size=n3)
df_g = pl.DataFrame(dict(idx=range(n3), x1=x1_g, group=group, y=y_g))
miss_g = rng3.random(n3) < 0.2
df_g = df_g.with_columns(
    pl.when(pl.Series(miss_g)).then(None).otherwise(pl.col("y")).alias("y")
)


def _mae_vs_group_truth(srmi):
    result = _collect(srmi).filter(pl.Series(miss_g))
    truth = result["group"].replace_strict({"a": 50.0, "b": -50.0}, return_dtype=pl.Float64)
    return (result["y"] - truth).abs().mean()


v_no_group = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(error=Parameters.ErrorDraw.Random),
)
srmi_no_group = _build_and_run(df_g, v_no_group, "no_group_levels")
mae_no_group = _mae_vs_group_truth(srmi_no_group)

v_group = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(
        error=Parameters.ErrorDraw.Random, group_levels=["group"], group_shrinkage_k=1.0
    ),
)
srmi_group = _build_and_run(df_g, v_group, "group_levels")
mae_group = _mae_vs_group_truth(srmi_group)

assert mae_group < mae_no_group * 0.5, (
    f"group_levels should pull predictions much closer to each group's own mean - "
    f"MAE with group_levels ({mae_group:.2f}) wasn't meaningfully better than without "
    f"({mae_no_group:.2f})"
)
logger.info(f"group_levels: MAE {mae_no_group:.2f} -> {mae_group:.2f} - PASSED")


#   ============================================================
#   4) donate_list (via parameters_pmm)
#   ============================================================
logger.info("=== donate_list ===")

rng4 = np.random.default_rng(20260910)
n4 = 1500
x1_d = rng4.normal(size=n4)
primary = 2.0 * x1_d + rng4.normal(scale=1.0, size=n4)
companion = primary * 10.0
df_d = pl.DataFrame(dict(idx=range(n4), x1=x1_d, primary=primary, companion=companion))
miss_d = rng4.random(n4) < 0.2
df_d = df_d.with_columns(
    pl.when(pl.Series(miss_d)).then(None).otherwise(pl.col("primary")).alias("primary")
)

v_donate_list = Variable(
    impute_var="primary",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(
        error=Parameters.ErrorDraw.pmm, parameters_pmm=Parameters.pmm(donate_list=["companion"])
    ),
)
srmi_dl = _build_and_run(df_d, v_donate_list, "donate_list")
assert "['primary', 'companion']" in read_variable_log(srmi_dl, "primary")
result_dl = _collect(srmi_dl).filter(pl.Series(miss_d))
n_consistent = result_dl.filter(
    (pl.col("companion") - pl.col("primary") * 10.0).abs() < 1e-6
).height
assert n_consistent == result_dl.height, (
    f"with donate_list, every imputed row's companion should equal the same "
    f"donor's primary * 10 - only {n_consistent}/{result_dl.height} did"
)
logger.info("donate_list: PASSED")


#   ============================================================
#   5) donate_by (via parameters_pmm)
#   ============================================================
logger.info("=== donate_by ===")

rng5 = np.random.default_rng(20260910)
n5 = 2000
group_b = rng5.choice(["a", "b"], size=n5)
group_offset_b = np.where(group_b == "a", 1000.0, -1000.0)
x1_b = rng5.normal(size=n5)
y_b = group_offset_b + rng5.normal(scale=1.0, size=n5)
df_b = pl.DataFrame(dict(idx=range(n5), x1=x1_b, group=group_b, y=y_b))
miss_b = rng5.random(n5) < 0.2
df_b = df_b.with_columns(
    pl.when(pl.Series(miss_b)).then(None).otherwise(pl.col("y")).alias("y")
)

v_donate_by = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(
        error=Parameters.ErrorDraw.pmm, parameters_pmm=Parameters.pmm(donate_by="group")
    ),
)
srmi_db = _build_and_run(df_b, v_donate_by, "donate_by")
assert "donate_by=['group']" in read_variable_log(srmi_db, "y")
result_db = _collect(srmi_db).filter(pl.Series(miss_b))
n_wrong_side = result_db.filter(
    ((pl.col("group") == "a") & (pl.col("y") < 0))
    | ((pl.col("group") == "b") & (pl.col("y") > 0))
).height
assert n_wrong_side == 0, (
    f"with donate_by='group', every imputed value should stay on its own group's "
    f"side - {n_wrong_side}/{result_db.height} didn't"
)
logger.info("donate_by: PASSED")


#   ============================================================
#   6) random_share
#   ============================================================
logger.info("=== random_share ===")

rng6 = np.random.default_rng(20260910)
n6 = 1500
x1_r = rng6.normal(size=n6)
y_r = 2.0 * x1_r + rng6.normal(scale=1.0, size=n6)
df_r = pl.DataFrame(dict(idx=range(n6), x1=x1_r, y=y_r))
miss_r = rng6.random(n6) < 0.2
df_r = df_r.with_columns(
    pl.when(pl.Series(miss_r)).then(None).otherwise(pl.col("y")).alias("y")
)

v_full_share = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(error=Parameters.ErrorDraw.Random, random_share=1.0),
)
srmi_full = _build_and_run(df_r, v_full_share, "random_share_full")
assert "Using a" not in read_variable_log(srmi_full, "y"), (
    "random_share=1.0 shouldn't trigger the subsample log line"
)

v_partial_share = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.Regression,
    parameters=Parameters.Regression(error=Parameters.ErrorDraw.Random, random_share=0.3),
)
srmi_partial = _build_and_run(df_r, v_partial_share, "random_share_partial")
assert "Using a 0.3 subsample" in read_variable_log(srmi_partial, "y"), (
    "random_share=0.3 should log that a subsample was used"
)
logger.info("random_share: PASSED")


#   ============================================================
#   7) cv_folds is genuinely not a Regression() option (documented
#      limitation - plain OLS/Logit isn't a flexible-enough model for
#      "refit on folds to correct in-sample bias" to be meaningful) -
#      confirm it's a hard TypeError, not silently accepted and ignored
#   ============================================================
logger.info("=== cv_folds is rejected outright for Regression() ===")
try:
    Parameters.Regression(cv_folds=5)
    raise AssertionError("expected Parameters.Regression(cv_folds=5) to raise a TypeError")
except AssertionError:
    raise
except TypeError as e:
    logger.info(f"Correctly rejected: {e}")

logger.info("regression_parameter_effects.py: all checks passed")
