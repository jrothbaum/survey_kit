"""
Same convention as lgbm_parameter_effects.py/regression_parameter_effects.py,
for Parameters.TwoSampleRegression() - see that file's own module docstring
for the overall rationale. Not part of the fast day-to-day regression loop -
run when imputation code changes.
"""

import os
import shutil

import numpy as np
import polars as pl

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.srmi import SRMI
from survey_kit import logger, config

from _imputation_test_utils import read_variable_log

path_scratch = config.path_temp_files


def _build_and_run(df, variable, path_suffix, index="idx", n_iterations=1):
    srmi = SRMI(
        df=df,
        variables=[variable],
        index=[index],
        replication=SRMI.Replication(n_implicates=1, n_iterations=n_iterations),
        parallel=SRMI.Parallel(enabled=False),
        bootstrap=SRMI.Bootstrap(enabled=False),
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_two_sample_param_effects_{path_suffix}",
            force_start=True,
        ),
    )
    srmi.run()
    return srmi


def _collect(srmi):
    import narwhals as nw

    return nw.from_native(srmi.implicates[0].df).lazy().collect().to_native()


#   ============================================================
#   1) fit + path_save, then load_from_save on a genuinely separate
#      sample that never has the model sample in memory
#   ============================================================
logger.info("=== fit+save vs. load_from_save on a disjoint sample ===")

path_save_1 = f"{path_scratch}/two_sample_param_effects_save_1"
shutil.rmtree(path_save_1, ignore_errors=True)

rng1 = np.random.default_rng(20260916)
n1 = 3000
x1_a = rng1.normal(size=n1)
x2_a = rng1.normal(size=n1)
y_a = 3.0 * x1_a - 2.0 * x2_a + rng1.normal(scale=1.0, size=n1)
df_a = pl.DataFrame(dict(idx=range(n1), x1=x1_a, x2=x2_a, y=y_a))
miss_a = rng1.random(n1) < 0.3
df_a = df_a.with_columns(
    pl.when(pl.Series(miss_a)).then(None).otherwise(pl.col("y")).alias("y")
)

v_fit = Variable(
    impute_var="y",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.TwoSampleRegression,
    parameters=Parameters.TwoSampleRegression(path_save=path_save_1, bins=5),
)
srmi_fit = _build_and_run(df_a, v_fit, "fit")
assert "load_from_save=False" in read_variable_log(srmi_fit, "y")
assert "Saved two-sample regression model to" in read_variable_log(srmi_fit, "y")
imputed_a = _collect(srmi_fit).filter(pl.Series(miss_a))["y"]
assert imputed_a.null_count() == 0, "every recipient row should get a drawn value"
assert abs(imputed_a.mean() - y_a.mean()) < 0.5, (
    f"fit+save imputed mean {imputed_a.mean()} too far from true mean {y_a.mean()}"
)

#   A second, disjoint sample - x1_b/x2_b/y_b never appear in df_a, and this
#   run's Impute never sees df_a at all.
n2 = 900
x1_b = rng1.normal(size=n2)
x2_b = rng1.normal(size=n2)
true_y_b = 3.0 * x1_b - 2.0 * x2_b + rng1.normal(scale=1.0, size=n2)
df_b = pl.DataFrame(
    dict(idx=range(n2), x1=x1_b, x2=x2_b, y=[None] * n2)
).with_columns(pl.col("y").cast(pl.Float64))

v_load = Variable(
    impute_var="y",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.TwoSampleRegression,
    parameters=Parameters.TwoSampleRegression(
        path_load=path_save_1, load_from_save=True
    ),
)
srmi_load = _build_and_run(df_b, v_load, "load")
log_load = read_variable_log(srmi_load, "y")
assert "load_from_save=True" in log_load
assert "Loaded two-sample regression model from" in log_load
imputed_b = _collect(srmi_load)["y"]
assert imputed_b.null_count() == 0, "load_from_save should impute every row"
assert abs(imputed_b.mean() - true_y_b.mean()) < 0.7, (
    f"load_from_save imputed mean {imputed_b.mean()} too far from true mean "
    f"{true_y_b.mean()} - reloaded model isn't reproducing the fit"
)
logger.info("fit+save vs. load_from_save: PASSED")


#   ============================================================
#   2) is_boolean - plausible, non-degenerate share, tracking the model's
#      own predicted probability rather than a fixed marginal
#   ============================================================
logger.info("=== is_boolean ===")

rng2 = np.random.default_rng(20260916)
n3 = 3000
x1_c = rng2.normal(size=n3)
p_true = 1 / (1 + np.exp(-1.5 * x1_c))
y_c = (rng2.random(n3) < p_true).astype(bool)
df_c = pl.DataFrame(dict(idx=range(n3), x1=x1_c, y=y_c))
miss_c = rng2.random(n3) < 0.25
df_c = df_c.with_columns(
    pl.when(pl.Series(miss_c)).then(None).otherwise(pl.col("y")).alias("y")
)

v_bool = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.TwoSampleRegression,
    parameters=Parameters.TwoSampleRegression(is_boolean=True, bins=5),
)
srmi_bool = _build_and_run(df_c, v_bool, "boolean")
result_bool = _collect(srmi_bool).filter(pl.Series(miss_c))
assert result_bool["y"].dtype == pl.Boolean
share_true = result_bool["y"].cast(pl.Int8).mean()
assert 0.05 < share_true < 0.95, (
    f"is_boolean should give a plausible mix of True/False, not a degenerate "
    f"always-True/always-False result - got share_true={share_true}"
)
logger.info("is_boolean: PASSED")


#   ============================================================
#   3) draw_error: True (draw a residual, add to yhat) vs. False (draw y
#      directly) - both valid, but should give measurably different value
#      distributions since one is centered on yhat's own bin-average and
#      the other on the bin's raw y distribution
#   ============================================================
logger.info("=== draw_error True vs False ===")

rng3 = np.random.default_rng(20260916)
n4 = 2500
x1_d = rng3.normal(size=n4)
y_d = 4.0 * x1_d + rng3.normal(scale=2.0, size=n4)
df_d = pl.DataFrame(dict(idx=range(n4), x1=x1_d, y=y_d))
miss_d = rng3.random(n4) < 0.3
df_d = df_d.with_columns(
    pl.when(pl.Series(miss_d)).then(None).otherwise(pl.col("y")).alias("y")
)

v_direct = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.TwoSampleRegression,
    parameters=Parameters.TwoSampleRegression(draw_error=False, bins=5),
)
srmi_direct = _build_and_run(df_d, v_direct, "draw_direct")
imputed_direct = _collect(srmi_direct).filter(pl.Series(miss_d))["y"]

v_err = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.TwoSampleRegression,
    parameters=Parameters.TwoSampleRegression(draw_error=True, bins=5),
)
srmi_err = _build_and_run(df_d, v_err, "draw_error")
imputed_err = _collect(srmi_err).filter(pl.Series(miss_d))["y"]

assert imputed_direct.null_count() == 0 and imputed_err.null_count() == 0
assert abs(imputed_direct.mean() - y_d.mean()) < 0.5
assert abs(imputed_err.mean() - y_d.mean()) < 0.5
#   Different draw mechanisms over the same data/seed-free RNG shouldn't
#   produce an (almost) identical set of imputed values.
n_close = sum(
    1
    for a, b in zip(sorted(imputed_direct.to_list()), sorted(imputed_err.to_list()))
    if abs(a - b) < 1e-9
)
assert n_close < len(imputed_direct) * 0.5, (
    "draw_error=True/False should give measurably different draws"
)
logger.info("draw_error True vs False: PASSED")


#   ============================================================
#   4) save_percentile_cuts freezes bin cutoffs across repeated calls
#      (e.g. SRMI iterations) while the regression/distribution still
#      refit fresh every time - independent of load_from_save
#   ============================================================
logger.info("=== save_percentile_cuts freeze ===")

path_freeze = f"{path_scratch}/two_sample_param_effects_freeze"
shutil.rmtree(path_freeze, ignore_errors=True)

v_freeze = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.TwoSampleRegression,
    parameters=Parameters.TwoSampleRegression(
        bins=4, path_save=path_freeze, save_percentile_cuts=True
    ),
)
srmi_freeze = _build_and_run(df_d, v_freeze, "freeze", n_iterations=2)
cuts_path = f"{path_freeze}/y_cuts.csv"
assert os.path.isfile(cuts_path), "save_percentile_cuts should freeze cuts to disk"

log_iter1 = read_variable_log(srmi_freeze, "y", iteration=1)
log_iter2 = read_variable_log(srmi_freeze, "y", iteration=2)
assert "Froze percentile cutoffs to" in log_iter1
assert "Reusing frozen percentile cutoffs from" in log_iter2
#   The regression itself still refits fresh each iteration (a fresh
#   R2 line logs both times) even though the bin boundaries didn't move.
assert "R2 =" in log_iter1 and "R2 =" in log_iter2
logger.info("save_percentile_cuts freeze: PASSED")


#   ============================================================
#   5) min_n_x_var actually restricts predictors (logged), and a
#      C(...)/factor term in model= is rejected upfront (portability of
#      the persisted model - see Parameters.TwoSampleRegression()'s
#      docstring)
#   ============================================================
logger.info("=== min_n_x_var / numeric-only validation ===")

v_min_n = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.TwoSampleRegression,
    parameters=Parameters.TwoSampleRegression(min_n_x_var=5, bins=4),
)
srmi_min_n = _build_and_run(df_d, v_min_n, "min_n_x_var")
assert "Restricting to X variables with more than 5 observations" in read_variable_log(
    srmi_min_n, "y"
)

try:
    v_bad = Variable(
        impute_var="y",
        model="~x1 + C(x1)",
        modeltype=Variable.ModelType.TwoSampleRegression,
        parameters=Parameters.TwoSampleRegression(),
    )
    v_bad.validate_inputs(df_d)
    raise AssertionError("expected ValueError for a C(...) term in model=")
except ValueError as e:
    assert "C(...)" in str(e)
logger.info("min_n_x_var / numeric-only validation: PASSED")


#   ============================================================
#   6) save_disclosure_support is opt-in - off by default (no audit
#      files persisted at all), on when asked (adds the audit files
#      without changing distribution.csv or the actual draw)
#   ============================================================
logger.info("=== save_disclosure_support ===")

path_disclosure_off = f"{path_scratch}/two_sample_param_effects_disclosure_off"
path_disclosure_on = f"{path_scratch}/two_sample_param_effects_disclosure_on"
shutil.rmtree(path_disclosure_off, ignore_errors=True)
shutil.rmtree(path_disclosure_on, ignore_errors=True)

v_disclosure_off = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.TwoSampleRegression,
    parameters=Parameters.TwoSampleRegression(path_save=path_disclosure_off, bins=5),
)
_build_and_run(df_d, v_disclosure_off, "disclosure_off")
var_dir_off = f"{path_disclosure_off}/y"
files_off = set(os.listdir(var_dir_off))
assert files_off == {"meta.json", "betas.csv", "cuts.csv", "distribution.csv"}, files_off

v_disclosure_on = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.TwoSampleRegression,
    parameters=Parameters.TwoSampleRegression(
        path_save=path_disclosure_on, bins=5, save_disclosure_support=True
    ),
)
_build_and_run(df_d, v_disclosure_on, "disclosure_on")
var_dir_on = f"{path_disclosure_on}/y"
files_on = set(os.listdir(var_dir_on))
assert "disclosure_support_bin_counts.csv" in files_on
assert "disclosure_support_n_at_cuts.csv" in files_on
bin_counts = pl.read_csv(f"{var_dir_on}/disclosure_support_bin_counts.csv")
n_at_cuts = pl.read_csv(f"{var_dir_on}/disclosure_support_n_at_cuts.csv")
distribution = pl.read_csv(f"{var_dir_on}/distribution.csv")
assert "n" not in distribution.columns, (
    "cell counts shouldn't leak into the main distribution.csv even when "
    "save_disclosure_support is on"
)
assert bin_counts.height == 5 and bin_counts["n"].sum() > 0
assert n_at_cuts.height == 4
logger.info("save_disclosure_support: PASSED")


logger.info("two_sample_regression_parameter_effects.py: all checks passed")
