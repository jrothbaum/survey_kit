"""
Behavioral tests for the tune=True/tuner= mechanism now shared by
RandomForest()/XGBoost()/CatBoost()/SklearnModel() (see
utilities.tuning.Tuner.run_estimator, SRMI._preprocess_tune_estimator,
Impute._prepare_tuning_data/_tuned_model_factory) - the same tune-before-run
pattern LightGBM already had (see lgbm_parameter_effects.py's own tune/tuner
section), generalized to any sklearn-style estimator. Not part of the fast
day-to-day regression loop - run when imputation/tuning code changes.
"""

import os

import numpy as np
import polars as pl

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.srmi import SRMI
from survey_kit.imputation.utilities.tuning import (
    Tuner,
    HyperparameterSpace,
    IntRange,
    load_tuned_params,
)
from survey_kit import logger, config

from _imputation_test_utils import capture_global_log

path_scratch = config.path_temp_files


def _build_and_run(df, variable, path_suffix):
    srmi = SRMI(
        df=df,
        variables=[variable],
        index=["idx"],
        replication=SRMI.Replication(n_implicates=1, n_iterations=1),
        parallel=SRMI.Parallel(enabled=False),
        bootstrap=SRMI.Bootstrap(enabled=False),
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_estimator_tune_{path_suffix}",
            force_start=True,
        ),
    )
    srmi.run()
    return srmi


rng = np.random.default_rng(20260910)
n = 500
x1 = rng.normal(size=n)
y = 2.0 * x1 + rng.normal(scale=0.3, size=n)
df = pl.DataFrame(dict(idx=range(n), x1=x1, y=y))
miss = rng.random(n) < 0.2
df = df.with_columns(
    pl.when(pl.Series(miss)).then(None).otherwise(pl.col("y")).alias("y")
)


#   ============================================================
#   1) tune=True actually runs a search and saves a tuned-params file,
#      whose values land in the ACTUAL fit (not just on disk, unused)
#   ============================================================
logger.info("=== RandomForest tune=True: runs + saves + actually applied ===")

path_save_dir_1 = f"{path_scratch}/py_estimator_tune_rf1/tuner_outputs"
#   A narrow range far from RandomForestRegressor's own defaults
#       (n_estimators=100, max_depth=None) - if the tuned value reached the
#       ACTUAL fit, a directly-rebuilt model_factory() must show it.
tuner_1 = Tuner(
    space=HyperparameterSpace(n_estimators=IntRange(5, 8), max_depth=IntRange(2, 3)),
    n_trials=5,
    seed=1,
    path_save_dir=path_save_dir_1,
    overwrite=True,
)
v_rf = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        tune=True, tuner=tuner_1, error=Parameters.ErrorDraw.Random
    ),
)
with capture_global_log() as get_log:
    srmi_rf = _build_and_run(df, v_rf, "rf1")
    log_rf = get_log()
assert "trials finished" in log_rf, "expected the tuning pass to have actually run"

pickle_path_1 = f"{path_save_dir_1}/y.pickle"
assert os.path.exists(pickle_path_1), f"expected a tuned-params file at {pickle_path_1}"
tuned_1 = load_tuned_params(pickle_path_1)
assert 5 <= tuned_1["n_estimators"] <= 8, f"tuned n_estimators out of range: {tuned_1}"
assert 2 <= tuned_1["max_depth"] <= 3, f"tuned max_depth out of range: {tuned_1}"

#   Confirm the ACTUAL per-iteration model factory (Impute._tuned_model_factory)
#       applies these, not just that they're sitting on disk unused.
from survey_kit.imputation.impute import Impute
from sklearn.ensemble import RandomForestRegressor


class _FakeSRMI:
    pass


v_check = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(),
)
v_check.parameters["tune_hyperparameter_path"] = path_save_dir_1
impute_check = Impute(
    df=None, parent=_FakeSRMI(), variable=v_check, index=["idx"],
    variable_number=0, implicate_number=0,
)
wrapped_factory = impute_check._tuned_model_factory(lambda: RandomForestRegressor())
model = wrapped_factory()
assert model.n_estimators == tuned_1["n_estimators"], (
    f"the actual fit-time factory should apply the tuned n_estimators, "
    f"got {model.n_estimators}, expected {tuned_1['n_estimators']}"
)
assert model.max_depth == tuned_1["max_depth"]
logger.info("RandomForest tune=True: PASSED")


#   ============================================================
#   2) skip re-tuning when already tuned (overwrite=False); force re-tune
#      when overwrite=True
#   ============================================================
logger.info("=== overwrite=False skips, overwrite=True re-tunes ===")

tuner_2 = Tuner(
    space=HyperparameterSpace(n_estimators=IntRange(5, 8), max_depth=IntRange(2, 3)),
    n_trials=5,
    seed=2,
    path_save_dir=path_save_dir_1,  # same dir/file as section 1 - already tuned
    overwrite=False,
)
v_skip = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        tune=True, tuner=tuner_2, error=Parameters.ErrorDraw.Random
    ),
)
with capture_global_log() as get_log:
    _build_and_run(df, v_skip, "rf_skip")
    log_skip = get_log()
assert "trials finished" not in log_skip, (
    "a pickle already exists and overwrite=False - tuning should have been skipped"
)

tuner_3 = Tuner(
    space=HyperparameterSpace(n_estimators=IntRange(5, 8), max_depth=IntRange(2, 3)),
    n_trials=5,
    seed=3,
    path_save_dir=path_save_dir_1,
    overwrite=True,
)
v_overwrite = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        tune=True, tuner=tuner_3, error=Parameters.ErrorDraw.Random
    ),
)
with capture_global_log() as get_log:
    _build_and_run(df, v_overwrite, "rf_overwrite")
    log_overwrite = get_log()
assert "trials finished" in log_overwrite, (
    "overwrite=True should force a fresh tuning pass even though a pickle "
    "already exists"
)
logger.info("overwrite=False/True: PASSED")


#   ============================================================
#   3) One tuner instance reused across multiple variables (even across
#      different modeltypes) - each gets its own independent result, not
#      contaminated by the other's search (see Tuner's own fresh-study
#      guarantee)
#   ============================================================
logger.info("=== one tuner instance, multiple variables ===")

n2 = 500
x1_b = rng.normal(size=n2)
y_b = -5.0 * x1_b + rng.normal(scale=0.3, size=n2)  # very different scale/sign
df_b = pl.DataFrame(dict(idx=range(n2), x1=x1_b, y=y_b))
miss_b = rng.random(n2) < 0.2
df_b = df_b.with_columns(
    pl.when(pl.Series(miss_b)).then(None).otherwise(pl.col("y")).alias("y")
)

shared_tuner = Tuner(
    space=HyperparameterSpace(n_estimators=IntRange(5, 8), max_depth=IntRange(2, 3)),
    n_trials=5,
    seed=4,
    path_save_dir=f"{path_scratch}/py_estimator_tune_shared/tuner_outputs",
    overwrite=True,
)

v_a = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        tune=True, tuner=shared_tuner, error=Parameters.ErrorDraw.Random
    ),
)
_build_and_run(df, v_a, "shared_a")
tuned_a = load_tuned_params(shared_tuner.path_save)

v_c = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        tune=True, tuner=shared_tuner, error=Parameters.ErrorDraw.Random
    ),
)
_build_and_run(df_b, v_c, "shared_c")
tuned_c = load_tuned_params(shared_tuner.path_save)

#   Both results must be independently valid (in-range) - not proof of
#       independence on its own, but a badly-contaminated shared study
#       could easily land outside a deliberately narrow range like this.
assert 5 <= tuned_a["n_estimators"] <= 8 and 5 <= tuned_c["n_estimators"] <= 8
logger.info(f"shared tuner: var A -> {tuned_a}, var C -> {tuned_c} - PASSED")


#   ============================================================
#   4) SklearnModel (the generic factory= escape hatch) also supports tune=True
#   ============================================================
logger.info("=== SklearnModel tune=True ===")

from sklearn.linear_model import Ridge
from survey_kit.imputation.utilities.tuning import FloatRange

tuner_ridge = Tuner(
    space=HyperparameterSpace(alpha=FloatRange(0.01, 100.0, log=True)),
    n_trials=5,
    seed=5,
    path_save_dir=f"{path_scratch}/py_estimator_tune_ridge/tuner_outputs",
    overwrite=True,
)
v_ridge = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.SklearnModel,
    parameters=Parameters.SklearnModel(
        factory=lambda: Ridge(),
        tune=True,
        tuner=tuner_ridge,
        error=Parameters.ErrorDraw.Random,
    ),
)
_build_and_run(df, v_ridge, "ridge")
tuned_ridge = load_tuned_params(tuner_ridge.path_save)
assert 0.01 <= tuned_ridge["alpha"] <= 100.0
logger.info(f"SklearnModel tune=True: alpha={tuned_ridge['alpha']} - PASSED")


#   ============================================================
#   5) XGBoost + native categorical_feature + tune=True - Tuner.run_estimator's
#      X must stay a dataframe (not a bare numpy array) all the way through
#      cv-fold selection, or XGBoost's native categorical dtype gets
#      coerced away and .fit() crashes ("could not convert string to
#      float"). Also exercises Tuner.run_estimator's polars-specific
#      fold-selection path (plain df[bool_mask] isn't supported by polars -
#      .filter() is required).
#   ============================================================
logger.info("=== XGBoost + categorical_feature + tune=True ===")

n3 = 500
x1_cat = rng.normal(size=n3)
cat = rng.choice(["a", "b", "c"], size=n3)
cat_effect = np.select([cat == "a", cat == "b", cat == "c"], [0.0, 5.0, -5.0])
y_cat = 2.0 * x1_cat + cat_effect + rng.normal(scale=0.3, size=n3)
df_cat = pl.DataFrame(dict(idx=range(n3), x1=x1_cat, cat=cat, y=y_cat))
miss_cat = rng.random(n3) < 0.2
df_cat = df_cat.with_columns(
    pl.when(pl.Series(miss_cat)).then(None).otherwise(pl.col("y")).alias("y")
)

tuner_xgb_cat = Tuner(
    space=HyperparameterSpace(n_estimators=IntRange(5, 8), max_depth=IntRange(2, 3)),
    n_trials=5,
    seed=6,
    path_save_dir=f"{path_scratch}/py_estimator_tune_xgb_cat/tuner_outputs",
    overwrite=True,
)
v_xgb_cat = Variable(
    impute_var="y",
    model=["x1", "cat"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(
        tune=True,
        tuner=tuner_xgb_cat,
        categorical_feature="cat",
        error=Parameters.ErrorDraw.Random,
    ),
)
_build_and_run(df_cat, v_xgb_cat, "xgb_cat")
tuned_xgb_cat = load_tuned_params(tuner_xgb_cat.path_save)
assert 5 <= tuned_xgb_cat["n_estimators"] <= 8
logger.info(f"XGBoost + categorical_feature: {tuned_xgb_cat} - PASSED")

logger.info("estimator_tuning_effects.py: all checks passed")
