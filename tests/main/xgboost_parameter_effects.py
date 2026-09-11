"""
Same convention as lgbm_parameter_effects.py, for Parameters.XGBoost() -
see that file's own module docstring for the overall rationale. Not part
of the fast day-to-day regression loop - run when imputation code changes.
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
            path_model=f"{path_scratch}/py_xgb_param_effects_{path_suffix}", force_start=True
        ),
    )
    srmi.run()
    return srmi


def _collect(srmi):
    import narwhals as nw

    return nw.from_native(srmi.implicates[0].df).lazy().collect().to_native()


#   ============================================================
#   1) cv_folds
#   ============================================================
logger.info("=== cv_folds ===")

rng1 = np.random.default_rng(20260910)
n1 = 1500
x1_c = rng1.normal(size=n1)
y_c = 2.0 * x1_c + rng1.normal(scale=3.0, size=n1)
df_c = pl.DataFrame(dict(idx=range(n1), x1=x1_c, y=y_c))
miss_c = rng1.random(n1) < 0.2
df_c = df_c.with_columns(
    pl.when(pl.Series(miss_c)).then(None).otherwise(pl.col("y")).alias("y")
)
#   Deliberately overfittable (deep, unregularized trees, many rounds)
overfit_xgb_params = dict(max_depth=12, n_estimators=300, learning_rate=0.3, reg_lambda=0)

v_cv_off = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(
        parameters=overfit_xgb_params, error=Parameters.ErrorDraw.pmm, cv_folds=0
    ),
)
srmi_cv_off = _build_and_run(df_c, v_cv_off, "cv_off")
assert "cv_folds=" not in read_variable_log(srmi_cv_off, "y")

v_cv_on = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(
        parameters=overfit_xgb_params, error=Parameters.ErrorDraw.pmm, cv_folds=5
    ),
)
srmi_cv_on = _build_and_run(df_c, v_cv_on, "cv_on")
assert "cv_folds=5: donor pool prediction via 5-fold cross-validation" in read_variable_log(
    srmi_cv_on, "y"
)

y_off = _collect(srmi_cv_off).filter(pl.Series(miss_c))["y"]
y_on = _collect(srmi_cv_on).filter(pl.Series(miss_c))["y"]
n_diff = (y_off != y_on).sum()
assert n_diff > 0, "cv_folds=5 should change the donor pool prediction vs. cv_folds=0"
logger.info("cv_folds: PASSED")


#   ============================================================
#   2) error: Random / pmm / leaf
#   ============================================================
logger.info("=== error: Random / pmm / leaf ===")

rng2 = np.random.default_rng(20260910)
n2 = 1500
x1_e = rng2.normal(size=n2)
y_e = 2.0 * x1_e + rng2.normal(scale=1.0, size=n2)
df_e = pl.DataFrame(dict(idx=range(n2), x1=x1_e, y=y_e))
miss_e = rng2.random(n2) < 0.2
observed_e_values = set(df_e.filter(~pl.Series(miss_e))["y"].to_list())
df_e = df_e.with_columns(
    pl.when(pl.Series(miss_e)).then(None).otherwise(pl.col("y")).alias("y")
)

v_pmm = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(error=Parameters.ErrorDraw.pmm),
)
srmi_pmm = _build_and_run(df_e, v_pmm, "error_pmm")
assert "error=pmm:" in read_variable_log(srmi_pmm, "y")
imputed_pmm = _collect(srmi_pmm).filter(pl.Series(miss_e))["y"].to_list()
assert all(v in observed_e_values for v in imputed_pmm)

v_random = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(error=Parameters.ErrorDraw.Random),
)
srmi_random = _build_and_run(df_e, v_random, "error_random")
assert "error=Random:" in read_variable_log(srmi_random, "y")
imputed_random = _collect(srmi_random).filter(pl.Series(miss_e))["y"].to_list()
n_exact = sum(1 for v in imputed_random if v in observed_e_values)
assert n_exact < len(imputed_random) * 0.2

v_leaf = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(error=Parameters.ErrorDraw.leaf),
)
srmi_leaf = _build_and_run(df_e, v_leaf, "error_leaf")
log_leaf = read_variable_log(srmi_leaf, "y")
assert "Extracting leaf indices" in log_leaf
imputed_leaf = _collect(srmi_leaf).filter(pl.Series(miss_e))["y"].to_list()
assert all(v in observed_e_values for v in imputed_leaf)
logger.info("error: Random / pmm / leaf - PASSED")


#   ============================================================
#   3) categorical_feature - native support (unlike RandomForest) -
#      a raw string predictor via LIST-form should work directly once
#      declared
#   ============================================================
logger.info("=== categorical_feature (native) ===")

rng3 = np.random.default_rng(20260910)
n3 = 1000
cat_levels = {"a": 0.0, "b": 5.0, "c": -5.0}
cat = rng3.choice(list(cat_levels.keys()), size=n3)
x1_cat = rng3.normal(size=n3)
y_cat = x1_cat + np.array([cat_levels[c] for c in cat]) + rng3.normal(scale=1.0, size=n3)
df_cat = pl.DataFrame(dict(idx=range(n3), x1=x1_cat, cat=cat, y=y_cat))
miss_cat = rng3.random(n3) < 0.2
df_cat = df_cat.with_columns(
    pl.when(pl.Series(miss_cat)).then(None).otherwise(pl.col("y")).alias("y")
)

v_no_cat_declared = Variable(
    impute_var="y",
    model=["x1", "cat"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(error=Parameters.ErrorDraw.pmm),
)
try:
    _build_and_run(df_cat, v_no_cat_declared, "cat_undeclared_crash")
    raise AssertionError(
        "expected a raw string predictor to fail when categorical_feature isn't "
        "declared, even for XGBoost (native support requires the declaration)"
    )
except AssertionError:
    raise
except Exception as e:
    logger.info(f"Correctly rejected (categorical_feature not declared): {type(e).__name__}: {e}")

v_cat_declared = Variable(
    impute_var="y",
    model=["x1", "cat"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(error=Parameters.ErrorDraw.pmm, categorical_feature=["cat"]),
)
srmi_cat = _build_and_run(df_cat, v_cat_declared, "cat_declared")
imputed_cat = _collect(srmi_cat).filter(pl.Series(miss_cat))["y"]
assert imputed_cat.is_not_null().all(), "categorical_feature=['cat'] should let this run cleanly"
logger.info("categorical_feature (native): PASSED")


#   ============================================================
#   4) group_levels/group_shrinkage_k
#   ============================================================
logger.info("=== group_levels ===")

rng4 = np.random.default_rng(20260910)
n4 = 2000
group = rng4.choice(["a", "b"], size=n4)
group_offset = np.where(group == "a", 50.0, -50.0)
x1_g = rng4.normal(size=n4)
y_g = group_offset + rng4.normal(scale=2.0, size=n4)
df_g = pl.DataFrame(dict(idx=range(n4), x1=x1_g, group=group, y=y_g))
miss_g = rng4.random(n4) < 0.2
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
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(error=Parameters.ErrorDraw.Random),
)
srmi_no_group = _build_and_run(df_g, v_no_group, "no_group_levels")
mae_no_group = _mae_vs_group_truth(srmi_no_group)

v_group = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(
        error=Parameters.ErrorDraw.Random, group_levels=["group"], group_shrinkage_k=1.0
    ),
)
srmi_group = _build_and_run(df_g, v_group, "group_levels")
mae_group = _mae_vs_group_truth(srmi_group)
assert mae_group < mae_no_group * 0.85, (
    f"group_levels should pull predictions closer to each group's own mean - "
    f"MAE with ({mae_group:.2f}) wasn't meaningfully better than without ({mae_no_group:.2f})"
)
logger.info(f"group_levels: MAE {mae_no_group:.2f} -> {mae_group:.2f} - PASSED")

logger.info("xgboost_parameter_effects.py: all checks passed")
