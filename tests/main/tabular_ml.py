"""
Coverage for this session's tabular-ML additions to imputation:
RandomForest()/XGBoost()/CatBoost()/SklearnModel()/Multinomial modeltypes,
cv_folds, categorical_feature (list-model and formula-model forms), and
save/load.

Nothing here was previously exercised by the test suite - it had only
been checked by hand, ad hoc, while building each feature.
"""

import numpy as np
import polars as pl
import narwhals as nw

from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import summary

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.srmi import SRMI

from survey_kit import logger, config

n_rows = 2_000
impute_share = 0.25

path_scratch = config.path_temp_files

df = (
    RandomData(n_rows=n_rows, seed=90210)
    .index("index")
    .float("x1", -3, 3)
    .float("x2", -3, 3)
    .np_distribution("epsilon_rf", "normal", scale=1)
    .np_distribution("epsilon_xgb", "normal", scale=1)
    .np_distribution("epsilon_cb", "normal", scale=1)
    .np_distribution("epsilon_sk", "normal", scale=1)
    .float("missing_rf", 0, 1)
    .float("missing_xgb", 0, 1)
    .float("missing_cb", 0, 1)
    .float("missing_sk", 0, 1)
    .to_df()
)

#   A native categorical predictor - shared across the XGBoost/CatBoost
#       variables to cover categorical_feature end to end.
rng = np.random.default_rng(90210)
cat_levels = {"a": 0.0, "b": 3.0, "c": -2.0}
df = df.with_columns(
    pl.Series("cat1", rng.choice(list(cat_levels.keys()), size=n_rows))
)
cat_effect = pl.col("cat1").replace_strict(cat_levels, return_dtype=pl.Float64)

c_x1 = pl.col("x1")
c_x2 = pl.col("x2")

df = df.with_columns(
    [
        (1.0 + 2.0 * c_x1 - 1.5 * c_x2 + pl.col("epsilon_rf")).alias("var_rf"),
        (1.0 + 2.0 * c_x1 - 1.5 * c_x2 + cat_effect + pl.col("epsilon_xgb")).alias("var_xgb"),
        (1.0 + 2.0 * c_x1 - 1.5 * c_x2 + cat_effect + pl.col("epsilon_cb")).alias("var_cb"),
        (1.0 + 2.0 * c_x1 - 1.5 * c_x2 + pl.col("epsilon_sk")).alias("var_sk"),
    ]
).drop("epsilon_rf", "epsilon_xgb", "epsilon_cb", "epsilon_sk")

#   Set the target variables missing according to the uniform random
#       "missing_*" columns, same convention as tests/main/srmi.py.
clear_missing = []
for suffix in ["rf", "xgb", "cb", "sk"]:
    vari = f"var_{suffix}"
    missingi = f"missing_{suffix}"
    clear_missing.append(
        pl.when(pl.col(missingi) < impute_share)
        .then(pl.lit(None))
        .otherwise(pl.col(vari))
        .alias(vari)
    )
df = df.with_columns(clear_missing).drop("missing_rf", "missing_xgb", "missing_cb", "missing_sk")

#   var_multi - an unordered categorical target (5 levels), genuinely
#       dependent on x1/x2 - for Variable.ModelType.Multinomial coverage.
#       group/wgt are reused across the extra Multinomial-only checks
#       below (donate_by, weight).
n_classes = 5
true_coefs = rng.normal(scale=1.5, size=(n_classes, 2))
X_multi = df.select("x1", "x2").to_numpy()
logits = X_multi @ true_coefs.T
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
y_multi = np.array([rng.choice(n_classes, p=probs[i]) for i in range(n_rows)])
missing_multi_mask = rng.random(n_rows) < impute_share
y_multi_with_missing = [
    None if missing_multi_mask[i] else int(y_multi[i]) for i in range(n_rows)
]

df = df.with_columns(
    [
        pl.Series("var_multi", y_multi_with_missing, dtype=pl.Int64),
        pl.Series("group", rng.choice(["north", "south"], size=n_rows)),
        pl.Series("wgt", rng.uniform(0.5, 1.5, size=n_rows)),
    ]
)

summary(df)

vars_impute = []

#   RandomForest - list-model form, cv_folds on. No categorical_feature -
#       RandomForest doesn't support it (no native categorical handling).
v_rf = Variable(
    impute_var="var_rf",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(cv_folds=3),
)
vars_impute.append(v_rf)

#   XGBoost - list-model form, cv_folds + categorical_feature together.
v_xgb = Variable(
    impute_var="var_xgb",
    model=["x1", "x2", "cat1"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(cv_folds=3, categorical_feature=["cat1"]),
)
vars_impute.append(v_xgb)

#   CatBoost - formula-model form (cat1 left out of the formula, added in
#       raw by categorical_feature), cv_folds + categorical_feature together.
v_cb = Variable(
    impute_var="var_cb",
    model="~1+x1+x2",
    modeltype=Variable.ModelType.CatBoost,
    parameters=Parameters.CatBoost(cv_folds=3, categorical_feature=["cat1"]),
)
vars_impute.append(v_cb)

#   SklearnModel - the bring-your-own-estimator escape hatch, cv_folds on.
from sklearn.linear_model import Ridge

v_sk = Variable(
    impute_var="var_sk",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.SklearnModel,
    parameters=Parameters.SklearnModel(factory=lambda: Ridge(alpha=0.5), cv_folds=3),
)
vars_impute.append(v_sk)

#   Multinomial - list-model form, plain (no donate_by/weight here - those
#       get their own dedicated checks below since they need scenarios
#       the shared SRMI run above doesn't set up).
v_multi = Variable(
    impute_var="var_multi",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.Multinomial,
    parameters=Parameters.Multinomial(parameters={"n_estimators": 100, "max_depth": 8}),
)
vars_impute.append(v_multi)


srmi = SRMI(
    df=df,
    variables=vars_impute,
    replication=SRMI.Replication(n_implicates=2, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False, testing=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_test", force_start=True
    ),
)

srmi.run()

impute_vars = [vari.impute_var for vari in srmi.variables]

srmi_loaded = srmi.load(f"{path_scratch}/py_tabular_ml_test")

dfs = srmi.df_implicates
dfs_loaded = srmi_loaded.df_implicates

for i in range(len(dfs)):
    logger.info(f"{i} (imputed):")
    df_i = dfs[i].select(impute_vars).collect()
    summary(df_i)

    logger.info("Assert no missing values remain after imputation")
    for vari in impute_vars:
        assert df_i[vari].null_count() == 0, f"{vari} still has nulls after imputation"

    logger.info("Assert imputed variables are equal across run/loaded")
    assert df_i.equals(dfs_loaded[i].select(impute_vars).collect())
    logger.info("\n\n")


#   categorical_feature's validation guard: a formula that references the
#       categorical column itself should still be rejected (it would get
#       auto one-hot-encoded, leaving nothing for categorical_feature's
#       native-categorical cast to apply to).
logger.info("Assert categorical_feature + a formula referencing it is rejected")
try:
    Variable(
        impute_var="var_cb",
        model="~1+x1+x2+cat1",
        modeltype=Variable.ModelType.CatBoost,
        parameters=Parameters.CatBoost(categorical_feature=["cat1"]),
    ).validate_inputs(df=df)
    raise AssertionError(
        "expected categorical_feature + formula-referencing-it to raise"
    )
except Exception as e:
    assert "categorical_feature" in str(e)
    logger.info(f"Correctly rejected: {e}")


#   Multinomial-specific coverage - paths the shared SRMI run above
#       doesn't exercise: formula-form model=, donate_by (including a
#       group with zero donors, which leaf_cooccurrence_match leaves
#       unmatched), weight=, random_share<1, and n_iterations>1.

def run_multinomial_case(
    name, path_suffix, model=None, parameters=None, n_iterations=1, weight="",
    bootstrap_enabled=True,
):
    var = Variable(
        impute_var="var_multi",
        model=model if model is not None else ["x1", "x2"],
        weight=weight,
        modeltype=Variable.ModelType.Multinomial,
        parameters=parameters if parameters is not None else Parameters.Multinomial(
            parameters={"n_estimators": 100, "max_depth": 8}
        ),
    )
    srmi_case = SRMI(
        df=df,
        variables=[var],
        replication=SRMI.Replication(n_implicates=1, n_iterations=n_iterations),
        parallel=SRMI.Parallel(enabled=False),
        bootstrap=SRMI.Bootstrap(enabled=bootstrap_enabled),
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_tabular_ml_{path_suffix}", force_start=True
        ),
    )
    srmi_case.run()
    df_out = srmi_case.df_implicates[0].select("var_multi").collect()
    n_null = df_out["var_multi"].null_count()
    logger.info(f"{name}: n_null_after={n_null}")
    return n_null


logger.info("Multinomial: formula-form model=")
n_null = run_multinomial_case(
    "Multinomial_formula", "multi_formula", model="~1+x1+x2"
)
assert n_null == 0

logger.info("Multinomial: donate_by grouping")
n_null = run_multinomial_case(
    "Multinomial_donate_by",
    "multi_donate_by",
    parameters=Parameters.Multinomial(
        parameters={"n_estimators": 100, "max_depth": 8}, donate_by="group"
    ),
)
assert n_null == 0

#   weight= needs its own SRMI setup, not run_multinomial_case: the
#       weight actually used for sample_weight during fitting
#       (Impute.weight) is the bootstrap weight when bootstrap is on
#       (bbweight__1, overriding whatever the Variable's own weight= says)
#       - it's original_variable.weight (the Variable's own declared
#       weight, untouched by bootstrap) that _post_impute_statistics's
#       descriptive display reads instead. Disabling bootstrap AND
#       setting SRMI.Defaults(weight=...) here makes both resolve to the
#       same column, so this exercises the weight column actually being
#       read and present, rather than the two diverging.
logger.info("Multinomial: weight=")
var_weight = Variable(
    impute_var="var_multi",
    model=["x1", "x2"],
    weight="wgt",
    modeltype=Variable.ModelType.Multinomial,
    parameters=Parameters.Multinomial(parameters={"n_estimators": 100, "max_depth": 8}),
)
srmi_weight = SRMI(
    df=df,
    variables=[var_weight],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=False),
    defaults=SRMI.Defaults(weight="wgt"),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_multi_weight", force_start=True
    ),
)
srmi_weight.run()
df_out_weight = srmi_weight.df_implicates[0].select("var_multi").collect()
n_null = df_out_weight["var_multi"].null_count()
logger.info(f"Multinomial_weight: n_null_after={n_null}")
assert n_null == 0

#   Same weight=, but with bootstrap ENABLED this time - self.weight
#       (bbweight__1, used for sample_weight) and
#       self.original_variable.weight ("wgt", read only by
#       _post_impute_statistics's display) now genuinely diverge. This
#       is the scenario that used to raise ColumnNotFoundError before
#       multinomial()'s keep_vars/impute_extra_cols/stats_model_cols were
#       fixed to read self.original_variable.weight, not self.weight, for
#       the stats-display column - the disabled-bootstrap test above
#       happens to sidestep the bug since the two columns coincide there.
logger.info("Multinomial: weight= with bootstrap enabled (weight columns diverge)")
var_weight_boot = Variable(
    impute_var="var_multi",
    model=["x1", "x2"],
    weight="wgt",
    modeltype=Variable.ModelType.Multinomial,
    parameters=Parameters.Multinomial(parameters={"n_estimators": 100, "max_depth": 8}),
)
srmi_weight_boot = SRMI(
    df=df,
    variables=[var_weight_boot],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_multi_weight_boot", force_start=True
    ),
)
srmi_weight_boot.run()
df_out_weight_boot = srmi_weight_boot.df_implicates[0].select("var_multi").collect()
assert df_out_weight_boot["var_multi"].null_count() == 0
logger.info("Multinomial: weight= with bootstrap enabled passed")

logger.info("Multinomial: random_share<1")
n_null = run_multinomial_case(
    "Multinomial_random_share",
    "multi_random_share",
    parameters=Parameters.Multinomial(
        parameters={"n_estimators": 100, "max_depth": 8}, random_share=0.5
    ),
)
assert n_null == 0

logger.info("Multinomial: n_iterations>1")
n_null = run_multinomial_case(
    "Multinomial_multi_iteration", "multi_iterations", n_iterations=3
)
assert n_null == 0


#   donate_by group with NO donors at all - leaf_cooccurrence_match
#       leaves those recipients unmatched (-1); confirm that survives the
#       full Impute.multinomial() -> _merge_imputes_to_df path as a null
#       rather than crashing or silently mismatching.
logger.info("Multinomial: donate_by group with zero donors stays null, doesn't crash")
df_empty_group = df.clone()
df_empty_group = df_empty_group.with_columns(
    pl.when(pl.col("group") == "south")
    .then(pl.lit(None))
    .otherwise(pl.col("var_multi"))
    .alias("var_multi")
)
assert df_empty_group.filter(pl.col("group") == "south")["var_multi"].null_count() == (
    df_empty_group.filter(pl.col("group") == "south").height
)

var_empty_group = Variable(
    impute_var="var_multi",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.Multinomial,
    parameters=Parameters.Multinomial(
        parameters={"n_estimators": 100, "max_depth": 8}, donate_by="group"
    ),
)
srmi_empty_group = SRMI(
    df=df_empty_group,
    variables=[var_empty_group],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_empty_group", force_start=True
    ),
)
srmi_empty_group.run()
df_out_empty_group = srmi_empty_group.df_implicates[0].select("var_multi", "group").collect()
n_null_south = df_out_empty_group.filter(pl.col("group") == "south")["var_multi"].null_count()
n_null_north = df_out_empty_group.filter(pl.col("group") == "north")["var_multi"].null_count()
logger.info(f"south (no donors) still null: {n_null_south}, north (has donors) still null: {n_null_north}")
assert n_null_south > 0, "expected south group (no donors) to stay null - nothing to donate from"
assert n_null_north == 0, "north group has donors and should have imputed fully"


logger.info("tabular_ml.py: multinomial checks passed")


#   ============================================================
#   group_levels (nested shrinkage-heuristic random-intercept
#       stand-in) coverage - RandomForest/XGBoost/CatBoost/SklearnModel,
#       single- and multi-level nesting, combined with cv_folds/
#       categorical_feature/donate_by, save/load, and the household-head
#       Where-restriction scenario this feature was originally motivated
#       by.
#   ============================================================

n_states = 4
counties_per_state = 3
hh_per_county = 15
members_per_hh_choices = [1, 2, 3, 4]

rng_g = np.random.default_rng(20260909)
state_ids = np.arange(n_states)
county_ids = np.arange(n_states * counties_per_state)
county_state = np.repeat(state_ids, counties_per_state)
hhid_ids = np.arange(n_states * counties_per_state * hh_per_county)
hh_county = np.repeat(county_ids, hh_per_county)
hh_state = county_state[hh_county]
members_per_hh = rng_g.choice(members_per_hh_choices, size=len(hhid_ids))

state_g = np.repeat(hh_state, members_per_hh)
county_g = np.repeat(hh_county, members_per_hh)
hhid_g = np.repeat(hhid_ids, members_per_hh)
n_g = len(hhid_g)

x1_g = rng_g.normal(size=n_g)
#   Real nested effects at all three levels, decreasing in scale as the
#       grouping gets finer (matching the state/county/hhid ordering).
state_effect = rng_g.normal(scale=10.0, size=n_states)[state_g]
county_effect = rng_g.normal(scale=5.0, size=len(county_ids))[county_g]
hh_effect = rng_g.normal(scale=3.0, size=len(hhid_ids))[hhid_g]
y_g = 2.0 * x1_g + state_effect + county_effect + hh_effect + rng_g.normal(scale=1.0, size=n_g)

#   is_head - exactly one row per household flagged as head, for the
#       Where-restricted scenario below (mirrors imputing a household-
#       level variable only for the head, then broadcasting).
is_head = np.zeros(n_g, dtype=bool)
_, first_idx = np.unique(hhid_g, return_index=True)
is_head[first_idx] = True

miss_share_g = 0.25
miss_mask_g = rng_g.random(n_g) < miss_share_g
y_g_missing = [None if miss_mask_g[i] else float(y_g[i]) for i in range(n_g)]

df_g = pl.DataFrame(
    dict(
        idx=np.arange(n_g),
        state=state_g.astype(str),
        county=county_g.astype(str),
        hhid=hhid_g.astype(str),
        x1=x1_g,
        y=y_g_missing,
        is_head=is_head,
        #   Kept as its own (never-imputed) column, not a positional mask,
        #       since row order after SRMI processing isn't guaranteed to
        #       match generation order - filtering by this column instead
        #       of a bare boolean array is the robust way to pull out just
        #       the previously-missing rows afterward.
        was_missing_g=miss_mask_g,
        wgt_g=rng_g.uniform(0.5, 1.5, size=n_g),
    )
)
df_g_true = pl.DataFrame(dict(idx=np.arange(n_g), y_true=y_g))


def run_group_case(
    name, path_suffix, modeltype, parameters, df_use=None, n_iterations=3,
    check_no_nulls=True, model=None,
):
    df_use = df_use if df_use is not None else df_g
    var = Variable(
        impute_var="y",
        model=model if model is not None else ["x1"],
        modeltype=modeltype,
        parameters=parameters,
    )
    srmi_g = SRMI(
        df=df_use,
        variables=[var],
        replication=SRMI.Replication(n_implicates=1, n_iterations=n_iterations),
        parallel=SRMI.Parallel(enabled=False),
        bootstrap=SRMI.Bootstrap(enabled=True),
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_tabular_ml_group_{path_suffix}",
            force_start=True,
        ),
    )
    srmi_g.run()
    df_out_g = srmi_g.implicates[0].df

    df_out_g = nw.from_native(df_out_g).lazy().collect().to_native()
    n_null = df_out_g["y"].is_null().sum()
    logger.info(f"{name}: n_null_after={n_null}")
    if check_no_nulls:
        assert n_null == 0, f"{name} left nulls"
    return df_out_g


logger.info("group_levels: single level (hhid), RandomForest")
df_out1 = run_group_case(
    "group_single_rf",
    "single_rf",
    Variable.ModelType.RandomForest,
    Parameters.RandomForest(
        parameters={"n_estimators": 100, "max_depth": 6}, group_levels=["hhid"]
    ),
)
assert "___group_intercept_y___" in df_out1.columns

logger.info("group_levels: multi-level nesting (state -> county -> hhid), RandomForest")
df_out2 = run_group_case(
    "group_multi_rf",
    "multi_rf",
    Variable.ModelType.RandomForest,
    Parameters.RandomForest(
        parameters={"n_estimators": 100, "max_depth": 6},
        group_levels=["state", "county", "hhid"],
    ),
)
df_check2 = df_out2.join(df_g_true, on="idx").filter(pl.col("was_missing_g"))
rmse_multi = float(np.sqrt(((df_check2["y"] - df_check2["y_true"]) ** 2).mean()))

logger.info("group_levels: baseline (no group_levels) for RMSE comparison")
df_out_base = run_group_case(
    "group_baseline_rf",
    "baseline_rf",
    Variable.ModelType.RandomForest,
    Parameters.RandomForest(parameters={"n_estimators": 100, "max_depth": 6}),
)
df_check_base = df_out_base.join(df_g_true, on="idx").filter(pl.col("was_missing_g"))
rmse_base = float(np.sqrt(((df_check_base["y"] - df_check_base["y_true"]) ** 2).mean()))

logger.info(f"RMSE without group_levels = {rmse_base:.3f}, "
            f"with 3-level group_levels = {rmse_multi:.3f}")
assert rmse_multi < rmse_base, (
    "multi-level group_levels should meaningfully reduce RMSE given the real "
    "nested state/county/hhid effects baked into this synthetic data"
)

logger.info("group_levels: XGBoost, combined with cv_folds")
run_group_case(
    "group_xgb_cv",
    "xgb_cv",
    Variable.ModelType.XGBoost,
    Parameters.XGBoost(
        parameters={"n_estimators": 100, "max_depth": 4},
        group_levels=["state", "hhid"],
        cv_folds=3,
    ),
)

logger.info("group_levels: CatBoost, combined with categorical_feature")
df_g_cat = df_g.with_columns(pl.Series("cat_g", rng_g.choice(["a", "b"], size=n_g)))
run_group_case(
    "group_cb_categorical",
    "cb_categorical",
    Variable.ModelType.CatBoost,
    Parameters.CatBoost(
        parameters={"iterations": 100, "depth": 4},
        group_levels=["hhid"],
        categorical_feature=["cat_g"],
    ),
    df_use=df_g_cat,
    model=["x1", "cat_g"],
)

logger.info("group_levels: SklearnModel")
from sklearn.linear_model import Ridge

run_group_case(
    "group_sklearn",
    "sklearn",
    Variable.ModelType.SklearnModel,
    Parameters.SklearnModel(factory=lambda: Ridge(alpha=0.5), group_levels=["state", "hhid"]),
)

logger.info("group_levels: combined with donate_by (a different grouping concern - "
            "donate_by restricts PMM donor matching, group_levels shrinks the "
            "prediction - same column, two different jobs, shouldn't collide)")
run_group_case(
    "group_donate_by",
    "donate_by",
    Variable.ModelType.RandomForest,
    Parameters.RandomForest(
        parameters={"n_estimators": 100, "max_depth": 6},
        group_levels=["state", "hhid"],
        parameters_pmm=Parameters.pmm(donate_by="state"),
    ),
)

logger.info("group_levels: save/load round-trip")
df_run = run_group_case(
    "group_saveload",
    "saveload",
    Variable.ModelType.RandomForest,
    Parameters.RandomForest(
        parameters={"n_estimators": 100, "max_depth": 6}, group_levels=["hhid"]
    ),
    n_iterations=2,
)
srmi_loaded_g = SRMI.load(f"{path_scratch}/py_tabular_ml_group_saveload")

df_loaded_g = nw.from_native(srmi_loaded_g.implicates[0].df).lazy().collect().to_native()
assert (
    df_run.sort("idx").select("y", "___group_intercept_y___")
    .equals(df_loaded_g.sort("idx").select("y", "___group_intercept_y___"))
), "group intercept column should round-trip exactly through save/load"

logger.info(
    "group_levels: Where-restricted to household heads only (the original "
    "motivating scenario) - impute a household-level variable for heads "
    "only, group_levels shrinks toward the hh's own history, broadcast "
    "to members separately"
)
var_head = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        parameters={"n_estimators": 100, "max_depth": 6}, group_levels=["state", "county"]
    ),
    sample=Variable.Sample(Where=nw.col("is_head")),
)
srmi_head = SRMI(
    df=df_g,
    variables=[var_head],
    replication=SRMI.Replication(n_implicates=1, n_iterations=2),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_group_head", force_start=True
    ),
)
srmi_head.run()
df_head_out = nw.from_native(srmi_head.implicates[0].df).lazy().collect().to_native()
#   Only head rows get imputed/modeled - non-head rows with originally
#       missing y stay null (by design - Where restricts who this
#       variable applies to at all), and the group intercept column only
#       has values for head rows too.
n_null_heads = df_head_out.filter(pl.col("is_head"))["y"].is_null().sum()
assert n_null_heads == 0, "all household heads should have y imputed"
n_intercept_present_heads = (
    df_head_out.filter(pl.col("is_head"))["___group_intercept_y___"].is_not_null().sum()
)
assert n_intercept_present_heads == df_head_out.filter(pl.col("is_head")).height, (
    "every head row should have a group intercept estimate"
)
logger.info("group_levels: Where-restricted-to-heads scenario passed")


logger.info(
    "group_levels: weight= combined with group_levels - weight actually "
    "used for the shrinkage (self.weight) is the bootstrap weight when "
    "bootstrap is on, overriding a declared weight= regardless, so this "
    "needs bootstrap disabled + SRMI.Defaults(weight=...) to make "
    "self.weight resolve to the declared column at all (same distinction "
    "found earlier for Multinomial's weight test)."
)
var_wgt = Variable(
    impute_var="y",
    model=["x1"],
    weight="wgt_g",
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        parameters={"n_estimators": 100, "max_depth": 6}, group_levels=["state", "hhid"]
    ),
)
srmi_wgt = SRMI(
    df=df_g,
    variables=[var_wgt],
    replication=SRMI.Replication(n_implicates=1, n_iterations=2),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=False),
    defaults=SRMI.Defaults(weight="wgt_g"),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_group_weight", force_start=True
    ),
)
srmi_wgt.run()
df_wgt_out = nw.from_native(srmi_wgt.implicates[0].df).lazy().collect().to_native()
assert df_wgt_out["y"].is_null().sum() == 0
assert "___group_intercept_y___" in df_wgt_out.columns
logger.info("group_levels: weight= combined with group_levels passed")


logger.info("tabular_ml.py: group_levels checks passed")


#   ============================================================
#   error=ErrorDraw.leaf - generic leaf co-occurrence donor matching for
#       RandomForest()/XGBoost()/CatBoost()/SklearnModel() mean-regression
#       targets, not just Multinomial's unordered categorical case - same
#       donor-selection mechanism (see utilities/leaf_donor_matching.py),
#       now reused via the shared _leaf_match_donor_positions/
#       _leaf_gather_donations helpers.
#   ============================================================

logger.info("error=ErrorDraw.leaf: RandomForest, no donate_by")
v_rf_leaf = Variable(
    impute_var="var_rf",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        parameters={"n_estimators": 100, "max_depth": 8},
        error=Parameters.ErrorDraw.leaf,
    ),
)
srmi_rf_leaf = SRMI(
    df=df,
    variables=[v_rf_leaf],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_leaf_rf", force_start=True
    ),
)
srmi_rf_leaf.run()
df_rf_leaf_out = srmi_rf_leaf.df_implicates[0].select("var_rf").collect()
assert df_rf_leaf_out["var_rf"].null_count() == 0

#   Donor-matching invariant: every imputed value must be an ACTUALLY
#       OBSERVED donor value (never invented, unlike error=Random) - the
#       defining property leaf-matching shares with pmm.
observed_var_rf = set(df.drop_nulls("var_rf")["var_rf"].to_list())
imputed_var_rf = set(df_rf_leaf_out["var_rf"].to_list())
assert imputed_var_rf.issubset(observed_var_rf), (
    "leaf-matched donations should only ever be real observed values"
)
logger.info("error=ErrorDraw.leaf: RandomForest passed")


logger.info("error=ErrorDraw.leaf: XGBoost with categorical_feature + donate_by")
v_xgb_leaf = Variable(
    impute_var="var_xgb",
    model=["x1", "x2", "cat1"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(
        parameters={"n_estimators": 100},
        categorical_feature=["cat1"],
        error=Parameters.ErrorDraw.leaf,
        parameters_pmm=Parameters.pmm(donate_by="group"),
    ),
)
srmi_xgb_leaf = SRMI(
    df=df,
    variables=[v_xgb_leaf],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_leaf_xgb", force_start=True
    ),
)
srmi_xgb_leaf.run()
df_xgb_leaf_out = srmi_xgb_leaf.df_implicates[0].select("var_xgb").collect()
assert df_xgb_leaf_out["var_xgb"].null_count() == 0
logger.info("error=ErrorDraw.leaf: XGBoost + donate_by passed")


logger.info("error=ErrorDraw.leaf: CatBoost, formula model + categorical_feature")
v_cb_leaf = Variable(
    impute_var="var_cb",
    model="~1+x1+x2",
    modeltype=Variable.ModelType.CatBoost,
    parameters=Parameters.CatBoost(
        parameters={"iterations": 100, "depth": 4},
        categorical_feature=["cat1"],
        error=Parameters.ErrorDraw.leaf,
    ),
)
srmi_cb_leaf = SRMI(
    df=df,
    variables=[v_cb_leaf],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_leaf_cb", force_start=True
    ),
)
srmi_cb_leaf.run()
df_cb_leaf_out = srmi_cb_leaf.df_implicates[0].select("var_cb").collect()
assert df_cb_leaf_out["var_cb"].null_count() == 0
logger.info("error=ErrorDraw.leaf: CatBoost passed")


#   donate_by group with NO donors at all - same zero-donor convention as
#       Multinomial's: those recipients stay null rather than crashing.
logger.info("error=ErrorDraw.leaf: donate_by group with zero donors stays null")
df_leaf_empty_group = df.clone().with_columns(
    pl.when(pl.col("group") == "south")
    .then(pl.lit(None))
    .otherwise(pl.col("var_rf"))
    .alias("var_rf")
)
v_rf_leaf_empty = Variable(
    impute_var="var_rf",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        parameters={"n_estimators": 100, "max_depth": 8},
        error=Parameters.ErrorDraw.leaf,
        parameters_pmm=Parameters.pmm(donate_by="group"),
    ),
)
srmi_leaf_empty = SRMI(
    df=df_leaf_empty_group,
    variables=[v_rf_leaf_empty],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_leaf_empty_group", force_start=True
    ),
)
srmi_leaf_empty.run()
df_leaf_empty_out = srmi_leaf_empty.df_implicates[0].select("var_rf", "group").collect()
n_null_south_leaf = df_leaf_empty_out.filter(pl.col("group") == "south")["var_rf"].null_count()
n_null_north_leaf = df_leaf_empty_out.filter(pl.col("group") == "north")["var_rf"].null_count()
assert n_null_south_leaf > 0, "expected south group (no donors) to stay null"
assert n_null_north_leaf == 0, "north group has donors and should have imputed fully"
logger.info("error=ErrorDraw.leaf: zero-donor donate_by group passed")


#   group_levels + error=leaf composed together - orthogonal features
#       (group_levels affects the fit target/prediction center; leaf
#       matching only cares about the fitted tree structure) - reuse
#       df_g's real 3-level nested data from the group_levels section
#       above.
logger.info("error=ErrorDraw.leaf: combined with group_levels")
v_leaf_group = Variable(
    impute_var="y",
    model=["x1"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        parameters={"n_estimators": 100, "max_depth": 6},
        error=Parameters.ErrorDraw.leaf,
        group_levels=["state", "county", "hhid"],
    ),
)
srmi_leaf_group = SRMI(
    df=df_g,
    variables=[v_leaf_group],
    replication=SRMI.Replication(n_implicates=1, n_iterations=3),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_leaf_group", force_start=True
    ),
)
srmi_leaf_group.run()
df_leaf_group_out = (
    nw.from_native(srmi_leaf_group.implicates[0].df).lazy().collect().to_native()
)
assert df_leaf_group_out["y"].is_null().sum() == 0
assert "___group_intercept_y___" in df_leaf_group_out.columns

#   Donor-matching invariant again, restricted to the previously-missing
#       rows (was_missing_g - the robust, order-independent filter used
#       throughout this section) - group_levels is orthogonal, donation
#       still only ever returns a real observed y.
observed_y_g = set(df_g.drop_nulls("y")["y"].to_list())
df_leaf_group_check = df_leaf_group_out.join(
    df_g.select("idx", "was_missing_g"), on="idx"
).filter(pl.col("was_missing_g"))
imputed_y_g = set(df_leaf_group_check["y"].to_list())
assert imputed_y_g.issubset(observed_y_g)
logger.info("error=ErrorDraw.leaf: group_levels composition passed")


#   Clear failure for an estimator with no leaf-index support at all (a
#       plain linear model via SklearnModel()) - error=leaf should raise
#       an actionable error rather than silently doing nothing.
logger.info("error=ErrorDraw.leaf: unsupported estimator raises a clear error")
v_sk_leaf = Variable(
    impute_var="var_sk",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.SklearnModel,
    parameters=Parameters.SklearnModel(
        factory=lambda: Ridge(alpha=0.5), error=Parameters.ErrorDraw.leaf
    ),
)
srmi_sk_leaf = SRMI(
    df=df,
    variables=[v_sk_leaf],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_leaf_sk_unsupported",
        force_start=True,
    ),
)
try:
    srmi_sk_leaf.run()
    raise AssertionError(
        "expected error=ErrorDraw.leaf on an estimator without leaf "
        "indices (Ridge) to raise"
    )
except AssertionError:
    raise
except Exception as e:
    assert "apply" in str(e) or "calc_leaf_indexes" in str(e), (
        f"expected a leaf-index-support error, got: {e}"
    )
    logger.info(f"Correctly rejected: {e}")


logger.info("tabular_ml.py: error=ErrorDraw.leaf checks passed")


#   ============================================================
#   Variable.ModelType.OrderedCategorical - like Multinomial(), donation-
#       based (never invents a category that wasn't observed), but for an
#       ORDERED categorical target: fits a mean-regression estimator
#       against an integer rank encoding of the declared category order,
#       then donates the real category from a matched donor (pmm on the
#       predicted rank, or leaf co-occurrence).
#   ============================================================

oc_categories = ["low", "medium", "high", "very_high"]
oc_rank_true = np.clip(
    np.round(1.5 * df["x1"].to_numpy() + rng.normal(scale=1.0, size=n_rows) + 1.5),
    0,
    3,
).astype(int)
oc_y_true = [oc_categories[r] for r in oc_rank_true]
oc_missing_mask = rng.random(n_rows) < impute_share
df = df.with_columns(
    pl.Series(
        "var_ordered",
        [None if oc_missing_mask[i] else oc_y_true[i] for i in range(n_rows)],
    )
)

logger.info("OrderedCategorical: pmm (default), RandomForestRegressor default")
v_oc_pmm = Variable(
    impute_var="var_ordered",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.OrderedCategorical,
    parameters=Parameters.OrderedCategorical(
        categories=oc_categories,
        parameters={"n_estimators": 100, "max_depth": 6},
    ),
)
srmi_oc_pmm = SRMI(
    df=df,
    variables=[v_oc_pmm],
    replication=SRMI.Replication(n_implicates=1, n_iterations=2),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_oc_pmm", force_start=True
    ),
)
srmi_oc_pmm.run()
df_oc_pmm_out = srmi_oc_pmm.df_implicates[0].select("var_ordered").collect()
assert df_oc_pmm_out["var_ordered"].null_count() == 0
assert set(df_oc_pmm_out["var_ordered"].to_list()).issubset(set(oc_categories)), (
    "OrderedCategorical should only ever donate a real, declared category"
)
logger.info("OrderedCategorical: pmm passed")


logger.info("OrderedCategorical: error=leaf, donate_by")
v_oc_leaf = Variable(
    impute_var="var_ordered",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.OrderedCategorical,
    parameters=Parameters.OrderedCategorical(
        categories=oc_categories,
        parameters={"n_estimators": 100, "max_depth": 6},
        error=Parameters.ErrorDraw.leaf,
        donate_by="group",
    ),
)
srmi_oc_leaf = SRMI(
    df=df,
    variables=[v_oc_leaf],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_oc_leaf", force_start=True
    ),
)
srmi_oc_leaf.run()
df_oc_leaf_out = srmi_oc_leaf.df_implicates[0].select("var_ordered").collect()
assert df_oc_leaf_out["var_ordered"].null_count() == 0
assert set(df_oc_leaf_out["var_ordered"].to_list()).issubset(set(oc_categories))
logger.info("OrderedCategorical: error=leaf passed")


logger.info("OrderedCategorical: weight= with bootstrap enabled (weight columns diverge)")
v_oc_wgt = Variable(
    impute_var="var_ordered",
    model=["x1", "x2"],
    weight="wgt",
    modeltype=Variable.ModelType.OrderedCategorical,
    parameters=Parameters.OrderedCategorical(
        categories=oc_categories, parameters={"n_estimators": 100, "max_depth": 6}
    ),
)
srmi_oc_wgt = SRMI(
    df=df,
    variables=[v_oc_wgt],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_oc_weight", force_start=True
    ),
)
srmi_oc_wgt.run()
df_oc_wgt_out = srmi_oc_wgt.df_implicates[0].select("var_ordered").collect()
assert df_oc_wgt_out["var_ordered"].null_count() == 0
logger.info("OrderedCategorical: weight= with bootstrap enabled passed")


logger.info("OrderedCategorical: an unobserved/unknown category in categories= raises")
df_oc_bad = df.with_columns(
    pl.when(pl.col("index") == 0)
    .then(pl.lit("not_a_declared_category"))
    .otherwise(pl.col("var_ordered"))
    .alias("var_ordered")
)
v_oc_bad = Variable(
    impute_var="var_ordered",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.OrderedCategorical,
    parameters=Parameters.OrderedCategorical(categories=oc_categories),
)
srmi_oc_bad = SRMI(
    df=df_oc_bad,
    variables=[v_oc_bad],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{path_scratch}/py_tabular_ml_oc_bad", force_start=True
    ),
)
try:
    srmi_oc_bad.run()
    raise AssertionError("expected an unknown category to raise")
except AssertionError:
    raise
except Exception as e:
    assert "categories" in str(e), f"expected a categories error, got: {e}"
    logger.info(f"Correctly rejected: {e}")

logger.info("tabular_ml.py: OrderedCategorical checks passed")

logger.info("tabular_ml.py: all checks passed")
