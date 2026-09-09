import narwhals as nw
import polars as pl
import polars.selectors as cs
import numpy as np

from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import summary

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.srmi import SRMI
from survey_kit.imputation.utilities.tune_estimator import tune_estimator

from survey_kit import logger, config


# %%
logger.info("Draw some random data")
logger.info(
    "   This tutorial covers the tabular-ML modeltypes: RandomForest, "
    "XGBoost, CatBoost, SklearnModel (bring your own estimator), and "
    "Multinomial. The first four are all 'mean regression with a "
    "different model plugged in' - see tutorials/srmi/regression.py for "
    "the plain OLS/Logit case and tutorials/srmi/gbm.py for LightGBM, "
    "which has its own dedicated quantile-regression machinery these "
    "don't. Multinomial is a genuinely different shape of imputation - "
    "classification for an unordered categorical variable, imputed by "
    "donor matching rather than a predicted value - see its own section "
    "below."
)

n_rows = 10_000
impute_share = 0.25

df = (
    RandomData(n_rows=n_rows, seed=8675309)
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
    .float("missing_multi", 0, 1)
    .to_df()
)

logger.info(
    "cat1 is a native categorical predictor - a, b, c, each with a "
    "different effect - shared by the XGBoost and CatBoost variables "
    "below to demonstrate categorical_feature."
)
rng = np.random.default_rng(8675309)
cat_levels = {"a": 0.0, "b": 3.0, "c": -2.0}
df = df.with_columns(pl.Series("cat1", rng.choice(list(cat_levels.keys()), size=n_rows)))
cat_effect = pl.col("cat1").replace_strict(cat_levels, return_dtype=pl.Float64)

c_x1 = pl.col("x1")
c_x2 = pl.col("x2")

df = (
    df.with_columns(
        [
            (1.0 + 2.0 * c_x1 - 1.5 * c_x2 + pl.col("epsilon_rf")).alias("var_rf"),
            (1.0 + 2.0 * c_x1 - 1.5 * c_x2 + cat_effect + pl.col("epsilon_xgb")).alias("var_xgb"),
            (1.0 + 2.0 * c_x1 - 1.5 * c_x2 + cat_effect + pl.col("epsilon_cb")).alias("var_cb"),
            (1.0 + 2.0 * c_x1 - 1.5 * c_x2 + pl.col("epsilon_sk")).alias("var_sk"),
        ]
    )
    .drop(["epsilon_rf", "epsilon_xgb", "epsilon_cb", "epsilon_sk"])
    .with_row_index(name="_row_index_")
)

logger.info(
    "var_multi is an unordered categorical outcome (5 levels), genuinely "
    "dependent on x1/x2 - like an occupation or industry code, just with "
    "far fewer categories here to keep the tutorial fast. region is a "
    "grouping variable used by Multinomial's donate_by below."
)
n_classes = 5
true_coefs_multi = rng.normal(scale=1.5, size=(n_classes, 2))
logits_multi = df.select("x1", "x2").to_numpy() @ true_coefs_multi.T
probs_multi = np.exp(logits_multi) / np.exp(logits_multi).sum(axis=1, keepdims=True)
y_multi = np.array([rng.choice(n_classes, p=probs_multi[i]) for i in range(n_rows)])
df = df.with_columns(
    [
        pl.Series("var_multi", y_multi, dtype=pl.Int64),
        pl.Series("region", rng.choice(["north", "south"], size=n_rows)),
    ]
)

df_original = df

#   Set variables to missing according to the uniform random variables missing_*
clear_missing = [
    #   Kept as its own (never-imputed) column so it survives SRMI's run
    #       unchanged - used below to pull out just the previously-missing
    #       var_multi rows for the chi-squared check.
    (pl.col("missing_multi") < impute_share).alias("was_missing_multi")
]
for suffix in ["rf", "xgb", "cb", "sk", "multi"]:
    vari = f"var_{suffix}"
    missingi = f"missing_{suffix}"
    clear_missing.append(
        pl.when(pl.col(missingi) < impute_share)
        .then(pl.lit(None))
        .otherwise(pl.col(vari))
        .alias(vari)
    )
df = df.with_columns(clear_missing).drop(cs.starts_with("missing_"))

summary(df)


vars_impute = []

# %%
logger.info("RandomForest - the simplest of the four")
logger.info(
    "   RandomForest has no native categorical support (unlike XGBoost/"
    "CatBoost below), so categorical predictors would need to be "
    "one-hot-encoded via a formula's C(...) first if you had any."
)
logger.info(
    "   cv_folds=5 turns on cross-validated donor-pool predictions: "
    "instead of matching donors on their in-sample (overfit) prediction, "
    "each donor's matching value comes from a model that was refit "
    "without that donor's own y - see Parameters._tabular_ml_params's "
    "cv_folds docstring for the full rationale. It's off (0) by default "
    "since it costs cv_folds+1 model fits instead of 1."
)
v_rf = Variable(
    impute_var="var_rf",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        parameters={"n_estimators": 100, "max_depth": 8},
        cv_folds=5,
    ),
)
vars_impute.append(v_rf)


# %%
logger.info("XGBoost - with categorical_feature and cv_folds together")
logger.info(
    "   categorical_feature declares which columns XGBoost should treat "
    "as native categoricals (its own histogram-based categorical splits) "
    "rather than one-hot encoding. It only works with model= as a plain "
    "column list (like here) or a formula that simply leaves the "
    "categorical column out entirely - never one that references it, "
    "since a bare reference there still gets auto one-hot-encoded "
    "regardless of categorical_feature."
)
v_xgb = Variable(
    impute_var="var_xgb",
    model=["x1", "x2", "cat1"],
    modeltype=Variable.ModelType.XGBoost,
    parameters=Parameters.XGBoost(
        parameters={"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        categorical_feature=["cat1"],
        cv_folds=5,
    ),
)
vars_impute.append(v_xgb)


# %%
logger.info("CatBoost - categorical_feature via the formula-model form this time")
logger.info(
    "   model= is a formula here ('~1+x1+x2') that simply never mentions "
    "cat1 - _run_regression adds categorical_feature's columns into the "
    "model matrix raw, alongside whatever the formula produces, so this "
    "works the same way the list form does above."
)
v_cb = Variable(
    impute_var="var_cb",
    model="~1+x1+x2",
    modeltype=Variable.ModelType.CatBoost,
    parameters=Parameters.CatBoost(
        parameters={"iterations": 200, "depth": 5},
        categorical_feature=["cat1"],
        cv_folds=5,
    ),
)
vars_impute.append(v_cb)


# %%
logger.info("SklearnModel - the escape hatch for any sklearn-compatible estimator")
logger.info(
    "   factory is a zero-arg callable returning a fresh, unfitted "
    "estimator - anything with .fit(X, y)/.predict(X) works, not just "
    "something with a dedicated Parameters.XXX() function."
)
logger.info(
    "   tune_estimator() runs an Optuna hyperparameter search (its own, "
    "separate cross-validation loop - not the same cv_folds as above, "
    "which is about donor-pool predictions, not hyperparameter search) "
    "and returns the best trial's hyperparameters as a plain dict, ready "
    "to splice into the factory."
)
from sklearn.linear_model import Ridge

df_observed = df.filter(pl.col("var_sk").is_not_null())
best_params_sk = tune_estimator(
    df=df_observed,
    y="var_sk",
    x=["x1", "x2"],
    model_factory=Ridge,
    param_space={"alpha": (0.01, 10.0, "log")},
    n_trials=20,
    cv_folds=3,
)
logger.info(f"   tune_estimator picked: {best_params_sk}")

v_sk = Variable(
    impute_var="var_sk",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.SklearnModel,
    parameters=Parameters.SklearnModel(
        factory=lambda: Ridge(**best_params_sk),
        cv_folds=5,
    ),
)
vars_impute.append(v_sk)


# %%
logger.info("Multinomial - unordered categorical imputation via donor matching")
logger.info(
    "   Fits a RandomForestClassifier, then imputes by donor matching on "
    "leaf co-occurrence: for each recipient, pool the donors that share "
    "a leaf with it across every tree, and draw one uniformly at random "
    "- the same donor-selection mechanism mice's rf method uses. This is "
    "genuinely different from the four models above: there's no scalar "
    "prediction being PMM-matched, the donor match itself comes directly "
    "from which trees group two rows together - see "
    "imputation/utilities/leaf_donor_matching.py for the mechanism, and "
    "Parameters.Multinomial()'s docstring for why there's no error=/"
    "cv_folds= here (both assume a scalar yhat, which doesn't exist for "
    "this method)."
)
logger.info(
    "   model= works as either a plain column list (as here) or an "
    "R-style formula string - RandomForestClassifier itself has no "
    "native categorical handling (same restriction as RandomForest() "
    "above), but a formula's own C(...) one-hot encoding still applies "
    "normally either way."
)
logger.info(
    "   donate_by restricts donor matching to within the recipient's own "
    "group - here, region. A group with zero donors leaves those "
    "recipients unmatched (null) rather than borrowing from elsewhere."
)
v_multi = Variable(
    impute_var="var_multi",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.Multinomial,
    parameters=Parameters.Multinomial(
        parameters={"n_estimators": 200, "max_depth": 8},
        donate_by="region",
    ),
)
vars_impute.append(v_multi)


# %%
logger.info("Set up the imputation")
srmi = SRMI(
    df=df,
    variables=vars_impute,
    index=["index"],
    replication=SRMI.Replication(n_implicates=2, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/py_srmi_test_tabular_ml",
        force_start=True,
    ),
)

# %%
logger.info("Run it")
srmi.run()

logger.info("It's automatically saved and can be loaded with (see path_model above):")
logger.info("path_model = f'{config.path_temp_files}/py_srmi_test_tabular_ml'")
logger.info("srmi = SRMI.load(path_model)")


# %%
logger.info("Get the results")
df_list = srmi.df_implicates

logger.info("\n\nLook at the original")
_ = summary(df_original, detailed=True, drb_round=True)

logger.info("\n\nLook at the imputes")
_ = df_list.pipe(summary, detailed=True, drb_round=True)


# %%
logger.info(
    "var_multi is an unordered category, not a continuous/ordinal "
    "variable - a mean of its codes (0-4) isn't a meaningful statistic. "
    "A crosstab of true vs. imputed category, plus a chi-squared test of "
    "independence on it, is the right diagnostic instead: for a working "
    "imputation, imputed category should be strongly ASSOCIATED with the "
    "true one (most mass on the diagonal), so a low p-value here is the "
    "good outcome - it means that association is real, not noise."
)
from scipy.stats import chi2_contingency

categories = list(range(n_classes))

for i, dfi in enumerate(df_list):
    dfi = nw.from_native(dfi).lazy().collect().to_native()
    df_check = dfi.join(
        df_original.select("index", pl.col("var_multi").alias("var_multi_true")),
        on="index",
    ).filter(pl.col("was_missing_multi"))

    #   A fixed n_classes x n_classes matrix via 2D bincount, not
    #       group_by/pivot - a category that's missing entirely from one
    #       side (e.g. never imputed) would otherwise silently drop a row
    #       or column instead of showing up as all zeros, and pivot's
    #       column order isn't guaranteed to match categories' order
    #       either.
    true_vals = df_check["var_multi_true"].to_numpy()
    imputed_vals = df_check["var_multi"].to_numpy()
    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(contingency, (true_vals, imputed_vals), 1)

    logger.info(f"\n\nImplicate {i}: crosstab of true (rows) vs. imputed (cols) category")
    logger.info(
        pl.DataFrame(contingency, schema=[str(c) for c in categories]).with_columns(
            pl.Series("var_multi_true", categories)
        ).select(["var_multi_true"] + [str(c) for c in categories])
    )

    stat, p_value, dof, expected = chi2_contingency(contingency)

    #   Cramer's V - the raw chi-squared statistic (and its p-value) grows
    #       with sample size and only answers "is there SOME association",
    #       not "how strong is it" - at n=2472 it'll reject independence
    #       for almost any non-trivial effect, useful or not. Cramer's V
    #       normalizes chi-squared by n and table size into a bounded
    #       [0, 1] effect size instead - 0 means no association
    #       (imputation no better than guessing the marginal distribution),
    #       1 means a perfect one-to-one match between true and imputed
    #       category. That's the actual "how much deviation" summary a
    #       mean of category codes was never going to give us.
    n_obs = contingency.sum()
    cramers_v = np.sqrt(stat / (n_obs * (n_classes - 1)))

    logger.info(
        f"Implicate {i}: chi-squared test of independence (true category "
        f"vs. imputed category) = {stat:.2f}, dof = {dof}, p = {p_value:.3g} "
        f"- Cramer's V (effect size, 0=no association, 1=perfect match) "
        f"= {cramers_v:.3f}"
    )


# %%
logger.info(
    "group_levels - a cheap shrinkage-heuristic stand-in for a "
    "random-intercept term, for nested clustered data (state/county/hhid- "
    "style). Not a real mixed model - no joint variance-component "
    "estimation, just an empirical-Bayes-style shrunk group mean, applied "
    "one nested level at a time. Available on Regression()/RandomForest()/ "
    "XGBoost()/CatBoost()/SklearnModel() (anything routing through "
    "_run_regression) - not on LightGBM() or the donor-matching methods "
    "(HotDeck/StatMatch/NearestNeighbor/Multinomial), which don't have a "
    "residual to decompose the same way."
)
logger.info(
    "   It re-estimates fresh from each SRMI iteration's fit residuals and "
    "persists the result into the working data as a plain column - "
    "'___group_intercept_<var>___' - the same way donate_list values ride "
    "along, so there's no separate inner convergence loop: it improves "
    "alongside everything else SRMI already refits iteration to iteration."
)

n_states = 4
counties_per_state = 3
hh_per_county = 15
members_per_hh_choices = [1, 2, 3, 4]

rng_hier = np.random.default_rng(20260909)
state_ids = np.arange(n_states)
county_ids = np.arange(n_states * counties_per_state)
county_state = np.repeat(state_ids, counties_per_state)
hhid_ids = np.arange(n_states * counties_per_state * hh_per_county)
hh_county = np.repeat(county_ids, hh_per_county)
hh_state = county_state[hh_county]
members_per_hh = rng_hier.choice(members_per_hh_choices, size=len(hhid_ids))

state_h = np.repeat(hh_state, members_per_hh)
county_h = np.repeat(hh_county, members_per_hh)
hhid_h = np.repeat(hhid_ids, members_per_hh)
n_h = len(hhid_h)

logger.info(
    "   var_hier depends on x1_h plus REAL nested effects at all three "
    "levels (state std=10, county std=5, hhid std=3) - a random-intercept- "
    "style structure a plain RandomForest fit on x1_h alone can't see."
)
x1_h = rng_hier.normal(size=n_h)
state_effect = rng_hier.normal(scale=10.0, size=n_states)[state_h]
county_effect = rng_hier.normal(scale=5.0, size=len(county_ids))[county_h]
hh_effect = rng_hier.normal(scale=3.0, size=len(hhid_ids))[hhid_h]
var_hier_true = (
    2.0 * x1_h + state_effect + county_effect + hh_effect + rng_hier.normal(scale=1.0, size=n_h)
)

missing_hier_mask = rng_hier.random(n_h) < impute_share
var_hier_with_missing = [
    None if missing_hier_mask[i] else float(var_hier_true[i]) for i in range(n_h)
]

df_hier = pl.DataFrame(
    dict(
        index=np.arange(n_h),
        state=state_h.astype(str),
        county=county_h.astype(str),
        hhid=hhid_h.astype(str),
        x1_h=x1_h,
        var_hier=var_hier_with_missing,
        #   Kept as its own column (never imputed) so it survives SRMI
        #       unchanged - used below to compare against the truth.
        was_missing_hier=missing_hier_mask,
    )
)
df_hier_true = pl.DataFrame(dict(index=np.arange(n_h), var_hier_true=var_hier_true))


def run_hier(name, group_levels):
    var = Variable(
        impute_var="var_hier",
        model=["x1_h"],
        modeltype=Variable.ModelType.RandomForest,
        parameters=Parameters.RandomForest(
            parameters={"n_estimators": 200, "max_depth": 6},
            group_levels=group_levels,
        ),
    )
    srmi_hier = SRMI(
        df=df_hier,
        variables=[var],
        index=["index"],
        replication=SRMI.Replication(n_implicates=1, n_iterations=4),
        parallel=SRMI.Parallel(enabled=False),
        bootstrap=SRMI.Bootstrap(enabled=True),
        storage=SRMI.Storage(
            path_model=f"{config.path_temp_files}/py_srmi_test_group_levels_{name}",
            force_start=True,
        ),
    )
    srmi_hier.run()
    df_out = nw.from_native(srmi_hier.implicates[0].df).lazy().collect().to_native()
    df_check = df_out.join(df_hier_true, on="index").filter(pl.col("was_missing_hier"))
    rmse = float(np.sqrt(((df_check["var_hier"] - df_check["var_hier_true"]) ** 2).mean()))
    logger.info(f"   {name}: RMSE vs. true value (recipients only) = {rmse:.3f}")
    return rmse


logger.info("Without group_levels - x1_h alone has to explain everything")
rmse_without = run_hier("without_group_levels", group_levels=None)

logger.info("With group_levels=['state', 'county', 'hhid'] - coarsest to finest")
rmse_with = run_hier("with_group_levels", group_levels=["state", "county", "hhid"])

logger.info(
    f"RMSE without group_levels = {rmse_without:.2f}, with = {rmse_with:.2f} - "
    f"the shrinkage-heuristic random intercepts pick up the state/county/"
    f"household structure that x1_h alone can't."
)


# %%
logger.info(
    "error=ErrorDraw.leaf - donor matching by tree leaf co-occurrence, "
    "generalized beyond Multinomial()"
)
logger.info(
    "   The default error draw for RandomForest()/XGBoost()/CatBoost()/"
    "SklearnModel() is ErrorDraw.pmm: fit the model, then match each "
    "recipient to a donor via knearest on the scalar prediction "
    "(___prediction), the same PMM mechanism plain Regression() uses. "
    "ErrorDraw.leaf is a different donor-selection rule for the same "
    "models: instead of comparing a single predicted number, it pools "
    "every donor that shares a terminal leaf with the recipient in ANY "
    "tree of the fitted ensemble, weighted by how often they co-occur "
    "across trees, and draws one donor from that pool. This is exactly "
    "the mechanism Multinomial() already uses for its RandomForestClassifier "
    "donor pool (mice's own rf method uses the identical idea for BOTH "
    "categorical and continuous targets) - ErrorDraw.leaf just makes it "
    "available for these models' regular mean-regression targets too, not "
    "only Multinomial's unordered-categorical case."
)
logger.info(
    "   Needs the fitted estimator to expose per-tree leaf indices - "
    ".apply() (scikit-learn's RandomForestRegressor, xgboost's "
    "XGBRegressor) or .calc_leaf_indexes() (CatBoostRegressor). Not "
    "usable on plain Regression() (OLS/Logit have no tree structure to "
    "match on) - trying it there raises a clear error naming what's "
    "missing, rather than silently doing nothing."
)

v_rf_leaf = Variable(
    impute_var="var_rf",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.RandomForest,
    parameters=Parameters.RandomForest(
        parameters={"n_estimators": 200, "max_depth": 8},
        error=Parameters.ErrorDraw.leaf,
    ),
)
srmi_leaf = SRMI(
    df=df,
    variables=[v_rf_leaf],
    index=["index"],
    replication=SRMI.Replication(n_implicates=1, n_iterations=1),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/py_srmi_test_leaf_rf",
        force_start=True,
    ),
)
srmi_leaf.run()
df_leaf_out = nw.from_native(srmi_leaf.implicates[0].df).lazy().collect().to_native()

logger.info(
    "   Donor matching (pmm or leaf) only ever assigns a value someone "
    "actually reported - never an invented number the way ErrorDraw.Random "
    "would. Check that here: every imputed var_rf value is in the set of "
    "originally-observed values."
)
observed_var_rf = set(df.drop_nulls("var_rf")["var_rf"].to_list())
imputed_var_rf = set(df_leaf_out["var_rf"].to_list())
assert imputed_var_rf.issubset(observed_var_rf), (
    "leaf-matched donations should only ever be real observed values"
)
logger.info(f"   All {len(imputed_var_rf)} distinct imputed values were real donor values.")


# %%
logger.info(
    "Variable.ModelType.OrderedCategorical - like Multinomial(), but for "
    "an ORDERED categorical target"
)
logger.info(
    "   Multinomial() treats every category as unordered - fine for "
    "something like industry or occupation code, wrong for something "
    "like an education level or a Likert scale, where the categories "
    "have a real order and a model should be able to say 'a bit higher/"
    "lower than predicted', not just 'which bucket'. OrderedCategorical() "
    "handles that: give it `categories=` in order (lowest/coarsest to "
    "highest/finest), and it fits a mean-regression estimator (default "
    "RandomForestRegressor, or your own factory - same shape as "
    "SklearnModel()'s `factory`) against an integer RANK encoding of "
    "that order, then donates the REAL observed category from a matched "
    "donor - never the numeric rank, and never a category that wasn't "
    "actually observed, same guarantee Multinomial() gives. Donor "
    "matching is either error=pmm (knearest on the predicted rank) or "
    "error=leaf (tree leaf co-occurrence - see the ErrorDraw.leaf "
    "section above)."
)

oc_categories = ["less_than_hs", "hs_grad", "some_college", "college_grad"]
oc_rank_true = np.clip(
    np.round(1.4 * df["x1"].to_numpy() - 0.6 * df["x2"].to_numpy() + rng.normal(scale=1.0, size=n_rows) + 1.5),
    0,
    3,
).astype(int)
oc_missing_mask = rng.random(n_rows) < impute_share
df = df.with_columns(
    pl.Series(
        "education",
        [
            None if oc_missing_mask[i] else oc_categories[oc_rank_true[i]]
            for i in range(n_rows)
        ],
    )
)

v_education = Variable(
    impute_var="education",
    model=["x1", "x2"],
    modeltype=Variable.ModelType.OrderedCategorical,
    parameters=Parameters.OrderedCategorical(
        categories=oc_categories,
        parameters={"n_estimators": 200, "max_depth": 6},
    ),
)
srmi_education = SRMI(
    df=df,
    variables=[v_education],
    index=["index"],
    replication=SRMI.Replication(n_implicates=1, n_iterations=2),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/py_srmi_test_ordered_categorical",
        force_start=True,
    ),
)
srmi_education.run()
df_education_out = (
    nw.from_native(srmi_education.implicates[0].df).lazy().collect().to_native()
)
assert df_education_out["education"].null_count() == 0

logger.info(
    "   Same real-values-only guarantee as leaf/pmm donation generally: "
    "every imputed education value must be one of the four declared "
    "categories, and specifically one that was actually observed."
)
observed_education = set(df.drop_nulls("education")["education"].to_list())
imputed_education = set(df_education_out["education"].to_list())
assert imputed_education.issubset(observed_education)
assert imputed_education.issubset(set(oc_categories))
logger.info(f"   Distribution of imputed values:\n{df_education_out['education'].value_counts()}")
