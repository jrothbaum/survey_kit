"""
SRMI response-propensity diagnostic - answers a real gap in
quality_diagnostics.py's density/strip/box plots: those compare
observed vs. imputed values MARGINALLY, but under MAR the missing
rows can legitimately have a different marginal distribution than the
observed ones (that's the entire point of a conditional imputation
model rather than mean-filling). A marginal comparison flags a
correct imputation as implausible whenever the missing rows differ
systematically on the predictors, and is equally blind to a bad
imputation that happens to preserve the marginal by luck.

The fix: fit a binary LightGBM model predicting each row's own
imputation_flag (was this row originally missing for this variable)
from that variable's own predictor list. The predicted probability is
that row's response propensity - a single scalar summary of "how
similar is this row's covariate profile to a typically-missing row."

Two ways to compare observed vs. imputed conditional on that
propensity, both provided:

1. residual_density (the default, propensity_residual_data() +
   quality_diagnostics.density_long_table()) - matches the diagnostic
   Raghunathan & Bondarenko (2007, "Diagnostics for Multiple
   Imputation") and Bondarenko & Raghunathan (2016) describe, and the
   one actually used in the user's own SRMI/CPS-ASEC paper (Hokayem,
   Raghunathan & Rothbaum): regress y on the propensity (single
   continuous predictor - a LightGBM regression here rather than the
   paper's OLS, for the same reason the propensity model itself is
   LightGBM rather than a logit, fit on the OBSERVED rows only), then
   compare the KERNEL DENSITY of the residuals - observed vs. each
   implicate's own imputed rows, using the SAME observed-fit model for
   everyone. A correctly-specified MAR imputation should give a
   residual density similar in location AND spread to the observed
   one; a shifted or differently-spread residual density for an
   implicate suggests the imputation model is missing something the
   missingness mechanism itself depends on. (Fitting the regression
   model on the observed rows only, then scoring every group against
   that fixed model, is a deliberate choice over the paper's own
   whole-completed-sample fit - it keeps a single, implicate-
   independent "Observed" reference group, consistent with how
   quality_diagnostics.py's own density/strip/box plots already treat
   "observed" as one pooled group rather than repeated M times.)

   The observed rows' own residuals come from CROSS-VALIDATED
   (out-of-fold) predictions, not the same in-sample fit used to score
   the implicate rows - a flexible model regressed on a single
   continuous feature can still overfit given enough leaves/trees, and
   an overfit fit makes the observed group's residuals artificially
   tight (near 0, since the model already saw that exact point) while
   every implicate's residuals stay genuinely out-of-sample - a
   spurious "observed is tighter than imputed" artifact that has
   nothing to do with imputation quality. Cross-validating just the
   observed side keeps the comparison apples-to-apples: both the
   observed group's out-of-fold residuals and the implicates' always-
   out-of-sample residuals come from a model that never saw that exact
   point during its own fit. Same out-of-fold mechanism impute.py's
   own PMM donor-pool cv_folds machinery already uses
   (_pmm_cv_out_of_fold_predictions), reimplemented standalone here
   since this module has no access to an Impute instance.

2. binned_mean (propensity_binned_table()) - a simpler, cruder view
   built before finding the paper above: bin rows by propensity
   (quantiles of the pooled sample) and compare mean(y) - not
   residuals - within each bin. Less standard, but still a reasonable
   secondary look, so kept as an alternative `kind`.

Deliberately in-sample (no train/test split) for the propensity model
itself: this is a descriptive plausibility check, not a causal
propensity-score estimate, so there's no held-out-prediction
requirement to satisfy.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import narwhals as nw
from narwhals.typing import IntoFrameT

from ...utilities.random import RandomNumberGenerator


#   Deliberately shallow/regularized - an unregularized GBM propensity
#       model overfits trivially (in-sample, no holdout) and collapses
#       predictions toward 0/1, which would make every row's propensity
#       nearly degenerate instead of a useful continuous stratifier.
DEFAULT_PROPENSITY_PARAMETERS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 15,
    "num_iterations": 100,
    "learning_rate": 0.05,
    "min_data_in_leaf": 30,
    "test_size": 0,
    "verbose": -1,
}

#   Single-feature (propensity) regression - even shallower than the
#       propensity classifier above, since there's only one split
#       dimension to work with; cross-validated anyway (see
#       propensity_residual_data()) as the real defense against
#       overfitting a single continuous predictor.
DEFAULT_RESIDUAL_MODEL_PARAMETERS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 8,
    "num_iterations": 50,
    "learning_rate": 0.05,
    "min_data_in_leaf": 30,
    "test_size": 0,
    "verbose": -1,
}


def fit_propensity(
    df: IntoFrameT,
    predictors: list[str] | str,
    flag_col: str,
    categorical_feature: list[str] | str | None = None,
    parameters: dict | None = None,
) -> np.ndarray:
    """
    Fit a binary LightGBM classifier predicting `flag_col` (True/1 =
    row was originally missing for this variable) from `predictors`,
    in-sample on `df`, and return the predicted probability
    (propensity) for every row of `df`, in row order.

    `categorical_feature` carries through whichever predictors the
    variable's OWN imputation model already treats as native
    categoricals (e.g. CatBoost()/XGBoost()'s own `categorical_feature`
    dict entry) - LightGBM has its own separate native-categorical
    mechanism, but the *set of columns* that are categorical is a
    property of the data, not of which model reads it, so it's reused
    here rather than falling back to LightGBM guessing (numeric-coding
    a real categorical would corrupt the propensity model's use of it,
    and could silently error on a non-numeric column with no
    categorical_feature declared at all).
    """
    from .lightgbm_wrapper import Survey_kit_Lightgbm as kit_lightgbm

    if categorical_feature is None:
        categorical_feature = []
    elif isinstance(categorical_feature, str):
        categorical_feature = [categorical_feature]
    else:
        categorical_feature = list(categorical_feature)

    params = dict(DEFAULT_PROPENSITY_PARAMETERS)
    if categorical_feature:
        params["categorical_feature"] = categorical_feature
    if parameters:
        params.update(parameters)

    df_native = (
        nw.from_native(df)
        .lazy()
        .collect()
        .with_columns(nw.col(flag_col).cast(nw.Int8))
        .to_native()
    )

    if categorical_feature:
        #   kit_lightgbm's list-formula path (used here, not the
        #       string-formula/R-model-matrix path) does no categorical
        #       recoding of its own - it hands columns straight to
        #       lightgbm's pyarrow ingestion, which rejects anything
        #       but integer/float dtypes outright. Recode to plain
        #       integer category codes ourselves; categorical_feature
        #       in `params` (set above) tells LightGBM to still split
        #       on them natively (by subset, not by threshold) rather
        #       than treating the codes as ordered numbers.
        df_native = df_native.with_columns(
            [
                pl.col(c).cast(pl.Categorical).to_physical().alias(c)
                for c in categorical_feature
                if c in df_native.columns
            ]
        )

    model = kit_lightgbm(
        df=df_native, y=flag_col, formula=predictors, parameters=params
    )
    model.train(show_eval=False)
    df_pred = nw.from_native(model.predict(name="___propensity")).lazy().collect()
    return df_pred["___propensity"].to_numpy()


def _fit_residual_model(y: np.ndarray, p: np.ndarray, parameters: dict):
    """One LightGBM regression of y on the single feature p."""
    from .lightgbm_wrapper import Survey_kit_Lightgbm as kit_lightgbm

    df = pl.DataFrame({"___y": y, "___propensity": p})
    model = kit_lightgbm(
        df=df, y="___y", formula=["___propensity"], parameters=dict(parameters)
    )
    model.train(show_eval=False)
    return model


def _predict_residual_model(model, p: np.ndarray) -> np.ndarray:
    df_predict = pl.DataFrame({"___propensity": p})
    df_pred = (
        nw.from_native(model.predict(df_predict=df_predict, name="___yhat"))
        .lazy()
        .collect()
    )
    return df_pred["___yhat"].to_numpy()


def propensity_residual_data(
    data: dict[str, dict],
    cv_folds: int = 5,
    parameters: dict | None = None,
) -> dict[str, dict]:
    """
    Fit y ~ propensity (a LightGBM regression, one continuous feature)
    on the OBSERVED rows only, then residualize every group - observed
    and each implicate's own imputed rows - against that fit. Returns
    the same {variable: {"observed": [...], "implicates": [[...],
    ...]}} shape _observed_vs_imputed_data() uses (residuals in place
    of raw values), so the result plugs directly into
    quality_diagnostics.density_long_table()/
    observed_vs_imputed_long_table() unmodified.

    The observed group's own residuals come from CROSS-VALIDATED
    (out-of-fold) predictions rather than the same in-sample fit used
    to score the implicate rows - see this module's own docstring for
    why an in-sample fit here would bias observed residuals tighter
    than implicate residuals for reasons that have nothing to do with
    imputation quality. A separate FINAL model, trained on every
    observed row, scores the implicate rows (already genuinely
    out-of-sample, so no CV needed there).

    A variable is skipped (omitted from the output) if there are fewer
    than 2*cv_folds finite (y, propensity) observed pairs (not enough
    for a meaningful fold split), or the observed propensity has zero
    variance (nothing to regress on) - logged by the caller, not here
    (this is a pure data-transform, no logging dependency).

    Parameters
    ----------
    data : dict[str, dict]
        {variable: {"observed": {"y": [...], "propensity": [...]},
        "implicates": [{"y": [...], "propensity": [...]}, ...]}} - the
        SRMI._propensity_data() output.
    cv_folds : int, optional
        Number of folds for the observed group's out-of-fold residuals,
        by default 5. Must be >= 2 to matter.
    parameters : dict | None, optional
        Overrides for the residual model's LightGBM parameters, merged
        onto DEFAULT_RESIDUAL_MODEL_PARAMETERS. By default None.
    """
    params = dict(DEFAULT_RESIDUAL_MODEL_PARAMETERS)
    if parameters:
        params.update(parameters)

    out = {}
    for vrb, groups in data.items():
        y_obs = np.asarray(groups["observed"]["y"], dtype=float)
        p_obs = np.asarray(groups["observed"]["propensity"], dtype=float)
        valid = np.isfinite(y_obs) & np.isfinite(p_obs)
        y_obs, p_obs = y_obs[valid], p_obs[valid]
        if cv_folds < 2 or y_obs.size < 2 * cv_folds or np.std(p_obs) == 0:
            continue

        #   Out-of-fold predictions for the observed group.
        rng = RandomNumberGenerator()
        fold_assignment = rng.integers(0, cv_folds, size=y_obs.size)
        fitted_obs = np.empty(y_obs.size, dtype=float)
        for foldi in range(cv_folds):
            is_holdout = fold_assignment == foldi
            fold_model = _fit_residual_model(
                y_obs[~is_holdout], p_obs[~is_holdout], params
            )
            fitted_obs[is_holdout] = _predict_residual_model(
                fold_model, p_obs[is_holdout]
            )
        residual_obs = (y_obs - fitted_obs).tolist()

        #   Final model, trained on every observed row, scores each
        #       implicate's already-out-of-sample imputed rows.
        final_model = _fit_residual_model(y_obs, p_obs, params)

        def _residualize_implicate(y_list, p_list, model=final_model):
            y = np.asarray(y_list, dtype=float)
            p = np.asarray(p_list, dtype=float)
            valid = np.isfinite(y) & np.isfinite(p)
            y, p = y[valid], p[valid]
            if y.size == 0:
                return []
            fitted = _predict_residual_model(model, p)
            return (y - fitted).tolist()

        out[vrb] = {
            "observed": residual_obs,
            "implicates": [
                _residualize_implicate(imp["y"], imp["propensity"])
                for imp in groups["implicates"]
            ],
        }

    return out


def propensity_binned_table(
    data: dict[str, dict],
    n_bins: int = 10,
) -> pl.DataFrame:
    """
    Bin rows by propensity (quantile bins of the pooled - observed +
    every implicate's imputed - propensity values for that variable)
    and compute mean(y) per (variable, group, bin).

    Parameters
    ----------
    data : dict[str, dict]
        {variable: {"observed": {"y": [...], "propensity": [...]},
        "implicates": [{"y": [...], "propensity": [...]}, ...]}}.
    n_bins : int, optional
        Number of quantile bins of the pooled propensity, by default
        10. Actual bin count may be smaller where propensity values
        are tied heavily enough to collapse quantile edges.

    Returns
    -------
    pl.DataFrame
        Columns: "vrb", "group", "bin", "propensity" (the bin's own
        mean propensity - the plotted x position), "mean_y", "n".
    """
    rows = {
        "vrb": [],
        "group": [],
        "bin": [],
        "propensity": [],
        "mean_y": [],
        "n": [],
    }

    for vrb, groups in data.items():
        all_p = list(groups["observed"]["propensity"])
        for imp in groups["implicates"]:
            all_p.extend(imp["propensity"])
        all_p = np.asarray(all_p, dtype=float)
        all_p = all_p[np.isfinite(all_p)]
        if all_p.size == 0:
            continue

        edges = np.unique(np.quantile(all_p, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:
            continue

        def _binned(y_list, p_list):
            y = np.asarray(y_list, dtype=float)
            p = np.asarray(p_list, dtype=float)
            valid = np.isfinite(y) & np.isfinite(p)
            y, p = y[valid], p[valid]
            if y.size == 0:
                return []
            bin_idx = np.clip(
                np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2
            )
            out = []
            for b in range(len(edges) - 1):
                mask = bin_idx == b
                if mask.sum() == 0:
                    continue
                out.append(
                    (b, float(p[mask].mean()), float(y[mask].mean()), int(mask.sum()))
                )
            return out

        for b, pmid, ymean, n in _binned(
            groups["observed"]["y"], groups["observed"]["propensity"]
        ):
            rows["vrb"].append(vrb)
            rows["group"].append("Observed")
            rows["bin"].append(b)
            rows["propensity"].append(pmid)
            rows["mean_y"].append(ymean)
            rows["n"].append(n)

        for i, imp in enumerate(groups["implicates"], start=1):
            for b, pmid, ymean, n in _binned(imp["y"], imp["propensity"]):
                rows["vrb"].append(vrb)
                rows["group"].append(f"Implicate {i}")
                rows["bin"].append(b)
                rows["propensity"].append(pmid)
                rows["mean_y"].append(ymean)
                rows["n"].append(n)

    if not rows["vrb"]:
        return pl.DataFrame({k: [] for k in rows})

    return pl.DataFrame(rows)
