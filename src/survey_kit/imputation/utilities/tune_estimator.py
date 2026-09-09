from __future__ import annotations

from typing import Callable

import numpy as np
import narwhals as nw
from narwhals.typing import IntoFrameT

from ...utilities.random import generate_seed
from ... import logger


def tune_estimator(
    df: IntoFrameT,
    y: str,
    x: list[str],
    model_factory: Callable[..., object],
    param_space: dict,
    n_trials: int = 50,
    cv_folds: int = 3,
    weight: str = "",
    scoring: str = "neg_root_mean_squared_error",
    direction: str = "maximize",
    seed: int = 0,
    categorical_feature: list[str] | str | None = None,
) -> dict:
    """
    Run Optuna hyperparameter search for model_factory(**trial_params),
    scored via cv_folds-fold cross-validation, and return the best trial's
    hyperparameters as a plain dict.

    This is a standalone, one-time tuning pass you run yourself against a
    representative slice of your own data (typically the observed rows of
    the variable you're about to impute) - it is not run automatically
    inside SRMI, so a tuning pass isn't repeated for every
    implicate/iteration. Tune once, then feed the returned dict into
    whichever dedicated Parameters function matches model_factory:
    `Parameters.RandomForest(parameters=tune_estimator(...))`,
    `Parameters.XGBoost(parameters=tune_estimator(...))`,
    `Parameters.CatBoost(parameters=tune_estimator(...))` (all three merge
    `parameters` straight into the underlying constructor, same as here),
    or - for a model without a dedicated function -
    `Parameters.SklearnModel(factory=lambda: MyModel(**tune_estimator(...)))`.

    Cross-validation is done with a plain manual fold loop (not
    sklearn's cross_val_score) so sample_weight can be passed straight
    to .fit() without needing sklearn's metadata-routing opt-in, which is
    fragile/version-sensitive for exactly this case.

    Parameters
    ----------
    df : IntoFrameT
        Data to tune against.
    y : str
        Target column name.
    x : list[str]
        Predictor column names - already numeric/model-matrix-ready,
        aside from categorical_feature's columns (see below). This does
        no formula parsing - prepare any other predictor transforms your
        estimator needs (e.g. one-hot encoding) before calling this.
    model_factory : Callable[..., object]
        Given one trial's keyword arguments (matching param_space's keys),
        returns a fresh, unfitted sklearn-compatible estimator. E.g.
        `lambda **p: XGBRegressor(**p)`.
    param_space : dict
        One entry per model_factory keyword argument. Each value is
        either:
            - a (low, high) tuple -> int or float uniform, chosen by
              whether low/high are Python int or float
            - a (low, high, "log") tuple -> log-scale float (typical for
              learning rates)
            - a list -> categorical choices
    n_trials : int, optional
        Number of Optuna trials, by default 50.
    cv_folds : int, optional
        Cross-validation folds used to score each trial - the same fold
        assignment is reused across all trials, so score differences
        reflect hyperparameter differences, not random fold-split noise.
        By default 3.
    weight : str, optional
        Weight column name, passed as sample_weight to .fit() if set, by
        default "" (unweighted). Not used for scoring itself, only
        fitting - most sklearn scorers don't accept weights either.
    scoring : str, optional
        Any sklearn scoring string (see
        sklearn.metrics.get_scorer_names()), by default
        "neg_root_mean_squared_error". Mean-regression scorers only -
        this whole mechanism is mean-regression, matching `estimator`'s
        own scope.
    direction : str, optional
        "maximize" or "minimize" - sklearn scorers are usually oriented
        so higher is better (hence "neg_..." for error metrics, so the
        default of maximizing them minimizes the actual error), by
        default "maximize".
    seed : int, optional
        Random seed for the fold split and Optuna's sampler, by default 0
        (random).
    categorical_feature : list[str] | str | None, optional
        Columns in x to cast to a fixed-category dtype (polars Enum)
        before tuning - the same casting Parameters.XGBoost()/CatBoost()
        apply via their own categorical_feature, so a search over
        model_factory's hyperparameters sees the same native-categorical
        dtype the actual imputation fit will use. Only meaningful for
        estimators with native categorical support (pass
        enable_categorical=True/cat_features=[...] etc. yourself in
        model_factory, same as those Parameters functions do internally -
        this only handles the dtype, not the constructor kwarg). By
        default None (no categorical columns).

    Returns
    -------
    dict
        The best trial's hyperparameters (study.best_params) - a plain
        dict of keyword arguments for model_factory.
    """
    #   Imported here, not at module level - optuna's base import costs
    #       real time and this is the only place in the module that needs
    #       it.
    import optuna
    from sklearn.metrics import get_scorer

    if seed == 0:
        seed = generate_seed()

    df_collected = nw.from_native(df).lazy().collect()
    X = df_collected.select(x).to_native()

    if categorical_feature:
        #   Imported here, not at module level - parameters.py pulls in
        #       polars/optuna's own weight, so keep this lazy like the
        #       optuna import just below.
        from ..parameters import Parameters
        import polars as pl

        cat_list = (
            [categorical_feature]
            if isinstance(categorical_feature, str)
            else list(categorical_feature)
        )
        dtypes = Parameters._categorical_enum_dtypes(cat_list, X)
        X = X.with_columns([pl.col(c).cast(dt) for c, dt in dtypes.items()])

    #   Flat 1D array, not a 1-column frame - fitting/scoring against a
    #       column-vector y works but some estimators (e.g.
    #       RandomForestRegressor) warn loudly about it on every single
    #       fold/trial fit.
    y_arr = df_collected.select(y).to_numpy().ravel()
    n_rows = df_collected.shape[0]

    w_arr = None
    if weight != "":
        w_arr = df_collected.select(weight).to_numpy().ravel()

    #   Same fold assignment for every trial, so score differences
    #       reflect hyperparameter differences, not fold-split noise.
    rng = np.random.default_rng(seed)
    fold_assignment = rng.integers(0, cv_folds, size=n_rows)

    scorer = get_scorer(scoring)

    def _suggest(trial, name, spec):
        if isinstance(spec, (list, tuple)) and len(spec) == 3 and spec[2] == "log":
            return trial.suggest_float(name, spec[0], spec[1], log=True)
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            low, high = spec
            if isinstance(low, int) and isinstance(high, int):
                return trial.suggest_int(name, low, high)
            return trial.suggest_float(name, low, high)
        #   Anything else (a plain list/tuple of any other length) ->
        #       categorical choices.
        return trial.suggest_categorical(name, list(spec))

    def _objective(trial):
        params = {
            name: _suggest(trial, name, spec) for name, spec in param_space.items()
        }

        scores = []
        for foldi in range(cv_folds):
            is_holdout = fold_assignment == foldi
            train_mask = ~is_holdout

            model = model_factory(**params)
            fit_kwargs = {}
            if w_arr is not None:
                fit_kwargs["sample_weight"] = w_arr[train_mask]

            model.fit(X.filter(train_mask), y_arr[train_mask], **fit_kwargs)
            scores.append(scorer(model, X.filter(is_holdout), y_arr[is_holdout]))

        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction=direction, sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(_objective, n_trials=n_trials)

    logger.info(f"tune_estimator: best {scoring} = {study.best_value:.5f}")
    logger.info(f"tune_estimator: best params = {study.best_params}")

    return study.best_params
