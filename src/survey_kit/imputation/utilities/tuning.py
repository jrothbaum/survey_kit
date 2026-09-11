from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from ...utilities.inputs import create_folders_if_needed
from ...utilities.random import generate_seed, RandomNumberGenerator
from ... import logger


def load_tuned_params(path: str) -> dict | None:
    """
    Read back a hyperparameter dict previously saved by Tuner.run() (via
    path_save) - returns None if path is empty or nothing has been saved
    there yet (e.g. the very first run, before any tuning pass has
    completed). Used both by SRMI's own tune-before-run preprocessing (to
    check whether a fresh tuning pass is even needed) and by the actual
    per-iteration fit (to re-load the tuned values fresh from disk every
    time, rather than trying to carry a fitted-in-memory value across a
    deepcopy(variable) or a parallel worker process boundary).
    """
    if path == "" or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@dataclass
class IntRange:
    """An integer hyperparameter, uniform (or log-uniform) between low/high."""

    low: int
    high: int
    log: bool = False

    def suggest(self, trial, name: str) -> int:
        return trial.suggest_int(name, self.low, self.high, log=self.log)


@dataclass
class FloatRange:
    """A float hyperparameter, uniform (or log-uniform) between low/high."""

    low: float
    high: float
    log: bool = False

    def suggest(self, trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, log=self.log)


@dataclass
class Categorical:
    """A hyperparameter chosen from a fixed set of values."""

    choices: list

    def suggest(self, trial, name: str):
        return trial.suggest_categorical(name, list(self.choices))


class HyperparameterSpace:
    """
    A named, typed hyperparameter search space - replaces a bare
    {"num_leaves": [2, 256]} dict with self-documenting IntRange/FloatRange/
    Categorical fields, e.g.:

        HyperparameterSpace(
            num_leaves=IntRange(2, 256),
            learning_rate=FloatRange(1e-3, 0.3, log=True),
            boosting=Categorical(["gbdt", "dart"]),
        )

    Extensible: any object with a `.suggest(trial, name)` method can be used
    as a field value - a new distribution type is just a new small class,
    not a change to HyperparameterSpace or Tuner.
    """

    def __init__(self, **fields):
        self.fields = fields

    def suggest(self, trial) -> dict:
        return {name: spec.suggest(trial, name) for name, spec in self.fields.items()}

    def __or__(self, other: "HyperparameterSpace") -> "HyperparameterSpace":
        """
        Combine two spaces - other's fields win on overlap. Lets a caller
        start from a preset default space and override/extend a few fields,
        e.g. `SomeDefaultSpace() | HyperparameterSpace(num_leaves=IntRange(2, 6))`.
        """
        return HyperparameterSpace(**{**self.fields, **other.fields})


class Objective(Enum):
    """
    A scoring function for comparing a model's raw predictions against
    holdout labels - used by Tuner.run_lightgbm (LightGBM's own .predict()
    doesn't go through an sklearn Estimator, so it can't use run_estimator's
    sklearn `scoring=` string instead). Values must NOT be plain functions -
    Enum silently treats function-valued (or otherwise descriptor-like, e.g.
    functools.partial on newer Python) class attributes as methods rather
    than registering them as real members, so string values are used here
    instead and the actual scoring function is looked up via
    _objective_scorers() below.
    """

    binary_accuracy = "binary_accuracy"
    #   Same Sum/Mean squared error
    sse = "sse"
    mse = "mse"
    mae = "mae"

    def __call__(self, *args, **kwargs):
        return _objective_scorers()[self](*args, **kwargs)


#   Built lazily (not at module level) so importing this module doesn't
#   require sklearn just to define the Objective enum.
_OBJECTIVE_SCORERS = None


def _objective_scorers() -> dict:
    global _OBJECTIVE_SCORERS
    if _OBJECTIVE_SCORERS is None:
        from sklearn.metrics import (
            accuracy_score,
            mean_squared_error,
            mean_absolute_error,
        )

        _OBJECTIVE_SCORERS = {
            Objective.binary_accuracy: accuracy_score,
            Objective.sse: mean_squared_error,
            Objective.mse: mean_squared_error,
            Objective.mae: mean_absolute_error,
        }
    return _OBJECTIVE_SCORERS


class Tuner:
    """
    One tuner class for every imputation model family - the same instance
    is meant to be reused across many Variables (e.g. via SRMI.Defaults),
    each getting its own independent search and its own result:

        tuner = Tuner(
            space=HyperparameterSpace(num_leaves=IntRange(2, 256), ...),
            n_trials=50,
            path_save_dir="tuner_outputs",
            overwrite=True,
        )
        Parameters.LightGBM(tune=True, tuner=tuner, ...)      # var A
        Parameters.LightGBM(tune=True, tuner=tuner, ...)      # var B, same tuner

    Every routing call below (run/run_lightgbm/run_estimator) builds a FRESH
    optuna study - reusing one study across variables would silently mix
    each variable's (unrelated) trials into the same search, and
    study.best_params/best_value would then reflect whichever variable
    happened to score best overall, not the one you just tuned. Only
    space/n_trials/direction/seed/path_save_dir/overwrite are shared state;
    the study itself never is.

    Three ways to use it, in increasing order of how much you write:
    - run_lightgbm(train_data, test_data, base_params) - built-in LightGBM
      routing (CV over lgb.basic.Dataset, scored via `objective`).
    - run_estimator(estimator, X, y, cv=..., scoring=..., fit_params=...) -
      built-in sklearn routing, argument names matching
      sklearn.model_selection's own (GridSearchCV-style: clone(estimator)
      .set_params(**trial_params) per trial).
    - run(score_fn) - the escape hatch for anything else: score_fn(params:
      dict) -> float is called once per trial with that trial's suggested
      hyperparameters; fit/score however you need to, inside your own
      closure. run_lightgbm/run_estimator are themselves just score_fn
      closures built for you and passed to this same method.
    """

    def __init__(
        self,
        space: HyperparameterSpace,
        n_trials: int = 50,
        direction: str | None = None,
        objective: Objective = Objective.mae,
        seed: int = 0,
        sampler=None,
        path_save_dir: str = "",
        overwrite: bool = False,
    ):
        self.space = space
        self.n_trials = n_trials
        #   objective is only meaningful to run_lightgbm (its own default,
        #       overridable per-call) - kept here too so it, like
        #       path_save_dir/overwrite, only needs setting once even
        #       though the same tuner gets reused across several variables.
        self.objective = objective
        self.direction = direction or (
            "maximize" if objective == Objective.binary_accuracy else "minimize"
        )
        self.seed = seed if seed != 0 else generate_seed()
        self._sampler = sampler
        #   A directory, not a single file - the same tuner is typically
        #       reused across several variables. path_save (the actual,
        #       per-variable file this run's result gets pickled to) is
        #       computed by the caller (see srmi.py's _preprocess_tune) as
        #       f"{path_save_dir}/{impute_var}.pickle" right before each
        #       variable's tuning pass.
        self.path_save_dir = path_save_dir
        self.overwrite = overwrite
        self.path_save = ""

    def _new_study(self, direction: str):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = self._sampler or optuna.samplers.TPESampler(seed=self.seed)
        return optuna.create_study(direction=direction, sampler=sampler)

    def run(
        self, score_fn: Callable[[dict], float], direction: str | None = None
    ) -> dict:
        """
        The generic primitive every routing method (including your own) is
        built on: score_fn(params: dict) -> float is called once per trial.
        Runs a FRESH study every call, saves the result to path_save (if
        set) and returns study.best_params.
        """
        study = self._new_study(direction=direction or self.direction)

        def _objective(trial):
            return score_fn(self.space.suggest(trial))

        study.optimize(_objective, n_trials=self.n_trials)

        logger.info(f"Tuner: {len(study.trials)} trials finished")
        logger.info(f"Tuner: best value = {study.best_value:.5f}")
        logger.info(f"Tuner: best params = {study.best_params}")

        best_params = dict(study.best_params)

        if self.path_save != "":
            create_folders_if_needed([os.path.dirname(self.path_save)], quietly=True)
            with open(self.path_save, "wb") as f:
                pickle.dump(best_params, f)

        return best_params

    def run_lightgbm(
        self,
        train_data,
        test_data,
        base_params: dict,
        objective: Objective | None = None,
    ) -> dict:
        """
        LightGBM's own CV: for each trial, train on train_data (an already-
        built lgb.basic.Dataset) with base_params merged with that trial's
        suggested hyperparameters, and score the fitted model's predictions
        on test_data against test_data.label via `objective` (falls back to
        this tuner's own self.objective, set at construction, if not given
        here).
        """
        import lightgbm as lgb

        if objective is None:
            objective = self.objective

        direction = (
            "maximize" if objective == Objective.binary_accuracy else "minimize"
        )

        def _score(trial_params: dict) -> float:
            params = {**base_params, **trial_params}

            if "num_iterations" in params:
                num_boost_round = params.pop("num_iterations")
            else:
                num_boost_round = 100

            gbm_model = lgb.train(
                params=params, train_set=train_data, num_boost_round=num_boost_round
            )
            preds = gbm_model.predict(test_data.data)

            return objective(test_data.label, preds)

        return self.run(_score, direction=direction)

    def run_estimator(
        self,
        estimator,
        X,
        y,
        cv: int = 3,
        scoring: str = "neg_root_mean_squared_error",
        fit_params: dict | None = None,
        direction: str = "maximize",
    ) -> dict:
        """
        sklearn-style CV, argument names matching sklearn.model_selection's
        own (GridSearchCV/cross_val_score: `cv`, `scoring`, `fit_params`) so
        this reads like ordinary sklearn code. For each trial, clones
        `estimator` (an unfitted, already-constructed sklearn-compatible
        estimator - the same clone()+set_params() idiom GridSearchCV itself
        uses), sets that trial's suggested hyperparameters on the clone,
        fits/scores it across `cv` folds (a fixed fold assignment reused
        across every trial, so score differences reflect hyperparameter
        differences, not fold-split noise), and returns the mean fold score.

        Splitting is done natively in polars: X (and y and any per-row
        fit_params, e.g. sample_weight) are attached as columns on one
        combined frame alongside a random fold-id column, then
        `partition_by(fold_id, include_key=False)` both groups AND drops
        the fold-id column in one step - so it can never leak into X as a
        feature. This is this codebase's own native dataframe library
        (used everywhere else in survey_kit) rather than reaching for
        sklearn's private `_safe_indexing` (no backward-compatibility
        guarantee per its own docstring) or supporting pandas (not used
        anywhere in this codebase).

        Parameters
        ----------
        estimator : sklearn-compatible estimator
            An unfitted, already-constructed instance (e.g.
            `RandomForestRegressor()`) - cloned fresh for every trial/fold,
            never fit in place.
        X : polars.DataFrame | numpy.ndarray
            Already numeric/model-matrix-ready training data - a polars
            DataFrame (as Impute._prepare_tuning_data produces - needed to
            keep XGBoost's/CatBoost's native categorical dtype intact
            through to .fit()) or a plain numpy array (wrapped into a
            polars DataFrame internally). No pandas.
        y : array-like
            Target values, length matching X.
        cv : int, optional
            Number of cross-validation folds, by default 3.
        scoring : str, optional
            Any sklearn scoring string (sklearn.metrics.get_scorer_names()),
            by default "neg_root_mean_squared_error".
        fit_params : dict, optional
            Extra keyword arguments passed to every fold's .fit() call (e.g.
            {"sample_weight": w}) - split to each fold the same way X/y
            are, if array-like of matching length; passed as-is otherwise.
            By default None.
        direction : str, optional
            "maximize" or "minimize" - by default "maximize" (matching
            sklearn's own scorer convention, where higher is better - hence
            "neg_..." for error metrics).
        """
        import numpy as np
        import polars as pl
        from sklearn.base import clone
        from sklearn.metrics import get_scorer

        fit_params = fit_params or {}
        scorer = get_scorer(scoring)
        n_rows = len(y)

        X_df = X if isinstance(X, pl.DataFrame) else pl.DataFrame(X)

        #   Same fold assignment reused across every trial (computed once,
        #       not inside _score), so score differences reflect
        #       hyperparameter differences, not fold-split noise. Same
        #       idiom as everywhere else this codebase draws a random
        #       generator (see Impute._pmm_cv_out_of_fold_predictions's own
        #       identical fold-assignment draw for the LightGBM/
        #       RandomForest/etc. cv_folds mechanism) - RandomNumberGenerator(),
        #       not a bare np.random.default_rng(). No set_seed() call here -
        #       SRMI.run() already seeds Python's random module once, at the
        #       top of the whole run; calling set_seed() again here (with
        #       self.seed, a DIFFERENT value than SRMI's own run seed) would
        #       reset that global state mid-run, perturbing every other
        #       random draw that happens afterward.
        rng = RandomNumberGenerator()
        fold_id = rng.integers(0, cv, size=n_rows)

        y_col = "___tuner_y___"
        #   Only fit_params values shaped like one-per-row (e.g.
        #       sample_weight) get bundled as columns and split per fold -
        #       anything else (a scalar, or already fixed for the whole
        #       fit) is passed through to every fold unchanged.
        sliceable_fit_params = {
            k: v for k, v in fit_params.items() if hasattr(v, "__len__") and len(v) == n_rows
        }
        fit_param_cols = {k: f"___tuner_fit_param_{k}___" for k in sliceable_fit_params}

        combined = X_df.with_columns(
            [
                pl.Series(y_col, np.asarray(y)),
                pl.Series("___fold_id___", fold_id),
            ]
            + [
                pl.Series(col, np.asarray(sliceable_fit_params[k]))
                for k, col in fit_param_cols.items()
            ]
        )
        extra_cols = [y_col, *fit_param_cols.values()]

        folds = {
            key[0]: df
            for key, df in combined.partition_by(
                "___fold_id___", include_key=False, as_dict=True
            ).items()
        }

        def _fold_data(df_fold: pl.DataFrame):
            X_fold = df_fold.drop(extra_cols)
            y_fold = df_fold[y_col].to_numpy()
            fold_fit_params = dict(fit_params)
            for k, col in fit_param_cols.items():
                fold_fit_params[k] = df_fold[col].to_numpy()
            return X_fold, y_fold, fold_fit_params

        def _score(trial_params: dict) -> float:
            scores = []
            for foldi in range(cv):
                test_df = folds.get(foldi)
                train_parts = [df for k, df in folds.items() if k != foldi]
                if test_df is None or not train_parts:
                    #   Can happen with very small data / large cv - no
                    #       rows landed in this fold at all.
                    continue
                train_df = pl.concat(train_parts, how="vertical")

                X_train, y_train, fold_fit_params = _fold_data(train_df)
                X_test, y_test, _ = _fold_data(test_df)

                model = clone(estimator).set_params(**trial_params)
                model.fit(X_train, y_train, **fold_fit_params)
                scores.append(scorer(model, X_test, y_test))

            return float(np.mean(scores))

        return self.run(_score, direction=direction)
