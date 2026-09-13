from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Callable

from .. import logger


class Parameters:
    """
    Factory class for creating parameter dictionaries for different imputation methods.

    Provides static methods to generate properly formatted parameter dictionaries
    with validation and default values for each imputation approach.
    """

    #   List of acceptable input parameters
    #       Commented out if not yet implemented
    class RegressionModel(Enum):
        """
        OLS = 0
        # Probit = 1
        Logit = 2
        # TwoSampleRegression = 3
        """

        OLS = 0
        # Probit = 1
        Logit = 2
        # TwoSampleRegression = 3

    class ErrorDraw(Enum):
        """
        Random = 0
        pmm = 1
        leaf = 2
        """

        Random = 0
        pmm = 1
        #   Donor matching via tree leaf co-occurrence, instead of
        #       PMM's knearest-on-scalar-yhat - only meaningful for a
        #       tree-ensemble estimator (RandomForest()/XGBoost()/
        #       CatBoost(), or a SklearnModel() wrapping one) that
        #       exposes per-tree leaf/node ids for a row (.apply() or
        #       .calc_leaf_indexes()). Not usable with Regression()
        #       (OLS/Logit have no tree structure to match on) - see
        #       utilities/leaf_donor_matching.py and _run_regression's
        #       "leaf" handling. Same mechanism Multinomial() already
        #       uses for its RandomForestClassifier donor pool, just
        #       generalized to any tree-ensemble mean-regression model.
        leaf = 2
        #   rif_binned = 1
        #   rif_pmm = 2

    @staticmethod
    def _check_package_installed(package: str, why: str):
        """
        Fail immediately, with an actionable message, if `package` isn't
        installed - used by RandomForest/XGBoost/CatBoost so a missing
        dev-only dependency is caught the moment you configure the
        variable, not later, mid-fit, deep inside an SRMI run.
        """
        import importlib.util

        if importlib.util.find_spec(package) is None:
            message = (
                f"{why} needs the '{package}' package, which isn't "
                f"installed - it's a dev-only dependency of survey_kit "
                f"(not required for everyone, since most usage doesn't "
                f"need it). Install it with `uv add --dev {package}` (or "
                f"`pip install {package}`) first."
            )
            logger.error(message)
            raise ImportError(message)

    @staticmethod
    def _warn_dropped_parameters_pmm_keys(
        method_name: str, parameters_pmm: dict, dropped_keys: tuple[str, ...]
    ) -> None:
        """
        Warn if parameters_pmm sets any of dropped_keys to something other
        than Parameters.pmm()'s own default for that key - i.e. the caller
        actually customized a PMM parameter that method_name silently
        doesn't use (e.g. "winsor" for LightGBM(), "model" for anything
        but Regression()/RandomForest()/etc.), so it would
        otherwise have no effect with no indication why. Never warns for a
        key still sitting at pmm()'s own default - parameters_pmm usually
        just defaults to Parameters.pmm() itself, and warning on every
        untouched default would be pure noise.
        """
        pmm_defaults = Parameters.pmm()
        for keyi in dropped_keys:
            if keyi in parameters_pmm and keyi in pmm_defaults:
                if parameters_pmm[keyi] != pmm_defaults[keyi]:
                    message = (
                        f"Parameters.{method_name}(): parameters_pmm sets "
                        f"{keyi}={parameters_pmm[keyi]!r}, but {method_name} "
                        f"doesn't use it - it will be ignored."
                    )
                    logger.warning(message)

    @staticmethod
    def pmm(
        knearest: int = 10,
        model: RegressionModel = RegressionModel.OLS,
        donate_list: list[str] = None,
        winsor: tuple[float, float] = [0, 1],
        donate_by: list[str] | str | None = None,
    ) -> dict:
        """
        Parameters for predictive mean matching imputation.

        Parameters
        ----------
        knearest : int, optional
            Number of nearest neighbors for matching, by default 10
        model : RegressionModel, optional
            Regression model type, by default RegressionModel.OLS
        donate_list : list[str], optional
            Additional variables to impute together, by default None
            i.e., you can predict earnings amount to find a donor,
            but then also impute hours worked and weeks worked with it.
        winsor : tuple[float, float], optional
            Winsorization percentiles, by default [0, 1]
        donate_by : list[str] | str | None, optional
            Grouping variables for donation, by default None

        Returns
        -------
        dict
            PMM parameter dictionary
        """

        if donate_by is None:
            donate_by = []
        elif type(donate_by) is str:
            donate_by = [donate_by]

        if donate_list is None:
            donate_list = []
        return deepcopy(locals())

    @staticmethod
    def LightGBM(
        tune: bool = False,
        parameters: dict | None = None,
        tuner=None,
        quantiles: list = None,
        error: Parameters.ErrorDraw = ErrorDraw.pmm,
        cv_folds: int = 0,
        parameters_pmm: dict | None = None,
    ) -> dict:
        """
        Parameters for LightGBM-based imputation.

        Parameters
        ----------
        tune : bool, optional
            Whether to tune hyperparameters for this run, by default False.
            No effect if `tuner` is None.
        parameters : dict | None, optional
            LightGBM model parameters, by default None
        tuner : Tuner, optional
            Hyperparameter tuner - also owns whether/where tuned parameters
            get cached to disk (Tuner's own path_save_dir/overwrite, since
            the same tuner is typically reused across several variables,
            even across different modeltypes - see utilities.tuning.Tuner /
            HyperparameterSpace), by default None
        quantiles : list, optional
            Quantiles for quantile regression, by default None
        error : Parameters.ErrorDraw, optional
            How to convert yhat from LGBM into imputes.
            If pmm, draw from nearest yhat neighbors, for example.
            The default is ErrorDraw.pmm.
        cv_folds : int, optional
            See Parameters._tabular_ml_params's cv_folds docstring -
            same tabular-ML cv_folds mechanism RandomForest()/XGBoost()/
            CatBoost()/SklearnModel() use, by default 0 (off).
        parameters_pmm : dict | None, optional
            PMM parameters if using PMM error drawing, by default None.
            "winsor" is dropped even if present - unlike pmm()/Regression(),
            LightGBM's own fit path never winsorizes impute_var before
            training (that's only done by _pmm_winsorize_for_fit, called
            from pmm()/regression() - the latter also covers RandomForest/
            XGBoost/CatBoost/SklearnModel, which route through regression(),
            but LightGBM has its own separate fit path that doesn't).

        Returns
        -------
        dict
            LightGBM parameter dictionary

        Raises
        ------
        Exception
            If quantiles are not between 0 and 1
        """

        if quantiles is None:
            quantiles = []

        if any([qi >= 1 for qi in quantiles]):
            message = f"LightGBM quantiles must be between 0 and 1 (passed {quantiles})"
            logger.error(message)
            raise Exception(message)

        return Parameters._tabular_ml_params(
            method_name="LightGBM",
            error=error,
            #   Not a LightGBM() kwarg - lightgbm()/kit_lightgbm's own fit
            #       path never reads "random_share" (that's only consumed
            #       by _run_regression, for RandomForest/XGBoost/CatBoost/
            #       SklearnModel/OLS/Logit), so there's nothing to expose
            #       here beyond this harmless fixed placeholder value.
            random_share=1.0,
            cv_folds=cv_folds,
            parameters_pmm=parameters_pmm,
            tune=tune,
            tuner=tuner,
            extra={
                "parameters": parameters,
                "quantiles": quantiles,
            },
            #   LightGBM's own fit path never winsorizes impute_var (see
            #       the parameters_pmm docstring above) - drop it even if
            #       a passed-in parameters_pmm carries one, so it doesn't
            #       sit in the returned dict implying an effect it
            #       doesn't have.
            reserved=("winsor",),
        )

    @staticmethod
    def HotDeck(
        model_list: list[str] | list[list[str]] = None,
        donate_list: list | None = None,
        n_hotdeck_array: int = 3,
        sequential_drop: bool = True,
    ) -> dict:
        """
        Parameters for hot HotDeck imputation

        Parameters
        ----------
        model_list : list[str] | list[list[str]]
            Each model is a list of variables that are used as match keys.
            model_list can either be a list of strings (the model itself)
            or it can be a list of lists of strings (sequential hot deck to match on)
        donate_list : list, optional
            Additional variables to impute together, by default None
            I.e., you can predict earnings amount to find a donor,
            but then also impute hours worked and weeks worked with it.
        n_hotdeck_array : int, optional
            Size of hot deck donor arrays, by default 3
        sequential_drop : bool, optional
            Drop variables sequentially until matches found, by default True
                If model_list is a list of strings (one model), should we
                sequentially drop the last variable until all recipients find
                a donor?  Makes it easier to set the hot deck/stat match up.
                Whatever recipients are STILL unmatched after every model in
                model_list (including every dropped-down level of this
                cascade) has been tried get matched fully at random instead,
                with a logged warning, rather than being left unmatched
                forever - see impute.py's hotdeck()/statmatch(), the
                guaranteed-to-succeed last resort applies regardless of
                sequential_drop or what model_list contains.

        Returns
        -------
        dict
            Hot deck parameter dictionary
        """
        #   Pending implementation (if I get to) for deterministic hot decks like the ACS
        #   sort_by:list=None):

        if donate_list is None:
            donate_list = []

        #   Placeholder if I ever want to implement an ACS-style sorted
        #       deterministic hot deck
        sort_by = None
        arguments = deepcopy(locals())

        if model_list is not None and len(model_list) > 0 and isinstance(model_list[0], str):
            #   model_list is a single model (list of variable names) - impute.py's
            #   statmatch()/hotdeck() always expect a list of models (list of lists)
            if sequential_drop:
                #   Create a list of models that sequentially drops the last item -
                #       impute.py's hotdeck()/statmatch() try each level in turn,
                #       then fall back to a fully random match (with a warning)
                #       for anyone still unmatched after the shortest level.
                seq_list = []
                for endi in range(len(model_list), 0, -1):
                    if len(model_list[0:endi]) > 0:
                        seq_list.append(model_list[0:endi])

                arguments["model_list"] = seq_list
            else:
                arguments["model_list"] = [model_list]

        #   Only shapes model_list above, at construction time - nothing
        #       downstream (hotdeck()/statmatch()) ever reads it back out
        #       of the stored parameter dict, so keeping it there would
        #       just be an inert, misleading echo of the input.
        del arguments["sequential_drop"]

        return arguments

    @staticmethod
    def StatMatch(
        model_list: list = None, donate_list: list = None, sequential_drop: bool = False
    ):
        """
        Parameters for hot statistical match imputation

        Stat match and hot deck are basically the same, but the hot deck
            iterates over the data carrying arrays of possible donor values
            whereas the stat match just does a join of donors and recipients

        Parameters
        ----------
        model_list : list[str] | list[list[str]]
            Each model is a list of variables that are used as match keys.
            model_list can either be a list of strings (the model itself)
            or it can be a list of lists of strings (sequential hot deck to match on)
        donate_list : list, optional
            Additional variables to impute together, by default None
            I.e., you can predict earnings amount to find a donor,
            but then also impute hours worked and weeks worked with it.
        sequential_drop : bool, optional
            Drop variables sequentially until matches found, by default False
                If model_list is a list of strings (one model), should we
                sequentially drop the last variable until all recipients find
                a donor?  Makes it easier to set the hot deck/stat match up.

        Returns
        -------
        dict
            stat match parameter dictionary
        """
        return Parameters.HotDeck(**locals())

    @staticmethod
    def NearestNeighbor(
        match_to: str | list[str], logit: bool = False, parameters_pmm: dict = None
    ) -> dict:
        """
        Convenience builder for nearest-neighbor PMM matching, purely for
        readability/discoverability - there's no dedicated
        Variable.ModelType.NearestNeighbor; this just returns
        Parameters.Regression()'s own dict shape, so use it with
        Variable.ModelType.Regression. Matching on a fitted OLS/Logit
        prediction (a principled notion of "similar") is strictly better
        than matching on raw, unweighted, unlearned multivariate distance
        across match_to's predictors directly, so this always routes
        through Regression rather than a distance-based implementation of
        its own.

        Parameters
        ----------
        match_to : str | list[str]
            The predictor(s) to match on. NOT stored anywhere in the
            returned dict - Parameters.XXX() functions never set
            Variable.model themselves (true of every one of them, not
            specific to this one), so pass this SAME list as the
            Variable's own `model=` too.
        logit : bool, optional
            impute_var is binary - fit Logit instead of OLS (still
            matched via PMM on the fitted probability, same as OLS's
            fitted mean). By default False (OLS).
        parameters_pmm : dict, optional
            knearest/donate_list/donate_by/winsor - same as pmm()/
            Regression(). By default None (Parameters.pmm()'s own
            defaults).

        Returns
        -------
        dict
            Parameters.Regression(model=OLS or Logit, error=pmm, ...)'s
            parameter dictionary.
        """
        return Parameters.Regression(
            model=(
                Parameters.RegressionModel.Logit
                if logit
                else Parameters.RegressionModel.OLS
            ),
            error=Parameters.ErrorDraw.pmm,
            parameters_pmm=parameters_pmm,
        )

    @staticmethod
    def Regression(
        model: RegressionModel = RegressionModel.OLS,
        error: ErrorDraw = ErrorDraw.pmm,
        random_share: float = 1.0,
        group_levels: list[str] | None = None,
        group_shrinkage_k: float = 10.0,
        parameters_pmm: dict = None,
    ) -> dict:
        """
        Parameters for regression-based imputation.

        Parameters
        ----------
        model : RegressionModel, optional
            Type of regression model, by default RegressionModel.OLS
        error : ErrorDraw, optional
            Method for drawing errors, by default ErrorDraw.Random.
            ErrorDraw.leaf isn't usable here (OLS/Logit have no tree
            structure to match donors on) - see RandomForest()/
            XGBoost()/CatBoost()/SklearnModel() for that.
            If pmm, draw from nearest yhat neighbors, for example.
            If Random, draw from observed errors for modeled observations
        random_share : float, optional
            Fraction of data to use for regression, by default 1.0
            Use less memory by running the regression on a random subset?
        group_levels : list[str] | None, optional
            Nested grouping columns, COARSEST to FINEST (e.g.
            ["state", "county", "hhid"]), for a cheap shrinkage-heuristic
            stand-in for a random-intercept term - not a real mixed
            model, see Impute._nested_group_shrinkage's docstring for the
            exact mechanism and its limits. Re-estimated fresh from each
            SRMI iteration's fit residuals and persisted into the working
            data (as a plain column, the same way donate_list values ride
            along) for the next iteration to build on - no separate inner
            convergence loop. By default None (off).
        group_shrinkage_k : float, optional
            Shrinkage constant for group_levels - a group needs roughly
            this many observations before its own mean starts to
            dominate over being pulled toward 0. Only meaningful when
            group_levels is set. By default 10.0.
        parameters_pmm : dict, optional
            PMM parameters if using PMM error drawing, by default None

        Returns
        -------
        dict
            Regression parameter dictionary
        """
        if group_levels is None:
            group_levels = []

        params = deepcopy(locals())
        if parameters_pmm is None:
            #   leaf needs donate_list/donate_by too (same shape as pmm's
            #       dict, just ignoring knearest/winsor) - see
            #       _tabular_ml_params's identical grouping below.
            if error in (Parameters.ErrorDraw.pmm, Parameters.ErrorDraw.leaf):
                parameters_pmm = Parameters.pmm()
            else:
                parameters_pmm = {}

        for keyi, valuei in parameters_pmm.items():
            if keyi != "model":
                params[keyi] = valuei

        del params["parameters_pmm"]
        return params

    @staticmethod
    def _categorical_enum_dtypes(
        categorical_feature: list[str], *dfs: object
    ) -> dict[str, object]:
        """
        Build {column: pl.Enum(categories)} for categorical_feature's
        columns, categories being the sorted union of unique values across
        every frame in dfs - one frame (e.g. tune_estimator(), which only
        ever sees the data it's tuning against) or several (e.g.
        _categorical_enum_prepare_data below, where the donor pool and
        recipients need the SAME category set or the same column could end
        up with mismatched Enum dtypes between them) both work the same
        way. This is the dtype XGBoost/CatBoost's sklearn APIs actually
        need for native categorical handling.
        """
        import polars as pl

        dtypes = {}
        for coli in categorical_feature:
            categories = set()
            for dfi in dfs:
                categories |= set(dfi[coli].unique().to_list())
            dtypes[coli] = pl.Enum(sorted(categories))
        return dtypes

    @staticmethod
    def _categorical_enum_prepare_data(
        categorical_feature: list[str],
    ) -> Callable[[object, object], tuple[object, object]]:
        """
        Build a prepare_data(df_model, df_impute) hook that casts
        categorical_feature's columns to a shared, fixed-category dtype
        (polars Enum) in both frames - see _categorical_enum_dtypes.
        """

        def _prepare_data(df_model, df_impute):
            import polars as pl

            dtypes = Parameters._categorical_enum_dtypes(
                categorical_feature, df_model, df_impute
            )
            for coli, enum_dtype in dtypes.items():
                df_model = df_model.with_columns(pl.col(coli).cast(enum_dtype))
                df_impute = df_impute.with_columns(pl.col(coli).cast(enum_dtype))
            return df_model, df_impute

        return _prepare_data

    @staticmethod
    def _tabular_ml_params(
        method_name: str,
        error: ErrorDraw,
        random_share: float,
        cv_folds: int,
        parameters_pmm: dict | None,
        extra: dict | None = None,
        reserved: tuple[str, ...] = (),
        tune: bool = False,
        tuner=None,
    ) -> dict:
        """
        Shared parameter-dict assembly for every tabular-ML imputation
        method - LightGBM()/RandomForest()/XGBoost()/CatBoost()/
        SklearnModel() all route their error/random_share/cv_folds/
        parameters_pmm/tune/tuner handling through here, so cv_folds (and
        the tune-before-run mechanism) is one consistently-named,
        consistently-behaved kwarg across all five instead of each function
        reinventing it.

        tune/tuner : see utilities.tuning.Tuner. Also derives
        "tune_overwrite"/"tune_hyperparameter_path" from tuner's own
        overwrite/path_save_dir (rather than exposing them as separate
        XXX() kwargs) - the same tuner is typically reused across several
        variables (even across different modeltypes), so this stays
        consistent automatically instead of needing to be repeated on
        every single Parameters.XXX() call. `tune` is forced to False if
        `tuner` is None (can't tune without a tuner).

        cv_folds is deliberately scoped to these tabular-ML methods only -
        Parameters.pmm()/Regression() (plain OLS/Logit) don't expose it at
        all, since "refit on folds to correct in-sample bias" isn't a
        meaningful notion for a fixed-form parametric regression the way
        it is for these more flexible, potentially-overfit models.

        Parameters
        ----------
        method_name : str
            Name of the calling Parameters.XXX() function, for warning
            messages when a dropped parameters_pmm key was actually
            customized (see _warn_dropped_parameters_pmm_keys).
        error : ErrorDraw
            Method for drawing errors - see Regression()'s error
            docstring.
        random_share : float
            Fraction of data to use for fitting.
        cv_folds : int
            If > 1, compute the donor pool's matching prediction
            (___prediction / ___yhat) via cv_folds-way cross-validation
            instead of the in-sample fit - the model refits on each fold's
            complement and predicts on the held-out fold, so every donor's
            matching value comes from a fit that never saw its own y. This
            corrects a real asymmetry: a model's in-sample predictions on
            its own training data are systematically closer to the truth
            than its predictions on genuinely new (to-be-imputed) rows,
            which can bias PMM matching toward an unrepresentatively
            narrow donor set. Costs cv_folds+1 model fits instead of 1. 0
            (off - use the in-sample fit) is the default everywhere this
            appears.
        parameters_pmm : dict | None
            PMM parameters if using PMM error drawing - defaults to
            Parameters.pmm() when error is pmm.
        extra : dict | None, optional
            Caller-specific keys (e.g. "estimator"/"categorical_feature"
            for the sklearn-style models, "parameters"/"quantiles" for
            LightGBM) merged into the result alongside
            error/random_share/cv_folds/tune/tuner, by default None.
        reserved : tuple[str, ...], optional
            Extra parameters_pmm keys to exclude from the merge, beyond
            "model"/"error"/"random_share"/"cv_folds" (always excluded) -
            e.g. the sklearn-style models also reserve
            "estimator"/"categorical_feature"/"estimator_prepare_data". By
            default ().

        Returns
        -------
        dict
            Assembled parameter dictionary.
        """
        if parameters_pmm is None:
            #   leaf (RandomForest()/XGBoost()/CatBoost()/SklearnModel()
            #       only - see ErrorDraw.leaf's docstring) still needs
            #       donate_list/donate_by from pmm()'s dict, even though
            #       it ignores knearest/winsor - same parameters_pmm shape
            #       as pmm itself, so it's simplest to default it the
            #       same way here rather than inventing a second default.
            parameters_pmm = (
                Parameters.pmm()
                if error in (Parameters.ErrorDraw.pmm, Parameters.ErrorDraw.leaf)
                else {}
            )

        if tuner is None:
            #   Can't tune without a tuner!
            tune = False

        params = {
            "error": error,
            "random_share": random_share,
            "cv_folds": cv_folds,
            "tune": tune,
            "tuner": tuner,
            "tune_overwrite": tuner.overwrite if tuner is not None else False,
            "tune_hyperparameter_path": (
                tuner.path_save_dir if tuner is not None else ""
            ),
        }
        if extra:
            params.update(extra)

        always_reserved = ("model", "error", "random_share", "cv_folds")
        Parameters._warn_dropped_parameters_pmm_keys(
            method_name, parameters_pmm, always_reserved + reserved
        )

        for keyi, valuei in parameters_pmm.items():
            if keyi not in always_reserved and keyi not in reserved:
                params[keyi] = valuei

        return params

    @staticmethod
    def _sklearn_model_params(
        method_name: str,
        model_factory: Callable[[], object],
        error: ErrorDraw,
        random_share: float,
        cv_folds: int,
        categorical_feature: list[str] | str | None,
        parameters_pmm: dict | None,
        prepare_data: Callable[[object, object], tuple[object, object]] | None = None,
        group_levels: list[str] | None = None,
        group_shrinkage_k: float = 10.0,
        tune: bool = False,
        tuner=None,
    ) -> dict:
        """
        Shared parameter-dict assembly for RandomForest()/XGBoost()/
        CatBoost()/SklearnModel() - all four just plug a different
        model_factory (and, where relevant, categorical_feature/
        prepare_data) into the same underlying machinery _run_regression
        already provides for OLS/Logit (formula/model-matrix handling,
        PMM donor matching, cv_folds, error draws, group_levels). This is
        the sklearn-style-only counterpart of Regression()'s own
        group_levels/group_shrinkage_k - see Regression()'s docstring for
        what they do (not LightGBM(), which routes through
        _tabular_ml_params directly rather than through here, and doesn't
        go through _run_regression at all - group_levels isn't wired into
        its own fit path). prepare_data, if any, is owned entirely by the
        model choice itself - _run_regression just calls it, generically,
        without knowing what it does.

        tune/tuner : see _tabular_ml_params - SRMI runs a Tuner.run_estimator()
        pass (using model_factory() as the template estimator) before the
        SRMI run starts, the same way it does for LightGBM's tune=True.
        """
        if categorical_feature is None:
            categorical_feature = []
        elif type(categorical_feature) is str:
            categorical_feature = [categorical_feature]

        if group_levels is None:
            group_levels = []

        return Parameters._tabular_ml_params(
            method_name=method_name,
            error=error,
            random_share=random_share,
            cv_folds=cv_folds,
            parameters_pmm=parameters_pmm,
            tune=tune,
            tuner=tuner,
            extra={
                "estimator": model_factory,
                "categorical_feature": categorical_feature,
                "estimator_prepare_data": prepare_data,
                "group_levels": group_levels,
                "group_shrinkage_k": group_shrinkage_k,
            },
            reserved=(
                "estimator",
                "categorical_feature",
                "estimator_prepare_data",
                "group_levels",
                "group_shrinkage_k",
            ),
        )

    @staticmethod
    def RandomForest(
        parameters: dict | None = None,
        error: ErrorDraw = ErrorDraw.pmm,
        random_share: float = 1.0,
        cv_folds: int = 0,
        group_levels: list[str] | None = None,
        group_shrinkage_k: float = 10.0,
        parameters_pmm: dict = None,
        tune: bool = False,
        tuner=None,
    ) -> dict:
        """
        Parameters for RandomForestRegressor-based imputation (mean
        regression only - scikit-learn's RandomForestRegressor has no
        native quantile-loss support). No native categorical handling
        either - categorical predictors need to be encoded (e.g. via a
        "~...+C(var)+..." formula) before reaching this model, same as
        OLS/Logit.

        Parameters
        ----------
        parameters : dict, optional
            Keyword arguments passed straight through to
            sklearn.ensemble.RandomForestRegressor(**parameters), by
            default None (RandomForestRegressor's own defaults).
        error : ErrorDraw, optional
            Method for drawing errors - see Regression()'s error
            docstring, by default ErrorDraw.pmm. ErrorDraw.leaf is also
            available here (unlike Regression()) - donates by leaf
            co-occurrence in the fitted tree ensemble instead of
            PMM's knearest-on-scalar-yhat, the same mechanism
            Multinomial() uses for its RandomForestClassifier donor
            pool. Requires the fitted model to expose .apply() or
            .calc_leaf_indexes() - see ErrorDraw.leaf's own docstring.
        random_share : float, optional
            Fraction of data to use for fitting, by default 1.0.
        cv_folds : int, optional
            See Parameters._tabular_ml_params's cv_folds docstring, by default 0 (off).
        group_levels : list[str] | None, optional
            See Regression()'s group_levels docstring, by default None (off).
        group_shrinkage_k : float, optional
            See Regression()'s group_shrinkage_k docstring, by default 10.0.
        parameters_pmm : dict, optional
            PMM parameters if using PMM error drawing, by default None.
        tune : bool, optional
            Whether to tune hyperparameters for this run, by default False.
            No effect if `tuner` is None. See utilities.tuning.Tuner -
            SRMI runs a Tuner.run_estimator() pass (over `tuner`'s own
            HyperparameterSpace, against RandomForestRegressor(**parameters)
            as the template estimator) before the SRMI run starts.
        tuner : Tuner, optional
            Hyperparameter tuner - also owns whether/where tuned parameters
            get cached to disk (Tuner's own path_save_dir/overwrite, since
            the same tuner is typically reused across several variables,
            even across different modeltypes), by default None.

        Returns
        -------
        dict
            RandomForest parameter dictionary
        """
        if parameters is None:
            parameters = {}

        def _factory():
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**parameters)

        return Parameters._sklearn_model_params(
            method_name="RandomForest",
            model_factory=_factory,
            error=error,
            random_share=random_share,
            cv_folds=cv_folds,
            categorical_feature=None,
            parameters_pmm=parameters_pmm,
            group_levels=group_levels,
            group_shrinkage_k=group_shrinkage_k,
            tune=tune,
            tuner=tuner,
        )

    @staticmethod
    def XGBoost(
        parameters: dict | None = None,
        error: ErrorDraw = ErrorDraw.pmm,
        random_share: float = 1.0,
        cv_folds: int = 0,
        categorical_feature: list[str] | str | None = None,
        group_levels: list[str] | None = None,
        group_shrinkage_k: float = 10.0,
        parameters_pmm: dict = None,
        tune: bool = False,
        tuner=None,
    ) -> dict:
        """
        Parameters for XGBoost-based imputation (mean regression only -
        see Parameters._tabular_ml_params's cv_folds docstring for why an
        honest donor-pool prediction matters more for flexible models
        like this one).

        Parameters
        ----------
        parameters : dict, optional
            Keyword arguments passed straight through to
            xgboost.XGBRegressor(**parameters), by default None
            (XGBRegressor's own defaults).
        error : ErrorDraw, optional
            Method for drawing errors - see Regression()'s error
            docstring, by default ErrorDraw.pmm. ErrorDraw.leaf is also
            available here (unlike Regression()) - donates by leaf
            co-occurrence in the fitted tree ensemble instead of
            PMM's knearest-on-scalar-yhat, the same mechanism
            Multinomial() uses for its RandomForestClassifier donor
            pool. Requires the fitted model to expose .apply() or
            .calc_leaf_indexes() - see ErrorDraw.leaf's own docstring.
        random_share : float, optional
            Fraction of data to use for fitting, by default 1.0.
        cv_folds : int, optional
            See Parameters._tabular_ml_params's cv_folds docstring, by default 0 (off).
        categorical_feature : list[str] | str | None, optional
            Predictor column names to treat as native categoricals
            (XGBoost's own histogram-based categorical splits, not
            one-hot encoding). Works with either form of the Variable's
            `model=` - a plain column list, or an R-style formula string
            as long as these columns are either left out of the formula
            entirely or referenced there only as a plain numeric term
            (see Variable._validate_estimator_available) - a bare
            reference to a non-numeric-dtype column in the formula still
            gets auto one-hot-encoded there regardless of this setting,
            which is what that validation catches. By default None (no
            native categoricals).
        group_levels : list[str] | None, optional
            See Regression()'s group_levels docstring, by default None (off).
        group_shrinkage_k : float, optional
            See Regression()'s group_shrinkage_k docstring, by default 10.0.
        parameters_pmm : dict, optional
            PMM parameters if using PMM error drawing, by default None.
        tune : bool, optional
            Whether to tune hyperparameters for this run, by default False.
            No effect if `tuner` is None. See utilities.tuning.Tuner -
            SRMI runs a Tuner.run_estimator() pass (over `tuner`'s own
            HyperparameterSpace, against the constructed XGBRegressor as
            the template estimator) before the SRMI run starts.
        tuner : Tuner, optional
            Hyperparameter tuner - also owns whether/where tuned parameters
            get cached to disk (Tuner's own path_save_dir/overwrite, since
            the same tuner is typically reused across several variables,
            even across different modeltypes), by default None.

        Returns
        -------
        dict
            XGBoost parameter dictionary
        """
        Parameters._check_package_installed("xgboost", "Parameters.XGBoost()")

        if categorical_feature is None:
            cat_list = []
        elif type(categorical_feature) is str:
            cat_list = [categorical_feature]
        else:
            cat_list = list(categorical_feature)

        if parameters is None:
            parameters = {}

        def _factory():
            from xgboost import XGBRegressor

            return XGBRegressor(enable_categorical=bool(cat_list), **parameters)

        return Parameters._sklearn_model_params(
            method_name="XGBoost",
            model_factory=_factory,
            error=error,
            random_share=random_share,
            cv_folds=cv_folds,
            categorical_feature=categorical_feature,
            parameters_pmm=parameters_pmm,
            prepare_data=(
                Parameters._categorical_enum_prepare_data(cat_list)
                if cat_list
                else None
            ),
            group_levels=group_levels,
            group_shrinkage_k=group_shrinkage_k,
            tune=tune,
            tuner=tuner,
        )

    @staticmethod
    def CatBoost(
        parameters: dict | None = None,
        error: ErrorDraw = ErrorDraw.pmm,
        random_share: float = 1.0,
        cv_folds: int = 0,
        categorical_feature: list[str] | str | None = None,
        group_levels: list[str] | None = None,
        group_shrinkage_k: float = 10.0,
        parameters_pmm: dict = None,
        tune: bool = False,
        tuner=None,
    ) -> dict:
        """
        Parameters for CatBoost-based imputation (mean regression only).

        Parameters
        ----------
        parameters : dict, optional
            Keyword arguments passed straight through to
            catboost.CatBoostRegressor(**parameters), by default None
            (CatBoostRegressor's own defaults, plus verbose=0).
        error : ErrorDraw, optional
            Method for drawing errors - see Regression()'s error
            docstring, by default ErrorDraw.pmm. ErrorDraw.leaf is also
            available here (unlike Regression()) - donates by leaf
            co-occurrence in the fitted tree ensemble instead of
            PMM's knearest-on-scalar-yhat, the same mechanism
            Multinomial() uses for its RandomForestClassifier donor
            pool. Requires the fitted model to expose .apply() or
            .calc_leaf_indexes() - see ErrorDraw.leaf's own docstring.
        random_share : float, optional
            Fraction of data to use for fitting, by default 1.0.
        cv_folds : int, optional
            See Parameters._tabular_ml_params's cv_folds docstring, by default 0 (off).
        categorical_feature : list[str] | str | None, optional
            Predictor column names to treat as native categoricals
            (CatBoost's own ordered-target-statistic categorical
            handling, not one-hot encoding) - see XGBoost()'s
            categorical_feature docstring for why this is separate from a
            formula's C(...) syntax. By default None (no native
            categoricals).
        group_levels : list[str] | None, optional
            See Regression()'s group_levels docstring, by default None (off).
        group_shrinkage_k : float, optional
            See Regression()'s group_shrinkage_k docstring, by default 10.0.
        parameters_pmm : dict, optional
            PMM parameters if using PMM error drawing, by default None.
        tune : bool, optional
            Whether to tune hyperparameters for this run, by default False.
            No effect if `tuner` is None. See utilities.tuning.Tuner -
            SRMI runs a Tuner.run_estimator() pass (over `tuner`'s own
            HyperparameterSpace, against the constructed CatBoostRegressor
            as the template estimator) before the SRMI run starts.
        tuner : Tuner, optional
            Hyperparameter tuner - also owns whether/where tuned parameters
            get cached to disk (Tuner's own path_save_dir/overwrite, since
            the same tuner is typically reused across several variables,
            even across different modeltypes), by default None.

        Returns
        -------
        dict
            CatBoost parameter dictionary
        """
        Parameters._check_package_installed("catboost", "Parameters.CatBoost()")

        if categorical_feature is None:
            cat_features_list = []
        elif type(categorical_feature) is str:
            cat_features_list = [categorical_feature]
        else:
            cat_features_list = list(categorical_feature)

        if parameters is None:
            parameters = {}

        def _factory():
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                cat_features=cat_features_list, verbose=0, **parameters
            )

        return Parameters._sklearn_model_params(
            method_name="CatBoost",
            model_factory=_factory,
            error=error,
            random_share=random_share,
            cv_folds=cv_folds,
            categorical_feature=categorical_feature,
            parameters_pmm=parameters_pmm,
            prepare_data=(
                Parameters._categorical_enum_prepare_data(cat_features_list)
                if cat_features_list
                else None
            ),
            group_levels=group_levels,
            group_shrinkage_k=group_shrinkage_k,
            tune=tune,
            tuner=tuner,
        )

    @staticmethod
    def SklearnModel(
        factory: Callable[[], object],
        error: ErrorDraw = ErrorDraw.pmm,
        random_share: float = 1.0,
        cv_folds: int = 0,
        categorical_feature: list[str] | str | None = None,
        prepare_data: Callable[[object, object], tuple[object, object]] | None = None,
        group_levels: list[str] | None = None,
        group_shrinkage_k: float = 10.0,
        parameters_pmm: dict = None,
        tune: bool = False,
        tuner=None,
    ) -> dict:
        """
        Parameters for imputation using any sklearn-compatible estimator
        you bring yourself (mean regression only) - the escape hatch for
        anything without a dedicated RandomForest()/XGBoost()/CatBoost()
        function.

        Parameters
        ----------
        factory : Callable[[], object]
            Zero-arg callable returning a fresh, unfitted
            sklearn-compatible estimator (supporting .fit(X, y) and
            .predict(X), plus .fit(..., sample_weight=...) if you're using
            a weight variable). E.g. `lambda: MyModel(some_hyperparam=5)`.
        error : ErrorDraw, optional
            Method for drawing errors - see Regression()'s error
            docstring, by default ErrorDraw.pmm. ErrorDraw.leaf is also
            available here (unlike Regression()) - donates by leaf
            co-occurrence in the fitted tree ensemble instead of
            PMM's knearest-on-scalar-yhat, the same mechanism
            Multinomial() uses for its RandomForestClassifier donor
            pool. Requires the fitted model to expose .apply() or
            .calc_leaf_indexes() - see ErrorDraw.leaf's own docstring.
        random_share : float, optional
            Fraction of data to use for fitting, by default 1.0.
        cv_folds : int, optional
            See Parameters._tabular_ml_params's cv_folds docstring, by default 0 (off).
        categorical_feature : list[str] | str | None, optional
            Predictor column names to cast to a fixed-category dtype
            (polars Enum) in the model matrix before fitting - useful if
            your own estimator wants that dtype for native categorical
            handling, the same way XGBoost/CatBoost do. This only casts
            the dtype; if your model needs the categorical columns
            communicated some other way too (e.g. a constructor kwarg
            naming them), use `prepare_data` instead/as well. By default
            None.
        prepare_data : Callable[[df_model, df_impute], (df_model, df_impute)], optional
            Full control over preparing the model matrix right before
            fitting/predicting - called once, on both the donor pool and
            recipient frames together (so anything derived from both, like
            fixed Enum categories, stays consistent between them), after
            the regular formula/model-matrix construction and before your
            factory's model is fit. By default None (no extra prep beyond
            categorical_feature, if any).
        group_levels : list[str] | None, optional
            See Regression()'s group_levels docstring, by default None (off).
        group_shrinkage_k : float, optional
            See Regression()'s group_shrinkage_k docstring, by default 10.0.
        parameters_pmm : dict, optional
            PMM parameters if using PMM error drawing, by default None.
        tune : bool, optional
            Whether to tune hyperparameters for this run, by default False.
            No effect if `tuner` is None. See utilities.tuning.Tuner - SRMI
            runs a Tuner.run_estimator() pass (over `tuner`'s own
            HyperparameterSpace, against factory() as the template
            estimator) before the SRMI run starts. Your estimator needs a
            real sklearn .set_params()/.get_params() (true of anything
            built on sklearn.base.BaseEstimator) for the tuned
            hyperparameters to actually apply.
        tuner : Tuner, optional
            Hyperparameter tuner - also owns whether/where tuned parameters
            get cached to disk (Tuner's own path_save_dir/overwrite, since
            the same tuner is typically reused across several variables,
            even across different modeltypes), by default None.

        Returns
        -------
        dict
            Custom sklearn-model parameter dictionary
        """
        if categorical_feature:
            cat_list = (
                [categorical_feature]
                if type(categorical_feature) is str
                else list(categorical_feature)
            )
            categorical_prep = Parameters._categorical_enum_prepare_data(cat_list)
            if prepare_data is not None:
                user_prep = prepare_data

                def _combined_prepare_data(df_model, df_impute):
                    df_model, df_impute = categorical_prep(df_model, df_impute)
                    return user_prep(df_model, df_impute)

                prepare_data = _combined_prepare_data
            else:
                prepare_data = categorical_prep

        return Parameters._sklearn_model_params(
            method_name="SklearnModel",
            model_factory=factory,
            error=error,
            random_share=random_share,
            cv_folds=cv_folds,
            categorical_feature=categorical_feature,
            parameters_pmm=parameters_pmm,
            prepare_data=prepare_data,
            group_levels=group_levels,
            group_shrinkage_k=group_shrinkage_k,
            tune=tune,
            tuner=tuner,
        )

    @staticmethod
    def Multinomial(
        parameters: dict | None = None,
        donate_list: list[str] | None = None,
        donate_by: list[str] | str | None = None,
        random_share: float = 1.0,
    ) -> dict:
        """
        Parameters for imputing an unordered categorical variable with 3+
        levels via a RandomForestClassifier and donor matching on leaf
        co-occurrence - see
        imputation/utilities/leaf_donor_matching.leaf_cooccurrence_match
        for the donor-selection mechanism (the same one mice's rf method
        uses: pool donors sharing a leaf with the recipient across every
        tree, draw one uniformly at random).

        This is a distinct imputation shape from
        RandomForest()/XGBoost()/CatBoost()/SklearnModel() above - those
        are all mean regression (a single continuous yhat, optionally
        PMM-matched); this is genuinely multi-class classification, with
        the donor match itself driven by which trees group two rows
        together, not by distance on a predicted scalar. There's no
        error=/cv_folds= here for that reason - PMM-style matching on a
        scalar prediction and "refit on folds to correct in-sample bias"
        both assume a scalar yhat, which doesn't exist for this method.

        Parameters
        ----------
        parameters : dict, optional
            Keyword arguments passed straight through to
            sklearn.ensemble.RandomForestClassifier(**parameters), by
            default None (RandomForestClassifier's own defaults). No
            native categorical predictor handling (same restriction as
            RandomForest()) - model= can be a plain column list or an
            R-style formula string; either way, a categorical predictor
            needs to end up numeric before reaching this model, via the
            formula's own C(...) one-hot encoding or, for the list form,
            by already being numeric-coded.
        donate_list : list[str], optional
            Additional variables to impute together from the same matched
            donor, by default None.
        donate_by : list[str] | str | None, optional
            Grouping variable(s) - donors are only matched within the
            recipient's own group, by default None.
        random_share : float, optional
            Fraction of data to use for fitting, by default 1.0.

        Returns
        -------
        dict
            Multinomial parameter dictionary
        """
        if parameters is None:
            parameters = {}

        if donate_list is None:
            donate_list = []

        if donate_by is None:
            donate_by = []
        elif type(donate_by) is str:
            donate_by = [donate_by]

        return {
            "parameters": parameters,
            "donate_list": donate_list,
            "donate_by": donate_by,
            "random_share": random_share,
        }

    @staticmethod
    def OrderedCategorical(
        categories: list,
        parameters: dict | None = None,
        estimator: Callable[[], object] | None = None,
        error: ErrorDraw = ErrorDraw.pmm,
        random_share: float = 1.0,
        categorical_feature: list[str] | str | None = None,
        estimator_prepare_data: Callable[[object, object], tuple[object, object]]
        | None = None,
        donate_list: list[str] | None = None,
        donate_by: list[str] | str | None = None,
        knearest: int = 10,
    ) -> dict:
        """
        Parameters for imputing an ORDERED categorical variable (e.g. an
        education level or a Likert scale) - unlike Multinomial()
        (unordered, classification), this fits a mean-regression
        estimator against an integer rank encoding of `categories`
        (lowest/coarsest to highest/finest), then donates the REAL
        observed category from a matched donor - never the numeric rank,
        and never a category that wasn't actually observed, the same
        guarantee Multinomial() and PMM/leaf donor matching already give
        elsewhere.

        This reuses the same donor-matching machinery RandomForest()/
        XGBoost()/CatBoost() etc. use for their own error=pmm/leaf, just
        matched on the predicted rank instead of impute_var's own
        (non-numeric) value - see impute.py's ordered_categorical().

        Parameters
        ----------
        categories : list
            Every observed value of impute_var, ordered from
            lowest/coarsest to highest/finest (e.g.
            ["less_than_hs", "hs_grad", "some_college", "college_grad"]).
            Imputation raises if any observed value isn't in this list.
        parameters : dict, optional
            Keyword arguments for the default estimator
            (sklearn.ensemble.RandomForestRegressor(**parameters)) -
            ignored if `estimator` is set. By default None.
        estimator : Callable[[], object], optional
            Zero-arg factory for a fresh, unfitted mean-regression
            estimator (e.g. `lambda: XGBRegressor(...)` or
            `lambda: CatBoostRegressor(...)`), same shape as
            SklearnModel()'s `factory`. By default None (uses
            RandomForestRegressor(**parameters)).
        error : ErrorDraw, optional
            pmm (knearest on the predicted rank) or leaf (tree leaf
            co-occurrence - needs `estimator` to expose leaf indices,
            e.g. the default RandomForestRegressor, or your own
            XGBRegressor/CatBoostRegressor factory - see ErrorDraw.leaf's
            docstring). ErrorDraw.Random isn't meaningful here (there's
            no natural way to add continuous noise to a category and get
            a valid category back). By default ErrorDraw.pmm.
        random_share : float, optional
            Fraction of data to use for fitting, by default 1.0.
        categorical_feature : list[str] | str | None, optional
            Predictor columns to cast to a fixed-category dtype before
            fitting, for an estimator with native categorical handling
            (XGBoost/CatBoost) - see RandomForest()/XGBoost()'s
            docstring. By default None.
        estimator_prepare_data : Callable[[df_model, df_impute], (df_model, df_impute)], optional
            See SklearnModel()'s prepare_data docstring. By default None.
        donate_list : list[str], optional
            Additional variables to impute together from the same
            matched donor, by default None.
        donate_by : list[str] | str | None, optional
            Grouping variable(s) - donors are only matched within the
            recipient's own group, by default None.
        knearest : int, optional
            Only used when error=pmm - number of nearest neighbors (on
            predicted rank) to draw from, by default 10.

        Returns
        -------
        dict
            OrderedCategorical parameter dictionary
        """
        if parameters is None:
            parameters = {}

        if estimator is None:

            def estimator():
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(**parameters)

        if categorical_feature is None:
            categorical_feature = []
        elif type(categorical_feature) is str:
            categorical_feature = [categorical_feature]

        if donate_list is None:
            donate_list = []

        if donate_by is None:
            donate_by = []
        elif type(donate_by) is str:
            donate_by = [donate_by]

        return {
            "categories": list(categories),
            "estimator": estimator,
            "error": error,
            "random_share": random_share,
            "categorical_feature": categorical_feature,
            "estimator_prepare_data": estimator_prepare_data,
            "donate_list": donate_list,
            "donate_by": donate_by,
            "knearest": knearest,
        }

    # @staticmethod
    # def TwoSampleRegression(parameters_regression:dict|None=None,
    #                         is_boolean:bool=False,
    #                         bins:int=10,
    #                         bin_by:list[str] | None=None,
    #                         percentile_cuts:list[float]|None=None,
    #                         save_percentile_cuts:bool=False,
    #                         round_impute_var_digits:int=4,
    #                         continuous_qtiles_y_cuts:int | list[float] | None=None,
    #                         continuous_qtiles_interpolate_by_bin:bool=False,
    #                         #   cond_match_bins:list[float]|None=None,
    #                         min_n_x_var:int=0,
    #                         draw_error:bool=False,
    #                         path_save:str="",
    #                         path_load:str="",
    #                         load_from_save:bool=False) -> dict:
    #     """
    #     Parameters for two-sample regression imputation.

    #     Parameters
    #     ----------
    #     parameters_regression : dict | None, optional
    #         Underlying regression parameters, by default None
    #         (default for Parameters.Regression()).
    #     is_boolean : bool, optional
    #         Whether variable is binary, by default False
    #     bins : int, optional
    #         Number of prediction bins, by default 10
    #         Separate the yhat into


# 				bins and get the actual
# 				p(y = 1|yhat) or impute for E(y|yhat) in bin
# 				to get reasonable impute values
# 				from the LPM. The default is 10.
#     bin_by : list[str] | None, optional
#         Variables for creating separate bins, by default None
#             i.e. have a different expected value and draws by age or something
#             The ddefault is None (ignore)
#     percentile_cuts : list[float] | None, optional
#         Custom percentile cut points, by default None
#     save_percentile_cuts : bool, optional
#         Whether to save cut point information, by default False
#     round_impute_var_digits : int, optional
#         Decimal places for rounding, by default 4
#             For the cut endpoints and y cut quantiles to
#             (for disclosure, generall)
#     continuous_qtiles_y_cuts : int | list[float] | None, optional
#         For continuous variables, what quantiles to run a quantile
#             regression on to approximate the distribution of values
#             to draw from.
#             Default is [0.1,0.25,0.5,0.75,0.9] if None is passed
#     continuous_qtiles_interpolate_by_bin : bool, optional
#         Interpolate quantiles within bins, by default False
#             If the interpolation range is too wide, it can cause problems
#             Set the interpolation range by bin to ensure coverage (at the cost of time)
#     min_n_x_var : int, optional
#         Minimum observations required per predictor, by default 0
#             For disclosure
#     draw_error : bool, optional
#         Draw errors rather than values, by default False
#     path_save : str, optional
#         Path for saving results, by default ""
#     path_load : str, optional
#         Path for loading existing results, by default ""
#     load_from_save : bool, optional
#         Use saved results instead of re-running, by default False

#     Returns
#     -------
#     dict
#         Two-sample regression parameter dictionary
#     """

#     # if additional_match_vars is None:
#     #     additional_match_vars = []

#     # if cond_match_vars is None:
#     #     cond_match_vars = []

#     if parameters_regression is None:
#         parameters_regression = Parameters.Regression()

#     if parameters_regression["model"] == Parameters.RegressionModel.Probit:
#         message = "Probit model not implemented, use Logit"
#         logger.error(message)
#         raise Exception(message)
#     if percentile_cuts is None:
#         percentile_cuts = []


#     if bin_by is None:
#         bin_by = []
#     # if cond_match_bins is None:
#     #     cond_match_bins = []

#     if continuous_qtiles_y_cuts is None:
#         continuous_qtiles_y_cuts = [0.1,0.25,0.5,0.75,0.9]

#     params = deepcopy(locals())
#     if parameters_regression is None:
#         parameters_regression = Parameters.Regression()

#     for keyi, valuei in parameters_regression.items():
#         params[keyi] = valuei

#     del params["parameters_regression"]
#     return params
