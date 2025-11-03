#   TODO - add a feature importance check
#        - Test for save in python/load in R to get important interactions
#           from EIX tool or use shap (try this first)

from __future__ import annotations

import os
import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT
import polars.selectors as pl_cs

from enum import Enum
import lightgbm as lgb
import optuna
import formulaic
import pickle
import random
import gc

from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from copy import deepcopy


from ...utilities.formula_builder import FormulaBuilder
from ...utilities.inputs import create_folders_if_needed
from ...utilities.dataframe import columns_from_list, concat_wrapper, NarwhalsType
from ...utilities.random import set_seed, generate_seed

from ... import logger


class Survey_kit_Lightgbm:
    def __init__(
        self,
        df: IntoFrameT,
        y: str = "",
        x: list | str | None = None,
        weight: str = "",
        formula: str = "",
        tuner=None,
        parameters: dict | None = None,
        formula_exclude_interactions: bool = True,
        formula_remove_factor: bool = True,
        formula_remove_scale: bool = True,
    ):
        if parameters is None:
            parameters = {}
        else:
            parameters = deepcopy(parameters)

        if x is None:
            x = []
        if type(x) is str:
            x = [x]

        self.df = df
        self.nw_type = NarwhalsType(df)
        self.y = y
        self.x = x
        self.weight = weight
        self._formula = formula

        self.formula_exclude_interactions = formula_exclude_interactions
        self.formula_remove_factor = formula_remove_factor
        self.formula_remove_scale = formula_remove_scale
        self.tuner = tuner

        self.formula_processed = False

        #   Set anything based on whats passed, with defaults
        self.parameters = {}
        self.set_parameters(**parameters)

        #   Will be set later in _prepare_params
        self._params_prepared = False
        self.nfold = 0
        self.test_size = 1.0
        self.categorical_feature = []
        self.num_boost_round = 0
        self.categoricals_by_name = []

        #   Set in _prepare_test_train
        self._test_train_prepared = False
        self.train_data = None
        self.test_data = None
        self.train_y = None
        self.test_y = None
        self.extra_eval = {}

        #   Set in train
        self.model = None
        self.evals_result = {}

    def __del__(self):
        if self.train_data is not None:
            self.train_data.data = None
        if self.test_data is not None:
            self.test_data.data = None
        if self.train_y is not None:
            self.train_y = None
        if self.test_y is not None:
            self.test_y = None

        if gc is not None:
            gc.collect()

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        self._formula = value
        self.formula_processed = False

    def set_parameters(self, **kwargs):
        valid_options = list(Survey_kit_Lightgbm._feature_characteristics())

        invalid_passed = list(set(list(kwargs.keys())).difference(valid_options))

        if len(invalid_passed) > 0:
            message = f"Invalid option(s) passed: {', '.join(invalid_passed)}\n"
            message += (
                f"               Acceptable options include: {', '.join(valid_options)}"
            )

            raise Exception(message)

        #   Set any defaults for lightgbm
        defaults = Survey_kit_Lightgbm._parameters_defaults()

        passed_keys = list(kwargs.keys())
        for key, value in defaults.items():
            if key not in passed_keys:
                #   logger.info(f"Adding default lightgbm option {(key + ':').ljust(25)}{value}")
                kwargs[key] = value

        self.parameters.update(kwargs)

    def process_formula(self, other_vars_to_keep: list | str | None = None):
        """


        Parameters
        ----------
        other_vars_to_keep : list|None, optional
            Any other variables to keep that aren't in the formula?
            The default is None.
        remove_factor : bool, optional
            Convert formula factor variables to lgbm categorical. The default is True.
        remove_scale : bool, optional
            Skip rescale, as it is not strictly necessary for lgbm. The default is True.
        exclude_interactions : bool, optional
            Avoid the direct inclusion of interactions and let lgbm handle it?
            The default is True.

        Returns
        -------
        None.

        """

        #   Only do the processing, if it's needed
        if self.formula != "":
            if other_vars_to_keep is None:
                other_vars_to_keep = []
            elif type(other_vars_to_keep) is str:
                other_vars_to_keep = [other_vars_to_keep]

            if type(self.formula) is str:
                self._process_formula_string(other_vars_to_keep=other_vars_to_keep)
            elif type(self.formula) is list:
                self._process_formula_list()

            #   Rename anything that needs to be renamed for light gbm
            self._rename_for_lgb()

            #   We have processed the formula
            self.formula_processed = True

    def _process_formula_string(self, other_vars_to_keep: list | str | None = None):
        categorical_feature = []
        additional_categoricals = []

        #       Parse the formula and see if we need to get a
        #   Simple proxy for needing to go to R and get the model matrix
        #       Does it have a "(" indicating some kind of transformation
        fb = FormulaBuilder(df=self.df, formula=self.formula)

        if self.formula_remove_factor or self.formula_remove_scale:
            [_, additional_categoricals] = fb.recode_to_continuous(
                remove_factor=self.formula_remove_factor,
                remove_scale=self.formula_remove_scale,
            )

        if self.formula_exclude_interactions:
            fb.exclude_interactions(b_exclude_powers=False)

        #   Do we need to get the model matrix from R?
        b_need_mm = fb.formula.find("(") > 0 or fb.formula.find(":") > 0

        if b_need_mm:
            #       Get analysis dataset from formulaic (the model matrix)
            fb.remove_constant()
            df_mm = formulaic.Formula(fb.formula).get_model_matrix(
                nw.from_native(self.df).lazy().collect()
            )

            self.x = nw.from_native(df_mm).lazy().collect_schema().names()

            if self.y == "":
                self.y = fb.lhs()

            y_weight = []
            if self.y != "":
                y_weight.append(self.y)

            if self.weight != "":
                if self.weight not in df_mm.columns:
                    y_weight.append(self.weight)

            if len(other_vars_to_keep) > 0:
                y_weight.extend(other_vars_to_keep)

            #   Replace the dataframe with the model matrix

            if len(
                set(y_weight).intersection(
                    nw.from_native(self.df).collect_schema().names()
                )
            ):
                self.df = concat_wrapper(
                    [
                        (
                            nw.from_native(self.df)
                            .select(y_weight)
                            .lazy()
                            .collect()
                            .to_native()
                        ),
                        (nw.from_native(df_mm).lazy().collect().to_native()),
                    ],
                    how="horizontal",
                )
            else:
                self.df = df_mm
        else:
            #   No need to go to R, just set x from the formula
            self.x = fb.columns_rhs

        #   Update the categoricals in self.parameters
        #       to match the ones in the formula/df that
        #       include the passed categorical/factor variables
        if "categorical_feature" in self.parameters:
            categorical_feature = self.parameters["categorical_feature"]

        if len(categorical_feature):
            categorical_feature = fb.interactions_with_cols_to_list(
                col_check=categorical_feature
            )

        #   Any factors?  Make them categorical features
        if len(additional_categoricals):
            categorical_feature.extend(additional_categoricals)

        if len(categorical_feature):
            self.parameters["categorical_feature"] = categorical_feature

            self.categoricals_by_name = categorical_feature

    def _process_formula_list(self):
        self.x = columns_from_list(df=self.df, columns=self.formula)

        #   Update the categoricals in parameters
        #       to match the ones in the formula/df that
        #       include the passed categorical variables
        categorical_feature = []
        if "categorical_feature" in self.parameters.keys():
            categorical_feature = self.parameters["categorical_feature"]

        if len(categorical_feature) > 0:
            categorical_feature = [
                coli for coli in categorical_feature if coli in self.x
            ]

        if len(categorical_feature):
            self.parameters["categorical_feature"] = categorical_feature
            self.categoricals_by_name = categorical_feature

    def _rename_for_lgb(self):
        rename = {}

        replace_set = {":": "_", "[": "(", "]": ")"}

        #   Just rename and return
        for coli in nw.from_native(self.df).lazy().collect_schema().names():
            rename_to = coli

            b_rename = False
            for keyi, valuei in replace_set.items():
                if keyi in coli:
                    rename_to = rename_to.replace(keyi, valuei)
                    b_rename = True
            if b_rename:
                rename[coli] = rename_to

        if len(rename) > 0:
            self.df = nw.from_native(self.df).rename(rename).to_native()

            categorical_feature = []
            if "categorical_feature" in self.parameters.keys():
                categorical_feature = self.parameters["categorical_feature"]

            if len(categorical_feature) > 0:
                for cati in categorical_feature:
                    if cati in rename.keys():
                        categorical_feature.remove(cati)
                        categorical_feature.append(rename[cati])

                self.parameters["categorical_feature"] = categorical_feature

            x_renamed = []
            for vari in self.x:
                if vari in rename.keys():
                    x_renamed.append(rename[vari])
                else:
                    x_renamed.append(vari)

            self.x = x_renamed

    def _prepare_params(self):
        #   Process formula, from defaults, if needed
        if not self.formula_processed and self.formula != "":
            self.process_formula()

        if self.y == "":
            message = "Must pass a y variable"
            raise Exception(message)

        if len(self.x) == 0:
            logger.info(f"Defaulting to x of all variables in df except {self.y}")
            drop_list = [self.y]

            if self.weight != "":
                logger.info(f"     and {self.weight}")

                drop_list.append(self.weight)

            self.x = [coli for coli in self.df.columns if coli not in drop_list]

        #   Keep weight in model (I know it's dropped above, but that's what I'm doing!)
        if self.weight != "":
            if self.weight not in self.x:
                self.x = self.x + [self.weight]

        #   n-fold validation?
        lgbm_keys = list(self.parameters.keys())
        if "nfold" in lgbm_keys:
            self.nfold = self.parameters["nfold"]
            del self.parameters["nfold"]
        elif "test_size" in lgbm_keys:
            self.test_size = self.parameters["test_size"]
            del self.parameters["test_size"]

        if "num_iterations" in lgbm_keys:
            self.num_boost_round = self.parameters["num_iterations"]
            del self.parameters["num_iterations"]
        else:
            self.num_boost_round = 100

        #   Convert categorical features to indices
        if "categorical_feature" in lgbm_keys:
            self.categorical_feature = [
                self.x.index(vari)
                for vari in self.parameters["categorical_feature"]
                if not vari.startswith("name:") and vari in self.x
            ]
            del self.parameters["categorical_feature"]
        else:
            self.categorical_feature = []

        if "seed" in lgbm_keys:
            set_seed(self.parameters["seed"])
            del self.parameters["seed"]

            self.parameters["seed"] = int(generate_seed())

        self._params_prepared = True

    def _prepare_test_train(self):
        x_train = nw.from_native(self.df).lazy().collect().select(self.x)
        y_train = nw.from_native(self.df).lazy().collect().select(self.y)

        data_params = {}

        list_data_params = ["min_data_in_bin"]
        for parami in list_data_params:
            if parami in self.parameters:
                data_params[parami] = self.parameters[parami]
                del self.parameters[parami]

        extra_data = {}
        if len(self.categorical_feature) > 0:
            extra_data["categorical_feature"] = self.categorical_feature

        if self.test_size > 0:
            x_train, x_test, y_train, y_test = train_test_split(
                x_train.to_arrow(),
                y_train.to_arrow(),
                test_size=self.test_size,
                random_state=int(generate_seed()),
            )
            extra_test = {}

            if self.weight != "":
                weight_test = nw.from_native(x_test).select(self.weight).to_native()
                x_test = nw.from_native(x_test).drop(self.weight).to_native()
                extra_test["weight"] = nw.from_native(weight_test).to_numpy().ravel()

            self.train_y = y_test
            self.test_data = lgb.Dataset(
                (
                    nw.from_native(x_test)
                    .with_columns(cs.boolean().cast(nw.Int8))
                    .to_arrow()
                ),
                label=nw.from_native(y_test).to_numpy().ravel(),
                **extra_test,
                **extra_data,
                free_raw_data=False,
                params=data_params,
            )

            self.extra_eval["valid_sets"] = [self.test_data]

        extra_train = {}
        if self.weight != "":
            weight_train = nw.from_native(x_train).select(self.weight)
            x_train = nw.from_native(x_train).drop(self.weight).to_native()
            extra_train["weight"] = weight_train.to_numpy().ravel()

        self.train_y = y_train

        self.train_data = lgb.Dataset(
            (
                nw.from_native(x_train)
                .with_columns(cs.boolean().cast(nw.Int8))
                .to_arrow()
            ),
            label=(nw.from_native(y_train).to_numpy().ravel()),
            **extra_train,
            **extra_data,
            free_raw_data=False,
            params=data_params,
        )

        self._test_train_prepared = True

    def train(self, show_eval: bool = True):
        #   Parse/process the input parameters, if needed
        if not self._params_prepared:
            self._prepare_params()

        #   Load the test/train data
        if not self._test_train_prepared:
            self._prepare_test_train()

        if self.test_size > 0 and show_eval:
            callbacks = [lgb.log_evaluation(), lgb.record_evaluation(self.evals_result)]
        else:
            callbacks = []

        logger.info(f"Running lightgbm model with parameters: {self.parameters}")
        logger.info(f"     Iterations:                        {self.num_boost_round}")
        logger.info(f"Model:     {self.y}=f({', '.join(self.x)})")
        logger.info(f"Categorical features: {self.categoricals_by_name}")

        if "data_sample_strategy" in self.parameters.keys():
            if self.parameters["data_sample_strategy"] == "goss":
                if "bagging_fraction" in self.parameters.keys():
                    logger.info(
                        "Dropping bagging fraction from parameters as sampling is goss"
                    )
                    del self.parameters["bagging_fraction"]
                if "bagging_freq" in self.parameters.keys():
                    logger.info(
                        "Dropping bagging frequency from parameters as sampling is goss"
                    )
                    del self.parameters["bagging_freq"]

        self.model = lgb.train(
            params=self.parameters,
            train_set=self.train_data,
            callbacks=callbacks,
            num_boost_round=self.num_boost_round,
            **self.extra_eval,
        )
        print("", flush=True)

    def tune(self) -> dict:
        if self.tuner is None:
            message = "Must pass a 'tuner' class"
            raise Exception(message)

        #   Parse/process the input parameters, if needed
        if not self._params_prepared:
            self._prepare_params()

        #   Load the test/train data
        if not self._test_train_prepared:
            self._prepare_test_train()

        self.parameters["verbose"] = -1

        #   Early stopping specified in the tuner, not here
        if "early_stopping_round" in self.parameters.keys():
            del self.parameters["early_stopping_round"]

        if type(self.tuner) is Tuner_optuna:
            if "seed" not in self.tuner.params.keys():
                self.tuner.params["seed"] = random.randint(1, 2**32 - 1)

            objective = self.tuner.get_objective(
                self.train_data, self.test_data, params_lgbm=self.parameters
            )

            self.tuner.study.optimize(objective, n_trials=self.tuner.n_trials)
            logger.info(f"Number of finished trials: {len(self.tuner.study.trials)}")
            logger.info(f"Best trial: {self.tuner.study.best_trial.value}")

            for keyi, valuei in self.tuner.study.best_trial.params.items():
                logger.info(f"{keyi}: {valuei}")

            #   The full final list of lightgbm parameters
            full_params = deepcopy(self.parameters)
            full_params.update(self.tuner.study.best_trial.params)
            print("", flush=True)

            #   Items that arent "real" parameters, but should
            #       be set on the actual run
            drop_params = ["seed", "num_threads", "verbose"]
            for itemi in drop_params:
                if itemi in full_params.keys():
                    del full_params[itemi]

            if self.tuner.path_save != "":
                create_folders_if_needed(
                    [os.path.dirname(self.tuner.path_save)], quietly=True
                )
                with open(self.tuner.path_save, "wb") as f:
                    pickle.dump(full_params, f)

            self.parameters = full_params
        else:
            return None

    def predict(
        self,
        df_predict: IntoFrameT | None = None,
        name: str = "___prediction",
        merged_to_input: bool = False,
    ) -> IntoFrameT:
        if df_predict is not None:
            #   Predict on new data
            nw_type = NarwhalsType(df_predict)

            temp_lgbm = Survey_kit_Lightgbm(
                df=df_predict,
                y=self.y,
                x=self.x,
                weight=self.weight,
                formula=self.formula,
                parameters=self.parameters,
                formula_exclude_interactions=self.formula_exclude_interactions,
                formula_remove_factor=self.formula_remove_factor,
                formula_remove_scale=self.formula_remove_scale,
            )

            temp_lgbm.process_formula()

            df_prediction = (
                nw.Series.from_numpy(
                    name=name,
                    values=self.model.predict(
                        data=(
                            nw.from_native(temp_lgbm.df)
                            .select(self.train_data.get_data().schema.names)
                            .with_columns(cs.boolean().cast(nw.Int8))
                            .lazy()
                            .collect()
                            .to_arrow()
                        ),
                        # predict_disable_shape_check=True,
                    ),
                    backend="polars",
                )
                .to_frame()
                .lazy_backend(nw_type)
                .to_native()
            )

            if merged_to_input:
                df_prediction = concat_wrapper(
                    [df_predict, df_prediction], how="horizontal"
                )

            return NarwhalsType.return_df(df_prediction, nw_type)
        else:
            #   Predict on model data
            df_prediction = (
                nw.Series.from_numpy(
                    name=name,
                    values=self.model.predict(
                        data=(
                            nw.from_native(self.df)
                            .select(self.train_data.get_data().schema.names)
                            .with_columns(cs.boolean().cast(nw.Int8))
                            .lazy()
                            .collect()
                            .to_arrow()
                        )
                    ),
                    #   schema={name:nw.Float64},
                    backend="polars",
                )
                .to_frame()
                .lazy_backend(self.nw_type)
            )

            if merged_to_input:
                df_prediction = concat_wrapper(
                    [self.df, df_prediction], how="horizontal"
                )

            return NarwhalsType.return_df(df_prediction, self.nw_type)

    def load_tuned_parameters(
        self, path: str = "", error_on_missing: bool = True
    ) -> bool:
        if path == "":
            path = self.tuner.path_save

        if os.path.exists(path):
            with open(path, "rb") as f:
                tuned_params = pickle.load(f)

            self.parameters.update(tuned_params)

            return True
        else:
            message = f"No parameters exist at {path}"
            logger.info(message)
            if error_on_missing:
                raise Exception(message)

            return False

    def importance(
        self,
        # interactions:bool=False,
        # use_r:bool=False,
        with_rank: bool = False,
    ) -> IntoFrameT:
        df = nw.from_native(self.df).select(self.x)

        importance_gain = self.model.feature_importance(importance_type="gain")
        importance_split = self.model.feature_importance(importance_type="split")

        df_importance = concat_wrapper(
            [
                nw.from_dicts(
                    {"Feature": df.lazy().collect_schema().names()}, backend="polars"
                ),
                nw.Series.from_numpy(
                    name="Gain", values=importance_gain, backend="polars"
                ).to_frame(),
                nw.Series.from_numpy(
                    name="Frequency", values=importance_split, backend="polars"
                ).to_frame(),
            ],
            how="horizontal",
        )

        df_importance = (
            nw.from_native(df_importance)
            .filter(nw.col("Frequency") > 0)
            .sort("Gain", descending=True)
            .with_columns(
                [
                    (nw.col("Gain") / nw.sum("Gain")).alias("Gain"),
                    (nw.col("Frequency") / nw.sum("Frequency")).alias("Frequency"),
                ]
            )
            .to_native()
        )

        if with_rank:
            df_importance = df_importance.with_columns(
                (~pl_cs.by_name("Feature")).rank(descending=True).name.prefix("rank_")
            )

        return nw.from_native(df_importance).lazy_backend(self.nw_type).to_native()

    def _feature_characteristics(
        feature: str = "", tunable_only=False
    ) -> tuple[type, bool] | dict:
        """
        Returns the type and a bool for whether the feature is tunable
        """

        tunable_ints = [
            "max_depth",
            "min_data_in_leaf",
            "bagging_freq",
            "min_data_per_group",
            "num_leaves",
            "max_bin",
            "num_iterations",
        ]

        tunable_floats = [
            "bagging_fraction",
            "feature_fraction",
            "lambda_l1",
            "lambda_l2",
            "min_gain_to_split",
            "learning_rate",
        ]

        non_tunable_strs = [
            "objective",
            "metric",
            "boosting",
            "data_sample_strategy",
            "tree_learner",
        ]

        non_tunable_floats = ["alpha", "test_size"]
        non_tunable_ints = [
            "num_threads",
            "seed",
            "num_class",
            "nfold",
            "early_stopping_round",
            "min_data_in_bin",
            "verbose",
        ]

        non_tunable_lists = ["categorical_feature"]

        if feature == "":
            d_out = {}
            d_out.update({feature: [int, True] for feature in tunable_ints})
            d_out.update({feature: [float, True] for feature in tunable_floats})

            if not tunable_only:
                d_out.update({feature: [str, False] for feature in non_tunable_strs})

                d_out.update(
                    {feature: [float, False] for feature in non_tunable_floats}
                )

                d_out.update({feature: [int, False] for feature in non_tunable_ints})

                d_out.update({feature: [list, False] for feature in non_tunable_lists})

            return d_out
        else:
            if feature in tunable_ints:
                return [int, True]
            elif feature in tunable_floats:
                return [float, True]
            elif feature in non_tunable_strs:
                return [str, False]
            elif feature in non_tunable_floats:
                return [float, False]
            elif feature in non_tunable_ints:
                return [int, False]
            elif feature in non_tunable_lists:
                return [list, False]
            else:
                return [None, False]

    def _parameters_defaults():
        params = {}

        params["objective"] = "regression"
        params["metric"] = "rmse"
        params["boosting"] = "gbdt"
        params["test_size"] = 0
        params["min_data_per_group"] = 25

        if (
            os.environ["OMP_NUM_THREADS"] is not None
            and os.environ["OMP_NUM_THREADS"] != ""
        ):
            try:
                cpus = int(os.environ["OMP_NUM_THREADS"])
                params["num_threads"] = max(1, cpus)
            except:
                pass

        params["seed"] = random.randint(1, 2**32 - 1)
        params["num_iterations"] = 100
        params["verbose"] = -1

        return params


class Tuner:
    def __init__(self):
        pass

    class Objectives(Enum):
        binary_accuracy = accuracy_score
        #   Same Sum/Mean squared error
        sse = mean_squared_error
        mse = mean_squared_error

        mae = mean_absolute_error


class Tuner_optuna:
    def __init__(
        self,
        n_trials: int = 10,
        params: dict | None = None,
        hyperparameters: dict | None = None,
        study: optuna.study = None,
        objective: Tuner.Objectives = Tuner.Objectives.sse,
        path_save: str = "",
        test_size: float = 0.5,
        nfold: int = 3,
    ):
        if params is None:
            params = {}
        if hyperparameters is None:
            hyperparameters = {}

        self.n_trials = n_trials
        self.params = params
        self.hyperparameters = hyperparameters
        self.objective = objective
        self.study = study
        self.path_save = path_save
        self.test_size = test_size
        self.nfold = nfold

    def parameters(
        self,
        study: optuna.study | None = None,
        sampler: optuna.sampler | None = None,
        #   Pass the callbacks
        callbacks: list | None = None,
        seed: int = 0,
        #   Or pass callback items
        n_early_stopping: int | None = None,
        n_log_evaluation: int | None = None,
    ):
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)

        bDefaultSampler = False
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed)

            bDefaultSampler = True

        if self.study is None:
            if bDefaultSampler:
                logger.info("Setting default optuna sampler: TPESampler")

            if self.objective in [tuner_objectives.binary_accuracy]:
                direction = "maximize"
            else:
                direction = "minimize"
            self.study = optuna.create_study(sampler=sampler, direction=direction)
        if callbacks is None:
            callbacks = []

            if n_early_stopping is not None:
                callbacks.append(early_stopping(n_early_stopping))

            if n_log_evaluation is not None:
                callbacks.append(log_evaluation(n_log_evaluation))

        if len(callbacks) > 0:
            self.params["callbacks"] = callbacks
        if seed > 0:
            self.params["optuna_seed"] = seed

    def _ranges(trial=None, params: dict = None):
        if params is None:
            params = {}
        else:
            params = deepcopy(params)

        valid_options = Survey_kit_Lightgbm._feature_characteristics(tunable_only=True)

        invalid_passed = list(
            set(list(params.keys())).difference(list(valid_options.keys()))
        )

        if len(invalid_passed) > 0:
            message = f"Invalid option(s) passed: {', '.join(invalid_passed)}\n"
            message += (
                f"               Acceptable options include: {', '.join(valid_options)}"
            )

            raise Exception(message)

        message = ""
        for key, value in params.items():
            [typei, _] = valid_options[key]

            if typei is int:
                if type(value[0]) is not int or type(value[1]) is not int:
                    message += f"               Invalid value passed for {key}: passed [{value[0]},{value[1]}] but expects {typei}\n"
                else:
                    params[key] = trial.suggest_int(key, value[0], value[1])
            if typei is float:
                if (type(value[0]) is not int and type(value[0]) is not float) or (
                    type(value[1]) is not int and type(value[1]) is not float
                ):
                    message += f"               Invalid value passed for {key}: passed [{value[0]},{value[1]}] but expects {typei}\n"
                else:
                    if len(value) > 2:
                        params[key] = trial.suggest_float(
                            key, value[0], value[1], log=value[2]
                        )
                    else:
                        params[key] = trial.suggest_float(key, value[0], value[1])

        if message != "":
            message = f"Invalid hyperparameter range input:\n{message}"
            raise Exception(message)

        return params

    def get_objective(
        self,
        d_train: lgb.basic.Dataset,
        d_test: lgb.basic.Dataset,
        params_lgbm: dict = None,
    ):
        params = deepcopy(params_lgbm)

        def _objective(trial):
            trial_hyperparams = tuner_optuna._optuna_ranges(trial, self.hyperparameters)
            for keyi, valuei in trial_hyperparams.items():
                params[keyi] = trial_hyperparams[keyi]

            if "num_iterations" in params.keys():
                num_boost_round = params["num_iterations"]
                del params["num_iterations"]
            else:
                num_boost_round = 100

            gbm_model = lgb.train(
                params=params, train_set=d_train, num_boost_round=num_boost_round
            )

            preds = gbm_model.predict(d_test.data)

            return self.objective(d_test.label, preds)

        return _objective


class tuner_objectives(Enum):
    binary_accuracy = accuracy_score
    #   Same Sum/Mean squared error
    sse = mean_squared_error
    mse = mean_squared_error

    mae = mean_absolute_error


class tuner_optuna:
    def __init__(
        self,
        n_trials: int = 10,
        params: dict | None = None,
        hyperparameters: dict | None = None,
        study: optuna.study = None,
        objective: tuner_objectives = tuner_objectives.sse,
        path_save: str = "",
        test_size: float = 0.5,
        nfold: int = 3,
    ):
        if params is None:
            params = {}
        if hyperparameters is None:
            hyperparameters = {}

        self.n_trials = n_trials
        self.params = params
        self.hyperparameters = hyperparameters
        self.objective = objective
        self.study = study
        self.path_save = path_save
        self.test_size = test_size
        self.nfold = nfold

    def parameters(
        self,
        study: optuna.study | None = None,
        sampler: optuna.sampler | None = None,
        #   Pass the callbacks
        callbacks: list | None = None,
        seed: int = 0,
        #   Or pass callback items
        n_early_stopping: int | None = None,
        n_log_evaluation: int | None = None,
    ):
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)

        bDefaultSampler = False
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed)

            bDefaultSampler = True

        if self.study is None:
            if bDefaultSampler:
                logger.info("Setting default optuna sampler: TPESampler")

            if self.objective in [tuner_objectives.binary_accuracy]:
                direction = "maximize"
            else:
                direction = "minimize"
            self.study = optuna.create_study(sampler=sampler, direction=direction)
        if callbacks is None:
            callbacks = []

            if n_early_stopping is not None:
                callbacks.append(early_stopping(n_early_stopping))

            if n_log_evaluation is not None:
                callbacks.append(log_evaluation(n_log_evaluation))

        if len(callbacks) > 0:
            self.params["callbacks"] = callbacks
        if seed > 0:
            self.params["optuna_seed"] = seed

    def _optuna_ranges(trial=None, params: dict = None):
        if params is None:
            params = {}
        else:
            params = deepcopy(params)

        valid_options = Survey_kit_Lightgbm._feature_characteristics(tunable_only=True)

        invalid_passed = list(
            set(list(params.keys())).difference(list(valid_options.keys()))
        )

        if len(invalid_passed) > 0:
            message = f"Invalid option(s) passed: {', '.join(invalid_passed)}\n"
            message += (
                f"               Acceptable options include: {', '.join(valid_options)}"
            )

            raise Exception(message)

        message = ""
        for key, value in params.items():
            [typei, _] = valid_options[key]

            if typei is int:
                if type(value[0]) is not int or type(value[1]) is not int:
                    message += f"               Invalid value passed for {key}: passed [{value[0]},{value[1]}] but expects {typei}\n"
                else:
                    params[key] = trial.suggest_int(key, value[0], value[1])
            if typei is float:
                if (type(value[0]) is not int and type(value[0]) is not float) or (
                    type(value[1]) is not int and type(value[1]) is not float
                ):
                    message += f"               Invalid value passed for {key}: passed [{value[0]},{value[1]}] but expects {typei}\n"
                else:
                    if len(value) > 2:
                        params[key] = trial.suggest_float(
                            key, value[0], value[1], log=value[2]
                        )
                    else:
                        params[key] = trial.suggest_float(key, value[0], value[1])

        if message != "":
            message = f"Invalid hyperparameter range input:\n{message}"
            raise Exception(message)

        return params

    def get_objective(
        self,
        d_train: lgb.basic.Dataset,
        d_test: lgb.basic.Dataset,
        params_lgbm: dict = None,
    ):
        params = deepcopy(params_lgbm)

        def _objective(trial):
            trial_hyperparams = tuner_optuna._optuna_ranges(trial, self.hyperparameters)
            for keyi, valuei in trial_hyperparams.items():
                params[keyi] = trial_hyperparams[keyi]

            if "num_iterations" in params.keys():
                num_boost_round = params["num_iterations"]
                del params["num_iterations"]
            else:
                num_boost_round = 100

            gbm_model = lgb.train(
                params=params, train_set=d_train, num_boost_round=num_boost_round
            )

            preds = gbm_model.predict(d_test.data)

            return self.objective(d_test.label, preds)

        return _objective


# if __name__ == "__main__":


#     def test_class():
#         import NEWS
#         from NEWS.CodeUtilities.Python.Parquet.IO import readParquet
#         from NEWS.CodeUtilities.Python.Random import set_seed,\
#                                                      RandomNumberGenerator
#         set_seed(53215684)


#         SurveyYear = 2019
#         parameters = NEWS.Parameters.NEWSParameters.create_parameters()


#         cps_path = parameters["DataRoot"] + "/Extracts/Survey/CPS_ASEC/" + str(SurveyYear) + ".parquet"
#         df = readParquet(parquetFullPath=cps_path,
#                          Obs=50_000,
#                          Columns=["ern_val",
#                                   "ss_val",
#                                   "a_age",
#                                   "a_hga",
#                                   "gereg",
#                                   "gestfips",
#                                   'wkswork',
#                                   'hrswk',
#                                   'prdtrace',
#                                   "marsupwt"],
#                          LazyLoad=False,
#                          Where="marsupwt is not null",
#                          quietly=True)

#         y = "ern_val"
#         weight = "marsupwt"
#         x = df.select(pl.exclude(y,weight)).columns


#         initial_seed = RandomNumberGenerator().integers(low=0,high=2**31-1,size=1)[0]
#         params_lgbm = {"test_size":0.25,
#                        "seed":initial_seed,
#                        "categorical_feature":["gereg",
#                                             "gestfips",
#                                             'prdtrace']}

#         from NEWS.CodeUtilities.Python.FormulaBuilder import FormulaBuilder
#         fb = FormulaBuilder(LHS="ern_val",
#                             Constant=True,
#                             df=df)
#         fb.Factor(Columns=["gereg"])
#         fb.Interaction(Clause1=FormulaBuilder.Factor(df=fb.df,
#                                                       Columns=["gereg"]),
#                        Clause2="hrswk+wkswork")
#         fb.Continuous(Columns=["gestfips",
#                                "a_hga",
#                                "wkswork",
#                                "hrswk",
#                                "prdtrace"])


#         tuner = Tuner_optuna(n_trials=5,
#                              objective=Tuner.Objectives.sse)
#         tuner.parameters()
#         tuner.hyperparameters["num_leaves"] = [2,256]
#         tuner.hyperparameters["max_depth"] = [2,256]
#         # tuner.hyperparameters["min_data_in_leaf"] = [10,250]
#         tuner.hyperparameters["num_iterations"] = [25,500]

#         lgbm = Survey_kit_Lightgbm(df=df,
#                              y=y,
#                              x=x,
#                              weight=weight,
#                              formula=fb.formula,
#                              parameters=params_lgbm,
#                              tuner=tuner,
#                              formula_exclude_interactions=False)


#         #   lgbm.tune()
#         lgbm.train()

#         logger.info(lgbm.predict(df))

#         #   logger.info(lgbm.importance(interactions=True))
#         return lgbm

#     def test_small_sample_predict():
#         import NEWS
#         from NEWS.CodeUtilities.Python.Parquet.IO import readParquet
#         from NEWS.CodeUtilities.Python.Random import set_seed,\
#                                                      RandomNumberGenerator
#         df = readParquet("/projects/data/NEWS/Test/program_impute_input.parquet")

#         import NEWS.Processing.Imputation._python_utilities.Survey.ModelDefaults as model_defaults

#         program = "tanf"
#         program_year = "t1"
#         model = model_defaults.model(Survey="CPS",
#                                      SurveyYear=2019,
#                                      small=True)

#         model = df.news.columns(model,
#                                 exclude=[f"{program}_*"])


#         var_yn = f"{program}_value_in_hh_head_yn_{program_year}"
#         from NEWS.CodeUtilities.Python.Bootstrap import bayes_bootstrap_weights

#         df = bayes_bootstrap_weights(df=df,
#                                      weight="marsupwt",
#                                      new_weight="wgt_test",
#                                      n_replicates=1)

#         lgbm = Survey_kit_Lightgbm(df=df.filter(pl.col(var_yn).filter(pl.col("bbweight__1") > 0)),
#                              y='tanf_value_in_hh_head_t1',
#                              x=model,
#                              weight="bbweight__1",
#                              parameters={'objective': 'regression', 'num_leaves': 12, 'min_data_in_leaf': 2, 'num_iterations': 77, 'test_size': 0.2, 'boosting': 'gbdt', 'categorical_feature': ['a_age_cat', 'nHH_ge_65', 'cit_cat', 'mar_cat', 'nHH_', 'ed_cat', 'lkweeks_cat', 'a_hrs1_cat', 'nHH_le_17', 'a_ind_group', 'gestfips', 'industry_group', 'sp_group', 'wkswork_cat', 'occup_group', 'geco', 'nHH_18_64', 'hrswk_cat', 'a_occ_group', 'nHH_le_5', 'nHH_le_1'], 'verbose': -99, 'min_data_in_bin': 2, 'metric': 'rmse', 'min_data_per_group': 2, 'num_threads': 10, 'seed': 2860830418, 'max_depth': 3, 'feature_fraction': 0.6036767753068146})

#         lgbm.parameters["seed"] = 1234
#         lgbm.parameters["test_size"] = 0.25
#         logger.info(lgbm.parameters)
#         lgbm.train()
#         logger.info(lgbm.importance())

#         test1 = lgbm.parameters
#         test2 = {'objective': 'regression', 'num_leaves': 12, 'min_data_in_leaf': 2, 'boosting': 'gbdt', 'verbose': -99, 'metric': 'rmse', 'min_data_per_group': 2, 'num_threads': 10, 'max_depth': 3, 'feature_fraction': 0.6036767753068146, 'seed': 1591752070}

#         for keyi, valuei in test1.items():
#             if keyi in test2:
#                 logger.info(f"{keyi}: {valuei}, {test2[keyi]}")
#             else:
#                 logger.info(f"{keyi}: {valuei}, NONE")

#         for keyi, valuei in test2.items():
#             if keyi not in test1:
#                 logger.info(f"{keyi}: NONE, {valuei}")


#         return df
#     test = test_class()
