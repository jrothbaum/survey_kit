#   TODO - add a feature importance check
#        - Test for save in python/load in R to get important interactions
#           from EIX tool or use shap (try this first)

from __future__ import annotations

import os
import numpy as np
import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT
import polars as pl
import polars.selectors as pl_cs

import pickle
import random

#   lightgbm/optuna/sklearn are all imported lazily at their points of use
#   below (not here) - their base imports cost roughly a second combined,
#   and this module gets imported unconditionally by both srmi.py and
#   impute.py regardless of whether LightGBM is actually the modeltype in
#   use. That cost is also paid per-process under SRMI's parallel execution,
#   since each worker re-imports the module from scratch.

from copy import deepcopy


from ...utilities.formula_builder import FormulaBuilder, get_model_frame
from ...utilities.dataframe import (
    columns_from_list,
    concat_wrapper,
    NarwhalsType,
    lazy_backend,
)
from ...utilities.random import set_seed, generate_seed
from .tuning import HyperparameterSpace, Tuner, Objective

from ... import logger


class Survey_kit_Lightgbm:
    def __init__(
        self,
        df: IntoFrameT,
        y: str = "",
        x: list | str | None = None,
        weight: str = "",
        formula: str = "",
        tuner: Tuner | None = None,
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
        #   {column: pl.Enum(categories)} - fixed at training time (see
        #       _process_formula_list) from the training data's own
        #       category values, then reused as-is by
        #       process_predict_frame() so a later predict() call
        #       recodes df_predict's raw categorical columns to the
        #       SAME integer codes training used (an independent local
        #       recast per call could assign different codes to the
        #       same category label, silently scrambling predictions -
        #       not just a crash risk).
        self._categorical_enum_dtypes = {}

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
        b_need_mm = fb.needs_model_matrix()

        if b_need_mm:
            #       Get analysis dataset (the model matrix)
            fb.remove_constant()
            df_mm = get_model_frame(
                fb.formula, nw.from_native(self.df).lazy().collect().to_native()
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

            #   Unlike the string-formula/R-model-matrix path (which
            #       auto one-hot-encodes), this list-form path does no
            #       categorical recoding of its own - lightgbm's own
            #       pyarrow ingestion rejects anything but integer/float
            #       dtypes outright, so a raw string/categorical column
            #       declared here would otherwise crash training. Recode
            #       to plain integer category codes (skipping any column
            #       that's already numeric - e.g. a pre-coded category
            #       id - so its original encoding isn't scrambled) via a
            #       FIXED pl.Enum built from this (training) data's own
            #       category values and stored on self - a later
            #       predict() call (process_predict_frame) reuses the
            #       same Enum so the same category always gets the same
            #       code, rather than each call assigning codes
            #       independently (which could silently scramble which
            #       physical group a code represents between calls).
            #       categorical_feature above still tells lightgbm to
            #       split on the resulting codes natively (by subset,
            #       not by threshold) rather than as ordered numbers.
            nw_type = NarwhalsType(self.df)
            df_pl = nw_type.to_polars()
            if isinstance(df_pl, pl.LazyFrame):
                df_pl = df_pl.collect()
            recode_cols = [
                coli
                for coli in categorical_feature
                if not df_pl.schema[coli].is_numeric()
            ]
            if recode_cols:
                for coli in recode_cols:
                    if coli not in self._categorical_enum_dtypes:
                        categories = sorted(
                            str(vali)
                            for vali in df_pl[coli].drop_nulls().unique().to_list()
                        )
                        self._categorical_enum_dtypes[coli] = pl.Enum(categories)
                df_pl = df_pl.with_columns(
                    [
                        pl.col(coli)
                        .cast(pl.String)
                        .cast(self._categorical_enum_dtypes[coli], strict=False)
                        .to_physical()
                        .alias(coli)
                        for coli in recode_cols
                    ]
                )
                self.df = nw_type.from_polars(df_pl)

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
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split

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
            #   Split on row indices rather than handing sklearn the arrow
            #   tables directly - train_test_split's array-like indexing of
            #   a raw pyarrow.Table is version-sensitive (broken on the
            #   scikit-learn<1.7 line that's still the only option on
            #   Python<3.10), while pyarrow's own .take() is stable, so use
            #   that to apply the same split instead.
            idx_train, idx_test = train_test_split(
                np.arange(x_train.shape[0]),
                test_size=self.test_size,
                random_state=int(generate_seed()),
            )
            x_train_arrow = x_train.to_arrow()
            y_train_arrow = y_train.to_arrow()
            x_train, x_test = (
                x_train_arrow.take(idx_train),
                x_train_arrow.take(idx_test),
            )
            y_train, y_test = (
                y_train_arrow.take(idx_train),
                y_train_arrow.take(idx_test),
            )
            extra_test = {}

            if self.weight != "":
                weight_test = nw.from_native(x_test).select(self.weight).to_native()
                x_test = nw.from_native(x_test).drop(self.weight).to_native()
                extra_test["weight"] = nw.from_native(weight_test).to_numpy().ravel()

            self.test_y = y_test
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
        import lightgbm as lgb

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
            message = "Must pass a 'tuner' (Tuner) instance"
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

        best_trial_params = self.tuner.run_lightgbm(
            train_data=self.train_data,
            test_data=self.test_data,
            base_params=self.parameters,
        )

        #   The full final list of lightgbm parameters
        full_params = deepcopy(self.parameters)
        full_params.update(best_trial_params)

        #   Items that arent "real" parameters, but should
        #       be set on the actual run
        drop_params = ["seed", "num_threads", "verbose"]
        for itemi in drop_params:
            if itemi in full_params.keys():
                del full_params[itemi]

        self.parameters = full_params
        return full_params

    def process_predict_frame(self, df_predict: IntoFrameT) -> IntoFrameT:
        """
        Run the same formula processing predict() applies to df_predict, and
        return the resulting processed dataframe on its own - lets a caller
        that will call predict() repeatedly against the same df_predict (e.g.
        once per quantile in a quantile-regression loop, where only the
        trained model changes between calls) do that processing once and
        reuse it via predict(df_predict_processed=...) instead of paying for
        formula parsing/model-matrix construction on every call.
        """
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

        if self.categoricals_by_name and self._categorical_enum_dtypes:
            #   temp_lgbm's own _process_formula_list() can't redo this
            #       recoding itself - self.parameters["categorical_feature"]
            #       was already consumed (converted to index form in
            #       self.categorical_feature) and deleted by THIS
            #       (trained) instance's own _prepare_params(), so
            #       temp_lgbm (built from self.parameters) never sees
            #       it. Reapply it here directly, reusing the SAME
            #       pl.Enum training fixed per column (see
            #       _process_formula_list) so df_predict's categories
            #       get the identical codes training used - not a fresh,
            #       independently-assigned set that could scramble which
            #       physical group a code represents.
            nw_type = NarwhalsType(temp_lgbm.df)
            df_pl = nw_type.to_polars()
            if isinstance(df_pl, pl.LazyFrame):
                df_pl = df_pl.collect()
            recode_cols = [
                coli
                for coli in self.categoricals_by_name
                if coli in self._categorical_enum_dtypes and coli in df_pl.columns
            ]
            if recode_cols:
                df_pl = df_pl.with_columns(
                    [
                        pl.col(coli)
                        .cast(pl.String)
                        .cast(self._categorical_enum_dtypes[coli], strict=False)
                        .to_physical()
                        .alias(coli)
                        for coli in recode_cols
                    ]
                )
                temp_lgbm.df = nw_type.from_polars(df_pl)

        return temp_lgbm.df

    def predict(
        self,
        df_predict: IntoFrameT | None = None,
        df_predict_processed: IntoFrameT | None = None,
        name: str = "___prediction",
        merged_to_input: bool = False,
    ) -> IntoFrameT:
        if df_predict is not None or df_predict_processed is not None:
            if df_predict_processed is not None and merged_to_input and df_predict is None:
                message = (
                    "predict(): merged_to_input=True needs the original df_predict "
                    "(to merge onto) - passing only df_predict_processed isn't enough."
                )
                logger.error(message)
                raise Exception(message)

            #   Predict on new data
            nw_type = NarwhalsType(
                df_predict if df_predict is not None else df_predict_processed
            )

            if df_predict_processed is not None:
                processed_df = df_predict_processed
            else:
                processed_df = self.process_predict_frame(df_predict)

            df_prediction = lazy_backend(
                nw.Series.from_numpy(
                    name=name,
                    values=self.model.predict(
                        data=(
                            nw.from_native(processed_df)
                            .select(self.train_data.get_data().schema.names)
                            .with_columns(cs.boolean().cast(nw.Int8))
                            .lazy()
                            .collect()
                            .to_arrow()
                        ),
                        # predict_disable_shape_check=True,
                    ),
                    backend="polars",
                ).to_frame(),
                nw_type,
            ).to_native()

            if merged_to_input:
                df_prediction = concat_wrapper(
                    [df_predict, df_prediction], how="horizontal"
                )

            return NarwhalsType.return_df(df_prediction, nw_type)
        else:
            #   Predict on model data
            df_prediction = lazy_backend(
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
                ).to_frame(),
                self.nw_type,
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

        return lazy_backend(nw.from_native(df_importance), self.nw_type).to_native()

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

        omp_num_threads = os.environ.get("OMP_NUM_THREADS", "")
        if omp_num_threads != "":
            try:
                cpus = int(omp_num_threads)
                params["num_threads"] = max(1, cpus)
            except:
                pass

        params["seed"] = random.randint(1, 2**32 - 1)
        params["num_iterations"] = 100
        params["verbose"] = -1

        return params


#   Tuner/Objective (imported at the top of this file, from tuning.py) are
#       intentionally still reachable as lightgbm_wrapper.Tuner/.Objective,
#       for callers that import tuning types from here rather than from
#       tuning.py directly.

