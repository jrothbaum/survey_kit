from __future__ import annotations
from typing import Optional, Callable
import os
import numpy as np
import polars as pl
import narwhals as nw
from narwhals.typing import IntoFrameT
import shutil
import random
import math
import inspect
from pathlib import Path

from ..utilities.inputs import create_folders_if_needed

from ..utilities.dataframe import (
    lazy_backend,
    safe_height,
    drop_if_exists,
    join_list,
    concat_wrapper,
    NarwhalsType,
)

from ..utilities.random import set_seed
from ..utilities.dataframe_list import DataFrameList

from ..orchestration.utilities import CallTypes, CallInputs
from ..orchestration.config import Config
from ..orchestration.from_python import FunctionFromPython
from ..orchestration.callers import run_function_list

from .utilities.lightgbm_wrapper import Survey_kit_Lightgbm as kit_lightgbm
from .utilities.convergence_diagnostics import convergence_table, convergence_long_table
from .utilities.quality_diagnostics import (
    observed_vs_imputed_long_table,
    density_long_table,
)

#   SRMI modules
from .variable import Variable
from .selection import Selection
from .implicate import Implicate

from ..serializable import Serializable
from .. import logger


class SRMI(Serializable):
    """
    Sequential Regression Multiple Imputation (SRMI) class for handling missing data imputation.

    This class manages the complete SRMI process including variable setup, model configuration,
    parallel execution, and result management across multiple implicates and iterations.

    Construction groups related settings into sub-objects - see __init__ for the full
    parameter list:

    df, variables, index, imputation_stats : flat, core settings
    replication : SRMI.Replication - n_implicates / n_iterations / seed
    parallel : SRMI.Parallel - enabled / variables_per_job / call_inputs / testing
    storage : SRMI.Storage - path_model / model_name / force_start /
        save_every_variable / save_every_iteration
    bootstrap : SRMI.Bootstrap - enabled / index / where
    defaults : SRMI.Defaults - weight / model / joint / selection / preselection /
        modeltype / parameters / ordered_categorical (fallback values applied to each
        Variable added via AddVariable() when that Variable doesn't specify its own)

    For code still using the pre-refactor flat-kwarg signature (n_implicates=...,
    bayesian_bootstrap=..., parallel_CallInputs=..., etc.), use SRMI.from_legacy(...)
    instead of SRMI(...).

    Raises
    ------
    Exception
        If replication.n_implicates < 1 or replication.n_iterations < 1
        If path equals path_model_new in load_to_continue_prior

    Examples
    --------
    Basic usage:

    >>> srmi = SRMI(
    ...     df=data,
    ...     variables=[var1, var2],
    ...     replication=SRMI.Replication(n_implicates=5, n_iterations=10),
    ...     storage=SRMI.Storage(path_model="/path/to/model"),
    ... )
    >>> srmi.run()

    With parallel execution:

    >>> srmi = SRMI(
    ...     df=data,
    ...     variables=vars_list,
    ...     replication=SRMI.Replication(n_implicates=5, n_iterations=10),
    ...     parallel=SRMI.Parallel(enabled=True, call_inputs=CallInputs(n_cpu=4, mem_in_mb=5000)),
    ... )
    """

    _save_suffix = "srmi"
    _save_exclude_items = ["implicates"]

    class Replication(Serializable):
        """How many implicates/iterations to run, and the base random seed."""

        _save_suffix = "srmi.replication"

        def __init__(self, n_implicates: int = 0, n_iterations: int = 0, seed: int = 0):
            """
            Parameters
            ----------
            n_implicates : int
                Number of separate implicates to impute.
            n_iterations : int
                Number of iterations in each implicate.
            seed : int, optional
                Random seed for replicability, by default 0 (no seed).
            """
            self.n_implicates = n_implicates
            self.n_iterations = n_iterations
            self.seed = seed

        def with_n_implicates(self, value: int) -> SRMI.Replication:
            return self._with(n_implicates=value)

        def with_n_iterations(self, value: int) -> SRMI.Replication:
            return self._with(n_iterations=value)

        def with_seed(self, value: int) -> SRMI.Replication:
            return self._with(seed=value)

    class Parallel(Serializable):
        """Parallel execution settings for running implicates."""

        _save_suffix = "srmi.parallel"

        def __init__(
            self,
            enabled: bool = True,
            variables_per_job: int = 0,
            call_inputs: CallInputs | None = None,
            testing: bool = False,
        ):
            """
            Parameters
            ----------
            enabled : bool, optional
                Run implicates in parallel, by default True.
            variables_per_job : int, optional
                Number of variables per parallel job (for memory management and to
                deal with memory leaks, if there are any), by default 0.
            call_inputs : CallInputs | None, optional
                Parameters for parallel execution such as memory and CPU allocation,
                by default None (auto-sized from Config().cpus and n_implicates).
            testing : bool, optional
                Test parallel jobs without running them, by default False.
            """
            self.enabled = enabled
            self.variables_per_job = variables_per_job
            self.call_inputs = call_inputs
            self.testing = testing

        def with_enabled(self, value: bool) -> SRMI.Parallel:
            return self._with(enabled=value)

        def with_variables_per_job(self, value: int) -> SRMI.Parallel:
            return self._with(variables_per_job=value)

        def with_call_inputs(self, value: CallInputs | None) -> SRMI.Parallel:
            return self._with(call_inputs=value)

        def with_testing(self, value: bool) -> SRMI.Parallel:
            return self._with(testing=value)

    class Storage(Serializable):
        """Where/how the imputation model's data and progress are saved."""

        _save_suffix = "srmi.storage"

        #   force_start is a one-time "wipe and restart" instruction for the
        #   construction call it's passed to - it must never be persisted, or
        #   replaying it via SRMI.load() (which reconstructs SRMI/Storage through
        #   this same __init__) would delete the very directory being loaded.
        _save_exclude_items = ["force_start"]

        def __init__(
            self,
            path_model: str = "",
            model_name: str = "",
            force_start: bool = False,
            save_every_variable: bool = False,
            save_every_iteration: bool = True,
        ):
            """
            Parameters
            ----------
            path_model : str, optional
                Directory to save model data and temporary files.
            model_name : str, optional
                Model name for continuing existing runs, by default "".
            force_start : bool, optional
                Restart imputation even if existing run exists, by default False.
                Not persisted - see _save_exclude_items above.
            save_every_variable : bool, optional
                Save data after each variable is imputed, by default False.
            save_every_iteration : bool, optional
                Save data after each iteration completes, by default True.
            """
            self.path_model = path_model
            self.model_name = model_name
            self.force_start = force_start
            self.save_every_variable = save_every_variable
            self.save_every_iteration = save_every_iteration

        def with_path_model(self, value: str) -> SRMI.Storage:
            return self._with(path_model=value)

        def with_model_name(self, value: str) -> SRMI.Storage:
            return self._with(model_name=value)

        def with_force_start(self, value: bool) -> SRMI.Storage:
            return self._with(force_start=value)

        def with_save_every_variable(self, value: bool) -> SRMI.Storage:
            return self._with(save_every_variable=value)

        def with_save_every_iteration(self, value: bool) -> SRMI.Storage:
            return self._with(save_every_iteration=value)

    class Bootstrap(Serializable):
        """Bayesian Bootstrap settings for accounting for coefficient uncertainty."""

        _save_suffix = "srmi.bootstrap"

        def __init__(
            self,
            enabled: bool = True,
            index: list[str] | None = None,
            where: str = "",
        ):
            """
            Parameters
            ----------
            enabled : bool, optional
                Use Bayesian Bootstrap to account for uncertainty in coefficients,
                by default True.
            index : list, optional
                Index variables for resampling (i.e. if you want to resample by
                household, not person), by default None.
            where : str, optional
                SQL condition for keeping observations when resampling, by default "".
            """
            self.enabled = enabled
            self.index = index
            self.where = where

        def with_enabled(self, value: bool) -> SRMI.Bootstrap:
            return self._with(enabled=value)

        def with_index(self, value: list[str] | None) -> SRMI.Bootstrap:
            return self._with(index=value)

        def with_where(self, value: str) -> SRMI.Bootstrap:
            return self._with(where=value)

    class Defaults(Serializable):
        """
        Fallback settings applied to each Variable added via AddVariable() when
        that Variable doesn't specify its own.
        """

        _save_suffix = "srmi.defaults"

        def __init__(
            self,
            weight: str = "",
            model: str | list = "",
            joint: dict = None,
            selection: Selection = None,
            preselection: Selection = None,
            modeltype: Variable.ModelType = None,
            parameters: dict = None,
            ordered_categorical: list[str] = None,
        ):
            """
            Parameters
            ----------
            weight : str, optional
                Weight variable name for imputation modeling, by default "".
            model : str | list, optional
                R string formula that is the default for the imputation.
            joint : dict, optional
                Key-value pairs of variables to be included together (i.e. if one is
                selected in the variable selection step, the other is too), by default None.
            selection : Selection, optional
                Variable selection method used within the imputation (if any), by default None.
            preselection : Selection, optional
                Variable selection done before SRMI starts to pre-prune inputs, by default None.
            modeltype : Variable.ModelType, optional
                Imputation model type from the ModelType enumeration, by default None.
            parameters : dict, optional
                Model parameters dictionary, by default None.
            ordered_categorical : list, optional
                List of categorical variables in model that are ordered, by default None
                    An example would be education (vs. a variable with no ordering like
                    state or county code).
            """
            self.weight = weight
            self.model = model
            self.joint = joint

            if selection is None:
                selection = Selection(method=Selection.Method.No)
            self.selection = selection

            if preselection is None:
                preselection = Selection(method=Selection.Method.No)
            self.preselection = preselection

            self.modeltype = modeltype

            if parameters is None:
                parameters = {}
            self.parameters = parameters

            self.ordered_categorical = ordered_categorical

        def with_weight(self, value: str) -> SRMI.Defaults:
            return self._with(weight=value)

        def with_model(self, value: str | list) -> SRMI.Defaults:
            return self._with(model=value)

        def with_joint(self, value: dict) -> SRMI.Defaults:
            return self._with(joint=value)

        def with_selection(self, value: Selection | None) -> SRMI.Defaults:
            return self._with(
                selection=value if value is not None else Selection(method=Selection.Method.No)
            )

        def with_preselection(self, value: Selection | None) -> SRMI.Defaults:
            return self._with(
                preselection=value
                if value is not None
                else Selection(method=Selection.Method.No)
            )

        def with_modeltype(self, value: Variable.ModelType) -> SRMI.Defaults:
            return self._with(modeltype=value)

        def with_parameters(self, value: dict) -> SRMI.Defaults:
            return self._with(parameters=value)

        def with_ordered_categorical(self, value: list[str]) -> SRMI.Defaults:
            return self._with(ordered_categorical=value)

    def __init__(
        self,
        df: IntoFrameT | None = None,
        variables: list[Variable] = None,
        index: list[str] = None,
        imputation_stats: list[str] | None = None,
        replication: SRMI.Replication = None,
        parallel: SRMI.Parallel = None,
        storage: SRMI.Storage = None,
        bootstrap: SRMI.Bootstrap = None,
        defaults: SRMI.Defaults = None,
    ):
        self.replication = replication if replication is not None else SRMI.Replication()
        self.parallel = parallel if parallel is not None else SRMI.Parallel()
        self.storage = storage if storage is not None else SRMI.Storage()
        self.bootstrap = bootstrap if bootstrap is not None else SRMI.Bootstrap()
        self.defaults = defaults if defaults is not None else SRMI.Defaults()

        #   Error checking
        if self.replication.n_implicates < 1:
            message = (
                f"Must pass at least 1 implicate (passed {self.replication.n_implicates})"
            )
            logger.error(message)
            raise Exception(message)

        if self.replication.n_iterations < 1:
            message = (
                f"Must pass at least 1 iteration (passed {self.replication.n_iterations})"
            )
            logger.error(message)
            raise Exception(message)

        if index is None:
            index = []

        self.df = df
        if df is not None:
            self.nw_type = NarwhalsType(df)
        else:
            self.nw_type = None

        if self.replication.seed > 0:
            set_seed(self.replication.seed)

        # Add an index, if there isn't one
        #   We need one to be able to put the file back together again
        if type(index) is str:
            index = [index]

        if len(index) == 0:
            self.index = ["___rownumber"]

            self.df = lazy_backend(
                nw.from_native(self.df)
                .lazy()
                .collect()
                .with_row_index(name=self.index[0]),
                self.nw_type,
            ).to_native()
        else:
            #   Needs to be unique
            if safe_height(
                nw.from_native(self.df).select(index).unique().to_native()
            ) != safe_height(self.df):
                logger.info("Adding row number to the index as it is not unique")
                self.df = self.df.with_row_index(name="___rownumber")
                index.append("___rownumber")

            self.index = index

        if self.parallel.enabled and self.parallel.call_inputs is None:
            n_available_cpus = Config().cpus
            n_parallel_cpus = max(
                int(n_available_cpus / self.replication.n_implicates), 1
            )

            self.parallel.call_inputs = CallInputs(
                call_type=CallTypes.shell,
                n_cpu=n_parallel_cpus,
                process_limit=min(n_available_cpus, self.replication.n_implicates),
            )

        self.imputation_stats = imputation_stats

        self.setup_complete = False

        #   No model path, use a temporary one (with temp, can't really continue)
        if self.storage.path_model == "":
            if Config().path_temp_files == "":
                message = "You must pass in a path to save the imputation files to (storage.path_model)"
                logger.error(message)
                raise Exception(message)
            else:
                self.storage.path_model = Config().path_temp_with_random()

        #   For safety against accidental deletes, file path has .srmi suffix
        if not self.storage.path_model.endswith(".srmi"):
            self.storage.path_model = self.storage.path_model + ".srmi"

        #   Force start? - then delete any saved data
        if os.path.isdir(self.storage.path_model) and self.storage.force_start:
            logger.info(f"Removing existing directory {self.storage.path_model}")
            shutil.rmtree(self.storage.path_model)

        self.variables = []
        if variables is not None:
            for vari in variables:
                self.AddVariable(vari)

        self.implicates = []

        #   Defaults to false
        self.is_continuing_srmi = False
        self.continuing_cols = []

    @classmethod
    def from_legacy(
        cls,
        df: IntoFrameT | None = None,
        variables: list[Variable] = None,
        model: str | list = "",
        selection: Selection = None,
        preselection: Selection = None,
        modeltype: Variable.ModelType = None,
        parameters: dict = None,
        joint: dict = None,
        ordered_categorical: list[str] = None,
        seed: int = 0,
        weight: str = "",
        n_implicates: int = 0,
        n_iterations: int = 0,
        bayesian_bootstrap: bool = True,
        bootstrap_index: list[str] = None,
        bootstrap_where: str = "",
        index: list[str] = None,
        parallel: bool = True,
        parallel_variables_per_job: int = 0,
        parallel_CallInputs: CallInputs | None = None,
        parallel_testing: bool = False,
        path_model: str = "",
        model_name: str = "",
        force_start: bool = False,
        save_every_variable: bool = False,
        save_every_iteration: bool = True,
        imputation_stats: list[str] | None = None,
    ) -> SRMI:
        """
        Construct an SRMI from the pre-refactor flat-kwarg signature.

        Migration aid only - new code should pass replication=SRMI.Replication(...),
        parallel=SRMI.Parallel(...), storage=SRMI.Storage(...), bootstrap=SRMI.Bootstrap(...),
        defaults=SRMI.Defaults(...) directly to SRMI() instead.
        """
        return cls(
            df=df,
            variables=variables,
            index=index,
            imputation_stats=imputation_stats,
            replication=cls.Replication(
                n_implicates=n_implicates, n_iterations=n_iterations, seed=seed
            ),
            parallel=cls.Parallel(
                enabled=parallel,
                variables_per_job=parallel_variables_per_job,
                call_inputs=parallel_CallInputs,
                testing=parallel_testing,
            ),
            storage=cls.Storage(
                path_model=path_model,
                model_name=model_name,
                force_start=force_start,
                save_every_variable=save_every_variable,
                save_every_iteration=save_every_iteration,
            ),
            bootstrap=cls.Bootstrap(
                enabled=bayesian_bootstrap, index=bootstrap_index, where=bootstrap_where
            ),
            defaults=cls.Defaults(
                weight=weight,
                model=model,
                joint=joint,
                selection=selection,
                preselection=preselection,
                modeltype=modeltype,
                parameters=parameters,
                ordered_categorical=ordered_categorical,
            ),
        )

    def AddVariable(self, variable: Variable) -> None:
        """
        Add a variable to the imputation model.

        This method validates the variable and applies default parameters
        from the SRMI instance if not specified in the variable.

        Parameters
        ----------
        variable : Variable
            The Variable instance to be added to the imputation sequence

        Notes
        -----
        - Applies SRMI-level defaults for weight, model, joint, selection, etc.
        - Validates variable inputs and excludes the variable from its own
            (i.e. don't regress x on x)
        - Variables are processed in the order they are added
        """

        str_override = ["weight", "model"]
        none_override = [
            "joint",
            "selection",
            "preselection",
            "modeltype",
            "parameters",
        ]

        for stri in str_override:
            if getattr(variable, stri) == "":
                setattr(variable, stri, getattr(self.defaults, stri))

        for obji in none_override:
            if getattr(variable, obji) is None:
                setattr(variable, obji, getattr(self.defaults, obji))

        if len(variable.parameters) == 0:
            variable.parameters = self.defaults.parameters

        #   Remove the variable itself from its own model
        #       and any variables in variable.predictors_exclude
        variable.exclude_variables_from_models(df=self.df)

        #   Do some pre-checks to catch any errors that will stop things later
        variable.validate_inputs(df=self.df, bootstrap_enabled=self.bootstrap.enabled)

        self.variables.append(variable)

    def run(self) -> None:
        """
        Execute the SRMI imputation process.

        Orchestrates the complete imputation workflow including initialization,
        preprocessing, and execution in parallel or sequential mode.

        Notes
        -----
        The method performs these steps:
        1. Creates folders and initializes implicates to be run
        2. Preprocesses data (variable selection, hyperparameter tuning)
        3. Runs imputation in parallel or sequential mode
        4. Saves results and statistics

        For parallel execution, creates job files for each iteration and variable subset.
        For sequential execution, runs each implicate directly.
        """

        #   Create folder if needed
        create_folders_if_needed(self.storage.path_model, quietly=True)

        #   Create/Load the implicates
        self._initialize_implicates()

        if not self.setup_complete:
            #   Save the initial input data

            #   Do we need to create any missing flags for bimpute_if_missing
            missing_cols = []

            var_index = 0
            for vari in self.variables:
                var_index += 1

                impute_flag = f"___imp_missing_{vari.impute_var}_{var_index}"
                vari.imputation_flag = impute_flag
                if vari.bimpute_if_missing:
                    missing_expr = (
                        nw.col(vari.impute_var)
                        .is_null()
                        .cast(nw.Boolean)
                        .alias(impute_flag)
                    )

                    missing_cols.append(missing_expr)

                    vari.where_impute_add_flag(impute_flag)

            if len(missing_cols) > 0:
                self.df = nw.from_native(self.df).with_columns(missing_cols).to_native()
            self._preprocess()

            #   Done, save the srmi information
            self.setup_complete = True
            self.save()

        if self.parallel.enabled:
            #   Set up jobs to run the implicates in parallel
            f_implicates = []
            for impi in self.implicates:
                for iterationi in range(1, self.replication.n_iterations + 1):
                    if self.parallel.variables_per_job > 0:
                        n_jobs_per_loop = math.ceil(
                            len(self.variables) / self.parallel.variables_per_job
                        )
                    else:
                        n_jobs_per_loop = 1

                    for sub_job in range(0, n_jobs_per_loop):
                        if sub_job == 0:
                            prior_sub = n_jobs_per_loop - 1
                            prior_iteration = iterationi - 1
                        else:
                            prior_sub = sub_job - 1
                            prior_iteration = iterationi

                        if self.parallel.variables_per_job == 0:
                            variable_start = 0
                            variable_end = 0
                        else:
                            variable_start = (
                                sub_job * self.parallel.variables_per_job + 1
                            )
                            variable_end = (
                                variable_start + self.parallel.variables_per_job - 1
                            )

                        #   Dummy inputs and outputs to order the iteration runs properly
                        if sub_job > 0 or iterationi > 1:
                            inputs = [
                                f"{self.storage.path_model}/logs/iteration_{impi}_{prior_iteration}_{prior_sub}.log"
                            ]
                        else:
                            inputs = []
                        outputs = [
                            f"{self.storage.path_model}/logs/iteration_{impi}_{iterationi}_{sub_job}.log"
                        ]

                        f_implicates.append(
                            FunctionFromPython(
                                function=run_implicate_async,
                                parameters={
                                    "path_model": Path(self.storage.path_model).as_posix(),
                                    "implicate": impi.number,
                                    "iteration": iterationi,
                                    "variable_start": variable_start,
                                    "variable_end": variable_end,
                                },
                                inputs=inputs,
                                outputs=outputs,
                            )
                        )

            log = run_function_list(
                function_list=f_implicates,
                call_input=self.parallel.call_inputs,
                run_all=True,
                testing=self.parallel.testing,
            )

        else:
            #   Just run it
            for impi in self.implicates:
                impi.run()

            if self.complete:
                drop_flags = []

                var_index = 0
                for vari in self.variables:
                    var_index += 1
                    if vari.bimpute_if_missing:
                        drop_flags.append(
                            f"___imp_missing_{vari.impute_var}_{var_index}"
                        )

                if len(drop_flags) > 0:
                    self.df = nw.from_native(self.df).drop(drop_flags).to_native()

    def _initialize_implicates(self) -> None:
        if self.is_continuing_srmi:
            #   Less processing for continuing srmi
            keep_vars = self.vars_implicate + self.continuing_cols

            for impi in self.implicates:
                impi.df = nw.from_native(impi.df).select(keep_vars).to_native()
                impi.seed = random.randint(1, 2**32 - 1)

                #   Reset progress
                impi.status_iteration = 0
                impi.status_variable = 0
                impi.status_iteration_complete = False
                impi.complete = False
                impi.in_progress = False
                impi.df_summary_stats = {}

                impi.save()
        else:
            keep_vars = self.vars_implicate

            #   implicate dataframe has ONLY the variables to be imputed
            #       and the index for merging
            df_initial = nw.from_native(self.df).select(keep_vars).to_native()

            #   run() calls _initialize_implicates() unconditionally,
            #       every time it's called - including a second call on
            #       an already-loaded SRMI (e.g. SRMI.load(path) then
            #       .run() again to extend to a higher n_iterations,
            #       after checking SRMI.convergence()). self.implicates
            #       is already fully populated in that case (by
            #       SRMI.load()'s own explicit per-implicate loading
            #       loop), so appending unconditionally here would
            #       silently duplicate every implicate - confirmed:
            #       len(self.implicates) doubles, the run loop (indexed
            #       by number, not iterating the list) only ever touches
            #       the ORIGINAL entries so the duplicates just sit
            #       there stale, but anything that iterates
            #       self.implicates directly (e.g. SRMI.convergence()'s
            #       own m = len(self.implicates)) silently corrupts on
            #       the phantom extras. Skip any number already present.
            existing_numbers = {impi.number for impi in self.implicates}
            for impi in range(self.replication.n_implicates):
                number = impi + 1
                if number in existing_numbers:
                    continue

                this_implicate = Implicate(
                    parent=self, number=number, seed=random.randint(1, 2**32 - 1)
                )

                if not this_implicate.in_progress:
                    this_implicate.df = df_initial

                    this_implicate.save()

                self.implicates.append(this_implicate)

    def _preprocess(self) -> None:
        #   Check for two-sample variables
        # for vari in self.variables:
        # if vari.modeltype == Variable.ModelType.TwoSampleRegression:
        #     #   Is there any data?
        #     vari.parameters["any_values"] = (
        #         nw.from_native(self.df)
        #         .select(nw.col(vari.impute_var).is_not_missing().cast(nw.Int64).sum())
        #         .item(0,0)
        #      ) > 0

        #     #   No selection if there are no values (this is the receiver of the imputes)
        #     if not vari.parameters["any_values"] or vari.parameters["load_from_save"]:
        #         vari.selection.method = Selection.Method.No
        #         vari.preselection.method = Selection.Method.No

        #   Pre-select variables for each model
        logger.info("Variable selection before SRMI run, if necessary")
        for vari in self.variables:
            self._preprocess_selection(vari)

        logger.info("Hyperparameter tuning before SRMI run, if necessary")
        for vari in self.variables:
            self._preprocess_tune(vari)

    def _preprocess_selection(self, variable: Variable):
        selection_method = variable.preselection.method

        logger.info(f"     {variable.impute_var}: {selection_method}")

        [fb, _, _] = variable.process_model(df=self.df, NoConstant=True)

        if variable.selection is not None:
            if variable.preselection.method == Selection.Method.LASSO:
                prior_optimal_lambda = variable.preselection.parameters[
                    "optimal_lambda"
                ] = None

        selected_model = variable.preselection.run(
            df=self.df,
            y=variable.impute_var,
            formula=fb.formula,
            weight=variable.weight,
        )
        if selected_model != "":
            variable.model = variable.union_required_predictors(selected_model)

            if variable.selection is not None:
                if variable.preselection.method == Selection.Method.LASSO:
                    if variable.selection.method == Selection.Method.LASSO:
                        if variable.preselection.parameters["optimal_lambda_from_pre"]:
                            variable.selection.parameters["optimal_lambda"] = (
                                variable.preselection.parameters["optimal_lambda"]
                            )

                    #   Reset the optimal lambda
                    variable.preselection.parameters["optimal_lambda"] = (
                        prior_optimal_lambda
                    )

    def _preprocess_tune(self, variable: Variable):
        #   Tunable models
        if variable.modeltype == Variable.ModelType.LightGBM:
            tune = variable.parameters["tune"]
            tune_overwrite = variable.parameters["tune_overwrite"]
            tune_hyperparameter_path = variable.parameters["tune_hyperparameter_path"]
            tuner = variable.parameters["tuner"]

            if tune_hyperparameter_path != "":
                tuner.path_save = (
                    f"{tune_hyperparameter_path}/{variable.impute_var}.pickle"
                )
            parameters = variable.parameters["parameters"]

            df_tune = (
                nw.from_native(self.df)
                .filter(~nw.col(variable.impute_var).is_null())
                .to_native()
            )
            df_tune = variable.df_where(df_tune)

            lgbm = kit_lightgbm(
                df=df_tune,
                y=variable.impute_var,
                formula=variable.model,
                weight=self.defaults.weight,
                parameters=parameters,
                tuner=tuner,
            )

            if lgbm.tuner is not None:
                if lgbm.tuner.path_save != "" and not (tune_overwrite and tune):
                    #   Load the tuned parameters (returns True if loaded, False if not)
                    if lgbm.load_tuned_parameters(error_on_missing=False):
                        tune = False
                if tune:
                    #   Run the tuning
                    #       This will save the file (if path set)
                    #       and update the parameters in the lgbm object
                    lgbm.tune()
                    logger.info("TUNING COMPLETE")

    @classmethod
    def load_to_continue_prior(
        cls,
        path: str,
        path_model_new: str,
        path_append: list[str] | None,
        append_condition: nw.Expr | None,
        append_to_index: list[str] | str | None,
        seed: int = 0,
        pipe=None,
        pipe_kwargs=None,
    ) -> SRMI:
        """
        Load a previous SRMI run and prepare it for continuation with new data.

        Supppose you have run an imputation model, but now you want to impute
            another set of variables.  This will load the model so you can
            do so.

        Parameters
        ----------
        path : str
            Path to the existing SRMI model to load
        path_model_new : str
            Path for the new continued model (must differ from path)
                This will save the new imputation model separately from the
                old and there is a function to load them both together
        path_append : list[str] | None
            List of paths to additional SRMI models to append
                This lets you run any number of downstream models
                and combine them into one "srmi" set of implicates
        append_condition : nw.Expr | None
            Condition for including appended data
                I.e., maybe one model was run on the cps in 2023
                    but another predicted SSI state payments in 2023
                    using data from 2017-2022.  This limits which
                    rows get merged so that only the 2023 data from one
                    merges to the 2023 data from the other
        append_to_index : list[str] | str | None
            Additional index columns to add
        seed : int, optional
            Random seed for the continued run, by default 0
        pipe : callable, optional
            Function to apply to data during loading, by default None
        pipe_kwargs : dict, optional
            Keyword arguments for pipe function, by default None

        Returns
        -------
        SRMI
            New SRMI instance ready for continuation

        Raises
        ------
        Exception
            If path equals path_model_new (would overwrite original)
        """

        if not path_model_new.endswith(".srmi"):
            path_model_new = path_model_new + ".srmi"

        if path == path_model_new:
            message = (
                "To avoid overwriting the prior SRMI, path cannot equal path_model_new"
            )
            logger.error(message)
            raise Exception(message)
        srmi_prior = SRMI.load(path)

        srmi_prior.replication.seed = seed

        if srmi_prior.replication.seed > 0:
            set_seed(srmi_prior.replication.seed)

        if path_append is not None:
            srmis_to_append = []
            for pathi in path_append:
                srmi_other = SRMI.load(pathi)

                do_append = True
                if append_condition is not None:
                    do_append = (
                        nw.from_native(srmi_other.df)
                        .select(append_condition)
                        .lazy()
                        .collect()
                        .item(0, 0)
                    )
                if do_append:
                    srmis_to_append.append(srmi_other)
            if len(srmis_to_append):
                for i in range(0, srmi_prior.replication.n_implicates + 1):
                    df_concat = [srmi_prior.df_containers[i].df] + [
                        srmi_other.df_containers[i].df for srmi_other in srmis_to_append
                    ]

                    df = concat_wrapper(df_concat, how="diagonal")

                    if pipe is not None:
                        if pipe_kwargs is None:
                            pipe_kwargs = {}

                        extra_args = {}
                        if _function_has_argument(pipe, "srmi_df"):
                            extra_args["srmi_df"] = i == 0
                        df = df.pipe(pipe, **extra_args, **pipe_kwargs)
                    srmi_prior.df_containers[i].df = df

        if append_to_index is not None:
            if isinstance(append_to_index, str):
                append_to_index = [append_to_index]
            srmi_prior.index.extend(append_to_index)

        srmi_prior.is_continuing_srmi = True
        srmi_prior.setup_complete = False
        srmi_prior.continuing_cols = srmi_prior.vars_imputed.copy()
        srmi_prior.storage.path_model = path_model_new

        srmi_prior.variables = []
        return srmi_prior

    class SRMIContinueLoad:
        def __init__(
            self, path: str, varlist: list[str], fill_null: dict | object | None = None
        ):
            self.path = path
            self.varlist = varlist
            self.fill_null = fill_null

    @classmethod
    def load_with_continued(
        cls,
        path: str,
        srmi_continue: list[SRMIContinueLoad],
        filter_cond: nw.Expr | None,
    ) -> SRMI:
        """
        Load SRMI with additional continued imputation results.
            This combines the results of multiple imputation models
            into one set of implicates to work with

        Parameters
        ----------
        path : str
            Path to the main SRMI model
        srmi_continue : list[SRMIContinueLoad]
            List of continuation data specifications
        filter_cond : nw.Expr | None
            Filter condition for continued data

        Returns
        -------
        SRMI
            SRMI instance with continued data merged in
        """
        srmi = cls.load(path)

        for srmi_continue_i in srmi_continue:
            srmi_i = cls.load(srmi_continue_i.path)

            #   Replace/append the value of varlist to the original SRMI
            for i in range(0, srmi.replication.n_implicates):
                dfi = srmi_i.implicates[i].df
                if filter_cond is not None:
                    dfi = nw.from_native(dfi).filter(filter_cond).to_native()
                dfi = (
                    nw.from_native(dfi)
                    .select(srmi.index + srmi_continue_i.varlist)
                    .to_native()
                )
                srmi.implicates[i].df = join_list(
                    [
                        drop_if_exists(
                            df=srmi.implicates[i].df, columns=srmi_continue_i.varlist
                        ),
                        dfi,
                    ],
                    on=srmi.index,
                    how="left",
                )

                if srmi_continue_i.fill_null is not None:
                    if type(srmi_continue_i.fill_null) is dict:
                        with_null_fill = []
                        for vari, valuei in srmi_continue_i.fill_null.items():
                            with_null_fill.append(nw.col(vari).fill_null(valuei))
                    else:
                        with_null_fill = [
                            nw.col(srmi_continue_i.varlist).fill_null(
                                srmi_continue_i.fill_null
                            )
                        ]

                    if len(with_null_fill):
                        srmi.implicates[i].df = (
                            nw.from_native(srmi.implicates[i].df)
                            .with_columns(with_null_fill)
                            .to_native()
                        )

        return srmi

    @property
    def in_progress(self) -> bool:
        #   Any in progress?
        for impi in self.implicates:
            if impi.in_progress:
                return True

        #   No
        return False

    @property
    def complete(self) -> bool:
        """
        Check if all implicates have completed imputation.

        Returns
        -------
        bool
            True if all implicates are complete, False otherwise
        """
        all_completed = True
        #   Any in progress?
        for impi in self.implicates:
            all_completed = all_completed and impi.complete

        return all_completed

    @property
    def vars_imputed(self) -> list[str]:
        """
        Get list of all variables that will be imputed.

        Returns
        -------
        list[str]
            Variable names including donated variables
        """
        keep_vars = []
        for vari in self.variables:
            keep_vars.append(vari.impute_var)

            if "donate_list" in vari.parameters.keys():
                if len(vari.parameters["donate_list"]):
                    keep_vars.extend(vari.parameters["donate_list"])

        #   remove duplicates
        keep_vars = list(set(keep_vars))
        return keep_vars

    @property
    def vars_imputed_ordered(self) -> list[str]:
        keep_vars = {}

        var_index = 0
        for vari in self.variables:
            var_index += 1
            if vari not in keep_vars:
                keep_vars[vari.impute_var] = var_index

                if "donate_list" in vari.parameters.keys():
                    if len(vari.parameters["donate_list"]):
                        for donatei in vari.parameters["donate_list"]:
                            if donatei not in keep_vars:
                                keep_vars[donatei] = var_index

        return list(keep_vars.keys())

    @property
    def vars_implicate(self) -> list[str]:
        keep_vars = self.vars_imputed
        keep_vars.extend(self.index)

        #   remove duplicates
        keep_vars = list(set(keep_vars))
        return keep_vars

    @property
    def df_implicates(self) -> DataFrameList:
        """
        Get all completed implicates as a DataFrameList.

        Returns
        -------
        DataFrameList
            List containing the full dataframe for each implicate
        """
        df_out = []
        for impi in self.implicates:
            df_out.append(impi.df_full(drop_flags=True))

        return DataFrameList(df_out)

    @property
    def df_implicates_with_appended_cols(self) -> DataFrameList:
        """
        Get all completed implicates as a DataFrameList with any downstream
            data appended to them.  We use this for getting
            post-imputation variables, such as NEWS's final household
            income estimates that are calculated separately for
            each implicate.

        Returns
        -------
        DataFrameList
            List containing the full dataframe for each implicate
        """
        df_out = []
        for impi in self.implicates:
            df_out.append(impi.df_full(drop_flags=True, with_appended_cols=True))

        return DataFrameList(df_out)

    def df_implicates_by_index(
        self, index: int, drop_flags: bool = False, with_appended_cols: bool = False
    ) -> DataFrameList:
        return self.implicates[index].df_full(
            drop_flags=drop_flags, with_appended_cols=with_appended_cols
        )

    def convergence(self, diagnostic: str = "all", parameter: str = "mean") -> IntoFrameT:
        """
        Convergence diagnostics for the imputed variables, across
        implicates and iterations - matches mice's own convergence()
        function (R/convergence.R) as closely as possible, including
        the exact algorithm it delegates to for the potential scale
        reduction factor (rstan::Rhat(), i.e. the rank-normalized,
        folded, split-Rhat of Vehtari et al. 2021 - see
        utilities/convergence_diagnostics.py's module docstring for the
        verified-against-source details).

        Callable any time after at least 3 iterations of at least 2
        implicates have run (mid-run is fine, not just after a
        completed SRMI) - it reads whatever each Impute.chain_mean/
        chain_std has already accumulated on self.implicates, the same
        mean/std of each variable's own newly-imputed values every
        iteration that Impute._post_impute_statistics computes (and
        that already respects the Variable's own Where restriction,
        since it's read from df_impute, which Impute.df_impute() has
        already filtered by Where before this ever sees it).

        Parameters
        ----------
        diagnostic : str, optional
            "all" (both ac and psrf), "ac" (lag-1 autocorrelation only),
            or "psrf"/"gr" (potential scale reduction factor only). By
            default "all".
        parameter : str, optional
            "mean" or "sd" - diagnose the chain means or the chain
            standard deviations. By default "mean".

        Returns
        -------
        IntoFrameT
            One row per (iteration, variable) - columns ".it", "vrb",
            and "ac"/"psrf" per `diagnostic`, same names mice uses. NaN
            wherever a variable has no numeric/boolean imputed values to
            track for that (iteration, implicate) - e.g. a Multinomial()/
            OrderedCategorical() category-label target, or an iteration
            where that variable had nothing to impute.
        """
        chain_mean, chain_std = self._collect_chain_arrays()

        return convergence_table(
            chain_mean=chain_mean,
            chain_std=chain_std,
            diagnostic=diagnostic,
            parameter=parameter,
        )

    def _collect_chain_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Shared data-gathering for convergence()/plot_convergence(): read
        each implicate's Impute.chain_mean/chain_std accumulation into
        {variable: (n_iterations, n_implicates) numpy array} pairs, with
        mice's own m>=2/iterations>=3 validation (convergence()
        docstring has the full explanation of what these values are and
        where they come from).
        """
        m = len(self.implicates)
        if m < 2:
            message = (
                f"SRMI.convergence()/plot_convergence(): the number of "
                f"implicates should be at least two (m > 1), got {m}."
            )
            logger.error(message)
            raise ValueError(message)

        #   Implicate.chain_mean/chain_std are {impute_var: {iteration:
        #       value}} - the inner "iteration" keys are plain Python
        #       ints on a freshly-run (in-memory) implicate, but come
        #       back as STRINGS after a save()/load() round trip (JSON
        #       object keys are always strings) - int(...) here makes
        #       this work identically either way, which matters for
        #       exactly the "reload and run more iterations" workflow
        #       convergence()/plot_convergence() exist to support.
        max_it = 0
        for impi in self.implicates:
            for chain_mean_var in impi.chain_mean.values():
                if chain_mean_var:
                    max_it = max(max_it, max(int(it) for it in chain_mean_var.keys()))
        if max_it < 3:
            message = (
                f"SRMI.convergence()/plot_convergence(): the number of "
                f"iterations should be at least three (maxit > 2), got "
                f"{max_it}."
            )
            logger.error(message)
            raise ValueError(message)

        vrbs = []
        for vari in self.variables:
            if vari.impute_var not in vrbs:
                vrbs.append(vari.impute_var)

        chain_mean = {v: np.full((max_it, m), np.nan) for v in vrbs}
        chain_std = {v: np.full((max_it, m), np.nan) for v in vrbs}

        for chain_idx, impi in enumerate(self.implicates):
            for v in vrbs:
                for it_raw, val in impi.chain_mean.get(v, {}).items():
                    it = int(it_raw)
                    if it <= max_it and val is not None:
                        chain_mean[v][it - 1, chain_idx] = val
                for it_raw, val in impi.chain_std.get(v, {}).items():
                    it = int(it_raw)
                    if it <= max_it and val is not None:
                        chain_std[v][it - 1, chain_idx] = val

        return chain_mean, chain_std

    def plot_convergence(
        self, parameter: str = "both", path: str | None = None
    ) -> "plotly.graph_objects.Figure":  # noqa: F821
        """
        Plot the trace lines of the SRMI algorithm - matches mice's own
        plot(imp) (plot.mids(), R/mids.R) as closely as possible: for
        each imputed variable, one line per implicate, the chain mean
        (and/or chain sd) against iteration number. On convergence, the
        lines within a panel should intermingle and be free of any
        trend - the same "worm plot" reading as mice's own.

        Purely a plotting convenience on top of the same data
        convergence() reads (Impute.chain_mean/chain_std, accumulated
        on self.implicates) - there's no "auto-plot" flag anywhere in
        SRMI's own run() - call this whenever you want, including on an
        SRMI you've just SRMI.load()ed in a completely different
        process/environment from the one that ran the imputation (e.g.
        one where plotly wasn't installed at run time, but is now).

        Requires the optional 'plotly' package - not a survey_kit
        dependency at all (imputation itself never needs it), so
        nothing about running SRMI requires having it installed; only
        calling this specific method does.

        Parameters
        ----------
        parameter : str, optional
            "both" (mice's own default - one row of panels for chain
            means, one for chain sds), "mean", or "sd". By default
            "both".
        path : str | None, optional
            If given, also save the figure there as a self-contained
            HTML file (fig.write_html) - no extra dependency beyond
            plotly itself needed for that, unlike a static image export
            (which would need kaleido too). By default None (don't
            save - just return the Figure; display it yourself, e.g.
            fig.show() in a script or automatically in a notebook).

        Returns
        -------
        plotly.graph_objects.Figure
            Always returned (even when `path` is also given) so you can
            further customize it, .show() it, or save it yourself in a
            different format.
        """
        try:
            import plotly.express as px
        except ImportError as e:
            message = (
                "SRMI.plot_convergence() needs the 'plotly' package, "
                "which isn't installed - it's optional (nothing about "
                "running SRMI itself needs it, only this diagnostic "
                "plot). Install it with `uv add --dev plotly` (or `pip "
                "install plotly`), then call this again - the "
                "underlying chain_mean/chain_std data is already saved "
                "with the SRMI, so this works just as well on an "
                "SRMI.load()ed object in a totally separate "
                "process/environment from the one that ran the "
                "imputation, any time after."
            )
            logger.error(message)
            raise ImportError(message) from e

        if parameter not in ("both", "mean", "sd"):
            message = (
                f"SRMI.plot_convergence(): parameter={parameter!r} not "
                f"recognized - use 'both', 'mean', or 'sd'."
            )
            logger.error(message)
            raise ValueError(message)

        chain_mean, chain_std = self._collect_chain_arrays()
        df_long = convergence_long_table(chain_mean=chain_mean, chain_std=chain_std)

        nw_df = nw.from_native(df_long)
        if parameter != "both":
            nw_df = nw_df.filter(nw.col("parameter") == parameter)
        df_pd = (
            nw_df.with_columns(nw.col("implicate").cast(nw.String))
            .lazy()
            .collect()
            .to_pandas()
        )

        fig = px.line(
            df_pd,
            x=".it",
            y="value",
            color="implicate",
            facet_row="parameter" if parameter == "both" else None,
            facet_col="vrb",
            labels={
                ".it": "Iteration",
                "value": "",
                "implicate": "Implicate",
                "vrb": "",
            },
            title="SRMI convergence (mice plot.mids()-equivalent trace plot)",
        )
        #   Free y-scale per panel - matches mice's own
        #       scales=list(y=list(relation="free")) (different
        #       variables/parameters are rarely on comparable scales).
        fig.update_yaxes(matches=None)

        if path is not None:
            fig.write_html(path)

        return fig

    def _resolve_variables(
        self, variable: str | int | list[str | int] | None
    ) -> list[Variable]:
        if variable is None:
            return list(self.variables)
        if not isinstance(variable, list):
            variable = [variable]
        resolved = []
        for vi in variable:
            if isinstance(vi, bool):
                raise TypeError(f"variable entry {vi!r} must be a str or int, not bool")
            if isinstance(vi, int):
                resolved.append(self.variables[vi])
            elif isinstance(vi, str):
                match = next((v for v in self.variables if v.impute_var == vi), None)
                if match is None:
                    message = (
                        f"No variable named {vi!r} in self.variables - have: "
                        f"{[v.impute_var for v in self.variables]}."
                    )
                    logger.error(message)
                    raise ValueError(message)
                resolved.append(match)
            else:
                raise TypeError(
                    f"variable entry {vi!r} must be a str (name) or int "
                    f"(0-indexed position in self.variables), got {type(vi)}"
                )
        return resolved

    def _observed_vs_imputed_data(
        self, variables: list[Variable]
    ) -> dict[str, dict[str, list]]:
        """
        {impute_var: {"observed": [...], "implicates": [[...], ...]}} -
        split via each Variable's own imputation_flag column (the exact
        same "was this row originally missing" indicator SRMI's own
        engine uses to decide what needed imputing in the first place -
        more robust than re-deriving it from self.df's own null
        pattern, since it's already the canonical source of truth,
        already present on every implicate's own df). "observed" is
        read from implicate 0 alone - the truly-observed values never
        differ across implicates, so there's nothing to gain (and
        something to lose - it'd overweight it M-fold) from repeating
        them M times, matching mice's own densityplot()/stripplot()
        convention of a single "observed" group set alongside M
        "imputed" ones.
        """
        data = {}
        df0 = nw.from_native(self.implicates[0].df).lazy().collect().to_native()
        for vari in variables:
            impute_col = vari.impute_var
            flag_col = vari.imputation_flag
            if impute_col not in df0.columns or flag_col not in df0.columns:
                continue

            observed = df0.filter(~pl.col(flag_col))[impute_col].drop_nulls().to_list()

            implicates_values = []
            for impi in self.implicates:
                dfi = nw.from_native(impi.df).lazy().collect().to_native()
                imputed_i = (
                    dfi.filter(pl.col(flag_col))[impute_col].drop_nulls().to_list()
                )
                implicates_values.append(imputed_i)

            data[impute_col] = {"observed": observed, "implicates": implicates_values}

        return data

    def plot_imputation_quality(
        self,
        variable: str | int | list[str | int] | None = None,
        kind: str = "density",
        sample_k: int | None = None,
        seed: int | None = None,
        path: str | None = None,
    ) -> "plotly.graph_objects.Figure":  # noqa: F821
        """
        Plot observed vs. imputed values - a DIFFERENT question from
        plot_convergence()'s "did the chain stabilize": here it's "do
        the imputed values look plausible next to the observed ones."
        Matches mice's own densityplot()/stripplot()/bwplot() as
        closely as possible (see utilities/quality_diagnostics.py's
        module docstring for the verified-against-source convention):
        one group for the truly observed values (pooled once), plus one
        group per implicate holding ONLY that implicate's own newly
        imputed values - never the whole column.

        Not built here (a materially bigger lift - needs a fitted
        propensity/detrending model, not just a reshape): mice's
        propensity-score xyplot() and its own "worm plot" (a detrended
        Q-Q plot conditional on a covariate - unrelated to the trace
        lines plot_convergence() draws, despite the similar-sounding
        name).

        No "auto-plot" flag in run() here either, same as
        plot_convergence() - call this whenever you want, including on
        an SRMI you've just SRMI.load()ed.

        Requires the optional 'plotly' package, same as
        plot_convergence() - raises a clear ImportError if it isn't
        installed, only when this is actually called.

        Parameters
        ----------
        variable : str | int | list[str | int] | None, optional
            Which variable(s) to plot - a name (impute_var), a 0-indexed
            position in self.variables, a list mixing either, or None
            for every variable in self.variables. By default None (all).
        kind : str, optional
            "density" (kernel density per group - numeric/boolean
            variables only, silently skips a group with fewer than 2
            distinct finite values, same wall mice's own densityplot()
            hits with no workaround), "strip" (every individual point,
            one column per group), or "box" (five-number-summary box
            plot per group). By default "density".
        sample_k : int | None, optional
            Only meaningful for kind="strip" - randomly sample at most
            this many points per (group, variable) to avoid overplotting
            a large dataset (mice's own guidance: stripplot is best for
            small datasets, use bwplot/box for large ones - this is the
            other way to cope with a large one and still see individual
            points). Ignored for "density"/"box", which should always
            use every point. By default None (no sampling).
        seed : int | None, optional
            Seed for the sample_k random sample. By default None.
        path : str | None, optional
            If given, also save the figure there as a self-contained
            HTML file. By default None.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        try:
            import plotly.express as px
        except ImportError as e:
            message = (
                "SRMI.plot_imputation_quality() needs the 'plotly' "
                "package, which isn't installed - it's optional (nothing "
                "about running SRMI itself needs it, only this diagnostic "
                "plot). Install it with `uv add --dev plotly` (or `pip "
                "install plotly`), then call this again - it works just "
                "as well on an SRMI.load()ed object as on one you just ran."
            )
            logger.error(message)
            raise ImportError(message) from e

        if kind not in ("density", "strip", "box"):
            message = (
                f"SRMI.plot_imputation_quality(): kind={kind!r} not "
                f"recognized - use 'density', 'strip', or 'box'."
            )
            logger.error(message)
            raise ValueError(message)

        variables = self._resolve_variables(variable)
        data = self._observed_vs_imputed_data(variables)

        if kind == "density":
            df_long = density_long_table(data)
            df_pd = nw.from_native(df_long).lazy().collect().to_pandas()
            fig = px.line(
                df_pd,
                x="x",
                y="density",
                color="group",
                facet_col="vrb",
                labels={"x": "", "density": "Density", "group": ""},
                title="Observed vs. imputed - density (mice densityplot()-equivalent)",
            )
        else:
            df_long = observed_vs_imputed_long_table(
                data, sample_k=sample_k if kind == "strip" else None, seed=seed
            )
            df_pd = nw.from_native(df_long).lazy().collect().to_pandas()
            plot_fn = px.strip if kind == "strip" else px.box
            fig = plot_fn(
                df_pd,
                x="group",
                y="value",
                color="group",
                facet_col="vrb",
                labels={"group": "", "value": ""},
                title=(
                    "Observed vs. imputed - individual points "
                    "(mice stripplot()-equivalent)"
                    if kind == "strip"
                    else "Observed vs. imputed - box plot (mice bwplot()-equivalent)"
                ),
            )

        fig.update_xaxes(matches=None)
        fig.update_yaxes(matches=None)

        if path is not None:
            fig.write_html(path)

        return fig

    def save_appended_cols_to_implicates(
        self, df_list: DataFrameList | list[IntoFrameT], columns: list[str], name: str
    ):
        for i in range(0, self.replication.n_implicates):
            self.implicates[i].save_appended_cols_to_implicate(
                df_list[i], columns=columns, name=name
            )

    def pipe(self, pipe, pipe_args=None):
        if pipe_args is None:
            pipe_args = {}

        for obji in self.df_containers:
            obji = obji.df.pipe(pipe, **pipe_args)

    @property
    def paths_full(self) -> list[str]:
        paths = [self.storage.path_model]

        for i in range(0, self.replication.n_implicates):
            paths.extend(self.implicates[i].paths_full)

        return paths

    @property
    def df_containers(self) -> list:
        #   A list to make editing the underlying dataframes easier
        containers = [self]
        for i in range(0, self.replication.n_implicates):
            containers.append(self.implicates[i])

        return containers

    #####################################################
    #   Serializable - BEGIN
    #####################################################
    def save(self):
        path = f"{self.storage.path_model}/SRMI"
        super().save(path)

        for impi in self.implicates:
            impi.save()

    @classmethod
    def _init_from_dict(cls, data: dict):
        return super()._init_from_dict(data)

    @classmethod
    def load(
        cls, path_model: str = "", implicate_number: int = 0, **df_kwargs
    ) -> SRMI | None:
        if isinstance(cls, SRMI) and path_model == "":
            path_model = cls.storage.path_model
        else:
            path_model = os.path.normpath(path_model)
            if not os.path.isdir(path_model) and os.path.isdir(
                os.path.normpath(f"{path_model}.{cls._save_suffix}")
            ):
                path_model = os.path.normpath(f"{path_model}.{cls._save_suffix}")

        obj = super().load(os.path.normpath(f"{path_model}/SRMI"), **df_kwargs)

        #   Load the implicates:
        for impi in range(1, obj.replication.n_implicates + 1):
            impi = Implicate(
                parent=obj,
                number=impi,
                load=(implicate_number == 0) or (implicate_number == impi),
                **df_kwargs,
            )
            impi.parent = obj
            obj.implicates.append(impi)
        return obj

    #####################################################
    #   Serializable - END
    #####################################################

    # return self


def run_implicate_async(
    path_model: str,
    implicate: int,
    iteration: int,
    variable_start: int = 0,
    variable_end: int = 0,
):
    srmi = SRMI.load(path_model=path_model, implicate_number=implicate)

    srmi.implicates[implicate - 1].run(
        iteration_number=iteration,
        variable_start=variable_start,
        variable_end=variable_end,
    )


def _function_has_argument(func, arg_name: str):
    return arg_name in inspect.signature(func).parameters
