from __future__ import annotations
from typing import Optional, Callable
import os
import narwhals as nw
from narwhals.typing import IntoFrameT
import shutil
import random
import math
import inspect
from pathlib import Path

from ..utilities.inputs import create_folders_if_needed

from ..utilities.dataframe import (
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

    Parameters
    ----------
    df : IntoFrameT
        Data to be used in the imputation
    variables : list[Variable]
        Variables to be imputed (as Variable class instances), by default None
    model : str | list, optional
        R string formula that is the default for the imputation
    selection : Selection, optional
        Variable selection method used within the imputation (if any), by default None
    preselection : Selection, optional
        Variable selection done before SRMI starts to pre-prune inputs, by default None
    modeltype : Variable.ModelType, optional
        Imputation model type from the ModelType enumeration, by default None
    parameters : dict, optional
        Model parameters dictionary, by default None
    joint : dict, optional
        Key-value pairs of variables to be included together (i.e. if one is selected
            in the variable selection step, the other is too), by default None
    ordered_categorical : list, optional
        List of categorical variables in model that are ordered, by default None
            An example would be education (vs. a variable with no ordering like state or county code)
    seed : int, optional
        Random seed for replicability, by default 0 (no seed)
    weight : str, optional
        Weight variable name for imputation modeling, by default ""
    n_implicates : int
        Number of separate implicates to impute
    n_iterations : int
        Number of iterations in each implicate
    bayesian_bootstrap : bool, optional
        Use Bayesian Bootstrap to account for uncertainty in coefficients, by default True
    bootstrap_index : list, optional
        Index variables for resampling in Bayesian Bootstrap (i.e. if you want to resample by household, not person), by default None
    bootstrap_where : str, optional
        SQL Condition for keeping observations when resampling, by default ""
    index : list, optional
        Columns that uniquely identify observations such as ["h_seq","pppos"], by default None
    parallel : bool, optional
        Run implicates in parallel, by default True
    parallel_variables_per_job : int, optional
        Number of variables per parallel job (for memory management and to deal with memory leaks, if there are any), by default 0
    parallel_CallInputs : CallInputs | None, optional
        Parameters for parallel execution such as memory and CPU allocation, by default None
    parallel_testing : bool, optional
        Test parallel jobs without running them, by default False
    path_model : str, optional
        Directory to save model data and temporary files
    model_name : str
        Model name for continuing existing runs, by default ""
    force_start : bool, optional
        Restart imputation even if existing run exists, by default False
    save_every_variable : bool, optional
        Save data after each variable is imputed, by default False
    save_every_iteration : bool, optional
        Save data after each iteration completes, by default True
    from_load : bool, optional
        Flag indicating object created from saved state, by default False
    imputation_stats : list[str] | None, optional
        List of statistics to calculate during imputation, by default None

    Raises
    ------
    Exception
        If n_implicates < 1 or n_iterations < 1
        If path equals path_model_new in load_to_continue_prior

    Examples
    --------
    Basic usage:

    >>> srmi = SRMI(
    ...     df=data,
    ...     variables=[var1, var2],
    ...     n_implicates=5,
    ...     n_iterations=10,
    ...     path_model="/path/to/model"
    ... )
    >>> srmi.run()

    With parallel execution:

    >>> srmi = SRMI(
    ...     df=data,
    ...     variables=vars_list,
    ...     n_implicates=5,
    ...     n_iterations=10,
    ...     parallel=True,
    ...     parallel_CallInputs=CallInputs(CPUs=4, MemInMB=5000)
    ... )
    """

    _save_suffix = "srmi"
    _save_exclude_items = ["implicates"]

    def __init__(
        self,
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
        from_load: bool = False,
    ):
        #   Error checking
        if n_implicates < 1:
            message = f"Must pass at least 1 implicate (passed {n_implicates})"
            logger.error(message)
            raise Exception(message)

        if n_iterations < 1:
            message = f"Must pass at least 1 iteration (passed {n_iterations})"
            logger.error(message)
            raise Exception(message)

        if index is None:
            index = []

        self.df = df
        if df is not None:
            self.nw_type = NarwhalsType(df)
        else:
            self.nw_type = None

        self.model = model
        if selection is None:
            selection = Selection(method=Selection.Method.No)
        self.selection = selection
        if preselection is None:
            preselection = Selection(method=Selection.Method.No)

        self.preselection = preselection
        self.modeltype = modeltype
        self.parameters = parameters
        self.joint = joint
        self.ordered_categorical = ordered_categorical
        self.seed = seed

        if self.seed > 0:
            set_seed(self.seed)

        self.weight = weight

        self.n_implicates = n_implicates
        self.n_iterations = n_iterations
        self.bayesian_bootstrap = bayesian_bootstrap
        self.bootstrap_index = bootstrap_index
        self.bootstrap_where = bootstrap_where

        self.save_every_variable = save_every_variable
        self.save_every_iteration = save_every_iteration

        # Add an index, if there isn't one
        #   We need one to be able to put the file back together again
        if type(index) is str:
            index = [index]

        if len(index) == 0:
            self.index = ["___rownumber"]

            self.df = (
                nw.from_native(self.df)
                .lazy()
                .collect()
                .with_row_index(name=self.index[0])
                .lazy_backend(self.nw_type)
                .to_native()
            )
        else:
            #   Needs to be unique
            if safe_height(
                nw.from_native(self.df).select(index).unique().to_native()
            ) != safe_height(self.df):
                logger.info("Adding row number to the index as it is not unique")
                self.df = self.df.with_row_index(name="___rownumber")
                index.append("___rownumber")

            self.index = index

        self.parallel = parallel
        self.parallel_variables_per_job = parallel_variables_per_job

        if self.parallel and parallel_CallInputs is None:
            #   Make sure the data is in memory to get the estimated size
            # file_size = NarwhalsType(self.df).to_polars().lazy().collect().estimated_size(unit="mb")

            # if file_size <= 1_000:
            #     MemInMB = 5_000
            # elif file_size <= 10_000:
            #     MemInMB = 30_000
            # elif file_size <= 25_000:
            #     MemInMB = 75_000
            # elif file_size <= 50_000:
            #     MemInMB = 125_000
            # elif file_size <= 75_000:
            #     MemInMB = 175_000
            # elif file_size <= 100_000:
            #     MemInMB = 250_000
            # elif file_size <= 150_000:
            #     MemInMB = 350_000
            # else:
            #     MemInMB = 500_000

            # if self.parallel_variables_per_job == 0:
            #     MemInMB = min(self.parallel_variables_per_job*5,500_000)
            # elif self.parallel_variables_per_job >= 100:
            #     MemInMB = min(self.parallel_variables_per_job*3,500_000)
            # elif self.parallel_variables_per_job >= 50:
            #     MemInMB = min(self.parallel_variables_per_job*2,500_000)

            n_available_cpus = Config().cpus
            n_parallel_cpus = max(int(n_available_cpus / self.n_implicates), 1)

            self.parallel_CallInputs = CallInputs(
                call_type=CallTypes.shell,
                n_cpu=n_parallel_cpus,
                process_limit=min(n_available_cpus, self.n_implicates),
            )
        else:
            self.parallel_CallInputs = parallel_CallInputs
        self.parallel_testing = parallel_testing

        self.imputation_stats = imputation_stats

        self.setup_complete = False

        #   No model path, use a temporary one (with temp, can't really continue)
        if path_model == "":
            if Config().path_temp_files == "":
                message = "You must pass in a path to save the imputation files to (path_model)"
                logger.error(message)
                raise Exception(message)
            else:
                path_model = Config().path_temp_with_random()

        #   For safety against accidental deletes, file path has .srmi suffix
        if not path_model.endswith(".srmi"):
            path_model = path_model + ".srmi"
        self.path_model = path_model

        #   Force start? - then delete any saved data
        if os.path.isdir(self.path_model) and force_start:
            logger.info(f"Removing existing directory {self.path_model}")
            shutil.rmtree(self.path_model)

        self.variables = []
        if variables is not None:
            for vari in variables:
                self.AddVariable(vari)

        self.implicates = []

        #   Defaults to false
        self.is_continuing_srmi = False
        self.continuing_cols = []

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
                setattr(variable, stri, getattr(self, stri))

        for obji in none_override:
            if getattr(variable, obji) is None:
                setattr(variable, obji, getattr(self, obji))

        if len(variable.parameters) == 0:
            variable.parameters = self.parameters

        #   Remove the variable itself from its own model
        #       and any variables in variable.predictors_exclude
        variable.exclude_variables_from_models(df=self.df)

        #   Do some pre-checks to catch any errors that will stop things later
        variable.validate_inputs(df=self.df)

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
        create_folders_if_needed(self.path_model, quietly=True)

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
                        .is_missing()
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

        if self.parallel:
            #   Set up jobs to run the implicates in parallel
            f_implicates = []
            for impi in self.implicates:
                for iterationi in range(1, self.n_iterations + 1):
                    if self.parallel_variables_per_job > 0:
                        n_jobs_per_loop = math.ceil(
                            len(self.variables) / self.parallel_variables_per_job
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

                        if self.parallel_variables_per_job == 0:
                            variable_start = 0
                            variable_end = 0
                        else:
                            variable_start = (
                                sub_job * self.parallel_variables_per_job + 1
                            )
                            variable_end = (
                                variable_start + self.parallel_variables_per_job - 1
                            )

                        #   Dummy inputs and outputs to order the iteration runs properly
                        if sub_job > 0 or iterationi > 1:
                            inputs = [
                                f"{self.path_model}/logs/iteration_{impi}_{prior_iteration}_{prior_sub}.log"
                            ]
                        else:
                            inputs = []
                        outputs = [
                            f"{self.path_model}/logs/iteration_{impi}_{iterationi}_{sub_job}.log"
                        ]

                        f_implicates.append(
                            FunctionFromPython(
                                function=run_implicate_async,
                                parameters={
                                    "path_model": Path(self.path_model).as_posix(),
                                    "implicate": impi.number,
                                    "iteration": iterationi,
                                    "variable_start": variable_start,
                                    "variable_end": variable_end,
                                },
                                inputs=inputs,
                                outputs=outputs,
                                #    namespace="NEWS.CodeUtilities.Python.SRMI.SRMI"                               )
                            )
                        )

            log = run_function_list(
                function_list=f_implicates,
                call_input=self.parallel_CallInputs,
                run_all=True,
                testing=self.parallel_testing,
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

            for impi in range(self.n_implicates):
                this_implicate = Implicate(
                    parent=self, number=impi + 1, seed=random.randint(1, 2**32 - 1)
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
            df=self.df, y=variable.impute_var, formula=fb.formula
        )
        if selected_model != "":
            variable.model = selected_model

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
                .filter(nw.col(variable.impute_var).is_not_missing())
                .to_native()
            )
            df_tune = variable.df_where(df_tune)

            lgbm = kit_lightgbm(
                df=df_tune,
                y=variable.impute_var,
                formula=variable.model,
                weight=self.weight,
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

        srmi_prior.seed = seed

        if srmi_prior.seed > 0:
            set_seed(srmi_prior.seed)

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
                for i in range(0, srmi_prior.n_implicates + 1):
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
        srmi_prior.path_model = path_model_new

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
            for i in range(0, srmi.n_implicates):
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
            data appended to them.  We use this in NEWS for getting
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

    def save_appended_cols_to_implicates(
        self, df_list: DataFrameList | list[IntoFrameT], columns: list[str], name: str
    ):
        for i in range(0, self.n_implicates):
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
        paths = [self.path_model]

        for i in range(0, self.n_implicates):
            paths.extend(self.implicates[i].paths_full)

        return paths

    @property
    def df_containers(self) -> list:
        #   A list to make editing the underlying dataframes easier
        containers = [self]
        for i in range(0, self.n_implicates):
            containers.append(self.implicates[i])

        return containers

    #####################################################
    #   Serializable - BEGIN
    #####################################################
    def save(self):
        path = f"{self.path_model}/SRMI"
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
            path_model = cls.path_model
        else:
            path_model = os.path.normpath(path_model)
            if not os.path.isdir(path_model) and os.path.isdir(
                os.path.normpath(f"{path_model}.{cls._save_suffix}")
            ):
                path_model = os.path.normpath(f"{path_model}.{cls._save_suffix}")

        obj = super().load(os.path.normpath(f"{path_model}/SRMI"), **df_kwargs)

        #   Load the implicates:
        for impi in range(1, obj.n_implicates + 1):
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
