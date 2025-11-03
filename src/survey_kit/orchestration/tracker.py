from __future__ import annotations
from typing import Callable

import os
import time

from ..serializable import SerializableDictionary

from .utilities import CallTypes, CallInputs, UpdateParams
from .need_to_run import InputChecker
from .function import Function
from .dependency_order import FunctionDependencyOrder
from .config import Config

from .. import logger


class FunctionTracker:
    """
    Manages parallel execution of multiple functions with dependency tracking.

    FunctionTracker orchestrates the execution of function lists, handling
    dependencies, resource limits, and progress monitoring. It supports
    various execution methods and provides detailed logging and status tracking.

    Parameters
    ----------
    update : UpdateParams, optional
        Configuration for determining which functions need execution.
        Default creates new UpdateParams.
    call_input : CallInputs, optional
        Default execution configuration for all functions.
        Default creates new CallInputs.
    use_function_call_inputs : bool, optional
        Whether to use function-specific CallInputs instead of global.
        Default is False.
    path_temp : str, optional
        Directory for temporary files. Default is "".
    no_print_log : bool, optional
        Whether to suppress detailed logging output. Default is False.
    on_complete_by_file : callable, optional
        Function to call when specific files are created. Default is None.
    on_complete_by_file_params : dict, optional
        Parameters for on_complete_by_file function. Default is None.

    Attributes
    ----------
    call_input : CallInputs
        Default execution configuration.
    update : UpdateParams
        Update checking configuration.
    use_function_call_inputs : bool
        Whether to use per-function execution settings.
    n_functions_running : dict
        Count of currently running functions by execution type.
    n_functions_run_total : dict
        Total count of functions run by execution type.
    all_complete : bool
        Whether all functions have completed.
    any_changed : bool
        Whether any function status has changed recently.

    Examples
    --------
    Basic usage (with a FunctionDependencyOrder object named function_ordering):

    >>> tracker = FunctionTracker(
    ...     update=UpdateParams(update_by_date=True),
    ...     call_input=CallInputs(call_type=CallTypes.shell)
    ... )
    >>> log = tracker.run_list(function_ordering)

    With resource limits:

    >>> tracker = FunctionTracker(
    ...     call_input=CallInputs(process_limit=4),
    ...     no_print_log=True
    ... )

    Notes
    -----
    FunctionTracker automatically manages process limits to prevent
    system overload and handles the complexity of dependency resolution
    and parallel execution coordination.
    """

    max_pbs_processes = 10

    def __init__(
        self,
        #   Arguments for what needs updating (input/output, Function name list, Date of files)
        update: UpdateParams = None,
        #  Overall call input to use for all the calls (if not use_function_call_inputs)
        call_input: CallInputs = None,
        #  Use function-specific call input information
        #      to determine if calls are shell/PBS and resources available
        use_function_call_inputs: bool = False,
        #   If > 0, limit the number of shell running processes (to avoid job kills)
        path_temp: str = "",
        no_print_log: bool = False,
        #  Run a function when files show up
        #      run when creating a file - works when you create it
        #      with a temp name then rename it when it's done
        on_complete_by_file=None,
        #  Dictionary of file names to check (keys)
        #      and values are dictionary of function kwargs
        on_complete_by_file_params: dict | None = None,
    ):
        if call_input is None:
            call_input = CallInputs()
        self.call_input = call_input

        if update is None:
            update = UpdateParams()
        self.update = update

        self.use_function_call_inputs = use_function_call_inputs
        self.path_temp = path_temp
        self.no_print_log = no_print_log

        #   Initialize the count of running functions and call number
        self.call_number = 0
        self.n_functions_running = {}
        self.n_functions_run_total = {}

        for index, item in enumerate(CallTypes):
            self.n_functions_running[item] = 0
            self.n_functions_run_total[item] = 0

        self.on_complete_by_file = on_complete_by_file
        self.on_complete_by_file_params = on_complete_by_file_params

        if self.call_input.process_limit == 0:
            if (self.call_input.call_type.value == CallTypes.shell.value) or (
                self.call_input.call_type.value == CallTypes.multiprocessing.value
            ):
                cpus = Config().cpus
                if cpus is not None:
                    if self.call_input.n_cpu < 1:
                        self.call_input.process_limit = int(cpus / 2)
                    else:
                        self.call_input.process_limit = int(
                            (cpus - 1) / (self.call_input.n_cpu)
                        )
                else:
                    #   No processor count?  Default to 4, I guess
                    self.call_input.process_limit = 4
            else:
                #   PBS jobs, let PBS pro handle running jobs, up to a point...
                self.call_input.process_limit = self.max_pbs_processes

        #   Initialize to any changed to run on the first loop
        self.any_changed = True
        #   Initialize to not complete
        self.all_complete = False

        #   Initialize the tracker's timer
        self.start_time = time.time()

    def run_list(
        self,
        function_ordering: FunctionDependencyOrder,
        testing: bool = False,
        function_check_every: Callable = None,
        params_check_every: dict | None = None,
    ):
        #   Loop over the functions and run them if they need to be run and they are ready
        #       Start with defaults, not all complete and anyChanged = True (to trigger run)
        if testing:
            #   Just run each function call with testing to create all the files
            for groupi in [
                function_ordering.run_initial,
                function_ordering.run_on_parent_complete,
            ]:
                for functioni in groupi:
                    #   This function is set to run, create the file
                    if functioni.run:
                        self.call_number = self.call_number + 1
                        functioni.call(
                            NamePrefix=str(self.call_number).zfill(3) + "_",
                            call_input=self.call_input,
                            testing=testing,
                        )

            #   No log to return
            return ""
        else:
            #   If anyChanged, check if any are ready to run
            if not self.no_print_log:
                logger.info("\n\nStarting call:")
            #   Keep checking until everything is finished or we run out of time (only if total_wait_seconds > 0)

            #   times_checked = 0

            #   Start the jobs that are ready to run
            pending_jobs = (
                function_ordering.run_initial + function_ordering.run_on_parent_complete
            )

            for functioni in function_ordering.run_initial:
                if functioni.run:
                    if self.run_new_process(functioni):
                        self.run_function(
                            function=functioni, no_print_log=self.no_print_log
                        )

            while not self.all_complete and (
                self.call_input.total_wait_seconds == 0
                or (
                    (time.time() - self.start_time) < self.call_input.total_wait_seconds
                )
            ):
                #   if (self.any_changed):
                #   Something finished (or first loop), check if anything else is ready to start
                for functioni in pending_jobs:
                    #   This function is set to run, but hasn't started
                    if functioni.run and not functioni.call_status.started:
                        #   Is it ready to go? If so, run it if we're ready to run another process
                        if self.ReadyToRun(function=functioni):
                            if self.run_new_process(function=functioni):
                                self.run_function(
                                    function=functioni, no_print_log=self.no_print_log
                                )

                #   Wait for set amount of time (for progress)
                time.sleep(self.call_input.check_every_x_seconds)

                #   times_checked += 1
                # if (times_checked % 50) == 0:
                #     logger.info(".")
                # else:
                #     logger.info(".[!n]")
                logger.info(".[!n]")

                #   Update completion status for next round in loop
                pending_jobs = self.update_complete(
                    function_ordering=function_ordering, pending_jobs=pending_jobs
                )

                #   Run any on_complete functions that are triggered by files
                self.check_for_complete_files()

                for functioni in pending_jobs:
                    #   Done starting new jobs, new run completion script if needed
                    if (
                        functioni.on_complete is not None
                        and functioni.call_status.complete
                    ):
                        try:
                            if len(functioni.on_complete_args):
                                functioni.on_complete(**functioni.on_complete_args)
                            else:
                                functioni.on_complete()
                        except Exception as error:
                            message = f"on_complete function failed: {error}"
                            logger.error(message)
                            raise Exception(message)
                        #   Clear it so it doesn't run again
                        functioni.on_complete = None

                if self.any_changed:
                    #   Next line from the dots
                    if not self.no_print_log:
                        logger.info("")

                if function_check_every is not None:
                    params_check_every = function_check_every(**params_check_every)

            #    logger.info("Timeout Time=" + str(call_input.total_wait_seconds))
            if self.call_number == 0:
                logger.info("\nNo functions needed to run.")
                full_log = ""
            else:
                #   Run any on_complete functions that are triggered by files
                self.check_for_complete_files(final_call=True)

                # if self.all_complete:
                #     #   Update files for functions run as in_process
                #     for groupi in [function_ordering.run_initial,
                #                    function_ordering.run_on_parent_complete]:
                #         for functioni in groupi:
                #             if functioni.call_status.call_input.call_type.value != CallTypes.in_process.value:
                #                 self.save_inputs_for_function(functioni)

                if not self.all_complete:
                    logger.info(
                        "Timeout="
                        + str(
                            (time.time() - self.start_time)
                            > self.call_input.total_wait_seconds
                        )
                    )

                if self.no_print_log:
                    logger.info("")
                else:
                    logger.info("Complete=" + str(self.all_complete))
                full_log = self.collect_results(function_ordering=function_ordering)
            #   logger.info(full_log)

            if function_check_every is not None:
                params_check_every = function_check_every(**params_check_every)

            return full_log

    def run_function(self, function: Function = None, no_print_log: bool = False):
        if not no_print_log:
            logger.info("")
            logger.info(
                "Starting call at "
                + (str(int(time.time() - self.start_time)))
                + " seconds"
            )
            logger.info("      Because:    " + str(function.run_because))
            logger.info(function)
        if self.use_function_call_inputs:
            thisCallType = function.call_status.call_input.call_type
            thiscall_input = function.call_status.call_input
        else:
            thisCallType = self.call_input.call_type
            thiscall_input = self.call_input

        #   This one started, add to list of running processes
        self.n_functions_running[thisCallType] += 1
        self.n_functions_run_total[thisCallType] += 1
        self.call_number = self.call_number + 1

        function.call(
            path_temp=self.path_temp,
            NamePrefix=str(self.call_number).zfill(3) + "_" + function.name + "_",
            call_input=thiscall_input,
        )

        function.call_status.call_number = self.call_number

    def run_new_process(self, function: Function):
        if self.use_function_call_inputs:
            thisCallType = function.call_status.call_input.call_type
        else:
            thisCallType = self.call_input.call_type

        if (
            thisCallType.value == CallTypes.shell.value
            or thisCallType.value == CallTypes.multiprocessing.value
            or thisCallType.value == CallTypes.in_process.value
        ):
            # logger.info(f"self.call_input.process_limit = {self.call_input.process_limit}")
            # logger.info(f"self.n_functions_running[thisCallType] = {self.n_functions_running[thisCallType]}")
            # logger.info(f"self.max_pbs_processes = {self.max_pbs_processes}")
            # logger.info(f"self.call_input.process_limit = {self.call_input.process_limit}")
            # logger.info(f"self.n_functions_running[thisCallType] = {self.n_functions_running[thisCallType]}")

            return (
                self.call_input.process_limit == 0
                and self.n_functions_running[thisCallType] < self.max_pbs_processes
            ) or (
                self.call_input.process_limit > self.n_functions_running[thisCallType]
            )
        elif thisCallType.value == CallTypes.PBS.value:
            return self.n_functions_running[thisCallType] <= self.max_pbs_processes

    def update_complete(
        self, function_ordering: FunctionDependencyOrder, pending_jobs: list[Function]
    ):
        self.any_changed = False
        self.all_complete = True

        for keyi in self.n_functions_running:
            self.n_functions_running[keyi] = 0

        any_updated = False
        for functioni in pending_jobs:
            if functioni.call_status.started:
                #   To be run, so we need to check if it's done
                if not functioni.call_status.complete:
                    functioni.call_status.check_completion()

                    if self.use_function_call_inputs:
                        thisCallType = functioni.call_status.call_input.call_type
                    else:
                        thisCallType = self.call_input.call_type

                    #   Done?, then Get the results of the call and update that one changed status
                    if functioni.call_status.complete:
                        any_updated = True
                        if not self.no_print_log:
                            logger.info(functioni.full_log)
                        else:
                            #   Do this and swallow to delete the log files
                            functioni.full_log

                        self.any_changed = True

                        if not thisCallType == CallTypes.in_process:
                            self.save_inputs_for_function(functioni)
                    else:
                        self.n_functions_running[thisCallType] += 1

            #   Update all complete to false if not finished
            self.all_complete = self.all_complete and (
                functioni.call_status.complete or not functioni.run
            )

        if any_updated:
            pending_jobs = self._update_pending_jobs(pending_jobs)

        return pending_jobs

    def _update_pending_jobs(self, pending_jobs: list[Function]) -> list[Function]:
        return [fi for fi in pending_jobs if (fi.run and not fi.call_status.complete)]

    def ReadyToRun(self, function: Function = None):
        #   Start at true (so if no parents, bReady == True)
        bReady = True
        for parenti in function.parent_functions:
            #   Run if parent is finished or parent doesn't need to run (for all parents)
            if bReady:
                bReady = bReady and (
                    (parenti.run and parenti.call_status.complete) or (not parenti.run)
                )

        return bReady

    def collect_results(self, function_ordering: FunctionDependencyOrder):
        fullresults = ""
        for groupi in [
            function_ordering.run_initial,
            function_ordering.run_on_parent_complete,
        ]:
            for functioni in groupi:
                if functioni.run and functioni.call_status.complete:
                    fullresults += "\n\n" + "BEGIN CALL FOR " + functioni.name + "\n"
                    fullresults += functioni.call_code(save_serialize_params=False)
                    fullresults += "END CALL\n\n"

                    fullresults += (
                        "Approximate execution time: "
                        + str(functioni.call_status.execution_time)
                        + " seconds\n\n"
                    )

                    #   fullresults = functioni.full_log

        return fullresults

    def save_inputs_for_function(self, functioni: Function) -> None:
        if hasattr(self, "no_print_log"):
            no_print_log = self.no_print_log
        else:
            no_print_log = False

        if functioni.call_status.complete:
            #   This function was run, does it have inputs and outputs
            if len(functioni.inputs) > 0 and len(functioni.outputs) > 0:
                if (
                    functioni.inputs_write_only is not None
                    and len(functioni.inputs_write_only) > 0
                ):
                    inputs_write = functioni.inputs_write_only
                else:
                    inputs_write = functioni.inputs

                InputChecker.save_inputs(
                    inputs=inputs_write,
                    outputs=functioni.outputs,
                    args=self.args_for_comparison(functioni),
                    quietly=no_print_log,
                )

    @classmethod
    def args_for_comparison(cls, functioni: Function):
        args = functioni.parameters
        if type(args) is SerializableDictionary:
            args = args._d

        args = args.copy()

        if len(functioni.tracker_args_to_drop):
            drop_list = functioni.tracker_args_to_drop
        else:
            drop_list = []

        if len(functioni.tracker_args_to_keep):
            drop_list.extend(
                set(args.keys()).difference(functioni.tracker_args_to_keep)
            )

        if len(drop_list):
            for keyi in drop_list:
                if keyi in args:
                    del args[keyi]

        return args

    def check_for_complete_files(self, final_call: bool = False):
        if self.on_complete_by_file is not None:
            if len(self.on_complete_by_file_params):
                completed_files = []

                for filei, paramsi in self.on_complete_by_file_params.items():
                    if os.path.exists(filei):
                        logger.info(f"\n  Running post-processing for {filei}")
                        self.on_complete_by_file(**paramsi)
                        completed_files.append(filei)

                if final_call:
                    #   Log which files never showed up
                    logger.info("\nPost-processing never run for:")
                    for keyi in self.on_complete_by_file_params.keys():
                        logger.info(f"     {keyi}")

                if len(completed_files):
                    for filei in completed_files:
                        del self.on_complete_by_file_params[filei]

    @classmethod
    def args_to_keep(cls, args: list[str]):
        if type(args) is str:
            args = [args]

        def decorator(func):
            func.args_to_keep = args
            return func

        return decorator

    @classmethod
    def args_to_drop(cls, args: list[str]):
        if type(args) is str:
            args = [args]

        def decorator(func):
            func.args_to_drop = args
            return func

        return decorator
