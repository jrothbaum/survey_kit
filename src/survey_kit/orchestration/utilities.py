from __future__ import annotations

import os
import importlib
import inspect
from enum import Enum

from .config import Config
from ..serializable import Serializable
from .. import logger


LINEBREAK = "\n"


class Languages(Enum):
    """
    Enumeration of supported programming languages for function execution.

    This enum defines the programming languages that can be used with the
    Function class for parallel or asynchronous execution. Each language
    has specific handling for code generation, execution, and logger.

    Options
    ----------
    - SAS
    - Python
    - Stata
    - R
    - Bash

    Examples
    --------
    >>> f = Function(language=Languages.Python, name="my_function")
    >>> f = Function(language=Languages.SAS, name="StandardizeCPS")
    """

    SAS = 0
    Python = 1
    Stata = 2
    R = 3
    Bash = 4  #  Just run a bash script and do nothing with it

    @classmethod
    def language_from_path(cls, path: str) -> Languages:
        if path.endswith(".py"):
            return Languages.Python
        elif path.endswith(".R"):
            return Languages.R
        elif path.endswith(".do") or path.endswith(".ado"):
            return Languages.Stata
        elif path.endswith(".sas"):
            return Languages.SAS
        elif path.endswith(".bash") or path.endswith(".sh"):
            return Languages.Bash
        else:
            message = f"Invalid file suffix ({path})"
            logger.error(message)
            raise Exception(message)


class CallTypes(Enum):
    """
    Enumeration of execution methods for function calls.

    This enum defines how functions should be executed, ranging from
    high-performance cluster jobs to local processing methods.

    Options
    ----------
    - PBS : PBS Pro job scheduler for cluster execution with resource allocation.
    - shell : Shell-based execution using subprocess calls.
    - multiprocessing : Python multiprocessing for parallel execution within a single node.
    - in_process : Direct execution within the current Python process.

    Examples
    --------
    >>> call_input = CallInputs(call_type=CallTypes.PBS, mem_in_mb=50000, n_cpu=4)
    >>> call_input = CallInputs(call_type=CallTypes.shell)
    """

    PBS = 0
    shell = 1
    multiprocessing = 2
    in_process = 3


class CallInputs(Serializable):
    def __init__(
        self,
        call_type: CallTypes = CallTypes.shell,
        check_every_x_seconds: float = 0.2,
        total_wait_seconds: int = 0,
        mem_in_mb: int = 5000,
        n_cpu: int = -1,
        process_limit: int = 0,
    ):
        """
        Configuration for function execution including resource allocation and timing.

        CallInputs specifies how functions should be executed, including the execution
        method, resource requirements, and monitoring parameters. This class provides
        a unified interface for configuring both local and PBS Pro cluster-based execution.

        Parameters
        ----------
        call_type : CallTypes, optional
            Execution method (PBS, shell, multiprocessing, in_process).
            Default is CallTypes.shell.
        check_every_x_seconds : float, optional
            Frequency for checking job completion status. Lower values for quick jobs,
            higher for slow ones. Default is 0.2 seconds.
        total_wait_seconds : int, optional
            Maximum time to wait for jobs to complete (0 = no timeout).
            Default is 0 (wait indefinitely).
        mem_in_mb : int, optional
            Memory allocation in MB for PBS Pro jobs. Default is 5000.
        n_cpu : int, optional
            Number of CPU cores to request. Auto-configured based on call_type if -1:
            - PBS: 4 cores
            - shell/multiprocessing/in_process: 1 core
            Default is -1 (auto-configure).
        process_limit : int, optional
            Maximum number of parallel processes to spawn simultaneously.
            0 means use system defaults. Default is 0.

        Attributes
        ----------
        call_type : CallTypes
            The execution method being used.
        check_every_x_seconds : float
            Job status check frequency.   You should leave this as is.
        total_wait_seconds : int
            Maximum wait time for completion. 0 = wait forever...
        mem_in_mb : int
            Memory allocation for PBS jobs.
        n_cpu : int
            Number of CPU cores allocated.
        process_limit : int
            Process concurrency limit.  You should leave this as is.

        Examples
        --------
        Basic shell execution:

        >>> call_input = CallInputs(call_type=CallTypes.shell)

        High-memory PBS job:

        >>> call_input = CallInputs(
        ...     call_type=CallTypes.PBS,
        ...     mem_in_mb=100000,
        ...     n_cpu=8,
        ... )

        Notes
        -----
        - Memory allocation (mem_in_mb) only applies to PBS Pro jobs
        """

        self.call_type = call_type
        self.check_every_x_seconds = check_every_x_seconds
        self.total_wait_seconds = total_wait_seconds

        self.mem_in_mb = mem_in_mb

        if n_cpu < 0:
            if self.call_type.value == CallTypes.PBS.value:
                n_cpu = 4
            elif self.call_type.value == CallTypes.shell.value:
                n_cpu = 1
            elif self.call_type.value == CallTypes.multiprocessing.value:
                n_cpu = 1
            elif self.call_type.value == CallTypes.in_process.value:
                #   Not relevant if in process
                #       Will only be used if not a python call
                n_cpu = 1

        self.n_cpu = n_cpu
        self.process_limit = process_limit

    def __str__(self):
        return convert_to_constructor(self)

    @property
    def is_in_process(self):
        return self.call_type == CallTypes.in_process


class DefaultCallInputs:
    """
    Pre-configured CallInputs instances for common use cases.

    This class provides convenient defaults for different computational
    workloads, automatically selecting appropriate execution methods
    and resource allocations based on system capabilities.

    Attributes
    ----------
    small : CallInputs
        Configuration for small jobs. Uses shell execution on systems
        with sufficient RAM, PBS with 25GB RAM otherwise.
    medium : CallInputs
        Configuration for medium jobs. PBS execution with 50GB RAM, 4 n_cpu.
    big : CallInputs
        Configuration for large jobs. PBS execution with 150GB RAM, 8 n_cpu.
    verybig : CallInputs
        Configuration for very large jobs. PBS execution with 250GB RAM, 8 n_cpu.
    huge : CallInputs
        Configuration for huge jobs. PBS execution with 400GB RAM, 8 n_cpu.
    veryhuge : CallInputs
        Configuration for huge jobs. PBS execution with 600GB RAM, 12 n_cpu.
    Examples
    --------
    >>> f = FunctionFromPython(function=my_function, call_input=DefaultCallInputs.big)
    >>> RunFunctionList(function_list, call_input=DefaultCallInputs.medium)

    """

    ram = Config().mem_in_mb

    if ram is None:
        ram = 0

    if ram < 25000:
        #       Really small, don't do a separate job, just run here
        small = CallInputs(call_type=CallTypes.PBS, mem_in_mb=25000, n_cpu=4)
    else:
        #       Really small, don't do a separate job, just run here
        small = CallInputs(call_type=CallTypes.shell)

    #       Medium, don't ask for THAT much RAM
    medium = CallInputs(call_type=CallTypes.PBS, mem_in_mb=50000, n_cpu=4)
    #       Big - more RAM
    big = CallInputs(call_type=CallTypes.PBS, mem_in_mb=150000, n_cpu=8)
    #       Really Big - tons of RAM
    verybig = CallInputs(call_type=CallTypes.PBS, mem_in_mb=250000, n_cpu=8)

    #       Huge - tons of RAM
    huge = CallInputs(call_type=CallTypes.PBS, mem_in_mb=400000, n_cpu=8)

    #       Biggest..
    veryhuge = CallInputs(call_type=CallTypes.PBS, mem_in_mb=600000, n_cpu=12)


class UpdateParams(Serializable):
    """
    Configuration for determining when functions need to be executed.

    UpdateParams controls the logic for deciding which functions in a
    dependency graph should be executed, based on various criteria like
    file timestamps, explicit lists, and dependency changes.

    Parameters
    ----------
    update_run : bool, optional
        Whether to check if functions need updating. If False, no functions
        will be automatically set to run. Default is True.
    update_by_date : bool, optional
        Check file timestamps to determine if inputs are newer than outputs.
        Default is False.
    update_by_output : list, optional
        List of specific output files that should trigger re-execution.
        Default is None (empty list).
    update_by_function_name : list, optional
        List of specific function names that should be re-executed.
        Default is None (empty list).
    update_by_used_file_list : bool, optional
        Check if the list of input files has changed since last execution.
        Default is False.



    Examples
    --------
    Basic update configuration:

    >>> update = UpdateParams(
    ...     update_by_date=True,
    ...     update_by_used_file_list=True
    ... )

    Force specific functions to run:

    >>> update = UpdateParams(
    ...     update_by_function_name=["load_data", "process_results"],
    ...     update_by_output=["/data/critical_output.parquet"]
    ... )

    Disable all automatic updates:

    >>> update = UpdateParams(update_run=False)

    Notes
    -----
    Multiple update criteria can be combined. If any criterion is met,
    the function will be set to run. The update_run flag provides a
    master switch to disable all automatic update checking.
    """

    def __init__(
        self,
        update_run: bool = True,
        update_by_date: bool = False,
        update_by_output: list = None,
        update_by_function_name: list = None,
        update_by_used_file_list: bool = False,
    ):
        if update_by_output is None:
            update_by_output = []
        if update_by_function_name is None:
            update_by_function_name = []

        self.update_run = update_run
        self.update_by_output = update_by_output
        self.update_by_function_name = update_by_function_name
        self.update_by_date = update_by_date
        self.update_by_used_file_list = update_by_used_file_list


def convert_to_constructor(item, from_init: bool = False):
    constructor = item.__class__.__name__ + "("
    add_comma = False

    if from_init:
        signature = inspect.signature(item.__init__)

        for keyi in signature.parameters.keys():
            if hasattr(item, keyi):
                attr = [keyi, getattr(item, keyi)]
                if add_comma:
                    constructor += ","

                if type(attr[1]) is str:
                    constructor += attr[0] + "='" + str(attr[1]) + "'"
                else:
                    constructor += attr[0] + "=" + str(attr[1])

                add_comma = True

    else:
        for attr in inspect.getmembers(item):
            if not attr[0].startswith("_") and not inspect.ismethod(attr[1]):
                if add_comma:
                    constructor += ","

                if type(attr[1]) is str:
                    constructor += attr[0] + "='" + str(attr[1]) + "'"
                else:
                    constructor += attr[0] + "=" + str(attr[1])

                add_comma = True
    constructor += ")"

    return constructor


class FileLoaderUtilities:
    """
    Class that holds parameters that determine what happens and is returned
    from file loaders

    args:
        obss: int, optional
            If > 0, load a subset of observations.
            Default is 0.

        run: bool, optional
            Should this be run to return or save data or return
            a :class:`~.function.Function`
            Default is True.
        return_data: bool, optional
            Do I need to load the data after the function?
            (No if you're just saving the file)
        track_outputs, optional
            When returning a :class:`~.function.Function`, include a list of inputs and outputs
            in it.
            Default is True
        testing, boolean, optional
            Do not run the function, but instead save a tempfile in
            /TempFiles in the data root
            Default is False
    """

    def __init__(
        self,
        obs: int = 0,
        run: bool = True,
        return_data: bool = True,
        track_outputs: bool = True,
        testing: bool = False,
    ):
        self.obs = obs
        self.run = run
        self.return_data = return_data
        self.track_outputs = track_outputs
        self.testing = testing

    def __str__(self):
        return convert_to_constructor(self)


def load_utility(
    path_full: str = "",
    file: str = "",
    folder: str = "",
    existing_utilities: dict | None = None,
    module_only=False,
):
    if existing_utilities is None:
        existing_utilities = {}

    if path_full != "":
        folder = os.path.dirname(path_full)
        file = os.path.basename(path_full)

    if file.endswith(".py"):
        file = file[0 : len(file) - 3]
    spec = importlib.util.spec_from_file_location(
        file, os.path.normpath(f"{folder}/{file}.py")
    )

    existing_utilities[file] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(existing_utilities[file])

    if module_only:
        return existing_utilities[file]
    else:
        return existing_utilities
