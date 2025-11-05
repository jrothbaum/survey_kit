from __future__ import annotations
from typing import Callable

from enum import Enum
import sys
import os
import tempfile
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
import polars as pl
import time
import importlib.util
import subprocess
from pathlib import Path

from .config import Config
from ..utilities.logging import run_with_temporary_logging
from .shared_memory import SharedMemoryUtility

from .utilities import (
    Languages,
    CallInputs,
    CallTypes,
    LINEBREAK,
    convert_to_constructor,
)

from .need_to_run import RunBecause
from .call_status import CallStatus

from ..serializable import (
    Serializable,
    SerializableDictionary,
    SerializableList,
)

from ..utilities.inputs import list_input, create_folders_if_needed
from .. import logger

from .path_with_fallbacks import PathWithFallbacks, as_output_path


class Function(Serializable):
    """
    A configurable function call that can be executed in various languages and environments.

    Function represents a single computational task that can be executed locally
    or remotely, with support for multiple programming languages, dependency
    tracking, and resource management. It encapsulates all information needed
    for execution including code, parameters, and execution context.

    Parameters
    ----------
    language : Languages, optional
        Programming language for the function. Default is Languages.Python.
    name : str, optional
        Name of the function or program to execute.
    parameters : dict | SerializableDictionary, optional
        Named arguments to pass to the function. Default is None.
            - In Stata, this would be like for "this=var_a" in do_something, this(var_a)
            - In python, it would be like do_something(this=var_a)
            - In SAS, %do_something(this=var_a)
            - In R, do_something(this=var_a)
    parameters_positional_pre : list, optional
        Positional arguments to place before named parameters in function call.
            - In Stata, this would be like for "var_a" in sum var_a or sum
                Set a value of "," where you want the comma, such as in
                ["var_a",",","detail"] for sum var_a, detail
            - In python, it would be like: sum(var_a)
            - In SAS, %sum(var_a)
            - In R, sum(var_a)
        Default is [].
    parameters_positional_post : list, optional
        Positional arguments to place after named parameters in function call.
            - In Stata, this would be like for "var_b" in do_something, this(var_a) var_b
            - In python, you can't pass a positional parameter after a named one, but it would try do_suming(this=var_a, var_b)
            - In SAS, %do_something(this=var_a, var_b) - like python, this will throw an error
            - In R, do_something(this=var_a, var_b)
        Default is [].
    pre_functions : list, optional
        List of code strings or Function objects to execute before main function.
        Default is [].
    post_functions : list, optional
        List of code strings or Function objects to execute after main function.
        Default is [].
    inputs : list, optional
        List of input file paths for dependency tracking. Default is [].
    outputs : list, optional
        List of output file paths for dependency tracking. Default is [].
    inputs_write_only : list, optional
        List of input files used only for writing (not reading). Default is [].
    on_complete : Callable, optional
        Function to call when execution completes successfully. Default is None.
    on_complete_args : dict, optional
        Arguments to pass to on_complete function. Default is {}.
    force : bool, optional
        force execution regardless of dependency status (i.e. even if the output data already exists). Default is False.
    load_parameters : bool, optional
        Whether to load NEWS environment parameters automatically. Default is True.
    load_utilities : bool, optional
        Whether to load language-specific utility functions. Default is True.
    call_input: CallInput, optional
        Memory and CPUs for call
        Default is None (use call_input setting from fu).
    parameters_as_is : bool | list, optional
        Parameters to pass as objects rather than strings. If True, all string
        parameters are passed as objects. If list, only specified parameters.
        The idea here is that if you set df = some data before, and you want to pass
        df to the function (not "df"), parameters_as_is=["df"] will do that
        Default is False.
    assign_to : str, optional
        Variable name to assign function result to in generated code.
        Default is "".
        Suppose assign_to is "df", this will change it df = do_something...
    shared_memory_items : dict, optional
        Key-value pairs of items to pass via shared memory for multiprocessing.
        Keys are names, values are data objects. Default is {}.
    in_main_block : bool, optional
        Whether to wrap Python code in 'if __name__ == "__main__":' block.
        Default is False.

    Attributes
    ----------
    language : Languages
        The programming language being used.
    name : str
        Function or program name.
    parameters : dict
        Named parameters for the function call.
    inputs : list
        Input file dependencies.
    outputs : list
        Output files that will be created.
    run : bool
        Whether this function should be executed.
    run_because : RunBecause
        Reason why the function is set to run.
    call_status : CallStatus
        Current execution status and metadata.
    parent_functions : list
        Functions that must complete before this one.

    Examples
    --------
    Python function call:

    >>> f = Function(
    ...     language=Languages.Python,
    ...     name="my_analysis_function",
    ...     parameters={"input_file": "data.csv", "threshold": 0.05},
    ...     inputs=["data.csv"],
    ...     outputs=["results.parquet"]
    ... )

    SAS procedure with pre/post code:

    >>> f = Function(
    ...     language=Languages.SAS,
    ...     name="StandardizeCPS",
    ...     parameters={"output_set": "Work.CPS2020", "CPSYear": 2020},
    ...     pre_functions=["%PUT 'Starting CPS processing';"],
    ...     post_functions=["%PUT 'CPS processing complete';"]
    ... )

    R script with resource requirements:

    >>> f = Function(
    ...     language=Languages.R,
    ...     name="statistical_model",
    ...     parameters={"data_path": "/data/analysis.csv"},
    ...     call_input=CallInputs(mem_in_mb=5000,
    ...                           n_cpu=4)
    ... )

    Bash script execution:

    >>> f = Function(
    ...     language=Languages.Bash,
    ...     name="/scripts/process_files.sh",
    ...     parameters={"INPUT_DIR": "/data", "OUTPUT_DIR": "/results"}
    ... )

    Notes
    -----
    - Functions are executed in isolated environments with appropriate language setup
    - Dependencies between functions are automatically resolved via inputs/outputs
    - Resource requirements can be specified per-function or globally
    - All languages have access to NEWS environment parameters and utilities
    """

    def __init__(
        self,
        language: Languages = None,
        name: str = "",
        parameters: dict | SerializableDictionary = None,
        parameters_positional_pre: list = None,
        parameters_positional_post: list = None,
        pre_functions: list[str | Function] = None,
        post_functions: list[str | Function] = None,
        inputs: list[str | PathWithFallbacks] = None,
        outputs: list[str | PathWithFallbacks] = None,
        inputs_write_only: list[str | PathWithFallbacks] = None,
        call_input: CallInputs | None = None,
        on_complete: Callable = None,
        on_complete_args: dict = None,
        force: bool = False,
        load_parameters: bool = True,
        load_utilities: bool = True,
        parameters_as_is: bool | list[str] = False,
        assign_to: str = "",
        shared_memory_items: dict = None,
        in_main_block: bool = False,
        tracker_args_to_keep: list[str] = None,
        tracker_args_to_drop: list[str] = None,
    ):
        # Handle None defaults for mutable types
        if parameters is None:
            parameters = {}

        if on_complete_args is None:
            on_complete_args = {}
        if shared_memory_items is None:
            shared_memory_items = {}

        # Positional parameters before/after the named ones
        self.parameters_positional_pre = list_input(parameters_positional_pre)
        self.parameters_positional_post = list_input(parameters_positional_post)

        # Initialize inputs and outputs
        self.inputs = list_input(inputs)
        self.outputs = list_input(outputs)
        self.inputs_write_only = list_input(inputs_write_only)

        # Set language default
        if language is None:
            language = Languages.Python

        # Assign basic attributes
        self.language = language
        self.name = name
        self.parameters = parameters

        # List of functions or lines of code to run first
        pre_functions = list_input(pre_functions)
        self.pre_functions = []
        for funci in pre_functions:
            self.add_pre_function(funci)

        # List of functions to run last (after postCode)
        self.post_functions = []
        post_functions = list_input(post_functions)
        for funci in post_functions:
            self.add_post_function(funci)

        # Initialize run state
        self.run = False
        self.run_because = RunBecause.Not
        self.force = force
        self.parent_functions = []
        self.child_functions = []

        # Initialize call status
        self.call_status = CallStatus(self.language)
        if call_input is None:
            call_input = CallInputs()
        self.call_status.call_input = call_input

        # Configuration flags
        self.load_parameters = load_parameters
        self.load_utilities = load_utilities

        # Completion callback
        self.on_complete = on_complete
        self.on_complete_args = on_complete_args
        self.on_complete_run = False

        # Resource allocation

        # Parameter handling configuration
        self.parameters_as_is = parameters_as_is

        # Variable assignment configuration
        self.assign_to = assign_to

        # Shared memory for multiprocessing
        self.shared_memory_items = shared_memory_items

        # Python-specific configuration
        self.in_main_block = in_main_block

        # Tracker configuration
        self.tracker_args_to_keep = list_input(tracker_args_to_keep)
        self.tracker_args_to_drop = list_input(tracker_args_to_drop)

        # Return module for in_process calls
        self.module = None

    @staticmethod
    def _hash_eq_attributes():
        return [
            "name",
            "language",
            "parameters",
            "parameters_positional_pre",
            "parameters_positional_post",
            "pre_functions",
            "post_functions",
        ]

    def __hash__(self):
        return SerializableList(self.call_code()).__hash__()

    def __eq__(self, other):
        if type(other) is not Function:
            return False
        else:
            for itemi in self._hash_eq_attributes():
                if getattr(self, itemi) is getattr(other, itemi):
                    #   Do nothing
                    pass
                else:
                    try:
                        if getattr(self, itemi) != getattr(other, itemi):
                            return False
                    except:
                        #   Can't be compared, assume they're different
                        return False

            return True

    def __str__(self):
        return (
            f"Function: {self.name}{LINEBREAK}"
            + f"language: {self.language}{LINEBREAK * 2}"
            + f"BEGIN CODE{LINEBREAK * 2}"
            + self.call_code(save_serialize_params=False)
            + LINEBREAK * 2
            + f"END CODE{LINEBREAK * 2}"
        )

    def call(
        self,
        path_temp="",
        NamePrefix="",
        NameSuffix="",
        testing=False,
        call_input: CallInputs | None = None,
    ):
        if call_input is None:
            call_input = CallInputs()

        if path_temp == "":
            #   Set a default path_temp because PBS won't work with none passed
            path_temp = Config().path_temp_files

        #   Get a random name for the file from the tempfile object
        filenameonly = (
            NamePrefix.replace("/", "_")
            + next(tempfile._get_candidate_names())
            + NameSuffix
            + "."
            + self.file_suffix
        )
        fPath = str(path_temp / filenameonly)

        #   Write the call file
        logsuffix = ".log"

        self.call_status.call_input = call_input
        self.call_status.callfile = fPath
        self.call_status.logfile = fPath.replace("." + self.file_suffix, logsuffix)
        self.call_status.logfile_pythonlogging = (
            self.call_status.logfile + "_logging.log"
        )

        if os.path.exists(self.call_status.logfile):
            os.remove(self.call_status.logfile)

        if self.language == Languages.Python:
            if os.path.exists(self.call_status.logfile_pythonlogging):
                os.remove(self.call_status.logfile_pythonlogging)

        code = self.call_code(logpath=self.call_status.logfile_pythonlogging)

        create_folders_if_needed(os.path.dirname(fPath))
        fCall = open(fPath, "w", encoding="utf-8")
        fCall.write(code)
        fCall.close()

        if testing:
            logger.info("TESTING FILE CREATED: " + fPath)

        if (
            self.language == Languages.Bash
            and self.call_status.call_input.call_type.value != CallTypes.PBS.value
        ):
            logger.info("Resetting bash call to PBS job")
            self.call_status.call_input.call_type = CallTypes.PBS

        if (
            self.call_status.call_input.call_type.value
            == CallTypes.multiprocessing.value
        ):
            if self.language == Languages.Python:
                mp_context = multiprocessing.get_context("spawn")

                #   Get the shared memory items and pass them to the subprocess
                memory_items = []

                try:
                    smm = SharedMemoryManager()
                    smm.start()
                    self.call_status.shared_memory_manager = smm
                    if len(self.shared_memory_items):
                        for keyi, valuei in self.shared_memory_items.items():
                            #   We're only passing polars Lazy/DataFrames and paths to load them
                            #       If it's something else, just make it an argument
                            if type(valuei) is str:
                                #   Confirm the file exists
                                if not os.path.isfile(valuei):
                                    sError = f"SharedMemoryItem {keyi}={valuei} is not a string, but not a file."
                                    logger.error(sError)
                                    raise Exception(sError)
                            elif (
                                type(valuei) is not pl.LazyFrame
                                and type(valuei) is not pl.DataFrame
                            ):
                                sError = f"SharedMemoryItem {keyi} is not a string, polars LazyFrame, or polars DataFrame."
                                logger.error(sError)
                                raise Exception(sError)

                            #   We're good, add it to the list of items getting passed
                            memory_items.append(
                                SharedMemoryUtility.df_to_arrow_shm(
                                    df=valuei, smm=smm, name=keyi
                                ).to_dict()
                            )
                except:
                    smm.shutdown()

                self.call_status.process = mp_context.Process(
                    target=Function.multiprocess_function,
                    args=(memory_items, self.call_status.logfile_pythonlogging, code),
                )

                self.call_status.process.start()
                self.call_status.start_time = time.time()
                #   Load the file and run
            else:
                logger.info(
                    "CallType multiprocessing is python only, converting to a shell process"
                )
                self.call_status.call_input.call_type = CallTypes.shell

        if (
            self.call_status.call_input.call_type.value == CallTypes.in_process.value
        ) and not testing:
            from .tracker import FunctionTracker

            if self.language == Languages.Python:
                #   Run the function
                logger.info(f"Run the in-process call for {self.name} - BEGIN")
                with run_with_temporary_logging():
                    self.call_status.start_time = time.time()
                    spec = importlib.util.spec_from_file_location(self.name, fPath)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.module = module
                self.call_status.end_time = time.time()
                FunctionTracker.save_inputs_for_function(
                    self=FunctionTracker, functioni=self
                )
                logger.info(f"                            {self.name} - COMPLETE")

            else:
                logger.info("In process jobs must be python, defaulting to shell jobs")
                self.call_status.call_input.call_type = CallTypes.shell

        if self.call_status.call_input.call_type.value == CallTypes.shell.value:
            if self.language == Languages.SAS:
                shellcommand = "sas " + fPath + " -log " + self.call_status.logfile
            elif self.language == Languages.Python:
                shellcommand = f"{sys.executable} " + fPath
            elif self.language == Languages.Stata:
                shellcommand = "stata-mp -q -b do " + fPath
            elif self.language == Languages.R:
                shellcommand = (
                    'R CMD BATCH --no-save --quiet "'
                    + fPath
                    + '" "'
                    + self.call_status.logfile
                    + '"'
                )

            self.call_status.start_time = time.time()

            if not testing:
                self.call_status.stdout_file = Path(f"{fPath}.stdout").as_posix()
                self.call_status.stderr_file = Path(f"{fPath}.stderr").as_posix()

                self.call_status.stdout_handle = open(
                    self.call_status.stdout_file, "w", encoding="utf-8"
                )
                self.call_status.stderr_handle = open(
                    self.call_status.stderr_file, "w", encoding="utf-8"
                )
                self.call_status.process = subprocess.Popen(
                    shellcommand,
                    stdout=self.call_status.stdout_handle,
                    stderr=self.call_status.stderr_handle,
                    text=True,
                    cwd=os.path.dirname(fPath),
                    shell=True,
                )

            self.call_status.start_time = time.time()
        elif self.call_status.call_input.call_type.value == CallTypes.PBS.value:
            if self.language == Languages.SAS:
                shellcommand = "qsas_news --sasprog=" + fPath
            elif self.language == Languages.Python:
                #   Different log file default for python qsub
                self.call_status.logfile = fPath + ".log"
                shellcommand = "qpy_news --programfile=" + fPath
                #   --q=testq
            elif self.language == Languages.Stata:
                shellcommand = "qstata_news --nologo --dofile=" + fPath
            elif self.language == Languages.R:
                shellcommand = (
                    "qR_news --program="
                    + fPath
                    + " --logfile="
                    + self.call_status.logfile
                    + "--quiet"
                )
            elif self.language == Languages.Bash:
                shellcommand = f"qsub {fPath}"

            #   Function-specific memory and CPUs?
            mem_in_mb = self.call_status.call_input.mem_in_mb
            n_cpu = self.call_status.call_input.n_cpu

            if self.language != Languages.Bash:
                shellcommand += (
                    " --cpucount=" + str(n_cpu) + " --memsize=" + str(mem_in_mb)
                )

            #   PBS Pro call
            self.call_status.start_time = time.time()
            if not testing:
                shellout = subprocess.run(
                    shellcommand.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                #   Set the information needed to check if the command is complete
                self.call_status.set_job_id(shellout.stdout)

    def call_code(self, logpath: str = "", save_serialize_params: bool = True):
        function_call = ""

        if self.language == Languages.Bash:
            #   Just load the bash script as the file

            bash_script = ""

            if len(self.parameters):
                for keyi, itemi in self.parameters.items():
                    if type(itemi) is str:
                        bash_script += f'{keyi}="{itemi}"' + LINEBREAK
                    else:
                        bash_script += f"{keyi}={itemi}" + LINEBREAK

            with open(self.name, "r", encoding="utf-8") as f:
                bash_script += f.read()

            return bash_script

        if self.language == Languages.Python and self.in_main_block:
            function_call += 'if __name__ == "__main__":' + LINEBREAK

        if self.load_parameters:
            CodeRoot = Config().code_root

            parameter_files = Config().parameter_files
            if str(self.language) in parameter_files:
                param_file = parameter_files[str(self.language)]
            else:
                param_file = None

            if param_file is not None:
                if self.language == Languages.SAS:
                    function_call += (
                        'FILENAME params "' + param_file + '" LRECL=10000;' + LINEBREAK
                    )
                    function_call += "%INCLUDE params;" + LINEBREAK
                elif self.language == Languages.Stata:
                    function_call += 'quietly do "' + param_file + '"' + LINEBREAK

                elif self.language == Languages.R:
                    function_call += 'source("' + param_file + '");' + LINEBREAK

            if self.language == Languages.SAS:
                #   Utilities always lo aded in SAS
                self.load_utilities = False

            elif self.language == Languages.Python:
                if (
                    self.call_status.call_input.call_type.value
                    != CallTypes.in_process.value
                ):
                    function_call += (
                        "from survey_kit.orchestration.config import Config" + LINEBREAK
                    )
                    function_call += (
                        f"Config().cpus = {self.call_status.call_input.n_cpu}"
                        + LINEBREAK
                    )

                function_call += (
                    "from survey_kit.utilities.logging import set_logging, PrintLogger"
                    + LINEBREAK
                )
                function_call += (
                    f"set_logging(path_log='{Path(logpath).as_posix()}',force=True,to_console=False,append_to_file=True,name='{Path(logpath).as_posix()}')"
                    + LINEBREAK
                )
                if (
                    self.call_status.call_input.call_type.value
                    != CallTypes.in_process.value
                ):
                    function_call += (
                        f"capture_prints_log = PrintLogger('{Path(logpath).as_posix()}',to_console=False)"
                        + LINEBREAK
                    )

        for funci in self.pre_functions:
            if type(funci) is str:
                function_call += funci + LINEBREAK
            elif type(funci) is Function:
                if self.load_parameters:
                    funci.load_parameters = False
                    if self.language == Languages.SAS:
                        funci.load_utilities = False

                if self.load_utilities:
                    funci.load_utilities = False

                function_call += (
                    funci.call_code(save_serialize_params=save_serialize_params)
                    + LINEBREAK
                )

        temp_dict = next(tempfile._get_candidate_names())
        path_dict_save = Path(Config().path_temp_files) / temp_dict

        if (
            self.language == Languages.Python
            and type(self.parameters) is SerializableDictionary
        ):
            #   Get a random name for the serialized dictionary
            if save_serialize_params:
                self.parameters.save(path_dict_save)

            function_call += (
                "from survey_kit.serializable import SerializableDictionary" + LINEBREAK
            )
            function_call += (
                f"__parameters__ = SerializableDictionary.load('{Path(path_dict_save).as_posix()}')"
                + LINEBREAK
            )

        elif type(self.parameters) is SerializableDictionary:
            logger.info(
                f"Serializing/saving parameters is not implemented for {self.language}, passing as a regular dictionary"
            )

        if self.language == Languages.SAS:
            if self.assign_to != "":
                function_call += "%LET " + self.assign_to + " = "

            function_call += "%" + self.name + "("

            bFirstParam = True
            for parami in self.parameters_positional_pre:
                if not bFirstParam:
                    function_call += ","

                function_call += str(parami)

                bFirstParam = False

            for keyi in self.parameters.keys():
                if not bFirstParam:
                    function_call += ","

                function_call += keyi + "=" + str(self.parameters[keyi])

                bFirstParam = False
            for parami in self.parameters_positional_post:
                if not bFirstParam:
                    function_call += ","

                function_call += str(parami)

                bFirstParam = False

            function_call += ");" + LINEBREAK
        elif self.language == Languages.Python or self.language == Languages.R:
            if self.language == Languages.R:
                Semicolon = ";"
            else:
                Semicolon = ""

            if self.assign_to != "":
                function_call += self.assign_to + " = "
            function_call += self.name + "("

            bFirstParam = True
            for parami in self.parameters_positional_pre:
                if not bFirstParam:
                    function_call += ","

                function_call += str(parami)

                bFirstParam = False

            if (
                self.language == Languages.Python
                and type(self.parameters) is SerializableDictionary
            ):
                if not bFirstParam:
                    function_call += ","
                function_call += "**__parameters__"

            #   Add the parameter items, if needed
            if not (
                self.language == Languages.Python
                and type(self.parameters) is SerializableDictionary
            ):
                for keyi in self.parameters.keys():
                    if not bFirstParam:
                        function_call += ","

                    function_call += (
                        keyi
                        + "="
                        + self.function_call_r_python_to_output(self.parameters[keyi])
                    )

                    bFirstParam = False
            for parami in self.parameters_positional_post:
                if not bFirstParam:
                    function_call += ","

                function_call += str(parami)

                bFirstParam = False

            function_call += ")" + Semicolon + LINEBREAK
        elif self.language == Languages.Stata:
            if self.assign_to != "":
                function_call += "local " + self.assign_to + " = "
            function_call += self.name + " "

            if len(self.parameters_positional_pre) == 0:
                self.parameters_positional_pre = [","]
            for parami in self.parameters_positional_pre:
                function_call += str(parami) + " "

            for keyi in self.parameters.keys():
                function_call += keyi + "(" + str(self.parameters[keyi]) + ") "

            for parami in self.parameters_positional_post:
                function_call += str(parami) + " "

            function_call += LINEBREAK

        for funci in self.post_functions:
            if type(funci) is str:
                function_call += funci + LINEBREAK
            elif type(funci) is Function:
                if self.load_parameters:
                    funci.load_parameters = False
                    if self.language == Languages.SAS:
                        funci.load_utilities = False

                if self.load_utilities:
                    funci.load_utilities = False

                function_call += (
                    funci.call_code(save_serialize_params=save_serialize_params)
                    + LINEBREAK
                )

        if (
            self.language == Languages.Python
            and type(self.parameters) is SerializableDictionary
        ):
            #   Clean up the serialized parameter file
            function_call += (
                f"SerializableDictionary.delete('{Path(path_dict_save).as_posix()}')"
                + LINEBREAK
            )

        #   Close the log, if needed
        if logpath != "" and (self.language == Languages.Python):
            function_call += "print('',flush=True)" + LINEBREAK
            if (
                self.call_status.call_input.call_type.value
                != CallTypes.in_process.value
            ):
                function_call += "capture_prints_log.close()" + LINEBREAK

        if self.language == Languages.Python and self.in_main_block:
            function_call = function_call.replace(LINEBREAK, f"{LINEBREAK}\t")
        return function_call

    def function_call_r_python_to_output(self, Item=None):
        output = ""

        if self.language == Languages.R:
            if type(Item) is PathWithFallbacks:
                Item = as_output_path(Item)

            if type(Item) is str:
                asis = False
                if type(self.parameters_as_is) is bool:
                    asis = self.parameters_as_is
                elif type(self.parameters_as_is) is list:
                    asis = Item in self.parameters_as_is

                if asis:
                    output += Item
                else:
                    output += "'" + Item + "'"
            # elif (type(Item) is PathWithFallbacks):
            #     output += ConvertToConstructor(Item,
            #                                    FromInit=True)
            elif type(Item) is bool:
                if Item:
                    output += "TRUE"
                else:
                    output += "FALSE"
            elif type(Item) is list:
                if len(Item) == 0:
                    output = "c()"
                else:
                    bFirst = True
                    for itemi in Item:
                        if bFirst:
                            output += "c("
                        else:
                            output += ","

                        output += self.function_call_r_python_to_output(itemi)
                        bFirst = False

                    output += ")"
            elif type(Item) is dict:
                output += "list("

                bFirst = True
                for subkeyi in Item:
                    itemi = Item[subkeyi]

                    if not bFirst:
                        output += ","

                    output += "'" + subkeyi + "'="

                    output += self.function_call_r_python_to_output(itemi)

                    bFirst = False
                output += ")"
            else:
                output += str(Item)
        else:
            if type(Item) is str:
                asis = False
                if type(self.parameters_as_is) is bool:
                    asis = self.parameters_as_is
                elif type(self.parameters_as_is) is list:
                    asis = Item in self.parameters_as_is

                if asis:
                    output += Item
                else:
                    output += "'" + Item + "'"
            elif type(Item) is PathWithFallbacks:
                output += convert_to_constructor(Item, from_init=True)
            elif callable(Item):
                #   Function
                output += Item.__name__
            elif type(Item) is list:
                bFirst = True

                if len(Item) == 0:
                    output = "[]"
                else:
                    for itemi in Item:
                        if bFirst:
                            output += "["
                        else:
                            output += ","

                        output += self.function_call_r_python_to_output(itemi)
                        bFirst = False

                    output += "]"
            elif type(Item) is dict:
                output += "{"

                bFirst = True
                for subkeyi in Item:
                    itemi = Item[subkeyi]

                    if not bFirst:
                        output += ","

                    output += "'" + subkeyi + "':"
                    output += self.function_call_r_python_to_output(itemi)

                    bFirst = False
                output += "}"
            else:
                output += str(Item)

        return output

    def code_as_bash(self, path_code: str, name: str = "", mail_to: str = "") -> str:
        code = "#!/bin/bash" + Function.LINEBREAK
        if name != "":
            code += f"#PBS -N {name}" + Function.LINEBREAK
        if mail_to != "":
            code += f"#PBS -M {mail_to}" + Function.LINEBREAK

        cpus = max(self.call_status.call_input.n_cpu, 1)

        code += (
            f"#PBS -l ncpus={cpus}:mem={self.call_status.call_input.mem_in_mb}mb"
            + Function.LINEBREAK
        )
        code += "#PBS -j oe" + Function.LINEBREAK

        # code += f"source {Config.code_root}/NEWSConfig.bash" + Function.LINEBREAK

        if self.language.value == Languages.Python.value:
            code += f"python {path_code} > {path_code}.log 2>&1" + Function.LINEBREAK
        elif self.language.value == Languages.R.value:
            code += f"cd {os.path.dirname(path_code)}" + Function.LINEBREAK
            code += (
                f"R CMD BATCH --quiet --no-save {os.path.basename(path_code)}"
                + Function.LINEBREAK
            )
        elif self.language.value == Languages.Stata.value:
            message = "Stata code as bash not implemented yet"
            logger.error(message)
            raise Exception(message)
        elif self.language.value == Languages.SAS.value:
            message = "SAS code as bash not implemented yet"
            logger.error(message)
            raise Exception(message)

        return code

    @property
    def file_suffix(self):
        if self.language is Languages.SAS:
            return "sas"
        elif self.language == Languages.Python:
            return "py"
        elif self.language == Languages.Stata:
            return "do"
        elif self.language == Languages.R:
            return "R"
        elif self.language == Languages.Bash:
            return "bash"

    def multiprocess_function(
        shm_memory_items: list[dict], logpath: str = "", code: str = ""
    ):
        from .shared_memory import SharedMemoryUtility

        df_memory_items = SharedMemoryUtility.arrow_shm_list_to_dict_df(
            shm_memory_items
        )
        del shm_memory_items

        #   I know this is not great, but it's the easiest way...
        exec(code)

    def add_pre_function(self, func=None):
        if type(func) is list:
            #   Pass each line of the list through this function
            for itemi in func:
                self.add_pre_function(func=itemi)
        elif type(func) is str:
            self.pre_functions.append(func)
        elif type(func) is Function:
            if self.language != func.language:
                raise Exception(
                    "Pre/Post functions must be in same language as function"
                )

            self.pre_functions.append(func)

    def add_post_function(self, func=None):
        if type(func) is list:
            #   Pass each line of the list through this function
            for itemi in func:
                self.add_post_function(func=itemi)
        elif type(func) is str:
            self.post_functions.append(func)
        elif type(func) is Function:
            if self.language != func.language:
                raise Exception(
                    "Pre/Post functions must be in same language as function"
                )

            self.post_functions.append(func)

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, force: bool):
        self._force = force
        self.run = self.run or force

    def convert_python_object_to_r(obj=None):
        f = Function(language=Languages.R)
        return f.function_call_r_python_to_output(Item=obj)

    @property
    def full_log(self):
        #   Done?, then Get the results of the call and update that one changed status
        if self.call_status.complete:
            self.call_status.get_output()
            fullresults = ""
            fullresults += "\nLOGGING INFORMATION FOR:\n"
            fullresults += "    FILE:       " + self.call_status.callfile + "\n"
            if self.call_status.call_input.call_type.value == CallTypes.PBS.value:
                fullresults += "    PBS Job:    " + str(self.call_status.job_id) + "\n"
            if self.call_status.stdout is not None and self.call_status.stdout != "":
                fullresults += "SHELL OUT BEGIN\n"
                fullresults += self.call_status.stdout
                fullresults += "SHELL OUT END\n\n"

            if self.call_status.stderr is not None and self.call_status.stderr != "":
                fullresults += "SHELL LOG/ERROR BEGIN\n"
                fullresults += self.call_status.stderr
                fullresults += "SHELL LOG/ERROR END\n\n"

            if self.call_status.log is not None and self.call_status.log != "":
                fullresults += "PRINT OUTPUT BEGIN\n"
                fullresults += self.call_status.log
                fullresults += "PRINT OUTPUT END\n\n"

            self.call_status._full_log = fullresults
        elif self.run:
            fullresults += "\n\n" + "BEGIN CALL FOR " + self.name + "\n"
            fullresults += self.call_code(save_serialize_params=False)
            fullresults += "END CALL\n\n"

            fullresults += "*********************************************\n"
            fullresults += "*********************************************\n"
            fullresults += "       THIS FUNCTION WAS NOT COMPLETED\n"
            fullresults += "*********************************************\n"
            fullresults += "*********************************************\n\n"

            return fullresults

        return self.call_status._full_log
