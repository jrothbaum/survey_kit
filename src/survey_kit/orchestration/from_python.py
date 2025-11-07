from __future__ import annotations
from typing import Callable

import os
from pathlib import Path
from .utilities import Languages, CallInputs, LINEBREAK
from .function import Function
from ..serializable import SerializableDictionary
from ..utilities.inputs import list_input


#   Convert any random python function into a Function class
#       to be called asynchronously
def FunctionFromPython(
    function: Callable,
    namespace: str = "",
    load_from_file: str = "",
    parameters: dict | None | SerializableDictionary = None,
    inputs_parameters: list | None = None,
    outputs_parameters: list | None = None,
    inputs: list | None = None,
    outputs: list | None = None,
    call_input: CallInputs | None = None,
    parameters_as_is: bool | list = False,
    import_all_in_namespace: bool = False,
    shared_memory_items: dict | None = None,
    on_complete: Callable | None = None,
    on_complete_args: dict | None = None,
    assign_to: str = "",
) -> Function:
    """
    Convert a Python function into a Function object for parallel execution.

    This function wraps any Python callable into a Function object that can
    be executed asynchronously in separate processes, shell commands, or PBS jobs.
    The resulting Function is self-contained and includes all necessary imports
    and parameter handling.

    Parameters
    ----------
    function : callable
        The Python function to be wrapped for parallel execution.
    namespace : str, optional
        Module namespace for importing the function. If empty, attempts to
        auto-detect from function.__module__. Default is "".
    load_from_file : str, optional
        Path to file containing the function. Overrides namespace if provided.
        Uses .utilities.load_utility for loading. Default is "".
    parameters : dict | SerializableDictionary | None, optional
        Keyword arguments to pass to the function. Default is None.
    inputs_parameters : list | None, optional
        Parameter names whose values should be added to the Function's inputs
        for dependency tracking. Default is None.
    outputs_parameters : list | None, optional
        Parameter names whose values should be added to the Function's outputs
        for dependency tracking. Default is None.
    inputs : list | None, optional
        Additional file inputs for dependency tracking. Default is None.
    outputs : list | None, optional
        Additional file outputs for dependency tracking. Default is None.
    call_input : CallInputs | None, optional
        Execution configuration (PBS, shell, etc.). Default is None.
    parameters_as_is : bool | list, optional
        Parameters to pass as objects rather than strings. If True, all
        string parameters are passed as objects. If list, only specified
        parameters. Default is False.
    import_all_in_namespace : bool, optional
        Use "from namespace import *" instead of importing specific function.
        Default is False.
    shared_memory_items : dict | None, optional
        Items to pass via shared memory for multiprocessing calls.
        Default is None.
    on_complete : callable | None, optional
        Function to call when execution completes. Default is None.
    on_complete_args : dict | None, optional
        Arguments for on_complete function. Default is None.
    assign_to : str, optional
        Variable name to assign function result to. Default is "".

    Returns
    -------
    Function
        Function object ready for parallel execution.

    Examples
    --------
    Basic function wrapping:

    >>> def my_analysis(data_path, output_path, param1=10):
    ...     # Function implementation
    ...     pass
    >>>
    >>> f = FunctionFromPython(
    ...     function=my_analysis,
    ...     parameters={"data_path": "/data/input.csv",
    ...                "output_path": "/data/output.csv",
    ...                "param1": 20},
    ...     inputs_parameters=["data_path"],
    ...     outputs_parameters=["output_path"]
    ... )

    Loading from external file:

    >>> f = FunctionFromPython(
    ...     function=external_function,
    ...     load_from_file="/path/to/my_module.py",
    ...     parameters={"n_rows": 1000}
    ... )

    With high-memory PBS execution:

    >>> f = FunctionFromPython(
    ...     function=memory_intensive_task,
    ...     parameters={"large_dataset": "data.parquet"},
    ...     call_input=CallInputs(call_type=CallTypes.PBS, mem_in_mb=100000)
    ... )

    Notes
    -----
    - Functions must be self-contained or have all dependencies importable
    - File paths in inputs/outputs enable automatic dependency resolution
    - SerializableDictionary parameters are automatically handled for complex objects
    - shared_memory_items are only used with multiprocessing execution
    """
    if parameters is None:
        parameters = {}

    inputs_parameters = list_input(inputs_parameters)
    outputs_parameters = list_input(outputs_parameters)

    inputs = list_input(inputs)
    outputs = list_input(outputs)

    def ParseParameters(findlist: list = None, indict: dict = None):
        if findlist is None:
            findlist = []
        if indict is None:
            indict = {}

        present = set(findlist).intersection(indict.keys())

        listout = []
        for parami in present:
            value = indict[parami]

            if type(value) is list:
                listout.extend(value)
            else:
                listout.append(value)

        return listout

    if type(parameters) is SerializableDictionary:
        parse_check = parameters._d
    else:
        parse_check = parameters

    inputs.extend(ParseParameters(findlist=inputs_parameters, indict=parse_check))
    outputs.extend(ParseParameters(findlist=outputs_parameters, indict=parse_check))

    namespaceimport = ""

    if load_from_file != "":
        #   Use SetupUtils.parameters to load the function
        load_folder = os.path.dirname(load_from_file)
        load_filename = os.path.basename(load_from_file)

        if load_filename.endswith(".py"):
            load_filename = load_filename[0 : len(load_filename) - 3]

        name = function.__name__
        namespaceimport = (
            "from survey_kit.orchestration.utilities import load_utility" + LINEBREAK
        )
        namespaceimport += f"{name} = load_utility(folder='{Path(load_folder).as_posix()}',file='{load_filename}',module_only=True).{name}"
    else:
        if namespace == "":
            namespace = function.__module__

        if namespace != "__main__":
            if import_all_in_namespace:
                namespaceimport = "from " + namespace + " import *"
            else:
                namespaceimport = "from " + namespace + " import " + function.__name__

    extra_args = {}

    if hasattr(function, "args_to_drop"):
        extra_args["tracker_args_to_drop"] = function.args_to_drop

    if hasattr(function, "args_to_keep"):
        extra_args["tracker_args_to_keep"] = function.args_to_keep

    f = Function(
        language=Languages.Python,
        name=function.__name__,
        pre_functions=[namespaceimport],
        parameters=parameters,
        inputs=inputs,
        outputs=outputs,
        shared_memory_items=shared_memory_items,
        parameters_as_is=parameters_as_is,
        on_complete=on_complete,
        on_complete_args=on_complete_args,
        assign_to=assign_to,
        **extra_args,
    )

    if call_input is not None:
        f.call_status.call_input = call_input
    return f
