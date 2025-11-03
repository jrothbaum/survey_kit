from __future__ import annotations
from typing import TYPE_CHECKING

from copy import deepcopy


from ..utilities.logging import set_logging
from ..utilities.inputs import list_input
from .utilities import CallInputs, UpdateParams

from .need_to_run import RunBecause
from .tracker import FunctionTracker
from .dependency_order import FunctionDependencyOrder

from .. import logger

if TYPE_CHECKING:
    from .function import Function


def run_function_list(
    function_list: list[Function] | Function,
    path_temp: str = "",
    call_input: CallInputs = None,
    update: UpdateParams = None,
    testing: bool = False,
    run_all: bool = False,
    use_function_call_inputs: bool = False,
    no_print_log: bool = False,
    # reset_log:bool=False,
    on_complete_by_file=None,
    on_complete_by_file_params: dict | None = None,
    function_check_every=None,
    params_check_every: dict | None = None,
):
    """
    Execute a list of functions with automatic dependency resolution and parallel execution.

    This is the main entry point for executing multiple functions in parallel
    while respecting dependencies, resource constraints, and update requirements.
    It handles the complete workflow from dependency analysis to execution monitoring.

    Parameters
    ----------
    function_list : list[Function] | Function
        Function objects to execute.
    path_temp : str, optional
        Directory for temporary execution files. Default is "" (/projects/Data/NEWS/TempFiles).
        You can find the logs of function calls during execution in this path
    call_input : CallInputs, optional
        Default execution configuration for all functions.
        Default creates new CallInputs.
    update : UpdateParams, optional
        Configuration for determining which functions need execution.
        Default creates new UpdateParams.
    testing : bool, optional
        If True, creates execution files but doesn't run them.
        Useful for debugging. Default is False.
    run_all : bool, optional
        If True, forces all functions to run regardless of dependencies.
        Default is False.
    use_function_call_inputs : bool, optional
        Whether to use function-specific CallInputs instead of global.
        Default is False.
    no_print_log : bool, optional
        Whether to suppress detailed progress logging. Default is False.
    reset_log : bool, optional
        Whether to reset logging configuration before execution.
        Default is False.
    on_complete_by_file : callable, optional
        Function to call when specific files are created. Default is None.
    on_complete_by_file_params : dict, optional
        File paths and parameters for on_complete_by_file. Default is None.
    function_check_every : callable, optional
        Function to call periodically during execution. Default is None.
    params_check_every : dict, optional
        Parameters for function_check_every. Default is None.

    Returns
    -------
    str
        Combined execution log from all completed functions.

    Examples
    --------
    Basic parallel execution:

    >>> functions = [func1, func2, func3]
    >>> log = run_function_list(
    ...     function_list=functions,
    ...     call_input=CallInputs(call_type=CallTypes.shell)
    ... )

    High-performance cluster execution:

    >>> log = run_function_list(
    ...     function_list=functions,
    ...     call_input=CallInputs(call_type=CallTypes.PBS, mem_in_mb=100000),
    ...     update=UpdateParams(update_by_date=True)
    ... )

    Testing mode for debugging:

    >>> log = run_function_list(
    ...     function_list=functions,
    ...     testing=True
    ... )

    Force execution of all functions:

    >>> log = run_function_list(
    ...     function_list=functions,
    ...     run_all=True,
    ...     no_print_log=True
    ... )

    Notes
    -----
    - Automatically resolves dependencies between functions based on inputs/outputs
    - Supports mixed execution types (some functions via PBS, others via shell)
    - Progress is shown with dots (.) indicating status checks.  If there are no new dots, the process is dead
    - Resource limits prevent system overload during parallel execution

    The execution process:
    1. Analyze dependencies and determine execution order
    2. Check which functions need to run based on update criteria
    3. Start functions in parallel as dependencies allow
    4. Monitor progress and collect results
    5. Return combined execution log
    """

    function_list = list_input(function_list)

    if update is None:
        update = UpdateParams()

    if call_input is None:
        call_input = CallInputs()

    #   Initialize function tracker class
    #       Tracks jobs running, completed, and pending
    function_tracker = FunctionTracker(
        update=update,
        call_input=call_input,
        use_function_call_inputs=use_function_call_inputs,
        path_temp=path_temp,
        no_print_log=no_print_log,
        on_complete_by_file=on_complete_by_file,
        on_complete_by_file_params=on_complete_by_file_params,
    )

    #   Get dependencies and order functions
    if run_all:
        for functioni in function_list:
            functioni.force = True
            functioni.run_because = RunBecause.SetToRun

    #   Deduplicate functions but keep in original order, more or less
    function_list = list(dict.fromkeys(function_list))
    #   Initialize dependency ordering class (more like a set of functions)
    #       This takes in a list of functions and UpdateParams
    #       and orders the functions into groups that can be run in batches
    function_ordering = FunctionDependencyOrder(
        function_list=function_list, update=update
    )

    #   Run the functions and track, returning the composite log
    return function_tracker.run_list(
        function_ordering=function_ordering,
        testing=testing,
        #    reset_log=reset_log,
        function_check_every=function_check_every,
        params_check_every=params_check_every,
    )


# #   Call an arbitrary function in a loop (shell or PBS call)
# #       See acs and cps asec multiyear loaders for example calls
# def looped_function_call(#    Function to be called
#                        Delegate=None,
#                        #    Key-value pairs to pass to function that
#                        #        do not change with each loop
#                        StableParameters:dict=None,
#                        #    Key-list pairs that do change with each loop
#                        #        Each list should be the same length
#                        LoopParameters:dict=None,
#                        testing:bool=False,
#                        SavePathListDict:dict=None,
#                        ReturnData:bool=False,
#                        partitionBy:list=None,
#                        SequentialCheckBy:list=None,
#                        downcast:bool=True,
#                        downcast_strToNum:bool=False,
#                        call_input:CallInputs=None,
#                        CollectFiles_CallInput:CallInputs=None,
#                        TemporarySave:bool=False):

#     if StableParameters is None:
#         StableParameters = {}

#     if SavePathListDict is None:
#         SavePathListDict = {}

#     if (CollectFiles_CallInput is None):
#         CollectFiles_CallInput = CallInputs()
#     if LoopParameters is None:
#         raise ValueError("Must pass a LoopParameters dictionary")


#     #   No partition if temporary
#     if (TemporarySave):
#         partitionBy = []

#     if (partitionBy is None):
#         partitionBy = []


#     calllist = []

#     for key,value in LoopParameters.items():
#         callindex = 0
#         for itemi in value:
#             if len(calllist) <= callindex:
#                 calllist.append(deepcopy(StableParameters))

#             calllist[callindex][key] = itemi
#             callindex = callindex + 1
#     logger.info("ASSIGNING FUNCTIONS")
#     function_list = []
#     for itemi in calllist:
#         if ("loadutils" in itemi.keys()):
#             itemi["loadutils"].run = False
#         else:
#             itemi["loadutils"] = FileLoaderUtilities.FileLoaderUtilities(run=False)

#         function_list.append(Delegate(**itemi))


#     log = run_function_list(function_list=function_list,
#                           run_all=True,
#                           call_input=call_input,
#                           testing=testing)


#     #   List for output (if needed)
#     df = []

#     #   List of functions for async/PBS call, if needed
#     fFuncList = []
#     if (not testing and (ReturnData or len(SavePathListDict) > 0)):
#         for indexi in range(0, len(SavePathListDict)):
#             if (type(SavePathListDict) is dict):
#                 SavePath = list(SavePathListDict)[indexi]
#                 SaveList = SavePathListDict[SavePath]

#             elif (type(SavePathListDict) is list):
#                 SavePath = ""
#                 SaveList = SavePathListDict[indexi]


#             if ((CollectFiles_CallInput.call_type.value == CallTypes.shell.value)
#                     or (CollectFiles_CallInput.call_type.value == CallTypes.multiprocessing.value)):
#                 #   Just run as is in this thread/job
#                 if (SavePath != "" and SavePath is not None):
#                     logger.info("Collecting files and saving at:")
#                     logger.info("     " + SavePath)
#                 else:
#                     logger.info("Collecting files and loading")


#                 if (len(SavePathListDict) == 1):
#                     df = _looped_function_call_append_file(SavePath=SavePath,
#                                                         SaveList=SaveList,
#                                                         partitionBy=partitionBy,
#                                                         SequentialCheckBy=SequentialCheckBy,
#                                                         downcast=downcast,
#                                                         downcast_strToNum=downcast_strToNum,
#                                                         TemporarySave=TemporarySave)

#                 else:
#                     df.append(_looped_function_call_append_file(SavePath=SavePath,
#                                                              SaveList=SaveList,
#                                                              partitionBy=partitionBy,
#                                                              SequentialCheckBy=SequentialCheckBy,
#                                                              downcast=downcast,
#                                                              downcast_strToNum=downcast_strToNum,
#                                                              TemporarySave=TemporarySave))
#             elif (CollectFiles_CallInput.call_type.value==CallTypes.PBS.value):
#                 #   Create the PBS pro job for it
#                 fFuncList.append(Function(language=Languages.Python,
#                                           pre_functions=["from NEWS.CodeUtilities.Python.Function.Callers import _looped_function_call_append_file"],
#                                           name="_looped_function_call_append_file",
#                                           parameters={"SavePath":SavePath,
#                                                       "SaveList":SaveList,
#                                                       "partitionBy":partitionBy,
#                                                       "SequentialCheckBy":SequentialCheckBy,
#                                                       "downcast":downcast,
#                                                       "downcast_strToNum":downcast_strToNum}
#                                           )
#                                  )
#         if (CollectFiles_CallInput.call_type.value==CallTypes.PBS.value):
#             #   Async call, run the call and get the results
#             logAppend = run_function_list(function_list=fFuncList,
#                                         CallInput=CollectFiles_CallInput,
#                                         run_all=True)


#             log += logAppend

#             if (ReturnData):
#                 #   Collect the files
#                 logger.info("Loading files ")
#                 for SavePath,SaveList in SavePathListDict.items():
#                     if (len(SavePathListDict) == 1):
#                         df = ReadParquet.SafeCollect(ReadParquet.readParquet(parquetFullPath=SavePath))
#                     else:
#                         df.append(ReadParquet.SafeCollect(ReadParquet.readParquet(parquetFullPath=SavePath)))


#                     if (TemporarySave):
#                         ReadParquet.DeleteParquet(SavePath)
#             else:
#                 df = None
#         if ReturnData:
#             return [log, function_list, calllist, df]
#         else:
#             return [log, function_list, calllist]
#     else:
#         return [log, function_list, calllist]

# def _looped_function_call_append_file(SavePath:str="",
#                                    SaveList:list=None,
#                                    partitionBy:list=None,
#                                    SequentialCheckBy:list=None,
#                                    downcast:bool=True,
#                                    downcast_strToNum:bool=False,
#                                    TemporarySave:bool=False):
#     if (partitionBy is None):
#         partitionBy = []
#     if (SaveList is None):
#         SaveList = []

#     logger.info("Collecting files and saving at:")
#     logger.info("     " + SavePath)
#     df = JoinParquet.AppendList(dflist=SaveList,
#                                 parquetFolderPath=os.path.dirname(SavePath),
#                                 parquetFileName=os.path.basename(SavePath),
#                                 partitionBy=partitionBy,
#                                 downcast=downcast,
#                                 downcast_strToNum=downcast_strToNum,
#                                 SequentialCheckBy=SequentialCheckBy)

#     if SavePath != "":
#         df = ReadParquet.readParquet(SavePath)
#     for filei in SaveList:
#         ReadParquet.DeleteParquet(filei)

#     if (TemporarySave):
#         df = ReadParquet.SafeCollect(df)
#         ReadParquet.DeleteParquet(SavePath)

#     return df


#   Pass in a list of functions and either
#       Call the ones that need to be run (run=True)
#       Call all of them (force_run_all=True)
#       Return the functions in the order in which they can be called
#           run=False, force_run_all=False, return_ordering=True
#           This will also list the functions in their groups
#               All of them or only the ones that need to run
#               depending on show_only_functions_set_to_run
#       Do nothing (all of them false)
def function_call_or_list(
    function_list: list = None,
    # Run the necessary loaders or not?
    #     If run = False, return the function list
    run: bool = True,
    #     If running, run everything?
    force_run_all: bool = False,
    #     return_ordering, if run=False,
    return_ordering: bool = False,
    # Show the log?
    show_log: bool = False,
    short_log: bool = False,
    #     If running only, save the log?
    log_file: str = "",
    #     Log - show only functions set to run (if not running)
    show_only_functions_set_to_run: bool = True,
    #     Determines when to update each extract
    # Defaults, update if
    #     1) Set of inputs have changed
    #     2) Newer input than output
    update: UpdateParams = None,
    testing: bool = False,
):
    function_list = list_input(function_list)

    #   Default update checks
    if update is None:
        update = UpdateParams(update_by_used_file_list=True, update_by_date=True)

    if run:
        set_logging(to_console=show_log, path_log=log_file)
        log = run_function_list(
            function_list=function_list,
            update=update,
            run_all=force_run_all,
            testing=testing,
            use_function_call_inputs=True,
        )
        return log

    else:
        if return_ordering:
            functionordering = FunctionDependencyOrder(
                function_list=function_list, update=update
            )

            if show_log:
                logger.info("\n\n\n\n")
                groupNumber = 0
                for groupi in [
                    functionordering.run_initial,
                    functionordering.run_on_parent_complete,
                ]:
                    groupNumber = groupNumber + 1
                    if groupNumber == 1:
                        logger.info("Functions ready to start")
                    else:
                        logger.info("Functions that must wait")

                    n_in_group = len(groupi)
                    run_count = 0
                    for index, functioni in enumerate(groupi):
                        if functioni.run or not show_only_functions_set_to_run:
                            if show_only_functions_set_to_run:
                                run_count += 1

                                logger.info(
                                    f"{index + 1} (of {n_in_group}): #{run_count} to run"
                                )
                            else:
                                logger.info(f"{index + 1}")

                            if short_log:
                                logger.info(
                                    "NAME:         "
                                    + functioni.name
                                    + ": "
                                    + str(functioni.parameters)
                                )
                                logger.info(
                                    "CALL PARAMS:  "
                                    + str(functioni.call_status.call_input)
                                )
                                logger.info(str(functioni.run_because))
                            else:
                                logger.info(
                                    "NAME:         "
                                    + functioni.name
                                    + ": "
                                    + str(functioni.parameters)
                                )
                                logger.info("RUN:          " + str(functioni.run))
                                logger.info("INPUTS:       " + str(functioni.inputs))
                                logger.info("OUTPUTS:      " + str(functioni.outputs))
                                logger.info(
                                    "CALL PARAMS:  "
                                    + str(functioni.call_status.call_input)
                                )
                                logger.info(
                                    "CODE:      \n"
                                    + str(
                                        functioni.call_code(save_serialize_params=False)
                                    )
                                )
                                logger.info(str(functioni.run_because))
                            # logger.info("call_inputs:   " + str(functioni.call_status.call_input.call_type))
                            # logger.info("              " + str(functioni.call_status.call_input.mem_in_mb))
                            # logger.info("              " + str(functioni.call_status.call_input.n_cpu))
                            logger.info("\n")
                    logger.info("\n\n")

            return functionordering
        else:
            #   Deduplicate functions but keep in original order, more or less
            function_list = list(dict.fromkeys(function_list))

            return function_list
