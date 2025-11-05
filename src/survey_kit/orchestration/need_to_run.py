from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from enum import Enum
import os
import sys
import polars as pl

from .utilities import UpdateParams
from .path_with_fallbacks import PathWithFallbacks
from ..serializable import SerializableDictionary

from .. import logger

if TYPE_CHECKING:
    from .function import Function


class RunBecause(Enum):
    Not = -1
    SetToRun = 0
    OutputDoesNotExist = 1
    UpdateByOutput = 2
    UpdateByFunctionName = 3
    UpdateByDate = 4
    UpdateByInputFileList = 5
    ParentSetToRun = 6
    UpdateByParameters = 7


def update_need_to_run(function: Function, update: UpdateParams) -> None:
    rerun_status = RerunStatus(False, RunBecause.Not)

    if not update.update_run:
        return [True, RunBecause.SetToRun]

    #   Not already set to rerun, check if it needs updating from the function name
    _check_if_run_by_function_name(
        function.name, update.update_by_function_name, rerun_status
    )

    for outputi in function.outputs:
        #   Check if output already exists
        _check_if_output_exists(outputi, rerun_status)

        #   Check if output is in list to be updated
        _check_if_output_listed_for_update(
            outputi,
            outputs_listed_for_update=update.update_by_output,
            rerun_status=rerun_status,
        )

        #   Check by output, if
        #       1. The list of inputs has changed
        #       2. if update.UpdateByDate, is any input is newer than any output
        _check_inputs_for_output(
            path_output=outputi,
            function=function,
            by_date=update.update_by_date,
            rerun_status=rerun_status,
        )

    function.run = rerun_status.run_needed
    function.run_because = rerun_status.run_because

    if function.run:
        set_output_paths_to_newest_on_run(function)


def set_output_paths_to_newest_on_run(function: Function) -> None:
    for i in range(0, len(function.outputs)):
        if type(function.outputs[i]) is PathWithFallbacks:
            function.outputs[i] = function.outputs[i].path_main


class RerunStatus:
    def __init__(self, run_needed: bool, run_because: RunBecause):
        self.run_needed = run_needed
        self.run_because = run_because


def _check_if_run_by_function_name(
    name: str, names_to_rerun: list[str] | None, rerun_status: RerunStatus
) -> None:
    if not rerun_status.run_needed:
        if names_to_rerun is not None and len(names_to_rerun):
            rerun_status.run_needed = name in names_to_rerun
            if rerun_status.run_needed:
                rerun_status.run_because = RunBecause.UpdateByFunctionName


def _check_if_output_exists(
    path_output: str | PathWithFallbacks, rerun_status: RerunStatus
) -> None:
    if not rerun_status.run_needed:
        if type(path_output) is PathWithFallbacks:
            path_output = path_output.resolve_path().path

        rerun_status.run_needed = not (
            os.path.isfile(path_output) or (os.path.exists(path_output))
        )

        if rerun_status.run_needed:
            rerun_status.run_because = RunBecause.OutputDoesNotExist


def _check_if_output_listed_for_update(
    path_output: str | PathWithFallbacks,
    outputs_listed_for_update: list[str | PathWithFallbacks],
    rerun_status: RerunStatus,
) -> None:
    if not rerun_status.run_needed:
        if outputs_listed_for_update is not None and len(outputs_listed_for_update):
            if type(path_output) is PathWithFallbacks:
                path_output = path_output.path_main

            rerun_status.run_needed = path_output in outputs_listed_for_update

            if rerun_status.run_needed:
                rerun_status.run_because = RunBecause.UpdateByOutput


def _check_inputs_for_output(
    path_output: str | PathWithFallbacks,
    function: Function,
    by_date: bool,
    rerun_status: RerunStatus,
) -> None:
    from .tracker import FunctionTracker

    if not rerun_status.run_needed:
        fc = InputChecker(
            output=path_output,
            inputs=function.inputs,
            parameters=FunctionTracker.args_for_comparison(function),
            by_date=by_date,
        )

        run_because = fc.check_inputs_changed()
        if run_because != RunBecause.Not:
            rerun_status.run_needed = True
            rerun_status.run_because = run_because


class InputChecker:
    """
    Manages checking a list of input paths or PathWithFallbacks objects
    for a given output of a Function to see if needs to run because the
    list of inputs has changed or the input itself is newer than the output

    Parameters
    ----------
    output:str | PathWithFallbacks
        Path of output file or PathWithFallbacks object
    inputs:str | PathWithFallbacks | list[str | PathWithFallbacks],
        List of input files or PathWithFallbacks objects
    parameters:dict, optional
        Function parameters
        The default is None
    only_if_exists:bool, optional
        Is the checking happening before a file run (i.e. the inputs might not exist)
        The default is False.
    by_date:bool, optional
        Flag to update output if any input was updated more recently than the output file
        The default is True
    """

    def __init__(
        self,
        output: str | PathWithFallbacks,
        inputs: str | PathWithFallbacks | list[str | PathWithFallbacks],
        parameters: dict | None = None,
        only_if_exists: bool = False,
        by_date: bool = True,
    ):
        if type(inputs) is not list:
            inputs = [inputs]

        self.output = output
        self.inputs = inputs

        if parameters is None:
            parameters = {}

        self.parameters = parameters
        self.only_if_exists = only_if_exists
        self.by_date = by_date

    def check_inputs_changed(self) -> RunBecause:
        return self.inputs_changed_by_output(
            path_output=self.output,
            inputs=self.inputs,
            only_if_exists=self.only_if_exists,
            parameters=self.parameters,
            by_date=self.by_date,
        )

    @classmethod
    def path_inputs(cls, path: str = "", for_writing: bool = False) -> str:
        path_sd = os.path.normpath(
            f"{os.path.dirname(path)}/.{os.path.basename(path)}_inputs.{SerializableDictionary._save_suffix}"
        )

        if for_writing:
            return path_sd
        else:
            path_csv = os.path.normpath(
                f"{os.path.dirname(path)}/.{os.path.basename(path)}_inputs.csv"
            )

            if os.path.isdir(path_sd):
                return path_sd
            elif os.path.isfile(path_csv):
                return path_csv
            else:
                return ""

    @classmethod
    def inputs_changed_by_output(
        cls,
        path_output: str | PathWithFallbacks,
        inputs: list[str | PathWithFallbacks] | str | PathWithFallbacks,
        only_if_exists: bool = False,
        parameters: dict | None = None,
        by_date: bool = True,
    ) -> RunBecause:
        if parameters is None:
            parameters = {}

        if type(path_output) is PathWithFallbacks:
            if only_if_exists:
                path_output = path_output.path_main
            else:
                path_output = path_output.resolve_path().path

        if path_output == "":
            return RunBecause.OutputDoesNotExist

        path_inputs = cls.path_inputs(path_output)

        if path_inputs == "":
            return RunBecause.UpdateByInputFileList

        #   Check the actual inputs
        if type(inputs) is not list:
            inputs = [inputs]

        inputs_resolved = []
        for inputi in inputs:
            if type(inputi) is str:
                inputs_resolved.append(inputi)
            elif type(inputi) is PathWithFallbacks:
                inputs_resolved.append(inputi.resolve_path().path)

        args = dict(
            path_output=path_output,
            path_inputs=path_inputs,
            inputs=inputs_resolved,
            parameters=parameters,
            by_date=by_date,
        )

        if path_inputs.endswith(f".{SerializableDictionary._save_suffix}"):
            return cls._check_inputs_dict(**args)
        elif path_inputs.endswith(".csv"):
            return cls._check_inputs_csv(**args)

    @classmethod
    def _check_inputs_dict(
        cls,
        path_output: str,
        path_inputs: str,
        inputs: list[str],
        parameters: dict | None = None,
        by_date: bool = True,
    ) -> RunBecause:
        d_inputs = SerializableDictionary.load(path_inputs)

        inputs_file = d_inputs["inputs"]
        parameters_file = d_inputs["parameters"]

        if not cls._compare_objects_nested(parameters, parameters_file):
            return RunBecause.UpdateByParameters
        else:
            inputs_file_normed = [os.path.normpath(ini) for ini in inputs_file]
            inputs_normed = [os.path.normpath(ini) for ini in inputs]

            if len(set(inputs_file_normed).symmetric_difference(inputs_normed)):
                return RunBecause.UpdateByInputFileList
            else:
                #   Check dates of input files against output
                if by_date and cls._any_inputs_newer_than_output(path_output, inputs):
                    return RunBecause.UpdateByDate

        #   We made it here, don't need to run
        return RunBecause.Not

    @classmethod
    def _compare_objects_nested(
        cls, obj1: object, obj2: object, verbose: bool = False
    ) -> bool:
        if (type(obj1) is str and type(obj2) is PathWithFallbacks) or (
            type(obj1) is PathWithFallbacks and type(obj2) is str
        ):
            return os.path.normpath(obj1) == os.path.normpath(obj2)

        if type(obj1) is not type(obj2):
            return False

        if obj1 is None:
            return obj2 is None
        elif type(obj1) is dict:  #    Then so is obj 2
            if len(obj1) != len(obj2):
                #   not the same # of keys
                if verbose:
                    logger.info(f"{obj1} != {obj2}: keys do not match (different #)")
                return False
            elif len(set(obj1.keys()).symmetric_difference(obj2.keys())):
                #   Not the same keys
                if verbose:
                    logger.info(f"{obj1} != {obj2}: keys do not match")
                return False
            else:
                for keyi in obj1.keys():
                    match = cls._compare_objects_nested(obj1[keyi], obj2[keyi])
                    #   Stop comparing if any object doesn't match
                    if not match:
                        if verbose:
                            logger.info(
                                f"{obj1} != {obj2}: {obj1[keyi]} != {obj2[keyi]}"
                            )
                        return False
        elif type(obj1) in [str, int, float, bool]:
            return obj1 == obj2
        elif type(obj1) in [tuple, list, set]:
            if len(obj1) != len(obj2):
                if verbose:
                    logger.info(f"{obj1} != {obj2}: lengths don't match")
                return False
            else:
                if len(set(obj1).symmetric_difference(obj2)):
                    obj1 = list(obj1)
                    obj2 = list(obj2)

                    #   Can they be sorted?
                    try:
                        obj1.sort()
                        obj2.sort()
                    except:
                        if verbose:
                            logger.info("Cannnot sort the objects in the list")
                    for i in range(len(obj1)):
                        match = cls._compare_objects_nested(obj1[i], obj2[i])

                        if not match:
                            if verbose:
                                logger.info(f"{obj1} != {obj2}: {obj1[i]} != {obj2[i]}")
                            return False
        elif type(obj1) is PathWithFallbacks:
            return (obj1.input_path == obj2.input_path) and (
                obj1.fallback_options == obj2.fallback_options
            )
        else:
            try:
                obj1_comp = vars(obj1)
                obj2_comp = vars(obj2)
            except:
                message = f"Comparison failure for {obj1}, {obj2}"
                logger.error(message)
                return False

            for keyi in obj1_comp.keys():
                if not keyi.startswith("_"):
                    match = cls._compare_objects_nested(
                        obj1_comp[keyi], obj2_comp[keyi]
                    )
                    if not match:
                        if verbose:
                            logger.info(
                                f"{obj1} != {obj2}: {obj1_comp[keyi]} != {obj2_comp[keyi]}"
                            )
                        return False

        return True

    @classmethod
    def _check_inputs_csv(
        cls,
        path_output: str,
        path_inputs: str,
        inputs: list[str],
        parameters: dict | None = None,
        by_date: bool = True,
    ) -> RunBecause:
        df_inputs = pl.read_csv(path_inputs)
        c_inputs = pl.col.inputs

        obs_file = 0
        df_obs = (
            df_inputs.with_columns(c_inputs.str.to_lowercase())
            .filter(c_inputs.str.starts_with("obs="))
            .with_columns(c_inputs.str.replace("obs=", "", literal=True))
        )
        if df_obs.height > 0:
            obs_file = df_obs.item(0)

        inputs_file = df_inputs.filter(
            pl.col.inputs.str.to_lowercase().str.starts_with("obs=").not_()
        )["inputs"].to_list()

        if "obs" in parameters:
            obs = parameters["obs"]
        elif "Obs" in parameters:
            obs = parameters["Obs"]
        else:
            obs = 0

        if type(obs) is str:
            obs = int(obs)

        if obs != obs_file:
            return RunBecause.UpdateByInputFileList

        else:
            if len(set(inputs_file).symmetric_difference(inputs)):
                return RunBecause.UpdateByInputFileList
            else:
                #   Check dates of input files against output
                if by_date and cls._any_inputs_newer_than_output(path_output, inputs):
                    return RunBecause.UpdateByDate

        #   We made it here, don't need to run
        return RunBecause.Not

    @classmethod
    def _any_inputs_newer_than_output(cls, output: str, inputs: list[str]) -> bool:
        tinput = 0
        for inputi in inputs:
            if os.path.isfile(inputi) or os.path.exists(inputi):
                tinput = max(tinput, os.path.getmtime(inputi))
            else:
                logger.info(f"Missing file = {inputi}")
                #   tinput = sys.float_info.max

        toutput = sys.float_info.max
        if os.path.isfile(output):
            toutput = min(toutput, os.path.getmtime(output))
        elif os.path.exists(output):
            toutput = min(toutput, os.path.getmtime(output))
        else:
            toutput = 0

        #   Is the newest input newer than the oldest output?
        return tinput > toutput

    @classmethod
    def save_inputs(
        cls,
        outputs: list[str],
        inputs: list[str],
        args: dict | None = None,
        quietly: bool = False,
    ) -> None:
        inputs_write = []
        for inputi in inputs:
            if type(inputi) is PathWithFallbacks:
                inputi = inputi.resolve_path().path
            inputs_write.append(inputi)

        d_save = SerializableDictionary(dict(inputs=inputs_write, parameters=args))

        #   In case I change my mind and want csv's again
        #   df_inputs = pl.DataFrame(list(set(inputs)),schema={"inputs":pl.String})

        for outputi in outputs:
            if outputi is not None and outputi != "":
                if os.path.exists(outputi) or os.path.isdir(outputi):
                    if not quietly:
                        logger.info("     Updating input list for " + outputi)

                    path_for_inputs = cls.path_inputs(outputi, for_writing=True)
                    cls.delete_existing_inputs(outputi, path_for_inputs)

                    d_save.save(path_for_inputs)

                #   In case I change my mind and want csv's again
                #   df_inputs.write_csv(path_for_inputs)

    @classmethod
    def delete_existing_inputs(cls, output: str, path_for_inputs: str = "") -> None:
        if path_for_inputs == "":
            path_for_inputs = cls.path_inputs(output, for_writing=True)
        path_root = os.path.splitext(path_for_inputs)[0]
        #   Delete any existing ones
        for extension in ["csv", "dict"]:
            path_to_check = f"{path_root}.{extension}"
            if os.path.isfile(path_to_check):
                os.remove(path_to_check)
