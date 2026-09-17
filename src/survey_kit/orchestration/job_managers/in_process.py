from __future__ import annotations
from typing import TYPE_CHECKING

import time
import importlib.util
from dataclasses import dataclass

from ...utilities.logging import run_with_temporary_logging
from ..call_status import State

from ... import logger

if TYPE_CHECKING:
    from ..function import Function


@dataclass
class Handle:
    def poll(self) -> State:
        return State.SUCCESS

    def get_std_log(self) -> list[str]:
        return ["", ""]


@dataclass
class Executor:
    name = "in_process"

    @staticmethod
    def submit(function: Function, testing: bool):
        call_status = function.call_status

        if not testing:
            logger.info(f"Run the in-process call for {function.name} - BEGIN")
            with run_with_temporary_logging():
                call_status.start_time = time.time()
                spec = importlib.util.spec_from_file_location(
                    function.name, call_status.callfile
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                function.module = module
            call_status.end_time = time.time()

            from ..tracker import FunctionTracker

            FunctionTracker.save_inputs_for_function(
                self=FunctionTracker, functioni=function
            )
            logger.info(f"                            {function.name} - COMPLETE")

            call_status.handle = Handle()
