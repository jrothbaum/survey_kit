from __future__ import annotations
from typing import TYPE_CHECKING

import os
import time
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
import polars as pl
from dataclasses import dataclass

from ..call_status import State
from ..shared_memory import SharedMemoryUtility

from ... import logger

if TYPE_CHECKING:
    from ..function import Function


@dataclass
class Handle:
    process: multiprocessing.process.BaseProcess | None = None
    shared_memory_manager: SharedMemoryManager | None = None
    exit_code: int = 0

    def poll(self) -> State:
        if self.process is None:
            return State.NOT_SET

        if self.process.is_alive():
            return State.IN_PROGRESS

        #   Finish the process and clean up the shared memory
        self.process.join()
        if self.shared_memory_manager is not None:
            self.shared_memory_manager.shutdown()

        self.exit_code = self.process.exitcode

        if self.exit_code == 0:
            return State.SUCCESS
        else:
            return State.FAILED

    def get_std_log(self) -> list[str]:
        return ["", ""]


@dataclass
class Executor:
    name = "multiprocessing"

    @staticmethod
    def _multiprocess_function(
        shm_memory_items: list[dict], logpath: str = "", code: str = ""
    ):
        df_memory_items = SharedMemoryUtility.arrow_shm_list_to_dict_df(
            shm_memory_items
        )
        del shm_memory_items

        #   I know this is not great, but it's the easiest way...
        exec(code)

    @staticmethod
    def submit(function: Function, testing: bool):
        call_status = function.call_status

        if not testing:
            mp_context = multiprocessing.get_context("spawn")

            #   Get the shared memory items and pass them to the subprocess
            memory_items = []

            try:
                smm = SharedMemoryManager()
                smm.start()

                if len(function.shared_memory_items):
                    for keyi, valuei in function.shared_memory_items.items():
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

            with open(call_status.callfile, "r", encoding="utf-8") as f:
                code = f.read()

            call_status.start_time = time.time()

            process = mp_context.Process(
                target=Executor._multiprocess_function,
                args=(memory_items, call_status.logfile_pythonlogging, code),
            )

            process.start()

            call_status.handle = Handle(
                process=process,
                shared_memory_manager=smm,
            )
