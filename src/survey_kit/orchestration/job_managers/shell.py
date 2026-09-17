from __future__ import annotations
from typing import TYPE_CHECKING

import os
import sys
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass

from ..utilities import Languages
from ..call_status import State

from ... import logger

if TYPE_CHECKING:
    from ..function import Function


def _read_and_delete(path: str) -> str:
    if path != "" and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            output = f.read()
        os.remove(path)
    else:
        output = ""

    return output


@dataclass
class Handle:
    process: subprocess.Popen | None = None
    stdout_file: str = ""
    stderr_file: str = ""
    stdout_handle: object = None
    stderr_handle: object = None
    exit_code: int = 0

    def poll(self) -> State:
        if self.process is None:
            return State.NOT_SET

        rc = self.process.poll()
        if rc is None:
            return State.IN_PROGRESS

        self.exit_code = rc

        try:
            self.stdout_handle.close()
            self.stderr_handle.close()
        except:
            logger.error("Failed to close file handles")

        if self.exit_code == 0:
            return State.SUCCESS
        else:
            return State.FAILED

    def get_std_log(self) -> list[str]:
        stdout = _read_and_delete(self.stdout_file)
        stderr = _read_and_delete(self.stderr_file)
        return [stdout, stderr]


@dataclass
class Executor:
    name = "shell"

    @staticmethod
    def command(function: Function) -> str:
        call_status = function.call_status

        if function.language == Languages.SAS:
            cmd = "sas " + call_status.callfile + " -log " + call_status.logfile
        elif function.language == Languages.Python:
            cmd = f"{sys.executable} " + call_status.callfile
        elif function.language == Languages.Stata:
            cmd = "stata-mp -q -b do " + call_status.callfile
        elif function.language == Languages.R:
            cmd = (
                'R CMD BATCH --no-save --quiet "'
                + call_status.callfile
                + '" "'
                + call_status.logfile
                + '"'
            )
        else:
            message = f"Shell execution not supported for {function.language}"
            logger.error(message)

        return cmd

    @staticmethod
    def submit(function: Function, testing: bool):
        call_status = function.call_status
        cmd = Executor.command(function)

        call_status.start_time = time.time()

        if not testing:
            call_status.stdout_file = Path(f"{call_status.callfile}.stdout").as_posix()
            call_status.stderr_file = Path(f"{call_status.callfile}.stderr").as_posix()

            stdout_handle = open(call_status.stdout_file, "w", encoding="utf-8")
            stderr_handle = open(call_status.stderr_file, "w", encoding="utf-8")

            process = subprocess.Popen(
                cmd,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                cwd=os.path.dirname(call_status.callfile),
                shell=True,
            )

            call_status.handle = Handle(
                process=process,
                stdout_file=call_status.stdout_file,
                stderr_file=call_status.stderr_file,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
            )
