from __future__ import annotations
from typing import Optional

import os
from enum import Enum
import time

from ..serializable import Serializable
from .utilities import Languages, CallInputs, LINEBREAK

from .log import remove_useless_errors


class State(Enum):
    NOT_SET = -1
    IN_PROGRESS = 0
    SUCCESS = 1
    FAILED = 2


class CallStatus(Serializable):
    def __init__(self, language: Languages, call_input: CallInputs | None = None):
        if call_input is None:
            call_input = CallInputs()

        #   Name of code file to be called
        self.callfile = ""
        #   Name of log file from the call (i.e. the sas log file)
        self.logfile = ""
        #   Python logfile for logging output
        self.logfile_pythonlogging = ""

        #   Output of shell/bash call
        self.stdout_file = ""
        self.stderr_file = ""

        self.stdout_handle = None
        self.stderr_handle = None

        self.stdout = ""
        self.stderr = ""

        #   call log contents
        self.log = ""
        self.output_retrieved = False
        self._full_log = ""

        #   Has the call started (times set on call and completion confirmation)
        #       Used to set properties Started, Complete
        self.start_time = 0
        self.end_time = 0

        self.call_input = call_input
        self.state = State.NOT_SET

        #   For keeping track of a call
        self.call_number = 0

        self.language = language

        #   Set by the job_managers.Executor that ran this call - carries
        #       whatever backend-specific state is needed to poll for
        #       completion and retrieve output (job_id/process/etc.)
        self.handle = None

    def check_completion(self):
        #   Only check if the call was made but not yet logged as finished
        if self.started and not self.complete:
            if self.handle is not None:
                self.state = self.handle.poll()

                if self.state != State.IN_PROGRESS:
                    self.end_time = time.time()

    def get_output(self):
        if self.complete and not self.output_retrieved:
            if self.handle is not None:
                [self.stdout, self.stderr] = self.handle.get_std_log()

            #   Load the log contents
            log = remove_useless_errors(
                self.get_file_contents(FilePath=self.logfile),
                language=self.language,
            )

            if self.logfile_pythonlogging != "":
                logging = self.get_file_contents(FilePath=self.logfile_pythonlogging)
                log += LINEBREAK + logging

            self.log = log

            #   PBS calls killed for exceeding resource limits report it in stdout only
            if self.stdout.find("PBS: job killed:") >= 0:
                self.log += LINEBREAK + self.stdout

            #   Delete the code file
            if os.path.isfile(self.callfile):
                os.remove(self.callfile)

            self.output_retrieved = True

    @property
    def job_id(self) -> int:
        return getattr(self.handle, "job_id", 0)

    @property
    def process(self):
        return getattr(self.handle, "process", None)

    @property
    def shared_memory_manager(self):
        return getattr(self.handle, "shared_memory_manager", None)

    def get_file_contents(self, FilePath: str = "", bDelete: bool = True):
        if os.path.isfile(FilePath):
            #   Load the file contents
            fFile = open(FilePath, "r", encoding="utf-8")
            output = fFile.read()
            fFile.close()

            #   Delete the file
            if bDelete:
                os.remove(FilePath)
        else:
            output = ""

        return output

    @property
    def started(self):
        return self.start_time > 0

    @property
    def complete(self):
        return self.end_time > 0

    @property
    def execution_time(self):
        if self.started and self.complete:
            return round(self.end_time - self.start_time, 0)
        else:
            return -1
