from __future__ import annotations
from typing import Optional

import os
from enum import Enum
import subprocess
import time

from ..serializable import Serializable
from .utilities import Languages, CallInputs, CallTypes, LINEBREAK

from .log import remove_useless_errors

from .config import Config

from .. import logger


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

        #   Information needed to get the results of a process
        #       job_id from PBS call
        self.job_id = 0
        #       Process from shell call
        self.process = None

        #   For keeping track of a call
        self.call_number = 0

        #   Manage shared memory to avoid memory leaks in multiprocessing calls
        self.shared_memory_manager = None

        self.language = language

    def check_completion(self):
        #   Only check if the call was made but not yet logged as finished
        if self.started and not self.complete:
            if self.call_input.call_type.value == CallTypes.shell.value:
                #   There is a process, check the status of it
                if self.process is not None:
                    #   process.poll() is not None if finished

                    #   Check pid instead?

                    # checkjobshell = "ps -p " + str(self.process.pid) + " | grep \"" + str(self.process.pid) + "\""
                    # shellout = subprocess.run(checkjobshell,
                    #                       stdout=subprocess.PIPE,
                    #                       stderr=subprocess.PIPE,
                    #                       text=True,
                    #                       shell=True)

                    #   if ((shellout.stdout == "") or (self.process.poll() is not None)):
                    if self.process.poll() is not None:
                        self.end_time = time.time()
                        try:
                            self.stdout_handle.close()
                            self.stderr_handle.close()

                            os.remove(self.stdout_file)
                            os.remove(self.stderr_file)
                        except:
                            logger.error("Failed to close file handles")

                        self.stdout = self.get_file_contents(self.stdout_file)
                        self.stderr = self.get_file_contents(self.stderr_file)

            elif self.call_input.call_type.value == CallTypes.multiprocessing.value:
                if self.process is not None:
                    if not self.process.is_alive():
                        #   Finish the process
                        self.process.join()
                        #   Clean up the shared memory
                        self.shared_memory_manager.shutdown()
                        self.end_time = time.time()

            elif self.call_input.call_type.value == CallTypes.PBS.value:
                #   If there is a jobid, check the status
                if self.job_id > 0:
                    #   Check the job number with qstat
                    #       job_state == "F" indicates finished
                    checkjobshell = (
                        "qstat -fxw " + str(self.job_id) + ' | grep -E "job_state"'
                    )
                    shellout = subprocess.run(
                        checkjobshell,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        shell=True,
                    )

                    statuscode = shellout.stdout.replace("/n", "").strip()

                    if len(statuscode) > 0:
                        statuscode = statuscode[len(statuscode) - 1]

                        if statuscode == "F":
                            self.end_time = time.time()

    def set_job_id(self, stdout=""):
        try:
            #   The job_id is the first part of a string that includes
            #       other info separated by periods
            self.job_id = int(stdout.split(".")[0])
        except:
            logger.info("Invalid stdout for getting a job id")

    def get_output(self):
        if self.complete and not self.output_retrieved:
            if self.call_input.call_type.value == CallTypes.shell.value:
                #   Get the shell output from the process
                # [self.stdout,self.stderr] = self.process.communicate()

                #   Load the log contents
                self.log = remove_useless_errors(
                    self.get_file_contents(FilePath=self.logfile),
                    language=self.language,
                )

                if self.logfile_pythonlogging != "":
                    logging = self.get_file_contents(
                        FilePath=self.logfile_pythonlogging
                    )
                    self.log += LINEBREAK + logging

            elif self.call_input.call_type.value == CallTypes.multiprocessing.value:
                self.log = self.get_file_contents(FilePath=self.logfile)

                if self.logfile_pythonlogging != "":
                    logging = self.get_file_contents(
                        FilePath=self.logfile_pythonlogging
                    )
                    self.log += LINEBREAK + logging

            elif self.call_input.call_type.value == CallTypes.PBS.value:
                stdpath = Config().pbs_log_path
                #   Get the shell output from the default file locations
                stdoutfile = stdpath + str(self.job_id) + ".hpc-pbs.OU"
                stderrfile = stdpath + str(self.job_id) + ".hpc-pbs.ER"

                self.stdout = self.get_file_contents(FilePath=stdoutfile)
                self.stderr = self.get_file_contents(FilePath=stderrfile)

                #   Load the log contents
                self.log = self.get_file_contents(FilePath=self.logfile)

                if self.logfile_pythonlogging != "":
                    logging = self.get_file_contents(
                        FilePath=self.logfile_pythonlogging
                    )
                    self.log += LINEBREAK + logging

                if self.stdout.find("PBS: job killed:") >= 0:
                    self.log += LINEBREAK + self.stdout

            #   Delete the code file
            if os.path.isfile(self.callfile):
                os.remove(self.callfile)

            self.output_retrieved = True

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
