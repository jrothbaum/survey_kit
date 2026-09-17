from __future__ import annotations
from typing import TYPE_CHECKING

import os
import getpass
import json
import subprocess
import time
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from cachetools import cached, TTLCache

from ..config import Config
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
    job_id: int
    job_data: dict | None = None

    def poll(self) -> State:
        self.job_data = JobInfo.get_job_info(self.job_id)

        if self.is_finished:
            if self.pbs_status == JobInfo.Status.SUCCESS:
                return State.SUCCESS
            else:
                return State.FAILED
        else:
            return State.IN_PROGRESS

    def get_std_log(self) -> list[str]:
        stdpath = Config().pbs_log_path
        stdoutfile = stdpath + str(self.job_id) + ".hpc-pbs.OU"
        stderrfile = stdpath + str(self.job_id) + ".hpc-pbs.ER"

        stdout = _read_and_delete(stdoutfile)
        stderr = _read_and_delete(stderrfile)

        return [stdout, stderr]

    @property
    def is_finished(self) -> bool:
        try:
            return self.job_data["is_finished"]
        except:
            return False

    @property
    def pbs_state_code(self) -> "JobInfo.JobState":
        return self.job_data["job_state_code"]

    @property
    def pbs_status(self) -> "JobInfo.Status":
        return self.job_data["exit_status_code"]

    @property
    def ram_used(self):
        try:
            return self.job_data["resources_used"]["mem"]
        except:
            return 0

    @property
    def cpu_used(self):
        try:
            return self.job_data["resources_used"]["cpupercent"]
        except:
            return 0

    @property
    def wall_time(self) -> int:
        try:
            t = datetime.strptime(
                self.job_data["resources_used"]["walltime"], "%H:%M:%S"
            )

            return t.hour * 3600 + t.minute * 60 + t.second
        except:
            return 0


@dataclass
class Executor:
    name = "pbs"

    @staticmethod
    def command(function: Function) -> str:
        call_status = function.call_status

        if function.language == Languages.SAS:
            cmd = "qsas_news --sasprog=" + call_status.callfile
        elif function.language == Languages.Python:
            #   Different log file default for python qsub
            call_status.logfile = call_status.callfile + ".log"
            cmd = "qpy_news --programfile=" + call_status.callfile
        elif function.language == Languages.Stata:
            cmd = "qstata_news --nologo --dofile=" + call_status.callfile
        elif function.language == Languages.R:
            cmd = (
                "qR_news --program="
                + call_status.callfile
                + " --logfile="
                + call_status.logfile
                + " --quiet"
            )
        elif function.language == Languages.Bash:
            cmd = f"qsub {call_status.callfile}"

        mem_in_mb = call_status.call_input.mem_in_mb
        n_cpu = call_status.call_input.n_cpu

        if function.language != Languages.Bash:
            cmd += " --cpucount=" + str(n_cpu) + " --memsize=" + str(mem_in_mb)

        return cmd

    @staticmethod
    def submit(function: Function, testing: bool):
        call_status = function.call_status

        cmd = Executor.command(function)

        call_status.start_time = time.time()

        if not testing:
            shellout = subprocess.run(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            #   Set the information needed to check if the command is complete
            try:
                #   The job_id is the first part of a string that includes
                #       other info separated by periods
                job_id = int(shellout.stdout.split(".")[0])
            except:
                logger.error("Invalid stdout for getting a job id")

                call_status.end_time = time.time()
                call_status.state = State.FAILED
                return

            call_status.handle = Handle(job_id=job_id)


class JobInfo:
    class JobState(Enum):
        Q = "Queued"
        R = "Running"
        E = "Exiting"
        F = "Finished"
        W = "Waiting"
        T = "Moving"
        X = "Expired"
        B = "Begun"
        S = "Suspended"
        U = "Suspended (workstation busy)"
        M = "Moved"

    class Status(Enum):
        NOT_SET = -999999
        SUCCESS = 0
        FAILED_TO_START = -1
        REQUEUED = -2
        DELETED_BEFORE_START = -3
        DELETED = -4
        RESOURCE_LIMIT = 271
        KILLED = 265

    _qstat_cache = TTLCache(maxsize=1, ttl=15)

    _d_finished = dict()

    @staticmethod
    @cached(_qstat_cache)
    def get_all_qstat_job_info() -> dict:
        user = getpass.getuser()
        job_list = subprocess.run(
            ["qstat", "-u", user], capture_output=True, text=True
        )

        job_list = [line.split() for line in job_list.stdout.splitlines()]
        job_list = [
            line[0].split(".")[0]
            for line in job_list
            if len(line) > 3 and line[0][0].isnumeric()
        ]

        if len(job_list) == 0:
            return {}

        job_data = subprocess.run(
            ["qstat", "-f"] + job_list + ["-F", "json"],
            capture_output=True,
            text=True,
        )

        d_jobs = json.loads(job_data.stdout).get("Jobs", {})

        d_jobs_return = {}
        for keyi, valuei in d_jobs.items():
            job_id = int(keyi.split(".")[0])
            valuei = JobInfo._process_job_state(valuei)

            if valuei["is_finished"]:
                JobInfo._d_finished[job_id] = valuei
            else:
                d_jobs_return[job_id] = valuei

        return d_jobs_return

    @staticmethod
    def get_job_info(job_id: int, get_if_finished: bool = True) -> dict:
        d_all_jobs = JobInfo.get_all_qstat_job_info()
        d_all_jobs.update(JobInfo._d_finished)

        if job_id in d_all_jobs:
            return d_all_jobs[job_id]

        if get_if_finished:
            x_flag = "x"
        else:
            x_flag = ""
        job_data = subprocess.run(
            ["qstat", f"-f{x_flag}", str(job_id), "-F", "json"],
            capture_output=True,
            text=True,
        )

        job_data = json.loads(job_data.stdout).get("Jobs", {})
        job_data = {
            int(keyi.split(".")[0]): JobInfo._process_job_state(valuei)
            for keyi, valuei in job_data.items()
        }

        job_data = job_data[job_id]

        add_to_cache = True
        if job_data["is_finished"]:
            JobInfo._d_finished[job_id] = job_data
            add_to_cache = False

        if add_to_cache:
            if () in JobInfo._qstat_cache:
                JobInfo._qstat_cache[()][job_id] = job_data

        return job_data

    @staticmethod
    def _process_job_state(d: dict) -> dict:
        d["job_state_code"] = JobInfo.JobState[d["job_state"]]
        d["is_finished"] = JobInfo._is_finished(d)

        if d["is_finished"]:
            exit_status = d["Exit_status"]
            exit_status_enum = None
            try:
                exit_status_enum = JobInfo.Status(exit_status)
            except:
                pass

            d["exit_status_code"] = exit_status_enum

        return d

    @staticmethod
    def _is_finished(job_data: dict) -> bool:
        return job_data["job_state_code"] in [JobInfo.JobState.X, JobInfo.JobState.F]
