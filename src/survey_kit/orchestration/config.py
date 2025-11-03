import os
import psutil
import json
import tempfile
from pathlib import Path


class TypedEnvVar:
    def __init__(self, env_name: str, default=None, convert=str):
        self.env_name = env_name
        self.default = default
        self.convert = convert

    def __get__(self, obj, objtype=None):
        value = os.getenv(self.env_name)
        if value is None:
            return self.default

        if self.convert in [list, dict]:
            return json.loads(value)

        return self.convert(value)

    def __set__(self, obj, value):
        # Convert to JSON when setting
        if isinstance(value, (list, dict)):
            os.environ[self.env_name] = json.dumps(value)
        else:
            os.environ[self.env_name] = str(value)


class Config:
    _code_root_key = "_SurveyRep_CodeRoot"
    _data_root_key = "_SurveyRep_DataRoot"
    _version_key = "_SurveyRep_Versions"
    _cpus_key = "_SurveyRep_CPUs"
    _path_temp_files_key = "_SurveyRep_path_temp_files"
    _ram_key = "_SurveyRep_RAM"
    _parameter_files_key = "_SurveyRep_parameter_files"
    _pbs_log_path_key = "_SurveyRep_pbs_log_path"

    code_root = TypedEnvVar(_code_root_key, default="", convert=str)
    data_root = TypedEnvVar(_data_root_key, default="", convert=str)
    versions = TypedEnvVar(_version_key, default=[], convert=list)
    _cpus = TypedEnvVar(_cpus_key, os.cpu_count(), int)
    _path_temp_files = TypedEnvVar(_path_temp_files_key, "", str)
    ram = TypedEnvVar(_ram_key, psutil.virtual_memory().total)
    parameter_files = TypedEnvVar(_parameter_files_key, {}, convert=dict)
    pbs_log_path = TypedEnvVar(_pbs_log_path_key, "", str)

    @property
    def latest_version(self) -> str:
        versions = self.versions

        if len(versions):
            return versions[0]

        return ""

    @property
    def data_with_version(self) -> str:
        versions = self.versions

        output = self.data_root
        if len(versions):
            latest = str(versions[0])
            output = os.path.join(output, latest)

        return output

    @property
    def cpus(self) -> int:
        return self._cpus

    @cpus.setter
    def cpus(self, value: int):
        self._cpus = value

        self._set_thread_limits()

    @property
    def path_temp_files(self) -> int:
        if self._path_temp_files != "":
            return self._path_temp_files
        else:
            if self.data_root == "":
                from .. import logger
                message = "You must set Configs().data_root to get a default temp file directory"
                logger.error(message)
                raise Exception(message)

            return Path(self.data_root) / "temp_files"

    @path_temp_files.setter
    def path_temp_files(self, value: str):
        self._path_temp_files = value

    def _set_thread_limits(self):
        n_cpus = self.cpus

        cpu_limits = [
            "POLARS_MAX_THREADS",
            "OMP_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ]
        for limiti in cpu_limits:
            os.environ[limiti] = str(n_cpus)

    @property
    def mem_in_gb(self) -> int:
        return self._mem("gb")

    @property
    def mem_in_mb(self) -> int:
        return self._mem("mb")

    @property
    def mem_in_kb(self) -> int:
        return self._mem("kb")

    def _mem(self, unit: str) -> int:
        unit = unit.lower()

        if unit == "gb":
            power = 3
        elif unit == "mb":
            power = 2
        elif unit == "kb":
            power = 1
        else:
            from .. import logger
            message = f"Must pass kb, mb, or gb ({unit})"
            logger.error(message)
            raise Exception(message)

        return int(self.ram / 1024**power)

    def path_temp_with_random(
        self, as_parquet: bool = False, underscore_prefix: bool = False
    ) -> str:
        if as_parquet:
            parquet_suffix = ".parquet"
        else:
            parquet_suffix = ""

        if underscore_prefix:
            prefix = "_"
        else:
            prefix = ""

        return os.path.normpath(
            f"{self.path_temp_files}/{prefix}{next(tempfile._get_candidate_names())}{parquet_suffix}"
        )
