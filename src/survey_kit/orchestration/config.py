import os
import psutil
import json
import tempfile
from pathlib import Path
from datetime import datetime


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
    """
    Global configuration for survey-kit.

    Config manages package-wide settings including paths, CPU limits, memory settings,
    and environment variables. Settings can be configured via environment variables
    or by directly setting attributes on the config instance.

    The config instance is typically accessed via:
    ```python
        from survey_kit import config
        config.data_root = "/path/to/data"
        config.cpus = 8
    ```

    Attributes
    ----------
    code_root : str
        Root directory for code files. Set via environment variable
        `_survey_kit_code_root_` or directly. Default is "".
    data_root : str
        Root directory for data files. Set via environment variable
        `_survey_kit_data_root_` or directly. Default is "".
    cpus : int
        Number of CPUs to use for parallel operations. Automatically sets
        thread limits for Polars, OpenBLAS, MKL, etc. when changed.
        Set via `_survey_kit_n_cpus_` or directly. Default is os.cpu_count().
    path_temp_files : str
        Directory for temporary files. Set via `_survey_kit_path_temp_files_`
        or directly. Default is {data_root}/temp_files.
    ram : int
        Total available RAM in bytes. Set via `_survey_kit_ram_`.
        Default is system total memory.
    mem_in_gb : int
        Available memory in gigabytes (read-only).
    mem_in_mb : int
        Available memory in megabytes (read-only).
    mem_in_kb : int
        Available memory in kilobytes (read-only).
    Examples
    --------
    Basic configuration:

    >>> from survey_kit import config
    >>> config.data_root = "/projects/data/myproject"
    >>> config.cpus = 16
    >>> print(config.path_temp_files)
    '/projects/data/myproject/temp_files'

    Using environment variables:
    ```bash
        export _survey_kit_data_root_="/projects/data/myproject"
        export _survey_kit_n_cpus_=16
    ```

    Memory information:

    >>> print(f"Available RAM: {config.mem_in_gb} GB")
    >>> print(f"Available RAM: {config.mem_in_mb} MB")

    Temporary files:

    >>> temp_path = config.path_temp_with_random(as_parquet=True)
    >>> print(temp_path)
    '/projects/data/myproject/temp_files/abc123xyz.parquet'

    Notes
    -----
    Setting `cpus` automatically updates thread limits for multiple libraries:
    - POLARS_MAX_THREADS
    - OMP_NUM_THREADS
    - NUMEXPR_NUM_THREADS
    - MKL_NUM_THREADS
    - OPENBLAS_NUM_THREADS

    """

    # parameter_files : dict
    #     Dictionary of parameter file paths. Set via `_survey_kit_parameter_files_`.
    #     Default is {}.
    # pbs_log_path : str
    #     Path for PBS job logs. Set via `_survey_kit_pbs_log_path_`.
    #     Default is "".
    # versions : list
    #     (IGNORE - INTENDED FOR FUTURE USE)
    #     List of version strings (e.g., ["V3", "V2"]). Used to construct
    #     versioned data paths. Set via `_survey_kit_versions_`. Default is [].
    # latest_version : str
    #     (IGNORE - INTENDED FOR FUTURE USE)
    #     Most recent version string from versions list (read-only).
    # data_with_version : str
    #     (IGNORE - INTENDED FOR FUTURE USE)
    #     Path combining data_root with latest_version (read-only).

    _code_root_key = "_survey_kit_code_root_"
    _data_root_key = "_survey_kit_data_root_"
    _version_key = "_survey_kit_versions_"
    _cpus_key = "_survey_kit_n_cpus_"
    _path_temp_files_key = "_survey_kit_path_temp_files_"
    _ram_key = "_survey_kit_ram_"
    _parameter_files_key = "_survey_kit_parameter_files_"
    _pbs_log_path_key = "_survey_kit_pbs_log_path_"

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
            from .. import logger

            logger.error(message)
            raise Exception(message)

        return int(self.ram / 1024**power)

    def path_temp_with_random(
        self, as_parquet: bool = False, underscore_prefix: bool = False
    ) -> str:
        """
        Generate a random temporary file path.

        Parameters
        ----------
        as_parquet : bool, optional
            Add .parquet extension. Default is False.
        underscore_prefix : bool, optional
            Prefix filename with underscore. Default is False.

        Returns
        -------
        str
            Full path to a uniquely-named temporary file.

        Examples
        --------
        >>> config.path_temp_with_random()
        '/data/temp_files/abc123xyz'

        >>> config.path_temp_with_random(as_parquet=True)
        '/data/temp_files/abc123xyz.parquet'
        """
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

    def clean_temp_directory(self, clean_older_than_days: int = 7):
        from .. import logger

        try:
            path_to_clean = self.path_temp_files

            CleanTempDirectory.clean_old_files(
                temp_dir_path=path_to_clean, clean_older_than_days=clean_older_than_days
            )
        except:
            from .. import logger

            logger.info("Not temp directory to clean")


class CleanTempDirectory:
    LAST_CLEANED_FILE = "survey_kit_last_cleaned.txt"

    @staticmethod
    def clean_old_files(temp_dir_path: str, clean_older_than_days: int = 7):
        """
        Clean old files from temp directory, at most once per day.

        Parameters
        ----------
        temp_dir_path : str, optional
            Path to the temp directory. If None, uses the default temp path.
        clean_older_than_days : int, optional
            Delete files older than this many days. Default is 7.
        """
        from .. import logger

        # Convert to Path object and ensure it exists
        temp_dir_path = Path(temp_dir_path)
        from .. import logger

        if not temp_dir_path.exists():
            logger.warning(f"Temp directory does not exist: {temp_dir_path}")
            return

        # Check if already cleaned today
        if CleanTempDirectory._already_cleaned_today(temp_dir_path):
            logger.info("Temp directory already cleaned today, skipping")
            return

        # Clean old files
        logger.info(
            f"Cleaning files older than {clean_older_than_days} days from {temp_dir_path}"
        )
        CleanTempDirectory._delete_old_files(temp_dir_path, clean_older_than_days)

        # Update last cleaned timestamp
        CleanTempDirectory._update_last_cleaned(temp_dir_path)

    @staticmethod
    def _already_cleaned_today(temp_dir_path: Path) -> bool:
        """Check if the temp directory was already cleaned today."""
        from .. import logger

        last_cleaned_file = temp_dir_path / CleanTempDirectory.LAST_CLEANED_FILE

        if not last_cleaned_file.is_file():
            return False

        try:
            last_cleaned_str = last_cleaned_file.read_text().strip()
            last_cleaned = datetime.strptime(last_cleaned_str, "%Y-%m-%d").date()
            current_date = datetime.now().date()

            return last_cleaned == current_date
        except Exception as e:
            logger.warning(f"Error reading last cleaned file: {e}")
            return False

    @staticmethod
    def _update_last_cleaned(temp_dir_path: Path):
        """Update the last cleaned timestamp file."""
        from .. import logger

        last_cleaned_file = temp_dir_path / CleanTempDirectory.LAST_CLEANED_FILE
        current_date = datetime.now().date().strftime("%Y-%m-%d")

        try:
            last_cleaned_file.write_text(current_date)
            logger.info(f"Updated last cleaned timestamp to {current_date}")
        except Exception as e:
            logger.error(f"Error updating last cleaned file: {e}")

    @staticmethod
    def _delete_old_files(temp_dir_path: Path, clean_older_than_days: int = 7):
        """
        Recursively delete files older than specified days from temp directory.

        Parameters
        ----------
        temp_dir_path : Path
            Path to clean
        clean_older_than_days : int, optional
            Delete files older than this many days. Default is 7.
        """
        from .. import logger

        current_date = datetime.now().date()

        try:
            for entry in temp_dir_path.iterdir():
                try:
                    # Skip the last cleaned file itself
                    if (
                        entry.is_file()
                        and entry.name == CleanTempDirectory.LAST_CLEANED_FILE
                    ):
                        continue

                    if entry.is_file():
                        days_since_modified = (
                            current_date
                            - datetime.fromtimestamp(entry.stat().st_mtime).date()
                        ).days

                        if days_since_modified > clean_older_than_days:
                            logger.info(f"Removing old file: {entry}")
                            try:
                                entry.unlink()
                            except Exception as e:
                                logger.warning(f"Failed to remove {entry}: {e}")

                    elif entry.is_dir():
                        # Recursively clean subdirectories
                        CleanTempDirectory._delete_old_files(
                            entry, clean_older_than_days
                        )

                        # Delete empty directories
                        try:
                            if not any(entry.iterdir()):  # Check if directory is empty
                                logger.info(f"Deleting empty directory: {entry}")
                                entry.rmdir()
                        except Exception as e:
                            logger.warning(f"Failed to delete directory {entry}: {e}")

                except Exception as e:
                    logger.warning(f"Error processing {entry}: {e}")
        except Exception as e:
            logger.error(f"Error scanning directory {temp_dir_path}: {e}")
