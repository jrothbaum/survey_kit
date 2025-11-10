import os
import sys
from pathlib import Path
import polars as pl
from survey_kit.orchestration.config import Config


c = Config()


print(c.ram)
print("CPUS ")
print(f"    Python: {c.cpus}")
# print(f"    Polars: {pl.thread_pool_size()}")
c.cpus = 2
print(f"    Set:    {c.cpus}")
print(f"    Polars: {pl.thread_pool_size()}")

c.code_root = os.path.normpath("C:/Users/jonro/OneDrive/Documents/Coding")
print(f"code root={c.code_root}")

c.versions = ["v3", "v2", "v1"]
print(f"versions = {c.versions}")

c.versions = [3, 2, 1]
print(f"versions = {c.versions}")

print(c.data_root)
print(c.data_with_version)


c.parameter_files = dict(
    Stata="/this/file/parameters.do",
    SAS="/this/file/parameters.sas",
    R="/this/file/parameters.R",
)

print(c.parameter_files)

c.pbs_log_path = "/pbs/log/path"


c.clean_temp_directory(2)
