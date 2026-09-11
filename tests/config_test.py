
import os

os.environ["POLARS_MAX_THREADS"] = "4"

from survey_kit.orchestration.config import Config


c = Config()

print(c.cpus)

for vari in Config._cpu_env_vars:
    print(os.getenv(vari))

c.cpus = 16
print(c.cpus)

for vari in Config._cpu_env_vars:
    print(os.getenv(vari))