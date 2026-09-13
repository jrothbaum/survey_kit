from __future__ import annotations

import os

import numpy as np
import polars as pl
from dotenv import load_dotenv

from survey_kit import logger
from survey_kit.statistics.multiple_imputation import mi_ses_from_function
from survey_kit.statistics.adapters import stata_adapter

# %%
logger.info("The simplest way to run a Stata regression from survey_kit:")
logger.info("stata_adapter() - pass any e-class command as a plain string.")
logger.info("")
logger.info("**UNTESTED** here (no Stata license in this development environment) -")
logger.info("see stata_arbitrary_estimators.py and _stata_interop's module")
logger.info("docstring for details/caveats. Requires `pip install survey-kit[stata]`")
logger.info("plus a licensed Stata 17+ install. Check your setup cheaply with:")
logger.info("    from survey_kit.statistics._stata_interop import check_stata_setup")
logger.info('    check_stata_setup(stata_path=r"C:\\Program Files\\Stata17")')

#   Machine-specific - set these in a local ".env" file (see .gitignore,
#   which excludes it from git) in the repo root rather than editing this
#   file or exporting them yourself:
#       _survey_kit_stata_path_=C:\Program Files\Stata17
#       _survey_kit_stata_edition_=se
#   Or, just as easily, set them directly in code instead of via env vars:
#       from survey_kit import config
#       config.stata_path = r"C:\Program Files\Stata17"
#       config.stata_edition = "se"
load_dotenv()

# %%
def make_implicate(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1 + 2 * x1 - 1.5 * x2 + rng.normal(size=n) * 0.4
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


df = make_implicate(0)

# %%
logger.info("\n\nOn one dataset, standalone - no MI at all:")
(df_estimates, df_ses, df_vcov, df_tidy) = stata_adapter(
    df, command="regress y x1 x2"
)
logger.info(df_estimates)
logger.info(df_ses)


# %%
logger.info("\n\nAcross multiple imputed datasets, combined via Rubin's rules:")
df_implicates = [make_implicate(seed) for seed in range(5)]
mi_reg = mi_ses_from_function(
    delegate=stata_adapter,
    df_implicates=df_implicates,
    join_on=["Variable"],
    arguments={
        "command": "regress y x1 x2",
    },
    round_output=False,
)
mi_reg.print(round_output=False)


# %%
logger.info("\n\nThat's it for the common case. For weighted/survey designs, other")
logger.info("commands (svy:/xtreg/areg/logit/a community-installed ado/...), or")
logger.info("pulling back custom r()/e() results via stata_results_adapter instead")
logger.info("of the usual e(b)/e(V)/r(table), see stata_arbitrary_estimators.py.")
