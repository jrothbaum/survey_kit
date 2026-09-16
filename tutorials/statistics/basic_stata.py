from __future__ import annotations

import os

import numpy as np
import polars as pl
from dotenv import load_dotenv

from survey_kit import logger
from survey_kit.statistics.adapters import stata_adapter

# %%
logger.info("The simplest way to run a Stata regression from survey_kit:")
logger.info("stata_adapter() - pass any e-class command as a plain string.")

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
rng = np.random.default_rng(0)
n = 300
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
y = 1 + 2 * x1 - 1.5 * x2 + rng.normal(size=n) * 0.4
df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})

# %%
logger.info("\n\nOn one dataset, standalone - no MI at all. stata_adapter() returns")
logger.info("an AdapterStats (a StatCalculator subclass), so it already works with")
logger.info(".print(), .compare(), survey_kit.plot, save/load - no extra step:")
stata_result = stata_adapter(df, command="regress y x1 x2")
stata_result.print()


# %%
logger.info(
    "\n\nThat's it for the basics. For multiple imputation (mi_ses_from_stata),"
)
logger.info("weighted/survey designs, other commands (svy:/xtreg/areg/logit/a")
logger.info("community-installed ado/...), replicate-weight bootstrapping, or pulling")
logger.info("back custom r()/e() results via stata_results_adapter instead of the")
logger.info("usual e(b)/e(V)/r(table), see stata_arbitrary_estimators.py.")
