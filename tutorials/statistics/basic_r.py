from __future__ import annotations

import numpy as np
import polars as pl

from survey_kit import logger
from survey_kit.statistics.multiple_imputation import mi_ses_from_function
from survey_kit.statistics.adapters import r_feols

# %%
logger.info("The simplest way to run an R regression from survey_kit: r_feols(),")
logger.info("a wrapper around fixest::feols() with robust SEs by default. Requires R")
logger.info("itself plus rpy2/rpy2-arrow (`pip install survey-kit[r]`) and the R")
logger.info("'fixest' package. Check your setup cheaply with:")
logger.info("    from survey_kit.statistics._r_interop import check_r_setup")
logger.info("    check_r_setup(['fixest'])")


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
(df_estimates, df_ses, df_vcov, df_tidy) = r_feols(df, formula="y ~ x1 + x2")
logger.info(df_estimates)
logger.info(df_ses)


# %%
logger.info("\n\nAcross multiple imputed datasets, combined via Rubin's rules:")
df_implicates = [make_implicate(seed) for seed in range(5)]
mi_reg = mi_ses_from_function(
    delegate=r_feols,
    df_implicates=df_implicates,
    join_on=["Variable"],
    arguments={"formula": "y ~ x1 + x2"},
    round_output=False,
)
mi_reg.print(round_output=False)


# %%
logger.info("\n\nThat's it for the common case. r_lm_adapter (base R lm()/glm()) and")
logger.info("r_fixest_adapter (any fixest estimator, not just feols) cover most of")
logger.info("what's left with named arguments. For anything none of those wrap - a")
logger.info("totally different R package or function - see")
logger.info("r_arbitrary_estimators.py for the generic escape hatch.")
