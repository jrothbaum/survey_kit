from __future__ import annotations

import numpy as np
import polars as pl

from survey_kit import logger
from survey_kit.statistics.multiple_imputation import mi_ses_from_function
from survey_kit.statistics.adapters import (
    statsmodels_adapter,
    linearmodels_adapter,
    pyfixest_adapter,
    polars_ds_adapter,
)

# %%
logger.info("survey_kit.statistics.adapters has four pure-Python regression")
logger.info("adapters, one per package - no R/Stata/rpy2/pystata needed. Every")
logger.info("adapter (these four, plus the R/Stata ones) returns the same")
logger.info("normalized shape:")
logger.info("    (df_estimates, df_ses, df_vcov, df_tidy)")
logger.info("so mi_ses_from_function/StatCalculator.from_function treat any of them")
logger.info("identically - pick whichever package you already use or have")
logger.info("installed; none of these are hard dependencies of survey_kit itself.")


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
logger.info("\n\nstatsmodels_adapter - y/x column lists rather than a formula, HC3")
logger.info("(heteroskedasticity-robust) SEs by default:")
(df_estimates, df_ses, df_vcov, df_tidy) = statsmodels_adapter(df, y="y", x=["x1", "x2"])
logger.info(df_estimates)

# %%
logger.info("\n\nlinearmodels_adapter - formula syntax, IV/panel-capable (plain OLS")
logger.info("via IV2SLS with no instruments, as here):")
(df_estimates, df_ses, df_vcov, df_tidy) = linearmodels_adapter(
    df, formula="y ~ 1 + x1 + x2"
)
logger.info(df_estimates)

# %%
logger.info("\n\npyfixest_adapter - fixest-syntax formula, fixed effects supported")
logger.info("directly in the formula (e.g. 'y ~ x1 + x2 | firm') - generally the")
logger.info("best default of the four unless you specifically need something it")
logger.info("doesn't cover (see its docstring):")
(df_estimates, df_ses, df_vcov, df_tidy) = pyfixest_adapter(df, formula="y ~ x1 + x2")
logger.info(df_estimates)

# %%
logger.info("\n\npolars_ds_adapter - stays entirely in polars/narwhals, no pandas")
logger.info("conversion at all; doesn't compute a covariance matrix, so df_vcov is")
logger.info("always None here:")
(df_estimates, df_ses, df_vcov, df_tidy) = polars_ds_adapter(df, y="y", x=["x1", "x2"])
logger.info(df_estimates)


# %%
logger.info("\n\nAny of these plugs into mi_ses_from_function the same way - here")
logger.info("with pyfixest_adapter, across multiple imputed datasets combined via")
logger.info("Rubin's rules:")
df_implicates = [make_implicate(seed) for seed in range(5)]
mi_reg = mi_ses_from_function(
    delegate=pyfixest_adapter,
    df_implicates=df_implicates,
    join_on=["Variable"],
    arguments={"formula": "y ~ x1 + x2"},
    round_output=False,
)
mi_reg.print(round_output=False)


# %%
logger.info("\n\nFor R or Stata instead, see basic_r.py/basic_stata.py (or")
logger.info("r_arbitrary_estimators.py/stata_arbitrary_estimators.py for the generic")
logger.info("escape hatch into any function either package provides).")
