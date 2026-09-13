from __future__ import annotations

from survey_kit import logger
from survey_kit.statistics.adapters import (
    statsmodels_adapter,
    linearmodels_adapter,
    pyfixest_adapter,
    polars_ds_adapter,
    mi_ses_from_statsmodels,
    mi_ses_from_linearmodels,
    mi_ses_from_pyfixest,
    mi_ses_from_polars_ds,
)
from survey_kit.utilities.dataframe import summary
from survey_kit.statistics.replicates import Replicates
from sample_data import make_implicates, with_bootstrap_weights

# %%
logger.info("survey_kit.statistics.adapters has four pure-Python regression")
logger.info("adapters, one per package - no R/Stata/rpy2/pystata needed. Every")
logger.info("adapter (these four, plus the R/Stata ones) returns the same")
logger.info("normalized shape:")
logger.info("    (df_estimates, df_ses, df_vcov, df_tidy)")
logger.info("and each has a matching mi_ses_from_<package>(...) - same shape as")
logger.info("mi_ses_from_r_fixest/mi_ses_from_stata - that runs it across")
logger.info("implicates, combined via Rubin's rules, taking the adapter's own")
logger.info("arguments directly instead of an arguments={} dict.")

# %%
df_implicates = make_implicates()
df = df_implicates[0]
logger.info("\n\nSample data (y = 1 + 2*x1 - 1.5*x2 + noise) - see sample_data.py:")
summary(df)

# %%
logger.info("\n\nstatsmodels - y/x column lists rather than a formula, HC3")
logger.info("(heteroskedasticity-robust) SEs by default:")
(df_estimates, df_ses, df_vcov, df_tidy) = statsmodels_adapter(df, y="y", x=["x1", "x2"])
logger.info("\n   Single df")
logger.info(df_estimates)

mi_sm = mi_ses_from_statsmodels(df_implicates=df_implicates, y="y", x=["x1", "x2"])
logger.info("\n   Multiple imputation")
mi_sm.print()

# %%
logger.info("\n\nlinearmodels - formula syntax, IV/panel-capable (plain OLS via")
logger.info("IV2SLS with no instruments, as here):")
(df_estimates, df_ses, df_vcov, df_tidy) = linearmodels_adapter(
    df, formula="y ~ 1 + x1 + x2"
)
logger.info("\n   Single df")
logger.info(df_estimates)

mi_lm = mi_ses_from_linearmodels(df_implicates=df_implicates, formula="y ~ 1 + x1 + x2")
logger.info("\n   Multiple imputation")
mi_lm.print()

# %%
logger.info("\n\npyfixest - fixest-syntax formula, fixed effects supported")
logger.info("directly (e.g. 'y ~ x1 + x2 | firm') - generally the best default")
logger.info("of the four unless you specifically need something it doesn't")
logger.info("cover (see its docstring):")
(df_estimates, df_ses, df_vcov, df_tidy) = pyfixest_adapter(df, formula="y ~ x1 + x2")
logger.info("\n   Single df")
logger.info(df_estimates)

mi_pf = mi_ses_from_pyfixest.feols(df_implicates=df_implicates, fml="y ~ x1 + x2")
logger.info("\n   Multiple imputation")
mi_pf.print()

# %%
logger.info("\n\nReplicate-weight bootstrapping instead of pyfixest's own vcov: pass")
logger.info("replicates=, and pyfixest_adapter runs once per replicate weight column")
logger.info("(point estimates only, vcov forced to \"iid\" since it's discarded")
logger.info("anyway) - the spread of estimates across replicates IS the SE, computed")
logger.info("by survey_kit's own Replicates/StatCalculator machinery, the same")
logger.info("approach mi_ses_from_stata/mi_ses_from_r_fixest's replicates= use. Every")
logger.info("mi_ses_from_<package> here (except .feglm/.femlm, whose underlying")
logger.info("estimators have no weights= argument to substitute a replicate column")
logger.info("into) supports this the same way.")

N_REPLICATES = 20
mi_pf_boot = mi_ses_from_pyfixest.feols(
    df_implicates=with_bootstrap_weights(df_implicates, n_replicates=N_REPLICATES),
    fml="y ~ x1 + x2",
    replicates=Replicates(weight_stub="replicate_", n_replicates=N_REPLICATES, bootstrap=True),
)
mi_pf_boot.print()

# %%
logger.info("\n\npolars_ds - stays entirely in polars/narwhals, no pandas")
logger.info("conversion at all; doesn't compute a covariance matrix, so df_vcov")
logger.info("is always None here:")
(df_estimates, df_ses, df_vcov, df_tidy) = polars_ds_adapter(df, y="y", x=["x1", "x2"])
logger.info("\n   Single df")
logger.info(df_estimates)

mi_pds = mi_ses_from_polars_ds(df_implicates=df_implicates, y="y", x=["x1", "x2"])
logger.info("\n   Multiple imputation")
mi_pds.print()

# %%
logger.info("\n\nFor R or Stata instead, see basic_r.py/basic_stata.py (or")
logger.info("r_arbitrary_estimators.py/stata_arbitrary_estimators.py for the generic")
logger.info("escape hatch into any function either package provides).")
