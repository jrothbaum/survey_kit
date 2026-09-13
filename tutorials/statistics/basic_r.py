from __future__ import annotations

from survey_kit import logger
from survey_kit.statistics.adapters import r_feols, mi_ses_from_r_fixest
from sample_data import make_implicates

# %%
logger.info("The simplest way to run an R regression from survey_kit: r_feols(),")
logger.info("a wrapper around fixest::feols() with robust SEs by default. Requires R")
logger.info("itself plus rpy2/rpy2-arrow (`pip install survey-kit[r]`) and the R")
logger.info("'fixest' package. Check your setup cheaply with:")
logger.info("    from survey_kit.statistics._r_interop import check_r_setup")
logger.info("    check_r_setup(['fixest'])")

# %%
df_implicates = make_implicates()
logger.info(f"\n\nSample data: {len(df_implicates)} implicates, {df_implicates[0].height}")
logger.info("rows each (y = 1 + 2*x1 - 1.5*x2 + noise) - see sample_data.py.")

# %%
logger.info("\n\nOn one dataset, standalone - no MI at all:")
(df_estimates, df_ses, df_vcov, df_tidy) = r_feols(df_implicates[0], formula="y ~ x1 + x2")
logger.info(df_estimates)
logger.info(df_ses)


# %%
logger.info("\n\nAcross multiple imputed datasets, combined via Rubin's rules -")
logger.info("mi_ses_from_r_fixest.feols(...) runs r_feols once per implicate and")
logger.info("combines the results, taking r_feols's own arguments directly:")
mi_reg = mi_ses_from_r_fixest.feols(
    df_implicates=df_implicates,
    formula="y ~ x1 + x2",
    round_output=False,
)
mi_reg.print(round_output=False)


# %%
logger.info("\n\nThat's it for the common case. mi_ses_from_r_fixest also has")
logger.info(".feglm/.fepois/.femlm for fixest's other estimators, all with the same")
logger.info("shape. r_lm_adapter (base R lm()/glm()) and r_fixest_adapter (any")
logger.info("fixest estimator via a func= string, including ones .feglm/.fepois/")
logger.info(".femlm don't cover) are the lower-level pieces these are built from -")
logger.info("see r_arbitrary_estimators.py for rolling your own with those, plus a")
logger.info("generic escape hatch into any R package/function at all.")
