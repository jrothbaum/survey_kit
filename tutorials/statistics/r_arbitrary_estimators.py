from __future__ import annotations

import numpy as np
import polars as pl

from survey_kit import logger
from survey_kit.statistics.multiple_imputation import mi_ses_from_function
from survey_kit.statistics.adapters import r_feols, mi_ses_from_r_fixest
from survey_kit.statistics.adapter_stats import AdapterStats
from survey_kit.statistics import _r_interop as _r
from survey_kit.statistics.replicates import Replicates
from sample_data import make_implicates, with_bootstrap_weights


# %%
logger.info("Every regression adapter in survey_kit.statistics.adapters is a plain")
logger.info("function returning an AdapterStats (a StatCalculator subclass built")
logger.info("directly from df_estimates/df_ses, optionally df_vcov/df_tidy too - see")
logger.info("survey_kit.statistics.adapter_stats). mi_ses_from_function() calls it")
logger.info("once per implicate and combines the results via Rubin's rules - there's")
logger.info("no special-casing for which package produced the estimates, and no")
logger.info("other return shape is accepted (build an AdapterStats even for a")
logger.info("one-off custom delegate, as Part 2 below does).")
logger.info("")
logger.info("This tutorial has two parts:")
logger.info("  1. A quick reminder of using a *named* R adapter (r_feols).")
logger.info("  2. How to reach ANY R function survey_kit hasn't wrapped - fixest and")
logger.info("     base lm()/glm() are just the ones with named wrappers; the plumbing")
logger.info("     underneath (survey_kit.statistics._r_interop) works for anything.")
logger.info("")
logger.info("Requires: R itself, plus rpy2/rpy2-arrow on the Python side")
logger.info("(`pip install survey-kit[r]`). Check your setup cheaply with:")
logger.info("    from survey_kit.statistics._r_interop import check_r_setup")
logger.info("    check_r_setup()")


# %%
df_implicates = make_implicates()


# %%
logger.info("\n\nPart 1: a named adapter (r_feols), via mi_ses_from_r_fixest - nothing")
logger.info("new here, just a reminder of the shape everything in this tutorial")
logger.info("produces.")

mi_feols = mi_ses_from_r_fixest.feols(
    df_implicates=df_implicates,
    formula="y ~ x1 + x2",
    round_output=False,
)
mi_feols.print(round_output=False)


# %%
logger.info("\n\nReplicate-weight bootstrapping instead of fixest's own vcov: pass")
logger.info("replicates=, and r_feols runs once per replicate weight column (point")
logger.info("estimates only, vcov forced to \"iid\" since it's discarded anyway) rather")
logger.info("than reading fixest's own SE - the spread of estimates across replicates")
logger.info("IS the SE, computed by survey_kit's own Replicates/StatCalculator")
logger.info("machinery. Each replicate weight column is passed via r_feols's own")
logger.info("`weight=` argument (no \"{weight}\" string placeholder needed in `formula`")
logger.info("the way Stata's raw `command` string needs one) - and each implicate is")
logger.info("converted to an R data.frame once, not once per replicate:")
logger.info("dataframe_to_r()'s passthrough-if-already-converted behavior (see the")
logger.info("caching section further below) makes that free to do with no")
logger.info("special-casing in r_feols itself.")

N_REPLICATES = 20

mi_boot = mi_ses_from_r_fixest.feols(
    df_implicates=with_bootstrap_weights(df_implicates, n_replicates=N_REPLICATES),
    formula="y ~ x1 + x2",
    replicates=Replicates(weight_stub="replicate_", n_replicates=N_REPLICATES, bootstrap=True),
    round_output=False,
)
mi_boot.print(round_output=False)


# %%
logger.info("\n\nPart 2: calling an R function survey_kit has no named adapter for.")
logger.info("Example: MASS::rlm() - robust (M-estimation) linear regression. MASS")
logger.info("ships with R itself, so this needs no extra R package install.")
logger.info("")
logger.info("rpy2's get_library(name) (a thin wrapper over importr()) returns an")
logger.info("object that already exposes every R function in that package as a")
logger.info("Python-callable attribute - dot-calling, the same way rpy2 always works.")
logger.info("So the pattern is just three calls into _r_interop:")
logger.info("  1. get_library(name)       - import (and cache) an R package; call")
logger.info("                                 its functions directly as attributes")
logger.info("  2. dataframe_to_r / formula - convert a polars df / formula string")
logger.info("                                 into the R objects the function needs")
logger.info("  3. extract_fit(fit)          - pull back coef()/vcov() from the")
logger.info("                                 already-fitted R object (works for ANY")
logger.info("                                 R model with those two generics - true")
logger.info("                                 of almost every R estimator)")
logger.info("  4. coef_table/ses_from_vcov/vcov_table - convert the raw rpy2")
logger.info("                                 objects into survey_kit's normalized")
logger.info("                                 tables")
logger.info("")
logger.info("No R call string to build, no RRaw escaping, no scratch global-env")
logger.info("variable - `fit` below is already a plain Python reference to the")
logger.info("fitted R object. Wrap that in a plain function that builds and returns")
logger.info("an AdapterStats and it's already a valid mi_ses_from_function delegate")
logger.info("- no need to add it to adapters.py unless you want to reuse it")
logger.info("elsewhere.")


def rlm_adapter(
    df,
    formula: str,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
) -> AdapterStats:
    mass = _r.get_library("MASS")

    fit = mass.rlm(_r.formula(formula), data=_r.dataframe_to_r(df))
    coef, vcov, _tidy = _r.extract_fit(fit)

    df_estimates = _r.coef_table(coef, join_on_name, value_name)
    df_ses = _r.ses_from_vcov(vcov, join_on_name, value_name)
    df_vcov = _r.vcov_table(vcov, join_on_name, value_name)

    return AdapterStats(
        df_estimates, df_ses, variable_ids=join_on_name, df_vcov=df_vcov, display=False
    )


# %%
logger.info("\n\nUse it exactly like any other delegate - standalone on one dataset:")
rlm_result = rlm_adapter(df_implicates[0], formula="y ~ x1 + x2")
rlm_result.print()

# %%
logger.info("\n\n...or across implicates, combined via Rubin's rules:")
mi_rlm = mi_ses_from_function(
    delegate=rlm_adapter,
    df_implicates=df_implicates,
    join_on=["Variable"],
    arguments={"formula": "y ~ x1 + x2"},
    round_output=False,
)
mi_rlm.print(round_output=False)


# %%
logger.info("\n\nIf you're going to fit SEVERAL models against the same implicates -")
logger.info("comparing specifications is a common workflow - converting each")
logger.info("implicate to R once and reusing that list is cheaper than letting every")
logger.info("mi_ses_from_function call re-convert the same data from scratch.")
logger.info("dataframe_to_r() materializes a real R data.frame (not a free")
logger.info("Arrow-backed view), and it passes an R object straight through")
logger.info("unchanged if you hand it one - so this needs no adapter code changes:")
logger.info("rlm_adapter works identically whether df is a polars frame or an")
logger.info("already-converted one.")

df_implicates_r = [_r.dataframe_to_r(dfi) for dfi in df_implicates]

mi_rlm_x1_only = mi_ses_from_function(
    delegate=rlm_adapter,
    df_implicates=df_implicates_r,
    join_on=["Variable"],
    arguments={"formula": "y ~ x1"},
    round_output=False,
)
mi_rlm_x1_only.print(round_output=False)

mi_rlm_both = mi_ses_from_function(
    delegate=rlm_adapter,
    df_implicates=df_implicates_r,
    join_on=["Variable"],
    arguments={"formula": "y ~ x1 + x2"},
    round_output=False,
)
mi_rlm_both.print(round_output=False)

logger.info("Same df_implicates_r list, two different formulas, no re-conversion")
logger.info("in between - dataframe_to_r() only ran once per implicate, back when")
logger.info("df_implicates_r was built above.")


# %%
logger.info("\n\nHow much does that actually save? Time it directly: run several")
logger.info("formulas against the same 5 implicates once with raw polars implicates")
logger.info("(re-converted to R on every single call) and once with the")
logger.info("df_implicates_r list already built above (converted once, total).")

import time

formulas_to_compare = ["y ~ x1", "y ~ x2", "y ~ x1 + x2", "y ~ x1 - 1", "y ~ x2 - 1"]

start = time.perf_counter()
for formulai in formulas_to_compare:
    mi_ses_from_function(
        delegate=rlm_adapter,
        df_implicates=df_implicates,
        join_on=["Variable"],
        arguments={"formula": formulai},
        round_output=False,
    )
elapsed_raw = time.perf_counter() - start

start = time.perf_counter()
for formulai in formulas_to_compare:
    mi_ses_from_function(
        delegate=rlm_adapter,
        df_implicates=df_implicates_r,
        join_on=["Variable"],
        arguments={"formula": formulai},
        round_output=False,
    )
elapsed_cached = time.perf_counter() - start

logger.info(
    f"{len(formulas_to_compare)} formulas x 5 implicates, raw polars "
    f"(re-converted every call): {elapsed_raw:.3f}s"
)
logger.info(
    f"{len(formulas_to_compare)} formulas x 5 implicates, pre-converted R "
    f"(converted once, reused): {elapsed_cached:.3f}s"
)
logger.info(f"Speedup: {elapsed_raw / elapsed_cached:.2f}x")
logger.info("")
logger.info("This tutorial's data is tiny (300 rows x 5 implicates), so rlm()'s own")
logger.info("fit time dominates and the gap here is modest - the win scales with")
logger.info("real data size (Arrow->R materialization cost grows with row/column")
logger.info("count) and with how many separate model calls reuse the same")
logger.info("implicates, which is exactly the shape of a real specification search.")


# %%
logger.info("\n\nA fancier example: quantreg::rq() for quantile regression. Needs an")
logger.info("extra R package (quantreg) - check first rather than finding out from a")
logger.info("cryptic mid-fit error:")
logger.info("    from survey_kit.statistics._r_interop import check_r_setup")
logger.info("    check_r_setup(['quantreg'])")
logger.info("")
logger.info("quantreg::rq() has no vcov() method at all (extract_fit tolerates that")
logger.info("gracefully - see its docstring) - the SE lives in summary() instead, so")
logger.info("this delegate builds both df_estimates and df_ses from a matrix table via")
logger.info("matrix_table() instead of coef_table()/ses_from_vcov(), using")
logger.info("extract_fit's tidy_fn= to call summary() directly (R's usual S3 dispatch")
logger.info("still applies to a dot-called generic like stats.summary(), so this")
logger.info("reaches summary.rq() exactly as calling summary(fit) would in R). Note")
logger.info("se='nid' - rq()'s summary() defaults to a rank-based CI table (columns:")
logger.info("coefficients/lower bd/upper bd) rather than the classical Value/Std.")
logger.info("Error/t value layout every other adapter in this tutorial produces;")
logger.info("se='nid' asks for that classical layout instead.")


def quantreg_adapter(
    df,
    formula: str,
    tau: float = 0.5,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
) -> AdapterStats:
    qr = _r.get_library("quantreg")

    fit = qr.rq(_r.formula(formula), data=_r.dataframe_to_r(df), tau=tau)
    #   rq() has no vcov() method at all - extract_fit() tolerates that and
    #   returns vcov=None rather than raising. Pull the coefficient table
    #   from summary(fit, se="nid") instead via tidy_fn - se="nid" asks for
    #   the classical Value/Std. Error/t value layout rather than rq()'s
    #   default rank-based CI table.
    base = _r.get_library("base")
    _coef, _vcov, tidy_matrix = _r.extract_fit(
        fit, tidy_fn=lambda f: base.summary(f, se="nid").rx2("coefficients")
    )
    df_tidy = _r.matrix_table(tidy_matrix, join_on_name)

    df_estimates = df_tidy.select(join_on_name, pl.col("Value").alias(value_name))
    df_ses = df_tidy.select(join_on_name, pl.col("Std. Error").alias(value_name))

    #   No vcov here (rq() doesn't have one - see above).
    return AdapterStats(
        df_estimates, df_ses, variable_ids=join_on_name, df_tidy=df_tidy, display=False
    )


# %%
logger.info("\n\nOnly run this cell if quantreg is installed (install.packages('quantreg')")
logger.info("in R) - it's not part of base R the way MASS is.")
setup = _r.check_r_setup(["quantreg"])
if setup["r_packages"].get("quantreg"):
    #   Still using df_implicates_r (built above) rather than df_implicates -
    #   quantreg_adapter's dataframe_to_r(df) call gets the passthrough for
    #   free, same as rlm_adapter did.
    mi_qr = mi_ses_from_function(
        delegate=quantreg_adapter,
        df_implicates=df_implicates_r,
        join_on=["Variable"],
        arguments={"formula": "y ~ x1 + x2", "tau": 0.5},
        round_output=False,
    )
    mi_qr.print(round_output=False)
else:
    logger.info("quantreg isn't installed - skipping (see the warning above for how).")


# %%
logger.info("\n\nSummary - to wire up any R estimator survey_kit doesn't already wrap:")
logger.info("  1. Find its R function and confirm it has coef()/vcov() methods (most")
logger.info("     do) - or a summary()/tidy-style table if not (see quantreg above).")
logger.info("  2. Write a small Python function: get_library(name) to import the R")
logger.info("     package, call its function directly as a Python attribute (formulas")
logger.info("     via formula(), data via dataframe_to_r() - everything else is a")
logger.info("     plain keyword argument, converted by rpy2 automatically), then")
logger.info("     extract_fit(fit) plus coef_table/ses_from_vcov/vcov_table/")
logger.info("     matrix_table to normalize the result, and build an AdapterStats")
logger.info("     from the pieces (that's the one shape mi_ses_from_function accepts).")
logger.info("  3. That function is already a valid mi_ses_from_function delegate, and")
logger.info("     already usable standalone on a single dataset with no MI at all.")
logger.info("")
logger.info("A few R arguments still need special handling rather than a plain")
logger.info("Python value: a one-sided formula like weights=~column (build it with")
logger.info("formula('~column')), or an argument that's itself an R function call")
logger.info("(e.g. fixest's ssc(fixef.K=\"full\") - call get_library('fixest').ssc(**")
logger.info("{'fixef.K': 'full'}) directly rather than writing it as a string). See")
logger.info("r_fixest_adapter/_fixest_fit in adapters.py for the older string-based")
logger.info("(_r_interop.call/RRaw/fit_r_model) approach those still use - it")
logger.info("remains available for cases where building a raw R call string is")
logger.info("genuinely more convenient than assembling the equivalent rpy2 objects.")
logger.info("")
logger.info("One more thing worth knowing if you call a delegate like rlm_adapter")
logger.info("repeatedly on the SAME underlying data - the multiple-formulas example")
logger.info("earlier in Part 2, or once per replicate weight via")
logger.info("StatCalculator.from_function, where the df object passed to the")
logger.info("delegate is identical across replicates and only which weight column is")
logger.info("referenced changes: dataframe_to_r() is real work (R materializes an")
logger.info("actual data.frame, not a free Arrow-backed view), and it passes an R")
logger.info("object straight through unchanged if you hand it one - convert once,")
logger.info("hold the result, and pass that instead of re-converting every call. No")
logger.info("cache to manage or clear - it's a plain Python variable, freed the")
logger.info("normal way (falls out of scope / reassigned) once you're done with it.")
logger.info("Stata's escape hatch (adapters.stata_adapter/stata_results_adapter) has")
logger.info("an analogous reuse_data=True option, but that one DOES need an explicit")
logger.info("clear_stata_cache() call afterward - Stata holds one single shared")
logger.info("in-memory dataset (a stateful resource), unlike R where dataframe_to_r")
logger.info("is just a pure conversion you can hold a reference to.")
