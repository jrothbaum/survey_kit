from __future__ import annotations

import os
import numpy as np
import polars as pl
from dotenv import load_dotenv


from survey_kit import logger
from survey_kit.statistics.adapters import stata_adapter, mi_ses_from_stata
from survey_kit.statistics import _stata_interop as _st
from survey_kit.statistics.replicates import Replicates
from survey_kit.utilities.random import set_seed, RandomNumberGenerator
from sample_data import make_implicates, with_bootstrap_weights

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
logger.info("Before running any cell here, point survey_kit at your Stata install and")
logger.info("check what's importable:")
logger.info('    from survey_kit.statistics._stata_interop import check_stata_setup')
logger.info('    check_stata_setup(stata_path=r"C:\\Program Files\\Stata18")')
logger.info("")
logger.info("Requires `pip install survey-kit[stata]` (polars_readstat, for writing")
logger.info(".dta files) plus Stata 17+ for pystata itself.")




# %%
df_implicates = make_implicates()


# %%
logger.info("\n\nPart 1: mi_ses_from_stata - mi_ses_from_function(delegate=stata_adapter,")
logger.info("...), with stata_adapter's own arguments (command, edition, stata_path,")
logger.info("...) taken directly instead of packed into an arguments={} dict. Returns")
logger.info("the same shape every other adapter in survey_kit.statistics.adapters")
logger.info("does under the hood: (df_estimates, df_ses, df_vcov, df_tidy), combined")
logger.info("across implicates via Rubin's rules.")
logger.info("")
logger.info("Data goes into Stata as a .dta file written by polars_readstat, not")
logger.info("through pystata's own DataFrame transfer - this part IS verified (the")
logger.info(".dta round-trips correctly through polars_readstat's own reader), even")
logger.info("though the Stata-side execution below isn't.")

mi_reg = mi_ses_from_stata(
    df_implicates=df_implicates,
    command="regress y x1 x2",
    round_output=False,
)
mi_reg.print(round_output=False)


# %%
logger.info("\n\nWeighted regression, and a survey design set up per-implicate:")
logger.info("`command` can be a list of Stata commands run in order (run after")
logger.info("`use` but before the estimation command) instead of a single string -")
logger.info("only the LAST command's e(b)/e(V)/r(table) are read back.")

mi_svy = mi_ses_from_stata(
    df_implicates=df_implicates,
    #   svyset's pweight must be a variable, not a literal - replace
    #   with your actual design (real psu/weight/strata variables).
    command=[
        "gen _svy_weight = 1",
        "svyset _n [pw=_svy_weight]",
        "svy: regress y x1 x2",
    ],
    round_output=False,
)
mi_svy.print(round_output=False)


# %%
logger.info("\n\nReplicate-weight bootstrapping instead of Stata's own e(V): pass")
logger.info("replicates=, and `command` runs once per replicate weight column (via")
logger.info("stata_results_adapter under the hood, reading back e(b) only) rather")
logger.info("than Stata's own bootstrap/brr/jackknife prefix - the spread of")
logger.info("estimates across replicates IS the SE, computed in Python by")
logger.info("survey_kit's own Replicates/StatCalculator machinery. Stata's own")
logger.info("replicate-estimation commands are usually faster/more idiomatic if")
logger.info("you're already set up for them - this is for matching SEs computed the")
logger.info("same way elsewhere in a project instead. `command` needs a \"{weight}\"")
logger.info("placeholder here, unlike the e(V)-based calls above.")

N_REPLICATES = 20

mi_boot = mi_ses_from_stata(
    df_implicates=with_bootstrap_weights(df_implicates, n_replicates=N_REPLICATES),
    command="regress y x1 x2 [pw={weight}]",
    replicates=Replicates(weight_stub="replicate_", n_replicates=N_REPLICATES, bootstrap=True),
    round_output=False,
)
mi_boot.print(round_output=False)


# %%
logger.info("\n\nPart 2: calling this like any other delegate, standalone on one")
logger.info("dataset with no MI at all:")
(df_estimates, df_ses, df_vcov, df_tidy) = stata_adapter(
    df_implicates[0], command="regress y x1 x2",
)
logger.info(df_estimates)
logger.info(df_ses)
logger.info("df_tidy is Stata's own r(table), transposed to one row per term:")
logger.info(df_tidy)


# %%
logger.info("\n\nPart 3: reaching a Stata command stata_adapter's generic `command=`")
logger.info("string already covers by itself - `stata_adapter` IS the 'arbitrary")
logger.info("estimator' entry point here (unlike the R side, there's no separate")
logger.info("named-wrapper-per-function layer to route around): any e-class command")
logger.info("works by just changing the `command` string, including svy: prefixes,")
logger.info("xtreg/xtlogit/areg, or a community-installed command (`ssc install`)")
logger.info("survey_kit has never heard of. Two examples:")

logger.info("\n  Fixed effects via areg:")
mi_areg = mi_ses_from_stata(
    df_implicates=[
        d.with_columns((pl.arange(0, pl.len()) % 10).alias("firm"))
        for d in df_implicates
    ],
    command="areg y x1 x2, absorb(firm)",
    round_output=False,
)
mi_areg.print(round_output=False)

logger.info("\n  Logit (any e-class command works the same way):")


def _with_ybin(d: pl.DataFrame, seed: int) -> pl.DataFrame:
    if seed > 0:
        set_seed(seed)
    rng = RandomNumberGenerator()
    p = 1 / (1 + np.exp(-d["x1"].to_numpy()))
    ybin = rng.binomial(1, p).astype(np.int8)
    return d.with_columns(pl.Series("ybin", ybin))


mi_logit = mi_ses_from_stata(
    df_implicates=[_with_ybin(d, seed) for seed, d in enumerate(df_implicates)],
    command="logit ybin x1 x2",
    round_output=False,
)
mi_logit.print(round_output=False)


# %%
logger.info("\n\nPart 4: if you need something below stata_adapter's command-string")
logger.info("level - a custom pre/post-processing step around the .dta write, or a")
logger.info("different way of pulling results back than e(b)/e(V)/r(table) - the")
logger.info("primitives in _stata_interop are the same ones stata_adapter itself is")
logger.info("built from, and you can call them directly:")
logger.info("  - dataframe_to_dta(df, path)        - write via polars_readstat")
logger.info("  - require_pystata(edition, stata_path) - get a live Stata session")
logger.info("  - run_stata_model(df, command, ...) - the full write+use+run+extract")
logger.info("                                         stata_adapter wraps")
logger.info("  - run_stata_results(df, command, results, ...) - like")
logger.info("                                         run_stata_model, but for ANY")
logger.info("                                         named r()/e() result rather")
logger.info("                                         than the fixed e(b)/e(V)/")
logger.info("                                         r(table) triplet")
logger.info("")
logger.info("A custom example - grabbing a scalar result that isn't part of")
logger.info("e(b)/e(V)/r(table) at all (say, e(N) or e(r2)):")

raw = _st.run_stata_results(
    df_implicates[0],
    command="regress y x1 x2",
    results=["e(N)", "e(r2)"],
)
logger.info(raw)

logger.info("")
logger.info("adapters.stata_results_adapter wraps run_stata_results the same way")
logger.info("stata_adapter wraps run_stata_model - it works for ANY command")
logger.info("(r-class or e-class) and any named result, not just an e-class fit's")
logger.info("coefficient table. It's shaped as a StatCalculator.from_function")
logger.info("delegate (point estimates only, no vcov - the SE comes from")
logger.info("resampling across replicate weights, not Stata's own e(V)) - it's what")
logger.info("mi_ses_from_stata uses under the hood for the replicates= case above.")


# %%
logger.info("\n\nSummary / troubleshooting checklist for getting this working:")
logger.info("  1. check_stata_setup(stata_path=...) - confirms pystata and")
logger.info("     polars_readstat both import; doesn't guarantee config.init()")
logger.info("     succeeds too.")
logger.info("  2. If a command errors immediately with just a bare 'r(####);' and no")
logger.info("     explanation, pass quietly=False to stata_adapter/run_stata_model/")
logger.info("     run_stata_results to see Stata's own error text in the console.")
logger.info("  3. If e(b)/e(V) come back empty, the command you ran probably isn't")
logger.info("     e-class (didn't leave results in e()) - `ereturn list` right after")
logger.info("     running it interactively in Stata will confirm.")
logger.info("  4. If r(table) shapes/names look different than expected, `matrix list")
logger.info("     r(table)` interactively will show you its real shape.")
