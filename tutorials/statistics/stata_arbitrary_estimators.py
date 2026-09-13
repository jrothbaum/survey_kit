from __future__ import annotations

import numpy as np
import polars as pl

from survey_kit import logger
from survey_kit.statistics.multiple_imputation import mi_ses_from_function
from survey_kit.statistics.adapters import stata_adapter
from survey_kit.statistics import _stata_interop as _st


# %%
logger.info("**UNTESTED** - written without a Stata installation available in the")
logger.info("environment this was developed in (pystata ships inside Stata 17+, not")
logger.info("on PyPI, so it couldn't be installed there to verify against). Expect to")
logger.info("need small fixes running this for real - see")
logger.info("survey_kit.statistics._stata_interop's module docstring for the specific")
logger.info("API points most likely to need adjustment, and treat this file as a")
logger.info("starting point rather than a guarantee.")
logger.info("")
logger.info("Before running any cell here, point survey_kit at your Stata install and")
logger.info("check what's importable:")
logger.info('    from survey_kit.statistics._stata_interop import check_stata_setup')
logger.info('    check_stata_setup(stata_path=r"C:\\Program Files\\Stata18")')
logger.info("")
logger.info("Requires `pip install survey-kit[stata]` (polars_readstat, for writing")
logger.info(".dta files) plus a licensed Stata 17+ install for pystata itself.")

STATA_PATH = r"C:\Program Files\Stata18"  # <-- change this to your install directory


# %%
def make_implicate(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1 + 2 * x1 - 1.5 * x2 + rng.normal(size=n) * 0.4
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


df_implicates = [make_implicate(seed) for seed in range(5)]


# %%
logger.info("\n\nPart 1: the named adapter (stata_adapter) - same shape every other")
logger.info("adapter in survey_kit.statistics.adapters returns:")
logger.info("(df_estimates, df_ses, df_vcov, df_tidy).")
logger.info("")
logger.info("Data goes into Stata as a .dta file written by polars_readstat, not")
logger.info("through pystata's own DataFrame transfer - this part IS verified (the")
logger.info(".dta round-trips correctly through polars_readstat's own reader), even")
logger.info("though the Stata-side execution below isn't.")

mi_reg = mi_ses_from_function(
    delegate=stata_adapter,
    df_implicates=df_implicates,
    join_on=["Variable"],
    arguments={
        "command": "regress y x1 x2",
        "stata_path": STATA_PATH,
    },
    round_output=False,
)
mi_reg.print(round_output=False)


# %%
logger.info("\n\nWeighted regression, and a survey design set up per-implicate via")
logger.info("pre_commands (run after `use` but before the estimation command):")

mi_svy = mi_ses_from_function(
    delegate=stata_adapter,
    df_implicates=df_implicates,
    join_on=["Variable"],
    arguments={
        "command": "svy: regress y x1 x2",
        "pre_commands": ["svyset _n [pw=1]"],  # replace with your actual design
        "stata_path": STATA_PATH,
    },
    round_output=False,
)
mi_svy.print(round_output=False)


# %%
logger.info("\n\nPart 2: calling this like any other delegate, standalone on one")
logger.info("dataset with no MI at all:")
(df_estimates, df_ses, df_vcov, df_tidy) = stata_adapter(
    df_implicates[0], command="regress y x1 x2", stata_path=STATA_PATH
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
mi_areg = mi_ses_from_function(
    delegate=stata_adapter,
    df_implicates=[
        d.with_columns((pl.arange(0, pl.len()) % 10).alias("firm"))
        for d in df_implicates
    ],
    join_on=["Variable"],
    arguments={"command": "areg y x1 x2, absorb(firm)", "stata_path": STATA_PATH},
    round_output=False,
)
mi_areg.print(round_output=False)

logger.info("\n  Logit (any e-class command works the same way):")
mi_logit = mi_ses_from_function(
    delegate=stata_adapter,
    df_implicates=[
        d.with_columns((pl.col("x1") > 0).cast(pl.Int8).alias("ybin"))
        for d in df_implicates
    ],
    join_on=["Variable"],
    arguments={"command": "logit ybin x1 x2", "stata_path": STATA_PATH},
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
    stata_path=STATA_PATH,
)
logger.info(raw)

logger.info("")
logger.info("adapters.stata_results_adapter wraps run_stata_results the same way")
logger.info("stata_adapter wraps run_stata_model - it works for ANY command")
logger.info("(r-class or e-class) and any named result, not just an e-class fit's")
logger.info("coefficient table. It's shaped as a StatCalculator.from_function")
logger.info("delegate (point estimates only, no vcov - the SE comes from")
logger.info("resampling across replicate weights, not Stata's own e(V)) rather")
logger.info("than an mi_ses_from_function delegate like stata_adapter - see its")
logger.info("docstring for the full reuse_data=/replicate-weight-loop story.")


# %%
logger.info("\n\nSummary / troubleshooting checklist for getting this working:")
logger.info("  1. check_stata_setup(stata_path=...) - confirms pystata and")
logger.info("     polars_readstat both import; doesn't guarantee config.init()")
logger.info("     succeeds (e.g. an expired license would still fail there).")
logger.info("  2. If `stata.run(...)` errors immediately, try it with quietly=False")
logger.info("     to see Stata's own error text in the console.")
logger.info("  3. If e(b)/e(V) come back empty, the command you ran probably isn't")
logger.info("     e-class (didn't leave results in e()) - `ereturn list` right after")
logger.info("     running it interactively in Stata will confirm.")
logger.info("  4. If r(table) shapes/names look different than expected here, that's")
logger.info("     the piece flagged as least certain in _stata_interop's docstring -")
logger.info("     `matrix list r(table)` interactively will show you its real shape.")
logger.info("  5. Please report back what needed changing - this file (and")
logger.info("     _stata_interop.py/stata_adapter) should get their 'untested'")
logger.info("     caveats removed once someone's actually run them against Stata.")
