"""
Minimal, self-contained rpy2 plumbing shared by the R-based adapters in
adapters.py. Not a public API - just enough to call into R, cache slow
package imports across calls, and convert data/coefficient tables back and
forth.

rpy2 (and R itself) are optional - nothing here is imported at module load
time, and every entry point raises a clear ImportError naming the install
command if rpy2 or the requested R package isn't available. This module
never installs anything into R itself; that's left to the user.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import polars as pl

from .. import logger


class RRaw(str):
    """
    Marks a string as already-valid R code to be emitted verbatim (unquoted)
    rather than treated as an R string literal - the escape hatch for
    py_to_r_literal/call. e.g. RRaw('ssc(fixef.K="full")') for an argument
    that takes a function call rather than a plain scalar/string/formula.
    """


def py_to_r_literal(value) -> str:
    """
    Convert a Python value to an R literal for building an R call string.

    - None -> omitted by call entirely (not passed as NULL - see there)
    - bool -> TRUE/FALSE
    - int/float -> as-is
    - str starting with "~" -> emitted raw (a one-sided formula, e.g. for
      fixest's cluster/panel.id/weights - by far the most common reason to
      need "raw" R code in this passthrough) - anything else stringy that
      needs to be raw (a function call like `ssc(fixef.K="full")`, a
      two-sided formula, ...) should be wrapped in RRaw(...) explicitly
    - other str -> a quoted R string literal (via json.dumps, whose
      double-quote/backslash/escape rules are a compatible subset of R's)
    - list/tuple -> an R vector: c(...)
    - dict -> an R named list: list(name=value, ...)
    """
    if isinstance(value, RRaw):
        return str(value)
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if value.strip().startswith("~"):
            return value
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "c(" + ", ".join(py_to_r_literal(v) for v in value) + ")"
    if isinstance(value, dict):
        return "list(" + ", ".join(f"{k}={py_to_r_literal(v)}" for k, v in value.items()) + ")"

    message = (
        f"Don't know how to pass {value!r} ({type(value).__name__}) to R - "
        "wrap it in RRaw(...) with the R code you want emitted verbatim."
    )
    logger.error(message)
    raise TypeError(message)


def call(fn: str, *raw_positional: str, **kwargs) -> str:
    """
    Build an R function-call expression string.

    Positional args are inserted verbatim (already R code, e.g. a formula):
        call("lm", "y ~ x1 + x2", data="{df}")
        -> 'lm(y ~ x1 + x2, data={df})'

    Keyword args are converted via py_to_r_literal; a value of None omits
    that argument entirely (so a delegate can pass every fixest/lm keyword
    it knows about and let the ones the caller didn't set just vanish,
    rather than forcing every unset argument to be reasoned about as an
    explicit R NULL, which isn't always equivalent to the argument being
    absent):
        call("fixest::feols", "y ~ x1 + x2", data="{df}",
               cluster="firm", weights=None)
        -> 'fixest::feols(y ~ x1 + x2, data={df}, cluster="firm")'
    """
    parts = list(raw_positional)
    for key, value in kwargs.items():
        if value is None:
            continue
        parts.append(f"{key}={py_to_r_literal(value)}")
    return f"{fn}({', '.join(parts)})"

#   Caches the (slow - often multiple seconds) R package imports so repeated
#   adapter calls across implicates don't reload the same package each time.
_loaded_r_packages: dict = {}

#   R packages the interop layer itself needs, independent of whichever
#   regression package (fixest, ...) a given adapter also requires. Data
#   moves in via the Arrow C Data Interface (rpy2-arrow), not reticulate -
#   reticulate is unreliable when R is itself embedded inside a Python
#   process via rpy2 (segfaults and silent discovery failures are both
#   documented upstream: rstudio/reticulate#98, #972; rpy2/rpy2#470, #942).
_INTEROP_R_PACKAGES = ["arrow"]


def check_r_setup(r_packages: list[str] | None = None) -> dict:
    """
    Cheaply check what's available for the R-based adapters, without
    starting R interactively, embedding it via rpy2, or loading any
    regression package. Logs an actionable summary (what's missing and how
    to install it) and returns the same information as a dict.

    Meant to be run once, up front, while setting up an R-based workflow -
    not called automatically before every adapter call, since spawning
    Rscript to check has real (if small, ~O(100-300ms)) cost you don't want
    to pay per-implicate.

    Parameters
    ----------
    r_packages : list[str] | None, optional
        R package names to check for, in addition to the interop layer's
        own requirements (reticulate, arrow). Pass the regression package(s)
        you plan to use, e.g. ["fixest"]. Default is None (interop packages
        only).

    Returns
    -------
    dict
        Keys: rscript_found, rscript_path, r_version, rpy2_installed,
        rpy2_version, r_packages (dict[str, bool]), missing (list of
        (what, how_to_install) tuples - empty if everything's ready).
    """
    r_packages = _INTEROP_R_PACKAGES + list(r_packages or [])
    #   De-dupe while preserving order (a caller might re-pass "reticulate").
    r_packages = list(dict.fromkeys(r_packages))

    report = {
        "rscript_found": False,
        "rscript_path": None,
        "r_version": None,
        "rpy2_installed": False,
        "rpy2_version": None,
        "rpy2_arrow_installed": False,
        "rpy2_arrow_version": None,
        "r_packages": {name: False for name in r_packages},
        "missing": [],
    }

    rscript_path = shutil.which("Rscript")
    report["rscript_found"] = rscript_path is not None
    report["rscript_path"] = rscript_path
    if not report["rscript_found"]:
        report["missing"].append(
            (
                "R",
                "install R (e.g. from https://cloud.r-project.org/) and make "
                "sure `Rscript` is on PATH.",
            )
        )

    import importlib.metadata

    try:
        report["rpy2_version"] = importlib.metadata.version("rpy2")
        report["rpy2_installed"] = True
    except importlib.metadata.PackageNotFoundError:
        report["missing"].append(("rpy2", "`uv add rpy2` (or `pip install survey-kit[r]`)"))

    try:
        report["rpy2_arrow_version"] = importlib.metadata.version("rpy2-arrow")
        report["rpy2_arrow_installed"] = True
    except importlib.metadata.PackageNotFoundError:
        report["missing"].append(
            ("rpy2-arrow", "`uv add rpy2-arrow` (or `pip install survey-kit[r]`)")
        )

    if report["rscript_found"]:
        check_expr = "; ".join(
            [
                "cat(R.version.string, '\\n')",
                "pkgs <- rownames(installed.packages())",
            ]
            + [f'cat("{p}:", "{p}" %in% pkgs, "\\n")' for p in r_packages]
        )
        try:
            result = subprocess.run(
                [rscript_path, "--vanilla", "-e", check_expr],
                capture_output=True,
                text=True,
                timeout=30,
            )
            lines = result.stdout.strip().splitlines()
            if lines:
                report["r_version"] = lines[0]
            for line in lines[1:]:
                name, _, present = line.partition(":")
                name = name.strip()
                if name in report["r_packages"]:
                    is_present = present.strip() == "TRUE"
                    report["r_packages"][name] = is_present
                    if not is_present:
                        report["missing"].append(
                            (
                                f"R package '{name}'",
                                f"install.packages('{name}') in R",
                            )
                        )
        except Exception as e:
            logger.warning(f"Could not query installed R packages: {e}")

    if report["missing"]:
        logger.warning("R interop setup is incomplete:")
        for what, how in report["missing"]:
            logger.warning(f"  - missing {what} -> install with: {how}")
    else:
        logger.info(
            "R interop setup looks complete: "
            f"R {report['r_version']}, rpy2 {report['rpy2_version']}, "
            f"rpy2-arrow {report['rpy2_arrow_version']}, "
            + ", ".join(f"{p} installed" for p in r_packages)
        )

    return report


def require_rpy2():
    try:
        import rpy2.robjects as ro
    except ImportError as e:
        message = (
            "This requires the 'rpy2' package and a working R installation - "
            "install rpy2 with `uv add rpy2` (R itself must already be "
            "installed separately and discoverable by rpy2)."
        )
        logger.error(message)
        raise ImportError(message) from e
    return ro


def get_library(name: str):
    """
    Import (and cache) an R package via rpy2, e.g. get_library("fixest").
    Raises ImportError with the R install command if the package isn't
    available in R - this never calls install.packages() itself.
    """
    if name in _loaded_r_packages:
        return _loaded_r_packages[name]

    import rpy2.robjects.packages as rpackages
    from rpy2.rinterface_lib.embedded import RRuntimeError

    try:
        package = rpackages.importr(name)
    except RRuntimeError as e:
        message = (
            f"R package '{name}' is not installed/available - install it in "
            f"R with install.packages('{name}')."
        )
        logger.error(message)
        raise ImportError(message) from e

    _loaded_r_packages[name] = package
    return package


def require_rpy2_arrow():
    from rpy2.robjects.packages import PackageNotInstalledError

    try:
        import rpy2_arrow.arrow as pyra
    except PackageNotInstalledError as e:
        #   rpy2_arrow.arrow imports R's own 'arrow' package at import time
        #   (see rpy2_arrow/arrow.py), so a missing R package surfaces here
        #   as an ImportError on the *Python* module even though rpy2-arrow
        #   itself is installed fine - distinguish the two so the message
        #   points at the actual problem.
        message = (
            "The 'rpy2-arrow' Python package is installed, but R's own "
            "'arrow' package isn't - install it with install.packages"
            "('arrow') in R."
        )
        logger.error(message)
        raise ImportError(message) from e
    except ImportError as e:
        message = (
            "This requires the 'rpy2-arrow' package - install it with "
            "`uv add rpy2-arrow` (R's 'arrow' package must also be "
            "installed: install.packages('arrow') in R)."
        )
        logger.error(message)
        raise ImportError(message) from e
    return pyra


def dataframe_to_r(df):
    """
    Convert a polars/pandas/narwhals-native dataframe to an R data.frame via
    the Arrow C Data Interface (rpy2-arrow) - zero-copy out of Python, no
    pandas round-trip, and no reticulate (see the module docstring above for
    why reticulate isn't used here).

    If `df` is already an R object (e.g. the return value of a previous
    dataframe_to_r call), it's returned unchanged rather than re-converted.
    This conversion is real work (R materializes an actual data.frame, not
    a free Arrow-backed view - most R functions need the former), so a
    caller that will reuse the same data across several calls (e.g. once
    per replicate weight in a StatCalculator.from_function loop, where the
    underlying df object is identical across replicates and only the
    weight column differs) can convert once, hold the result in a
    variable, and pass that directly on later calls instead of the
    original polars/narwhals df - no separate cache to manage, since
    Python's own variable scoping already owns the object's lifetime.
    """
    ro = require_rpy2()
    import rpy2.rinterface as rinterface

    if isinstance(df, rinterface.Sexp):
        return df

    pyra = require_rpy2_arrow()
    get_library("arrow")  # registers as.data.frame.Table's S3 dispatch

    import narwhals as nw

    arrow_table = nw.from_native(df).lazy().collect().to_arrow()

    r_table = pyra.converter.py2rpy(arrow_table)
    return ro.r["as.data.frame"](r_table)


def formula(expr: str):
    """
    Wrap a formula string as an actual R formula object (`ro.Formula`), for
    passing directly to a dot-called R function, e.g.:
        mass = get_library("MASS")
        mass.rlm(formula("y ~ x1 + x2"), data=dataframe_to_r(df))
    rpy2's importr() already exposes every function in an R package as a
    Python-callable attribute (dots in the R name become underscores) and
    converts plain Python scalars/strings/lists automatically - a formula is
    the one common argument type that still needs explicit wrapping, since
    R evaluates it unevaluated (non-standard evaluation) rather than as a
    plain string.
    """
    ro = require_rpy2()
    return ro.Formula(expr)


def extract_fit(fit, tidy_fn=None):
    """
    Given an already-fitted R model object - the return value of calling a
    dot-mapped R function directly, e.g.
    `mass.rlm(formula(expr), data=dataframe_to_r(df))` - pull back
    coef()/vcov() (and optionally a tidy coefficient table), the same result
    shape fit_r_model() produces from a fit_code string, but without
    building any R call string or stashing anything in R's global
    environment: `fit` is already a Python-side reference to the R object,
    so R's own coef()/vcov()/summary() generics can be called on it
    directly via rpy2's dot-calling.

    Parameters
    ----------
    tidy_fn : a Python callable taking `fit` and returning the package's own
        tidy coefficient matrix/table (term names as rownames), e.g.
        `lambda f: get_library("stats").summary(f).rx2("coefficients")`
        for base R, or `lambda f: get_library("fixest").coeftable(f)` for
        any fixest model - the dot-call equivalent of fit_r_model's
        tidy_code string. Default is None (skip).

    Returns
    -------
    tuple
        (coef, vcov, tidy) - rpy2 R objects (tidy is None if tidy_fn wasn't
        given); pass to coef_table/vcov_table/matrix_table. vcov is
        subset down to exactly coef's terms, in coef's order (see
        fit_r_model's docstring for why), or None if the model has no
        vcov() method at all (e.g. quantreg::rq() - its SE lives in
        summary() instead, via tidy_fn).
    """
    from rpy2.rinterface_lib.embedded import RRuntimeError

    stats = get_library("stats")
    coef = stats.coef(fit)
    try:
        vcov_full = stats.vcov(fit)
        terms = get_library("base").names(coef)
        #   drop=False: R's default `[` behavior collapses a 1x1 result
        #   (a single-coefficient model, e.g. "y ~ x1 - 1") down to a plain
        #   vector, losing rownames - coef_table/ses_from_vcov/vcov_table
        #   all need vcov to stay a matrix regardless of term count.
        vcov = vcov_full.rx(terms, terms, drop=False)
    except RRuntimeError as e:
        logger.info(
            "extract_fit: this model has no vcov() method "
            f"({e}) - continuing with vcov=None."
        )
        vcov = None
    tidy = tidy_fn(fit) if tidy_fn is not None else None
    return coef, vcov, tidy


def fit_r_model(df, fit_code: str, tidy_code: str | None = None):
    """
    Convert df to an R data.frame, fit a model with it, and pull back the
    fitted model's coef()/vcov() (and optionally a "tidy" coefficient
    table). `fit_code` is an R expression string that fits the model, with
    "{df}" as a placeholder for the R variable name holding the converted
    data, e.g.:

        fit_r_model(df, 'lm(y ~ x1 + x2, data={df}, weights=w)')
        fit_r_model(df, 'fixest::feols(y ~ x1 + x2, data={df}, vcov="hetero")')

    Uses fixed, prefixed scratch names in R's global environment for the
    data/fit objects and removes them afterward - fine for the sequential,
    one-call-at-a-time usage pattern of an mi_ses_from_function delegate;
    not meant for concurrent use.

    Parameters
    ----------
    tidy_code : an R expression string, with "{fit}" as a placeholder for
        the fitted model object, that returns the package's own tidy
        coefficient matrix/table (term names as rownames), e.g.
        'summary({fit})$coefficients' for base R, or
        'fixest::coeftable({fit})' for any fixest model. Evaluated (and
        cleaned up) inside the same try/finally as the fit itself, so it's
        available before `.survey_kit_fit` is removed. Default is None
        (skip - not every caller wants this).

    Returns
    -------
    tuple
        (coef, vcov, tidy) - rpy2 R objects (tidy is None if tidy_code
        wasn't given); pass to coef_table/vcov_table/matrix_table.
        vcov is subset down to exactly coef's terms, in coef's order, even
        if the model's own vcov() includes extra nuisance/dispersion terms
        coef() doesn't (e.g. fixest::femlm(family="negbin") includes a
        ".theta" row/column in vcov() that coef() omits) - without this,
        coef and vcov would disagree on how many terms there are, silently
        corrupting the term alignment downstream.
    """
    ro = require_rpy2()
    from rpy2.rinterface_lib.embedded import RRuntimeError

    #   R identifiers can't start with "_" (unlike Python) - a leading "."
    #   is R's own convention for a scratch/hidden name instead.
    df_r = dataframe_to_r(df)
    ro.globalenv[".survey_kit_df"] = df_r
    try:
        ro.r(f".survey_kit_fit <- {fit_code.format(df='.survey_kit_df')}")
        coef = ro.r("coef(.survey_kit_fit)")
        try:
            vcov = ro.r(
                "{v <- vcov(.survey_kit_fit); v[names(coef(.survey_kit_fit)), "
                "names(coef(.survey_kit_fit)), drop=FALSE]}"
            )
        except RRuntimeError as e:
            #   Not every model class implements vcov() (e.g. quantreg's
            #   rq() has none by default - its SE lives in summary()
            #   instead, via tidy_code). Missing coef() would be a real
            #   problem; missing vcov() just means no df_vcov this time.
            logger.info(
                "fit_r_model: this model has no vcov() method "
                f"({e}) - continuing with vcov=None."
            )
            vcov = None
        tidy = (
            ro.r(tidy_code.format(fit=".survey_kit_fit"))
            if tidy_code is not None
            else None
        )
    finally:
        ro.r(
            "rm(list=intersect(c('.survey_kit_df', '.survey_kit_fit'), "
            "ls(envir=globalenv())), envir=globalenv())"
        )
    return coef, vcov, tidy


def coef_table(coef, join_on_name: str, value_name: str) -> pl.DataFrame:
    """coef: an R named numeric vector (e.g. from R's coef())."""
    return pl.DataFrame({join_on_name: list(coef.names), value_name: list(coef)})


def matrix_table(matrix, join_on_name: str) -> pl.DataFrame:
    """
    Convert an R numeric matrix with rownames (terms) and colnames (stat
    names, e.g. "Estimate"/"Std. Error"/"t value"/"Pr(>|t|)") into a polars
    DataFrame - one row per term, one column per original R column name,
    used as-is (this is meant to hand back the package's own native
    coefficient/summary table, not to normalize its column names).
    """
    import numpy as np

    terms = list(matrix.rownames)
    stat_names = list(matrix.colnames)
    values = np.array(matrix)
    data = {join_on_name: terms}
    for j, name in enumerate(stat_names):
        data[name] = values[:, j].tolist()
    return pl.DataFrame(data)


def ses_from_vcov(vcov, join_on_name: str, value_name: str) -> pl.DataFrame:
    """
    Standard errors as sqrt(diag(vcov)) - derived from the same vcov() call
    already needed for vcov_table, rather than a separate summary() call,
    so the SE and the covariance matrix are guaranteed consistent.
    """
    import numpy as np

    terms = list(vcov.rownames)
    diag = np.diag(np.array(vcov))
    return pl.DataFrame({join_on_name: terms, value_name: np.sqrt(diag).tolist()})


def vcov_table(vcov, join_on_name: str, value_name: str) -> pl.DataFrame:
    """vcov: an R named square matrix (e.g. from R's vcov())."""
    import numpy as np

    terms = list(vcov.rownames)
    values = np.array(vcov)  # np.array() respects R's dim attribute correctly
    records = [
        {
            f"{join_on_name}_1": t1,
            f"{join_on_name}_2": t2,
            value_name: float(values[i, j]),
        }
        for i, t1 in enumerate(terms)
        for j, t2 in enumerate(terms)
    ]
    return pl.DataFrame(records)
