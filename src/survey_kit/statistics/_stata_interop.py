"""
Minimal, self-contained pystata plumbing shared by the Stata-based adapter
in adapters.py. Mirrors _r_interop.py's shape and division of labor.

**UNTESTED**: written without a Stata installation available in this
development environment - pystata ships inside a Stata 17+ install (in its
`utilities` subfolder), not on PyPI, so it can't be installed here to
verify against. Everything below is based on StataCorp's documented
pystata/sfi API surface. The parts most likely to need adjustment once run
against a real Stata instance:

- Whether `sfi.Matrix.get("r(table)")` returns exactly the 9-row layout
  (b/se/z-or-t/pvalue/ll/ul/df/crit/eform) documented for postestimation
  use, or something that needs a small reshape - test against a plain
  `regress` first.
- `pystata.config.init(edition)` does NOT auto-detect the edition from the
  license when left None - it raises `ValueError('Stata edition must be one
  of be, se, or mp')` - confirmed against a real Stata 17 SE install. Every
  caller must pass edition="be"/"se"/"mp" explicitly, matching whichever
  edition's exe (e.g. StataSE-64.exe) is present in the Stata install dir.
- Quoting/escaping of the temp .dta path in the generated `use "..."`
  command on Windows (backslashes) - str(Path) should already produce
  forward slashes that Stata accepts, but worth a first check.

Data moves into Stata via a .dta file written by polars_readstat's
`write_readstat`, not through pystata's own DataFrame transfer
(`pystata.stata.pdataframe_to_data`) - by design (see the survey_kit
conversation this was built from): `.dta` is Stata's own native, most
battle-tested ingestion path, and this keeps survey_kit independent of
whichever transfer mechanism pystata favors internally.
"""

from __future__ import annotations

import os
import sys
import tempfile

from .. import config, logger

_stata_initialized = False


def check_stata_setup(stata_path: str | None = None) -> dict:
    """
    Cheaply check what's available for the Stata adapter, without
    initializing Stata. Stata isn't discoverable on PATH the way R's
    Rscript is, so pystata usually needs its `utilities` subfolder added to
    sys.path before it can be imported - pass `stata_path` (your Stata
    install directory, e.g. r"C:\\Program Files\\Stata18") to check
    that and add it for you, or add it yourself beforehand and call this
    with no argument.

    Parameters
    ----------
    stata_path : str | None, optional
        Path to your Stata installation (the directory containing
        `utilities`, `StataMP-64.exe`, etc.), or None to fall back to
        `survey_kit.config.stata_path` (env var `_survey_kit_stata_path_`)
        and, if that's unset too, only check what's already importable.
        Default is None.

    Returns
    -------
    dict
        Keys: stata_path, utilities_dir, utilities_found, pystata_importable,
        polars_readstat_importable, missing (list of (what, how_to_fix)
        tuples - empty if everything looks ready, though pystata importing
        successfully doesn't guarantee `config.init()` will succeed too,
        e.g. an expired license would still fail at that step).
    """
    if stata_path is None:
        stata_path = config.stata_path or None

    report = {
        "stata_path": stata_path,
        "utilities_dir": None,
        "utilities_found": False,
        "pystata_importable": False,
        "polars_readstat_importable": False,
        "missing": [],
    }

    if stata_path:
        utilities_dir = os.path.join(stata_path, "utilities")
        report["utilities_dir"] = utilities_dir
        report["utilities_found"] = os.path.isdir(utilities_dir)
        if report["utilities_found"]:
            if utilities_dir not in sys.path:
                sys.path.insert(0, utilities_dir)
        else:
            report["missing"].append(
                (
                    f"Stata 'utilities' folder at {utilities_dir}",
                    "check stata_path points at your Stata install directory",
                )
            )

    try:
        import pystata  # noqa: F401

        report["pystata_importable"] = True
    except ImportError:
        report["missing"].append(
            (
                "pystata",
                "pass stata_path=<your Stata install dir> here, or add "
                "'<Stata install dir>/utilities' to sys.path yourself - "
                "pystata ships inside Stata 17+, not on PyPI",
            )
        )

    try:
        import polars_readstat  # noqa: F401

        report["polars_readstat_importable"] = True
    except ImportError:
        report["missing"].append(("polars_readstat", "`uv add polars_readstat`"))

    if report["missing"]:
        logger.warning("Stata interop setup is incomplete:")
        for what, how in report["missing"]:
            logger.warning(f"  - missing {what} -> fix with: {how}")
    else:
        logger.info(
            "Stata interop setup looks importable (pystata, polars_readstat) "
            "- config.init() itself is only tried on first actual use."
        )

    return report


def require_polars_readstat():
    try:
        import polars_readstat
    except ImportError as e:
        message = (
            "This requires the 'polars_readstat' package - install it with "
            "`uv add polars_readstat` (or `pip install survey-kit[stata]`)."
        )
        logger.error(message)
        raise ImportError(message) from e
    return polars_readstat


def require_pystata(edition: str | None = None, stata_path: str | None = None):
    """
    Import (if needed, adding `stata_path`'s utilities folder to sys.path
    first) and initialize pystata. Caches initialization across calls -
    like R package loading, this is slow enough (a real Stata instance
    starting up) that repeated per-implicate calls should only pay it once.

    `edition`/`stata_path` fall back to `survey_kit.config.stata_edition`/
    `survey_kit.config.stata_path` (env vars `_survey_kit_stata_edition_`/
    `_survey_kit_stata_path_`) when not passed. Raises RuntimeError
    immediately if no edition is available from either source, rather than
    letting pystata's own `config.init(None)` fail with a cryptic
    ValueError.
    """
    global _stata_initialized

    if stata_path is None:
        stata_path = config.stata_path or None
    if edition is None:
        edition = config.stata_edition or None

    if not edition:
        message = (
            "Stata edition is required - pass edition='be'/'se'/'mp' "
            "(matching your license) to the adapter, or set it once via "
            "survey_kit.config.stata_edition (env var "
            "_survey_kit_stata_edition_)."
        )
        logger.error(message)
        raise RuntimeError(message)

    if stata_path:
        utilities_dir = os.path.join(stata_path, "utilities")
        if utilities_dir not in sys.path:
            sys.path.insert(0, utilities_dir)

    try:
        from pystata import config as pystata_config
    except ImportError as e:
        message = (
            "This requires Stata 17+ and its bundled 'pystata' package - "
            "pystata ships inside your Stata installation's 'utilities' "
            "subfolder, not on PyPI. Pass stata_path=<your Stata install "
            "dir> to the adapter, or add that folder to sys.path yourself "
            "before calling it. See check_stata_setup() for a diagnostic."
        )
        logger.error(message)
        raise ImportError(message) from e

    if not _stata_initialized:
        pystata_config.init(edition)
        #   pystata's default streamout='on' polls Stata's output buffer
        #   from a background thread while the main thread is still
        #   executing the command - pystata's embedded Stata engine isn't
        #   safe for that cross-thread sfi access (can surface as "Unable
        #   to find thread to evaluate variable reference", or as an error
        #   raised with only a truncated tail of the real message, e.g. a
        #   bare "r(2000);" with the actual explanation lost). Capturing
        #   output synchronously in the same thread avoids both.
        pystata_config.set_streaming_output_mode("off")
        _stata_initialized = True

    #   pystata.stata calls config.check_initialized() at import time, so it
    #   can only be imported after config.init() above has actually run.
    from pystata import stata

    return stata


def dataframe_to_dta(df, path: str) -> None:
    """Write df to a .dta file via polars_readstat - not pystata's own data transfer."""
    polars_readstat = require_polars_readstat()
    import narwhals as nw

    df_pl = nw.from_native(df).lazy().collect().to_polars()
    polars_readstat.write_readstat(df_pl, path, format="dta")


#   Holds a strong reference to the df object currently `use`d in Stata
#   when reuse_data=True - identity is checked against this reference
#   (not a bare id(df)), since id() can be recycled by the garbage
#   collector for an unrelated object and would otherwise risk silently
#   reusing the wrong dataset.
_stata_loaded_df: object | None = None


def _run_in_stata(
    df,
    command: str,
    pre_commands: list[str] | None,
    edition: str | None,
    stata_path: str | None,
    reuse_data: bool = False,
    quietly: bool = True,
):
    """
    Write df to a temp .dta, `use` it in the running (persistent, embedded)
    Stata instance, then run any `pre_commands` followed by `command`.
    Shared setup behind run_stata_model/run_stata_results - callers read
    back whatever e()/r() results they need via `sfi` afterward (the temp
    .dta is only needed for the `use`, so it's fine for the
    TemporaryDirectory to clean up before that read - sfi reads from
    Stata's own memory, not the file).

    pystata embeds a single long-running Stata process in this Python
    process (the same model as rpy2 embedding R), so `e()`/`r()` results,
    globals, and locals from a *previous* call persist here unless cleared
    - `use ..., clear` only replaces the in-memory dataset. Without
    `ereturn clear`/`return clear`, a caller asking for a result name that
    this call's `command` doesn't happen to populate would silently get a
    stale value left over from an earlier call instead of an error - a
    real risk for run_stata_results's generic named-result lookup used in
    a bootstrap/replicate loop that calls the same command repeatedly.

    Parameters
    ----------
    reuse_data : if True and `df` is the exact same object (by identity)
        as the one loaded by the previous reuse_data=True call, skip
        rewriting/re-`use`-ing the .dta entirely and just rerun
        pre_commands/command against the dataset already sitting in
        Stata's memory. Useful in a replicate-weight loop, where
        StatCalculator.from_function/mi_ses_from_function pass the exact
        same df object across every replicate call for one implicate (only
        the weight column referenced in `command` changes) - re-exporting
        and re-`use`-ing identical data on every single replicate is pure
        waste. Default is False (always reload): unlike the R side's
        dataframe_to_r (a pure conversion, safe to always skip when
        redundant), skipping `use` here also skips replacing Stata's one
        shared in-memory dataset, so this needs an explicit opt-in from a
        caller who knows the same object will really be reused - call
        clear_stata_cache() once done to free it.
    quietly : suppresses Stata's own console output for `pre_commands`/
        `command` on success (default True) - avoids dumping a full
        regression table/iteration log per implicate/replicate. Has no
        effect on failures: a failing command is always retried
        non-quietly so the real explanation reaches the raised
        SystemError, since Stata's `quietly` prefix would otherwise
        suppress that error text too, collapsing it to a bare
        "r(####);".
    """
    global _stata_loaded_df

    def run_line(cmd: str, line_quietly: bool):
        try:
            stata.run(cmd, quietly=line_quietly)
        except SystemError:
            if line_quietly:
                stata.run(cmd, quietly=False)
            raise

    stata = require_pystata(edition, stata_path)

    if not (reuse_data and _stata_loaded_df is not None and _stata_loaded_df is df):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dta_path = os.path.join(tmp_dir, "survey_kit_implicate.dta")
            dataframe_to_dta(df, dta_path)

            #   forward slashes work fine as path separators inside Stata
            #   command strings, including on Windows.
            stata_path_str = dta_path.replace(os.sep, "/")
            run_line(f'use "{stata_path_str}", clear', True)
        _stata_loaded_df = df if reuse_data else None

    run_line("ereturn clear", True)
    run_line("return clear", True)

    for pre in pre_commands or []:
        run_line(pre, quietly)

    run_line(command, quietly)

    return stata


def clear_stata_cache() -> None:
    """
    Drop Stata's cached in-memory dataset from a previous reuse_data=True
    call (see _run_in_stata) and reset reuse tracking, freeing Stata's own
    RAM copy of the data. Call this once a replicate loop that used
    reuse_data=True has finished with one implicate's data, before moving
    on to the next (or at the end of the run) - otherwise that copy stays
    resident in Stata's memory indefinitely, since nothing else triggers
    Stata to drop it.
    """
    global _stata_loaded_df
    if _stata_loaded_df is not None:
        stata = require_pystata()
        stata.run("clear", quietly=True)
        _stata_loaded_df = None


def run_stata_model(
    df,
    command: str,
    pre_commands: list[str] | None = None,
    edition: str | None = None,
    stata_path: str | None = None,
    reuse_data: bool = False,
    quietly: bool = True,
):
    """
    Write df to a temp .dta, `use` it in a running Stata instance, run any
    `pre_commands` (e.g. `svyset`) then `command` (must be e-class - i.e.
    leave e(b)/e(V) populated, true of most estimation commands), and pull
    back the coefficient vector, covariance matrix, and Stata's own
    postestimation results table.

    Parameters
    ----------
    df : the data to estimate on (already merged with any design/weight
        columns needed).
    command : the Stata command to run, e.g. "regress y x1 x2" or
        "regress y x1 x2 [pw=w]" or "xtreg y x1 x2, fe" or, given a prior
        `svyset` (pass it via pre_commands), "svy: regress y x1 x2".
    pre_commands : commands to run after `use` but before `command`, e.g.
        `["svyset psu [pw=weight], strata(strata)"]`. Default is None.
    edition, stata_path : see require_pystata.
    reuse_data : see `_run_in_stata`'s docstring - skips re-exporting/
        re-`use`-ing df when it's the same object as a previous
        reuse_data=True call. Default is False. Call `clear_stata_cache()`
        when done with data reused this way.
    quietly : pass False to see Stata's own console output/error text for
        `pre_commands`/`command` - see `_run_in_stata`'s docstring. Default
        is True.

    Returns
    -------
    tuple
        (b, b_names, V, table, table_row_names, table_col_names):
        - b : list[float], the k coefficients (e(b), in order).
        - b_names : list[str], their names (e(b)'s column names).
        - V : list[list[float]], the k x k covariance matrix (e(V)).
        - table : list[list[float]], Stata's r(table) (rows are stats -
          b/se/t-or-z/pvalue/ll/ul/..., columns are terms).
        - table_row_names, table_col_names : r(table)'s row/column names.

    **UNTESTED** - see this module's docstring for the specific pieces most
    likely to need adjustment against a real Stata instance.
    """
    _run_in_stata(
        df, command, pre_commands, edition, stata_path, reuse_data=reuse_data, quietly=quietly
    )
    import sfi

    b = sfi.Matrix.get("e(b)")[0]
    b_names = sfi.Matrix.getColNames("e(b)")
    V = sfi.Matrix.get("e(V)")
    table = sfi.Matrix.get("r(table)")
    table_row_names = sfi.Matrix.getRowNames("r(table)")
    table_col_names = sfi.Matrix.getColNames("r(table)")

    return b, b_names, V, table, table_row_names, table_col_names


def run_stata_results(
    df,
    command: str,
    results: list[str],
    pre_commands: list[str] | None = None,
    edition: str | None = None,
    stata_path: str | None = None,
    reuse_data: bool = False,
    quietly: bool = True,
) -> dict[str, object]:
    """
    Write df to a temp .dta, `use` it, run any `pre_commands` then
    `command`, and pull back exactly the r()/e() results named in
    `results` - by scalar name (e.g. "r(mean)", "e(N)") or matrix name
    (e.g. "e(b)", "r(table)"), whichever `command` actually populates.

    Unlike run_stata_model (which assumes an e-class fit and always reads
    the fixed e(b)/e(V)/r(table) triplet), this works for ANY command that
    populates r()/e() results - including r-class-only commands
    (summarize, tabstat, ci, ...) that never populate e(b) at all. Meant
    as the Stata side of a
    [`StatCalculator.from_function`][survey_kit.statistics.calculator.StatCalculator.from_function]
    bootstrap/replicate delegate (see `stata_results_adapter` in
    adapters.py): run once per replicate weight, returning point estimates
    only - the variance comes from resampling across replicate weights
    (survey_kit's own Replicates/StatCalculator machinery), not from
    Stata's own e(V)/r(table) SEs.

    Parameters
    ----------
    results : names to pull back, e.g. ["r(mean)", "r(Var)", "r(N)"] or
        ["e(b)"]. Each is tried as a scalar first (sfi.Scalar.getValue),
        then as a matrix (sfi.Matrix.get) if that fails - covers the
        overwhelming majority of r()/e() results without the caller
        needing to say which kind it is. A name this call's `command`
        didn't actually populate raises rather than returning a stale
        value from an earlier call, since `_run_in_stata` clears e()/r()
        before running `command` - see its docstring.
    reuse_data : see `_run_in_stata`'s docstring - skips re-exporting/
        re-`use`-ing df across repeated calls with the same replicate data
        (the common case here, one call per replicate weight). Default is
        False. Call `clear_stata_cache()` when done with data reused this
        way.
    quietly : pass False to see Stata's own console output/error text for
        `pre_commands`/`command` - see `_run_in_stata`'s docstring. Default
        is True.

    Returns
    -------
    dict[str, object]
        name -> float (scalar) or (values, row_names, col_names) (matrix,
        as returned by sfi.Matrix.get/getRowNames/getColNames).

    **UNTESTED** - see this module's docstring. In particular, which
    exception `sfi.Scalar.getValue` raises for a name that isn't a scalar
    (used here to decide to fall back to `sfi.Matrix.get`) isn't confirmed
    against a real Stata instance.
    """
    _run_in_stata(
        df, command, pre_commands, edition, stata_path, reuse_data=reuse_data, quietly=quietly
    )
    import sfi

    out: dict[str, object] = {}
    for name in results:
        try:
            out[name] = sfi.Scalar.getValue(name)
        except Exception:
            out[name] = (
                sfi.Matrix.get(name),
                sfi.Matrix.getRowNames(name),
                sfi.Matrix.getColNames(name),
            )
    return out
