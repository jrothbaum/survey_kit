# Regression Adapters

## What Is It

`survey_kit.statistics.adapters` wraps regression packages - four pure-Python (statsmodels, linearmodels, pyfixest, polars_ds), R (via rpy2/fixest), and Stata (via pystata) - behind one normalized shape:

```python
(df_estimates, df_ses, df_vcov, df_tidy)
```

so every estimator plugs into the same downstream machinery ([`mi_ses_from_function`](../api/multiple_imputation.md), [`StatCalculator.from_function`](../api/basic_standard_errors.md)) with no special-casing for which package produced the numbers. Each adapter also has a matching `mi_ses_from_<package>(...)` shortcut that runs it across multiple-imputation implicates directly - the adapter's own arguments (formula, weight, vcov, ...) come through as plain keywords instead of being packed into an `arguments={}` dict, so your IDE shows the right parameters for the one you're actually calling.

## Why Use It

The adapters handle the fiddly parts of wiring a regression package into `mi_ses_from_function` - term-name alignment between `df_estimates`/`df_ses`/`df_vcov`, a missing `vcov()` method, converting the data into whatever the underlying package expects - so:

- **Any package plugs into MI/replicate-weight machinery identically** - swap `pyfixest_adapter` for `r_feols` for `stata_adapter` without changing anything downstream.
- **Simple call shape for the common case** - `mi_ses_from_pyfixest.feols(df_implicates=..., fml="y ~ x1 + x2 | firm", ...)` runs the fit across every implicate and combines the results.
- **Replicate-weight bootstrapping built in** - pass `replicates=` to run the same command once per replicate weight column and get the SE from the spread across replicates, instead of the package's own `vcov`/`cov_type` - useful when you need SEs computed the same way elsewhere in a project rather than trusting a given package's own variance estimator. The data conversion to whatever the underlying package needs (pandas, an R `data.frame`, a Stata `.dta`) happens once per implicate, not once per replicate.
- **An escape hatch when you need it** - every language also exposes its underlying primitives (`_r_interop`, `_stata_interop`) for writing a custom delegate when the named adapters don't cover what you need. See [Rolling Your Own](#rolling-your-own) below.

## Key Features

- **Same shape everywhere** - `(df_estimates, df_ses, df_vcov, df_tidy)` regardless of package.
- **`mi_ses_from_<package>` shortcuts** - one call combines across implicates via Rubin's rules.
- **Replicate-weight bootstrapping** - `replicates=` on every `mi_ses_from_*` that has a weight argument to substitute a column into.
- **No hard dependencies** - none of statsmodels/linearmodels/pyfixest/polars_ds/rpy2/pystata are required by survey_kit itself; each adapter raises a clear, actionable error (with the install command) only if you actually call it without the package installed.
- **Data conversion caching** - each package's own "already converted, don't redo the work" passthrough (an already-pandas frame, an already-`data.frame` R object, Stata's `reuse_data=`) means repeated calls against the same implicate - the replicate-weight loop being the main case - don't redo an expensive conversion on every call.

## When to Use What

| Use Case | Tool | Why |
|----------|------|-----|
| Already using statsmodels/linearmodels/pyfixest/polars_ds | matching `*_adapter`/`mi_ses_from_*` | No R/Stata install needed |
| Fixed effects, clustered SEs | `pyfixest_adapter`/`mi_ses_from_pyfixest` | Generally the best default of the four Python adapters |
| Already have R code/packages you trust | `r_feols`/`r_fixest_adapter`/`r_lm_adapter` | Reuses your existing R model specifications |
| Already have Stata code/do-files | `stata_adapter`/`mi_ses_from_stata` | `command=` takes the Stata syntax directly |
| Need SEs computed the same way as elsewhere in a project | any `mi_ses_from_*` with `replicates=` | Bootstraps from replicate weights instead of the package's own vcov |
| A package/function with no named adapter | `_r_interop`/`_stata_interop` primitives directly | See [Rolling Your Own](#rolling-your-own) |

## API

See the [Regression Adapters API reference](../api/adapters.md) for the full parameter list of every adapter and `mi_ses_from_*` shortcut.

## Example/Tutorial

=== "Python (statsmodels/linearmodels/pyfixest/polars_ds)"
    No R/Stata install needed - these four wrap packages already available in Python.

    === "Code"
        ```python
        --8<-- "tutorials/statistics/basic_python_adapters.py"
        ```

    === "Log"
        [View in separate window](../tutorials/statistics/basic_python_adapters.html){:target="_blank"}
        <iframe src="../../tutorials/statistics/basic_python_adapters.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "R (fixest)"
    Requires R itself plus rpy2/rpy2-arrow (`pip install survey-kit[r]`) and the R `fixest` package. Check your setup cheaply with `from survey_kit.statistics._r_interop import check_r_setup; check_r_setup(['fixest'])`.

    === "Code"
        ```python
        --8<-- "tutorials/statistics/basic_r.py"
        ```

    === "Log"
        [View in separate window](../tutorials/statistics/basic_r.html){:target="_blank"}
        <iframe src="../../tutorials/statistics/basic_r.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "Stata"
    Requires Stata 17+ plus `pip install survey-kit[stata]`. Check your setup cheaply with `from survey_kit.statistics._stata_interop import check_stata_setup; check_stata_setup(stata_path=r"C:\Program Files\Stata18")`.

    === "Code"
        ```python
        --8<-- "tutorials/statistics/basic_stata.py"
        ```

    === "Log"
        <iframe src="../../tutorials/statistics/basic_stata.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

## Rolling Your Own

Each language's tutorial above only covers the named adapters (`statsmodels_adapter`, `r_feols`, `stata_adapter`, ...). For a package or function with no named wrapper, every language exposes the primitives those adapters are themselves built from, so writing a custom delegate is a few lines rather than a new adapter:

=== "R"
    `_r_interop`'s `get_library`/`dataframe_to_r`/`formula`/`extract_fit`/`coef_table` etc. - any R function becomes a Python-callable attribute via rpy2's dot-calling, with no R call string to build. Covers reaching any R package/function at all, plus how to convert an implicate to R once and reuse it across several model calls (or once per replicate weight) instead of re-converting on every call.

    === "Code"
        ```python
        --8<-- "tutorials/statistics/r_arbitrary_estimators.py"
        ```

    === "Log"
        [View in separate window](../tutorials/statistics/r_arbitrary_estimators.html){:target="_blank"}
        <iframe src="../../tutorials/statistics/r_arbitrary_estimators.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "Stata"
    `stata_adapter`'s generic `command=` string already covers any e-class Stata command by itself (svy:/xtreg/areg/logit/a community-installed ado/...) - there's no separate named-wrapper-per-estimator layer to route around the way fixest has on the R side. `_stata_interop`'s lower-level primitives (`run_stata_model`, `run_stata_results`) are what `stata_adapter`/`stata_results_adapter` are themselves built from, for reaching something below the command-string level (a custom pre/post-processing step, or pulling back an arbitrary r()/e() result).

    === "Code"
        ```python
        --8<-- "tutorials/statistics/stata_arbitrary_estimators.py"
        ```

    === "Log"
        [view in separate window](../tutorials/statistics/stata_arbitrary_estimators.html){:target="_blank"} if available.
        <iframe src="../../tutorials/statistics/stata_arbitrary_estimators.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>
