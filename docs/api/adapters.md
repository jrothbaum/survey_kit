# Regression Adapters

Every adapter below returns the same normalized shape - `(df_estimates, df_ses, df_vcov, df_tidy)` - so [`mi_ses_from_function`](multiple_imputation.md) and [`StatCalculator.from_function`](basic_standard_errors.md) treat any of them identically. Each also has a matching `mi_ses_from_<package>(...)` shortcut that runs it across multiple-imputation implicates directly, taking the adapter's own arguments as keywords instead of an `arguments={}` dict - see [Using the Regression Adapters](../user-guide/adapters.md) for a walkthrough.

## Pure Python (statsmodels, linearmodels, pyfixest, polars_ds)

No R/Stata/rpy2/pystata needed - these four wrap packages already available in Python.

::: survey_kit.statistics.adapters.statsmodels_adapter
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.mi_ses_from_statsmodels
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.linearmodels_adapter
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.mi_ses_from_linearmodels
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.pyfixest_adapter
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.mi_ses_from_pyfixest
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3
        show_source: false

::: survey_kit.statistics.adapters.polars_ds_adapter
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.mi_ses_from_polars_ds
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

## R (rpy2)

Requires R itself plus rpy2/rpy2-arrow (`pip install survey-kit[r]`). `r_feols`/`r_feglm`/`r_fepois`/`r_femlm` (and their `mi_ses_from_r_fixest` shortcuts) cover fixest's four estimators with named arguments; `r_fixest_adapter` is the fully generic `func=` escape hatch for anything else fixest offers (`feNmlm`, `feglm.fit`, ...); `r_lm_adapter` covers base R's `lm()`/`glm()`. For any other R package/function entirely, see [`_r_interop`](../user-guide/adapters.md#rolling-your-own) and `r_arbitrary_estimators.py`.

::: survey_kit.statistics.adapters.r_lm_adapter
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.r_fixest_adapter
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.r_feols
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.r_feglm
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.r_fepois
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.r_femlm
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.mi_ses_from_r_fixest
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3
        show_source: false

## Stata (pystata)

Requires Stata 17+ plus `pip install survey-kit[stata]`. `stata_adapter` (and its `mi_ses_from_stata` shortcut) runs any e-class command as a plain string - there's no separate named-wrapper-per-estimator layer to route around the way fixest has, since any Stata command already works by just changing the `command` string. `stata_results_adapter` reaches any r()/e() result rather than the fixed e(b)/e(V)/r(table) triplet, and is what `mi_ses_from_stata`'s `replicates=` option uses under the hood for replicate-weight bootstrapping.

::: survey_kit.statistics.adapters.stata_adapter
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.mi_ses_from_stata
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.statistics.adapters.stata_results_adapter
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3
