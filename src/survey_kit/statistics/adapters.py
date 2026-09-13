"""
Adapters that let common Python regression/estimation packages be used as
the `delegate` in
[`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function],
so their per-implicate coefficient estimates and standard errors get
combined into proper multiple-imputation standard errors via Rubin's rules -
with no dependency on StatCalculator/ReplicateStats.

None of the underlying packages are hard dependencies of survey_kit - each
adapter imports its package lazily and raises a clear ImportError naming the
`uv add` command if it isn't installed.

Every adapter returns the same shape, normalized regardless of what the
underlying package natively calls things:

    (df_estimates, df_ses, df_vcov, df_tidy)

- df_estimates / df_ses: one row per term, with the term-identifier column
  named `join_on_name` (default "Variable") and the coefficient/SE column
  named `value_name` (default "estimate") - matching survey_kit's own
  join_on convention used everywhere else in this library.
- df_vcov: the term-by-term covariance matrix in the long/pairwise form
  [`MultipleImputation`][survey_kit.statistics.multiple_imputation.MultipleImputation]
  expects (join_on columns suffixed "_1"/"_2" plus one value column), or
  None when the package doesn't compute one (e.g. polars_ds). This is what
  lets `.compare()` compute a correct joint standard error for a contrast
  between two terms of the same fit instead of assuming independence.
- df_tidy: the underlying package's own native coefficient/summary table
  for this one fit (whatever columns it naturally has - estimate, SE, t/z,
  p-value, CI, R², ...), held as-is rather than normalized. This is a
  per-implicate diagnostic snapshot only - MultipleImputation does NOT
  combine it across implicates (a single implicate's own t-stats/p-values
  aren't valid MI inference), it just stays accessible via
  `mi_result.implicate_stats[i].df_tidy`.
"""

from __future__ import annotations

import polars as pl

from .. import logger


def _coef_table_from_series(
    coef, se, join_on_name: str, value_name: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df_estimates = pl.DataFrame(
        {join_on_name: list(coef.index), value_name: list(coef.to_numpy())}
    )
    df_ses = pl.DataFrame(
        {join_on_name: list(se.index), value_name: list(se.to_numpy())}
    )
    return df_estimates, df_ses


def _vcov_table_from_frame(
    cov_df, join_on_name: str, value_name: str
) -> pl.DataFrame:
    terms = list(cov_df.index)
    records = [
        {
            f"{join_on_name}_1": t1,
            f"{join_on_name}_2": t2,
            value_name: float(cov_df.loc[t1, t2]),
        }
        for t1 in terms
        for t2 in terms
    ]
    return pl.DataFrame(records)


def _tidy_from_parts(
    join_on_name: str,
    estimate,
    se,
    statistic,
    pvalue,
    conf_int,
) -> pl.DataFrame:
    """
    Assemble a tidy coefficient table from a package's separate params/se/
    tstat-or-zstat/pvalue Series and a conf_int DataFrame (columns "lower"/
    "upper" - linearmodels' own naming, confirmed consistent across its
    model classes), for packages that don't already expose one ready-made
    table the way statsmodels' summary2()/pyfixest's tidy()/R's coeftable()
    do.
    """
    return pl.DataFrame(
        {
            join_on_name: list(estimate.index),
            "estimate": list(estimate.to_numpy()),
            "std_error": list(se.to_numpy()),
            "statistic": list(statistic.to_numpy()),
            "p_value": list(pvalue.to_numpy()),
            "conf_low": list(conf_int["lower"].to_numpy()),
            "conf_high": list(conf_int["upper"].to_numpy()),
        }
    )


def _to_pandas(df):
    import pandas as pd

    if isinstance(df, pd.DataFrame):
        #   Already converted - returned unchanged rather than re-converted,
        #   same as _r_interop.dataframe_to_r does for an already-R object.
        #   Lets a caller that will reuse the same data across several
        #   calls (e.g. once per replicate weight) convert once and pass
        #   the pandas frame directly on later calls instead of the
        #   original polars/narwhals df.
        return df

    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    #   narwhals-native with no to_pandas (e.g. a raw pyarrow/duckdb frame).
    import narwhals as nw

    return nw.from_native(df).lazy().collect().to_pandas()


def _to_polars(df) -> pl.DataFrame:
    """
    Materializes df to an actual pl.DataFrame, collecting a LazyFrame if
    given one, so a caller reusing the same implicate across many calls
    (e.g. once per replicate weight) collects once and passes the
    materialized frame on later calls - passing an uncollected LazyFrame
    through unchanged instead would re-run its whole upstream plan (joins/
    filters/etc.) from scratch on every one of those calls.
    """
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()

    import narwhals as nw

    return nw.from_native(df).lazy().collect().to_polars()


def _mi_ses_replicates_delegate(
    adapter,
    base_arguments: dict,
    join_on_name: str,
    replicates,
    convert=None,
):
    """
    Build an `mi_ses_from_function` delegate (called once per implicate)
    that instead runs `adapter` once per replicate weight column via
    `StatCalculator.from_function`, taking just its point estimates
    (element 0 of the (df_estimates, df_ses, df_vcov, df_tidy) tuple every
    adapter in this module returns) - the SE then comes from the spread of
    estimates across replicates rather than the adapter's own df_ses/
    df_vcov, the same replicate-weight-bootstrap approach
    `stata_results_adapter` uses for Stata.

    `convert`, if given (e.g. `_to_pandas`, `_to_polars`,
    `_r_interop.dataframe_to_r`), converts df once per implicate - on the
    first replicate call, memoized in a closure for the rest - rather than
    leaving `adapter` to redo that conversion on every one of potentially
    dozens of replicate-weight calls for the same underlying data. This
    can't just pre-convert `df` before calling `StatCalculator.from_function`
    below: its own `df=` parameter has to stay narwhals-compatible (pandas/
    polars/pyarrow/...) since it inspects it directly - an R data.frame
    (an rpy2 object, not narwhals-native) would break that immediately, so
    the conversion has to happen inside `_point_estimate` instead, after
    `StatCalculator.from_function` has already done its own (cheap, since
    `df` here is never actually partitioned by `by=`) handling of the
    original df. None (the default) skips this - appropriate for an
    adapter that's already cheap to call repeatedly on the same data (e.g.
    `polars_ds_adapter` with an already-materialized polars df).
    """
    from .calculator import StatCalculator

    def delegate(df):
        converted = {}

        def _point_estimate(df, weight):
            if convert is not None:
                if not converted:
                    converted["df"] = convert(df)
                df = converted["df"]
            return adapter(df, weight=weight, **base_arguments)[0]

        return StatCalculator.from_function(
            delegate=_point_estimate,
            estimate_ids=[join_on_name],
            df=df,
            replicates=replicates,
            display=False,
        )

    return delegate


def statsmodels_adapter(
    df,
    y: str,
    x: list[str] | str,
    weight: str | None = None,
    add_constant: bool = True,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    cov_type: str = "HC3",
    cov_kwds: dict | None = None,
    model_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Fit an OLS/WLS regression with statsmodels and return its coefficient
    table in survey_kit's normalized (df_estimates, df_ses, df_vcov,
    df_tidy) shape.

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    y : dependent variable column name.
    x : predictor column name(s).
    weight : column name for weighted least squares (WLS), or None for
        unweighted OLS. Default is None.
    add_constant : whether to add an intercept term (named "const"). Default
        is True.
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE/covariance value column in the
        output. Default is "estimate".
    cov_type : passed to `.fit()`. Defaults to "HC3" (heteroskedasticity-
        robust) rather than statsmodels' own classical default, since HC3
        performs well even in small-to-moderate samples and there's rarely a
        reason to assume homoskedasticity for survey/implicate data. Pass
        "nonrobust" to opt back into classical SEs.
    cov_kwds : extra keywords for the covariance estimator (e.g.
        `{"groups": df["cluster"]}` for cluster-robust SEs). Default is None.
    model_kwargs : extra keywords forwarded to the model constructor
        (`sm.OLS`/`sm.WLS`). Default is None.
    fit_kwargs : extra keywords forwarded to `.fit()` besides cov_type/
        cov_kwds (e.g. `{"maxiter": 200}`). Default is None.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (df_estimates, df_ses, df_vcov, df_tidy) - df_vcov is always
        populated here since statsmodels computes it for free alongside
        .bse. df_tidy is statsmodels' own coefficient table
        (`results.summary2().tables[1]`: Coef./Std.Err./z or t/P>|z|/CI),
        held as-is - a single implicate's own t-stats/p-values aren't valid
        MI inference, so this is a diagnostic snapshot only, never combined
        across implicates.
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:
        message = (
            "statsmodels_adapter requires the 'statsmodels' package - "
            "install it with `uv add statsmodels`."
        )
        logger.error(message)
        raise ImportError(message) from e

    df_pd = _to_pandas(df)
    x = [x] if isinstance(x, str) else list(x)
    model_kwargs = dict(model_kwargs or {})
    fit_kwargs = dict(fit_kwargs or {})

    exog = df_pd[x]
    if add_constant:
        exog = sm.add_constant(exog, has_constant="add")

    if weight:
        model = sm.WLS(df_pd[y], exog, weights=df_pd[weight], **model_kwargs)
    else:
        model = sm.OLS(df_pd[y], exog, **model_kwargs)

    results = model.fit(cov_type=cov_type, cov_kwds=cov_kwds, **fit_kwargs)

    df_estimates, df_ses = _coef_table_from_series(
        results.params, results.bse, join_on_name, value_name
    )
    df_vcov = _vcov_table_from_frame(results.cov_params(), join_on_name, value_name)
    df_tidy = pl.from_pandas(
        results.summary2().tables[1].reset_index(names=join_on_name)
    )

    return (df_estimates, df_ses, df_vcov, df_tidy)


def mi_ses_from_statsmodels(
    df_implicates,
    y: str,
    x: list[str] | str,
    weight: str | None = None,
    add_constant: bool = True,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    cov_type: str = "HC3",
    cov_kwds: dict | None = None,
    model_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
    replicates=None,
    path_srmi: str = "",
    index: list | None = None,
    df_noimputes=None,
    parallel: bool = False,
    parallel_inputs=None,
    rounding=None,
    round_output: bool = True,
):
    """
    `mi_ses_from_function(delegate=statsmodels_adapter, ...)`, with
    `statsmodels_adapter`'s own arguments taken directly as keyword
    arguments instead of packed into an `arguments={}` dict.

    Parameters
    ----------
    df_implicates, path_srmi, index, df_noimputes, parallel,
        parallel_inputs, rounding, round_output : see
        [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
    y, x, weight, add_constant, join_on_name, value_name, cov_type,
        cov_kwds, model_kwargs, fit_kwargs : see `statsmodels_adapter`.
    replicates : `survey_kit.statistics.replicates.Replicates`, optional.
        When given, per-implicate SEs come from resampling across
        replicate weights instead of statsmodels' own cov_type/cov_kwds -
        `statsmodels_adapter` is run once per replicate weight column
        (point estimates only), the same replicate-weight-bootstrap
        approach `mi_ses_from_stata`'s `replicates=` uses. `weight` is
        ignored in this mode (each replicate call supplies its own weight
        column) - the implicate is converted to pandas once, not once per
        replicate. Default is None (use `statsmodels_adapter`'s own
        cov_type/cov_kwds).

    Returns
    -------
    MultipleImputation
        Same as `mi_ses_from_function`.

    See Also
    --------
    statsmodels_adapter : the delegate this wraps.
    """
    from .multiple_imputation import mi_ses_from_function

    base_arguments = {
        "y": y,
        "x": x,
        "add_constant": add_constant,
        "join_on_name": join_on_name,
        "value_name": value_name,
        "model_kwargs": model_kwargs,
        "fit_kwargs": fit_kwargs,
    }

    if replicates is None:
        delegate = statsmodels_adapter
        arguments = {**base_arguments, "weight": weight, "cov_type": cov_type, "cov_kwds": cov_kwds}
    else:
        if weight:
            logger.warning(
                "mi_ses_from_statsmodels: weight is ignored when replicates "
                "is given - each replicate call supplies its own weight column."
            )
        #   cov_type/cov_kwds are discarded for these point-estimate-only
        #   replicate calls (the SE comes from the spread across
        #   replicates, not statsmodels' own covariance) - force the
        #   cheapest (classical) option rather than whatever the caller
        #   passed, to skip computing a robust covariance matrix that's
        #   about to be thrown away, on every one of potentially dozens of
        #   replicate calls.
        delegate = _mi_ses_replicates_delegate(
            statsmodels_adapter,
            {**base_arguments, "cov_type": "nonrobust", "cov_kwds": None},
            join_on_name,
            replicates,
            convert=_to_pandas,
        )
        arguments = {}

    return mi_ses_from_function(
        delegate=delegate,
        df_implicates=df_implicates,
        join_on=[join_on_name],
        path_srmi=path_srmi,
        index=index,
        df_noimputes=df_noimputes,
        arguments=arguments,
        parallel=parallel,
        parallel_inputs=parallel_inputs,
        rounding=rounding,
        round_output=round_output,
    )


def linearmodels_adapter(
    df,
    formula: str,
    weight: str | None = None,
    model: str = "IV2SLS",
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    cov_type: str = "robust",
    cov_kwds: dict | None = None,
    model_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Fit a linearmodels model (IV/panel) and return its coefficient table in
    survey_kit's normalized (df_estimates, df_ses, df_vcov, df_tidy) shape.

    Unlike statsmodels_adapter/polars_ds_adapter, this takes a formula
    string rather than y/x lists - linearmodels' bracket syntax
    (`"y ~ 1 + x1 + [x2 ~ z1 + z2]"` for instrumenting x2 with z1/z2) is how
    IV/panel specifications are expressed, and there's no y/x-list
    equivalent for that.

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    formula : linearmodels formula string, e.g. "y ~ 1 + x1 + x2" for plain
        OLS via IV2SLS, or with a bracketed `[endog ~ instruments]` clause
        for actual instrumental variables.
    weight : column name for weighted estimation, or None. Default is None.
    model : one of "IV2SLS", "PanelOLS", "PooledOLS", "RandomEffects",
        "BetweenOLS". Default is "IV2SLS" (also covers plain OLS, with no
        instruments in the formula). Panel models require df to have the
        entity/time MultiIndex they expect - set that up before calling
        mi_ses_from_function (e.g. via a `pre`/index-setting step), this
        adapter doesn't build one for you.
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE/covariance value column in the
        output. Default is "estimate".
    cov_type : passed to `.fit()`. Defaults to "robust" - linearmodels' own
        vocabulary for heteroskedasticity-consistent SEs ("HC0-3" is a
        statsmodels-specific term with no direct equivalent here); "robust"
        is the closest counterpart to statsmodels_adapter's HC3 default,
        for the same reason (rarely safe to assume homoskedasticity).
    cov_kwds : extra keywords for the covariance estimator (e.g.
        `{"clusters": df["cluster"]}` for cluster-robust SEs), spread as
        `**cov_kwds` into `.fit()` since linearmodels takes them as loose
        kwargs rather than a nested dict. Default is None.
    model_kwargs : extra keywords forwarded to the model constructor.
        Default is None.
    fit_kwargs : extra keywords forwarded to `.fit()` besides cov_type/
        cov_kwds. Default is None.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (df_estimates, df_ses, df_vcov, df_tidy) - df_vcov is always
        populated here since linearmodels computes it for free alongside
        .std_errors. df_tidy assembles estimate/std_error/statistic/
        p_value/conf_low/conf_high from linearmodels' own .params/
        .std_errors/.tstats/.pvalues/.conf_int() (linearmodels has no
        single ready-made tidy table the way statsmodels/pyfixest do) - a
        diagnostic snapshot only, never combined across implicates.
    """
    try:
        from linearmodels.iv import IV2SLS
        from linearmodels.panel import BetweenOLS, PanelOLS, PooledOLS, RandomEffects
    except ImportError as e:
        message = (
            "linearmodels_adapter requires the 'linearmodels' package - "
            "install it with `uv add linearmodels`."
        )
        logger.error(message)
        raise ImportError(message) from e

    model_classes = {
        "IV2SLS": IV2SLS,
        "PanelOLS": PanelOLS,
        "PooledOLS": PooledOLS,
        "RandomEffects": RandomEffects,
        "BetweenOLS": BetweenOLS,
    }
    if model not in model_classes:
        message = f"Unknown linearmodels model '{model}'; expected one of {list(model_classes)}."
        logger.error(message)
        raise ValueError(message)

    df_pd = _to_pandas(df)
    model_kwargs = dict(model_kwargs or {})
    fit_kwargs = dict(fit_kwargs or {})
    cov_kwds = dict(cov_kwds or {})

    if weight:
        model_kwargs["weights"] = df_pd[weight]

    model_obj = model_classes[model].from_formula(formula, data=df_pd, **model_kwargs)
    results = model_obj.fit(cov_type=cov_type, **cov_kwds, **fit_kwargs)

    df_estimates, df_ses = _coef_table_from_series(
        results.params, results.std_errors, join_on_name, value_name
    )
    df_vcov = _vcov_table_from_frame(results.cov, join_on_name, value_name)
    df_tidy = _tidy_from_parts(
        join_on_name,
        results.params,
        results.std_errors,
        results.tstats,
        results.pvalues,
        results.conf_int(),
    )

    return (df_estimates, df_ses, df_vcov, df_tidy)


def mi_ses_from_linearmodels(
    df_implicates,
    formula: str,
    weight: str | None = None,
    model: str = "IV2SLS",
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    cov_type: str = "robust",
    cov_kwds: dict | None = None,
    model_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
    replicates=None,
    path_srmi: str = "",
    index: list | None = None,
    df_noimputes=None,
    parallel: bool = False,
    parallel_inputs=None,
    rounding=None,
    round_output: bool = True,
):
    """
    `mi_ses_from_function(delegate=linearmodels_adapter, ...)`, with
    `linearmodels_adapter`'s own arguments taken directly as keyword
    arguments instead of packed into an `arguments={}` dict.

    Parameters
    ----------
    df_implicates, path_srmi, index, df_noimputes, parallel,
        parallel_inputs, rounding, round_output : see
        [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
    formula, weight, model, join_on_name, value_name, cov_type, cov_kwds,
        model_kwargs, fit_kwargs : see `linearmodels_adapter`.
    replicates : `survey_kit.statistics.replicates.Replicates`, optional.
        When given, per-implicate SEs come from resampling across
        replicate weights instead of linearmodels' own cov_type/cov_kwds -
        `linearmodels_adapter` is run once per replicate weight column
        (point estimates only, with cov_type forced to "unadjusted" since
        the covariance is discarded anyway), the same replicate-weight-
        bootstrap approach `mi_ses_from_stata`'s `replicates=` uses.
        `weight` is ignored in this mode. Default is None (use
        `linearmodels_adapter`'s own cov_type/cov_kwds).

    Returns
    -------
    MultipleImputation
        Same as `mi_ses_from_function`.

    See Also
    --------
    linearmodels_adapter : the delegate this wraps.
    """
    from .multiple_imputation import mi_ses_from_function

    base_arguments = {
        "formula": formula,
        "model": model,
        "join_on_name": join_on_name,
        "value_name": value_name,
        "model_kwargs": model_kwargs,
        "fit_kwargs": fit_kwargs,
    }

    if replicates is None:
        delegate = linearmodels_adapter
        arguments = {**base_arguments, "weight": weight, "cov_type": cov_type, "cov_kwds": cov_kwds}
    else:
        if weight:
            logger.warning(
                "mi_ses_from_linearmodels: weight is ignored when replicates "
                "is given - each replicate call supplies its own weight column."
            )
        delegate = _mi_ses_replicates_delegate(
            linearmodels_adapter,
            {**base_arguments, "cov_type": "unadjusted", "cov_kwds": None},
            join_on_name,
            replicates,
            convert=_to_pandas,
        )
        arguments = {}

    return mi_ses_from_function(
        delegate=delegate,
        df_implicates=df_implicates,
        join_on=[join_on_name],
        path_srmi=path_srmi,
        index=index,
        df_noimputes=df_noimputes,
        arguments=arguments,
        parallel=parallel,
        parallel_inputs=parallel_inputs,
        rounding=rounding,
        round_output=round_output,
    )


def pyfixest_adapter(
    df,
    formula: str,
    func: str = "feols",
    family: str | None = None,
    weight: str | None = None,
    vcov: str | dict | None = "hetero",
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    **kwargs,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, pl.DataFrame]:
    """
    Fit a pyfixest regression and return its coefficient table in
    survey_kit's normalized (df_estimates, df_ses, df_vcov, df_tidy) shape.
    pyfixest mirrors R's fixest syntax/functionality (fixed effects via formula,
    e.g. "y ~ x1 | firm", robust/clustered SEs, OLS/GLM/Poisson) natively in
    Python - no R/rpy2 needed, and generally the better default over
    `r_feols`/`r_feglm`/`r_fepois`/`r_fixest_adapter` unless you specifically
    need something pyfixest doesn't cover (those pull in a full R + rpy2 +
    rpy2-arrow dependency chain for no extra benefit if pyfixest already
    does the job).

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    formula : fixest-syntax formula string, e.g. "y ~ x1 + x2 | firm" for a
        fit with a firm fixed effect.
    func : which pyfixest estimator to call - "feols" (default), "feglm",
        or "fepois". Called as `getattr(pyfixest, func)(...)`.
    family : passed to feglm as its required `family=` argument (e.g.
        "logit", "probit", "poisson"). Not used for feols/fepois - leave as
        None.
    weight : column name for weighted estimation, or None. Passed straight
        through as pyfixest's own `weights=` (a plain column name - no
        formula/raw-string gymnastics needed, unlike the R adapters).
    vcov : pyfixest's own `vcov=` argument - a string like "hetero" (robust,
        the default here for the same reason statsmodels_adapter defaults
        to HC3: rarely safe to assume homoskedasticity), "iid" (classical),
        or a dict for clustering, e.g. `{"CRV1": "firm"}`.
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE/covariance value column in the
        output. Default is "estimate".
    **kwargs : any other argument the chosen estimator takes (`ssc`,
        `fixef_rm`, `split`/`fsplit`, `offset` for fepois, ...) - forwarded
        verbatim, no conversion needed since this stays in pure Python.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, pl.DataFrame]
        (df_estimates, df_ses, df_vcov, df_tidy). df_vcov is populated from
        the fitted model's internal covariance matrix when its shape
        matches the coefficient vector, else None (with a warning) rather
        than risking a silent term misalignment - pyfixest doesn't expose
        this as public API, so this reads a private attribute defensively.
        df_tidy is pyfixest's own `fit.tidy()` table (Estimate/Std. Error/
        t value/Pr(>|t|)/CI) - a diagnostic snapshot only, never combined
        across implicates.
    """
    try:
        import pyfixest as pf
    except ImportError as e:
        message = (
            "pyfixest_adapter requires the 'pyfixest' package - install it "
            "with `uv add pyfixest` (or `pip install survey-kit[pyfixest]`)."
        )
        logger.error(message)
        raise ImportError(message) from e

    df_pd = _to_pandas(df)

    call_kwargs = dict(kwargs)
    if weight is not None:
        call_kwargs["weights"] = weight
    if family is not None:
        call_kwargs["family"] = family
    if vcov is not None:
        call_kwargs["vcov"] = vcov

    fit = getattr(pf, func)(formula, data=df_pd, **call_kwargs)

    coef = fit.coef()
    se = fit.se()

    df_estimates = pl.DataFrame(
        {join_on_name: list(coef.index), value_name: coef.to_numpy()}
    )
    df_ses = pl.DataFrame({join_on_name: list(se.index), value_name: se.to_numpy()})

    df_vcov = None
    vcov_matrix = getattr(fit, "_vcov", None)
    if vcov_matrix is not None:
        import numpy as np

        terms = list(coef.index)
        n = len(terms)
        vcov_matrix = np.asarray(vcov_matrix)
        if vcov_matrix.shape == (n, n):
            df_vcov = pl.DataFrame(
                [
                    {
                        f"{join_on_name}_1": terms[i],
                        f"{join_on_name}_2": terms[j],
                        value_name: float(vcov_matrix[i, j]),
                    }
                    for i in range(n)
                    for j in range(n)
                ]
            )
        else:
            logger.warning(
                "pyfixest_adapter: the fitted model's internal covariance "
                f"matrix shape {vcov_matrix.shape} doesn't match the "
                f"{n} coefficient terms - returning df_vcov=None rather "
                "than risk misaligning terms."
            )

    df_tidy = pl.from_pandas(fit.tidy().reset_index(names=join_on_name))

    return (df_estimates, df_ses, df_vcov, df_tidy)


class mi_ses_from_pyfixest:
    """
    Namespace for per-estimator `mi_ses_from_function(delegate=
    pyfixest_adapter, ...)` wrappers - `mi_ses_from_pyfixest.feols(...)`,
    `.fepois(...)`, `.feglm(...)` - each naming the common `pyfixest_adapter`
    arguments (`fml`, `weight`, `vcov`, `family`) explicitly instead of
    picking the estimator via a `func="..."` string, so IDEs show the right
    parameters for the one you're actually calling. Everything else
    `pyfixest.feols`/`fepois`/`feglm` accepts still flows through **kwargs
    exactly as in `pyfixest_adapter`.
    """

    @staticmethod
    def feols(
        df_implicates,
        fml: str,
        weight: str | None = None,
        vcov: str | dict | None = "hetero",
        join_on_name: str = "Variable",
        value_name: str = "estimate",
        replicates=None,
        path_srmi: str = "",
        index: list | None = None,
        df_noimputes=None,
        parallel: bool = False,
        parallel_inputs=None,
        rounding=None,
        round_output: bool = True,
        **kwargs,
    ):
        """
        `mi_ses_from_function(delegate=pyfixest_adapter, ...)` for
        `pyfixest.feols` specifically - OLS/IV with fixed effects, fixest
        formula syntax (e.g. "y ~ x1 + x2 | firm").

        Parameters
        ----------
        df_implicates, path_srmi, index, df_noimputes, parallel,
            parallel_inputs, rounding, round_output : see
            [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
        fml, weight, vcov, join_on_name, value_name : see
            `pyfixest_adapter` (`fml` is `pyfixest_adapter`'s `formula`).
        replicates : `survey_kit.statistics.replicates.Replicates`,
            optional. When given, per-implicate SEs come from resampling
            across replicate weights instead of `vcov` - `pyfixest_adapter`
            is run once per replicate weight column (point estimates only,
            with `vcov` forced to "iid" since it's discarded anyway), the
            same replicate-weight-bootstrap approach
            `mi_ses_from_stata`'s `replicates=` uses. `weight` is ignored
            in this mode. Default is None (use `pyfixest_adapter`'s own
            `vcov`).
        **kwargs : any other `pyfixest.feols` argument (`ssc`, `fixef_rm`,
            `split`/`fsplit`, ...) - forwarded verbatim, see
            `pyfixest_adapter`.

        Returns
        -------
        MultipleImputation
            Same as `mi_ses_from_function`.

        See Also
        --------
        pyfixest_adapter : the delegate this wraps.
        """
        from .multiple_imputation import mi_ses_from_function

        base_arguments = {
            "func": "feols",
            "formula": fml,
            "join_on_name": join_on_name,
            "value_name": value_name,
            **kwargs,
        }

        if replicates is None:
            delegate = pyfixest_adapter
            arguments = {**base_arguments, "weight": weight, "vcov": vcov}
        else:
            if weight:
                logger.warning(
                    "mi_ses_from_pyfixest.feols: weight is ignored when "
                    "replicates is given - each replicate call supplies "
                    "its own weight column."
                )
            delegate = _mi_ses_replicates_delegate(
                pyfixest_adapter,
                {**base_arguments, "vcov": "iid"},
                join_on_name,
                replicates,
                convert=_to_pandas,
            )
            arguments = {}

        return mi_ses_from_function(
            delegate=delegate,
            df_implicates=df_implicates,
            join_on=[join_on_name],
            path_srmi=path_srmi,
            index=index,
            df_noimputes=df_noimputes,
            arguments=arguments,
            parallel=parallel,
            parallel_inputs=parallel_inputs,
            rounding=rounding,
            round_output=round_output,
        )

    @staticmethod
    def fepois(
        df_implicates,
        fml: str,
        weight: str | None = None,
        vcov: str | dict | None = "hetero",
        join_on_name: str = "Variable",
        value_name: str = "estimate",
        replicates=None,
        path_srmi: str = "",
        index: list | None = None,
        df_noimputes=None,
        parallel: bool = False,
        parallel_inputs=None,
        rounding=None,
        round_output: bool = True,
        **kwargs,
    ):
        """
        `mi_ses_from_function(delegate=pyfixest_adapter, ...)` for
        `pyfixest.fepois` specifically - Poisson regression with fixed
        effects.

        Parameters
        ----------
        df_implicates, path_srmi, index, df_noimputes, parallel,
            parallel_inputs, rounding, round_output : see
            [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
        fml, weight, vcov, join_on_name, value_name : see
            `pyfixest_adapter` (`fml` is `pyfixest_adapter`'s `formula`).
        replicates : see `mi_ses_from_pyfixest.feols` - same
            replicate-weight-bootstrap option, `vcov` forced to "iid" and
            `weight` ignored in this mode. Default is None.
        **kwargs : any other `pyfixest.fepois` argument (`ssc`,
            `fixef_rm`, `offset`, `iwls_tol`/`iwls_maxiter`, ...) -
            forwarded verbatim, see `pyfixest_adapter`.

        Returns
        -------
        MultipleImputation
            Same as `mi_ses_from_function`.

        See Also
        --------
        pyfixest_adapter : the delegate this wraps.
        """
        from .multiple_imputation import mi_ses_from_function

        base_arguments = {
            "func": "fepois",
            "formula": fml,
            "join_on_name": join_on_name,
            "value_name": value_name,
            **kwargs,
        }

        if replicates is None:
            delegate = pyfixest_adapter
            arguments = {**base_arguments, "weight": weight, "vcov": vcov}
        else:
            if weight:
                logger.warning(
                    "mi_ses_from_pyfixest.fepois: weight is ignored when "
                    "replicates is given - each replicate call supplies "
                    "its own weight column."
                )
            delegate = _mi_ses_replicates_delegate(
                pyfixest_adapter,
                {**base_arguments, "vcov": "iid"},
                join_on_name,
                replicates,
                convert=_to_pandas,
            )
            arguments = {}

        return mi_ses_from_function(
            delegate=delegate,
            df_implicates=df_implicates,
            join_on=[join_on_name],
            path_srmi=path_srmi,
            index=index,
            df_noimputes=df_noimputes,
            arguments=arguments,
            parallel=parallel,
            parallel_inputs=parallel_inputs,
            rounding=rounding,
            round_output=round_output,
        )

    @staticmethod
    def feglm(
        df_implicates,
        fml: str,
        family: str,
        vcov: str | dict | None = "hetero",
        join_on_name: str = "Variable",
        value_name: str = "estimate",
        path_srmi: str = "",
        index: list | None = None,
        df_noimputes=None,
        parallel: bool = False,
        parallel_inputs=None,
        rounding=None,
        round_output: bool = True,
        **kwargs,
    ):
        """
        `mi_ses_from_function(delegate=pyfixest_adapter, ...)` for
        `pyfixest.feglm` specifically - GLM with fixed effects (`family=`
        required, e.g. "logit", "probit", "poisson"). No `replicates=`
        option here (unlike `.feols`/`.fepois`) - pyfixest's `feglm()` has
        no `weights=` argument to substitute a replicate weight column
        into.

        Parameters
        ----------
        df_implicates, path_srmi, index, df_noimputes, parallel,
            parallel_inputs, rounding, round_output : see
            [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
        fml, family, vcov, join_on_name, value_name : see
            `pyfixest_adapter` (`fml` is `pyfixest_adapter`'s `formula`).
        **kwargs : any other `pyfixest.feglm` argument (`ssc`,
            `fixef_rm`, `iwls_tol`/`iwls_maxiter`, ...) - forwarded
            verbatim, see `pyfixest_adapter`.

        Returns
        -------
        MultipleImputation
            Same as `mi_ses_from_function`.

        See Also
        --------
        pyfixest_adapter : the delegate this wraps.
        """
        from .multiple_imputation import mi_ses_from_function

        return mi_ses_from_function(
            delegate=pyfixest_adapter,
            df_implicates=df_implicates,
            join_on=[join_on_name],
            path_srmi=path_srmi,
            index=index,
            df_noimputes=df_noimputes,
            arguments={
                "func": "feglm",
                "formula": fml,
                "family": family,
                "vcov": vcov,
                "join_on_name": join_on_name,
                "value_name": value_name,
                **kwargs,
            },
            parallel=parallel,
            parallel_inputs=parallel_inputs,
            rounding=rounding,
            round_output=round_output,
        )


def polars_ds_adapter(
    df,
    y: str,
    x: list[str] | str,
    weight: str | None = None,
    add_bias: bool = True,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    std_err: str = "hc3",
    null_policy: str = "raise",
) -> tuple[pl.DataFrame, pl.DataFrame, None, pl.DataFrame]:
    """
    Fit an OLS/WLS regression with polars_ds's `lin_reg_report` and return
    its coefficient table in survey_kit's normalized (df_estimates, df_ses,
    None, df_tidy) shape. Stays entirely in polars/narwhals - no pandas
    conversion.

    polars_ds doesn't expose a coefficient covariance matrix, so df_vcov is
    always None here: contrasts between two terms of the same fit
    (`.compare(other, compare_list_variables=[...])`) aren't calculable from
    this adapter's output alone. Use statsmodels_adapter or
    linearmodels_adapter if you need that.

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    y : dependent variable column name.
    x : predictor column name(s).
    weight : column name for weighted least squares, or None. **Note**: when
        weight is given, polars_ds always falls back to homoskedastic
        standard errors regardless of `std_err` - a limitation of the
        underlying library (it doesn't yet implement weighted HC0-3), not
        of this adapter. A warning is logged when this happens.
    add_bias : whether to add an intercept term (named "__bias__", polars_ds's
        own naming). Default is True.
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE value column in the output.
        Default is "estimate".
    std_err : one of "se" (classical/homoskedastic), "hc0", "hc1", "hc2",
        "hc3". Defaults to "hc3", for the same reason statsmodels_adapter
        defaults there - rarely safe to assume homoskedasticity. Silently
        ignored (falls back to "se") when weight is given - see the weight
        parameter above.
    null_policy : how to handle nulls in the predictors, passed straight to
        polars_ds. Default is "raise".

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, None, pl.DataFrame]
        (df_estimates, df_ses, None, df_tidy) - df_tidy is polars_ds's full
        `lin_reg_report` output as-is (features/beta/se/t/p/CI/r2/adj_r2),
        just with its term column renamed to join_on_name - a diagnostic
        snapshot only, never combined across implicates.
    """
    try:
        import polars_ds as pds
    except ImportError as e:
        message = (
            "polars_ds_adapter requires the 'polars_ds' package - "
            "install it with `uv add polars_ds`."
        )
        logger.error(message)
        raise ImportError(message) from e

    if weight and std_err != "se":
        logger.warning(
            "polars_ds ignores std_err when weight is given and falls back to "
            "homoskedastic standard errors - see polars_ds_adapter's docstring."
        )

    x = [x] if isinstance(x, str) else list(x)

    if isinstance(df, pl.DataFrame):
        df_pl = df.lazy()
    elif isinstance(df, pl.LazyFrame):
        df_pl = df
    else:
        import narwhals as nw

        df_pl = nw.from_native(df).lazy().collect().to_polars().lazy()

    report = (
        df_pl.select(
            pds.lin_reg_report(
                *x,
                target=y,
                weights=weight,
                add_bias=add_bias,
                null_policy=null_policy,
                std_err=std_err,
            ).alias("__report__")
        )
        .collect()
        .unnest("__report__")
    )

    se_col = "std_err" if (weight or std_err == "se") else f"{std_err}_se"

    df_estimates = report.select(
        pl.col("features").alias(join_on_name), pl.col("beta").alias(value_name)
    )
    df_ses = report.select(
        pl.col("features").alias(join_on_name), pl.col(se_col).alias(value_name)
    )
    df_tidy = report.rename({"features": join_on_name})

    return (df_estimates, df_ses, None, df_tidy)


def mi_ses_from_polars_ds(
    df_implicates,
    y: str,
    x: list[str] | str,
    weight: str | None = None,
    add_bias: bool = True,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    std_err: str = "hc3",
    null_policy: str = "raise",
    replicates=None,
    path_srmi: str = "",
    index: list | None = None,
    df_noimputes=None,
    parallel: bool = False,
    parallel_inputs=None,
    rounding=None,
    round_output: bool = True,
):
    """
    `mi_ses_from_function(delegate=polars_ds_adapter, ...)`, with
    `polars_ds_adapter`'s own arguments taken directly as keyword
    arguments instead of packed into an `arguments={}` dict.

    Parameters
    ----------
    df_implicates, path_srmi, index, df_noimputes, parallel,
        parallel_inputs, rounding, round_output : see
        [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
    y, x, weight, add_bias, join_on_name, value_name, std_err, null_policy :
        see `polars_ds_adapter`.
    replicates : `survey_kit.statistics.replicates.Replicates`, optional.
        When given, per-implicate SEs come from resampling across
        replicate weights instead of `std_err` - `polars_ds_adapter` is
        run once per replicate weight column (point estimates only, with
        `std_err` forced to "se" since it's discarded anyway), the same
        replicate-weight-bootstrap approach `mi_ses_from_stata`'s
        `replicates=` uses. `weight` is ignored in this mode; the
        implicate is converted to polars once (if it isn't already), not
        once per replicate. Default is None (use `polars_ds_adapter`'s own
        `std_err`).

    Returns
    -------
    MultipleImputation
        Same as `mi_ses_from_function`.

    See Also
    --------
    polars_ds_adapter : the delegate this wraps.
    """
    from .multiple_imputation import mi_ses_from_function

    base_arguments = {
        "y": y,
        "x": x,
        "add_bias": add_bias,
        "join_on_name": join_on_name,
        "value_name": value_name,
        "null_policy": null_policy,
    }

    if replicates is None:
        delegate = polars_ds_adapter
        arguments = {**base_arguments, "weight": weight, "std_err": std_err}
    else:
        if weight:
            logger.warning(
                "mi_ses_from_polars_ds: weight is ignored when replicates "
                "is given - each replicate call supplies its own weight column."
            )
        delegate = _mi_ses_replicates_delegate(
            polars_ds_adapter,
            {**base_arguments, "std_err": "se"},
            join_on_name,
            replicates,
            convert=_to_polars,
        )
        arguments = {}

    return mi_ses_from_function(
        delegate=delegate,
        df_implicates=df_implicates,
        join_on=[join_on_name],
        path_srmi=path_srmi,
        index=index,
        df_noimputes=df_noimputes,
        arguments=arguments,
        parallel=parallel,
        parallel_inputs=parallel_inputs,
        rounding=rounding,
        round_output=round_output,
    )


def r_lm_adapter(
    df,
    formula: str,
    weight: str | None = None,
    family: str | None = None,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    **r_kwargs,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Fit a base-R lm()/glm() model and return its coefficient table in
    survey_kit's normalized (df_estimates, df_ses, df_vcov, df_tidy) shape.
    Needs only R itself (no extra R packages) plus rpy2/rpy2-arrow on the
    Python side - see `survey_kit.statistics._r_interop.check_r_setup()`.

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    formula : R formula string, e.g. "y ~ x1 + x2".
    weight : column name for weighted least squares, or None. Default is
        None. Passed as lm()/glm()'s `weights=` argument, referencing the
        column directly in R (base R's non-standard evaluation resolves it
        against `data=`, the same way the formula itself does) - unlike
        most other arguments here, this can't be a plain quoted string.
    family : a plain R family name (e.g. "binomial", "poisson") to fit via
        glm() instead of lm() - base R resolves the string to the family
        function itself. For a non-default link function, wrap the full
        expression in `_r_interop.RRaw`, e.g.
        `family=RRaw('binomial(link="probit")')`. Default is None (lm(),
        i.e. OLS/WLS).
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE/covariance value column in the
        output. Default is "estimate".
    **r_kwargs : any other lm()/glm() argument (e.g. `subset=`, `na.action=`,
        `offset=`), converted to R via `_r_interop.py_to_r_literal`:
        None omits the argument, bool/int/float/str/list/dict convert
        naturally, a string starting with "~" is passed through raw (a
        formula), and `_r_interop.RRaw` wraps any other
        literal R code you need verbatim (e.g. a bare column reference).
        R argument names with a "." (e.g. `na.action`) can't be Python
        keyword names - pass those via a dict: `r_kwargs={"na.action": ...}`
        merged into this call, or just use `**{"na.action": ...}`.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (df_estimates, df_ses, df_vcov, df_tidy) - df_vcov is always
        populated here since vcov() is free alongside coef() in R. df_tidy
        is R's own `summary(fit)$coefficients` (Estimate/Std. Error/t or z
        value/Pr(>|.|)) - a diagnostic snapshot only, never combined across
        implicates.

    Notes
    -----
    Base R's lm()/glm() only provide classical (non-robust) standard errors
    - there's no HC0-3 equivalent without the 'sandwich' package, which this
    adapter deliberately doesn't pull in (keeping the R-side dependency at
    just R itself). Use `r_fixest_adapter` for built-in robust/cluster SEs.
    """
    from . import _r_interop as _r

    fn = "glm" if family else "lm"
    fit_code = _r.call(
        fn,
        _r.RRaw(formula),
        data=_r.RRaw("{df}"),
        weights=_r.RRaw(weight) if weight else None,
        family=family,
        **r_kwargs,
    )

    coef, vcov, tidy = _r.fit_r_model(
        df, fit_code, tidy_code="summary({fit})$coefficients"
    )

    df_estimates = _r.coef_table(coef, join_on_name, value_name)
    df_ses = _r.ses_from_vcov(vcov, join_on_name, value_name)
    df_vcov = _r.vcov_table(vcov, join_on_name, value_name)
    df_tidy = _r.matrix_table(tidy, join_on_name)

    return (df_estimates, df_ses, df_vcov, df_tidy)


def r_fixest_adapter(
    df,
    formula: str,
    func: str = "feols",
    weight: str | None = None,
    family: str | None = None,
    vcov: str | None = None,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    **r_kwargs,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Fit any fixest regression and return its coefficient table in
    survey_kit's normalized (df_estimates, df_ses, df_vcov, df_tidy) shape.
    fixest supports fixed effects directly in the formula (e.g. "y ~ x1 |
    firm + year") and computes robust/clustered SEs natively - no 'sandwich'
    needed. Requires the R 'fixest' package - see
    `survey_kit.statistics._r_interop.check_r_setup(["fixest"])`.

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    formula : fixest formula string, e.g. "y ~ x1 + x2 | firm" for a fit
        with a firm fixed effect.
    func : which fixest estimator to call, e.g. "feols" (default), "feglm",
        "fepois", "femlm", "feNmlm", "feglm.fit" - anything in the fixest
        namespace. Called as `fixest::{func}(...)`.
    weight : column name for weighted estimation, or None. Default is None.
        Passed as fixest's `weights=~column` (a one-sided formula - unlike
        most other arguments here, this can't be a plain quoted string).
    family : a plain R family name (e.g. "binomial", "poisson") - only
        meaningful for func="feglm"/"femlm". For a non-default link
        function, wrap the full expression in `_r_interop.RRaw`. Default
        is None.
    vcov : fixest's own `vcov=` argument - a string like "hetero" (robust)
        or "iid" (classical), or a one-sided formula string like "~firm"
        for cluster-robust SEs. Defaults to "hetero" for the same reason
        statsmodels_adapter defaults to HC3 - rarely safe to assume
        homoskedasticity - *unless* you pass `cluster=` via r_kwargs, in
        which case vcov is left unset so fixest can infer clustered SEs
        from it: fixest raises if vcov and cluster are both given.
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE/covariance value column in the
        output. Default is "estimate".
    **r_kwargs : any other argument any fixest estimator takes - `cluster`,
        `panel.id`, `split`/`fsplit`, `se`, `ssc`, `lean`, `notes`,
        `verbose`, etc. - converted to R via `_r_interop.py_to_r_literal`:
        None omits the argument, bool/int/float/str/list/dict convert
        naturally (a plain column name like `cluster="firm"` works exactly
        as it does when you type it directly in fixest), a string starting
        with "~" is passed through raw (a formula, e.g.
        `panel.id="~id+time"`), and `_r_interop.RRaw` wraps anything else
        that needs to be emitted as literal R code (e.g.
        `ssc=RRaw('ssc(fixef.K="full")')`). R argument names with a "."
        (e.g. `panel.id`) can't be Python keyword names directly - build the
        kwargs dict separately and splat it: `**{"panel.id": "~id+time"}`.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (df_estimates, df_ses, df_vcov) - df_vcov is always populated here
        since vcov() is free alongside coef() in fixest's results.
    """
    return _fixest_fit(
        df, func, formula, weight, family, vcov, join_on_name, value_name, r_kwargs
    )


def _fixest_fit(
    df,
    func: str,
    formula: str,
    weight: str | None,
    family: str | None,
    vcov: str | None,
    join_on_name: str,
    value_name: str,
    r_kwargs: dict,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Shared core behind r_fixest_adapter and the r_feols/feglm/fepois/femlm
    wrappers. Returns (df_estimates, df_ses, df_vcov, df_tidy) - df_tidy is
    fixest's own `coeftable()` (Estimate/Std. Error/t value/Pr(>|t|)), a
    diagnostic snapshot of this one fit only, never combined across
    implicates.
    """
    from . import _r_interop as _r

    _r.get_library("fixest")

    #   fixest rejects vcov= combined with cluster= - let cluster= alone
    #   drive clustered SEs when the caller doesn't explicitly set vcov.
    if vcov is None and "cluster" not in r_kwargs:
        vcov = "hetero"

    fit_code = _r.call(
        f"fixest::{func}",
        _r.RRaw(formula),
        data=_r.RRaw("{df}"),
        weights=_r.RRaw(f"~{weight}") if weight else None,
        family=family,
        vcov=vcov,
        **r_kwargs,
    )

    coef, vcov_r, tidy = _r.fit_r_model(
        df, fit_code, tidy_code="fixest::coeftable({fit})"
    )

    df_estimates = _r.coef_table(coef, join_on_name, value_name)
    df_ses = _r.ses_from_vcov(vcov_r, join_on_name, value_name)
    df_vcov = _r.vcov_table(vcov_r, join_on_name, value_name)
    df_tidy = _r.matrix_table(tidy, join_on_name)

    return (df_estimates, df_ses, df_vcov, df_tidy)


def _fixest_named_kwargs(
    cluster,
    panel_id,
    ssc,
    fixef,
    lean,
    notes,
    verbose,
    r_kwargs: dict,
) -> dict:
    """
    Merge fixest's most commonly-tuned arguments (given as proper Python
    parameters, for IDE-discoverability) with any overflow **r_kwargs, into
    one dict of R argument name -> Python value ready for call. Explicit
    r_kwargs entries win if there's a collision.
    """
    named = {
        "cluster": cluster,
        "panel.id": panel_id,
        "ssc": ssc,
        "fixef": fixef,
        "lean": lean,
        "notes": notes,
        "verbose": verbose,
    }
    named = {k: v for k, v in named.items() if v is not None}
    named.update(r_kwargs)
    return named


_FIXEST_COMMON_PARAMS_DOC = """\
    cluster : column name(s) for cluster-robust SEs, or None. Default is
        None. A plain column name works directly (`cluster="firm"`); for
        two-way clustering pass a list (`cluster=["firm", "year"]`) or a
        formula string (`cluster="~firm+year"`). Setting this leaves `vcov`
        unset (see above) so fixest infers clustering from it.
    panel_id : panel identifier(s) for lagged/leaded variables in the
        formula - a formula string (`panel_id="~id+time"`) or a 2-element
        list (`panel_id=["id", "time"]`). Default is None. Maps to fixest's
        `panel.id` (which can't be a Python keyword name directly).
    ssc : small-sample correction settings, e.g.
        `ssc=RRaw('ssc(fixef.K="full")')` (fixest's `ssc()` helper returns
        an object, not a plain scalar/string - wrap it in
        [`RRaw`][survey_kit.statistics._r_interop.RRaw]). Default is None
        (fixest's own default).
    fixef : fixed-effect column name(s) as an alternative to writing them
        into `formula` after "|". Default is None.
    lean : if True, drop large intermediate objects from the fitted result
        to save memory - irrelevant here since only coef()/vcov() are read
        back, but harmless to set. Default is None (fixest's own default,
        False).
    notes : whether to print fixest's usual notes (about NA-dropping,
        collinearity, etc.) to the R console. Default is None (fixest's own
        default).
    verbose : fixest's verbosity level for iterative fitting. Default is
        None (fixest's own default, 0).
    **r_kwargs : anything else fixest takes (`se`, `split`/`fsplit`,
        `fixef.tol`, `nthreads`, ...) - same conversion rules as
        `r_fixest_adapter`'s `**r_kwargs`.
"""


def r_feols(
    df,
    formula: str,
    weight: str | None = None,
    vcov: str | None = None,
    cluster=None,
    panel_id=None,
    ssc=None,
    fixef=None,
    lean: bool | None = None,
    notes: bool | None = None,
    verbose: int | None = None,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    **r_kwargs,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    fixest::feols() (linear regression, with fixed effects) with arguments
    mirroring fixest's own. See `r_fixest_adapter` for the fully generic
    `func=` version (feNmlm, feglm.fit, ...) and the general conversion
    rules; this is the same thing with feols's common arguments spelled out.

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    formula : fixest formula string, e.g. "y ~ x1 + x2 | firm" for a fit
        with a firm fixed effect.
    weight : column name for weighted estimation, or None. Passed as
        `weights=~column` (fixest needs a formula here, not a plain string).
    vcov : "hetero" (robust, the default), "iid" (classical), or a one-sided
        formula string like "~firm" for cluster-robust SEs.
{params}
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE/covariance value column in the
        output. Default is "estimate".

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (df_estimates, df_ses, df_vcov, df_tidy) - df_tidy is fixest's own
        coeftable() (Estimate/Std. Error/t value/Pr(>|t|)), a diagnostic
        snapshot only, never combined across implicates.
    """
    r_kwargs = _fixest_named_kwargs(cluster, panel_id, ssc, fixef, lean, notes, verbose, r_kwargs)
    return _fixest_fit(df, "feols", formula, weight, None, vcov, join_on_name, value_name, r_kwargs)


def r_feglm(
    df,
    formula: str,
    family: str = "gaussian",
    weight: str | None = None,
    vcov: str | None = None,
    cluster=None,
    panel_id=None,
    ssc=None,
    fixef=None,
    lean: bool | None = None,
    notes: bool | None = None,
    verbose: int | None = None,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    **r_kwargs,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    fixest::feglm() (GLM, with fixed effects) with arguments mirroring
    fixest's own. See `r_fixest_adapter` for the fully generic `func=`
    version and the general conversion rules.

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    formula : fixest formula string, e.g. "y ~ x1 + x2 | firm".
    family : a plain R family name, e.g. "binomial", "poisson" - passed as a
        quoted string (fixest resolves it to the family function itself).
        For a non-default link function, wrap the full expression in
        [`RRaw`][survey_kit.statistics._r_interop.RRaw], e.g.
        `family=RRaw('binomial(link="probit")')`. Default is "gaussian"
        (matching feglm's own default) - for a pure Poisson fit,
        `r_fepois` is faster and doesn't need this.
    weight : column name for weighted estimation, or None. Passed as
        `weights=~column`.
    vcov : "hetero" (robust, the default), "iid" (classical), or a one-sided
        formula string like "~firm" for cluster-robust SEs.
{params}
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE/covariance value column in the
        output. Default is "estimate".

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (df_estimates, df_ses, df_vcov, df_tidy) - df_tidy is fixest's own
        coeftable() (Estimate/Std. Error/t value/Pr(>|t|)), a diagnostic
        snapshot only, never combined across implicates.
    """
    r_kwargs = _fixest_named_kwargs(cluster, panel_id, ssc, fixef, lean, notes, verbose, r_kwargs)
    return _fixest_fit(df, "feglm", formula, weight, family, vcov, join_on_name, value_name, r_kwargs)


def r_fepois(
    df,
    formula: str,
    weight: str | None = None,
    vcov: str | None = None,
    cluster=None,
    panel_id=None,
    ssc=None,
    fixef=None,
    lean: bool | None = None,
    notes: bool | None = None,
    verbose: int | None = None,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    **r_kwargs,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    fixest::fepois() (Poisson regression, with fixed effects) with arguments
    mirroring fixest's own. See `r_fixest_adapter` for the fully generic
    `func=` version and the general conversion rules.

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    formula : fixest formula string, e.g. "y ~ x1 + x2 | firm".
    weight : column name for weighted estimation, or None. Passed as
        `weights=~column`.
    vcov : "hetero" (robust, the default), "iid" (classical), or a one-sided
        formula string like "~firm" for cluster-robust SEs.
{params}
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE/covariance value column in the
        output. Default is "estimate".

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (df_estimates, df_ses, df_vcov, df_tidy) - df_tidy is fixest's own
        coeftable() (Estimate/Std. Error/t value/Pr(>|t|)), a diagnostic
        snapshot only, never combined across implicates.
    """
    r_kwargs = _fixest_named_kwargs(cluster, panel_id, ssc, fixef, lean, notes, verbose, r_kwargs)
    return _fixest_fit(df, "fepois", formula, weight, None, vcov, join_on_name, value_name, r_kwargs)


def r_femlm(
    df,
    formula: str,
    family: str = "poisson",
    vcov: str | None = None,
    cluster=None,
    panel_id=None,
    ssc=None,
    fixef=None,
    lean: bool | None = None,
    notes: bool | None = None,
    verbose: int | None = None,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    **r_kwargs,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    fixest::femlm() (max-likelihood: Poisson/negative binomial/logit/
    Gaussian, with fixed effects) with arguments mirroring fixest's own.
    See `r_fixest_adapter` for the fully generic `func=` version and the
    general conversion rules. Note: femlm has no `weights=` argument
    (unlike feols/feglm/fepois).

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    formula : fixest formula string, e.g. "y ~ x1 + x2 | firm".
    family : one of "poisson" (default), "negbin", "logit", "gaussian" -
        a plain string (femlm, unlike feglm, only accepts one of these four
        exact names - there's no link-function customization here).
    vcov : "hetero" (robust, the default), "iid" (classical), or a one-sided
        formula string like "~firm" for cluster-robust SEs.
{params}
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE/covariance value column in the
        output. Default is "estimate".

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (df_estimates, df_ses, df_vcov, df_tidy) - df_tidy is fixest's own
        coeftable() (Estimate/Std. Error/t value/Pr(>|t|)), a diagnostic
        snapshot only, never combined across implicates.
    """
    r_kwargs = _fixest_named_kwargs(cluster, panel_id, ssc, fixef, lean, notes, verbose, r_kwargs)
    return _fixest_fit(df, "femlm", formula, None, family, vcov, join_on_name, value_name, r_kwargs)


for _fn in (r_feols, r_feglm, r_fepois, r_femlm):
    _fn.__doc__ = _fn.__doc__.format(params=_FIXEST_COMMON_PARAMS_DOC)
del _fn


class mi_ses_from_r_fixest:
    """
    Namespace for per-estimator `mi_ses_from_function(delegate=r_feols/
    r_feglm/r_fepois/r_femlm, ...)` wrappers - `mi_ses_from_r_fixest.feols(...)`,
    `.feglm(...)`, `.fepois(...)`, `.femlm(...)` - each naming that
    estimator's own arguments (`formula`, `weight`, `vcov`, `cluster`,
    `panel_id`, ...) directly instead of packing them into an
    `arguments={}` dict, so IDEs show the right parameters for the one
    you're actually calling. Everything else the R side of fixest accepts
    still flows through `**r_kwargs` exactly as in `r_feols`/`r_feglm`/
    `r_fepois`/`r_femlm`. See `r_fixest_adapter` for the fully generic
    `func=` escape hatch (feNmlm, feglm.fit, ...) this doesn't cover.
    """

    @staticmethod
    def feols(
        df_implicates,
        formula: str,
        weight: str | None = None,
        vcov: str | None = None,
        cluster=None,
        panel_id=None,
        ssc=None,
        fixef=None,
        lean: bool | None = None,
        notes: bool | None = None,
        verbose: int | None = None,
        join_on_name: str = "Variable",
        value_name: str = "estimate",
        replicates=None,
        path_srmi: str = "",
        index: list | None = None,
        df_noimputes=None,
        parallel: bool = False,
        parallel_inputs=None,
        rounding=None,
        round_output: bool = True,
        **r_kwargs,
    ):
        """
        `mi_ses_from_function(delegate=r_feols, ...)` - fixest::feols()
        (linear regression, with fixed effects) across implicates.

        Parameters
        ----------
        df_implicates, path_srmi, index, df_noimputes, parallel,
            parallel_inputs, rounding, round_output : see
            [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
        formula, weight, vcov, cluster, panel_id, ssc, fixef, lean, notes,
            verbose, join_on_name, value_name, **r_kwargs : see `r_feols`.
        replicates : `survey_kit.statistics.replicates.Replicates`,
            optional. When given, per-implicate SEs come from resampling
            across replicate weights instead of `vcov`/`cluster` - `r_feols`
            is run once per replicate weight column (point estimates only,
            with `vcov` forced to "iid" and `cluster` forced off since
            they're discarded anyway), the same replicate-weight-bootstrap
            approach `mi_ses_from_stata`'s `replicates=` uses. `weight` is
            ignored in this mode; the implicate is converted to an R
            data.frame once, not once per replicate. Default is None (use
            `r_feols`'s own `vcov`/`cluster`).

        Returns
        -------
        MultipleImputation
            Same as `mi_ses_from_function`.

        See Also
        --------
        r_feols : the delegate this wraps.
        """
        from .multiple_imputation import mi_ses_from_function

        base_arguments = {
            "formula": formula,
            "panel_id": panel_id,
            "ssc": ssc,
            "fixef": fixef,
            "lean": lean,
            "notes": notes,
            "verbose": verbose,
            "join_on_name": join_on_name,
            "value_name": value_name,
            **r_kwargs,
        }

        if replicates is None:
            delegate = r_feols
            arguments = {**base_arguments, "weight": weight, "vcov": vcov, "cluster": cluster}
        else:
            if weight:
                logger.warning(
                    "mi_ses_from_r_fixest.feols: weight is ignored when "
                    "replicates is given - each replicate call supplies "
                    "its own weight column."
                )
            from . import _r_interop as _r

            delegate = _mi_ses_replicates_delegate(
                r_feols,
                {**base_arguments, "vcov": "iid", "cluster": None},
                join_on_name,
                replicates,
                convert=_r.dataframe_to_r,
            )
            arguments = {}

        return mi_ses_from_function(
            delegate=delegate,
            df_implicates=df_implicates,
            join_on=[join_on_name],
            path_srmi=path_srmi,
            index=index,
            df_noimputes=df_noimputes,
            arguments=arguments,
            parallel=parallel,
            parallel_inputs=parallel_inputs,
            rounding=rounding,
            round_output=round_output,
        )

    @staticmethod
    def feglm(
        df_implicates,
        formula: str,
        family: str = "gaussian",
        weight: str | None = None,
        vcov: str | None = None,
        cluster=None,
        panel_id=None,
        ssc=None,
        fixef=None,
        lean: bool | None = None,
        notes: bool | None = None,
        verbose: int | None = None,
        join_on_name: str = "Variable",
        value_name: str = "estimate",
        replicates=None,
        path_srmi: str = "",
        index: list | None = None,
        df_noimputes=None,
        parallel: bool = False,
        parallel_inputs=None,
        rounding=None,
        round_output: bool = True,
        **r_kwargs,
    ):
        """
        `mi_ses_from_function(delegate=r_feglm, ...)` - fixest::feglm()
        (GLM, with fixed effects) across implicates.

        Parameters
        ----------
        df_implicates, path_srmi, index, df_noimputes, parallel,
            parallel_inputs, rounding, round_output : see
            [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
        formula, family, weight, vcov, cluster, panel_id, ssc, fixef,
            lean, notes, verbose, join_on_name, value_name, **r_kwargs :
            see `r_feglm`.
        replicates : see `mi_ses_from_r_fixest.feols` - same
            replicate-weight-bootstrap option, `vcov` forced to "iid",
            `cluster` forced off, and `weight` ignored in this mode.
            Default is None.

        Returns
        -------
        MultipleImputation
            Same as `mi_ses_from_function`.

        See Also
        --------
        r_feglm : the delegate this wraps.
        """
        from .multiple_imputation import mi_ses_from_function

        base_arguments = {
            "formula": formula,
            "family": family,
            "panel_id": panel_id,
            "ssc": ssc,
            "fixef": fixef,
            "lean": lean,
            "notes": notes,
            "verbose": verbose,
            "join_on_name": join_on_name,
            "value_name": value_name,
            **r_kwargs,
        }

        if replicates is None:
            delegate = r_feglm
            arguments = {**base_arguments, "weight": weight, "vcov": vcov, "cluster": cluster}
        else:
            if weight:
                logger.warning(
                    "mi_ses_from_r_fixest.feglm: weight is ignored when "
                    "replicates is given - each replicate call supplies "
                    "its own weight column."
                )
            from . import _r_interop as _r

            delegate = _mi_ses_replicates_delegate(
                r_feglm,
                {**base_arguments, "vcov": "iid", "cluster": None},
                join_on_name,
                replicates,
                convert=_r.dataframe_to_r,
            )
            arguments = {}

        return mi_ses_from_function(
            delegate=delegate,
            df_implicates=df_implicates,
            join_on=[join_on_name],
            path_srmi=path_srmi,
            index=index,
            df_noimputes=df_noimputes,
            arguments=arguments,
            parallel=parallel,
            parallel_inputs=parallel_inputs,
            rounding=rounding,
            round_output=round_output,
        )

    @staticmethod
    def fepois(
        df_implicates,
        formula: str,
        weight: str | None = None,
        vcov: str | None = None,
        cluster=None,
        panel_id=None,
        ssc=None,
        fixef=None,
        lean: bool | None = None,
        notes: bool | None = None,
        verbose: int | None = None,
        join_on_name: str = "Variable",
        value_name: str = "estimate",
        replicates=None,
        path_srmi: str = "",
        index: list | None = None,
        df_noimputes=None,
        parallel: bool = False,
        parallel_inputs=None,
        rounding=None,
        round_output: bool = True,
        **r_kwargs,
    ):
        """
        `mi_ses_from_function(delegate=r_fepois, ...)` - fixest::fepois()
        (Poisson regression, with fixed effects) across implicates.

        Parameters
        ----------
        df_implicates, path_srmi, index, df_noimputes, parallel,
            parallel_inputs, rounding, round_output : see
            [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
        formula, weight, vcov, cluster, panel_id, ssc, fixef, lean, notes,
            verbose, join_on_name, value_name, **r_kwargs : see `r_fepois`.
        replicates : see `mi_ses_from_r_fixest.feols` - same
            replicate-weight-bootstrap option, `vcov` forced to "iid",
            `cluster` forced off, and `weight` ignored in this mode.
            Default is None.

        Returns
        -------
        MultipleImputation
            Same as `mi_ses_from_function`.

        See Also
        --------
        r_fepois : the delegate this wraps.
        """
        from .multiple_imputation import mi_ses_from_function

        base_arguments = {
            "formula": formula,
            "panel_id": panel_id,
            "ssc": ssc,
            "fixef": fixef,
            "lean": lean,
            "notes": notes,
            "verbose": verbose,
            "join_on_name": join_on_name,
            "value_name": value_name,
            **r_kwargs,
        }

        if replicates is None:
            delegate = r_fepois
            arguments = {**base_arguments, "weight": weight, "vcov": vcov, "cluster": cluster}
        else:
            if weight:
                logger.warning(
                    "mi_ses_from_r_fixest.fepois: weight is ignored when "
                    "replicates is given - each replicate call supplies "
                    "its own weight column."
                )
            from . import _r_interop as _r

            delegate = _mi_ses_replicates_delegate(
                r_fepois,
                {**base_arguments, "vcov": "iid", "cluster": None},
                join_on_name,
                replicates,
                convert=_r.dataframe_to_r,
            )
            arguments = {}

        return mi_ses_from_function(
            delegate=delegate,
            df_implicates=df_implicates,
            join_on=[join_on_name],
            path_srmi=path_srmi,
            index=index,
            df_noimputes=df_noimputes,
            arguments=arguments,
            parallel=parallel,
            parallel_inputs=parallel_inputs,
            rounding=rounding,
            round_output=round_output,
        )

    @staticmethod
    def femlm(
        df_implicates,
        formula: str,
        family: str = "poisson",
        vcov: str | None = None,
        cluster=None,
        panel_id=None,
        ssc=None,
        fixef=None,
        lean: bool | None = None,
        notes: bool | None = None,
        verbose: int | None = None,
        join_on_name: str = "Variable",
        value_name: str = "estimate",
        path_srmi: str = "",
        index: list | None = None,
        df_noimputes=None,
        parallel: bool = False,
        parallel_inputs=None,
        rounding=None,
        round_output: bool = True,
        **r_kwargs,
    ):
        """
        `mi_ses_from_function(delegate=r_femlm, ...)` - fixest::femlm()
        (max-likelihood: Poisson/negative binomial/logit/Gaussian, with
        fixed effects) across implicates. Note: femlm has no `weight=`
        argument (unlike feols/feglm/fepois), so - unlike those three -
        there's no `replicates=` option here either: nothing to substitute
        a replicate weight column into.

        Parameters
        ----------
        df_implicates, path_srmi, index, df_noimputes, parallel,
            parallel_inputs, rounding, round_output : see
            [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
        formula, family, vcov, cluster, panel_id, ssc, fixef, lean, notes,
            verbose, join_on_name, value_name, **r_kwargs : see `r_femlm`.

        Returns
        -------
        MultipleImputation
            Same as `mi_ses_from_function`.

        See Also
        --------
        r_femlm : the delegate this wraps.
        """
        from .multiple_imputation import mi_ses_from_function

        return mi_ses_from_function(
            delegate=r_femlm,
            df_implicates=df_implicates,
            join_on=[join_on_name],
            path_srmi=path_srmi,
            index=index,
            df_noimputes=df_noimputes,
            arguments={
                "formula": formula,
                "family": family,
                "vcov": vcov,
                "cluster": cluster,
                "panel_id": panel_id,
                "ssc": ssc,
                "fixef": fixef,
                "lean": lean,
                "notes": notes,
                "verbose": verbose,
                "join_on_name": join_on_name,
                "value_name": value_name,
                **r_kwargs,
            },
            parallel=parallel,
            parallel_inputs=parallel_inputs,
            rounding=rounding,
            round_output=round_output,
        )


def stata_adapter(
    df,
    command: str | list[str],
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    edition: str | None = None,
    stata_path: str | None = None,
    reuse_data: bool = False,
    quietly: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Run an arbitrary Stata e-class estimation command (regress, logit,
    xtreg, areg, svy: ..., or anything from an installed community package)
    and return its coefficient table in survey_kit's normalized
    (df_estimates, df_ses, df_vcov, df_tidy) shape.

    See `survey_kit.statistics._stata_interop`'s module docstring for
    details, and `check_stata_setup()` there for a setup diagnostic.

    Data moves into Stata via a .dta file written by polars_readstat
    (`write_readstat`) rather than pystata's own DataFrame transfer - `.dta`
    is Stata's own native, most battle-tested ingestion path, so this
    doesn't depend on however pystata's transfer mechanism behaves.

    Parameters
    ----------
    df : the merged implicate data (supplied by mi_ses_from_function).
    command : the Stata command to run, e.g. "regress y x1 x2", or
        "regress y x1 x2 [pw=w]", or "xtreg y x1 x2, fe". Must leave
        e(b)/e(V) populated - true of most estimation commands. Pass a
        list of commands run in order, instead of a single string, when
        you need setup (`gen`/`svyset`/...) between `use` and the
        estimation command - only the LAST command's e(b)/e(V)/r(table)
        are read back, e.g.
        `["svyset psu [pw=weight], strata(strata)", "svy: regress y x1 x2"]`.
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the coefficient/SE/covariance value column in the
        output. Default is "estimate".
    edition, stata_path : forwarded to
        `_stata_interop.require_pystata` - stata_path (your Stata install
        directory) is only needed if pystata's utilities folder isn't
        already on sys.path; edition ("be"/"se"/"mp") only if it can't be
        auto-detected. Only used on the first call in a
        process - Stata stays initialized afterward, like a package import.
    reuse_data : if True, skip re-exporting/re-`use`-ing df when it's the
        same object (by identity) as a previous reuse_data=True call -
        useful when calling this repeatedly for the same underlying data
        (e.g. once per replicate weight, if `command` already references
        the weight column directly rather than needing a different df per
        call). Default is False. Call
        `survey_kit.statistics._stata_interop.clear_stata_cache()` once
        done with data reused this way, to free Stata's own copy.
    quietly : pass False to let `command` stream Stata's own console
        output on success too (failures always surface the real error
        text regardless - see `_stata_interop._run_in_stata`'s
        docstring). Default is True.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (df_estimates, df_ses, df_vcov, df_tidy) - df_vcov from e(V).
        df_tidy is Stata's own r(table) (the matrix `ci`/`test`/etc. use
        internally: b/se/t-or-z/p-value/CI, transposed to one row per term)
        populated automatically after any estimation command - a
        diagnostic snapshot only, never combined across implicates.
    """
    from . import _stata_interop as _st

    b, b_names, V, table, table_row_names, table_col_names = _st.run_stata_model(
        df,
        command,
        edition=edition,
        stata_path=stata_path,
        reuse_data=reuse_data,
        quietly=quietly,
    )

    df_estimates = pl.DataFrame({join_on_name: b_names, value_name: b})

    import numpy as np

    v_arr = np.asarray(V)
    se = np.sqrt(np.diag(v_arr)).tolist()
    df_ses = pl.DataFrame({join_on_name: b_names, value_name: se})

    n = len(b_names)
    df_vcov = pl.DataFrame(
        [
            {
                f"{join_on_name}_1": b_names[i],
                f"{join_on_name}_2": b_names[j],
                value_name: float(v_arr[i, j]),
            }
            for i in range(n)
            for j in range(n)
        ]
    )

    #   r(table) is stat-by-term (rows=stats, cols=terms) - transpose to
    #   the one-row-per-term shape every other adapter's df_tidy uses.
    table_arr = np.asarray(table)
    tidy_data = {join_on_name: table_col_names}
    for i, stat_name in enumerate(table_row_names):
        tidy_data[stat_name] = table_arr[i, :].tolist()
    df_tidy = pl.DataFrame(tidy_data)

    return (df_estimates, df_ses, df_vcov, df_tidy)


def mi_ses_from_stata(
    df_implicates,
    command: str | list[str],
    path_srmi: str = "",
    index: list | None = None,
    df_noimputes=None,
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    edition: str | None = None,
    stata_path: str | None = None,
    reuse_data: bool = False,
    quietly: bool = True,
    replicates=None,
    parallel: bool = False,
    parallel_inputs=None,
    rounding=None,
    round_output: bool = True,
):
    """
    `mi_ses_from_function(delegate=stata_adapter, ...)`, with
    `stata_adapter`'s own arguments (`command`, `edition`, `stata_path`,
    ...) taken directly as keyword arguments instead of packed into an
    `arguments={}` dict - the common case of running one Stata command
    across implicates.

    Parameters
    ----------
    df_implicates, path_srmi, index, df_noimputes, parallel,
        parallel_inputs, rounding, round_output : see
        [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function].
    command, join_on_name, value_name, edition, stata_path, reuse_data,
        quietly : see `stata_adapter`. When `replicates` is given, `command`
        needs a "{weight}" placeholder instead (see below) - `reuse_data`
        is then forced on internally regardless of what's passed here (see
        `replicates`).
    replicates : `survey_kit.statistics.replicates.Replicates`, optional.
        When given, per-implicate SEs come from resampling across
        replicate weights instead of `command`'s own e(V) - `command` is
        run once per replicate weight column (via `stata_results_adapter`,
        reading back e(b) only) rather than Stata's own `bootstrap`/`brr`/
        `jackknife` prefix, so it needs a "{weight}" placeholder, e.g.
        `"regress y x1 x2 [pw={weight}]"`. Stata's own replicate-estimation
        commands are usually faster and more idiomatic if you're already
        set up for them - this is for reusing survey_kit's own
        Replicates/StatCalculator machinery (e.g. to match SEs computed
        the same way elsewhere in a project) instead. Default is None (use
        `command`'s own e(V), the `stata_adapter` path).

    Returns
    -------
    MultipleImputation
        Same as `mi_ses_from_function`.

    Examples
    --------
    >>> mi_reg = mi_ses_from_stata(
    ...     df_implicates=df_implicates,
    ...     command="regress y x1 x2",
    ... )

    Replicate-weight SEs instead of Stata's own e(V):

    >>> from survey_kit.statistics.replicates import Replicates
    >>> mi_reg = mi_ses_from_stata(
    ...     df_implicates=df_implicates,
    ...     command="regress y x1 x2 [pw={weight}]",
    ...     replicates=Replicates(weight_stub="replicate_", n_replicates=80),
    ... )

    See Also
    --------
    stata_adapter, stata_results_adapter : the delegates this wraps.
    [`mi_ses_from_function`][survey_kit.statistics.multiple_imputation.mi_ses_from_function] :
        the general-purpose function this specializes.
    """
    from .calculator import StatCalculator
    from .multiple_imputation import mi_ses_from_function
    from . import _stata_interop as _st

    if replicates is None:
        delegate = stata_adapter
        arguments = {
            "command": command,
            "join_on_name": join_on_name,
            "value_name": value_name,
            "edition": edition,
            "stata_path": stata_path,
            "reuse_data": reuse_data,
            "quietly": quietly,
        }
    else:

        def delegate(df):
            calc = StatCalculator.from_function(
                delegate=stata_results_adapter,
                estimate_ids=[join_on_name],
                df=df,
                arguments={
                    "command": command,
                    "results": ["e(b)"],
                    "join_on_name": join_on_name,
                    "value_name": value_name,
                    "edition": edition,
                    "stata_path": stata_path,
                    #   the same implicate df is reused across every
                    #   replicate weight in this loop - only `weight`
                    #   changes - so re-exporting/re-`use`-ing it every
                    #   call would be pure waste.
                    "reuse_data": True,
                    "quietly": quietly,
                },
                replicates=replicates,
                display=False,
            )
            _st.clear_stata_cache()
            return calc

        arguments = {}

    return mi_ses_from_function(
        delegate=delegate,
        df_implicates=df_implicates,
        join_on=[join_on_name],
        path_srmi=path_srmi,
        index=index,
        df_noimputes=df_noimputes,
        arguments=arguments,
        parallel=parallel,
        parallel_inputs=parallel_inputs,
        rounding=rounding,
        round_output=round_output,
    )


def stata_results_adapter(
    df,
    command: str | list[str],
    results: list[str],
    weight: str = "",
    join_on_name: str = "Variable",
    value_name: str = "estimate",
    edition: str | None = None,
    stata_path: str | None = None,
    reuse_data: bool = False,
    quietly: bool = True,
) -> pl.DataFrame:
    """
    Run an arbitrary Stata command (r-class or e-class) and return a flat
    table of exactly the r()/e() results you name - the Stata counterpart
    to survey_kit's generic R/rpy2 escape hatch (`get_library`/
    `extract_fit` in `_r_interop.py`), but shaped as a
    [`StatCalculator.from_function`][survey_kit.statistics.calculator.StatCalculator.from_function]
    delegate (df, weight -> one row per estimate) rather than an
    `mi_ses_from_function` delegate: point estimates only, no vcov. Use
    this for bootstrap/replicate-weight variance - the spread of estimates
    across replicate weights IS the SE (computed by
    `Replicates`/`StatCalculator`), not Stata's own e(V) - the same reason
    the R tutorial's ad-hoc `run_regression` delegate returns just a plain
    coefficient table with no SE of its own.

    Unlike `stata_adapter` (which assumes an e-class fit and always reads
    the fixed e(b)/e(V)/r(table) triplet), this works for ANY command that
    populates r()/e() results - `summarize`, `tabstat`, `ci`, `svy: mean`,
    or a full e-class regression - you just name what you want back.

    Parameters
    ----------
    df : one replicate/bootstrap draw's data (supplied by
        StatCalculator.from_function per replicate weight).
    command : the Stata command to run, with "{weight}" as a placeholder
        for the weight column name if `weight` is given, e.g.
        "regress y x1 x2 [pw={weight}]" or "summarize x [aw={weight}]".
        Stata's weight syntax/placement varies enough by command (pw/aw/
        fw/iw, and where the bracket goes) that this doesn't try to build
        it for you - write it the way you'd type it directly in Stata.
        Pass a list of commands run in order, instead of a single string,
        when you need setup (a `svyset`, say) between `use` and the
        command whose results you actually want back - each element may
        use "{weight}", e.g. `["svyset psu [pw={weight}]", "svy: mean x"]`.
    results : names to pull back, e.g. ["r(mean)", "r(Var)"] or ["e(b)"].
        A scalar becomes one row (`join_on_name`=name); a matrix's columns
        (e.g. e(b)'s term names) each become their own row.
    weight : column name to substitute into "{weight}" in `command`, or ""
        (the StatCalculator.from_function convention - see its
        `weight_argument_name`) if `command` needs no weight or already
        hardcodes one. Default is "".
    join_on_name : name of the term-identifier column in the output.
        Default is "Variable".
    value_name : name of the estimate column in the output. Default is
        "estimate".
    edition, stata_path : forwarded to
        `_stata_interop.require_pystata`.
    reuse_data : if True, skip re-exporting/re-`use`-ing df when it's the
        same object (by identity) as the previous call - this is exactly
        the common case here: StatCalculator.from_function calls this
        delegate once per replicate weight with the *same* df object every
        time (only `weight`, spliced into `command`, changes), so the
        underlying data doesn't need re-exporting on every replicate.
        Default is False (safe/explicit opt-in - see
        `_stata_interop._run_in_stata`'s docstring for why this can't be a
        default the way the R side's dataframe_to_r caching is). Call
        `survey_kit.statistics._stata_interop.clear_stata_cache()` once
        the replicate loop is done, to free Stata's own copy of the data.
    quietly : pass False to let `command` stream Stata's own console
        output on success too (failures always surface the real error
        text regardless - see `_stata_interop._run_in_stata`'s
        docstring). Default is True.

    Returns
    -------
    pl.DataFrame
        One row per named scalar, or per column of a named matrix -
        exactly the shape a StatCalculator.from_function delegate needs
        (no vcov/tidy - see the note above on where the variance actually
        comes from).
    """
    from . import _stata_interop as _st

    if weight:
        if isinstance(command, str):
            full_command = command.format(weight=weight)
        else:
            full_command = [c.format(weight=weight) for c in command]
    else:
        full_command = command

    raw = _st.run_stata_results(
        df,
        full_command,
        results,
        reuse_data=reuse_data,
        edition=edition,
        stata_path=stata_path,
        quietly=quietly,
    )

    import numpy as np

    rows = []
    for name, value in raw.items():
        if isinstance(value, tuple):
            values, row_names, col_names = value
            arr = np.asarray(values)
            if arr.ndim == 1 or 1 in arr.shape:
                for term, v in zip(col_names, arr.reshape(-1)):
                    rows.append({join_on_name: term, value_name: float(v)})
            else:
                for i, rowi in enumerate(row_names):
                    for j, colj in enumerate(col_names):
                        rows.append(
                            {
                                join_on_name: f"{name}[{rowi},{colj}]",
                                value_name: float(arr[i, j]),
                            }
                        )
        else:
            rows.append({join_on_name: name, value_name: float(value)})

    return pl.DataFrame(rows)
