# Statistics & Standard Errors

## What Is It

The Statistics module provides tools for calculating summary statistics with proper standard errors for complex survey data and multiple imputation. It handles three key sources of uncertainty:

- **Sampling variance** - From complex survey designs (stratification, clustering, unequal weights) with replicate weights (for examples, see [CPS ASEC documentation](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/2025_ASEC_Replicate_Weight_Usage_Instructions.docx), [ACS documentation](https://www.census.gov/programs-surveys/acs/data/variance-tables.html), [Consumer Expenditure Survey documentation](https://www.bls.gov/cex/pumd-getting-started-guide.htm), [SCF documentation](https://www.federalreserve.gov/econres/scfindex.htm), [MEPS documentation](https://www.federalreserve.gov/econres/scfindex.htm), ...)
- **Imputation uncertainty** - When working with multiply imputed data
- **Both combined** - Full uncertainty accounting when you have imputed data with replicate weights

The package supports:
- Quick exploratory summaries with `summary()`
- Replicate weight standard errors (Bootstrap and Balanced Repeated Replication)
- Multiple imputation inference using Rubin's rules
- Statistical comparisons with proper standard errors
- Custom analysis functions with automatic variance estimation

## Why Use It

Standard statistical software often gets uncertainty wrong:

**The problem:**
- Using design weights without replicate weights → Standard errors too small
- Simple variance formulas with complex samples → Wrong confidence intervals
- Ignoring imputation uncertainty → Overstated precision
- Ad-hoc comparisons → Invalid hypothesis tests.  

For example, replicate weights are intended to account for correlation over space and time in sample selection, so using them for comparisons will account for correlation in estimates across the different replicate factors.  To give a concrete example, addresses in the CPS are in sample for consecutive months, so if you want to get the correct confidence intervals on the change in unemployment, you need to use the replicate weights in the month-to-month comparison.  The standard errors will be much narrower than if you compared the unemployment rates as if they were two independent samples.


**This package:**
- Implements proper variance estimation for complex surveys (as used by Census Bureau, BLS, etc.)
- Correctly combines uncertainty from multiple imputations
- Makes it easy to do the right thing (comparisons, custom functions, etc.)
- Fast enough for production use with millions of observations (assuming you use polars)

## Key Features

- **DataFrame agnostic** - Works with Polars, Pandas, Arrow, DuckDB via Narwhals
- **Fast** - Optimized for large datasets (100K+ rows, tested on millions)
- **Flexible** - Custom functions get variance estimation for free
- **Parallel processing** - Run imputations in parallel
- **Proper inference** - Implements Rubin's rules for MI, supports BRR/bootstrap for replicate weights
- **Production-ready** - Used in Census Bureau's National Experimental Wellbeing Statistics

## Implementation Notes

### Replicate Weights

Two variance estimation approaches:
- **Bootstrap** (`bootstrap=True`) - Standard bootstrap resampling variance
- **BRR** (`bootstrap=False`) - Balanced Repeated Replication as used by Census Bureau

### Multiple Imputation

Implements Rubin's (1987) combining rules:
- **Combined estimate** = average across implicates
- **Total variance** = within-imputation variance + between-imputation variance
- **Degrees of freedom** account for finite number of implicates
- **Missing information rate** quantifies uncertainty from imputation

### When to Use What

| Use Case | Tool | Why |
|----------|------|-----|
| Quick exploration | `summary()` | Fast, no SEs needed |
| Survey data with replicate weights | `StatCalculator` + `Replicates` | Proper complex survey variance |
| Imputed data | `mi_ses_from_function()` | Accounts for imputation uncertainty |
| Imputed survey data | Both combined | Full uncertainty (sampling + imputation) |
| Custom analysis | `StatCalculator.from_function()` | Any function gets proper SEs |


## API

See the [Basic Statistics and Standard Errors documentation](../api/basic_standard_errors.md) estimating summary stats and standard errors.
See the [Multiple Imputation documentation](../api/multiple_imputation.md) estimating statistics with multiply imputed data.

## Example/Tutorial

How to get standard errors (SEs) using replicate weights or the bootstrap + multiple imputation (MI).

=== "Quick Summaries"
    The `summary()` function provides easy data summaries (with more control than the default `describe()` methods in pandas or polars)

    === "Code"
        ```python
        --8<-- "tutorials/statistics/basic.py"
        ```

    === "Log"
        [View in separate window](../../tutorials/statistics/basic.html){:target="_blank"}
        <iframe src="../../tutorials/statistics/basic.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "Bootstrap/Replicate SEs"
    For proper variance estimation with complex survey designs or using bootstrap weights

    === "Code (basic stats)"
        ```python
        --8<-- "tutorials/statistics/standard_errors.py"
        ```

    === "Log (basic stats)"
        [View in separate window](../../tutorials/statistics/standard_errors.html){:target="_blank"}
        <iframe src="../../tutorials/statistics/standard_errors.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>

    === "Code (any custom stat)"
        This works with **any function** that takes a dataframe and weight and returns estimates. You get proper variance estimation automatically.

        ```python
        --8<-- "tutorials/statistics/arbitrary_bootstrap.py"
        ```

    === "Log (any custom stat)"
        This works with **any function** that takes a dataframe and weight and returns estimates. You get proper variance estimation automatically.

        [View in separate window](../../tutorials/statistics/arbitrary_bootstrap.html){:target="_blank"}
        <iframe src="../../tutorials/statistics/arbitrary_bootstrap.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "MI SEs"
    Combine estimates properly across multiply imputed datasets using Rubin's rule.  This incorporates the bootstrap/replicate factor code internally.


    === "Code (basic stats)"
        ```python
        --8<-- "tutorials/statistics/multiple_imputation.py"
        ```

    === "Log (basic stats)"
        [View in separate window](../../tutorials/statistics/multiple_imputation.html){:target="_blank"}
        <iframe src="../../tutorials/statistics/multiple_imputation.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>

    === "Code for (any custom stat)"
        This works with **any function** that takes a dataframe and weight and returns estimates. You get proper variance estimation (replicate weights + MI) automatically.

        ```python
        --8<-- "tutorials/statistics/arbitrary_srmi.py"
        ```

    === "Log (any custom stat)"
        This works with **any function** that takes a dataframe and weight and returns estimates. You get proper variance estimation (replicate weights + MI) automatically.
        
        [View in separate window](../../tutorials/statistics/arbitrary_srmi.html){:target="_blank"}
        <iframe src="../tutorials/statistics/arbitrary_srmi.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>



=== "Using Real Data"
    Here are tutorials for replicate-weight standard errors for household income from the public use Current Population Annual Social and Economic Supplement (CPS ASEC)  and for multiple imputation + replicate-weight standard errors from the Consumer Expenditure Survey (CEX).

    === "Code (CPS ASEC)"
        ```python
        --8<-- "tutorials/survey_data/cps_asec.py"
        ```

    === "Log (CPS ASEC)"
        [View in separate window](../../tutorials/survey_data/cps_asec.html){:target="_blank"}
        <iframe src="../../tutorials/survey_data/cps_asec.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>
 

    === "Code (CEX)"
        ```python
        --8<-- "tutorials/survey_data/cex.py"
        ```

    === "Log (CEX)"
        [View in separate window](../../tutorials/survey_data/cex.html){:target="_blank"}
        <iframe src="../../tutorials/survey_data/cex.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>
