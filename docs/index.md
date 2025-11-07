# Survey Kit

Tools for addressing missing data problems (nonresponse bias and item missingness) including extremely fast calibration weighting and machine learning-based imputation.

## What Is It

Survey Kit is a Python package designed to help researchers and data scientists tackle common challenges in survey research, primarily:

1. **Nonresponse bias** - When your sample doesn't represent the population
2. **Item missingness** - When some records have incomplete data
3. **Proper measures of uncertainty** - with the various strategies surveys use to address missing data, it can be a challenge to estimate proper measures of uncertainty (like standard errors for regression coefficients).  This package tries to cover your bases by having very flexible tools to handle multiply imputed data and replicate weights.

**Note: these are not necessarily only survey issues**

Administrative data have these challenges, too.  Nonresponse bias can be caused by adminstrative rules or selective compliance.  For example, many low income individuals are not required to file taxes.  Some workers may be paid "under the table".  Likewise, reported information in administrative data may be incomplete.  I've worked with job-level administrative data with missing gross earnings for millions of jobs. 


## Why Use It

- **Nonresponse bias** through calibration weighting.  Calibration weighting is the standard practice of weighting the sample to match some set of known or estimated totals (like race x age x gender cells or race x income cells)
- **Item missingness** use Sequential Regression Multivariate Imputation (SRMI) to draw *plausible* values for missing data (NOT the mean or mode, but from the estimated distribution of values conditional on some set of observable characteristics in the data).  Tools to implement widely-used (but problematic) imputation strategies (hot deck), more advanced approaches (regression-based estimatation and predicted mean matching), and machine-learning and quantile-statistic based approaches (quantile-based machine learning with LightGBM).
- **Summary stats and proper standard errors** there are some (hopefully) useful tools for exploring the data.  Also, tools to work with bootsrapped and replicate-weight based standard errors (as done for ***many*** data sets, see [CPS ASEC documentation](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/2025_ASEC_Replicate_Weight_Usage_Instructions.docx), [ACS documentation](https://www.census.gov/programs-surveys/acs/data/variance-tables.html), [Consumer Expenditure Survey documentation](https://www.bls.gov/cex/pumd-getting-started-guide.htm), [SCF documentation](https://www.federalreserve.gov/econres/scfindex.htm), [MEPS documentation](https://www.federalreserve.gov/econres/scfindex.htm), ...)) and multiple imputation.  With these tools, you can estimate statistics once (costly over all replicates) and then get SEs for any arbitrary set of comparisons quickly.

## Key Features

- **DataFrame agnostic** - Works with Polars, Pandas, Arrow, DuckDB, etc. via Narwhals and that's what you'll get back (sometimes under the hood data will be converted to polars for calculations)
- **Fast** - Optimized for large datasets - it has been used for weighting and imputation of data with 10s to 100s of millions of rows and thousands of columns.
- **Flexible** - Allows the use of R-style formulas via [formulaic](https://github.com/matthewwardrop/formulaic) in both calibration and imputation (to reduce the amount of work you need to spend manually creating interactions and recoding data), machine-learning imputation via [LightGBM](https://lightgbm.readthedocs.io/en/stable/) (which again makes model specification less time consuming for imputation - you don't have to make so many explicit choices about whether to code variable as a log or exponential, which interactions to include, etc.).
- **Serializable** - Save and load calibration, imputation, and calculated statistics so you can store results from long-running estimation processes or for use later in calculations or regressions.  That way, if you want to estimate something like earnings for men and women, then compare them over time, you run the time-consuming estimates once and save them.  Then you can calculate the standard errors for comparisons instantly without having to go back to the original data and recalculating things for each replicate weight and imputation.   
- **Parallel processing** - Run imputations and calculate statistics in parallel



**Compared to other tools:**

- Fastest calibration tool that I'm aware of
- More flexible than scikit-learn's imputation (and others) and properly handles imputation uncertainty (Bayesian Bootstrap within imputation process, multiple imputation, etc.)
- Works across dataframe backends (Polars, Pandas, Arrow)
- Designed specifically for survey data workflows

**Built for:**

- Survey researchers and data scientists
- Government statistical agencies
- Market research firms
- Academic researchers

## Getting Started

### Installation
```bash
pip install survey-kit
```
or better yet,

```bash
uv add survey-kit
```

### User Guides

Want to learn more, go to the guides for each tool in the package.  Each guide 

- [Calibration Guide](user-guide/calibration.md) - nonresponse bias
- [Imputation Guide](user-guide/srmi.md) - missing data
- [Statistics Guide](user-guide/statistics.md) - proper standard errors

## Repository

View the source code on [GitHub](https://github.com/jrothbaum/survey_kit)

## Support

Report issues or ask questions on [GitHub Issues](https://github.com/jrothbaum/survey_kit/issues)