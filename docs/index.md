# Survey Kit

Tools for addressing missing data problems (nonresponse bias and item missingness) including extremely fast calibration weighting and machine learning-based imputation.

## Overview

Survey Kit is a Python package designed to help researchers and data scientists tackle common challenges in survey research, primarily:

1. **Nonresponse bias** - When your sample doesn't represent the population
2. **Item missingness** - When respondents skip questions

***Note, this is not only a survey issue***

Administrative data (adrecs) have these challenges, too.  Nonresponse bias can be caused by adminstrative rules on required compliance or selective compliance.  For example, many low income individuals are not required to file taxes.  Some workers may be paid "under the table".  Likewise, reported information in administrative data may be incomplete.


- **Nonresponse bias** through calibration weighting.  Calibration weighting is the standard practice of weighting the sample to match some set of known or estimated totals (like race x age x gender cells or race x income cells)
- **Item missingness** use Sequential Regression Multivariate Imputation (SRMI) to draw *plausible* values for missing data (NOT the mean or mode, but from the estimated distribution of values conditional on some set of observable characteristics in the data).  Tools for basic imputation (hot deck), more standard approaches (regression-based estimatation and predicted mean matching), and better approaches (quantile-based machine learning with LightGBM).
- **Survey data manipulation** some useful tools for exploring the data.  More usefully, tools to work with bootsrapped and replicate-weight based standard errors (as done by the U.S. Census Bureau) and multiple imputation.  With these tools, you can estimate statistics once (costly over all replicates) and then get SEs for any arbitrary set of comparisons quickly.

The package provides three core components:

### Calibration Weighting
Extremely fast implementation of calibration weighting (akin to raking) to adjust survey samples to match known population totals (from census data, administrative records, etc.). 

This helps correct for nonresponse bias and which can make the sample more representative.  Similar to packages like [Entropy Balancing](https://ngreifer.github.io/WeightIt/reference/method_ebal.html).  It's been a while since I've benchmarked other tools in this space, but the algorithm in Carl Sanders's [Accelerated Entropy Balancing package](https://github.com/uscensusbureau/entropy-balance-weighting) has been the fastest and most robust (as in it converges or can be used to get a "best possible"-ish set of weights) I've found. 

### Imputation using Sequential Regression Multivariate Imputation (SRMI)
Any imputation model assigns plausible values of $y$ given some set of observable characteristics $X$, through the estimation and drawing from the distribution $f(y|X)$.  How you define and estimate $f$ and take draws varies across methods, but the basic idea is the same.  This package implements hot deck imputation (and its near-twin statistical matching), corresponding the methods used by the U.S. Census Bureau and other statistical agencies.  It also implements a regression-based imputation (where $f(y|X)$ is estimated through OLS regression) with values drawn either based on the regression results (y hat and e hat) or using predicted mean matching.  

However, the method I have found to be the best uses [LightGBM](https://github.com/microsoft/LightGBM) to estimate $f(y|X)$ either as in a regression or more akin to a quantile regression.  With the machine-learning estimate of $f(y|X)$, you can draw values 1. using predicted mean matching (which I'd recommend for small-ish samples in the thousands) or 2. by imputing a rank in the distribution drawn from the estimated $f(y|X)$ quantile regressions and a separately estimated marginal distribution (also estimated from $f(y|X)$, but not from the quantile regressions which can be badly biased in the tails).  

The machine-learning approach can be much better than other approaches because it allows for 1. flexibly estimating $f(y|X)$ with fewer assumptions about what should be included in $X$ (the variables themselves, which should be interacted, etc.), 2. it is much easier to estimate unconditional quantiles (i.e. rather than just estimating the expected earnings, we can quickly estimate the 90th percentile of $y|X) so that we can incorporate that information in our imputation.  For example, suppose college graduates earn a% more than high school graduates, but the 90th percentile of college graduates earns b% more and the 10th percentile earns c% more.  We can incorporate estimates of a, b, and c in the imputation through this approach.  

This is not a trivial issue, many imputations fail in their intended use because of practical limitations on what can be included in $X$ (see [this paper](https://academic.oup.com/jssam/article-abstract/10/1/81/5943180) for an example).

### Estimation of summary stats with standard errors (Multiple Imputation and Bootstrapped)  
The package has useful tools for exploring the data and estimating basic summary stats, and maybe more usefully, tools to work with bootsrapped and replicate-weight based standard errors (as done by the U.S. Census Bureau) and multiple imputation.  With these tools, you can estimate statistics once (costly over all replicates) and then get SEs for any arbitrary set of comparisons quickly.


## Key Features

- **DataFrame agnostic** - Works with Polars, Pandas, Arrow, DuckDB, etc. via Narwhals
- **Fast** - Optimized for large datasets (100K+ rows)
- **Flexible formulas** - R-style formulas via Formulaic
- **Serializable** - Save and load calibration/imputation so you can store results from long-running estimation processes for use later
- **Parallel processing** - Run imputations in parallel


## Quick Example
## Quick Example: Calibration

```python
from survey_kit.calibration import Calibration, Moment
from survey_kit.utilities.formula_builder import FormulaBuilder
import polars as pl

# Your survey data (doesn't match population)
df = pl.read_csv('survey.csv')

# Define the moments (statistics) to match from target population
f = FormulaBuilder(df=df, constant=False)
f.continuous(columns=["age", "income"])

m = Moment(
    df=target_population_data,
    formula=f.formula,
    weight="population_weight",
    index="id"
)

# Calibrate your survey to match the population
c = Calibration(
    df=df,
    moments=m,
    weight="survey_weight"
)

# Run calibration
diagnostics = c.run(min_obs=10, bounds=(0.1, 10.0))

# Get calibrated weights
calibrated_weights = c.df['___final_weight']
```


## Quick Example: Imputation

```python
from survey_kit.imputation.srmi import SRMI
from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters

# Define variables to impute
variables = [
    Variable(
        impute_var="income",
        modeltype=Variable.ModelType.LightGBM,
        model=["age", "education", "occupation"],
        parameters=Parameters.LightGBM(tune=True)
    ),
    Variable(
        impute_var="satisfaction",
        modeltype=Variable.ModelType.pmm,
        model="~ age + income + education"
    )
]

# Run Sequential Regression Multiple Imputation
srmi = SRMI(
    df=df,
    variables=variables,
    n_implicates=5,
    n_iterations=10,
    path_model="output/srmi"
)

srmi.run()

# Get imputed datasets
imputed_dfs = srmi.df_implicates
```

## Quick Example: Statistics

```python
from survey_kit.statistics.calculator import StatCalculator
from survey_kit.statistics.statistics import Statistics
from survey_kit.statistics.replicates import Replicates

# Define replicate weights for variance estimation
replicates = Replicates(weight_stub="repwt_", n_replicates=80)

# Calculate statistics with proper variance estimation
sc = StatCalculator(
    df=df,
    statistics=Statistics(
        stats=["mean", "median"],
        columns=["income", "satisfaction"]
    ),
    weight="final_weight",
    replicates=replicates,
    by=dict(region=["region"], year=["year"])
)
```

## Installation

```bash
pip install survey-kit
```
or better yet,

```bash
uv add survey-kit
```

## Why Survey Kit?

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

Ready to dive in? Check out:

- [Installation Guide](getting-started/installation.md) - Get set up
- [Calibration Guide](user-guide/calibration.md) - Fix nonresponse bias
- [Imputation Guide](user-guide/imputation.md) - Handle missing data
- [Statistics Guide](user-guide/statistics.md) - Calculate proper standard errors
- [Examples](examples/complete-workflow.md) - See it all together

## Repository

View the source code on [GitHub](https://github.com/jrothbaum/survey_kit)

## Support

- Report issues on [GitHub Issues](https://github.com/jrothbaum/survey_kit/issues)
- Questions? Open a discussion
