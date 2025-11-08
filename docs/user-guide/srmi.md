# Imputation/SRMI

## What Is It

Suppose you have some variable ($y$) that has missing observations (some people didn't respond to the survey or data was left blank in the administrative record, etc.).  You also have some set of characteristics that are observable, with no missing information ($X$).  The basic idea of imputation is to estimate the distribution of $y$ conditional on $X$, or $f(y|X)$ and then use that estimate to draw plausible values of $y$ for each missing observation so that you can get an estimate of some statistic of $Q(y)$ (the mean, the median, a regression coefficient that conditions on $y$, ...).  The imputation model must be "congenial" to estimate a given statistic (or "proper"), which means the imputes should be drawn such that $Q(\hat{y})$ is unbiased and the estimates have valid confidence intervals (i.e. uncertainty is accounted for correctly - see [van Buren, 2018](https://stefvanbuuren.name/fimd) for a much more accurate and thorough discussion).   

When actually doing the imputation you have several choices:

1. **How will you estimate $f(y|X)$?** - What functional form will you use and how will you estimate the parameters?  A regression, a hot deck, machine learning?  Embedded in this is what do you include in $X$.
1. **How will you draw values of $y$ given your estimate of $f(y|X)$?** - draw from the error distribution from your regression ($\hat{e}$), use the emprirical distribution of observed values with the same/similar expected values ($\hat{y}$) as in predicted mean matching or a hot deck, etc.
1. **How will you account for the relationship between imputed variables in your model?** Suppose you have several variables with missing values, such that you want to estimate $f(y_1|X, y_2)$ and $f(y_2|X,y_1)$, how will you account for the relationship between $y_1$ and $y_2$?  This is what [Sequential Regression Multivariate Imputation (SRMI)](https://www150.statcan.gc.ca/n1/en/pub/12-001-x/2001001/article/5857-eng.pdf?st=TzHpqnV2) is for.  You iteratively run the estimation, where in the first iteration $^{(1)}$, you estimate $f(y_1^{(1)}|X)$ (ignoring the relationship between $y_1$ and $y_2$), then impute $f(y_2^{(1)}|X,y_1^{(1)})$ (not ignoring the relationship between $y_2$ and $y_1$).  Then, in the second iteration $^{(2)}$, you estimate $f(y_1^{(2)}|X,y_2^{(1)})$, with the imputed $y_2$ from the prior iteration to incorporate the relationship between the two $y$ variables.  You then re-impute $y_2^{(2)}$ with the newly imputed values for $y_1^{(2)}$.  You continue with additional iterations (up to $k$, for example) until you have converged to the covariate distributions of $f(y_1^{(k)}|X,y_2^{(k-1)})$=$f(y_1|X,y_2)$ and $f(y_2^{(k)}|X,y_1^{(k)})$=$f(y_2|X,y_1)$.  Many imputations (such as most done at the U.S. Census Bureau) do not do this and stop after the first iteration, where $f(y_1^{(1)}|X)$ was estimated without conditioning on $y_2$ (or other variables with missing information). 
1. **How do account for uncertainty?**  You should use multiple imputation ([Rubin, 1987](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316696)), which means that you take some number of independent draws of $y$ for each missing observation.  This allows you to account for uncertainty in $f(y|X)$.  However, most estimation methods involve estimating $f(y|X,\theta)$, where $\theta$ could be the regression coefficients from an OLS regression.  Since you are estimating $\hat{\theta}$, the estimates have uncertainty as well, which should be incorporated into the imputation.  This package handles this through taking a [Bayes Bootstrap](https://pmc.ncbi.nlm.nih.gov/articles/PMC3703677/) of the sample at each step of the estimation.

The tutorials below show how you can make some of these choices.  For example, by choosing the imputation model type and the method for drawing imputed values, you are choosing $f$.  By specifying the imputation formula (either as an R-style formula or a list of variables names), you are specifying $X$.  By setting the number of iterations (n_iterations), you are choosing to/not to do SRMI.  By setting the number of implicates (n_implicates), you choose whether to do single/multiple imputation.

## Why Use It

This package was built to handle production-scale multiple imputation for complex survey data. It provides capabilities that go beyond typical imputation packages:

* **Speed and Scale** - Built on [polars](https://pola.rs) to handle large datasets efficiently. Impute datasets with hundreds of thousands or millions of observations on a standard laptop in minutes rather than hours.  You can pass data in any form supported by [Narwhals](https://narwhals-dev.github.io/narwhals/) and you'll get your data back in that form (Pandas, Polars, DuckDB, PyArrow, etc.).  We just use polars under the hood for speed.

* **Production-Grade Reliability** - Designed for long-running imputation pipelines with managed state and checkpointing. If your imputation is interrupted (power outage, server maintenance, hitting time limits,  you want to use your laptop to play a game), you can resume exactly where you left off without losing work.

* **Flexible Method Mixing** - Different variables need different approaches. Impute demographic variables via hot deck, income with gradient boosting, and binary indicators with PMM—all in one coherent framework with proper iteration.

* **Complex Dependency Handling** - Real survey data has intricate relationships. For example, you may want to:

    * Impute an earnings flag first, then earnings source conditional on that flag (primary earnings from wage and salary, self-employment, or farm self-employment)
    * Impute by subgroups (wage earners, self-employed, farm self-employed) based on previously imputed variables
    * Impute spouse 1's income using spouse 2's, then vice versa, then recalculate household totals to use to impute other variables (like interest income, pensions, etc.)
    
    This package handles this naturally without requiring you to reshape your data or run totally separate imputation models (which would make it impossible to use SRMI properly).


* **Arbitrary Pre/Post Processing** - Execute custom functions at any point in the imputation sequence. Recalculate derived variables, apply business rules, update relationships—- whatever your data requires. Keep your data in its natural structure (person-level, household-level) and use the flexibility of the package make that complexity easy to manage (it's easy to write a function that updates spousal earnings and just call that when you need it).

* **Modern Methods** - Native support for gradient boosting ([LightGBM](https://lightgbm.readthedocs.io/en/stable/)) with hyperparameter tuning, quantile regression for continuous variables, etc. Methods typically unavailable in other imputation packages.  

Other tools can be integrated into this package, but the default is LightGBM for its speed and accuracy (some other tools include [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_intro.html), [Catboost](https://catboost.ai/docs/en/), [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)).

### Tested at Scale

This package integrates knowledge gained from years of research at the U.S. Census Bureau:

- [Income imputation in the Current Population Survey's Annual Social and Economic Supplement](https://academic.oup.com/jssam/article-abstract/10/1/81/5943180)
- [Analysis of administrative data from the Supplemental Nutrition Assistance Program](https://www.aeaweb.org/articles?id=10.1257/pandp.20221040)  
- [Development of the National Experimental Well-being Statistics Project](https://www.census.gov/data/experimental-data-products/national-experimental-wellbeing-statistics.html)

It was designed to handle imputing hundreds of variables across multiple iterations (SRMI) with samples ranging from hundreds of thousands to hundreds of millions of records, with complex dependency structures between variables.

### When to Use This vs. Other Packages

**Use this package when you:**

- Work with large datasets (>100K rows) where performance matters
- Need production reliability with checkpoint/resume capability
- Want to mix different imputation methods intelligently
- Have complex variable dependencies that require custom logic
- Need modern ML approaches (gradient boosting, quantile regression)
- Require hot deck or statistical matching methods

**Use mice or similar packages when you:**

- Work with smaller datasets (<100K rows)
- Want extensive built-in convergence diagnostics
- Just want something that works and you can cite
- Want a simpler API with simpler defaults


**Note:** This package assumes familiarity with imputation methodology. It provides powerful, flexible tools for implementing complex imputation strategies correctly at scale. If you need a point-and-click solution with extensive guardrails, traditional packages may be more appropriate.

## API

See the full [Imputation/SRMI API documentation](../api/srmi.md) 


## Examples/Tutorials

=== "Hot Deck/Statistical Match"
    Stat match uses a join and hot deck fill forward from an array across the file, but there is no real difference between them theoretically 

    === "Code"
        ```python
        --8<-- "tutorials/srmi/hotdeck.py"
        ```

    === "Log"
        [View in separate window](../../tutorials/srmi/hotdeck.html){:target="_blank"}
        <iframe src="../../tutorials/srmi/hotdeck.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "Regression"
    Logit and/or OLS-based imputation

    === "Code"
        ```python
        --8<-- "tutorials/srmi/regression.py"
        ```

    === "Log"
        [View in separate window](../../tutorials/srmi/regression.html){:target="_blank"}
        <iframe src="../../tutorials/srmi/regression.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "Machine Learning"
    Imputation with LightGBM, see the [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/) for additional information on some of the options.

    === "Code"
        ```python
        --8<-- "tutorials/srmi/gbm.py"
        ```

    === "Log"
        [View in separate window](../../tutorials/srmi/gbm.html){:target="_blank"}
        <iframe src="../../tutorials/srmi/gbm.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>
