# Getting Started with `simple_model()`

`SRMI.simple_model()` is the fastest way to go from a raw dataframe to a working multiple
imputation. Give it a dataframe and an id column, and it does the rest:

1. Looks at every other column and finds the ones with missing values.
2. For each one, decides whether it's binary (only two values, like a yes/no flag) or
   continuous (any other number).
3. Picks a model for each - LightGBM by default, which works well for both.
4. Builds the [Variable](../../api/srmi.md) objects and hands them to `SRMI`, ready to run.

Nothing here is hidden or magic - it's building exactly the same `Variable`/`SRMI` objects
you'd build by hand (see [Advanced/Manual Construction](advanced-manual-construction.md)), just
picking sensible defaults for you. You can always inspect what it decided before running
anything, and override any part of it - later pages in this section show how.

## Walkthrough

=== "Code"
    ```python
    --8<-- "tutorials/srmi/simple_model_getting_started.py"
    ```

=== "Log"
    [View in separate window](../../tutorials/srmi/simple_model_getting_started.html){:target="_blank"}
    <iframe src="../../../tutorials/srmi/simple_model_getting_started.html"
        style="width: 100%; height: 800px; border: none;">
    </iframe>

## What to do next

Once you've run something like the above, the natural next questions are usually:

- "One of my columns is really a category, not just binary/continuous" → see
  [Variable Types & Models](variable-types-and-models.md)
- "One of my predictors is a category, or I want to borrow strength across a grouping
  variable" → see [Categorical Predictors & Group Effects](categorical-and-group-effects.md)
- "One of my variables is mostly zero, with a real amount for everyone else" → see
  [Semicontinuous Variables](semicontinuous-variables.md)
- "I need to keep a variable out of a model" → see
  [Controlling Predictors](controlling-predictors.md)
- "Did this actually converge, and do the imputed values look right?" → see
  [Convergence Diagnostics](diagnostics-convergence.md) and
  [Imputation Quality & Propensity Diagnostics](diagnostics-quality-propensity.md)
