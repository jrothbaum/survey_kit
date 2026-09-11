# Controlling Predictors

By default, `simple_model()` uses every other column as a predictor for each variable it
imputes - a brute-force approach that lets the model's own regularization figure out what
actually matters. Sometimes you need to override that, most often to avoid circularity: a
column that's really just a restatement of, or downstream of, the thing you're imputing.

For example, "hours worked last week" only exists because someone is employed - using it (or
anything derived from it) as a predictor for an employment flag would let the model just read
the answer off a column that's really asking the same question.

## Excluding predictors

- `exclude={variable_name: [...]}` - keep specific columns out of just *one* variable's model.
- `exclude_global=[...]` - keep specific columns out of *every* variable's model. Useful for
  things like a raw survey weight or a free-text field that should never be a real predictor
  for anything.

## Choosing exactly which variables to impute

By default, `simple_model()` scans every column (other than the id column) for missingness and
imputes anything it finds. Pass `variables_to_impute=[...]` to impute an exact list instead -
nothing else gets scanned, even if it has missing values of its own. This is handy when you
only want to touch one or two columns in a much wider dataframe.

## Walkthrough

=== "Code"
    ```python
    --8<-- "tutorials/srmi/simple_model_controlling_predictors.py"
    ```

=== "Log"
    [View in separate window](../../tutorials/srmi/simple_model_controlling_predictors.html){:target="_blank"}
    <iframe src="../../tutorials/srmi/simple_model_controlling_predictors.html"
        style="width: 100%; height: 800px; border: none;">
    </iframe>
