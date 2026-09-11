# Categorical Predictors & Group Effects

The [Variable Types & Models](variable-types-and-models.md) page covers variables *being
imputed* that are categorical. This page covers two different, related situations: a
categorical column being used as a *predictor* for something else, and a grouping variable
(like state or household) whose effect you want captured without treating it as an ordinary
predictor.

## Categorical predictors

If a text/category column (like an industry code) is used as a predictor and you don't tell
`simple_model()` about it, the model behind the scenes would either mis-handle it as a plain
number or crash outright. Declare it with `categorical_predictors=[...]` and it's handled
correctly wherever it shows up as a predictor:

- For models with native categorical support (LightGBM, XGBoost, CatBoost), it's passed
  through directly.
- For models without native support (RandomForest, and the categorical-variable default
  models), it's automatically one-hot encoded instead.

You don't need to separately list a variable that's already declared `ordered_categorical` or
`unordered_categorical` in `classes` - that's picked up automatically wherever it's used as a
predictor for something else.

## Group effects

`group_levels="state"` (or a list of column names) lets a model borrow statistical strength
across observations that share a group - people in the same state should be a little more
alike than the overall population, without treating "state" as an ordinary dummy-coded
predictor (which would need one dummy per state, and wouldn't generalize well to states with
few observations). It's automatically excluded from being used as a plain predictor too.

Not every model supports this - only Regression, RandomForest, XGBoost, CatBoost, and
SklearnModel do. LightGBM (the default for continuous/binary variables) doesn't, so if you want
group effects for a continuous variable, override its model - see
[Variable Types & Models](variable-types-and-models.md).

## Walkthrough

=== "Code"
    ```python
    --8<-- "tutorials/srmi/simple_model_categorical_and_group.py"
    ```

=== "Log"
    [View in separate window](../../tutorials/srmi/simple_model_categorical_and_group.html){:target="_blank"}
    <iframe src="../../tutorials/srmi/simple_model_categorical_and_group.html"
        style="width: 100%; height: 800px; border: none;">
    </iframe>
