# Variable Types & Models

`simple_model()` can only guess so much on its own. It can tell binary and continuous columns
apart automatically (a column with only two values, or a boolean column, is binary; everything
else defaults to continuous). It can **never** guess that a column is really a category -
whether the categories have a natural order (like an education level: less than high school,
high school, some college, bachelor's or more) or not (like a region) can't be inferred from
the data alone, so those always need to be told explicitly.

## Declaring categorical variables

Use `classes={variable_name: Variable.Class...}` to declare a variable's type:

- `Variable.Class.ordered_categorical` - needs `ordered_categories={variable_name: [...]}`
  too, giving the category order from lowest to highest. There's no way around this - the data
  itself can't tell you whether "some college" ranks above or below "high school grad".
- `Variable.Class.unordered_categorical` - just needs the declaration, no order to give.
- `Variable.Class.binary` / `Variable.Class.continuous` - only needed if you want to override
  what auto-detection would have picked.

## Choosing the model

Every variable needs a model to actually do the imputing. The built-in defaults are:

| Class | Default model |
|---|---|
| binary | LightGBM |
| continuous | LightGBM |
| ordered_categorical | RandomForest (predicts a rank, donates the real category) |
| unordered_categorical | RandomForest (classifies, donates by matching) |

Override with `model={Variable.Class...: ...}`. Two ways to do it:

- Pass just a `ModelType` (e.g. `Variable.ModelType.RandomForest`) - uses a sensible built-in
  default configuration for that model.
- Pass `(ModelType, parameters)` - full control, using exactly the `Parameters` you give it.

## Walkthrough

=== "Code"
    ```python
    --8<-- "tutorials/srmi/simple_model_variable_types.py"
    ```

=== "Log"
    [View in separate window](../../tutorials/srmi/simple_model_variable_types.html){:target="_blank"}
    <iframe src="../../tutorials/srmi/simple_model_variable_types.html"
        style="width: 100%; height: 800px; border: none;">
    </iframe>
