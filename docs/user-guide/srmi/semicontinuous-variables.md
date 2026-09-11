# Semicontinuous Variables

Some variables are really two questions in one: "does this person have any self-employment
income at all?" and, only for the people who do, "how much?" Most people have exactly $0 (they
don't have this income source), and everyone else has some genuinely continuous positive
amount. Fitting one plain continuous model to a variable like this fights itself - it has to
somehow explain both the large point mass at zero and the spread of real dollar amounts with
the same model.

The standard fix (a "two-part" or "hurdle" model) is to split it into two separate imputations:
a yes/no flag, and the dollar amount, fit only among the people who said yes. `simple_model()`
does this for you with `yn_pairs={value_variable: yn_variable}`.

If the yes/no variable already exists in your data (as in the example below) and has no
missing values of its own, only the dollar-amount variable actually needs imputing.  If it has
its own missing values (or doesn't exist yet at all), `simple_model()` builds and imputes it
too, using the same predictor list.

## Walkthrough

=== "Code"
    ```python
    --8<-- "tutorials/srmi/simple_model_semicontinuous.py"
    ```

=== "Log"
    [View in separate window](../../tutorials/srmi/simple_model_semicontinuous.html){:target="_blank"}
    <iframe src="../../tutorials/srmi/simple_model_semicontinuous.html"
        style="width: 100%; height: 800px; border: none;">
    </iframe>
