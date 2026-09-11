# Imputation Quality & Propensity Diagnostics

Convergence diagnostics ask "did the chain settle down." These ask a different question:
do the imputed values themselves look plausible next to the values that were actually observed?

## Imputation quality

`SRMI.plot_imputation_quality()` compares observed and imputed values directly, one group per
implicate plus a pooled "Observed" group. Three views are available (`kind="density"`,
`"strip"`, or `"box"`) - density is the default, showing the full shape of each group's
distribution.

This marginal comparison has a real limitation worth knowing about: under a "missing at
random" assumption, the missing rows can legitimately have a *different* marginal distribution
than the observed rows - that's the entire point of a model that conditions on other variables,
rather than just filling in the overall mean. A correct imputation can look "off" by this
measure, and an incorrect one can look fine by luck.

## Response propensity

`SRMI.plot_propensity()` checks the same question conditional on each row's predicted
*response propensity* - how similar a row's other characteristics are to a typically-missing
row. Rows are compared within bins of similar propensity rather than all together, which is a
fairer test of whether the imputation is behaving the way the data actually supports.

## Example

<iframe src="../../tutorials/srmi/diagnostics/quality_density.html"
    style="width: 100%; height: 650px; border: none;">
</iframe>

<iframe src="../../tutorials/srmi/diagnostics/propensity_density.html"
    style="width: 100%; height: 650px; border: none;">
</iframe>

## Walkthrough

=== "Code"
    ```python
    --8<-- "tutorials/srmi/diagnostics_quality_propensity.py"
    ```

=== "Log"
    [View in separate window](../../tutorials/srmi/diagnostics_quality_propensity.html){:target="_blank"}
    <iframe src="../../tutorials/srmi/diagnostics_quality_propensity.html"
        style="width: 100%; height: 800px; border: none;">
    </iframe>
