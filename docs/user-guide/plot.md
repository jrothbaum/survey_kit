# Plotting

## What Is It

`survey_kit.plot` builds interactive [plotly](https://plotly.com/python/) figures directly from a [`StatCalculator`](../api/basic_standard_errors.md)'s or [`MultipleImputation`](../api/multiple_imputation.md)'s own estimates - no manual reshaping. [`combine`](../api/plot.md#survey_kit.plot.combine) puts several figures on one page, switched between with dropdowns.

## Key Features

- **No manual reshaping** - point a plot function at a `StatCalculator`/`MultipleImputation`/`AdapterStats` and it finds the right columns itself.
- **DRB rounding by default** - values are rounded per Census disclosure-review rules before plotting.
- **A group dropdown, not just a legend** - `line()`/`quantiles()`/`coefplot()` can group series into a compact dropdown (`group_by`) instead of a long legend.
- **Confidence intervals built in** - `ci_level=` draws error bars or, with `ci_area=True`, a shaded band.
- **`combine()`** - nests any number of dropdown levels, remembering each level's last choice. Label each level (`label=["Run:", "CI:"]`), or put a level on its own row with a `"\n"` in its label.
- **Shared legend state** - isolating a series by name on one figure carries over to any other figure with a same-named series, once it's shown.
- **Still a plain plotly figure** - every function takes a `layout: dict` applied via `fig.update_layout(**layout)`, and returns something you can call any plotly method on.

## When to Use What

| Use Case | Tool |
|----------|------|
| Any set of columns plotted against each other | `line()` |
| A stat item's own quantile columns, percentile 0-100 | `quantiles()` |
| One row per category with a CI whisker (disclosure-review style) | `coefplot()` |
| Several stat items that should sum to a shown total | `stacked_bar()` |
| Switching between whole figures (runs, vintages, scenarios) | `combine()` |

## API

See the [Plotting API reference](../api/plot.md) for the full parameter list of every function.

## Example/Tutorial

=== "line() / quantiles()"
    `quantiles()` for a stat item's own quantile columns, `line()` for any other set of columns. Covers confidence intervals and the group dropdown (`group_by`).

    **Figures:** [quantiles()](../tutorials/plot/figures/quantiles.html){:target="_blank"} ·
    [with CI](../tutorials/plot/figures/quantiles_ci_area.html){:target="_blank"} ·
    [grouped](../tutorials/plot/figures/quantiles_grouped.html){:target="_blank"} ·
    [line()](../tutorials/plot/figures/line_selected_quantiles.html){:target="_blank"}

    ```python
    --8<-- "tutorials/plot/line_and_quantiles.py"
    ```

=== "coefplot()"
    A disclosure-review-style point-and-whisker chart, with and without `series=` for multiple offset points per row.

    **Figures:** [with series](../tutorials/plot/figures/coefplot.html){:target="_blank"} ·
    [single series](../tutorials/plot/figures/coefplot_single_series.html){:target="_blank"}

    ```python
    --8<-- "tutorials/plot/coefplot.py"
    ```

=== "stacked_bar()"
    Layers/categories can mix positive and negative values - each bar stacks positive layers right of zero and negative ones left, with the total label landing on whichever side its net value falls on.

    **Figures:** [all-negative](../tutorials/plot/figures/stacked_bar.html){:target="_blank"} ·
    [mixed sign](../tutorials/plot/figures/stacked_bar_mixed_sign.html){:target="_blank"}

    ```python
    --8<-- "tutorials/plot/stacked_bar.py"
    ```

=== "combine()"
    Two independent runs, each with and without a confidence band. Covers a 2-level tree (Run -> CI), a 3-level one (Run -> CI -> Year), labeling each level, and putting a level on its own row.

    **Figures:** [2 levels](../tutorials/plot/figures/combine.html){:target="_blank"} ·
    [3 levels](../tutorials/plot/figures/combine_3_levels.html){:target="_blank"}

    ```python
    --8<-- "tutorials/plot/combine.py"
    ```

=== "combine() + shared legend state"
    `coefplot()`/`stacked_bar()` have no group dropdown of their own, but their legend clicks are still shared by trace name - isolating a series/layer in one branch and switching to a sibling with the same name shows it isolated there too.

    **Figures:** [coefplot()](../tutorials/plot/figures/combine_two_coefplots.html){:target="_blank"} ·
    [stacked_bar()](../tutorials/plot/figures/combine_two_stacked_bars.html){:target="_blank"}

    ```python
    --8<-- "tutorials/plot/combine_two_coefplots.py"
    ```

    ```python
    --8<-- "tutorials/plot/combine_two_stacked_bars.py"
    ```
