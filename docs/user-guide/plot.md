# Plotting

## What Is It

`survey_kit.plot` builds interactive [plotly](https://plotly.com/python/) figures directly from a [`StatCalculator`](../api/basic_standard_errors.md)'s or [`MultipleImputation`](../api/multiple_imputation.md)'s own `df_estimates`/CI columns - there's no separate reshaping step to do yourself, and it works the same way whether the object came from raw microdata, replicate weights, or a [regression adapter](adapters.md) (any `AdapterStats` is a `StatCalculator`).

Four core plot functions - [`line`](../api/plot.md#survey_kit.plot.line), [`quantiles`](../api/plot.md#survey_kit.plot.quantiles), [`coefplot`](../api/plot.md#survey_kit.plot.coefplot), [`stacked_bar`](../api/plot.md#survey_kit.plot.stacked_bar) - each return a plain `plotly.graph_objects.Figure`, so anything that works on a plotly figure (`fig.update_layout(...)`, `fig.write_html(...)`, embedding in a notebook or a web page) works here too. [`combine`](../api/plot.md#survey_kit.plot.combine) puts several such figures into a single page, switched between with nested dropdowns.

## Why Use It

- **No manual reshaping** - point a plot function at a `StatCalculator`/`MultipleImputation`/`AdapterStats` and it finds the right columns itself (quantile columns for `quantiles()`, a `column=` for `coefplot()`/`stacked_bar()`, any set of columns for `line()`).
- **DRB rounding by default** - values are rounded per Census disclosure-review rules before plotting (`round_output=True`), so a figure never shows an unrounded number a raw table wouldn't.
- **A group dropdown, not just a legend** - every `line()`/`quantiles()`/`coefplot()` figure gets a compact dropdown for switching which group of series is shown, with per-series state (on/off, isolated) preserved across group switches - see `group_by` below and [`add_group_dropdown`](../api/plot.md#survey_kit.plot.add_group_dropdown).
- **Confidence intervals built in** - `ci_level=` on `line()`/`quantiles()`/`coefplot()` draws error bars or (`ci_area=True`) a shaded band, computed from the stat item's own CI machinery.
- **Arbitrary-depth combining** - `combine()` nests any number of dropdown levels (not just two), and remembers each level's last choice independently when you switch a shallower one.
- **Any plotly kwarg still works** - every plot function takes a `layout: dict` applied last via `fig.update_layout(**layout)`, so custom formatting doesn't require dropping down to plotly yourself.

## Key Features

- **`line()`** - the general building block: one line per series across any set of `df_estimates` columns (e.g. one column per year, or per quantile). Numeric-looking column names (`"2016"`, `"q10"`) are placed on a real numeric x-axis automatically.
- **`quantiles()`** - `line()` specialized to a stat item's own quantile-stat columns (`q10`, `q25`, ... or `median`), plotted across percentile 0-100.
- **`coefplot()`** - the classic disclosure-review chart: one row per category, a point-and-whisker per row (and, with `series=`, several offset points per row for comparing e.g. multiple years), with optional `headers=` section dividers.
- **`stacked_bar()`** - several `StatCalculator`/`MultipleImputation` objects that share a category axis, stacked to show how they add up to a `total_key` total shown as a text label.
- **`combine()`** - nests whole figures under as many dropdown levels as the dict you pass it has, preserving each leaf figure's own internal group dropdown. Label each level (`label=["Run:", "CI:"]`), put a level on its own row of the control bar (a `"\n"` in that level's label), and a level with only one option (e.g. a branch that's a bare figure, not a dict) shows no dropdown at all.
- **`group_by`** - splits each series label on a separator so related series (e.g. two historical years vs. a recent one) land in the same dropdown entry instead of getting one each.
- **Shared legend state across figures** - isolating/toggling a series by name on one figure's legend carries over to any sibling figure with a same-named series once `combine()` switches to it (or once it's shown at all) - works the same way for `line()`/`quantiles()`'s group dropdown and for `coefplot()`/`stacked_bar()`'s plain legend, even though only the former has a dropdown of its own.

## When to Use What

| Use Case | Tool | Why |
|----------|------|-----|
| Any set of `df_estimates` columns plotted against each other (e.g. by year) | `line()` | The general building block every other plot function is built from |
| A stat item's own quantile columns across percentile 0-100 | `quantiles()` | Finds `q10`/`q25`/... itself, no column list to write out |
| One row per category with a CI whisker (disclosure-review style) | `coefplot()` | Built for exactly this chart shape, including multi-series offsets and section headers |
| Several stat items that should sum to a shown total | `stacked_bar()` | Stacks layers and checks/labels the total for you |
| Comparing runs/vintages, or any other switchable set of whole figures | `combine()` | Nested dropdowns, arbitrarily deep, with state preserved across switches |

## API

See the [Plotting API reference](../api/plot.md) for the full parameter list of every function.

## Example/Tutorial

=== "line() / quantiles()"
    The two column-plotting functions - `quantiles()` for a stat item's own quantile columns, `line()` for any other set of columns (including plain numeric-named ones like a year). Covers confidence intervals (error bars or a shaded band) and the group dropdown (`group_by`).

    === "Code"
        ```python
        --8<-- "tutorials/plot/line_and_quantiles.py"
        ```

    === "Log: quantiles()"
        [View in separate window](../tutorials/plot/figures/quantiles.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/quantiles.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

    === "Log: with CI"
        [View in separate window](../tutorials/plot/figures/quantiles_ci_area.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/quantiles_ci_area.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

    === "Log: grouped"
        [View in separate window](../tutorials/plot/figures/quantiles_grouped.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/quantiles_grouped.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

    === "Log: line()"
        [View in separate window](../tutorials/plot/figures/line_selected_quantiles.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/line_selected_quantiles.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "coefplot()"
    A disclosure-review-style point-and-whisker chart, with and without a `series=` for multiple offset points per row.

    === "Code"
        ```python
        --8<-- "tutorials/plot/coefplot.py"
        ```

    === "Log: with series"
        [View in separate window](../tutorials/plot/figures/coefplot.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/coefplot.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

    === "Log: single series"
        [View in separate window](../tutorials/plot/figures/coefplot_single_series.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/coefplot_single_series.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "stacked_bar()"
    Several `StatCalculator` objects (one per age group, plus an overall total) stacked into one bar per category, with the total shown as a text label. Layers/categories may mix positive and negative values (second tab below) - each bar stacks its positive layers right of zero and its negative ones left, and the total label lands on whichever side its own net value falls on.

    === "Code"
        ```python
        --8<-- "tutorials/plot/stacked_bar.py"
        ```

    === "Log: all-negative"
        [View in separate window](../tutorials/plot/figures/stacked_bar.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/stacked_bar.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

    === "Log: mixed sign"
        [View in separate window](../tutorials/plot/figures/stacked_bar_mixed_sign.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/stacked_bar_mixed_sign.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "combine()"
    Two independent runs, each plotted with and without a confidence band - a 2-level dropdown tree (Run -> CI). Also covers nesting a third level (Run -> CI -> Year), labeling each level (`label=["Run:", "\nCI:", "\nYear:"]`), and putting each level on its own row of the control bar (the `"\n"` prefix).

    === "Code"
        ```python
        --8<-- "tutorials/plot/combine.py"
        ```

    === "Log: 2 levels"
        [View in separate window](../tutorials/plot/figures/combine.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/combine.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

    === "Log: 3 levels"
        [View in separate window](../tutorials/plot/figures/combine_3_levels.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/combine_3_levels.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "combine() + shared legend state"
    `coefplot()`/`stacked_bar()` have no group dropdown of their own, but (like `line()`/`quantiles()`) their legend clicks are shared by trace name across figures - isolating a series/layer in one branch and switching to a sibling branch with the same name shows it isolated there too, without touching that sibling's own legend. Two examples, one per plot type, each with two branches sharing the same trace names.

    === "Code: coefplot()"
        ```python
        --8<-- "tutorials/plot/combine_two_coefplots.py"
        ```

    === "Log: coefplot()"
        [View in separate window](../tutorials/plot/figures/combine_two_coefplots.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/combine_two_coefplots.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

    === "Code: stacked_bar()"
        ```python
        --8<-- "tutorials/plot/combine_two_stacked_bars.py"
        ```

    === "Log: stacked_bar()"
        [View in separate window](../tutorials/plot/figures/combine_two_stacked_bars.html){:target="_blank"}
        <iframe src="../../tutorials/plot/figures/combine_two_stacked_bars.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>
