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
- **`combine()`** - nests whole figures under as many dropdown levels as the dict you pass it has, preserving each leaf figure's own internal group dropdown.
- **`group_by`** - splits each series label on a separator so related series (e.g. two historical years vs. a recent one) land in the same dropdown entry instead of getting one each.

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

    === "Log"
        [View in separate window](../tutorials/plot/line_and_quantiles.html){:target="_blank"}
        <iframe src="../../tutorials/plot/line_and_quantiles.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "coefplot()"
    A disclosure-review-style point-and-whisker chart, with and without a `series=` for multiple offset points per row.

    === "Code"
        ```python
        --8<-- "tutorials/plot/coefplot.py"
        ```

    === "Log"
        [View in separate window](../tutorials/plot/coefplot.html){:target="_blank"}
        <iframe src="../../tutorials/plot/coefplot.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "stacked_bar()"
    Several `StatCalculator` objects (one per age group, plus an overall total) stacked into one bar per category, with the total shown as a text label.

    === "Code"
        ```python
        --8<-- "tutorials/plot/stacked_bar.py"
        ```

    === "Log"
        [View in separate window](../tutorials/plot/stacked_bar.html){:target="_blank"}
        <iframe src="../../tutorials/plot/stacked_bar.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "combine()"
    Two independent runs, each plotted with and without a confidence band - a 3-level dropdown tree (Run -> CI). Also covers nesting a third level (Run -> CI -> Year).

    === "Code"
        ```python
        --8<-- "tutorials/plot/combine.py"
        ```

    === "Log"
        [View in separate window](../tutorials/plot/combine.html){:target="_blank"}
        <iframe src="../../tutorials/plot/combine.html"
            style="width: 100%; height: 800px; border: none;">
        </iframe>
