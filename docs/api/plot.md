# Plotting

`survey_kit.plot` builds interactive plotly figures directly from a [`StatCalculator`](basic_standard_errors.md)'s or [`MultipleImputation`](multiple_imputation.md)'s own `df_estimates`/CI columns - no separate reshaping step. Every figure carries a hand-rolled dropdown (not plotly's native `updatemenus`) for switching which group of series is shown, with state preserved across switches and shared across figures on the same page - see [Plotting](../user-guide/plot.md) for a walkthrough.

## Core Plots

::: survey_kit.plot.line
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.plot.quantiles
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.plot.coefplot
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

::: survey_kit.plot.stacked_bar
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

## Combining Figures

::: survey_kit.plot.combine
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3

## Lower-Level

`add_group_dropdown` is what `line()`/`quantiles()`/`coefplot()`/`stacked_bar()` each call internally to attach their own group dropdown - use it directly only if you're building a custom figure (a raw `plotly.graph_objects.Figure`) that should get the same dropdown/state-preservation behavior.

::: survey_kit.plot.add_group_dropdown
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3
