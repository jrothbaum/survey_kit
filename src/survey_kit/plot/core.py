from __future__ import annotations

import narwhals as nw
import polars as pl

from ._colors import default_colors, default_dashes
from ._dropdown import add_group_dropdown, attach_shared_legend_state
from ._reshape import (
    apply_rename,
    long_frame,
    numeric_x_values,
    quantile_columns,
    series_label,
    sort_to_order,
    stat_index_columns,
)


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        message = (
            "survey_kit.plot needs the optional 'plotly' package, which "
            "isn't installed. Install it with `uv add --dev plotly` (or "
            "`pip install plotly`), then call this again."
        )
        raise ImportError(message) from e
    return go


def line(
    stat_item,
    x_columns: list[str],
    x_values: dict[str, float] | None = None,
    series: str | None = None,
    ci_level: float | None = None,
    ci_area: bool = False,
    filter_expr: nw.Expr | None = None,
    rename: dict[str, str] | None = None,
    order: list[str] | None = None,
    round_output: bool = True,
    group_by: str | None = None,
    group_first: bool = True,
    group_label: str = "Show:",
    x_axis_title: str = "",
    y_axis_title: str = "",
    x_axis_range: list[float] | None = None,
    y_axis_range: list[float] | None = None,
    x_tick_frequency: float | None = None,
    colors: list[str] | None = None,
    use_dashes: bool = False,
    color_repeat_frequency: int = 0,
    color_n_before_change: int = 0,
    layout: dict | None = None,
    fig: "plotly.graph_objects.Figure | None" = None,  # noqa: F821
) -> "plotly.graph_objects.Figure":  # noqa: F821
    """
    One line per series (a StatCalculator/MultipleImputation index
    value, e.g. "Variable" or a by-group) across a set of columns of
    stat_item.df_estimates treated as the x-axis - e.g. quantile
    columns (see quantiles(), a thin wrapper around this), or any other
    set of numeric-valued columns you want plotted against each other
    (e.g. one column per year).

    Parameters
    ----------
    stat_item : StatCalculator | MultipleImputation
    x_columns : list[str]
        Columns of stat_item.df_estimates to plot, one point per
        column per series.
    x_values : dict[str, float] | None, optional
        {column: x position}. If None, inferred from each column's own
        name (see numeric_x_values()) when every column in x_columns is
        itself numeric - either literally (e.g. "2016", "2017", "2018" -
        a wide-by-year table plots on a real year axis instead of
        0,1,2) or as a quantile column ("q10"/"q0_1" -> 10, the
        percentile); else each column's 0-indexed position in
        x_columns.
    series : str | None, optional
        Which stat_index_columns(stat_item) column identifies one line.
        If None and there's only one index column, that one is used; if
        there's more than one, they're joined into a single label
        (series_label) and each unique combination is one line.
    ci_level : float | None, optional
        If given, also draw confidence intervals at this level (e.g.
        0.95) via stat_item._df_ci(ci_level).
    ci_area : bool, optional
        Draw the CI as a shaded band around the line instead of
        per-point error bars. Only used when ci_level is given.
    filter_expr : nw.Expr | None, optional
        Applied to stat_item.df_estimates (and _df_ci, if used) before
        plotting.
    rename : dict[str, str] | None, optional
        {old series label: new series label} - renames the legend/dropdown
        entries; unmatched labels pass through unchanged.
    order : list[str] | None, optional
        Explicit series order (also filters to only these series) - by
        the *renamed* label if `rename` is also given.
    round_output : bool, optional
        Round values per Census DRB disclosure rules before plotting.
        Default True.
    group_by : str | None, optional
        Every figure's legend is replaced with a compact dropdown (see
        add_group_dropdown); this controls how series are grouped into
        dropdown entries. None (default) puts every series in one
        group, so the dropdown has a single entry and all series show
        at once - visually the same as a plain legend. Given a
        separator (e.g. ":"), each series label is split on it once -
        the prefix and suffix, one of which becomes the dropdown
        group (see group_first) and the other the series' name within
        that group - so related series (e.g. "allhh:Survey" and
        "allhh:NEWS") land in the same group and are shown together. A
        label without the separator falls back to its own group.
    group_first : bool, optional
        When group_by splits a label, use the part before the
        separator as the group name (True, default) or the part after
        it (False).
    group_label : str, optional
        Text shown next to the group dropdown (see add_group_dropdown).
        Default "Show:".
    x_axis_title, y_axis_title : str, optional
    x_axis_range, y_axis_range : list[float] | None, optional
    x_tick_frequency : float | None, optional
        Spacing between x-axis ticks.
    colors : list[str] | None, optional
        One color per series, cycled if shorter. Defaults to plotly's
        qualitative palettes.
    use_dashes : bool, optional
        Cycle line dash styles across series in addition to color.
    color_repeat_frequency : int, optional
        Cycle colors (and dashes, if use_dashes) through only the first
        `color_repeat_frequency` palette entries instead of the whole
        palette - e.g. so a recurring series (same year across several
        groups) always gets the same color at each recurrence. 0
        (default) uses the full palette.
    color_n_before_change : int, optional
        Hold each color (and dash) for this many *consecutive* series
        before advancing to the next one, instead of advancing every
        series - e.g. so several consecutive related series share one
        color as a block. Takes priority over color_repeat_frequency if
        both are given; 0 (default) advances every series.
    layout : dict | None, optional
        Extra plotly layout properties, applied last via
        fig.update_layout(**layout) - so this always overrides any
        survey_kit default (e.g. {"title": "...", "font": {"size": 16}}).
        The returned figure is a plain plotly Figure regardless, so
        fig.update_layout(...)/fig.update_xaxes(...)/etc. also work fine
        called yourself afterward - this is only a same-call convenience.
    fig : plotly.graph_objects.Figure | None, optional
        Add these lines to an existing figure instead of a new one.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _plotly()

    index_cols = stat_index_columns(stat_item)
    series_col = series or (index_cols[0] if len(index_cols) == 1 else "___series___")

    if x_values is None:
        x_values = numeric_x_values(x_columns)
        if x_values is None:
            x_values = {c: float(i) for i, c in enumerate(x_columns)}

    frames = long_frame(
        stat_item,
        value_columns=x_columns,
        ci_level=ci_level,
        filter_expr=filter_expr,
        round_output=round_output,
    )

    for key, dfi in frames.items():
        if series_col == "___series___":
            dfi = series_label(dfi, index_cols)
        if rename:
            dfi = apply_rename(dfi, series_col, rename)
        dfi = dfi.with_columns(
            pl.col("Column")
            .replace_strict(x_values, return_dtype=pl.Float64)
            .alias("X")
        )
        frames[key] = dfi

    df_est = frames["estimates"]
    if order is not None:
        df_est = sort_to_order(df_est, series_col, order)
        series_list = order
    else:
        series_list = df_est[series_col].unique(maintain_order=True).to_list()

    if fig is None:
        fig = go.Figure()

    colors = colors or default_colors(
        len(series_list),
        repeat_frequency=color_repeat_frequency,
        n_before_change=color_n_before_change,
    )
    dashes = (
        default_dashes(
            len(series_list),
            repeat_frequency=color_repeat_frequency,
            n_before_change=color_n_before_change,
        )
        if use_dashes
        else ["solid"] * len(series_list)
    )

    trace_groups: dict[str, list[int]] = {}
    companions: dict[int, int] = {}
    for i, seriesi in enumerate(series_list):
        group_name, item_label = _split_group(str(seriesi), group_by, group_first)

        dfi = df_est.filter(pl.col(series_col) == seriesi).sort("X")

        extra = {}
        if ci_level is not None and not ci_area:
            df_ci_i = frames["ci"].filter(pl.col(series_col) == seriesi).sort("X")
            extra["error_y"] = dict(type="data", array=df_ci_i["Value"].to_list())

        fig.add_trace(
            go.Scatter(
                x=dfi["X"],
                y=dfi["Value"],
                mode="lines",
                name=item_label,
                line=dict(color=colors[i], dash=dashes[i]),
                **extra,
            )
        )
        line_idx = len(fig.data) - 1
        trace_indices = [line_idx]

        if ci_level is not None and ci_area:
            df_ci_i = frames["ci"].filter(pl.col(series_col) == seriesi).sort("X")
            x = dfi["X"].to_list()
            y = dfi["Value"].to_list()
            half_width = df_ci_i["Value"].to_list()
            lower = [v - h for v, h in zip(y, half_width)]
            upper = [v + h for v, h in zip(y, half_width)]

            color_rgb = _hex_to_rgb(colors[i])
            fig.add_trace(
                go.Scatter(
                    x=x + x[::-1],
                    y=lower + upper[::-1],
                    fill="toself",
                    fillcolor=f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fill_idx = len(fig.data) - 1
            trace_indices.append(fill_idx)
            #   So clicking (or isolating) the line's legend entry hides/shows
            #   its CI band with it, instead of leaving an orphaned shaded
            #   region once the line itself is gone.
            companions[line_idx] = fill_idx

        trace_groups.setdefault(group_name, []).extend(trace_indices)

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=x_axis_title,
            range=x_axis_range,
            dtick=x_tick_frequency,
            showgrid=False,
            zeroline=True,
            zerolinecolor="lightgray",
        ),
        yaxis=dict(
            title=y_axis_title,
            range=y_axis_range,
            showgrid=False,
            zeroline=True,
            zerolinecolor="lightgray",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="center",
            y=-0.2,
            x=0.5,
            title_text="",
        ),
    )

    fig = add_group_dropdown(
        fig, trace_groups, companions=companions, label=group_label
    )
    if layout:
        fig.update_layout(**layout)
    return fig


def quantiles(
    stat_item,
    quantile_list: list[float] | None = None,
    x_axis_title: str = "Percentile",
    x_axis_range: list[float] | None = None,
    **kwargs,
) -> "plotly.graph_objects.Figure":  # noqa: F821
    """
    line(), specialized to a StatCalculator/MultipleImputation's own
    quantile-stat columns (see quantile_columns()) - one line per
    series across percentile 0-100 on the x-axis.

    Parameters
    ----------
    stat_item : StatCalculator | MultipleImputation
    quantile_list : list[float] | None, optional
        Which quantiles to plot (as fractions 0-1 or percentiles
        0-100 - either is accepted). None (default) plots every
        quantile column present.
    x_axis_title : str, optional
        Default "Percentile".
    x_axis_range : list[float] | None, optional
        Default [0, 100].
    **kwargs
        Passed through to line() (series, ci_level, ci_area,
        filter_expr, rename, order, group_by, group_first, group_label,
        colors, ...).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    q_cols = quantile_columns(stat_item, quantiles=quantile_list)
    if not q_cols:
        raise ValueError(
            "quantiles(): stat_item.df_estimates has no quantile-stat columns "
            "matching quantile_list (or none at all) - these come from "
            "Statistics(stats=['q10', 'q50', ...]) / stats=['median']."
        )

    x_columns = sorted(q_cols, key=lambda c: q_cols[c])
    x_values = {c: q_cols[c] * 100 for c in x_columns}

    if x_axis_range is None:
        x_axis_range = [0, 100]

    return line(
        stat_item,
        x_columns=x_columns,
        x_values=x_values,
        x_axis_title=x_axis_title,
        x_axis_range=x_axis_range,
        **kwargs,
    )


def coefplot(
    stat_item,
    column: str,
    ci_level: float | None = None,
    category: str | None = None,
    series: str | None = None,
    filter_expr: nw.Expr | None = None,
    rename: dict[str, str] | None = None,
    order: list[str] | None = None,
    headers: dict[str, str] | None = None,
    round_output: bool = True,
    item_spacing: float = 1.0,
    offset_shift: float = 0.1,
    x_axis_title: str = "",
    y_axis_title: str = "",
    x_axis_range: list[float] | None = None,
    colors: list[str] | None = None,
    height: int | None = None,
    width: int = 1000,
    layout: dict | None = None,
    fig: "plotly.graph_objects.Figure | None" = None,  # noqa: F821
) -> "plotly.graph_objects.Figure":  # noqa: F821
    """
    Stata coefplot-style horizontal dot-and-whisker chart: one row per
    category (e.g. one per demographic subgroup), one point per row
    per series (e.g. one color per year, offset vertically so they
    don't overlap) - matches the classic disclosure-review chart of
    "estimate +/- CI, one row per subgroup, one color per year/source".

    Has no group dropdown of its own (there's nothing to switch between -
    every series is always shown), but single/double-click toggle/isolate
    on the legend is still shared by trace name with any other
    line()/quantiles()/coefplot()/stacked_bar() figure on the same page
    (see add_group_dropdown) - isolating "2018" here and switching to a
    sibling figure inside combine() that also has a "2018" series shows
    it isolated there too.

    Parameters
    ----------
    stat_item : StatCalculator | MultipleImputation
    column : str
        Which column of stat_item.df_estimates to plot (e.g. "poverty",
        or a compare()d "ratio"/"difference" column).
    ci_level : float | None, optional
        If given, draw horizontal error bars at this level (e.g. 0.95)
        via stat_item._df_ci(ci_level).
    category : str | None, optional
        Which stat_index_columns(stat_item) column is the row/category
        axis. Defaults to the one index column that isn't `series` (or
        the only index column, if there's just one).
    series : str | None, optional
        Which index column offsets multiple points per category row
        (e.g. "year") - each unique value becomes one color/trace. None
        (default) plots a single, unoffset point per category.
    filter_expr : nw.Expr | None, optional
        Applied to stat_item.df_estimates (and _df_ci, if used) before
        plotting.
    rename : dict[str, str] | None, optional
        {old category label: new category label}.
    order : list[str] | None, optional
        Explicit category order, top-to-bottom - by the *renamed*
        label if `rename` is also given. Also filters to only these
        categories.
    headers : dict[str, str] | None, optional
        {category label: header text} - inserts a bold divider row
        with that text directly above the given category's row. Keyed
        the same way as `order` (post-rename).
    round_output : bool, optional
        Round values per Census DRB disclosure rules before plotting.
        Default True.
    item_spacing : float, optional
        Vertical spacing between category rows. Default 1.0.
    offset_shift : float, optional
        Vertical offset between each series' points within a category
        row (only matters when `series` is given). Default 0.1.
    x_axis_title, y_axis_title : str, optional
    x_axis_range : list[float] | None, optional
    colors : list[str] | None, optional
        One color per series (or a single color, if `series` is None).
    height : int | None, optional
        Defaults to scaling with the number of category rows.
    width : int, optional
        Default 1000.
    layout : dict | None, optional
        Extra plotly layout properties, applied last via
        fig.update_layout(**layout) - overrides any survey_kit default.
    fig : plotly.graph_objects.Figure | None, optional
        Add these points to an existing figure instead of a new one.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _plotly()

    index_cols = stat_index_columns(stat_item)
    if category is None:
        remaining = [c for c in index_cols if c != series]
        if len(remaining) != 1:
            raise ValueError(
                f"coefplot(): can't infer `category` from stat_index_columns(stat_item)="
                f"{index_cols} with series={series!r} - pass `category` explicitly."
            )
        category = remaining[0]

    frames = long_frame(
        stat_item,
        value_columns=[column],
        ci_level=ci_level,
        filter_expr=filter_expr,
        round_output=round_output,
    )

    for key, dfi in frames.items():
        if rename:
            dfi = apply_rename(dfi, category, rename)
        frames[key] = dfi

    df_est = frames["estimates"]
    if order is not None:
        df_est = sort_to_order(df_est, category, order)
        category_order = order
    else:
        category_order = df_est[category].unique(maintain_order=True).to_list()

    headers = headers or {}
    display_rows = []
    for cati in category_order:
        if cati in headers:
            display_rows.append((True, headers[cati]))
        display_rows.append((False, cati))

    category_position = {}
    tickvals, ticktext = [], []
    for i, (is_header, label) in enumerate(display_rows):
        pos = i * item_spacing
        tickvals.append(pos)
        ticktext.append(f"<b>{label}</b>" if is_header else label)
        if not is_header:
            category_position[label] = pos

    series_list = (
        [None]
        if series is None
        else df_est[series].unique(maintain_order=True).to_list()
    )
    colors = colors or default_colors(len(series_list))

    if fig is None:
        fig = go.Figure()

    for i, seriesi in enumerate(series_list):
        dfi = df_est if seriesi is None else df_est.filter(pl.col(series) == seriesi)
        dfi = dfi.filter(pl.col(category).is_in(list(category_position.keys())))

        y = [category_position[c] + offset_shift * i for c in dfi[category].to_list()]
        x = dfi["Value"].to_list()

        extra = {}
        if ci_level is not None:
            df_ci_i = (
                frames["ci"]
                if seriesi is None
                else frames["ci"].filter(pl.col(series) == seriesi)
            )
            df_ci_i = df_ci_i.filter(
                pl.col(category).is_in(list(category_position.keys()))
            )
            ci_by_category = dict(
                zip(df_ci_i[category].to_list(), df_ci_i["Value"].to_list())
            )
            half_width = [ci_by_category[c] for c in dfi[category].to_list()]
            extra["error_x"] = dict(type="data", array=half_width, color=colors[i])
            hover_text = [
                f"{c}={v}+/-{h}"
                for c, v, h in zip(dfi[category].to_list(), x, half_width)
            ]
        else:
            hover_text = [f"{c}={v}" for c, v in zip(dfi[category].to_list(), x)]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=str(seriesi) if seriesi is not None else column,
                text=hover_text,
                hoverinfo="text",
                marker=dict(color=colors[i]),
                **extra,
            )
        )

    if height is None:
        height = len(display_rows) * 25 * item_spacing + 250

    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=x_axis_title,
            range=x_axis_range,
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="gray",
            linewidth=1,
            linecolor="gray",
        ),
        yaxis=dict(
            title=y_axis_title,
            tickvals=tickvals,
            ticktext=ticktext,
            range=[
                -item_spacing,
                (len(display_rows) - 1) * item_spacing + item_spacing,
            ],
            showgrid=False,
            linewidth=1,
            linecolor="gray",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="center",
            y=-0.12,
            x=0.5,
            yref="container",
        ),
    )

    if layout:
        fig.update_layout(**layout)
    return attach_shared_legend_state(fig)


def stacked_bar(
    items: dict[str, object],
    column: str,
    category: str | None = None,
    total_key: str | None = None,
    filter_expr: nw.Expr | None = None,
    rename: dict[str, str] | None = None,
    order: list[str] | None = None,
    round_output: bool = True,
    horizontal: bool = True,
    label_scale: float = 1.0,
    label_round_digits: int | None = None,
    value_axis_range: list[float] | None = None,
    x_axis_title: str = "",
    y_axis_title: str = "",
    colors: list[str] | None = None,
    height: int | None = None,
    width: int = 1000,
    layout: dict | None = None,
    fig: "plotly.graph_objects.Figure | None" = None,  # noqa: F821
) -> "plotly.graph_objects.Figure":  # noqa: F821
    """
    Stacked bar decomposition: {name: StatCalculator|MultipleImputation}
    where every item shares the same category axis (e.g. "which
    safety-net program was removed") - each dict entry becomes one
    layer of the stack (e.g. one age group's contribution), stacked per
    category. A shared blue gradient (lightest first) colors the layers
    in dict order, since these are typically ordered meaningfully (e.g.
    smallest to largest group). Layers, and categories, may mix positive
    and negative values freely - each bar stacks its positive layers to
    the right of zero and its negative ones to the left, and a total_key
    label lands on whichever side its own net value falls on.

    Has no group dropdown of its own, but single/double-click toggle/
    isolate on the legend is shared by trace (layer) name with any other
    line()/quantiles()/coefplot()/stacked_bar() figure on the same page
    (see add_group_dropdown) - isolating "Under 18" here and switching to
    a sibling figure inside combine() that also has an "Under 18" layer
    shows it isolated there too.

    Parameters
    ----------
    items : dict[str, StatCalculator | MultipleImputation]
        One item per stack layer, in the order they should stack.
    column : str
        Which column of each item's df_estimates to plot.
    category : str | None, optional
        Which stat_index_columns(item) column is the category axis (the
        bars/rows). Defaults to the one index column, if every item has
        exactly one.
    total_key : str | None, optional
        A key in `items` to exclude from the stack and instead render
        as a text total next to each category's fully-stacked bar (its
        own value, not a re-derived sum - useful as a sanity check that
        the stack's layers add up to an independently-computed total).
    filter_expr : nw.Expr | None, optional
        Applied to every item's df_estimates before plotting.
    rename : dict[str, str] | None, optional
        {old category label: new category label} - applied uniformly to
        every item.
    order : list[str] | None, optional
        Explicit category order - by the *renamed* label if `rename` is
        also given. Also filters to only these categories.
    round_output : bool, optional
        Round values per Census DRB disclosure rules before plotting.
        Default True.
    horizontal : bool, optional
        Horizontal (default) or vertical bars.
    label_scale : float, optional
        Multiplier applied to `total_key`'s values before display (e.g.
        1/1_000_000 to show millions). Default 1 (no rescaling).
    label_round_digits : int | None, optional
        Round the displayed total label to this many digits.
    value_axis_range : list[float] | None, optional
        Range of the value axis (x if horizontal, else y).
    x_axis_title, y_axis_title : str, optional
        Titles for the physical x/y axis, whichever one that ends up
        being (the value axis when horizontal, the category axis when
        not).
    colors : list[str] | None, optional
        One color per stacked layer (excluding total_key), in dict
        order. Defaults to a light-to-dark blue gradient.
    height, width : int | None, optional
        height defaults to scaling with the number of categories; width
        defaults to 1000.
    layout : dict | None, optional
        Extra plotly layout properties, applied last via
        fig.update_layout(**layout) - overrides any survey_kit default.
    fig : plotly.graph_objects.Figure | None, optional
        Add these bars to an existing figure instead of a new one.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _plotly()

    stack_keys = [k for k in items if k != total_key]
    if not stack_keys:
        raise ValueError(
            "stacked_bar(): `items` has nothing to stack (every key is total_key)."
        )

    values_by_key: dict[str, dict[str, float]] = {}
    category_order = order
    for key, item in items.items():
        index_cols = stat_index_columns(item)
        cat_col = category or (index_cols[0] if len(index_cols) == 1 else None)
        if cat_col is None:
            raise ValueError(
                f"stacked_bar(): can't infer `category` for items[{key!r}] with "
                f"stat_index_columns={index_cols} - pass `category` explicitly."
            )

        df = long_frame(
            item,
            value_columns=[column],
            filter_expr=filter_expr,
            round_output=round_output,
        )["estimates"]
        if rename:
            df = apply_rename(df, cat_col, rename)
        if category_order is None:
            category_order = df[cat_col].unique(maintain_order=True).to_list()

        values_by_key[key] = dict(zip(df[cat_col].to_list(), df["Value"].to_list()))

    colors = colors or _blue_gradient(len(stack_keys))

    if fig is None:
        fig = go.Figure()

    for i, key in enumerate(stack_keys):
        values = [values_by_key[key].get(c) for c in category_order]
        bar_kwargs = dict(name=key, marker=dict(color=colors[i]))
        if horizontal:
            fig.add_trace(
                go.Bar(x=values, y=category_order, orientation="h", **bar_kwargs)
            )
        else:
            fig.add_trace(go.Bar(x=category_order, y=values, **bar_kwargs))

    if total_key is not None:
        for cat in category_order:
            value = values_by_key[total_key].get(cat)
            if value is None:
                continue
            display_value = value * label_scale
            if label_round_digits is not None:
                display_value = round(display_value, label_round_digits)
            shift = 15 if value >= 0 else -15
            if horizontal:
                #   xanchor pins the text's near edge (not its center) to
                #   the shifted point, so the label extends outward, away
                #   from the bar, instead of straddling it - without this,
                #   the default center anchor put roughly half of a
                #   negative-value label even further left than intended,
                #   overlapping the category axis labels.
                fig.add_annotation(
                    x=value,
                    y=cat,
                    text=str(display_value),
                    showarrow=False,
                    xshift=shift,
                    xanchor="left" if value >= 0 else "right",
                )
            else:
                fig.add_annotation(
                    x=cat,
                    y=value,
                    text=str(display_value),
                    showarrow=False,
                    yshift=shift,
                    yanchor="bottom" if value >= 0 else "top",
                )

    if height is None:
        height = len(category_order) * 35 + 400

    if value_axis_range is None and total_key is not None:
        #   Without this, a total label sits right at (or past) the axis's
        #   own auto-computed edge - rangemode="tozero" only guarantees 0 is
        #   included, not any headroom past the data's own extreme, so the
        #   most extreme total's label had nowhere to go but into the
        #   category axis's own label area. The bound has to come from the
        #   actual plotted stack extents (each category's positive layers
        #   summed separately from its negative ones, matching how
        #   barmode="stack" itself splits mixed-sign layers on either side
        #   of zero), not just total_key's own values - a category can have
        #   a layer that reaches further than its net total does (e.g. one
        #   positive layer partly offsetting two negative ones), and
        #   bounding on the total alone would clip that layer's bar.
        pos_extents, neg_extents = [0.0], [0.0]
        for cat in category_order:
            layer_vals = [values_by_key[k].get(cat) for k in stack_keys]
            pos_extents.append(sum(v for v in layer_vals if v is not None and v > 0))
            neg_extents.append(sum(v for v in layer_vals if v is not None and v < 0))
        total_values = [v for v in values_by_key[total_key].values() if v is not None]
        lo = min(neg_extents + total_values)
        hi = max(pos_extents + total_values)
        pad = max(abs(lo), abs(hi)) * 0.12
        value_axis_range = [lo - pad if lo < 0 else lo, hi + pad if hi > 0 else hi]

    value_axis = dict(
        range=value_axis_range,
        showgrid=True,
        gridcolor="lightgray",
        zeroline=True,
        zerolinecolor="gray",
        linewidth=1,
        linecolor="gray",
        rangemode="tozero",
    )
    category_axis = dict(showgrid=False, linewidth=0)

    fig.update_layout(
        width=width,
        height=height,
        barmode="stack",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=x_axis_title, **(value_axis if horizontal else category_axis)),
        yaxis=dict(title=y_axis_title, **(category_axis if horizontal else value_axis)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="center",
            traceorder="normal",
            y=-0.2,
            x=0.5,
            title_text="",
        ),
    )

    if layout:
        fig.update_layout(**layout)
    return attach_shared_legend_state(fig)


def _blue_gradient(n: int) -> list[str]:
    darkest = (15, 15, 80)
    lightest = (230, 230, 255)
    if n <= 1:
        return ["rgb({},{},{})".format(*darkest)]
    return [
        "rgb({},{},{})".format(
            *(
                int(d + (i / (n - 1)) * (light - d))
                for d, light in zip(darkest, lightest)
            )
        )
        for i in range(n)
    ]


def _split_group(
    label: str, group_by: str | None, group_first: bool
) -> tuple[str, str]:
    """(dropdown group name, trace's name within that group) for one series label."""
    if group_by is None:
        return "All", label

    parts = label.split(group_by, 1)
    if len(parts) != 2:
        return label, label

    prefix, suffix = parts
    return (prefix, suffix) if group_first else (suffix, prefix)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    if color.startswith("#"):
        h = color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    if color.startswith("rgb"):
        nums = color[color.find("(") + 1 : color.find(")")].split(",")
        return tuple(int(float(n)) for n in nums[:3])
    raise ValueError(
        f"Can't parse color {color!r} as hex or rgb(...) for a CI band fill."
    )
