from __future__ import annotations

import json
import math
import os
import re
import shutil
import types
from io import StringIO


def add_group_dropdown(
    fig,
    groups: dict[str, list[int]],
    companions: dict[int, int] | None = None,
    default_group: str | None = None,
    label: str = "Show:",
    legend_space_per: float = 0.1,
    width: int | None = None,
    height: int | None = None,
) -> "plotly.graph_objects.Figure":  # noqa: F821
    """
    Replace a figure's legend with a compact dropdown that swaps which
    named subset of traces is shown - for a figure with many traces
    (e.g. one line per state), a full legend is unwieldy, but a
    dropdown listing named groups (which need not be single traces -
    each can be several, e.g. an estimate line plus its CI band) is
    not.

    Within whichever group is currently selected, clicking a legend
    entry still toggles that one trace on/off (single click) or
    isolates it (double click, matching plotly's own legend
    convention) - both preserved across switching which group is
    selected, and both preserved across a fig.write_html() round trip
    (the state-tracking JS this injects is self-contained, no
    survey_kit code needed to view the saved HTML later). This click
    handling is always attached, even when there's only one group (so
    no dropdown widget is shown) - it's what makes double-click-isolate
    and companion-trace linking (see `companions`) work consistently
    whether or not there's anything to switch between.

    Which traces are toggled on/off is tracked in a single object shared
    by every figure on the page (not kept private to each figure), keyed
    by each trace's own group + name (its legend label) rather than its
    position - so if "2016" is isolated in one figure, a sibling figure
    with a same-named "2016" series in a same-named group can pick up
    that exact same isolation (via the exposed `gd._applyVisible()`)
    instead of resetting to "everything visible", and two *unrelated*
    series that happen to occupy the same list position in different
    groups never interfere with each other. combine() calls
    `gd._applyVisible()` automatically whenever it switches which whole
    figure is shown.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to modify in place (and also returned, for chaining).
    groups : dict[str, list[int]]
        {group label: [trace indices in fig.data belonging to that
        group]}. A trace may belong to more than one group's list if
        that's useful, though the common case is a partition of
        range(len(fig.data)).
    companions : dict[int, int] | None, optional
        {trace index: companion trace index} - whenever the first
        trace's visibility changes (by any means: single-click toggle,
        double-click isolate, or a group switch), the companion is set
        to match. For a CI band drawn as its own `showlegend=False`
        fill trace (see line()'s ci_area), this is what makes clicking
        its line's legend entry hide the band along with it, instead of
        leaving an orphaned shaded region with no line.
    default_group : str | None, optional
        Which group to show initially. Defaults to the first key in
        `groups`.
    label : str, optional
        Text shown next to the dropdown. Default "Show:".
    legend_space_per : float, optional
        Extra vertical margin reserved below the plot per 10 items in
        the largest group (the native plotly legend still renders for
        whichever traces are visible). Default 0.1.
    width, height : int | None, optional
        If given, fixes the figure's size instead of leaving it
        autosized.

    Returns
    -------
    plotly.graph_objects.Figure
        The same `fig`, modified in place - a `_group_map` and
        `_default_group` attribute are attached (read by any future
        code, e.g. a multi-figure combiner, that wants to know this
        figure already has a group dropdown), and `fig.write_html` is
        replaced with a version that also injects the click-handling
        (and, if there's more than one group, dropdown widget) JS.
    """
    if default_group is None:
        default_group = next(iter(groups.keys()))

    for i in range(len(fig.data)):
        fig.data[i].visible = i in groups[default_group]
        fig.data[i].legendgroup = None
        fig.data[i].legendgrouptitle = None

    if width is not None:
        fig.update_layout(width=width, autosize=False)
    if height is not None:
        fig.update_layout(height=height, autosize=False)
    if width is None and height is None:
        fig.update_layout(autosize=True, height=None, width=None)

    fig.update_layout(
        legend=dict(
            y=-0.15 - legend_space_per * math.ceil(len(groups[default_group]) / 10)
        ),
    )

    fig._group_map = groups
    fig._default_group = default_group
    fig._dropdown_label = label
    fig._companions = companions or {}

    return _attach_dropdown_js(fig, show_widget=len(groups) > 1)


def _attach_dropdown_js(fig, show_widget: bool) -> "plotly.graph_objects.Figure":  # noqa: F821
    group_map = fig._group_map
    default_group = fig._default_group
    label = getattr(fig, "_dropdown_label", "Show:")
    companions = {str(k): v for k, v in getattr(fig, "_companions", {}).items()}

    group_map_js = json.dumps(group_map)
    default_group_js = json.dumps(default_group)
    label_js = json.dumps(label)
    companions_js = json.dumps(companions)
    show_widget_js = json.dumps(show_widget)
    metadata = json.dumps({"group_map": group_map, "default_group": default_group})

    fig._dropdown_js_template = f"""
(function() {{
    var groupTraceMap = {group_map_js};
    var currentGroup = {default_group_js};
    var dropdownLabel = {label_js};
    var companions = {companions_js};
    var showWidget = {show_widget_js};
    var gd = null;

    //   Identifies a trace by "group name:its own trace name" (read live
    //   from gd.data[idx].name - the same label shown in the legend, set
    //   by line()'s own `name=` on each trace) rather than by its
    //   position in groupTraceMap[group]. Position was fragile two ways:
    //   two groups can have a "position 0" that mean completely
    //   unrelated things (e.g. this figure's own "Historical" group's
    //   position 0 is a "2016" series, its "Recent" group's position 0
    //   an unrelated "2018" series - isolating one used to incorrectly
    //   hide the other), and it silently assumed sibling figures (e.g.
    //   combine()'s Run A vs Run B) build their traces in the exact same
    //   order. Keying by name sidesteps both: unrelated series never
    //   collide (their names differ), and related ones match up (their
    //   names agree) regardless of trace order.
    //
    //   Shared across every figure on the page (not just this one's own
    //   closure) via `window`, so state set in one figure carries over
    //   when a sibling figure (see gd._applyVisible) shows the same
    //   group+name combination - e.g. combine()'s outer switch between
    //   Run A and Run B.
    function stateKey(group, idx) {{ return group + ':' + gd.data[idx].name; }}

    if (!window.__survey_kit_name_visible__) window.__survey_kit_name_visible__ = {{}};
    var nameVisible = window.__survey_kit_name_visible__;

    function ensureDefaults() {{
        Object.keys(groupTraceMap).forEach(function(g) {{
            groupTraceMap[g].forEach(function(idx) {{
                var key = stateKey(g, idx);
                if (nameVisible[key] === undefined) nameVisible[key] = true;
            }});
        }});
    }}

    function applyCompanions(visArray) {{
        Object.keys(companions).forEach(function(lineIdx) {{
            visArray[companions[lineIdx]] = visArray[lineIdx];
        }});
    }}

    function applyRangeUnion(xRange, yRange) {{
        var newX = gd.layout.xaxis && gd.layout.xaxis.range ? gd.layout.xaxis.range.slice() : null;
        var newY = gd.layout.yaxis && gd.layout.yaxis.range ? gd.layout.yaxis.range.slice() : null;
        var relayoutArgs = {{}};
        if (xRange && newX) {{
            relayoutArgs['xaxis.range'] = [Math.min(xRange[0], newX[0]), Math.max(xRange[1], newX[1])];
        }}
        if (yRange && newY) {{
            relayoutArgs['yaxis.range'] = [Math.min(yRange[0], newY[0]), Math.max(yRange[1], newY[1])];
        }}
        if (Object.keys(relayoutArgs).length > 0) Plotly.relayout(gd, relayoutArgs);
    }}

    //   Rebuilds this figure's own trace visibility from the CURRENT
    //   group + the shared nameVisible state - exposed as
    //   gd._applyVisible so an outer combine() can resync a sibling
    //   figure (e.g. one just switched into view) to whatever
    //   isolate/toggle pattern was left active elsewhere on the page.
    function applyVisible() {{
        var visArray = new Array(gd.data.length).fill(false);
        groupTraceMap[currentGroup].forEach(function(idx) {{
            visArray[idx] = nameVisible[stateKey(currentGroup, idx)] ? true : 'legendonly';
        }});
        applyCompanions(visArray);
        return Plotly.restyle(gd, {{visible: visArray}});
    }}

    function applyGroup(g) {{
        var xRange = gd.layout.xaxis && gd.layout.xaxis.range ? gd.layout.xaxis.range.slice() : null;
        var yRange = gd.layout.yaxis && gd.layout.yaxis.range ? gd.layout.yaxis.range.slice() : null;
        currentGroup = g;
        applyVisible().then(function() {{ applyRangeUnion(xRange, yRange); }});
    }}

    function wireClickHandling() {{
        var clickTimer = null;
        gd.on('plotly_legendclick', function(e) {{
            var clickedIdx = e.curveNumber;
            if (groupTraceMap[currentGroup].indexOf(clickedIdx) === -1) return;

            if (clickTimer !== null) {{
                clearTimeout(clickTimer);
                clickTimer = null;

                //   Isolate: only the clicked trace's name visible - only
                //   this group's own names are touched, an unrelated
                //   group's state is left untouched.
                groupTraceMap[currentGroup].forEach(function(idx) {{
                    nameVisible[stateKey(currentGroup, idx)] = (idx === clickedIdx);
                }});
                applyVisible();
                return false;
            }}

            clickTimer = setTimeout(function() {{
                clickTimer = null;
                var currentVis = gd.data[clickedIdx].visible;
                nameVisible[stateKey(currentGroup, clickedIdx)] = (currentVis === 'legendonly');
                applyVisible();
            }}, 300);
            return false;
        }});
    }}

    function buildSelectWidget() {{
        var leftMargin = (gd.layout.margin && gd.layout.margin.l) ? gd.layout.margin.l : 80;

        var wrapper = document.createElement('div');
        wrapper.style.position = 'absolute';
        wrapper.style.top = '4px';
        wrapper.style.left = leftMargin + 'px';
        wrapper.style.zIndex = '1000';
        wrapper.style.display = 'flex';
        wrapper.style.alignItems = 'center';
        wrapper.style.gap = '6px';

        var labelEl = document.createElement('span');
        labelEl.textContent = dropdownLabel;
        labelEl.style.fontSize = '12px';
        labelEl.style.fontFamily = 'sans-serif';

        var select = document.createElement('select');
        select.style.fontSize = '12px';
        select.style.fontFamily = 'sans-serif';

        Object.keys(groupTraceMap).forEach(function(opt) {{
            var el = document.createElement('option');
            el.value = opt;
            el.textContent = opt;
            if (opt === currentGroup) el.selected = true;
            select.appendChild(el);
        }});

        select.addEventListener('change', function() {{ applyGroup(select.value); }});

        wrapper.appendChild(labelEl);
        wrapper.appendChild(select);
        gd.style.position = 'relative';
        gd.appendChild(wrapper);
    }}

    function tryInit() {{
        var el = document.getElementById('PLOT_ID');
        if (el && el.data) {{
            gd = el;
            ensureDefaults();
            gd._applyVisible = applyVisible;
            wireClickHandling();
            if (showWidget) buildSelectWidget();
        }} else {{
            setTimeout(tryInit, 50);
        }}
    }}
    tryInit();
}})();
"""
    fig._dropdown_metadata = metadata

    def write_html(self, file, **kwargs):
        import plotly
        import plotly.graph_objects as go

        kwargs.setdefault("include_plotlyjs", "directory")

        buf = StringIO()
        go.Figure.write_html(self, buf, **kwargs)
        html = buf.getvalue()

        div_id_match = re.search(r'<div id="([^"]+)" class="plotly-graph-div"', html)
        if div_id_match:
            plot_id = div_id_match.group(1)
            js = self._dropdown_js_template.replace("PLOT_ID", plot_id)
            meta_tag = f'<script id="plot-dropdown-metadata-{plot_id}" type="application/json">{self._dropdown_metadata}</script>'
            html = html.replace(
                "</body>", f"<script>{js}</script>\n{meta_tag}\n</body>"
            )

        if kwargs["include_plotlyjs"] == "directory" and isinstance(file, str):
            src = os.path.join(
                os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"
            )
            dst = os.path.join(os.path.dirname(file), "plotly.min.js")
            if not os.path.isfile(dst):
                shutil.copy(src, dst)

        if isinstance(file, str):
            with open(file, "w", encoding="utf-8") as f:
                f.write(html)
        else:
            file.write(html)

    fig.write_html = types.MethodType(write_html, fig)
    return fig
