from __future__ import annotations

import json
import os
import re
import shutil
from io import StringIO

from ._dropdown import DEFAULT_LEFT_MARGIN_PX


def combine(
    tree: dict[str, object],
    label: str | list[str | None] = "View:",
    width: int | None = None,
    height: int | None = 700,
    layout: dict | None = None,
    dropdowns_padding_left: int = DEFAULT_LEFT_MARGIN_PX,
) -> "CombinedFigure":
    """
    Combine several figures (from line()/quantiles()/coefplot()/
    stacked_bar(), or any plotly Figure) into one HTML page, switched
    between by one cascading dropdown per nesting level - arbitrarily
    deep, e.g. {"Income": {"2016": fig_a, "2017": fig_b}, "Poverty":
    {"2016": fig_c, "2017": fig_d}} gets two dropdowns (top:
    Income/Poverty, second: year); picking a year under one top choice
    keeps that same year selected if you then switch the top dropdown
    to the other branch (falling back to that branch's first option if
    the previously-picked one doesn't exist there).

    A leaf figure's own internal group dropdown (see
    add_group_dropdown - every line()/coefplot()/stacked_bar() figure
    already has one, even if it's a single inert group covering
    everything) keeps working independently once its figure is shown -
    this only adds the *outer* switching between whole figures, it
    doesn't touch what's inside any of them.

    Parameters
    ----------
    tree : dict[str, ...]
        Nested dict, any depth (branches may differ in depth), with
        plotly Figures at the leaves.
    label : str | list[str | None], optional
        Text shown next to each level's dropdown. A single string only
        labels the top level (deeper ones are unlabeled) - the default,
        "View:". Pass a list to label (or skip, with None or "") each
        level individually, e.g. ["Run:", "CI:"] for a 2-level tree, or
        ["Run:", None, "Year:"] to leave the middle level unlabeled in
        a 3-level one. A "\\n" anywhere in a level's label (it's stripped
        from the displayed text) starts that level's whole control -
        label and dropdown together - on a new row of the control bar,
        e.g. ["Run:", "\\nCI:", "\\nYear:"] puts each level on its own row.
    width, height : int | None, optional
        Applied to every leaf figure before rendering. height default
        700; width default None (each figure's own autosize setting is
        left alone).
    layout : dict | None, optional
        Extra plotly layout properties applied to every leaf figure
        (via fig.update_layout(**layout)), after width/height - useful
        for formatting that should be consistent across every figure in
        the tree (e.g. a shared font). Since each leaf is a plain figure
        you built yourself before handing it to combine(), you can
        always fig.update_layout(...) an individual one beforehand
        instead, if only that one needs something different.
    dropdowns_padding_left : int, optional
        Left padding (px) on the label/dropdown table, default 15 - also
        applied to any leaf's own internal group dropdown (see
        add_group_dropdown), so the two line up instead of each floating
        at a different offset. This is cosmetic spacing for the dropdowns
        (and text) themselves, not an attempt to line them up with a
        figure's plotted data area - that area's left edge floats with
        each figure's own axis-label width (via Plotly's automargin) and
        can differ leaf to leaf, so no fixed padding here would track it
        correctly anyway.

    Returns
    -------
    CombinedFigure
        Call .write_html(path) on it - there's no single Figure object
        to return (this holds several at once), so unlike the other
        plot functions there's nothing to .show() directly in a
        notebook; open the saved HTML instead.
    """
    return CombinedFigure(
        tree,
        label=label,
        width=width,
        height=height,
        layout=layout,
        dropdowns_padding_left=dropdowns_padding_left,
    )


class CombinedFigure:
    def __init__(
        self,
        tree: dict[str, object],
        label: str | list[str | None] = "View:",
        width: int | None = None,
        height: int | None = 700,
        layout: dict | None = None,
        dropdowns_padding_left: int = DEFAULT_LEFT_MARGIN_PX,
    ):
        self.tree = tree
        self.label = label
        self.width = width
        self.height = height
        self.layout = layout
        self.dropdowns_padding_left = dropdowns_padding_left

    def write_html(
        self, path: str, include_plotlyjs: str = "directory", **kwargs
    ) -> None:
        import plotly
        import plotly.graph_objects as go

        leaves = list(_iter_leaves(self.tree))

        for _, leaf in leaves:
            if self.width is not None:
                leaf.update_layout(width=self.width, autosize=False)
            if self.height is not None:
                leaf.update_layout(height=self.height, autosize=False)
            if self.layout:
                leaf.update_layout(**self.layout)

        div_ids = {}
        fragments = {}
        for leaf_path, leaf in leaves:
            buf = StringIO()
            #   Never leaf.write_html(...) here - that's each leaf's OWN
            #   monkeypatched method (from add_group_dropdown), built to
            #   splice its click-handling JS before a "</body>" tag in a
            #   *standalone* page. A fragment (full_html=False) has no
            #   <body> at all, so that splice would silently no-op and
            #   every leaf embedded here would lose its own dropdown/
            #   companion-linking/double-click JS entirely. Get the plain
            #   fragment from the base class instead, then splice each
            #   leaf's own JS in ourselves, right after its fragment.
            go.Figure.write_html(
                leaf, buf, full_html=False, include_plotlyjs=False, **kwargs
            )
            fragment = buf.getvalue()
            m = re.search(r'<div id="([^"]+)" class="plotly-graph-div"', fragment)
            if not m:
                raise ValueError(
                    f"combine(): couldn't find a plotly div id for leaf at {list(leaf_path)!r}."
                )
            plot_id = m.group(1)
            div_ids[leaf_path] = plot_id

            js_template = getattr(leaf, "_dropdown_js_template", None)
            if js_template is not None:
                #   Force this leaf's own group-dropdown widget (if shown)
                #   to the same left offset as our own outer table - see
                #   DEFAULT_LEFT_MARGIN_PX - rather than the leaf's default
                #   of chasing its own (leaf-dependent) axis margin.
                js = js_template.replace("PLOT_ID", plot_id).replace(
                    "LEFT_MARGIN_PX", str(self.dropdowns_padding_left)
                )
                meta = getattr(leaf, "_dropdown_metadata", "{}")
                meta_tag = f'<script id="plot-dropdown-metadata-{plot_id}" type="application/json">{meta}</script>'
                fragment = f"{fragment}\n<script>{js}</script>\n{meta_tag}"

            fragments[leaf_path] = fragment

        js_tree = _tree_to_js(self.tree, div_ids)
        max_depth = max(len(p) for p, _ in leaves)

        container_ids = {
            leaf_path: f"combine-leaf-{i}" for i, (leaf_path, _) in enumerate(leaves)
        }
        container_divs = [
            f'<div id="{container_ids[leaf_path]}" style="display:{"block" if i == 0 else "none"}; '
            f'width:100%; height:100%">{fragments[leaf_path]}</div>'
            for i, (leaf_path, _) in enumerate(leaves)
        ]

        div_to_container = {div_ids[p]: container_ids[p] for p, _ in leaves}

        labels = [self.label] if isinstance(self.label, str) else list(self.label)
        labels = [
            (labels[d] if d < len(labels) else None) or "" for d in range(max_depth)
        ]

        js = _combine_js(
            tree=js_tree,
            div_to_container=div_to_container,
            max_depth=max_depth,
            labels=labels,
            dropdowns_padding_left=self.dropdowns_padding_left,
        )

        if include_plotlyjs == "directory":
            plotlyjs_tag = '<script src="plotly.min.js"></script>'
        elif include_plotlyjs == "cdn":
            plotlyjs_tag = (
                '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
            )
        else:
            plotlyjs_tag = f'<script src="{include_plotlyjs}"></script>'

        html = f"""<!DOCTYPE html>
<html>
<head>
<style>html,body{{height:100%;margin:0;padding:0;}} .svg-container{{height:100%;}}</style>
<meta charset="utf-8" />
{plotlyjs_tag}
</head>
<body>
<div id="combine-root" style="height:100%;">
{"".join(container_divs)}
</div>
<script>{js}</script>
</body>
</html>
"""

        if include_plotlyjs == "directory" and isinstance(path, str):
            src = os.path.join(
                os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"
            )
            dst = os.path.join(os.path.dirname(path), "plotly.min.js")
            if not os.path.isfile(dst):
                shutil.copy(src, dst)

        if isinstance(path, str):
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        else:
            path.write(html)


def _iter_leaves(node, path: tuple = ()):
    if isinstance(node, dict):
        if not node:
            raise ValueError(
                f"combine(): tree{list(path)!r} is an empty dict - every branch needs at least one entry."
            )
        for k, v in node.items():
            yield from _iter_leaves(v, path + (k,))
    else:
        yield path, node


def _tree_to_js(node, div_ids: dict, path: tuple = ()) -> dict:
    """
    {order: [keys...], children: {key: subtree}} per internal node,
    {leaf: div_id} at leaves - explicit `order` (rather than relying on
    JS object key order, which silently reorders integer-looking string
    keys like "2016" ahead of everything else) is what keeps this safe
    for arbitrary key names.
    """
    if isinstance(node, dict):
        return {
            "order": list(node.keys()),
            "children": {
                k: _tree_to_js(v, div_ids, path + (k,)) for k, v in node.items()
            },
        }
    return {"leaf": div_ids[path]}


def _combine_js(
    tree: dict,
    div_to_container: dict,
    max_depth: int,
    labels: list[str],
    dropdowns_padding_left: int,
) -> str:
    tree_js = json.dumps(tree)
    div_to_container_js = json.dumps(div_to_container)
    labels_js = json.dumps(labels)

    return f"""
(function() {{
    var tree = {tree_js};
    var divToContainer = {div_to_container_js};
    var maxDepth = {max_depth};
    var levelLabels = {labels_js};
    var preferredPath = [];
    var currentPath = [];
    var currentLeafDivId = null;
    var selects = [];
    var controlsBar = null;
    var resizing = false;

    function isLeaf(node) {{ return node.leaf !== undefined; }}

    function childrenAt(pathPrefix) {{
        var node = tree;
        for (var i = 0; i < pathPrefix.length; i++) {{
            if (isLeaf(node)) return null;
            node = node.children[pathPrefix[i]];
        }}
        return isLeaf(node) ? null : node;
    }}

    function resolveFullPath(pathPrefix) {{
        var path = pathPrefix.slice();
        var node = childrenAt(path);
        while (node !== null) {{
            var depth = path.length;
            var chosen = (preferredPath[depth] !== undefined && node.order.indexOf(preferredPath[depth]) !== -1)
                ? preferredPath[depth] : node.order[0];
            path.push(chosen);
            var child = node.children[chosen];
            node = isLeaf(child) ? null : child;
        }}
        return path;
    }}

    function leafDivIdAt(path) {{
        var node = tree;
        for (var i = 0; i < path.length; i++) node = node.children[path[i]];
        return node.leaf;
    }}

    function applyRangeUnion(gd, xRange, yRange) {{
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

    function showLeaf(path) {{
        var divId = leafDivIdAt(path);
        var targetContainer = divToContainer[divId];

        //   Capture the currently-shown leaf's zoom/pan range before
        //   switching away from it, so the newly-shown one can be
        //   unioned with it below - flipping between related figures
        //   (e.g. same data, With CI vs No CI) shouldn't reset your
        //   zoom every time.
        var xRange = null, yRange = null;
        if (currentLeafDivId) {{
            var oldGd = document.getElementById(currentLeafDivId);
            if (oldGd && oldGd.layout) {{
                xRange = oldGd.layout.xaxis && oldGd.layout.xaxis.range ? oldGd.layout.xaxis.range.slice() : null;
                yRange = oldGd.layout.yaxis && oldGd.layout.yaxis.range ? oldGd.layout.yaxis.range.slice() : null;
            }}
        }}

        Object.keys(divToContainer).forEach(function(id) {{
            var cid = divToContainer[id];
            var visible = (cid === targetContainer);
            var el = document.getElementById(cid);
            el.style.display = visible ? 'block' : 'none';
            if (visible) el.style.height = '100%';
        }});
        resizeFigures();

        var newGd = document.getElementById(divId);
        //   Pick up whatever isolate/toggle pattern is currently active
        //   (shared across every figure on the page - see
        //   add_group_dropdown) instead of resetting to "everything
        //   visible" just because this particular figure wasn't the one
        //   last interacted with.
        if (newGd && newGd._applyVisible) {{
            newGd._applyVisible();
        }}
        if (newGd && (xRange || yRange)) {{
            applyRangeUnion(newGd, xRange, yRange);
        }}
        currentLeafDivId = divId;
    }}

    function resizeFigures() {{
        //   A hidden (display:none) container's plot gets drawn at 0x0 and
        //   *stays* there - relayout({{autosize:true}}) is a no-op once
        //   autosize is already true (relayout only redraws on an actual
        //   value change), so every figure (shown or not) gets an explicit
        //   height instead, forcing a real redraw when it's later shown.
        if (resizing) return;
        var root = document.getElementById('combine-root');
        var barHeight = controlsBar ? controlsBar.offsetHeight : 0;
        var containerHeight = root.offsetHeight - barHeight;
        if (containerHeight > 0) {{
            resizing = true;
            var promises = Object.keys(divToContainer).map(function(divId) {{
                var gd = document.getElementById(divId);
                return gd ? Plotly.relayout(gd, {{height: containerHeight}}) : Promise.resolve();
            }});
            Promise.all(promises)
                .catch(function(err) {{ console.error('combine(): resizeFigures failed', err); }})
                .then(function() {{ resizing = false; }});
        }}
    }}

    function populateSelect(depth, fullPath) {{
        //   fullPath is the WHOLE resolved path (not sliced to `depth`) -
        //   childrenAt needs the prefix *before* this depth to find this
        //   select's options, but marking the right one selected needs
        //   fullPath[depth] itself. Passing an already-sliced path here
        //   (as an earlier version did) left fullPath[depth] always
        //   undefined, so no option was ever marked selected and the
        //   browser's own "select the first option" default silently
        //   took over - the correct leaf was still shown (that path came
        //   from `currentPath` directly), but the dropdown widget itself
        //   displayed the wrong value.
        var cells = selects[depth].cells;
        var select = selects[depth].select;
        var node = childrenAt(fullPath.slice(0, depth));
        //   No control at all when there's nothing to choose between - a
        //   single-option level (node.order.length === 1) is inert the
        //   same way add_group_dropdown skips its own widget for a single
        //   group; node === null means this depth doesn't exist on the
        //   current branch at all (uneven-depth trees).
        if (node === null || node.order.length <= 1) {{
            cells[0].style.display = 'none';
            cells[1].style.display = 'none';
            return;
        }}
        cells[0].style.display = '';
        cells[1].style.display = '';
        select.innerHTML = '';
        node.order.forEach(function(k) {{
            var opt = document.createElement('option');
            opt.value = k;
            opt.textContent = k;
            if (k === fullPath[depth]) opt.selected = true;
            select.appendChild(opt);
        }});
    }}

    function render(path, fromDepth) {{
        //   fromDepth: only rebuild <option> lists for this depth and
        //   deeper - never the depth that's mid-event right now (its
        //   options and value are already correct; rebuilding a <select>
        //   from inside its own "change" handler is what left it stuck).
        //   Shallower depths are untouched too - their options and
        //   selected value can't have changed either.
        currentPath = path;
        preferredPath = path.slice();
        for (var d = fromDepth; d < maxDepth; d++) populateSelect(d, path);
        showLeaf(path);
    }}

    function onChange(depth, newKey) {{
        try {{
            var prefix = currentPath.slice(0, depth);
            prefix.push(newKey);
            preferredPath[depth] = newKey;
            render(resolveFullPath(prefix), depth + 1);
        }} catch (err) {{
            console.error('combine(): onChange failed for depth ' + depth + ', key ' + newKey, err);
        }}
    }}

    function buildControls() {{
        //   A <table> rather than a flex bar so that when a "\\n" break
        //   puts each level on its own <tr>, every row's label <td> and
        //   select <td> share the same two table columns - the browser
        //   then auto-sizes column 1 to the widest label, which is what
        //   left-aligns every dropdown in column 2 regardless of how long
        //   the label text next to it is.
        var bar = document.createElement('table');
        bar.style.borderCollapse = 'collapse';
        bar.style.margin = '4px 0 8px {dropdowns_padding_left}px';
        bar.style.fontFamily = 'sans-serif';
        bar.style.fontSize = '12px';

        var row = document.createElement('tr');
        bar.appendChild(row);

        for (var d = 0; d < maxDepth; d++) {{
            (function(depth) {{
                var rawLabel = levelLabels[depth] || '';
                if (rawLabel.indexOf('\\n') !== -1) {{
                    //   A "\\n" anywhere in this level's label means "start
                    //   this whole control (label + dropdown together) on a
                    //   new row" - not a line break within the label text
                    //   itself, which is why it's stripped out below.
                    row = document.createElement('tr');
                    bar.appendChild(row);
                    rawLabel = rawLabel.split('\\n').join('');
                }}

                var labelTd = document.createElement('td');
                labelTd.style.padding = '2px 6px 2px 0';
                labelTd.style.whiteSpace = 'nowrap';
                labelTd.textContent = rawLabel;

                var selectTd = document.createElement('td');
                selectTd.style.padding = '2px 12px 2px 0';
                selectTd.style.textAlign = 'left';

                var select = document.createElement('select');
                select.style.fontSize = '12px';
                select.style.fontFamily = 'sans-serif';
                select.addEventListener('change', function() {{ onChange(depth, select.value); }});

                selectTd.appendChild(select);
                row.appendChild(labelTd);
                row.appendChild(selectTd);
                selects.push({{cells: [labelTd, selectTd], select: select}});
            }})(d);
        }}

        var root = document.getElementById('combine-root');
        root.insertBefore(bar, root.firstChild);
        controlsBar = bar;
    }}

    function init() {{
        buildControls();
        render(resolveFullPath([]), 0);

        var observer = new ResizeObserver(resizeFigures);
        observer.observe(document.getElementById('combine-root'));
    }}

    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', init);
    }} else {{
        init();
    }}
}})();
"""
