from __future__ import annotations
from itertools import chain, cycle, islice


def default_colors(
    n: int, repeat_frequency: int = 0, n_before_change: int = 0
) -> list[str]:
    """
    n qualitative colors, cycling through plotly's own qualitative
    palettes once they run out (G10, then Dark24, then Prism) rather
    than repeating a single 10-color palette.

    repeat_frequency/n_before_change reproduce the original grouped-color
    convention: with both 0 (the default), colors just cycle through the
    full palette one per series. repeat_frequency > 0 instead cycles
    through only its first `repeat_frequency` colors (e.g. so series
    that recur across several groups, like "2016"/"2017"/"2018" inside
    each of several regions, get the same color per recurring position).
    n_before_change > 0 (takes priority if both are given) instead holds
    each color for `n_before_change` consecutive series before advancing
    - the opposite grouping, for when several consecutive series (e.g.
    a line and its own CI band, or several related sub-series) should
    share one color as a block.
    """
    import plotly.express as px

    palette = (
        px.colors.qualitative.G10
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Prism
    )
    return _cycle_palette(palette, n, repeat_frequency, n_before_change)


def default_dashes(
    n: int, repeat_frequency: int = 0, n_before_change: int = 0
) -> list[str]:
    """Same grouping conventions as default_colors(), over plotly's dash styles instead of colors."""
    palette = ["solid", "dash", "dot", "longdash", "dashdot", "longdashdot"]
    return _cycle_palette(palette, n, repeat_frequency, n_before_change)


def _cycle_palette(
    palette: list[str], n: int, repeat_frequency: int, n_before_change: int
) -> list[str]:
    if n_before_change > 0:
        source = chain.from_iterable([c] * n_before_change for c in cycle(palette))
    else:
        frequency = repeat_frequency if repeat_frequency > 0 else len(palette)
        prefix = list(islice(cycle(palette), frequency))
        source = cycle(prefix)
    return list(islice(source, n))
