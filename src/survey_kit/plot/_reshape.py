from __future__ import annotations

import narwhals as nw
import polars as pl

from ..utilities.rounding import drb_round_table


def stat_index_columns(stat_item) -> list[str]:
    """
    Row-identifying columns for a StatCalculator or MultipleImputation.

    The two classes' own `summarize_vars` mean different things despite
    the shared name, so this can't be one expression for both:
    - StatCalculator: `summarize_vars` is `_by_vars()` - the grouping
      columns from its own `by=`, distinct from variable_ids and from
      its stat-value columns. The row index is variable_ids +
      summarize_vars.
    - MultipleImputation: `summarize_vars` is *every* non-join_on column
      of df_estimates - i.e. the stat-value columns themselves (its
      analogue of StatCalculator's stat columns, not of its by-vars).
      Any real grouping dimension has to already be folded into
      `join_on` at construction (mi_ses_from_function(join_on=[...])),
      since MultipleImputation has no separate `by=` concept - so the
      row index is join_on alone; appending summarize_vars here would
      wrongly fold value columns into the index.
    """
    join_on = getattr(stat_item, "join_on", None)
    if join_on is not None:
        return list(join_on)
    return list(stat_item.variable_ids) + list(stat_item.summarize_vars)


def _quantile_fraction(column: str) -> float | None:
    """
    The quantile fraction (0-1) a column name encodes, or None if it
    doesn't look like one. survey_kit names quantile-stat columns either
    "q<percentile>" (e.g. "q10", "q50" - what Statistics(stats=[...])
    produces directly) or "q<fraction, "_" for ".">" (e.g. "q0_1",
    "q0_5" - Statistics.stat_suffix()'s convention) - both recognized.
    """
    if not column.startswith("q"):
        return None
    rest = column[1:]
    if not rest.replace("_", "").isdigit():
        return None
    return float(rest.replace("_", ".")) if "_" in rest else float(rest) / 100


def quantile_columns(
    stat_item, quantiles: list[float] | None = None
) -> dict[str, float]:
    """
    {column_name: quantile_fraction (0-1)} for every quantile-stat column
    on stat_item.df_estimates (see _quantile_fraction for the naming
    conventions recognized). If `quantiles` is given (as fractions or
    0-100 percentiles - either is accepted), only matching columns are
    kept.
    """
    cols = nw.from_native(stat_item.df_estimates).lazy().collect_schema().names()
    index_cols = set(stat_index_columns(stat_item))

    found = {}
    for coli in cols:
        if coli in index_cols:
            continue
        fraction = _quantile_fraction(coli)
        if fraction is not None:
            found[coli] = fraction

    if quantiles is not None:
        wanted = {(q / 100 if q > 1 else q) for q in quantiles}
        epsilon = 1e-9
        found = {
            coli: value
            for coli, value in found.items()
            if any(abs(value - w) < epsilon for w in wanted)
        }

    return found


def numeric_x_values(x_columns: list[str]) -> dict[str, float] | None:
    """
    {column: x position} if every one of x_columns is itself numeric,
    directly or as a quantile (e.g. "q10" -> 10, "q0_1" -> 10, "2016" ->
    2016 - a wide-by-year table plots on a real year axis instead of
    0,1,2) - None if even one column isn't, so the caller can fall back
    to something else (e.g. positional order) instead of half-numeric,
    half-guessed x positions.
    """
    values = {}
    for c in x_columns:
        fraction = _quantile_fraction(c)
        if fraction is not None:
            values[c] = fraction * 100
            continue
        try:
            values[c] = float(c)
        except ValueError:
            return None
    return values


def long_frame(
    stat_item,
    value_columns: list[str],
    ci_level: float | None = None,
    filter_expr: nw.Expr | None = None,
    round_output: bool = True,
    variable_name: str = "Column",
    value_name: str = "Value",
) -> dict[str, pl.DataFrame]:
    """
    Long-format {"estimates": df, ["ci": df]} for the given
    value_columns of stat_item.df_estimates (and, if ci_level is given,
    stat_item._df_ci(ci_level)) - index columns (stat_index_columns)
    kept as-is, value_columns unpivoted into variable_name/value_name.
    Rounded per Census DRB disclosure rules (drb_round_table) before
    reshaping unless round_output=False.
    """
    index_cols = stat_index_columns(stat_item)

    frames = {"estimates": nw.from_native(stat_item.df_estimates).lazy()}
    if ci_level is not None:
        frames["ci"] = nw.from_native(stat_item._df_ci(ci_level=ci_level)).lazy()

    out = {}
    for key, dfi in frames.items():
        if filter_expr is not None:
            dfi = dfi.filter(filter_expr)
        dfi = dfi.select(index_cols + value_columns)
        if round_output:
            dfi = nw.from_native(
                drb_round_table(dfi.to_native(), columns_exclude=index_cols)
            ).lazy()
        dfi = dfi.unpivot(
            on=value_columns,
            index=index_cols,
            variable_name=variable_name,
            value_name=value_name,
        )
        out[key] = dfi.collect().to_native()

    return out


def series_label(
    df: pl.DataFrame, columns: list[str], sep: str = " / "
) -> pl.DataFrame:
    """
    Add a "___series___" column joining `columns` together as a single
    string - the legend/group identity when more than one index column
    (e.g. a Variable id plus a by-group) needs to collapse to one label.
    A single column is used as-is (cast to string, no join needed).
    """
    if len(columns) == 1:
        expr = pl.col(columns[0]).cast(pl.String)
    else:
        expr = pl.concat_str(
            [pl.col(c).cast(pl.String) for c in columns], separator=sep
        )
    return df.with_columns(expr.alias("___series___"))


def apply_rename(df: pl.DataFrame, column: str, rename: dict[str, str]) -> pl.DataFrame:
    """Rename values in `column` per `rename` ({old: new}); values not in `rename` pass through unchanged."""
    when = pl
    for old, new in rename.items():
        when = when.when(pl.col(column) == old).then(pl.lit(new))
    return df.with_columns(when.otherwise(pl.col(column)).alias(column))


def sort_to_order(df: pl.DataFrame, column: str, order: list[str]) -> pl.DataFrame:
    """Filter `df` to only rows whose `column` value is in `order`, sorted to match that order."""
    var_index = "___sort_order___"
    when = pl
    for i, value in enumerate(order):
        when = when.when(pl.col(column) == value).then(pl.lit(i))
    df = df.with_columns(when.otherwise(None).alias(var_index))
    return df.filter(pl.col(var_index).is_not_null()).sort(var_index).drop(var_index)
