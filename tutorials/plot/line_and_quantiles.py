import os
import polars as pl

from survey_kit.utilities.random import RandomData, set_seed, generate_seed
from survey_kit.statistics.calculator import StatCalculator
from survey_kit.statistics.statistics import Statistics
from survey_kit.statistics.replicates import Replicates
from survey_kit.statistics.bootstrap import bayes_bootstrap
from survey_kit import logger, config, plot

path_docs_plot = os.path.join(config.code_root, "..", "..", "docs", "tutorials", "plot")
path_docs_figures = os.path.join(path_docs_plot, "figures")
os.makedirs(path_docs_figures, exist_ok=True)

# %%
logger.info(
    "survey_kit.plot builds plotly figures directly from a StatCalculator's or "
    "MultipleImputation's own df_estimates/_df_ci - there's no separate "
    "reshaping step to do yourself. line()/quantiles() plot a set of stat "
    "columns (e.g. several quantiles) against each other, one line per index "
    "value or by-group."
)

set_seed(20260915)
n_rows = 2_000
n_replicates = 10

df = (
    RandomData(n_rows=n_rows, seed=generate_seed())
    .index("index")
    .integer("income", 0, 100_000)
    .integer("year", 2016, 2018)
).to_df()

df = pl.concat(
    [
        df,
        bayes_bootstrap(
            n_rows=n_rows,
            n_draws=n_replicates + 1,
            seed=generate_seed(),
            initial_weight_index=0,
            prefix="weight_",
        ),
    ],
    how="horizontal",
)

stats = Statistics(stats=["q10", "q25", "q50", "q75", "q90"], columns=["income"])
replicates = Replicates(
    weight_stub="weight_", n_replicates=n_replicates, bootstrap=True
)

sc = StatCalculator(
    df,
    statistics=stats,
    weight="weight_0",
    replicates=replicates,
    by={"year": ["year"]},
)
sc.print()

# %%
logger.info(
    "quantiles() is a thin wrapper around line() - it finds every quantile-stat "
    "column on its own (q10, q25, ...) and plots them across percentile 0-100. "
    "With more than one index column (here: Variable + year), each unique "
    "combination becomes its own line - one per year, since there's only one "
    "Variable (income) in this example."
)
fig_quantiles = plot.quantiles(sc)
fig_quantiles.write_html(
    os.path.join(path_docs_figures, "quantiles.html"), include_plotlyjs="directory"
)

# %%
logger.info(
    "The legend itself is always interactive, whether or not a figure has a "
    "group dropdown (see below): single-click a legend entry to toggle just "
    "that line on/off; double-click one to isolate it, hiding every other "
    "line at once (double-click again, or on another entry, to bring the "
    "rest back)."
)

# %%
logger.info(
    "Add confidence intervals with ci_level - either as error bars (the "
    "default) or, with ci_area=True, a shaded band around each line."
)
fig_quantiles_ci = plot.quantiles(sc, ci_level=0.95, ci_area=True)
fig_quantiles_ci.write_html(
    os.path.join(path_docs_figures, "quantiles_ci_area.html"), include_plotlyjs="directory"
)

# %%
logger.info(
    "Every line()/quantiles() figure carries a dropdown for switching which "
    "group of lines is shown - by default there's a single group (everything "
    "shown at once, same as a plain legend, and no extra JavaScript at all, "
    "since there's nothing to switch between). group_by splits each line's "
    "label on a separator and groups lines that share the same prefix - handy "
    "when several related series (e.g. two 'historical' years vs. a 'recent' "
    "one) should travel together in the dropdown instead of each getting its "
    "own entry."
)
fig_grouped = plot.quantiles(
    sc,
    rename={
        "income / 2016.0": "Historical:2016",
        "income / 2017.0": "Historical:2017",
        "income / 2018.0": "Recent:2018",
    },
    group_by=":",
    group_first=True,
)
logger.info(f"Groups: {fig_grouped._group_map}")
fig_grouped.write_html(
    os.path.join(path_docs_figures, "quantiles_grouped.html"), include_plotlyjs="directory"
)

# %%
logger.info(
    "line() itself is more general than quantiles() - point it at any set of "
    "df_estimates columns you want plotted against each other. Here we only "
    "plot 3 of the 5 quantile columns, in an explicit order (also filters to "
    "just those series)."
)
fig_line = plot.line(
    sc,
    x_columns=["q10", "q50", "q90"],
    x_axis_title="Percentile",
    y_axis_title="Income",
    order=["income / 2018.0", "income / 2017.0", "income / 2016.0"],
)
fig_line.write_html(
    os.path.join(path_docs_figures, "line_selected_quantiles.html"),
    include_plotlyjs="directory",
)
