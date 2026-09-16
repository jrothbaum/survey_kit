import os
import narwhals as nw
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
    "coefplot() is the classic disclosure-review chart: one row per category "
    "(e.g. a demographic subgroup), a point with a confidence interval whisker "
    "for the estimate, and (optionally) several offset points per row when "
    "there's more than one series to compare (e.g. one color per year)."
)

set_seed(20260915)
n_rows = 3_000
n_replicates = 8
labels = ["Male", "Female", "White", "Black", "Under 18", "65+"]

df = (
    RandomData(n_rows=n_rows, seed=generate_seed())
    .index("index")
    .integer("poverty", 0, 1)
    .integer("subgroup_id", 0, len(labels) - 1)
    .integer("year", 2016, 2018)
).to_df()

when = pl
for i, label in enumerate(labels):
    when = when.when(pl.col("subgroup_id") == i).then(pl.lit(label))
df = df.with_columns(when.otherwise(pl.lit("Other")).alias("subgroup")).drop(
    "subgroup_id"
)

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

stats = Statistics(stats=["mean"], columns=["poverty"])
replicates = Replicates(
    weight_stub="weight_", n_replicates=n_replicates, bootstrap=True
)

sc = StatCalculator(
    df,
    statistics=stats,
    weight="weight_0",
    replicates=replicates,
    by={"group": ["subgroup", "year"]},
)
sc.print()

# %%
logger.info(
    "category is the row axis (subgroup here), series offsets multiple points "
    "per row (year). headers insert a bold divider row above a given "
    "category - handy for grouping related subgroups (e.g. every "
    "gender/race/age category under one section heading)."
)
fig_coefplot = plot.coefplot(
    sc,
    column="mean",
    ci_level=0.95,
    category="subgroup",
    series="year",
    headers={"Male": "Gender", "White": "Race", "Under 18": "Age"},
    x_axis_title="Poverty rate",
)
fig_coefplot.write_html(
    os.path.join(path_docs_figures, "coefplot.html"), include_plotlyjs="directory"
)

# %%
logger.info(
    "Without `series`, coefplot() draws a single unoffset point per category - "
    "useful for a simple one-estimate-per-row chart (e.g. just the most "
    "recent year)."
)
sc_2018 = sc.filter(nw.col("year") == 2018)
fig_coefplot_single = plot.coefplot(
    sc_2018,
    column="mean",
    ci_level=0.95,
    category="subgroup",
    headers={"Male": "Gender", "White": "Race", "Under 18": "Age"},
    x_axis_title="Poverty rate, 2018",
)
fig_coefplot_single.write_html(
    os.path.join(path_docs_figures, "coefplot_single_series.html"),
    include_plotlyjs="directory",
)
