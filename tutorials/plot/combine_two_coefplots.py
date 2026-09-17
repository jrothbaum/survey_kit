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
    "coefplot() has no group dropdown of its own (unlike line()/quantiles() - "
    "see add_group_dropdown) and never calls add_group_dropdown, so its own "
    "legend-click toggle/isolate is plain, unwired Plotly - each figure's "
    "own legend state, not the shared name-keyed state that lets a "
    "line()/quantiles() sibling pick up the same isolate pattern when "
    "combine() switches to it. Two coefplots with the SAME series names "
    "(here: year) under two different combine() branches is the test - "
    "isolate '2018' in Region A, switch to Region B, switch back: does "
    "Region A still show only 2018, or did switching reset it?"
)

set_seed(20260915)
n_rows = 3_000
n_replicates = 8
subgroups = ["Male", "Female", "White", "Black"]
regions = ["Region A", "Region B"]


def make_coefplot(region: str, seed: int):
    df = (
        RandomData(n_rows=n_rows, seed=seed)
        .index("index")
        .integer("poverty", 0, 1)
        .integer("subgroup_id", 0, len(subgroups) - 1)
        .integer("year", 2016, 2018)
    ).to_df()

    when = pl
    for i, label in enumerate(subgroups):
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
    return plot.coefplot(
        sc,
        column="mean",
        ci_level=0.95,
        category="subgroup",
        series="year",
        x_axis_title=f"Poverty rate, {region}",
    )


tree = {region: make_coefplot(region, generate_seed()) for region in regions}

# %%
logger.info(
    "Both figures have the same trace names (2016/2017/2018 from series="
    "'year') - if you toggle/isolate one via the legend on Region A, switch "
    "to Region B, then back to Region A, check whether Region A's own "
    "toggle state survived (it should - it's the same DOM element, just "
    "hidden/shown - the open question is only whether Region B ALSO picked "
    "up the same isolate pattern when you first switched to it, the way a "
    "quantiles() sibling would)."
)
combined = plot.combine(tree, label="Region:", layout={"margin": {"t": 30}})
combined.write_html(os.path.join(path_docs_figures, "combine_two_coefplots.html"))
logger.info(os.path.join(path_docs_figures, "combine_two_coefplots.html"))
