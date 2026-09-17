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
    "Same question as combine_two_coefplots.py, for stacked_bar() instead - "
    "it also never calls add_group_dropdown, so its legend is plain, "
    "unwired Plotly too. Two stacked bars with the SAME layer names (age "
    "groups) under two different combine() branches: isolate/hide a layer "
    "via the legend on Scenario A, switch to Scenario B and back, and see "
    "whether Scenario A's own state survived and whether Scenario B picked "
    "up the same pattern when first switched to."
)

set_seed(20260915)
programs = ["no_ss", "no_snap", "no_ctc", "no_housing"]
age_groups = ["Under 18", "18 to 64", "65+"]
scenarios = ["Scenario A", "Scenario B"]


def make_stacked_bar(scenario: str, seed: int):
    n_rows = 3_800
    df = (
        RandomData(n_rows=n_rows, seed=seed)
        .index("index")
        .float("impact_no_ss", -5, 0)
        .float("impact_no_snap", -3, 0)
        .float("impact_no_ctc", -2, 0)
        .float("impact_no_housing", -1, 0)
        .integer("age_group_id", 0, len(age_groups) - 1)
    ).to_df()

    when = pl
    for i, label in enumerate(age_groups):
        when = when.when(pl.col("age_group_id") == i).then(pl.lit(label))
    df = df.with_columns(when.otherwise(pl.lit("Other")).alias("age_group")).drop(
        "age_group_id"
    )
    df = pl.concat(
        [
            df,
            bayes_bootstrap(
                n_rows=n_rows,
                n_draws=9,
                seed=generate_seed(),
                initial_weight_index=0,
                prefix="weight_",
            ),
        ],
        how="horizontal",
    )

    stats = Statistics(stats=["sum"], columns=[f"impact_{p}" for p in programs])
    replicates = Replicates(weight_stub="weight_", n_replicates=8, bootstrap=True)
    items = {
        label: StatCalculator(
            df.filter(pl.col("age_group") == label),
            statistics=stats,
            weight="weight_0",
            replicates=replicates,
        )
        for label in age_groups
    }
    items["Overall"] = StatCalculator(
        df, statistics=stats, weight="weight_0", replicates=replicates
    )
    rename = {f"impact_{p}": p for p in programs}
    return plot.stacked_bar(
        items,
        column="sum",
        total_key="Overall",
        rename=rename,
        order=list(rename.values()),
        label_round_digits=0,
        x_axis_title=f"Change in number of people in poverty, {scenario}",
    )


tree = {
    scenario: make_stacked_bar(scenario, generate_seed()) for scenario in scenarios
}

# %%
logger.info(
    "Both figures have the same layer/trace names (no_ss/no_snap/no_ctc/"
    "no_housing) - same check as the coefplot version: does a legend "
    "toggle on Scenario A survive switching away and back, and does "
    "switching to Scenario B pick up the same pattern the way a "
    "quantiles() sibling would?"
)
combined = plot.combine(tree, label="Scenario:", layout={"margin": {"t": 30}})
combined.write_html(os.path.join(path_docs_figures, "combine_two_stacked_bars.html"))
logger.info(os.path.join(path_docs_figures, "combine_two_stacked_bars.html"))
