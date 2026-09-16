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
    "stacked_bar() answers a different question than line()/coefplot(): given "
    "several StatCalculator/MultipleImputation objects that all share the same "
    "category axis (e.g. 'which safety-net program was removed'), how do their "
    "contributions add up? Each dict entry becomes one layer of the stack (e.g. "
    "one age group's share of the total poverty-count impact)."
)

set_seed(20260915)
programs = ["no_ss", "no_snap", "no_ctc", "no_housing"]
age_groups = ["Under 18", "18 to 64", "65+"]

# %%
logger.info(
    "One shared dataset, split by age group - 'Overall' is the StatCalculator "
    "over the whole thing, and each age group is the StatCalculator over its "
    "own filtered slice. Since 'sum' is additive over a partition of rows, "
    "Overall's total is guaranteed to equal the sum of the three age groups' "
    "own sums (unlike building each one from independent, unrelated data,  "
    "which would leave the 'total' label matching nothing on the chart)."
)
n_rows = 3_800
df = (
    RandomData(n_rows=n_rows, seed=generate_seed())
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

for name, item in items.items():
    logger.info(name)
    item.print()

# %%
logger.info(
    "total_key excludes that entry from the stack and instead shows its own "
    "value as a text label next to each fully-stacked bar - here it lands "
    "right at the tip of each bar, since the age-group layers exactly sum "
    "to Overall by construction."
)
rename = {f"impact_{p}": p for p in programs}
fig_stacked_bar = plot.stacked_bar(
    items,
    column="sum",
    total_key="Overall",
    rename=rename,
    order=list(rename.values()),
    label_round_digits=0,
    x_axis_title="Change in number of people in poverty",
)
fig_stacked_bar.write_html(
    os.path.join(path_docs_figures, "stacked_bar.html"), include_plotlyjs="directory"
)
