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
    "combine.py's tutorial leaves are all single-group quantiles() figures, "
    "so their own internal 'Show:' dropdown (see add_group_dropdown) never "
    "actually renders - only combine()'s outer table is visible. This file "
    "builds a leaf that HAS more than one group, so its own floating dropdown "
    "shows up too, to see how the two currently coexist before any work to "
    "merge a leaf's own dropdown into combine()'s outer table."
)

set_seed(20260915)
n_rows = 1_500
n_replicates = 8

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
replicates = Replicates(weight_stub="weight_", n_replicates=n_replicates, bootstrap=True)
sc = StatCalculator(
    df,
    statistics=stats,
    weight="weight_0",
    replicates=replicates,
    by={"year": ["year"]},
)

# %%
logger.info(
    "group_by=':' + a rename splits each of the 3 year-series into its own "
    "group ('2016'/'2017'/'2018'), same pattern as line_and_quantiles.py's "
    "'grouped' example but with 3 groups instead of 2 - enough for "
    "add_group_dropdown to actually draw its own 'Show:' widget (it only "
    "renders when there's more than one group)."
)
fig_multi_group = plot.quantiles(
    sc,
    rename={
        "income / 2016.0": "2016:income",
        "income / 2017.0": "2017:income",
        "income / 2018.0": "2018:income",
    },
    group_by=":",
    group_first=True,
)
logger.info(f"Groups on this leaf: {fig_multi_group._group_map}")

# %%
logger.info(
    "A single-leaf tree is enough to see it - open combine_with_leaf_dropdown.html "
    "and look at the top-left of the figure itself: a second, differently-styled "
    "'Show:' dropdown floats there (from add_group_dropdown, positioned inside "
    "the plot's own top margin), separate from combine()'s outer 'View:' table "
    "above it. Two dropdown widgets, two different look-and-feels, is exactly "
    "the inconsistency to fix by merging a leaf's own group dropdown into "
    "combine()'s outer table when the leaf is shown inside combine() instead of "
    "standalone."
)
tree = {"Only leaf": fig_multi_group}
combined = plot.combine(tree, label="View:", layout={"margin": {"t": 30}})
combined.write_html(os.path.join(path_docs_figures, "combine_with_leaf_dropdown.html"))
logger.info(os.path.join(path_docs_figures, "combine_with_leaf_dropdown.html"))
