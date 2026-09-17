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
    "combine() puts several whole figures (from line()/quantiles()/coefplot()/ "
    "stacked_bar(), or any plotly Figure) into one HTML page, switched between "
    "by a dropdown per nesting level - as many levels deep as the dict you "
    "pass it. Picking a value at a deeper level is remembered when you switch "
    "a shallower one, and falls back to that branch's first option if the "
    "previous choice doesn't exist there. Each leaf figure's own internal "
    "group dropdown (see line_and_quantiles.py/coefplot.py) keeps working "
    "independently once it's shown - combine() only adds this outer layer."
)

set_seed(20260915)
n_rows = 1_500
n_replicates = 8


def make_quantiles_sc(seed: int) -> StatCalculator:
    df = (
        RandomData(n_rows=n_rows, seed=seed)
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
    return StatCalculator(
        df,
        statistics=stats,
        weight="weight_0",
        replicates=replicates,
        by={"year": ["year"]},
    )


# %%
logger.info(
    "Two independent StatCalculators (as if from two different runs/vintages) "
    "give us something worth switching between at the top level; each is "
    "plotted twice (with and without a confidence band), giving a 3-level "
    "tree: Run -> CI -> (nothing further, since quantiles() already puts every "
    "year on one figure)."
)
sc_run_a = make_quantiles_sc(generate_seed())
sc_run_b = make_quantiles_sc(generate_seed())

tree = {
    "Run A": {
        "With CI": plot.quantiles(sc_run_a, ci_level=0.95, ci_area=True),
        "No CI": plot.quantiles(sc_run_a),
    },
    "Run B": {
        "With CI": plot.quantiles(sc_run_b, ci_level=0.95, ci_area=True),
        "No CI": plot.quantiles(sc_run_b),
    },
}

# %%
logger.info(
    "Every leaf here already has its own (inert, single-group) dropdown from "
    "quantiles() - combine() adds the 'Run A'/'Run B' and 'With CI'/'No CI' "
    "switches on top, in one page. layout={'margin': {'t': 30}} trims Plotly's "
    "default top margin (~100px of otherwise-blank space above the plot, "
    "reserved for a title none of these figures use) - a stopgap until "
    "quantiles()/line() default to something less white-spacey on their own."
)
combined = plot.combine(tree, label=["Run:", "CI:"], layout={"margin": {"t": 30}})
combined.write_html(os.path.join(path_docs_figures, "combine.html"))

# %%
logger.info(
    "Nesting isn't limited to 2 levels - here's a 3-level tree (Run -> CI -> "
    "Year), built by giving quantiles() a `filter_expr` so each leaf covers "
    "just one year instead of all three. Switching 'Run' while you're looking "
    "at, say, 2017 keeps 2017 selected on the other run too, as long as 2017 "
    "exists there (it does here, since both runs cover the same years)."
)


def year_branches(
    sc: StatCalculator, ci_level: float | None, ci_area: bool = False
) -> dict:
    return {
        str(year): plot.quantiles(
            sc, ci_level=ci_level, ci_area=ci_area, filter_expr=nw.col("year") == year
        )
        for year in [2016, 2017, 2018]
    }


tree_deep = {
    "Run A": {
        "With CI": year_branches(sc_run_a, ci_level=0.95, ci_area=True),
        "No CI": year_branches(sc_run_a, ci_level=None),
    },
    "Run B": {
        "With CI": year_branches(sc_run_b, ci_level=0.95, ci_area=True),
        "No CI": year_branches(sc_run_b, ci_level=None),
    },
}

combined_deep = plot.combine(
    tree_deep,
    label=["Run:", "\nCI:", "\nYear:"],
    dropdowns_padding_left=50,
    layout={"margin": {"t": 30}},
)
combined_deep.write_html(os.path.join(path_docs_figures, "combine_3_levels.html"))
logger.info(os.path.join(path_docs_figures, "combine_3_levels.html"))
