from __future__ import annotations

import polars as pl
from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import summary
from survey_kit.statistics.calculator import StatCalculator, ComparisonItem
from survey_kit.statistics.statistics import Statistics
from survey_kit.statistics.replicates import Replicates
from survey_kit.statistics.bootstrap import bayes_bootstrap
from survey_kit.utilities.random import set_seed, generate_seed

from survey_kit import logger, config


# %%
# Draw some random data
def gen_random_table(n_rows: int, n_replicates: int, seed: int):
    set_seed(seed)
    df = (
        RandomData(n_rows=n_rows, seed=generate_seed())
        .index("index")
        .integer("v_1", 0, 10)
        .integer("year", 2016, 2021)
        .integer("month", 1, 12)
        .integer("income", 0, 100_000)
        .integer("income_2", 0, 120_000)
    ).to_df()

    #   Attach bootstrap weights to the data
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
    ).lazy()

    df = df.with_columns(
        pl.when(pl.col("year").ne(2016)).then(pl.col("income")).otherwise(pl.lit(0))
    )

    return df


n_rows = 1_000
n_replicates = 10

logger.info(
    "Create two datasets to compare, with bootstrap weights (or repliate weights)"
)
df = gen_random_table(n_rows, n_replicates, seed=1230)
df_compare = gen_random_table(n_rows, n_replicates, seed=9324)

# %%
logger.info("What's the data look like")
logger.info(df.head(10).collect())
logger.info(df_compare.head(10).collect())

# %%
logger.info("Use the stat calculator class to get stats")
logger.info("   The basic option, just get some stats")
stats = statistics = Statistics(
    stats=["mean", "median|not0"], columns=["v_1", "income", "income_2"]
)

logger.info("What is available:")
logger.info(Statistics.available_stats())

# %%
logger.info("Estimate them")
sc = StatCalculator(
    df,
    statistics=stats,
    weight="weight_0",
)


# %%
logger.info("   Want standard errors?")
logger.info("   Set up 'Replicates'")
logger.info("       from data (with df)")
logger.info("       NOTE:   if pass 'bootstrap=True', it will calculate ")
logger.info(
    "               regular bootstrap SEs rather than balanced repeated replication SEs"
)
logger.info("               (if you don't know what that means, pass bootstrap=True)")
replicates_from_df = Replicates(df=df, weight_stub="weight_", bootstrap=True)

logger.info("       or just tell it the number (n_replicates)")
replicates = Replicates(
    weight_stub="weight_", n_replicates=n_replicates, bootstrap=True
)
logger.info("       Same either way")
logger.info(replicates_from_df.rep_list)
logger.info(replicates.rep_list)
del replicates_from_df

# %%
logger.info("\n\nNow let's calculate the stats with replicate-weighted standard errors")
logger.info("   for df")
logger.info("   (where each estimate has the SE below the estimate)")
sc = StatCalculator(df, statistics=stats, weight="weight_0", replicates=replicates)

logger.info("   for df_compare")
sc_compare = StatCalculator(
    df_compare, statistics=stats, weight="weight_0", replicates=replicates
)


logger.info("Save them for later")
sc.save(f"{config.path_temp_files}/sc")
sc_compare.save(f"{config.path_temp_files}/sc_compare")

del sc
del sc_compare


# %%
logger.info("Load them back up")
sc = StatCalculator.load(f"{config.path_temp_files}/sc")
sc_compare = StatCalculator.load(f"{config.path_temp_files}/sc_compare")


# %%
logger.info("\n\nWe can compare them easily")
d_comparisons = sc.compare(sc_compare)

logger.info(
    "Which gives us a dictionary of comparisons, with keys=['difference','ratio']"
)
logger.info(d_comparisons["difference"])
logger.info(d_comparisons["ratio"])


# %%
logger.info(
    "\n\nLet's just get the difference between the means to medians, within the same data"
)
logger.info("   For df")
sc_mean_median = sc.compare(
    sc,
    ratio=False,
    compare_list_columns=[
        ComparisonItem.Column("mean", "median (not 0)", name="median_mean")
    ],
)["difference"]

logger.info("   For df_compare")
sc_compare_mean_median = sc_compare.compare(
    sc_compare,
    ratio=False,
    compare_list_columns=[
        ComparisonItem.Column("mean", "median (not 0)", name="median_mean")
    ],
)["difference"]

# %%
logger.info(
    "\n\nNow, we can use the same tool to get a difference in difference comparison"
)
logger.info(
    "   i.e. [df(median|not0) - df(mean)] - [df_compare(median|not0) - df_compare(mean)]"
)
d_diff_in_diffs = sc_mean_median.compare(sc_compare_mean_median, ratio=False)


# %%
logger.info("\n\nLet's get the ratio two variables")
logger.info(
    "   rather than the difference between two statistics for the same variable)"
)

logger.info("   For df")
sc_inc_inc_2 = sc.compare(
    sc,
    difference=False,
    compare_list_variables=[
        ComparisonItem.Variable(value1="income", value2="income_2", name="income_comp")
    ],
)["ratio"]

logger.info("   For df_compare")
sc_inc_inc_2_compare = sc_compare.compare(
    sc_compare,
    difference=False,
    compare_list_variables=[
        ComparisonItem.Variable(value1="income", value2="income_2", name="income_comp")
    ],
)["ratio"]


# %%
logger.info("\n\nAnd for fun, the diff-in-diff")
logger.info(
    "   i.e. [df(income_2)/df(income)] - [df_compare(income_2)/df_compare(income)]"
)

d_diff_in_diffs_2 = sc_inc_inc_2.compare(sc_inc_inc_2_compare, ratio=False)
