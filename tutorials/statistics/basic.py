from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import summary
from survey_kit.statistics.statistics import Statistics
from survey_kit import logger

# %%
logger.info("Draw some random data")
n_rows = 1_000
df = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_int", 0, 10)
    .boolean("v_bool")
    .float("v_float", -1, 1)
    .integer("weight_0", 100, 1_000_000)
    .integer("year", 2016, 2018)
    .integer("quarter", 1, 4)
    .to_df()
    .lazy()
)

# %%
logger.info("The simplest option: just call summary(df)")
logger.info(
    "  Note the '_ =' is to prevent it from being printed twice in jupyter when generating the html, you can ignore"
)
_ = summary(df)

# %%
logger.info("\n\n + Weighted")
_ = summary(df, weight="weight_0")


# %%
logger.info("\n\n + by something")
_ = summary(df, weight="weight_0", by="year")
_ = summary(df, weight="weight_0", by=["quarter", "year"])


# %%
logger.info("\n\n with detailed stats and 4-sig digit rounding")
_ = summary(df, weight="weight_0", detailed=True, drb_round=True)


# %%
logger.info("\n\n with additional stats")
logger.info("What is available:")
logger.info(Statistics.available_stats())
_ = summary(
    df, weight="weight_0", additional_stats=["q10", "q95", "n|not0", "share|not0"]
)


logger.info("Get them (but no need to print):")

df_stats = summary(
    df,
    weight="weight_0",
    additional_stats=["q10", "q95", "n|not0", "share|not0"],
    print=False,
)

logger.info(df_stats.collect())
