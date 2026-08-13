import polars as pl

from survey_kit import logger

from survey_kit.statistics.bootstrap import bayes_bootstrap_weights

from survey_kit.utilities.random import RandomData, generate_seed, set_seed
from survey_kit.statistics.calculator import StatCalculator
from survey_kit.statistics.statistics import Statistics
from survey_kit.statistics.replicates import Replicates

n_rows = 1_000
n_replicates = 10


def gen_random_table(n_rows: int, n_replicates: int, seed: int):
    set_seed(seed)
    df = (
        RandomData(n_rows=n_rows, seed=int(generate_seed()))
        .index("index")
        .integer("v_1", 0, 10)
        .boolean("v_bool")
        .float("v_f_continuous", -1, 1)
        .float("v_f_scale", -1, 1)
        .float("v_f_center", -1, 1)
        .float("v_extra", -1, 1)
        .integer("year", 2016, 2021)
        .integer("month", 1, 12)
        .integer("income", 0, 100_000)
        .np_distribution("weight_0", "normal", loc=100, scale=5)
    )

    c_w = pl.col("weight_0")
    df = (
        df.to_df()
        .with_columns(pl.when(c_w.gt(0)).then(c_w).otherwise(pl.lit(0)))
        .with_columns(c_w / c_w.sum() * n_rows)
        .lazy()
    )

    df = bayes_bootstrap_weights(
        df=df,
        weight="weight_0",
        prefix="weight_",
        n_replicates=n_replicates,
        seed=int(generate_seed()),
        sum_to=n_rows,
    )

    df = df.with_columns(
        pl.when(pl.col("year").ne(2016)).then(pl.col("income")).otherwise(pl.lit(0))
    )

    return df


df = gen_random_table(n_rows, n_replicates, seed=1230)
df_compare = gen_random_table(n_rows, n_replicates, seed=9324)


replicates = Replicates(
    weight_stub="weight_",
    df=df,
    #   n_replicates=10,
    bootstrap=True,
)

sc = StatCalculator(
    df,
    statistics=Statistics(stats=["mean", "median|not0"], columns=["v_1", "income"]),
    weight="weight_0",
    replicates=replicates,
    by=dict(year=["year"]),
)

sc = StatCalculator(
    df.collect().to_pandas(),
    statistics=Statistics(stats=["mean", "median|not0"], columns=["v_1", "income"]),
    weight="weight_0",
    replicates=replicates,
    by=dict(year=["year"]),
)

logger.info(type(sc.df_estimates))
