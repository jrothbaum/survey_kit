import polars as pl
from survey_kit.utilities.random import RandomData
from survey_kit.statistics.calculator import StatCalculator
from survey_kit.statistics.statistics import Statistics
from survey_kit.statistics.replicates import Replicates
from survey_kit.statistics.bootstrap import bayes_bootstrap
from survey_kit.utilities.random import set_seed, generate_seed

n_rows = 1_000
n_replicates = 10


def gen_random_table(n_rows: int, n_replicates: int, seed: int):
    set_seed(seed)

    df = (
        RandomData(n_rows=n_rows, seed=generate_seed())
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
    ).lazy()

    df = df.with_columns(
        pl.when(pl.col("year").ne(2016)).then(pl.col("income")).otherwise(pl.lit(0))
    )

    df = df.with_columns(pl.lit("a").alias("v_string"))
    return df


df = gen_random_table(n_rows, n_replicates, seed=1230)
df_compare = gen_random_table(n_rows, n_replicates, seed=9324)
# print(df.schema)
# print(df.describe())

replicates = Replicates(
    weight_stub="weight_", n_replicates=n_replicates, bootstrap=True
)


print("Polars")
sc = StatCalculator(
    df,
    statistics=Statistics(
        stats=["mean", "median|not0"], columns=["v_1", "income", "v_string"]
    ),
    weight="weight_0",
    replicates=replicates,
    by=dict(year=["year"]),
)


print("Pandas")
sc = StatCalculator(
    df.lazy().collect().to_pandas(),
    statistics=Statistics(
        stats=["mean", "median|not0", "median"], columns=["v_1", "income"]
    ),
    weight="weight_0",
    replicates=replicates,
    #   allow_slow_pandas=True,
    by=dict(year=["year"]),
)


sc_1 = StatCalculator(
    df,
    statistics=Statistics(stats=["mean"], columns=["v_1", "income"]),
    weight="weight_0",
    replicates=replicates,
    by=dict(year=["year"]),
)

sc_2 = StatCalculator(
    df_compare,
    statistics=Statistics(stats=["mean"], columns=["v_1", "income"]),
    weight="weight_0",
    replicates=replicates,
    by=dict(year=["year"]),
)

d_compare = sc_1.compare(sc_2)


sc_1 = StatCalculator(
    df.lazy().collect().to_pandas(),
    statistics=Statistics(stats=["mean"], columns=["v_1", "income"]),
    weight="weight_0",
    replicates=replicates,
    by=dict(year=["year"]),
)

sc_2 = StatCalculator(
    df_compare.lazy().collect().to_pandas(),
    statistics=Statistics(stats=["mean"], columns=["v_1", "income"]),
    weight="weight_0",
    replicates=replicates,
    by=dict(year=["year"]),
)

d_compare_pandas = sc_1.compare(sc_2)


d_compare["difference"].print()
d_compare_pandas["difference"].print()
