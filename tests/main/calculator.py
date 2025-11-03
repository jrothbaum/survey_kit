import polars as pl
from survey_kit.utilities.random import RandomData
from survey_kit.statistics.calculator import StatCalculator
from survey_kit.statistics.statistics import Statistics
from survey_kit.statistics.replicates import Replicates


n_rows = 1_000
n_replicates = 10


def gen_random_table(n_rows: int, n_replicates: int, seed: int):
    df = (
        RandomData(n_rows=n_rows, seed=seed)
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
    )

    for i in range(0, n_replicates + 1):
        df = df.np_distribution(f"weight_{i}", "normal", loc=1, scale=1)

    df = df.to_df().lazy()

    for i in range(0, n_replicates + 1):
        df = df.with_columns(
            pl.when(pl.col(f"weight_{i}") >= 0)
            .then(pl.col(f"weight_{i}"))
            .otherwise(pl.lit(0.0))
        )
    df = df.with_columns(
        pl.when(pl.col("year").ne(2016)).then(pl.col("income")).otherwise(pl.lit(0))
    )

    return df


df = gen_random_table(n_rows, n_replicates, seed=1230)
df_compare = gen_random_table(n_rows, n_replicates, seed=9324)
# print(df.schema)
# print(df.describe())

replicates = Replicates(weight_stub="weight_", n_replicates=n_replicates)


print("Polars")
sc = StatCalculator(
    df,
    statistics=Statistics(stats=["mean", "median|not0"], columns=["v_1", "income"]),
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
