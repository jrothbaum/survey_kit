import polars as pl
from survey_kit.utilities.random import RandomData
from survey_kit.statistics.basic_calculations import calculate_by
from survey_kit.statistics.statistics import Statistics
from survey_kit.utilities.dataframe import summary, safe_sum_cast

test_calculate_by = False
test_statistics = True

n_rows = 1_000
df = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_1", 0, 10)
    .integer("weight_0", 100, 1_000_000)
    .integer("year", 2016, 2021)
    .integer("income", 0, 100_000)
    .to_df()
    .lazy()
)

df = df.with_columns(
    pl.when(pl.col("year").ne(2016)).then(pl.col("income")).otherwise(pl.lit(0))
)

summary(df, weight="weight_0")

c_weight = pl.col("weight_0")
cols = ["v_1", "income", "year"]

df = safe_sum_cast(df, columns=["weight_0"])
print(
    df.select(
        [(pl.col(coli) * c_weight).sum() / c_weight.sum() for coli in cols]
    ).collect()
)

if test_calculate_by:
    d_polars = calculate_by(
        df=df,
        column_stats={
            "v_1": ["mean", "sum"],
            "income|not0": ["median", "q10", "q90", "count", "gini"],
            "v_1|share": ["mean", "sum"],
        },
        weight="weight_0",
        by=dict(all=[], year=["year"]),
        quantile_interpolated=True,
    )

    print(d_polars)

if test_statistics:
    stats = Statistics(stats=["mean", "median|not0"], columns=["v_1", "income"])
    df_out = stats.calculate(df, weight="weight_0")

    print(df_out)
    print(type(df_out))
