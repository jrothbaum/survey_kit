import narwhals as nw
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
    # .boolean("v_bool")
    # .float("v_f_continuous", -1, 1)
    # .float("v_f_scale", -1, 1)
    # .float("v_f_center", -1, 1)
    # .float("v_extra", -1, 1)
    # #   .np_distribution("weight_0", "normal", loc=1, scale=1)
    .integer("weight_0", 100, 1_000_000)
    # .np_distribution("weight_1", "normal", loc=1, scale=1)
    .integer("year", 2016, 2021)
    # .integer("month", 1, 12)
    .integer("income", 0, 100_000)
    .to_df()
    .lazy()
)

df = df.with_columns(
    pl.when(pl.col("year").ne(2016)).then(pl.col("income")).otherwise(pl.lit(0))
)
# print(df.schema)
# print(df.describe())

summary(df, weight="weight_0")

c_weight = pl.col("weight_0")
cols = ["v_1", "income", "year"]

df = safe_sum_cast(df, columns=["weight_0"])
print(
    df.select(
        [(pl.col(coli) * c_weight).sum() / c_weight.sum() for coli in cols]
    ).collect()
)

if True:
    if test_calculate_by:
        d_polars = calculate_by(
            df=df,  # .collect().to_pandas(),
            column_stats={
                "v_1": ["mean", "sum"],
                "income|not0": ["median", "q10", "q90", "count", "gini"],
                "v_1|share": ["mean", "sum"],
                "v_bool": ["mean"],
            },
            weight="weight_0",
            by=dict(all=[], year=["year"]),
            quantile_interpolated=True,
        )
        d_pandas = calculate_by(
            df=df.collect().to_pandas(),
            column_stats={
                "v_1": ["mean", "sum"],
                "income|not0": ["median", "q10", "q90", "count", "gini"],
                "v_1|share": ["mean", "sum"],
                "v_bool": ["mean"],
            },
            weight="weight_0",
            by=dict(all=[], year=["year"]),
            quantile_interpolated=True,
        )

        print(d_polars)
        print(d_pandas)

    if test_statistics:
        stats = Statistics(stats=["mean", "median|not0"], columns=["v_1", "income"])
        df_out = stats.calculate(df, weight="weight_0")

        df_duckdb = stats.calculate(
            nw.from_native(df.collect().to_arrow()).lazy(backend="duckdb"),
            weight="weight_0",
        )

        df_pandas = stats.calculate(df.lazy().collect().to_pandas(), weight="weight_0")

        print(df_out)
        print(type(df_out))

        print(df_pandas)
        print(type(df_pandas))

        print(df_duckdb)
        print(type(df_duckdb))

    # d_polars = calculate_by(
    #     df=df, #.collect().to_pandas(),
    #     column_stats={"v_1":["mean"]},
    #     weight="weight_0",
    #     by=dict(all=[],
    #             year=["year"]),
    #     no_suffix=True
    # )

    # print(d_polars)`
