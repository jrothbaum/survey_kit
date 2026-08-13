import narwhals as nw

from survey_kit.utilities.random import RandomData

from survey_kit.utilities.dataframe import join_list, join_wrapper

n_rows = 100
df = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_int8", 0, 10)
    .boolean("v_bool")
    .float("v_float", -1, 1)
    .to_df()
)

df_2 = (
    RandomData(n_rows=n_rows, seed=23421234)
    .index("index")
    .integer("v_int8", 0, 10)
    .boolean("v_bool")
    .float("v_float", -1, 1)
    .to_df()
)


df_3 = (
    RandomData(n_rows=n_rows, seed=8923)
    .index("index")
    .integer("v_int8", 0, 10)
    .boolean("v_bool")
    .float("v_float", -1, 1)
    .to_df()
)


df_polars = join_wrapper(df=df, df_to=df_2, how="left", on=["index"])
print(df.lazy().collect())


df_pandas = join_wrapper(
    df=df.lazy().collect().to_pandas(),
    df_to=df_2.lazy().collect().to_pandas(),
    how="left",
    on=["index"],
)
print(df_pandas)


df_polars_3 = join_list(
    [df, df_2, df_3],
    on=["index"],
    how="left",
    prefixes=["", "p2_", "p3_"],
    suffixes=["", "_2", "_3"],
)
print(df_polars_3.lazy().collect())
print(df_polars_3.lazy().collect_schema().names())


df_pandas_3 = join_list(
    [df.to_pandas(), df_2.to_pandas(), df_3.to_pandas()],
    on=["index"],
    how="left",
    prefixes=["", "p2_", "p3_"],
    suffixes=["", "_2", "_3"],
)
print(df_pandas_3)
print(df_pandas_3.columns)


df_duckdb = nw.from_native(df.to_arrow()).lazy(backend="duckdb")
df_duckdb_2 = nw.from_native(df_2.to_arrow()).lazy(backend="duckdb")

df_duckdb_joined = join_wrapper(
    df=df_duckdb, df_to=df_duckdb_2, how="left", on=["index"]
).to_native()
print(df_duckdb_joined)
print(type(df_duckdb_joined))
print(nw.from_native(df_duckdb_joined).collect_schema().names())
