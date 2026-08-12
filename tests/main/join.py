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


df_polars_3 = join_list(
    [df, df_2, df_3],
    on=["index"],
    how="left",
    prefixes=["", "p2_", "p3_"],
    suffixes=["", "_2", "_3"],
)
print(df_polars_3.lazy().collect())
print(df_polars_3.lazy().collect_schema().names())
