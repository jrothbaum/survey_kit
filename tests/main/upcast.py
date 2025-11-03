from survey_kit.utilities.dataframe import safe_upcast_list
import polars as pl

from survey_kit.utilities.random import RandomData


n_rows = 100
df = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_int8", 0, 10)
    .boolean("v_bool")
    .float("v_float", -1, 1)
    .to_df()
)

df_bigger = df.with_columns(
    pl.col("v_int8").cast(pl.Int16), pl.col("v_bool").cast(pl.Int8)
)

df_biggest = df.with_columns(
    pl.col("v_int8").cast(pl.Int32),
    pl.col("v_bool").cast(pl.Int16),
    pl.col("v_bool").alias("v_bool_2"),
)

df_smaller = df.with_columns(
    pl.col("v_float").cast(pl.Float32), pl.col("v_int8").cast(pl.Int64).alias("v_int_2")
)

inputs = [df, df_bigger, df_biggest, df_smaller]
outputs = safe_upcast_list(inputs)
for i in range(len(inputs)):
    print(f"before:     {inputs[i].schema}")
    print(f"after:      {outputs[i].schema}")

    assert outputs[i].schema["v_int8"] == pl.Int32
    assert outputs[i].schema["v_bool"] == pl.Int16
    assert outputs[i].schema["v_float"] == pl.Float64
