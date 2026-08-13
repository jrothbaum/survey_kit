import narwhals as nw

from survey_kit.utilities.random import RandomData
from survey_kit.utilities.compress import compress_df
from survey_kit.utilities.dataframe import safe_height, NarwhalsType

n_rows = 100
df = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_int8", 0, 100000)
    .boolean("v_bool")
    .float("v_float", -1, 1)
    .integer("n", 0, 1_000_000)
    .integer("n_small", 0, 100)
    .to_df(compress=False)
)
nw_type = NarwhalsType(df)
print(df.lazy().collect_schema())
df_compressed = compress_df(df).lazy()
print(df_compressed.collect_schema())


df_rounded = compress_df(df.lazy().collect().to_pandas())
print(nw.from_native(df_rounded).lazy().collect_schema())

print(safe_height(df))
print(safe_height(df.lazy()))
print(safe_height(df.to_pandas()))

print(df)
