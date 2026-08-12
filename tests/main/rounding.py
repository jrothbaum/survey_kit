from survey_kit.utilities.random import RandomData
from survey_kit.utilities.rounding import drb_round_table


n_rows = 100
df = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_int8", 0, 100000)
    .boolean("v_bool")
    .float("v_float", -1, 1)
    .integer("n", 0, 1_000_000)
    .integer("n_small", 0, 100)
    .to_df()
)

df_rounded = drb_round_table(df)
print(df_rounded.lazy().collect())


df_rounded = drb_round_table(df, columns_n=["n", "n_small"])
print(df_rounded.lazy().collect())


df_rounded = drb_round_table(df.lazy(), columns_n=["n", "n_small"])
print(df_rounded.lazy().collect())
