import polars as pl
import narwhals as nw
from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import summary, columns_from_list, NarwhalsType

from survey_kit.utilities.formula_builder import FormulaBuilder
from survey_kit.imputation.utilities.lightgbm_wrapper import (
    Survey_kit_Lightgbm as lightgbm_kit,
    Tuner,
    Tuner_optuna,
)


n_rows = 100_000

to_pandas = True
to_duckdb = False

as_formula = False
with_tuning = True


df = (
    RandomData(n_rows=n_rows, seed=32565437)
    .index("index")
    .float("x_1", -1, 1)
    .float("x_2", -1, 1)
    .float("x_3", -1, 1)
    .float("x_4", -1, 1)
    .float("x_5", -1, 1)
    .integer("x_cat", 0, 10)
    .float("z_1", -1, 1)
    .float("z_2", -1, 1)
    .np_distribution("e", "normal", loc=0, scale=4)
    .to_df()
)

df_predict = (
    RandomData(n_rows=n_rows, seed=437201)
    .index("index")
    .float("x_1", -1, 1)
    .float("x_2", -1, 1)
    .float("x_3", -1, 1)
    .float("x_4", -1, 1)
    .float("x_5", -1, 1)
    .integer("x_cat", 0, 10)
    .float("z_1", -1, 1)
    .float("z_2", -1, 1)
    .np_distribution("e", "normal", loc=0, scale=4)
    .to_df()
)

c_cat = pl.col("x_cat")
df = df.with_columns(
    (
        pl.col("x_1")
        - 2 * pl.col("x_2")
        + 3 * pl.col("x_3")
        - 5 * pl.col("x_4")
        + 0.01 * pl.col("x_5")
        + 5 * c_cat.eq(0)
        + 2 * c_cat.eq(1)
        + 3 * c_cat.eq(2)
        + -1 * c_cat.eq(3)
        + 4 * c_cat.eq(4)
        + -20 * c_cat.eq(5)
        + 3 * c_cat.eq(6)
        + 11 * c_cat.eq(7)
        + 10 * c_cat.eq(8)
        + 5 * c_cat.eq(9)
        + pl.col("e")
    ).alias("y")
)

if to_pandas:
    df = df.to_pandas()
    df_predict = df_predict.to_pandas()
elif to_duckdb:
    df = nw.from_native(df).lazy().lazy_backend(NarwhalsType(backend="duckdb"))
    df_predict = (
        nw.from_native(df_predict).lazy().lazy_backend(NarwhalsType(backend="duckdb"))
    )
summary(df)

fb = FormulaBuilder(df=df)
fb.continuous(columns=["x_*", "z_*"])
fb.factor(columns=["x_cat"])
fb.simple_interaction(columns=["x_1", "x_2"])
print(fb.formula)


tuner = Tuner_optuna(n_trials=5, objective=Tuner.Objectives.sse)
tuner.parameters()
tuner.hyperparameters["num_leaves"] = [2, 256]
tuner.hyperparameters["max_depth"] = [2, 256]
# tuner.hyperparameters["min_data_in_leaf"] = [10,250]
tuner.hyperparameters["num_iterations"] = [25, 500]

if as_formula:
    if with_tuning:
        parameters = dict(test_size=0.5)
    else:
        parameters = {}

    lgbm = lightgbm_kit(
        df=df, y="y", formula=fb.formula, tuner=tuner, parameters=parameters
    )
else:
    parameters = dict(
        categorical_feature=["x_cat"],
    )
    if with_tuning:
        parameters["test_size"] = 0.5

    lgbm = lightgbm_kit(
        df=df,
        y="y",
        formula=columns_from_list(df=df, columns=["x_*", "z_*"]),
        parameters=parameters,
        tuner=tuner,
    )


if with_tuning:
    lgbm.tune()
lgbm.train()
df_prediction = lgbm.predict(df_predict=df_predict, merged_to_input=True)
summary(df_prediction)
print(nw.from_native(lgbm.importance(with_rank=True)).lazy().collect().to_native())
