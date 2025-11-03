import polars as pl
from survey_kit.utilities.random import RandomData
from survey_kit.utilities.dataframe import summary

from survey_kit.utilities.formula_builder import FormulaBuilder
from survey_kit.imputation.utilities.lasso import Lasso
from sklearn.linear_model import LinearRegression

n_rows = 10_000
df = (
    RandomData(n_rows=n_rows, seed=32565437)
    .index("index")
    .float("x_1", -1, 1)
    .float("x_2", -1, 1)
    .float("x_3", -1, 1)
    .float("x_4", -1, 1)
    .float("x_5", -1, 1)
    .float("z_1", -1, 1)
    .float("z_2", -1, 1)
    .float("z_3", -1, 1)
    .float("z_4", -1, 1)
    .float("z_5", -1, 1)
    .float("z_6", -1, 1)
    .np_distribution("e", "normal", loc=0, scale=4)
    .to_df()
)

df = df.with_columns(
    (
        pl.col("x_1")
        - 2 * pl.col("x_2")
        + 3 * pl.col("x_3")
        - 5 * pl.col("x_4")
        + 0.01 * pl.col("x_5")
        + pl.col("e")
    ).alias("y")
)

summary(df)

fb = FormulaBuilder(df=df)
fb.continuous(columns=["x_*", "z_*"])
fb.simple_interaction(columns=["x_1", "x_2"])
print(fb.formula)
lasso = Lasso(df=df, y="y", formula=fb.formula)

vars_lasso = lasso.run()


print(vars_lasso)
model = LinearRegression()
model.fit(X=lasso.df.select(vars_lasso), y=lasso.df.select(lasso.y))
print(model.coef_)
print(model.intercept_)

import polars_ds as pds


from survey_kit.utilities.inputs import list_input


def regression(df: pl.DataFrame, x: list[str] | str, y: str, weight: str = ""):
    x = list_input(x)
    d_extra_args = {}
    if weight != "":
        d_extra_args["weights"] = weight

    df_coefficients = (
        df.select(
            pds.lin_reg(*x, target=y, add_bias=True)
            .list.to_struct(upper_bound=len(x) + 1)
            .alias("reg_out")
        )
        .unnest("reg_out")
        .rename({f"field_{i}": vari for i, vari in enumerate(x + ["_Intercept_"])})
    )

    return df_coefficients


df_regs = regression(df=lasso.df, x=vars_lasso, y=lasso.y)
# df_regs = (
#     lasso.df.select(
#         pds.lin_reg(
#             *vars_lasso,
#             target=lasso.y,
#             # weights=lasso.weight,
#             add_bias=True,
#         ).list.to_struct(upper_bound=len(vars_lasso)+1).alias("reg_out")
#     )
#     .unnest("reg_out")
#     .rename(
#         {
#             f"field_{i}":vari for i, vari in enumerate(vars_lasso + ["_Intercept_"])
#         }
#     )
# )
print(df_regs)


# import polars_ds.linear_models as pds_linear

# # elastic = pds_linear.LR(has_bias=True,
# #                         lambda_=float(lasso.optimal_lambda)
# # )
# elastic = pds_linear.ElasticNet(l1_reg=float(lasso.optimal_lambda),
#                                 l2_reg=0.0,
#                                 has_bias=True)

# out = elastic.fit_df(df=lasso.df,features=lasso.x,target=lasso.y)
# # df_lasso = (
# #     lasso.df.select(
# #         pds.lin_reg(
# #             *lasso.x,
# #             target=lasso.y,
# #             # weights=lasso.weight,
# #             l1_reg=float(lasso.optimal_lambda),
# #             add_bias=True,
# #         ).list.to_struct(upper_bound=len(lasso.x)+1).alias("lasso_out")
# #     )
# #     .unnest("lasso_out")
# #     .rename(
# #         {
# #             f"field_{i}":vari for i, vari in enumerate(lasso.x + ["_Intercept_"])
# #         }
# #     )
# # )

# print(out.coeffs)
# # print(df_lasso)
