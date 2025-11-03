from survey_kit.utilities.random import RandomData
from survey_kit.utilities.formula_builder import FormulaBuilder

from formulaic import Formula

from survey_kit.utilities.dataframe import summary

n_rows = 100_000
df = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_1", 0, 10)
    .float("v_f_continuous", -1, 1)
    .float("v_f_scale", -1, 1)
    .float("v_f_center", -1, 1)
    .float("v_f_standardize", -1, 1)
    .float("v_f_p2_1", -1, 1)
    .float("v_f_p2_2", -1, 1)
    .float("v_f_p2_center_1", -1, 1)
    .float("v_f_p2_center_2", -1, 1)
    .float("v_f_int_1", -1, 1)
    .float("v_f_int_2", -1, 1)
    .float("v_f_int_no_base_1", -1, 1)
    .float("v_f_int_no_base_2", -1, 1)
    .float("v_f_int_clause_1a", -1, 1)
    .float("v_f_int_clause_1b", -1, 1)
    .float("v_f_int_clause_2a", -1, 1)
    .float("v_f_int_clause_2b", -1, 1)
    .float("v_f_int_clause_no_base_1a", -1, 1)
    .float("v_f_int_clause_no_base_1b", -1, 1)
    .float("v_f_int_clause_no_base_2a", -1, 1)
    .float("v_f_int_clause_no_base_2b", -1, 1)
    .float("v_log", 1, 10_000)
    .float("v_bs", 1, 10_000)
    .np_distribution("weight_0", "normal", loc=1, scale=1)
    .np_distribution("weight_1", "normal", loc=1, scale=1)
    .integer("v_factor", 0, 5)
    .integer("v_factor_base_2", 0, 5)
    .to_df()
)

f = FormulaBuilder(df=df, constant=True)
f.continuous(columns=["v_1", "v_f_continuous", "v_f_p2_*"])
# f.scale(columns="v_f_scale")
# f.center(columns="v_f_center")
# f.standardize(columns="v_f_standardize")
# f.polynomial(columns=["v_f_p2_1", "v_f_p2_2"], degree=2)
# f.polynomial(columns="v_f_p2_center_*", degree=2, center=True)

# f.simple_interaction(columns=["v_f_int_1", "v_f_int_2"])

f.simple_interaction(columns=["v_f_int_no_base_*"], no_base=True)

f.interact_clauses(
    clause1="v_f_int_clause_1a + v_f_int_clause_1b",
    clause2="v_f_int_clause_2a + v_f_int_clause_2b",
)
# f.interact_clauses(
#     clause1="v_f_int_clause_no_base_1a + v_f_int_clause_no_base_1b",
#     clause2="v_f_int_clause_no_base_2a + v_f_int_clause_no_base_2b",
#     no_base=True,
# )
# f.factor(df=df, columns="v_factor")

# f.factor(df=df, columns="v_factor_base_2", reference=2)


# f.function(df=df, columns=["v_log"], function_item="log")

# f.function(df=df, columns=["v_bs"], function_item="bs", degree=5)

f_out = Formula(f.__str__())

print(f"original: {f}")
print(f"formulaic: {f_out}")

columns_in_formula = FormulaBuilder.columns_from_formula(formula=f.__str__())
print(columns_in_formula)


df_mm = f_out.get_model_matrix(df)
print(df_mm.schema)
summary(df_mm)

print(FormulaBuilder._is_factor("C(v_factor)[T.5]"))

print(f.expand())
