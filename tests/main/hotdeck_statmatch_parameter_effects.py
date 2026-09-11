"""
Same convention as lgbm_parameter_effects.py, for Parameters.HotDeck()/
StatMatch() - see that file's own module docstring for the overall
rationale. Not part of the fast day-to-day regression loop - run when
imputation code changes.
"""

import numpy as np
import polars as pl

from survey_kit.imputation.variable import Variable
from survey_kit.imputation.parameters import Parameters
from survey_kit.imputation.srmi import SRMI
from survey_kit import logger, config

from _imputation_test_utils import read_variable_log

path_scratch = config.path_temp_files


def _build_and_run(df, variable, path_suffix, index="idx"):
    srmi = SRMI(
        df=df,
        variables=[variable],
        index=[index],
        replication=SRMI.Replication(n_implicates=1, n_iterations=1),
        parallel=SRMI.Parallel(enabled=False),
        bootstrap=SRMI.Bootstrap(enabled=False),
        storage=SRMI.Storage(
            path_model=f"{path_scratch}/py_hd_param_effects_{path_suffix}", force_start=True
        ),
    )
    srmi.run()
    return srmi


def _collect(srmi):
    import narwhals as nw

    return nw.from_native(srmi.implicates[0].df).lazy().collect().to_native()


#   ============================================================
#   1) model_list actually controls matching
#   ============================================================
logger.info("=== model_list ===")

rng1 = np.random.default_rng(20260910)
n1 = 3000
cell = rng1.choice(["c1", "c2", "c3", "c4"], size=n1)
cell_value = {"c1": 10.0, "c2": 20.0, "c3": 30.0, "c4": 40.0}
#   y is EXACTLY determined by cell - any donor from the same cell is
#       guaranteed correct; a donor from the wrong cell is guaranteed wrong
y_cell = np.array([cell_value[c] for c in cell])
irrelevant = rng1.choice(["x", "y"], size=n1)  # carries no info about y
df_m = pl.DataFrame(dict(idx=range(n1), cell=cell, irrelevant=irrelevant, y=y_cell))
miss_m = rng1.random(n1) < 0.2
df_m = df_m.with_columns(
    pl.when(pl.Series(miss_m)).then(None).otherwise(pl.col("y")).alias("y")
)

v_correct_match = Variable(
    impute_var="y",
    modeltype=Variable.ModelType.StatMatch,
    parameters=Parameters.StatMatch(model_list=["cell"]),
)
srmi_correct = _build_and_run(df_m, v_correct_match, "model_list_correct")
result_correct = _collect(srmi_correct).filter(pl.Series(miss_m))
n_correct_wrong = result_correct.filter(
    pl.col("y") != pl.col("cell").replace_strict(cell_value, return_dtype=pl.Float64)
).height
assert n_correct_wrong == 0, (
    f"model_list=['cell'] should ALWAYS donate the correct cell value - "
    f"{n_correct_wrong}/{result_correct.height} were wrong"
)

v_irrelevant_match = Variable(
    impute_var="y",
    modeltype=Variable.ModelType.StatMatch,
    parameters=Parameters.StatMatch(model_list=["irrelevant"], sequential_drop=False),
)
srmi_irrelevant = _build_and_run(df_m, v_irrelevant_match, "model_list_irrelevant")
result_irrelevant = _collect(srmi_irrelevant).filter(pl.Series(miss_m))
n_irrelevant_wrong = result_irrelevant.filter(
    pl.col("y") != pl.col("cell").replace_strict(cell_value, return_dtype=pl.Float64)
).height
assert n_irrelevant_wrong > 0, (
    "model_list=['irrelevant'] (no real info about y's cell) should get plenty of "
    "cell-mismatched donors - got 0, which would be a wild coincidence"
)
logger.info(
    f"model_list: correct-key wrong={n_correct_wrong}/{result_correct.height}, "
    f"irrelevant-key wrong={n_irrelevant_wrong}/{result_irrelevant.height} - PASSED"
)


#   ============================================================
#   2) donate_list
#   ============================================================
logger.info("=== donate_list ===")

rng2 = np.random.default_rng(20260910)
n2 = 1500
cell2 = rng2.choice(["c1", "c2", "c3"], size=n2)
primary = rng2.normal(size=n2) + (
    np.array([{"c1": 0, "c2": 10, "c3": 20}[c] for c in cell2])
)
companion = primary * 10.0
df_d = pl.DataFrame(
    dict(idx=range(n2), cell=cell2, primary=primary, companion=companion)
)
miss_d = rng2.random(n2) < 0.2
df_d = df_d.with_columns(
    pl.when(pl.Series(miss_d)).then(None).otherwise(pl.col("primary")).alias("primary")
)

v_donate_list = Variable(
    impute_var="primary",
    modeltype=Variable.ModelType.StatMatch,
    parameters=Parameters.StatMatch(model_list=["cell"], donate_list=["companion"]),
)
srmi_dl = _build_and_run(df_d, v_donate_list, "donate_list")
result_dl = _collect(srmi_dl).filter(pl.Series(miss_d))
n_consistent = result_dl.filter(
    (pl.col("companion") - pl.col("primary") * 10.0).abs() < 1e-6
).height
assert n_consistent == result_dl.height, (
    f"with donate_list, every imputed row's companion should equal the same "
    f"donor's primary * 10 - only {n_consistent}/{result_dl.height} did"
)
logger.info("donate_list: PASSED")


#   ============================================================
#   3) sequential_drop - direct dict-shape check, plus end-to-end
#      fallback behavior when the full match set has no donor
#   ============================================================
logger.info("=== sequential_drop ===")

params_seq_true = Parameters.HotDeck(model_list=["a", "b", "c"], sequential_drop=True)
assert params_seq_true["model_list"] == [["a", "b", "c"], ["a", "b"], ["a"]], (
    f"sequential_drop=True should progressively drop the last match variable "
    f"(hotdeck()/statmatch() themselves fall back to a fully-random match for "
    f"anyone still unmatched after ['a']), got: {params_seq_true['model_list']}"
)

params_seq_false = Parameters.HotDeck(model_list=["a", "b", "c"], sequential_drop=False)
assert params_seq_false["model_list"] == [["a", "b", "c"]], (
    f"sequential_drop=False should keep exactly one (full) model, got: "
    f"{params_seq_false['model_list']}"
)

#   StatMatch defaults sequential_drop=False (unlike HotDeck's True default)
params_statmatch_default = Parameters.StatMatch(model_list=["a", "b", "c"])
assert params_statmatch_default["model_list"] == [["a", "b", "c"]], (
    "StatMatch's own default (sequential_drop=False) differs from HotDeck's "
    f"(True) - got: {params_statmatch_default['model_list']}"
)
logger.info("sequential_drop: dict-shape checks PASSED")

#   End-to-end: an impossible full-match cell (unique combination that has
#       no donor) - sequential_drop=True should still find SOME donor via a
#       shorter fallback match; sequential_drop=False should leave it unmatched
rng3 = np.random.default_rng(20260910)
n3 = 500
#   (a2, b2) appears EXACTLY ONCE in the whole dataset (donors included) -
#       its full-match cell genuinely has zero possible donors
lonely_idx = n3 - 1
a3 = rng3.choice(["a1", "a2"], size=n3 - 1).tolist() + ["a2"]
b3 = rng3.choice(["b1", "b2"], size=n3 - 1).tolist() + ["b2"]
#   Re-roll any OTHER accidental (a2, b2) row among the first n3-1
for i in range(n3 - 1):
    while a3[i] == "a2" and b3[i] == "b2":
        b3[i] = rng3.choice(["b1", "b2"])
y3 = rng3.normal(size=n3)
df_seq = pl.DataFrame(dict(idx=range(n3), a=a3, b=b3, y=y3))
assert df_seq.filter((pl.col("a") == "a2") & (pl.col("b") == "b2")).height == 1, (
    "test setup bug - (a2, b2) should appear exactly once"
)
df_seq = df_seq.with_columns(
    pl.when(pl.col("idx") == lonely_idx).then(None).otherwise(pl.col("y")).alias("y")
)
#   Also clear a handful of ordinary (plenty-of-donors) rows so the run
#       has real matching work to do
extra_missing = rng3.random(n3) < 0.1
df_seq = df_seq.with_columns(
    pl.when(pl.Series(extra_missing) & (pl.col("idx") != lonely_idx))
    .then(None)
    .otherwise(pl.col("y"))
    .alias("y")
)

v_seq_true = Variable(
    impute_var="y",
    modeltype=Variable.ModelType.StatMatch,
    parameters=Parameters.StatMatch(model_list=["a", "b"], sequential_drop=True),
)
srmi_seq_true = _build_and_run(df_seq, v_seq_true, "seq_true")
result_seq_true = _collect(srmi_seq_true).filter(pl.col("idx") == lonely_idx)
assert result_seq_true["y"].null_count() == 0, (
    "sequential_drop=True should still find a donor for the impossible-cell row "
    "via a shorter fallback match"
)
#   the ['a'] level (a real, if looser, match) should have caught this row -
#       the universal fully-random fallback shouldn't have been needed
log_seq_true = read_variable_log(srmi_seq_true, v_seq_true)
assert "NO match key at all" not in log_seq_true, (
    "sequential_drop=True should match the impossible-cell row via the "
    "shorter ['a'] level, not need the fully-random fallback"
)

v_seq_false = Variable(
    impute_var="y",
    modeltype=Variable.ModelType.StatMatch,
    parameters=Parameters.StatMatch(model_list=["a", "b"], sequential_drop=False),
)
srmi_seq_false = _build_and_run(df_seq, v_seq_false, "seq_false")
result_seq_false = _collect(srmi_seq_false).filter(pl.col("idx") == lonely_idx)
#   sequential_drop=False never tries a shorter match - but the universal
#       fully-random fallback (see _match_remaining_at_random) still
#       guarantees a donor, now with a logged warning instead of a crash
#       or a silently unmatched row
assert result_seq_false["y"].null_count() == 0, (
    "sequential_drop=False should still find a donor for the impossible-cell "
    "row via the universal fully-random fallback"
)
log_seq_false = read_variable_log(srmi_seq_false, v_seq_false)
assert "NO match key at all" in log_seq_false, (
    "sequential_drop=False's impossible-cell row should only be matched via "
    "the fully-random fallback (with its warning), since ['a', 'b'] is the "
    "only real match level tried"
)
logger.info("sequential_drop: end-to-end fallback checks PASSED")


#   ============================================================
#   4) n_hotdeck_array - a bigger carried-donor array should give more
#      donor diversity (less repeated reuse of the same donor) within a
#      single match cell
#   ============================================================
logger.info("=== n_hotdeck_array ===")

rng4 = np.random.default_rng(20260910)
n4 = 2000
#   One single cell for everyone (a constant match key - functionally
#       equivalent to a genuinely empty match key (see section 6 below),
#       but spelled out explicitly here since donor-array-size diversity
#       isn't what's under test in this section), plenty of
#       distinct donor values
y4 = rng4.normal(size=n4)
df_hd = pl.DataFrame(dict(idx=range(n4), const=["c"] * n4, y=y4))
miss4 = rng4.random(n4) < 0.5
df_hd = df_hd.with_columns(
    pl.when(pl.Series(miss4)).then(None).otherwise(pl.col("y")).alias("y")
)


def _n_distinct_donors_used(srmi):
    return _collect(srmi).filter(pl.Series(miss4))["y"].n_unique()


v_array_small = Variable(
    impute_var="y",
    modeltype=Variable.ModelType.HotDeck,
    parameters=Parameters.HotDeck(model_list=["const"], n_hotdeck_array=1),
)
srmi_small = _build_and_run(df_hd, v_array_small, "array_1")
n_distinct_small = _n_distinct_donors_used(srmi_small)

v_array_large = Variable(
    impute_var="y",
    modeltype=Variable.ModelType.HotDeck,
    parameters=Parameters.HotDeck(model_list=["const"], n_hotdeck_array=20),
)
srmi_large = _build_and_run(df_hd, v_array_large, "array_20")
n_distinct_large = _n_distinct_donors_used(srmi_large)

assert n_distinct_large > n_distinct_small, (
    f"a bigger n_hotdeck_array should give more donor diversity (fewer repeats) - "
    f"n_hotdeck_array=1 gave {n_distinct_small} distinct donors, "
    f"n_hotdeck_array=20 gave {n_distinct_large}"
)
logger.info(
    f"n_hotdeck_array: 1 -> {n_distinct_small} distinct, 20 -> {n_distinct_large} distinct - PASSED"
)


#   ============================================================
#   5) HotDeck and StatMatch both only ever donate real observed values
#      (no fitted model at all - pure matching)
#   ============================================================
logger.info("=== HotDeck/StatMatch only donate real observed values ===")

rng5 = np.random.default_rng(20260910)
n5 = 1500
y5 = rng5.normal(size=n5)
df5 = pl.DataFrame(dict(idx=range(n5), const=["c"] * n5, y=y5))
miss5 = rng5.random(n5) < 0.2
observed5 = set(df5.filter(~pl.Series(miss5))["y"].to_list())
df5 = df5.with_columns(
    pl.when(pl.Series(miss5)).then(None).otherwise(pl.col("y")).alias("y")
)

for modeltype, label in [
    (Variable.ModelType.HotDeck, "HotDeck"),
    (Variable.ModelType.StatMatch, "StatMatch"),
]:
    params = (
        Parameters.HotDeck(model_list=["const"])
        if modeltype == Variable.ModelType.HotDeck
        else Parameters.StatMatch(model_list=["const"])
    )
    v = Variable(impute_var="y", modeltype=modeltype, parameters=params)
    srmi_donate = _build_and_run(df5, v, f"donate_check_{label.lower()}")
    imputed_vals = _collect(srmi_donate).filter(pl.Series(miss5))["y"].to_list()
    assert all(v in observed5 for v in imputed_vals), (
        f"{label} should only ever donate real observed values"
    )
    logger.info(f"{label}: PASSED")

#   ============================================================
#   6) empty match key ([] / model_list=[]) - match fully at random with
#      a logged warning, instead of crashing or leaving rows unmatched.
#      Also confirm sequential_drop=True's cascade actually reaches this
#      fallback level (not just the single-variable level) when every
#      real match variable fails to find a donor.
#   ============================================================
logger.info("=== empty match key ===")

rng6 = np.random.default_rng(20260910)
n6 = 300
y6 = rng6.normal(size=n6)
df6 = pl.DataFrame(dict(idx=range(n6), y=y6))
miss6 = rng6.random(n6) < 0.3
df6 = df6.with_columns(
    pl.when(pl.Series(miss6)).then(None).otherwise(pl.col("y")).alias("y")
)

for modeltype, param_fn, label in [
    (Variable.ModelType.HotDeck, Parameters.HotDeck, "HotDeck"),
    (Variable.ModelType.StatMatch, Parameters.StatMatch, "StatMatch"),
]:
    v = Variable(impute_var="y", modeltype=modeltype, parameters=param_fn(model_list=[]))
    #   a bare model_list=[] means "zero models to try" - hotdeck()/
    #       statmatch() dispatch straight to the fully-random fallback
    #       themselves once the (empty) cascade leaves everyone unmatched,
    #       rather than needing [] preserved as a marker inside model_list
    assert v.parameters["model_list"] == [], (
        f"{label}: model_list=[] should stay a bare empty list, got "
        f"{v.parameters['model_list']}"
    )
    srmi_empty = _build_and_run(df6, v, f"empty_key_{label.lower()}")
    result_empty = _collect(srmi_empty).filter(pl.Series(miss6))
    assert result_empty["y"].null_count() == 0, (
        f"{label}: an empty match key should still match every recipient "
        f"(fully at random), not leave any unmatched"
    )
    log_text = read_variable_log(srmi_empty, v)
    assert "NO match key at all" in log_text, (
        f"{label}: expected the no-match-key warning in the variable log, "
        f"got:\n{log_text}"
    )
    logger.info(f"{label}: empty match key matched with warning - PASSED")

#   sequential_drop=True must actually reach the [] fallback level when
#       even the shortest single-variable match fails for some recipient
rng6b = np.random.default_rng(20260910)
n6b = 300
#   'a' has only ONE value among donors, but the lonely recipient's row
#       is forced to a value that appears NOWHERE among the donors - so
#       even the single-variable ['a'] fallback level can't find it a
#       match, and the cascade must fall all the way to []
a6b = ["same"] * (n6b - 1) + ["never_a_donor_value"]
y6b = rng6b.normal(size=n6b)
df6b = pl.DataFrame(dict(idx=range(n6b), a=a6b, y=y6b))
lonely_idx6b = n6b - 1
df6b = df6b.with_columns(
    pl.when(pl.col("idx") == lonely_idx6b).then(None).otherwise(pl.col("y")).alias("y")
)

v_cascade = Variable(
    impute_var="y",
    modeltype=Variable.ModelType.HotDeck,
    parameters=Parameters.HotDeck(model_list=["a"], sequential_drop=True),
)
srmi_cascade = _build_and_run(df6b, v_cascade, "empty_key_cascade")
result_cascade = _collect(srmi_cascade).filter(pl.col("idx") == lonely_idx6b)
assert result_cascade["y"].null_count() == 0, (
    "sequential_drop=True should reach the [] fallback and still find a "
    "donor for a recipient whose match-variable value no donor shares"
)
cascade_log_text = read_variable_log(srmi_cascade, v_cascade)
assert "NO match key at all" in cascade_log_text, (
    f"expected the cascade to actually reach the empty-key fallback level "
    f"(and log its warning), got:\n{cascade_log_text}"
)
logger.info("sequential_drop cascade reaches [] fallback - PASSED")

logger.info("hotdeck_statmatch_parameter_effects.py: all checks passed")
