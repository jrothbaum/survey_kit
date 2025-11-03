import time
import numpy as np
import polars as pl

from survey_kit.imputation.utilities.draw_from_quantiles import DrawFromQuantileVectors
from survey_kit.utilities.random import set_seed, RandomData, generate_seed
from survey_kit.utilities.dataframe import columns_from_list, summary
from survey_kit import logger

n_rows = 100_000

set_seed(12309324)
df = (
    RandomData(n_rows=n_rows, seed=generate_seed())
    .index("index")
    .integer("income_p10", 10_001, 25_000)
    .integer("income_p25", 2_000, 10_000)
    # .integer("income_p10", 2_000, 10_000)
    # .integer("income_p25", 10_001, 25_000)
    .integer("income_p50", 25_001, 75_000)
    .integer("income_p75", 75_001, 150_000)
    .integer("income_p90", 150_001, 500_000)
    # .integer("income_p0.1", 1, 2)
    # .integer("income_p1", 100, 101)
    # .integer("income_p10", 9_999, 10_000)
    # .integer("income_p25", 24_999, 25_000)
    # .integer("income_p50", 49_999, 50_000)
    # .integer("income_p75", 74_999, 75_000)
    # .integer("income_p90", 149_999, 150_000)
    .to_df(compress=False)
)


translation = 0
# print(q_income)
# print(q_income.size)
alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
# alphas = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9]

start_draw = time.time()
dist = DrawFromQuantileVectors(
    df_quantiles=df.select(columns_from_list(df, "income_*")).with_columns(
        pl.all() + translation
    ),
    alphas=alphas,
    seed=generate_seed(),
    tails=("lognormal", "lognormal"),
    # tails="exponential"
    # tails="lognormal"
    # tails="pareto"
    # tails=("exponential","exponential")
)
elapsed_tails = time.time() - start_draw
df_v = dist.draw_random_values(n_draws=1)


df_v = df_v.select(pl.col(columns_from_list(df_v, "values*")) - translation)

elapsed_draw = time.time() - start_draw
logger.info(f"elapsed (tails only): {elapsed_tails:0.4f}")
logger.info(f"elapsed (draw only):  {elapsed_draw - elapsed_tails:0.4f}")

logger.info(f"Draws/sec:            {n_rows / elapsed_draw:_.0f}")

summary(
    df_v,
    drb_round=True,
    stats=["n", "q0.1", "q1", "q5", "q10", "q25", "q50", "q75", "q90", "q99", "q99.9"],
)

# # First call compiles the JIT functions (slower)
# print("  Warming up JIT compiler...")
# _ = dist.draw_random_values(10)

# # Subsequent calls are fast
# start = time.time()
# samples = dist.draw_random_values(1000)
# elapsed = time.time() - start

# print(f"  Time: {elapsed*1000:.2f} ms")
# print(f"  Mean: {np.mean(samples):.3f}, Std: {np.std(samples):.3f}")

# # Large scale test
# print("\nTest 2: 1000 distributions, 1000 samples each (1M total)")
# quantiles_large = np.random.randn(1000, 9).cumsum(axis=1)
# alphas_large = np.linspace(0.1, 0.9, 9)

# dist_large = DrawFromQuantileVectors(quantiles_large, alphas_large)

# # Warmup
# _ = dist_large.draw_random_values(10)

# start = time.time()
# (p_large, v_large) = dist_large.draw_random_values(1000)
# elapsed = time.time() - start

# print(f"  Time: {elapsed:.3f} seconds")
# print(f"  Throughput: {v_large.size / elapsed:,.0f} samples/second")

# # Test quantile function
# print("\nTest 3: Quantile function evaluation")
# probs = np.array([0.25, 0.5, 0.75])

# start = time.time()
# quantile_vals = dist_large.ppf(probs)
# elapsed = time.time() - start

# print(f"  Time: {elapsed*1000:.2f} ms for {quantile_vals.size} evaluations")
# print(f"  Shape: {quantile_vals.shape}")
# print(f"  Sample medians: {quantile_vals[:5, 1]}")

# print("\n" + "="*70)
# print("All tests completed!")
# print("="*70)
