import os
import numpy as np
import polars as pl

from survey_kit.imputation.srmi import SRMI
from survey_kit import logger, config


# %%
# Convergence diagnostics answer "did the SRMI iteration settle down, or is
# it still drifting" - run several iterations and implicates so there's
# something to actually plot

n_rows = 3_000
rng = np.random.default_rng(20260913)

x1 = rng.normal(size=n_rows)
x2 = rng.normal(size=n_rows)
y1 = 2.0 * x1 - 1.0 * x2 + rng.normal(scale=1.0, size=n_rows)
y2 = -1.5 * x1 + 0.5 * y1 + rng.normal(scale=1.0, size=n_rows)

df = pl.DataFrame(dict(row_id=range(n_rows), x1=x1, x2=x2, y1=y1, y2=y2))
for col, share in [("y1", 0.2), ("y2", 0.2)]:
    missing = rng.random(n_rows) < share
    df = df.with_columns(
        pl.when(pl.Series(missing)).then(None).otherwise(pl.col(col)).alias(col)
    )

srmi = SRMI.simple_model(
    df=df,
    index="row_id",
    replication=SRMI.Replication(n_implicates=4, n_iterations=10),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_diagnostics_convergence",
        force_start=True,
    ),
)
srmi.run()

# %%
logger.info(
    "convergence() gives the numbers: ac (lag-1 autocorrelation across "
    "implicates) and psrf (potential scale reduction factor - values near 1 "
    "mean the implicates have mixed well)"
)
convergence_table = srmi.convergence()
logger.info(convergence_table.tail(8))

# %%
logger.info(
    "plot_convergence() gives the trace plot version - one line per implicate, "
    "faceted by variable. Lines that intermingle without a trend mean it converged"
)
path_docs_diagnostics = os.path.join(
    config.code_root, "..", "..", "docs", "tutorials", "srmi", "diagnostics"
)
os.makedirs(path_docs_diagnostics, exist_ok=True)

fig_convergence = srmi.plot_convergence(
    path=os.path.join(path_docs_diagnostics, "convergence_trace.html")
)
