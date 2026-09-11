import os
import numpy as np
import polars as pl

from survey_kit.imputation.srmi import SRMI
from survey_kit import logger, config


# %%
# Two "does this look right" diagnostics, both answering a different
# question from convergence: not "did the chain settle down" but "do the
# imputed values themselves look plausible"

n_rows = 3_000
rng = np.random.default_rng(20260913)

x1 = rng.normal(size=n_rows)
y = 2.0 * x1 + rng.normal(scale=1.0, size=n_rows)

df = pl.DataFrame(dict(row_id=range(n_rows), x1=x1, y=y))
missing = rng.random(n_rows) < 0.25
df = df.with_columns(
    pl.when(pl.Series(missing)).then(None).otherwise(pl.col("y")).alias("y")
)

srmi = SRMI.simple_model(
    df=df,
    index="row_id",
    replication=SRMI.Replication(n_implicates=3, n_iterations=3),
    parallel=SRMI.Parallel(enabled=False),
    bootstrap=SRMI.Bootstrap(enabled=True),
    storage=SRMI.Storage(
        path_model=f"{config.path_temp_files}/tutorial_diagnostics_quality_propensity",
        force_start=True,
    ),
)
srmi.run()

path_docs_diagnostics = os.path.join(
    config.code_root, "..", "..", "docs", "tutorials", "srmi", "diagnostics"
)
os.makedirs(path_docs_diagnostics, exist_ok=True)

# %%
logger.info(
    "plot_imputation_quality() compares observed vs. imputed values directly - a "
    "density plot showing the whole shape of the distribution for each"
)
fig_quality = srmi.plot_imputation_quality(
    kind="density",
    path=os.path.join(path_docs_diagnostics, "quality_density.html"),
)

# %%
logger.info(
    "This marginal comparison has a real limitation: under MAR, the missing rows "
    "can legitimately have a different marginal distribution than the observed "
    "ones - that's the whole point of a conditional imputation model, not "
    "mean-filling. plot_propensity() checks the same question CONDITIONAL on how "
    "similar a row's covariates are to a typically-missing row (its predicted "
    "response propensity)"
)
fig_propensity = srmi.plot_propensity(
    path=os.path.join(path_docs_diagnostics, "propensity_density.html")
)
