"""
Shared helpers for the per-modeltype "does each parameter actually do what
it claims" test files (see lgbm_parameter_effects.py, the first of these -
more to follow for other modeltypes). These tests deliberately check BOTH
a statistical/behavioral effect AND a direct log-file signal that the
intended code path actually executed - a statistical check alone can pass
by coincidence (or fail to catch a silently-skipped branch, as happened
with the _lightgbm_simple cv_folds bug these tests exist to guard against).
"""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager


@contextmanager
def capture_global_log():
    """
    Temporarily attach an in-memory handler to survey_kit's module-level
    `logger` and yield a callable returning everything logged to it so far.

    Some lower-level helpers (e.g. Survey_kit_Lightgbm.train()'s own
    "Running lightgbm model with parameters: ..." line, or Tuner's
    "trials finished" line) log via the plain module-level
    `logger`, not the per-variable sub-logger impute.py's own methods use
    (which is what variable_log_path()/read_variable_log() capture) - use
    this instead for anything that logs at that level.
    """
    from survey_kit import logger as sk_logger

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    sk_logger.addHandler(handler)
    try:
        yield buffer.getvalue
    finally:
        sk_logger.removeHandler(handler)


def variable_log_path(
    srmi, variable, implicate: int = 1, iteration: int | None = None
) -> str:
    """
    Path to the per-variable, per-iteration log file SRMI already writes to
    disk for every imputation - {path_model}/logs/{implicate}/
    {iteration:03d}.{variable_index:04d}.{impute_var}.log (variable_index is
    1-based - see implicate.py's _run_one_iteration).

    Parameters
    ----------
    srmi : SRMI
        The (already .run()) SRMI instance.
    variable : Variable | str
        The Variable object, or its impute_var name, to find the log for.
    implicate : int, optional
        1-based implicate number, by default 1.
    iteration : int | None, optional
        Which iteration's log to read - by default None, meaning the last
        iteration actually run (srmi.replication.n_iterations).
    """
    impute_var = variable if isinstance(variable, str) else variable.impute_var
    variable_index = next(
        (i + 1 for i, v in enumerate(srmi.variables) if v.impute_var == impute_var),
        None,
    )
    if variable_index is None:
        raise ValueError(f"No variable named {impute_var!r} in srmi.variables")

    if iteration is None:
        iteration = srmi.replication.n_iterations

    return (
        f"{srmi.storage.path_model}/logs/{implicate}/"
        f"{iteration:03d}.{variable_index:04d}.{impute_var}.log"
    )


def read_variable_log(
    srmi, variable, implicate: int = 1, iteration: int | None = None
) -> str:
    """Read the per-variable log file's full text - see variable_log_path()."""
    path = variable_log_path(srmi, variable, implicate=implicate, iteration=iteration)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
