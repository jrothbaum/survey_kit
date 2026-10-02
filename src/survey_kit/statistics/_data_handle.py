"""
A dataset loaded once into an embedded runtime (R, Stata) and reused across
adapter calls, instead of re-exporting it from Python on every call.

Pass the handle as `df` to that runtime's adapters. Subsetting is a
per-run `filter=` argument on the adapter - a condition string in the
runtime's own syntax, applied for that one run only and never changing the
loaded data (R subsets a copy inside R; Stata wraps the run in `preserve` /
`keep if` / `restore`). Rows where the condition is NA are dropped in R, as
in polars' filter; Stata follows its own rules (a missing value counts as
larger than any number), so add `& !missing(x)` where that matters.
Subclasses: RData (_r_interop.py), StataData (_stata_interop.py).
"""

from __future__ import annotations

import narwhals as nw
import polars as pl


class DataHandle:
    _language = ""

    def __init__(self, df):
        self._closed = False
        self._load(nw.from_native(df).lazy().collect().to_polars())

    def equals_condition(self, values: dict) -> str:
        """A condition (in the runtime's syntax) true where each column equals its value (None = missing) - e.g. one group of a `by`."""
        return " & ".join(self._equals(col, value) for col, value in values.items())

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("This data handle has been closed.")

    def close(self) -> None:
        """Release the loaded data; the handle can't be used afterward."""
        if not self._closed:
            self._release()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    #   --- runtime-specific ---
    def _load(self, df: pl.DataFrame) -> None:
        raise NotImplementedError

    def _release(self) -> None:
        pass

    def _equals(self, column: str, value) -> str:
        raise NotImplementedError

    def unique_values(self, columns: list[str], filter: str | None = None) -> pl.DataFrame:
        """Unique combinations of `columns` among the rows `filter` keeps."""
        raise NotImplementedError
