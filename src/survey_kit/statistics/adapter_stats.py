from __future__ import annotations

import narwhals as nw
from narwhals.typing import IntoFrameT

from .calculator import StatCalculator
from .replicates import ReplicateStats


class AdapterStats(StatCalculator):
    """
    A StatCalculator built directly from an already-computed set of
    estimates + standard errors - e.g. a regression adapter's own
    analytic output (see survey_kit.statistics.adapters, every one of
    which returns this) - rather than from raw microdata. No
    replicate-weight recomputation at all; use
    StatCalculator.from_function instead if you want SEs derived from
    the spread across replicate weights rather than df_ses's own
    values.

    Matches StatCalculator's own API and function surface exactly
    (comparisons, printing, save/load, and - since MultipleImputation's
    own combination code reads whatever a delegate returns generically -
    mi_ses_from_function/MultipleImputation work with this the same way
    they do a plain StatCalculator) since it IS one, just constructed
    differently - StatCalculator.copy() (which every chainable method
    starts from) preserves the real subclass rather than rebuilding a
    plain StatCalculator, so this stays an AdapterStats through
    filter()/select()/sort()/rename()/with_columns()/scale_by()/etc.

    filter()/rename() are additionally extended here to keep df_vcov in
    sync - the base class's versions only ever touch
    df_estimates/df_ses/df_replicates (df_vcov didn't exist before this
    class), so a plain filter()/rename() would otherwise just drop it
    (with a warning) rather than correctly narrow/rename it.
    with_columns()/drop()/pipe()/concat_with() still drop it - those
    operate on value columns or arbitrary user logic, where there's no
    safe, generic way to know whether/how df_vcov (tied to one specific
    coefficient column) should follow along; select() doesn't touch it
    for the same reason as those, not because it was overlooked.

    .compare() (inherited from StatCalculator) computes a difference/
    ratio SE from df_ses under an independence assumption between the
    two objects being compared, but always clears the result's df_vcov
    (it described the original fit's own term-by-term covariance, which
    doesn't carry over to a cross-object difference/ratio without more
    information than either object has).
    """

    def __init__(
        self,
        df_estimates: IntoFrameT,
        df_ses: IntoFrameT,
        variable_ids: list[str] | str = "Variable",
        by: dict[str, list[str]] | list | None = None,
        df_vcov: IntoFrameT | None = None,
        df_tidy: IntoFrameT | None = None,
        display: bool = True,
        display_all_vars: bool = True,
        display_max_vars: int = 20,
        round_output: bool | int = True,
    ):
        """
        Parameters
        ----------
        df_estimates : IntoFrameT
            One row per estimate - variable_ids (+ any `by` columns)
            plus one column per estimated quantity (e.g. a regression's
            coefficients).
        df_ses : IntoFrameT
            Same shape as df_estimates - standard errors for each value.
        variable_ids : list[str] | str, optional
            Column name(s) identifying each row. Default "Variable"
            (matches every adapter in survey_kit.statistics.adapters,
            which all use this as their own join_on_name default).
        by : dict[str,list[str]] | list | None, optional
            Grouping columns already present in df_estimates/df_ses
            (e.g. if the estimator was run separately per group and the
            results concatenated) - same shape as StatCalculator's own
            `by`. Default None.
        df_vcov : IntoFrameT | None, optional
            Variance-covariance matrix, long/pairwise: for each id in
            variable_ids, two copies of that column suffixed "_1" and
            "_2" (row term, column term), plus one value column matching
            df_estimates' own single non-join_on column - only
            meaningful when df_estimates has exactly one such column
            (e.g. a regression coefficient table). This is what lets
            .compare() compute a correct joint SE between two correlated
            rows of this same fit (e.g. two coefficients), instead of
            assuming independence. Default None.
        df_tidy : IntoFrameT | None, optional
            The underlying package's own native summary table, kept
            as-is for reference - never used in any computation here,
            and not kept in sync by filter()/select() (its own id
            column(s), if any, aren't guaranteed to match variable_ids).
            Default None.
        display, display_all_vars, display_max_vars, round_output
            Same as StatCalculator's own constructor.
        """
        if isinstance(variable_ids, str):
            variable_ids = [variable_ids]

        super().__init__(
            df=None,
            by=by,
            display=display,
            display_all_vars=display_all_vars,
            display_max_vars=display_max_vars,
            round_output=round_output,
            calculate=False,
        )
        self.variable_ids = variable_ids
        self.replicate_stats = ReplicateStats(
            df_estimates=df_estimates,
            df_ses=df_ses,
            df_vcov=df_vcov,
            df_tidy=df_tidy,
        )
        self.df_estimates = self.round_results(df=self.df_estimates)
        self.df_ses = self.round_results(df=self.df_ses)

        if display:
            self.print()

    def print(
        self, round_output=None, estimates_per_page: int = 0, sub_log=None
    ) -> None:
        #   The base class's print() decides whether to show SEs by
        #   checking df_replicates (raw per-replicate draws) - AdapterStats
        #   never has those (its SEs come directly from the adapter, not
        #   from replicate weights), so that check would wrongly hide
        #   df_ses, which IS populated here. Check df_ses directly
        #   instead; _print_replicates itself only ever reads
        #   df_estimates/df_ses; despite the name, it never touches
        #   df_replicates.
        if self.df_ses is not None:
            self._print_replicates(
                round_output=round_output,
                estimates_per_page=estimates_per_page,
                sub_log=sub_log,
            )
        else:
            self._print_estimates(
                round_output=round_output,
                estimates_per_page=estimates_per_page,
                sub_log=sub_log,
            )

    def filter(self, filter_expr: nw.Expr) -> AdapterStats:
        #   ReplicateStats.filter() (via _invalidate_extras) already
        #   drops df_vcov itself - it has no generic way to reshape
        #   term-pair rows - logging a warning as it does. Capture the
        #   original here, before that happens, so it can be correctly
        #   re-narrowed afterward instead of lost.
        original_vcov = self.replicate_stats.df_vcov

        self = super().filter(filter_expr)

        if original_vcov is not None:
            self.replicate_stats.df_vcov = _vcov_semi_join(
                original_vcov, self.variable_ids, self.df_estimates
            )
        return self

    def select(
        self, select_expr: nw.Expr | str | list[str] | list[nw.Expr]
    ) -> AdapterStats:
        #   Column selection narrows which *estimate* columns remain -
        #   df_vcov describes the covariance of a single coefficient
        #   column (see its own docstring above) and df_tidy is an
        #   unrelated native snapshot, so neither one has a "selected
        #   columns" concept of its own to keep in sync here; only rows
        #   (filter(), above) can make them inconsistent.
        return super().select(select_expr)

    def rename(self, d_rename: dict[str, str]) -> AdapterStats:
        #   Same _invalidate_extras drop-and-warn as filter() - capture
        #   df_vcov first and re-derive it afterward, applying the same
        #   rename to whichever of its own columns are affected: an id
        #   column (variable_ids) renames both its "_1"/"_2" copies, the
        #   value column (if renamed) renames directly.
        original_vcov = self.replicate_stats.df_vcov

        self = super().rename(d_rename)

        if original_vcov is not None:
            vcov_cols = set(
                nw.from_native(original_vcov).lazy().collect_schema().names()
            )
            vcov_rename = {}
            for old, new in d_rename.items():
                if f"{old}_1" in vcov_cols:
                    vcov_rename[f"{old}_1"] = f"{new}_1"
                if f"{old}_2" in vcov_cols:
                    vcov_rename[f"{old}_2"] = f"{new}_2"
                if old in vcov_cols:
                    vcov_rename[old] = new

            self.replicate_stats.df_vcov = (
                nw.from_native(original_vcov).rename(vcov_rename).to_native()
                if vcov_rename
                else original_vcov
            )
        return self


def _vcov_semi_join(
    df_vcov: IntoFrameT, variable_ids: list[str], df_estimates: IntoFrameT
) -> IntoFrameT:
    """Keep only df_vcov rows where BOTH the "_1" and "_2" term still exist in df_estimates."""
    nw_vcov = nw.from_native(df_vcov).lazy()
    keep_ids = nw.from_native(df_estimates).lazy().select(variable_ids).unique()

    keep_1 = keep_ids.rename({v: f"{v}_1" for v in variable_ids})
    keep_2 = keep_ids.rename({v: f"{v}_2" for v in variable_ids})

    out = nw_vcov.join(keep_1, on=[f"{v}_1" for v in variable_ids], how="inner")
    out = out.join(keep_2, on=[f"{v}_2" for v in variable_ids], how="inner")

    return out.collect().to_native()
