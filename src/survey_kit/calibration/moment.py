from __future__ import annotations

from collections import defaultdict

import narwhals as nw
from narwhals.typing import IntoFrameT
from ..utilities.inputs import list_input
from ..utilities.formula_builder import FormulaBuilder, get_model_frame
from ..utilities.dataframe import (
    lazy_backend,
    concat_wrapper,
    fill_missing,
    join_wrapper,
    safe_height,
    safe_sum_cast,
    NarwhalsType,
)
from ..utilities.deepcopy_with_copy_fallback import deepcopy_with_fallback
from ..statistics.statistics import column_stats_builder
from ..statistics.basic_calculations import calculate_by
from ..serializable import Serializable
from .. import logger


class Moment(Serializable):
    """
    Statistical moments for survey calibration.

    A Moment represents target statistics (means, proportions) that survey
    weights should be calibrated to match. Moments can be simple (e.g., overall mean)
    interactions (i.e. share in a and b).  They can also be
    stratified by groups, and can include submoments for more complex constraints.

    Parameters
    ----------
    df : IntoFrameT
        Dataframe containing the data.
    formula : str, optional
        Formulaic formula specifying variables for the moment (e.g., "C(gender) + age").
        Default is "".
    weight : str, optional
        Column name for weights. If empty, uniform weights are created. Default is "".
    index : str | list[str] | None, optional
        Column name(s) for unique observation identifiers. Default is None.
    sort_by : str | list[str] | None, optional
        Column name(s) to sort by when creating row index. Default is None.
    rescale : bool, optional
        Whether to rescale model matrix by dividing by target values. This implies a
        tradeoff between the tolerated miss (i.e. how close do we need to get
        in the calibration) and the target of this moment. Default is True.
    by : list[str] | str | None, optional
        Variables or formula to stratify moment by, creating submoments for each
        i.e. if the formula is race and gender and by is state,
        the moments would race x gender x state
        group. Default is None.
    missing_to_zero : bool, optional
        Fill missing values with zero. Default is True.
    keep_full_group : bool, optional
        When using 'by', whether to keep the overall moment in addition to
        submoments. Default is False.
    equalize_by : bool, optional
        Weight submoments equally rather than by their sample proportions.
        Default is False.
    equalize_by_obs_share : bool, optional
        Weight submoments (within by) by their observation counts. Default is False.
    equalize_by_weight_share : bool, optional
        Weight submoments (within by) by their weight sums. Default is False.

    Attributes
    ----------
    model_matrix : IntoFrameT | None
        Design matrix for calibration (X in the moment equations).
    targets : IntoFrameT | None
        Target values to calibrate to.
    non_zero : IntoFrameT | None
        Count of non-zero observations for each moment variable.
    sub_moments : list[Moment]
        List of submoments when stratifying by groups.
    columns : list[str]
        Column names used in calibration after any restrictions.
    n_observations : int
        Number of observations in this moment.

    Examples
    --------
    Simple moment for categorical variable:

    >>> import polars as pl
    >>> from survey_kit.calibration.moment import Moment
    >>>
    >>> df = pl.DataFrame({
    >>>     "id": range(100),
    >>>     "gender": ["M", "F"] * 50,
    >>>     "weight": [1.0] * 100
    >>> })
    >>>
    >>> moment = Moment(df=df, formula="C(gender)", weight="weight")

    Moment with continuous and categorical variables:

    >>> df = pl.DataFrame({
    >>>     "id": range(100),
    >>>     "age": range(20, 120),
    >>>     "education": ["HS", "College", "Grad"] * 33 + ["HS"],
    >>>     "weight": [1.0] * 100
    >>> })
    >>>
    >>> moment = Moment(
    >>>     df=df,
    >>>     formula="age + C(education)",
    >>>     weight="weight"
    >>> )

    Stratified moment with subgroups:

    >>> moment = Moment(
    >>>     df=df,
    >>>     formula="age",
    >>>     by="C(education)",
    >>>     weight="weight",
    >>>     equalize_by=True
    >>> )
    >>> # This creates separate age moments for each education level

    Notes
    -----
    - Formulas use R/survey_kit_formula-style syntax, with C() for categorical variables.
    - When using 'by' parameter, submoments are automatically created for each group.
    - The rescale option divides model matrix values by targets, which can improve
    convergence but changes the interpretation of calibration parameters.
    - Submoments allow for complex calibration schemes like raking or post-stratification.
    """

    _save_suffix = "moment"
    _save_exclude_items = ["nw_type"]

    def __init__(
        self,
        df: IntoFrameT | None = None,
        nw_type: NarwhalsType | None = None,
        formula: str = "",
        weight: str = "",
        index: str | list[str] | None = None,
        sort_by: str | list[str] | None = None,
        rescale: bool = True,
        by: list[str] | str | None = None,
        missing_to_zero: bool = True,
        keep_full_group: bool = False,
        equalize_by: bool = False,
        equalize_by_obs_share: bool = False,
        equalize_by_weight_share: bool = False,
        by_share: float = 1.0,
        target_moment: bool = True,
        is_sub_moment: bool = False,
        initial_processing: bool = True,
    ):
        if missing_to_zero and df is not None:
            df = fill_missing(df=df)

        self.df = df

        self.formula = formula
        self.weight = weight
        self.index = list_input(index)
        self.rescale = rescale

        if by is None:
            by = []
        self.by = by

        self.equalize_by = equalize_by
        self.equalize_by_obs_share = equalize_by_obs_share
        self.equalize_by_weight_share = equalize_by_weight_share
        self.keep_full_group = keep_full_group

        # Derived variables
        #   Any sub_moments of this?
        self.sub_moments = []
        #   Processed By into a list of where statements
        self.by_where_expressions = []
        self.by_where_strings = []
        # Parallel to by_where_expressions/strings: the exact {column: value}
        # pairs and the column-set each expression tests, so a batched groupby's
        # rows can be matched back to the right sub_moment without re-parsing
        # by_where_strings.
        self.by_where_values = []
        self.by_where_columns = []
        # By group adjustment to weights (i.e. this group's share of the total weight)
        self.by_share = by_share

        # List of variables used in by
        self.byvars = []
        # Columns to use in calibration
        #   All, until restricted for full rank or minimum observation restrictions
        self.columns = []

        # Final model matrix for weighting
        self.model_matrix = None
        # Vector of derived moment scale factors
        self.scale = None
        # Vector of target values
        self.targets = None
        # Vector of non-zero counts
        self.non_zero = None
        # When weighting, carry the zero count of the target, too
        self.non_zero_target = None

        # Has the rank already been checked on this?
        self.rank_checked = False

        # Number of observations
        self.n_observations = 0
        # When weighting, placeholder for target nObs
        self.n_observations_target = 0

        self.nw_type = nw_type
        #   Only do the processing, if data is passed in
        if df is not None:
            self.nw_type = NarwhalsType(df)
            if len(self.index) == 0:
                # Add an index, if there isn't one
                #   We need one to be able to put the file back together again
                sort_by = list_input(sort_by)

                if len(sort_by) == 0:
                    sort_by = (
                        lazy_backend(nw.from_native(df), self.nw_type)
                        .collect_schema()
                        .names()[0]
                    )

                self.index = ["___rownumber"]
                self.df = (
                    nw.from_native(self.df)
                    .with_row_index(name=self.index[0], order_by=sort_by)
                    .to_native()
                )

            #   Weights
            if self.weight == "":
                self.weight = "___ones"
                self.df = (
                    nw.from_native(self.df)
                    .with_columns(nw.lit(1).alias(self.weight))
                    .to_native()
                )

            #   Are there by groups that generate subgroup moments?
            if type(self.by) is str and self.by != "":
                self.byvars = FormulaBuilder.columns_from_formula(formula=self.by)
            elif type(self.by) is list:
                self.byvars = self.by

            if not is_sub_moment and initial_processing:
                self.initialize_moment(target_moment)

    def initialize_moment(
        self,
        # Is this a target moment (i.e. a list of moment constraints)?
        target_moment: bool = True,
    ):
        # Start the list of vars to keep with the vars in the formula
        keepvars = FormulaBuilder.columns_from_formula(formula=self.formula)
        keepvars.extend(self.index)
        keepvars.append(self.weight)
        keepvars.extend(self.byvars)

        #   Restrict to the relevant observations and variables, as needed
        self.df = nw.from_native(self.df).select(keepvars).to_native()

        #   Do we need to calculate targets?
        if target_moment:
            self._get_by_wheres()

            self._get_model_matrix()
            self._get_targets()
            self._create_sub_moments()

    def _get_by_wheres(self):
        if type(self.by) is str:
            f_by = FormulaBuilder(df=self.df, formula=self.by)

            f_by.remove_constant()
            self.by = f_by.formula
            df_by = nw.from_native(
                get_model_frame(
                    f_by.formula,
                    lazy_backend(nw.from_native(self.df), self.nw_type)
                    .collect()
                    .to_native(),
                )
            )

        elif type(self.by) is list:
            if len(self.by):
                df_by = nw.from_native(self.df).select(self.by)
            else:
                #   No by
                return

        #   For interactions, get the unique values of the variables
        interactioncols = [
            coli
            for coli in lazy_backend(df_by, self.nw_type).collect_schema().names()
            if coli.find(":") >= 0
        ]

        if len(interactioncols) > 0:
            for interi in interactioncols:
                subcols = interi.split(":")
                df_inter = lazy_backend(
                    df_by.select(subcols).unique().sort(subcols), self.nw_type
                ).collect()

                for i, rowi in enumerate(df_inter.rows()):
                    if i > 0 or not self.keep_full_group:
                        wherei = None
                        valuesi = {}

                        for j, coli in enumerate(subcols):
                            condi = nw.col(coli) == rowi[j]
                            stringi = f"{coli}=={rowi[j]}"
                            valuesi[coli] = rowi[j]
                            if wherei is None:
                                wherei = condi
                                wherei_string = stringi
                            else:
                                wherei = wherei & condi
                                wherei_string += f"_{stringi}"

                        self.by_where_expressions.append(wherei)
                        self.by_where_strings.append(wherei_string)
                        self.by_where_values.append(valuesi)
                        self.by_where_columns.append(tuple(subcols))

            #   Now drop the interactions and the sub-items
            droplist = []
            for interi in interactioncols:
                subcols = interi.split(":")

                droplist.append(interi)
                droplist.extend(subcols)

            df_by = df_by.drop(list(set(droplist)))

        #   Any non-interacted columns (remaining?)
        for coli in lazy_backend(df_by, self.nw_type).collect_schema().names():
            df_inter = lazy_backend(
                df_by.select(coli).unique().sort(coli), self.nw_type
            ).collect()

            for rowi in df_inter.rows():
                condi = nw.col(coli) == rowi[0]
                stringi = f"{coli}=={rowi[0]}"
                self.by_where_expressions.append(condi)
                self.by_where_strings.append(stringi)
                self.by_where_values.append({coli: rowi[0]})
                self.by_where_columns.append((coli,))

    def _get_model_matrix(self):
        fb = FormulaBuilder(df=self.df, formula=self.formula)

        fb.remove_constant()

        model_matrix = nw.from_native(
            get_model_frame(
                fb.formula,
                lazy_backend(nw.from_native(self.df), self.nw_type)
                .collect()
                .to_native(),
            )
        )

        #   The dataframe only needs the weights and the index, now
        keep_df = [self.weight]
        keep_df.extend(self.index)
        keep_df.extend(self.byvars)
        self.df = nw.from_native(self.df).select(keep_df).to_native()

        #   Append the index to the model matrix, so we can link to it
        self.model_matrix = lazy_backend(
            concat_wrapper(
                [
                    lazy_backend(
                        nw.from_native(self.df).select(self.index), self.nw_type
                    ).collect(),
                    lazy_backend(nw.from_native(model_matrix), self.nw_type).collect(),
                ],
                how="horizontal",
            ),
            self.nw_type,
        ).to_native()

    def _get_targets(self, non_zero_only: bool = False):
        col_df = [self.weight]
        col_df.extend(self.index)

        df_targets = join_wrapper(
            self.model_matrix,
            nw.from_native(self.df).select(col_df),
            on=self.index,
            how="left",
        )

        if not non_zero_only:
            #   Get the targets themselves
            summary_mean = column_stats_builder(
                df=self.model_matrix, cols_exclude=self.index, stat="mean"
            )

            self.targets = (
                nw.from_native(
                    calculate_by(
                        df=df_targets,
                        column_stats=summary_mean,
                        weight=self.weight,
                        no_suffix=True,
                    )
                )
                .with_columns(nw.all() * self.by_share)
                .to_native()
            )

        #   How many values are non-zero?
        summary_nonzero = column_stats_builder(
            df=self.model_matrix,
            cols_include="*",
            cols_exclude=self.index,
            stat="rawcount_not0",
        )

        self.non_zero = calculate_by(
            df=df_targets,
            column_stats=summary_nonzero,
            weight=self.weight,
            no_suffix=True,
        )

    def _rescale_targets(self):
        if self.rescale and self.targets is not None:
            cols_scale = []
            cols_targets = []

            targets = nw.from_native(self.targets)

            for coli in lazy_backend(targets, self.nw_type).collect_schema().names():
                cols_scale.append((nw.col(coli) ** -1).alias(coli))
                cols_targets.append(nw.lit(1).alias(coli))
            self.scale = targets.with_columns(cols_scale).to_native()
            self.targets = targets.with_columns(cols_targets).to_native()

    def _create_sub_moments(self):
        if len(self.by_where_expressions) > 0:
            # Need the ratio of by group weights to overall weights
            if self.equalize_by:
                if self.equalize_by_obs_share:
                    total_obs = safe_height(self.df)

                #   We need to get or load the full group targets for the subgroup, if we want to equalize them
                if self.targets is None:
                    # Get the targets of the parent moment without affecting the actual object (self)
                    m_targets_only = deepcopy_with_fallback(self)
                    m_targets_only._get_model_matrix()
                    m_targets_only._get_targets()

                    targets_equalize = m_targets_only.targets
                    del m_targets_only
                else:
                    targets_equalize = self.targets

        #   Make sure the weight sums do not overflow
        self.df = safe_sum_cast(df=self.df, columns=[self.weight])
        total_weight = (
            lazy_backend(nw.from_native(self.df), self.nw_type)
            .select(nw.col(self.weight).sum())
            .collect()
            .item(0, 0)
        )

        #   Don't recreate the model matrix/df, if we don't have to
        if self.model_matrix is not None:
            bykeep = []
            bykeep.append(self.weight)
            bykeep.extend(self.index)
            bykeep.extend(self.byvars)
            df_by = lazy_backend(nw.from_native(self.df), self.nw_type).select(bykeep)
        else:
            df_by = lazy_backend(nw.from_native(self.df), self.nw_type)

        #   Materialize once so each group's .filter() below doesn't re-run
        #   the whole upstream lazy plan from scratch on every group.
        df_by = lazy_backend(df_by.collect(), self.nw_type)

        n_groups = len(self.by_where_expressions)

        if n_groups > 0:
            try:
                self._create_sub_moments_batched(
                    df_by=df_by,
                    total_weight=total_weight,
                    total_obs=total_obs if (self.equalize_by and self.equalize_by_obs_share) else None,
                    targets_equalize=targets_equalize if self.equalize_by else None,
                    n_groups=n_groups,
                )
            except Exception as err:
                logger.warning(
                    f"Batched by-group computation failed ({err!r}); falling back "
                    "to the slower per-group loop for this Moment (e.g. some "
                    "backends -- pyarrow in particular -- don't support the "
                    "compound aggregation a weighted mean needs inside a "
                    "group_by)."
                )
                self.sub_moments = []
                self._create_sub_moments_looped(
                    df_by=df_by,
                    total_weight=total_weight,
                    total_obs=total_obs if (self.equalize_by and self.equalize_by_obs_share) else None,
                    targets_equalize=targets_equalize if self.equalize_by else None,
                    n_groups=n_groups,
                )

        if len(self.by_where_expressions) == 0:
            #   No sub_moments
            self.n_observations = safe_height(self.df)

        #   Rescale the Target moments, if necessary
        self._rescale_targets()

        for subi in self.sub_moments:
            subi._rescale_targets()

        #   Don't need df or model_matrix anymore
        self.df = None
        self.model_matrix = None

        #   Get rid of the other dataframes if this isn't going to be
        #       used as a moment to match to
        if len(self.by_where_expressions) > 0 and not self.keep_full_group:
            self.targets = None
            self.non_zero = None
            self.scale = None

    def _create_sub_moments_batched(
        self,
        df_by,
        total_weight: float,
        total_obs: int | None,
        targets_equalize,
        n_groups: int,
    ) -> None:
        """
        Fast path: compute weight/obs sums for every group in one .select(),
        and weighted-mean targets / non-zero counts for every group via one
        calculate_by(..., by=[cols]) groupby call per distinct column-set
        (interaction terms share a call; each plain by-column gets its own),
        instead of a separate join + aggregation per group.

        Raises on any backend that can't run the compound weighted-mean
        aggregation inside a group_by (e.g. pyarrow) -- the caller falls back
        to _create_sub_moments_looped in that case.
        """
        #   Weight sum (and row count) for every group in one pass.
        group_select = []
        for i, byi in enumerate(self.by_where_expressions):
            mask = byi.cast(nw.Int64)
            group_select.append(
                (nw.col(self.weight) * mask).sum().alias(f"___by_group_weight_{i}")
            )
            group_select.append(mask.sum().alias(f"___by_group_obs_{i}"))

        group_stats_row = df_by.select(group_select).collect()

        def _group_weight(i: int) -> float:
            return group_stats_row.item(0, f"___by_group_weight_{i}")

        def _group_obs(i: int) -> int:
            return group_stats_row.item(0, f"___by_group_obs_{i}")

        #   Weighted-mean targets and non-zero counts for every group, batched
        #   by which columns each group's condition depends on.
        need_targets = not self.equalize_by
        df_targets_all = join_wrapper(
            self.model_matrix,
            nw.from_native(self.df).select([self.weight] + self.index + self.byvars),
            on=self.index,
            how="left",
        )
        summary_mean = column_stats_builder(
            df=self.model_matrix, cols_exclude=self.index, stat="mean"
        )
        summary_nonzero = column_stats_builder(
            df=self.model_matrix,
            cols_include="*",
            cols_exclude=self.index,
            stat="rawcount_not0",
        )

        indices_by_column_set = defaultdict(list)
        for i, cols in enumerate(self.by_where_columns):
            indices_by_column_set[cols].append(i)

        batched_targets_by_index = {}
        batched_nonzero_by_index = {}
        for cols, indices in indices_by_column_set.items():
            cols_list = list(cols)

            batched_nonzero = nw.from_native(
                calculate_by(
                    df=df_targets_all,
                    column_stats=dict(summary_nonzero),
                    weight=self.weight,
                    by=[cols_list],
                    no_suffix=True,
                )
            ).lazy()
            if need_targets:
                batched_targets = nw.from_native(
                    calculate_by(
                        df=df_targets_all,
                        column_stats=dict(summary_mean),
                        weight=self.weight,
                        by=[cols_list],
                        no_suffix=True,
                    )
                ).lazy()

            for i in indices:
                values_i = self.by_where_values[i]
                match_expr = None
                for coli in cols_list:
                    condi = nw.col(coli) == values_i[coli]
                    match_expr = condi if match_expr is None else match_expr & condi

                batched_nonzero_by_index[i] = (
                    batched_nonzero.filter(match_expr)
                    .drop(cols_list)
                    .collect()
                    .to_native()
                )
                if need_targets:
                    batched_targets_by_index[i] = (
                        batched_targets.filter(match_expr)
                        .drop(cols_list)
                        .collect()
                        .to_native()
                    )

        for i_by, byi in enumerate(self.by_where_expressions):
            group_weight = _group_weight(i_by)

            #  What is the share of the weight that should go to the
            #      group identified by this byi
            if self.equalize_by:
                if self.equalize_by_obs_share:
                    by_share = _group_obs(i_by) / total_obs
                elif self.equalize_by_weight_share:
                    by_share = group_weight / total_weight
                else:
                    by_share = 1 / n_groups
            else:
                by_share = group_weight / total_weight

            sub_moment = Moment(
                df=None,
                nw_type=self.nw_type,
                formula=self.formula,
                weight=self.weight,
                index=self.index,
                rescale=self.rescale,
                by_share=by_share,
                is_sub_moment=True,
            )

            sub_moment.non_zero = batched_nonzero_by_index[i_by]
            sub_moment.n_observations = _group_obs(i_by)

            if self.equalize_by:
                sub_moment.targets = (
                    nw.from_native(targets_equalize)
                    .with_columns(nw.all() * by_share)
                    .to_native()
                )
            else:
                sub_moment.targets = (
                    nw.from_native(batched_targets_by_index[i_by])
                    .with_columns(nw.all() * by_share)
                    .to_native()
                )

            sub_moment.by_where_expressions = [byi]
            sub_moment.by_where_strings = [self.by_where_strings[i_by]]
            self.sub_moments.append(sub_moment)

    def _create_sub_moments_looped(
        self,
        df_by,
        total_weight: float,
        total_obs: int | None,
        targets_equalize,
        n_groups: int,
    ) -> None:
        """Slow path: the original one-group-at-a-time implementation, kept
        as a fallback for backends _create_sub_moments_batched can't handle."""
        for i_by, byi in enumerate(self.by_where_expressions):
            df_byi = df_by.filter(byi)

            group_weight = (
                lazy_backend(df_byi, self.nw_type)
                .select(nw.col(self.weight).sum())
                .collect()
                .item(0, 0)
            )

            #  What is the share of the weight that should go to the
            #      group identified by this byi
            if self.equalize_by:
                if self.equalize_by_obs_share:
                    group_obs = safe_height(df_byi)
                    by_share = group_obs / total_obs
                elif self.equalize_by_weight_share:
                    by_share = group_weight / total_weight
                else:
                    by_share = 1 / n_groups
            else:
                by_share = group_weight / total_weight

            sub_moment = Moment(
                df=df_byi,
                nw_type=self.nw_type,
                formula=self.formula,
                weight=self.weight,
                index=self.index,
                rescale=self.rescale,
                by_share=by_share,
                is_sub_moment=True,
            )

            #  Get the model matrix by subsetting the parent model matrix
            sub_moment.model_matrix = join_wrapper(
                nw.from_native(sub_moment.df).select(sub_moment.index),
                self.model_matrix,
                how="left",
                on=self.index,
            )

            sub_moment._get_targets(non_zero_only=self.equalize_by)

            if self.equalize_by:
                sub_moment.targets = (
                    nw.from_native(targets_equalize)
                    .with_columns(nw.all() * by_share)
                    .to_native()
                )

            sub_moment.by_where_expressions = [byi]
            sub_moment.by_where_strings = [self.by_where_strings[i_by]]
            self.sub_moments.append(sub_moment)

            sub_moment.n_observations = safe_height(sub_moment.df)
            sub_moment.df = None
            sub_moment.model_matrix = None

    def rescaled_model_matrix(self, narrow: bool = False):
        if narrow and len(self.columns) > 0:
            #   Only pass the subset of columns
            df_out = nw.from_native(self.model_matrix).select(self.columns).to_native()
        else:
            df_out = nw.from_native(self.model_matrix).sort(self.index).to_native()

        if self.rescale and self.scale is not None:
            with_columns = []

            cols_main = (
                lazy_backend(nw.from_native(df_out), self.nw_type)
                .collect_schema()
                .names()
            )
            cols_scale = (
                lazy_backend(nw.from_native(self.scale), self.nw_type)
                .collect_schema()
                .names()
            )
            for coli in set(cols_main).intersection(cols_scale):
                if coli not in self.index:
                    #   Get the scale adjustment from self.scale
                    valuei = (
                        lazy_backend(nw.from_native(self.scale), self.nw_type)
                        .select(coli)
                        .collect()
                        .item(0, 0)
                    )
                    with_columns.append((nw.col(coli) * valuei).alias(coli))

            df_out = (
                lazy_backend(nw.from_native(df_out), self.nw_type)
                .with_columns(with_columns)
                .to_native()
            )

        return df_out

    #####################################################
    #   Serializable - BEGIN
    #####################################################
    @classmethod
    def _init_from_dict(cls, data: dict):
        return super()._init_from_dict(data, initial_processing=False)

    @classmethod
    def load(
        cls,
        path: str = "",
        delete: bool = False,
        delete_only: bool = False,
        **df_kwargs,
    ) -> Moment | None:
        return super().load(
            path=path, delete=delete, delete_only=delete_only, **df_kwargs
        )

    #####################################################
    #   Serializable - END
    #####################################################
