from __future__ import annotations


import narwhals as nw
from narwhals.typing import IntoFrameT
import narwhals.selectors as cs
import polars as pl

from ..serializable import Serializable

from .moment import Moment

import scipy.sparse as sp
import numpy as np


from ..utilities.inputs import list_input
from ..utilities.dataframe import (
    fill_missing,
    safe_height,
    NarwhalsType,
    rename_with_prefix_suffix,
    join_wrapper,
    concat_wrapper,
    safe_sum_cast,
    backend_eager,
)

from ..utilities.compress import compress_df
from ..utilities.deepcopy_with_copy_fallback import deepcopy_with_fallback

from ..entropy_balance_weighting import ebw_routines
from ..entropy_balance_weighting.legacy import survey_calibration as legacy_calibration

from .trim import Trim

from ..statistics.statistics import (
    column_stats_builder,
)

from ..statistics.basic_calculations import calculate_by

from .. import logger


class Calibration(Serializable):
    """
    Survey calibration using entropy balancing methods.

    Calibration adjusts survey weights to match known population moments (targets) while
    minimizing the distance from base weights.

    Parameters
    ----------
    df : IntoFrameT
        Input dataframe containing the survey data to be calibrated.
    moments : list[Moment | str] | Moment | str | None, optional
        Moment objects or paths to saved moments defining calibration targets.
        Can be a single Moment, list of Moments, or paths to saved Moment files.
        Default is None.
    missing_to_zero : bool, optional
        Whether to fill missing values with zero before calibration. Default is True.
    index : list[str] | None, optional
        Column names to use as unique identifiers for observations. If None,
        a row number index will be created. Default is None.
    weight : str, optional
        Column name containing base weights. If empty string, uniform weights
        of 1 will be created. Default is "".
    initial_guess : str, optional
        Column name for initial weight guesses to improve convergence. Default is "".
    final_weight : str, optional
        Column name for output calibrated weights. If empty, defaults to
        "___final_weight". Default is "".
    aggregation : str, optional
        How to combine multiple moments: "Combined" (all at once), "Sequential"
        (one at a time), or "Both" (sequential first, then combined if needed).
        Default is "Combined".
    iterations : int, optional
        Maximum number of iterations for calibration algorithm. If -1, uses
        method-specific defaults (50 for aebw, 250 for legacy_ebw). Default is -1.
    iterations_loop : int, optional
        Maximum number of loops for sequential/trimmed calibration. Default is 20.
    tolerance : float, optional
        Convergence tolerance for maximum difference between targets and estimates.
        Default is 0.00001.

    Attributes
    ----------
    df : IntoFrameT
        Dataframe with calibrated weights in the final_weight column.
    moments : list[Moment]
        List of Moment objects used for calibration.
    diagnostics_out : dict | None
        Dictionary containing convergence status and diagnostic dataframe after
        running calibration.

    Examples
    --------
    Basic calibration to match population totals:

    >>> import polars as pl
    >>> from survey_kit.calibration import Calibration, Moment
    >>>
    >>> # Create sample data
    >>> df = pl.DataFrame({
    >>>     "id": range(100),
    >>>     "age_group": ["18-34", "35-54", "55+"] * 33 + ["18-34"],
    >>>     "region": ["North", "South"] * 50,
    >>>     "base_weight": [1.0] * 100
    >>> })
    >>>
    >>> # Define target moments
    >>> age_moment = Moment(df=df, formula="C(age_group)", weight="base_weight")
    >>>
    >>> # Run calibration
    >>> c = Calibration(df=df, moments=[age_moment], weight="base_weight")
    >>> results = c.run()
    >>>
    >>> # Get the calibrated weights (appended to the original data)
    >>> calibrated_df = c.get_final_weights(df)

    Multiple moments with sequential calibration:

    >>> region_moment = Moment(df=df, formula="C(region)", weight="base_weight")
    >>> c = Calibration(
    >>>     df=df,
    >>>     moments=[age_moment, region_moment],
    >>>     aggregation="Sequential",
    >>>     weight="base_weight"
    >>> )
    >>> results = c.run(min_obs=10)

    With weight trimming:

    >>> from survey_kit.calibration.trim import Trim
    >>> trim_params = Trim(trim=True, min_val=0.2, max_val=3.0)
    >>> results = c.run(trim=trim_params, min_obs=[0, 10, 50])

    Notes
    -----
    - The calibration preserves the total sum of weights while adjusting individual
    weight values to match target moments.
    - Sequential aggregation calibrates to each moment one at a time, while combined
    aggregation solves for all moments simultaneously.
    - The "Both" aggregation tries sequential first and falls back to combined if
    convergence criteria aren't met.
    - Use min_obs parameter to exclude moments with too few non-zero observations,
    which can cause convergence issues.
    """

    _save_suffix = "calibration"
    _save_exclude_items = ["nw_type"]

    def __init__(
        self,
        df: IntoFrameT,
        moments: list[Moment | str] | Moment | str | None = None,
        missing_to_zero: bool = True,
        index: list[str] | None = None,
        weight: str = "",
        initial_guess: str = "",
        final_weight: str = "",
        aggregation: str = "Combined",
        iterations: int = -1,
        iterations_loop: int = 20,
        tolerance: float = 0.00001,
    ):
        self.nw_type = NarwhalsType(df)

        moments = list_input(moments)
        index = list_input(index)

        #   Pending other optimizations?
        method = "aebw"

        acceptableMethods = ["aebw", "legacy_ebw"]
        bAcceptableMethods = method in acceptableMethods

        acceptableAggregations = ["Combined", "Sequential", "Both"]
        bAcceptableAggregation = aggregation in acceptableAggregations

        if not bAcceptableMethods:
            message = f"ONLY {', '.join(acceptableMethods)} ARE ACCEPTABLE METHODS - passed({method})"

            logger.error(message)
            raise Exception(message)

        if not bAcceptableAggregation:
            message = f"ONLY {', '.join(acceptableAggregations)} ARE ACCEPTABLE METHODS - passed({aggregation})"

            logger.error(message)
            raise Exception(message)

        if missing_to_zero and df is not None:
            df = fill_missing(df=df, value=0)

        df = nw.from_native(df)
        #   Check that base weight is > 0
        if weight != "":
            c_weight = nw.col(weight)
            c_invalid_weights = c_weight.le(0) | c_weight.is_null()

            n_invalid = safe_height(
                nw.from_native(df).lazy().select(weight).filter(c_invalid_weights)
            )
            if n_invalid:
                logger.info(
                    f"  Dropping {n_invalid} observation(s) with invalid base weights (<= 0 or missing)"
                )
                df = df.filter(~c_invalid_weights)
        self.df = df.lazy_backend(self.nw_type)

        # Add an index, if there isn't one
        #   We need one to be able to put the file back together again

        if len(index) == 0:
            self.index = ["___rownumber"]
            self.df = (
                self.df.collect()
                .with_row_index(name=self.index[0])
                .lazy_backend(self.nw_type)
            )
        else:
            self.index = index

        if weight == "":
            weight = "___ones"
            self.df = self.df.with_columns(nw.lit(1).alias(weight))
        self.weight = weight
        self.initial_guess = initial_guess

        if final_weight == "":
            final_weight = "___final_weight"
        self.final_weight = final_weight
        self.df = self.df.with_columns(nw.col(self.weight).alias(self.final_weight))

        self.aggregation = aggregation
        self.method = method

        #   Default values if not passed
        if iterations < 0:
            if self.method == "aebw":
                iterations = 50
            else:
                iterations = 250

        self.iterations = iterations
        self.tolerance = tolerance
        self.iterations_loop = iterations_loop

        # Process each "target" moment passed in
        #   and create a paired "weight" moment object
        self.moments = []
        if len(moments) > 0:
            for momenti in moments:
                self.moments.append(self.process_single_moment(momenti))

        #   Placeholder for combined moments when running both sequential and combined
        self.c_combined = None

        #   Placeholder for post-run diagnostics information
        self.diagnostics_out = None

    def process_single_moment(
        self, m_input: Moment | str, df: IntoFrameT | None = None
    ) -> Moment:
        # Clone the moment object - the processed one will be based on the data to calibrate,
        #                           the input one is based on the data passed originally
        #                           which need not be the same (something should generally differ...)
        if type(m_input) is str:
            m = Moment.load(m_input)
        else:
            m = deepcopy_with_fallback(m_input)

        m.non_zero_target = m_input.non_zero
        m.n_observations_target = m_input.n_observations

        if df is None:
            m.df = self.df
        else:
            m.df = df

        m.weight = self.weight

        m.index = self.index

        m.initialize_moment(target_moment=False)

        #   Does this moment have any targets?
        if m.targets is not None:
            m._get_model_matrix()
            m._get_targets(non_zero_only=True)
            m.n_observations = safe_height(m.model_matrix)

        #   Process any submoments
        if len(m.sub_moments) > 0:
            sub_moments = []
            for subi in m.sub_moments:
                subi = self.process_single_moment(subi, m.df)

                sub_moments.append(subi)

            m.sub_moments = sub_moments

        self.compare_moment_variables(m_target=m_input, m_weight=m)

        return m

    def compare_moment_variables(self, m_target: Moment, m_weight: Moment) -> None:
        """
        Do the variables match across source and weighting moments?
           Won't if a value doesn't exist in one data set that does in the other
           If don't match, set to the intersection of variables - can only weight to the things in both samples

        Parameters
        ----------
        m_target : Moment
            Input moment with targets calculated
        m_weight : Moment, optional
            moment for calibration

        Returns
        -------
        None
        """

        if m_target.targets is not None and m_weight.model_matrix is not None:
            cols_target = (
                nw.from_native(m_target.targets).lazy().collect_schema().names()
            )
            cols_weight = (
                nw.from_native(m_weight.model_matrix).lazy().collect_schema().names()
            )
            m_weight.columns = list(set(cols_target).intersection(cols_weight))

    def censor_to_min_obs(self, min_obs: int = 0) -> None:
        """
        In calibration, do we want to limit to moments with greater than some
           threshold of observations where x != 0

        Parameters
        ----------
        min_obs : int, optional
            The minimum number of != 0 observations.
            The default is 0. (don't censor)

        Returns
        -------
        None

        """
        for momenti in self.moments:
            self._censort_to_min_obs_by_moment(min_obs=min_obs, m=momenti)

            for subi in momenti.sub_moments:
                self._censort_to_min_obs_by_moment(min_obs=min_obs, m=subi)

        if self.c_combined is not None:
            self.c_combined.censor_to_min_obs(min_obs=min_obs)

    def _censort_to_min_obs_by_moment(self, min_obs: int, m: Moment) -> None:
        """
        For each moment, censor, if needed

        Parameters
        ----------
        min_obs : int
            Minimum non-zero observations to censor at
        m : Moment
            Moment to censor

        Returns
        -------
        None
        """

        if m.non_zero_target is not None and len(m.columns) > 0:
            nw_type = NarwhalsType(m.non_zero_target)
            nonzero_cols = (
                nw_type.to_polars().select(m.columns).transpose(include_header=True)
            )

            m.columns = nonzero_cols.filter(
                pl.col(nonzero_cols.lazy().collect_schema().names()[1]) >= min_obs
            )[nonzero_cols.lazy().collect_schema().names()[0]].to_list()

        if m.non_zero is not None and len(m.columns) > 0:
            nw_type = NarwhalsType(m.non_zero)

            nonzero_cols = (
                nw_type.to_polars().select(m.columns).transpose(include_header=True)
            )

            m.columns = nonzero_cols.filter(
                pl.col(nonzero_cols.lazy().collect_schema().names()[1]) >= min_obs
            )[nonzero_cols.lazy().collect_schema().names()[0]].to_list()

    def combine_moments(self, all: bool = False, sub_moments: bool = True) -> None:
        """
        To simultaneously weight to lots of moments, we need to combine them
            This directly edits the items in self.moments

        Parameters
        ----------
        all : bool, optional
            Combine all or just submoments into parent moments?
            The default is False.
        sub_moments : bool, optional
            Combine submoments?
            The default is True.
        Returns
        -------
        None

        """

        if all or sub_moments:
            logger.info("Aggregating any sub_moments")
            for i in range(len(self.moments)):
                if len(self.moments[i].sub_moments) > 0:
                    self.moments[i] = self._combine_moment_with_sub_moments(
                        self.moments[i]
                    )

        if all and len(self.moments) > 0:
            #   Check that submoments are combined
            for momenti in self.moments:
                if len(momenti.sub_moments) > 0:
                    message = "Must combine sub moments before combining all moments"
                    logger.error(message)
                    raise Exception(message)
            #   Merge aggregate the moments together into one
            self.moments = [
                self._combine_moment_list(m_list=self.moments, with_count_prefix=True)
            ]

    def _combine_moment_with_sub_moments(self, m: Moment) -> Moment:
        """
        Combine this moment and its submoments (i.e. moments by state)
            into one moment for weighting

        Parameters
        ----------
        m : Moment
            Input moment

        Returns
        -------
        Moment
            The combined moment with no submoments
        """

        if len(m.sub_moments) == 0:
            # No submoments
            return m
        else:
            # Combine this with sub, adding by prefix to submoments only
            m_list = [m]
            prefix_list = [False]

            for subi in m.sub_moments:
                m_list.append(subi)
                prefix_list.append(True)

            m = self._combine_moment_list(m_list=m_list, with_by_prefix=prefix_list)
        return m

    def _combine_moment_list(
        self,
        m_list: list[Moment],
        with_by_prefix: list[bool] | bool = None,
        with_count_prefix: bool = False,
    ) -> Moment:
        #   Combined moment
        m_combined = None

        for i_moment, m in enumerate(m_list):
            if with_by_prefix is None:
                with_by_prefixi = False
            elif type(with_by_prefix) is list:
                with_by_prefixi = with_by_prefix[i_moment]
            else:
                with_by_prefixi = with_by_prefix

            #   Add the prefix?
            prefix = self._combine_prefix(
                with_by_prefix=with_by_prefixi,
                with_count_prefix=with_count_prefix,
                count_value=i_moment,
                m=m,
            )

            if with_by_prefixi:
                sub_moment_column_name = "_in"
            else:
                sub_moment_column_name = ""

            m_combined = self._combine_moments_concatenate_data(
                m_add=m,
                m_out=m_combined,
                sub_moment_column_name=sub_moment_column_name,
                prefix=prefix,
            )

        #   Fill in the missing values left by submoment subgroups
        m_combined.model_matrix = fill_missing(m_combined.model_matrix, value=0)

        #   m_combined.df = m_combined.df.sort(self.index)
        #   m_combined.ModelMatrix = m_combined.ModelMatrix.sort(self.index)

        m_combined.by_where_expressions = []
        m_combined.sub_moments = []

        return m_combined

    def _combine_prefix(
        self,
        with_by_prefix: bool = False,
        with_count_prefix: bool = False,
        count_value: int = 0,
        m: Moment = None,
    ):
        prefix = ""
        if with_by_prefix:
            prefix = f"{m.by_where_strings[0]}:"
        if with_count_prefix:
            prefix = f"m{count_value}_{prefix}"

        return prefix

    #   Actually put them moments together
    def _combine_moments_concatenate_data(
        self,
        m_add: Moment | None = None,
        m_out: Moment | None = None,
        sub_moment_column_name: str = "",
        prefix: str = "",
    ) -> Moment:
        #   NonZero always should exist if this is actually a moment to be estimated
        #       Rather than just a parent to submoments,
        if m_add.non_zero is None:
            #   if it's null, don't add anything from this and just return the combined moment
            return m_out

        #   NonZero always should exist if this is actually a moment to be estimated
        #       so get the list of column names from it
        columns_to_add = nw.from_native(m_add.non_zero).lazy().collect_schema().names()

        #   Nothing to add?, if so, return m_out as we're doing nothing here
        if len(columns_to_add) == 0:
            return m_out

        # Add the dummy for being in this submoment
        if sub_moment_column_name != "":
            #   Non-zero count is the number of observations in group
            m_add.non_zero = (
                nw.from_native(m_add.non_zero)
                .with_columns(
                    nw.lit(m_add.n_observations).alias(sub_moment_column_name)
                )
                .to_native()
            )

            m_add.non_zero_target = (
                nw.from_native(m_add.non_zero_target)
                .with_columns(
                    nw.lit(m_add.n_observations_target).alias(sub_moment_column_name)
                )
                .to_native()
            )

            #   Target is share in this Moment (set to 1 as it should be ByShare, which it will be multiplied by )
            if m_add.rescale:
                m_add.targets = (
                    nw.from_native(m_add.targets)
                    .with_columns(nw.lit(1).alias(sub_moment_column_name))
                    .to_native()
                )
            else:
                m_add.targets = (
                    nw.from_native(m_add.targets)
                    .with_columns(nw.lit(m_add.by_share).alias(sub_moment_column_name))
                    .to_native()
                )

            #   No scaling needed
            if m_add.scale is not None:
                m_add.scale = nw.from_native(m_add.scale).with_columns(
                    nw.lit(m_add.by_share).alias(sub_moment_column_name)
                )

            #   Model matrix - add dummy (1) as this
            if m_add.model_matrix is not None:
                if m_add.rescale:
                    m_add.model_matrix = (
                        nw.from_native(m_add.model_matrix)
                        .with_columns(
                            nw.lit(1 / (m_add.by_share**2)).alias(
                                sub_moment_column_name
                            )
                        )
                        .to_native()
                    )
                else:
                    m_add.model_matrix = (
                        nw.from_native(m_add.model_matrix)
                        .with_columns(nw.lit(1).alias(sub_moment_column_name))
                        .to_native()
                    )

            m_add.columns.append(sub_moment_column_name)

        #   Add the prefix to all the column names
        if prefix != "":
            for itemi in [
                "model_matrix",
                "targets",
                "scale",
                "non_zero",
                "non_zero_target",
            ]:
                dfi = getattr(m_add, itemi)
                if dfi is not None:
                    setattr(
                        m_add,
                        itemi,
                        rename_with_prefix_suffix(
                            df=dfi, prefix=prefix, exclude_list=m_add.index
                        ),
                    )

        #   Also to the list of columns to be calibrated against
        m_add.columns = [prefix + coli for coli in m_add.columns]
        m_add.by_share = 1
        #   If m_out is None, we're done, and m_add is the "combined" moment
        if m_out is None:
            return m_add

        #   If m_out is not None, we need to combined m_out and m_add

        #   Update the name of the added columns, if anything was changed above
        columns_to_add = nw.from_native(m_add.non_zero).lazy().collect_schema().names()

        #   Combine the "data" (df and ModelMatrix)
        if m_out.df is None:
            m_out.df = m_add.df
            m_out.model_matrix = m_add.model_matrix
        else:
            df_add = (
                nw.from_native(
                    join_wrapper(
                        m_add.df,
                        nw.from_native(m_out.df)
                        .select(m_add.index)
                        .with_columns(nw.lit(1).alias("bExist"))
                        .to_native(),
                        on=m_add.index,
                        how="left",
                    )
                )
                .filter(nw.col("bExist").is_missing())
                .drop("bExist")
                .to_native()
            )

            m_out.df = concat_wrapper([m_out.df, df_add], how="diagonal")

            m_out.model_matrix = join_wrapper(
                m_out.model_matrix, m_add.model_matrix, on=m_add.index, how="full"
            )

        #   The summary data
        for itemi in ["targets", "scale", "non_zero", "non_zero_target"]:
            dfi_add = getattr(m_add, itemi)
            dfi_out = getattr(m_out, itemi)
            if dfi_out is None:
                setattr(m_out, itemi, dfi_add)
            else:
                setattr(
                    m_out, itemi, concat_wrapper([dfi_out, dfi_add], how="horizontal")
                )

        m_out.columns.extend(m_add.columns)

        return m_out

    def run(
        self,
        min_obs: int | list[int] = 0,
        skip_setup: bool = False,
        print_diagnostics: bool = True,
        skip_diagnostics: bool = False,
        trim: Trim | None = None,
        **additional_params,
    ) -> dict:
        """
        Run the calibration to estimate the weights from the moments.

        Parameters
        ----------
        min_obs : int | list[int], optional
            Limit the moments to those with more than some number of
            non-zero observations.
            You can pass a list of increasing values so that if
            the model doesn't converge with the lowest number
            (the most constraints), then try again dropping
            constraints with more non-zero observations and try again.
            I.e. if you pass [0,10,100], it will see if the model converges
            with min_obs = 0, if not, it will try with min_obs = 10, etc.
            The default is 0.
        print_diagnostics : bool, optional
            Print the diagnostics (do the weights match the target moments)?
            The default is True.
        skip_diagnostics : bool, optional
            Don't even calculate the diagnostics?
            They can take a while to calculate, but
            you really shouldn't skip this.
            The default is False.
        trim : Trim | None, optional
            Pass a Trim object with bounds on the weights.
            The default is None.
        **additional_params : dict
            Any additional params that are specific to a
            calibration algorithm.

        Returns
        -------
        dict
            A dictionary with diagnostics information including:

            - 'converged' : bool
                Whether the calibration converged.
            - 'max_diff' : float
                Maximum difference between targets and estimates (if diagnostics calculated).
            - 'diagnostics' : DataFrame
                Detailed diagnostics by moment (if skip_diagnostics=False).

        Examples
        --------
        Basic calibration run::

            diagnostics = c.run(min_obs=50)
            print(f"Converged: {diagnostics['converged']}")
            print(f"Max difference: {diagnostics['max_diff']}")

        Run with fallback min_obs levels::

            # Try with no minimum first, then 10, then 100 if previous fails
            diagnostics = c.run(min_obs=[0, 10, 100])

        Run with weight trimming::

            trim_params = Trim(trim=True, min_val=0.1, max_val=5.0)
            diagnostics = c.run(trim=trim_params)
        """

        #  Is min_obs a list of values?
        #       If so, run this recursively on the remaining obs, if it doesn't converge on this one
        min_obs_remaining = None
        if type(min_obs) is list:
            if len(min_obs) > 1:
                min_obs_remaining = min_obs[1 : len(min_obs)]
                min_obs = min_obs[0]
            else:
                min_obs = min_obs[0]
                min_obs_remaining = None

        if min_obs > 0:
            self.censor_to_min_obs(min_obs=min_obs)

        #   Need to loop if trimmed
        #   bLoopNeeded = self.Trimmed
        bLoopNeeded = False

        if self.aggregation == "Sequential":
            #   Sequential - need loop
            bLoopNeeded = True
        elif self.aggregation == "Combined":
            if not skip_setup:
                self.combine_moments(all=True, sub_moments=True)
        elif self.aggregation == "Both":
            if not skip_setup:
                #   Clone this and create a version with the moments combined
                self.c_combined = deepcopy_with_fallback(self)
                self.c_combined.aggregation = "Combined"
                # Do not trim within the combined calibration (handled in self)
                #   self.c_combined.Trimmed = False

                self.c_combined.combine_moments(all=True, sub_moments=True)

        logger.info("Calibrating weights using " + self.method)
        if min_obs > 0:
            logger.info(f"      min obs = {min_obs}")

        if bLoopNeeded:
            n_loops = self.iterations_loop
        else:
            n_loops = 1

        #   Loop counter
        iLoop = 0

        #   Within-loop completion flag
        b_complete = False
        diagnostics = None

        while (iLoop < n_loops) and not b_complete:
            if n_loops > 1:
                logger.info("\n\n\n\nRunning loop #" + str(iLoop + 1))

            diagnostics = self._run_one_loop(trim=trim, **additional_params)
            converged = diagnostics["converged"]

            if "diagnostics" not in list(diagnostics.keys()):
                #   Not full diagnostics, set to null to load later
                diagnostics = None

            #   Default to complete unless set to False below
            b_complete = True
            if trim is not None:
                #   Need trimming?
                #       Don't trim if aebw without separately passed bounds
                if trim.trim and (
                    self.method != "aebw" and "bounds" in additional_params.keys()
                ):
                    b_complete = trim.trim_in_loop(c=self, iLoop=iLoop, n_loops=n_loops)

            if self.aggregation == "Sequential":
                # Check the max deviation against the tolerance
                diagnostics = self.diagnostics()
                max_diff = diagnostics["max_diff"]

                converged = max_diff <= self.Tolerance_Loop
                b_complete = converged

            iLoop += 1

        if not skip_diagnostics:
            if diagnostics is None:
                diagnostics = self.diagnostics()

            if n_loops == 1:
                b_complete = diagnostics["max_diff"] <= self.tolerance
            else:
                b_complete = diagnostics["max_diff"] <= self.tolerance

            if not (b_complete or converged) and (min_obs_remaining is not None):
                #   Recursively call this again, if needed

                #   Reset the "final weight" (current weight) to the original one passed
                #       (ignores the failed run's output)
                self.df = (
                    nw.from_native(self.df)
                    .with_columns(nw.col(self.weight).alias(self.final_weight))
                    .to_native()
                )

                diagnostics = self.run(
                    min_obs=min_obs_remaining,
                    skip_setup=True,
                    print_diagnostics=print_diagnostics,
                    skip_diagnostics=skip_diagnostics,
                    **additional_params,
                )

            else:
                diagnostics["converged"] = converged

            if "diagnostics" in diagnostics.keys():
                diagnostics["diagnostics"] = (
                    nw.from_native(diagnostics["diagnostics"])
                    .lazy_backend(self.nw_type)
                    .to_native()
                )
            self.diagnostics_out = diagnostics

            if print_diagnostics:
                self.print_diagnostics()

        return diagnostics

    def censor_to_min_obs(self, min_obs: int = 0) -> None:
        """
        In calibration, do we want to limit to moments with greater than some
           threshold of observations where x != 0

        Parameters
        ----------
        min_obs : int, optional
            The minimum number of != 0 observations.
            The default is 0. (don't censor)

        Returns
        -------
        None

        """
        for momenti in self.moments:
            self._censor_to_min_obs_by_moment(min_obs=min_obs, m=momenti)

            for subi in momenti.sub_moments:
                self._censor_to_min_obs_by_moment(min_obs=min_obs, m=subi)

        if self.c_combined is not None:
            self.c_combined.censor_to_min_obs(min_obs=min_obs)

    def _censor_to_min_obs_by_moment(self, min_obs: int, m: Moment) -> None:
        """
        For each moment, censor, if needed

        Parameters
        ----------
        min_obs : int
            Minimum non-zero observations to censor at
        m : Moment
            Moment to censor

        Returns
        -------
        None
        """

        if m.non_zero_target is not None and len(m.columns) > 0:
            nw_type = NarwhalsType(m.non_zero_target)
            nonzero_cols = (
                nw_type.to_polars().select(m.columns).transpose(include_header=True)
            )

            m.columns = nonzero_cols.filter(
                pl.col(nonzero_cols.lazy().collect_schema().names()[1]) >= min_obs
            )[nonzero_cols.lazy().collect_schema().names()[0]].to_list()

        if m.non_zero is not None and len(m.columns):
            nw_type = NarwhalsType(m.non_zero)
            nonzero_cols = (
                nw_type.to_polars().select(m.columns).transpose(include_header=True)
            )

            m.columns = nonzero_cols.filter(
                pl.col(nonzero_cols.lazy().collect_schema().names()[1]) >= min_obs
            )[nonzero_cols.lazy().collect_schema().names()[0]].to_list()

    def _run_one_loop(self, trim: Trim | None = None, **additional_params):
        if additional_params is None:
            additional_params = {}

        #   Delegate for actual calibration call
        fCalibrate = None

        additional_list = []

        if self.method == "legacy_ebw":
            fCalibrate = self._legacy_ebw

        elif self.method == "aebw":
            fCalibrate = self._accelerated_ebw_moment

            #   Bound ratio adjustment to ebw
            additional_list.append("bounds")

            #   Constraint violation penalty parameter
            additional_list.append("eta")

            #   If passed, Sanders's version of Hainmueller's EBW
            additional_list.append("dual_only")
            additional_list.append("dense")
            additional_list.append("scale_weights_to_n")
            additional_list.append("fallback_bounded")
            additional_list.append("fallback_iterations")

            if trim is not None:
                if "bounds" in additional_params.keys():
                    logger.info("Bounds passed directly, ignoring trim parameters: ")
                    logger.info(trim)
                else:
                    logger.info("Do I want to pass trim bounds to the calibration?")
                    # additional_params["bounds"] = (trim.min_val,
                    #                                trim.max_val)

        #   Keep only relevant parameters
        additional_params = {
            keyi: valuei
            for keyi, valuei in additional_params.items()
            if keyi in additional_list
        }

        #   Create an empty diagnostics variable - some aggregations populate this
        diagnostics = None

        if self.aggregation == "Combined":
            converged = self._run_combined(fCalibrate=fCalibrate, **additional_params)

        elif self.aggregation == "Sequential":
            self._run_sequential(fCalibrate=fCalibrate, **additional_params)

            #   This is just passed, convergence is checked later
            #       through the diagnostics
            converged = True

        elif self.aggregation == "Both":
            self._run_sequential(fCalibrate=fCalibrate, **additional_params)
            diagnostics = self.diagnostics()

            if diagnostics["max_diff"] > self.Tolerance:
                logger.info(
                    "     Did not converge for sequential, running combined calibration"
                )

                cols_df = []
                cols_df.extend(self.index)
                cols_df.append(self.final_weight)
                self.c_combined.df = join_wrapper(
                    nw.from_native(self.c_combined.df)
                    .drop(self.final_weight)
                    .to_native(),
                    nw.from_native(self.df).select(cols_df).to_native(),
                    on=self.index,
                    how="left",
                )

                self.c_combined.run(**additional_params)
                #   Get the final diagnostics from the combined run
                diagnostics = self.c_combined.diagnostics_out

                # Update the weights back
                self.df = join_wrapper(
                    nw.from_native(self.df).drop(self.final_weight).to_native(),
                    nw.from_native(self.c_combined.df).select(cols_df).to_native(),
                    on=self.index,
                    how="left",
                )
            else:
                logger.info(
                    "     Converged for sequential, skipping combined calibration"
                )

        if diagnostics is None:
            diagnostics = {"converged": converged}

        return diagnostics

    def _run_combined(self, fCalibrate=None, **additional_params):
        m = self.moments[0]

        if m.model_matrix is not None:
            logger.info("     Calibration using combined moments")
            converged = fCalibrate(m=m, **additional_params)

        return converged

    def _run_sequential(self, fCalibrate=None, **additional_params):
        for i_moment, m in enumerate(self.moments):
            logger.info(f"    Moment: {i_moment + 1}")

            if m.ModelMatrix is not None:
                converged = fCalibrate(m=m, sequential=True, **additional_params)

            for i_sub, subi in enumerate(m.sub_moments):
                logger.info(f"    Moment: {i_moment}.{i_sub + 1}")

                if subi.model_matrix is not None:
                    converged = fCalibrate(m=subi, sequential=True, **additional_params)

        return converged

    def _legacy_ebw(self, m: Moment = None, sequential=False):
        [initial_guess, base_weight, xi, targetsi, tol_adj] = self._moment_inputs(
            m=m, sequential=sequential
        )

        nw_type = NarwhalsType(xi)

        #   Initial weights
        if base_weight is None:
            base_weight = (
                nw.from_native(initial_guess).lazy().collect().to_numpy().ravel()
            )
            initial_guess = None

            #   aebw_options["initial_ratio_guess"] = base_weight
        else:
            base_weight = (
                nw.from_native(base_weight).lazy().collect().to_numpy().ravel()
            )
            initial_guess = (
                nw.from_native(initial_guess).lazy().collect().to_numpy().ravel()
            )

            #   aebw_options["initial_ratio_guess"] = initial_guess

        xi = (
            nw.from_native(xi)
            .with_columns(
                [cs.boolean().cast(nw.Float32), cs.numeric().cast(nw.Float64)]
            )
            .to_native()
        )

        xi = (
            nw.from_native(xi)
            .with_columns(nw.all().cast(nw.Float64))
            .lazy()
            .collect()
            .to_numpy()
        )

        targetsi = nw.from_native(targetsi).lazy().collect().to_numpy().ravel()

        b_converged = False

        try:
            results = legacy_calibration(
                mean_population_moments=targetsi,
                x_sample=xi,
                weights0=base_weight,
                penalty_fn="log_diff",
                rank_regularization_term=self.tolerance * tol_adj,
                initial_guess=initial_guess,
                n_iterations=self.iterations,
            )

            #   Make sure the print log goes with the code that executed it (rather than at the end of the log)
            print("", flush=True)

            b_converged = results[1] == "success"

            w_out = nw.from_native(
                nw.from_arrow(
                    pl.from_numpy(
                        results[0], schema={"column_0": pl.Float64}
                    ).to_arrow(),
                    backend=backend_eager(nw_type.backend),
                )
            )

        except Exception as e:
            logger.error(e)

        if w_out is not None:
            self._merge_on_weights(m=m, weights=w_out, sequential=sequential)

        return b_converged

    def _accelerated_ebw_moment(
        self,
        m: Moment,
        sequential: bool = False,
        dense: bool = True,
        bounds: tuple[float, float] | None = None,
        #  eta=None,
        fallback_bounded: bool = False,
        fallback_iterations: int = -1,
        dual_only: bool = False,
        only_bounds: bool = False,
        scale_weights_to_n: bool = True,
    ) -> bool:
        """
        Call accelerated ebw calibration code
            (https://github.t26.it.census.gov/sande440/entropy-balance-weighting)

        Parameters
        ----------
        m : Moment
            Moment to be calibrated to
        sequential : TYPE, optional
            Is this part of a sequential optimization. The default is False.
        dense : bool, optional
            Dense matrix (vs. sparse). The default is True.
        bounds : tuple(float,float) | None, optional
            tuple of lower, upper bound.
            If bounds are set, the "elastic" calibration is called,
            which tries to match as well as possible, but can
            handle unsatisfiable constraints
            (by trying to get as closs as possible)
            Can be (0,None) for no bounds elastic calibration.
            The default is None.
        #   eta : TYPE, optional
        #       Elastic calibration penalty parameter. The default is None.
        fallback_bounded : bool, optional
            If unbounded calibration fails to converge, fallback to bounded
            for "as close as possible" calibration
            with no bounds (bounds=[epsilon_weight_lower_bound,None])
            The default is False.
        fallback_iterations : int, optional
            Number of iterations if falling back to elastic, bounded optimization.
            The default is self.Iterations
        dual_only : bool, optional
            Use dual only (Hainmueller-like) optimization. The default is False.
        only_bounds : bool, optional
            If you have bounds, do  you want to start with the
            bounds or see if the normal procedure satisfies the bounds?
            If true, start with the bounded elastic,
            if False, trie without the bounds and verify if they are satisfied.
            The default is False.
        scale_weights_to_n : bool, optional
            Scaling weights to sum to n of sample seems
            to run faster, so do that? The default is True.

        Returns
        -------
        b_converged : boolean
            Converged?  It also sets the weights in self

        """
        epsilon_weight_lower_bound = 10e-5

        if not hasattr(np, "concat"):
            np.concat = np.concatenate

        [initial_guess, base_weight, xi, targetsi, tol_adj] = self._moment_inputs(
            m=m, sequential=sequential
        )

        nw_type = NarwhalsType(xi)

        n_weights = safe_height(initial_guess)
        aebw_options = dict(dense=dense, dual_only=dual_only)

        # if eta is not None:
        #     aebw_options["eta"] = eta

        #   If you pass in [0,None], then I'm assuming only_bounds
        #       because that is unbounded
        if bounds is not None:
            if bounds[0] <= epsilon_weight_lower_bound and bounds[1] is None:
                only_bounds = True

            if only_bounds:
                aebw_options["bounds"] = bounds

        # WriteParquet(Xi,
        #               "/projects/data/NEWS/TempFiles/x_sample.parquet")
        # WriteParquet(initial_guess,
        #               "/projects/data/NEWS/TempFiles/weights0.parquet")
        # WriteParquet(Targetsi,
        #               "/projects/data/NEWS/TempFiles/mean_population_moments.parquet")

        #   Initial weights
        if base_weight is None:
            base_weight = (
                nw.from_native(initial_guess).lazy().collect().to_numpy().ravel()
            )
            initial_guess = None

            #   aebw_options["initial_ratio_guess"] = base_weight
        else:
            base_weight = (
                nw.from_native(base_weight).lazy().collect().to_numpy().ravel()
            )
            initial_guess = (
                nw.from_native(initial_guess).lazy().collect().to_numpy().ravel()
            )

            #   aebw_options["initial_ratio_guess"] = initial_guess

        xi = (
            nw.from_native(xi)
            .with_columns(
                [cs.boolean().cast(nw.Float32), cs.numeric().cast(nw.Float64)]
            )
            .to_native()
        )

        if dense:
            xi = (
                nw.from_native(xi)
                .with_columns(nw.all().cast(nw.Float64))
                .lazy()
                .collect()
                .to_numpy()
            )

        else:
            xi = sp.csc_array(
                nw.from_native(xi)
                .with_columns(nw.all().cast(nw.Float64))
                .lazy()
                .collect()
                .to_numpy()
            )

        targetsi = nw.from_native(targetsi).lazy().collect().to_numpy().ravel()

        if scale_weights_to_n:
            sum_weights = base_weight.sum()

            adjustment = n_weights / sum_weights
            base_weight = base_weight * adjustment
            if initial_guess is not None:
                initial_guess = initial_guess * adjustment

        b_converged = False

        results = None
        w_out = None

        try:
            aebw_options["max_steps"] = self.iterations
            aebw_options["optimality_violation"] = self.tolerance * tol_adj
            #   aebw_options["save_problem_data"] = "/projects/data/NEWS/TempFiles/ebw.zip"
            results = ebw_routines.entropy_balance(
                mean_population_moments=targetsi,
                x_sample=xi,
                weights0=base_weight,
                options=aebw_options,
            )

            #   Make sure the print log goes with the code that executed it (rather than at the end of the log)
            print("", flush=True)

            b_converged = results.converged
        except Exception as e:
            logger.error("Failure of initial calibration")
            logger.error(e)

        check_bounds = not only_bounds and (bounds is not None)
        rerun_with_bounds = False
        if b_converged:
            w_out = nw.from_native(
                nw.from_arrow(
                    pl.from_numpy(
                        results.new_weights, schema={"column_0": pl.Float64}
                    ).to_arrow(),
                    backend=backend_eager(nw_type.backend),
                )
            )

        ratio = None
        if check_bounds:
            #   It converged, but are we finished?
            df_base_weight = nw.from_arrow(
                pl.from_numpy(base_weight, schema={"base": pl.Float64}).to_arrow(),
                backend=backend_eager(nw_type.backend),
            )

            c_ratio = nw.col("new") / nw.col("base")
            ratio = nw.from_native(
                concat_wrapper(
                    [df_base_weight, w_out.rename({"column_0": "new"})],
                    how="horizontal",
                )
            ).select([c_ratio.min().alias("min"), c_ratio.max().alias("max")])

            min_ratio = ratio.select("min").item(0, 0)
            max_ratio = ratio.select("max").item(0, 0)

            lower_bound = bounds[0]
            upper_bound = bounds[1]

            if lower_bound is not None and lower_bound > 0:
                rerun_with_bounds = lower_bound > min_ratio
            if not rerun_with_bounds and upper_bound is not None and upper_bound > 0:
                rerun_with_bounds = rerun_with_bounds or (upper_bound < max_ratio)

            if rerun_with_bounds:
                logger.info("\n\nBounds satisfied:")
                logger.info(f"     bounds = {bounds}")
                logger.info(f"     limits = {[min_ratio, max_ratio]}")

        if (not b_converged and fallback_bounded) or rerun_with_bounds:
            #   Failed, did you use bounds the first time?
            #       Bounds will run the "as close as possible" version
            if bounds is None:
                bounds = [epsilon_weight_lower_bound, None]
            aebw_options["bounds"] = bounds
            logger.info(f"\n\nAttempting to calibrate with bounds: {bounds}")

            #   Set the initial guess
            ratio = None
            # if w_out is not None:
            #     df_base_weight = pl.from_numpy(base_weight,
            #                                     schema={"base":pl.Float64})

            #     c_ratio = pl.col("new")/pl.col("base")
            #     ratio = (pl.concat([df_base_weight,
            #                         w_out.rename({"column_0":"new"})],
            #                         how="horizontal")
            #               .select(c_ratio)
            #               )

            #   Initial guess doesn't work for bounded value: line 434 (inv_h_sqrt = sp..)
            # if ratio is not None:
            #     aebw_options["initial_ratio_guess"] = ratio.to_numpy().ravel()
            if fallback_iterations > 0:
                aebw_options["max_steps"] = fallback_iterations

            results = ebw_routines.entropy_balance_elastic(
                mean_population_moments=targetsi,
                x_sample=xi,
                weights0=base_weight,
                options=aebw_options,
            )

            b_converged = results.converged
            w_out = nw.from_native(
                nw.from_arrow(
                    pl.from_numpy(
                        results.new_weights, schema={"column_0": pl.Float64}
                    ).to_arrow(),
                    backend=backend_eager(nw_type.backend),
                )
            )
            # except Exception as e:
            #     logger.error(f"Failure of calibration with bounds: {bounds}")
            #     logger.error(e)

        if w_out is not None:
            self._merge_on_weights(m=m, weights=w_out, sequential=sequential)

        return b_converged

    def _moment_inputs(self, m: Moment = None, sequential=False, add_intercept=True):
        #   Initial weights
        cols_weights = [self.final_weight]
        if self.initial_guess != "":
            cols_weights.append(self.initial_guess)
        cols_df = self.index + cols_weights

        qi = (
            nw.from_native(
                join_wrapper(
                    nw.from_native(m.model_matrix).select(m.index).to_native(),
                    nw.from_native(self.df).select(cols_df).to_native(),
                    on=self.index,
                    how="left",
                )
            )
            .sort(m.index)
            .select(cols_weights)
            .to_native()
        )

        qi = (
            nw.from_native(safe_sum_cast(qi))
            .with_columns([nw.col(coli) / nw.col(coli).sum() for coli in cols_weights])
            .to_native()
        )

        if self.initial_guess != "":
            qi_base = qi.select(self.final_weight)
            qi = qi.select(self.initial_guess)

            qi_base = (
                nw.from_native(safe_sum_cast(df=qi_base, columns=[self.final_weight]))
                .with_columns(
                    (nw.col(self.final_weight) / nw.col(self.final_weight).sum()).alias(
                        self.final_weight
                    )
                )
                .to_native()
            )
        else:
            qi_base = None
        col_qi = nw.from_native(qi).lazy().collect_schema().names()[0]
        qi = nw.from_native(qi)
        qi = (
            nw.from_native(safe_sum_cast(df=qi, columns=col_qi))
            .with_columns(
                (nw.col(col_qi) / nw.col(col_qi).sum()).alias(
                    qi.lazy().collect_schema().names()[0]
                )
            )
            .to_native()
        )

        xi = m.rescaled_model_matrix(narrow=True)

        # Targets need to be adjusted for submoments
        targetsi = (
            nw.from_native(m.targets)
            .select(
                nw.col(nw.from_native(xi).lazy().collect_schema().names()) / m.by_share
            )
            .to_native()
        )

        #   Adjust tolerance for subgroup byshare to match actual passed tolerance
        if sequential:
            tol_adj = m.by_share
        else:
            tol_adj = 1

        if add_intercept:
            xi = (
                nw.from_native(xi)
                .with_columns(nw.lit(1).alias("___weighting_intercept___"))
                .to_native()
            )

            targetsi = (
                nw.from_native(targetsi)
                .with_columns(nw.lit(1).alias("___weighting_intercept___"))
                .to_native()
            )

        return [qi, qi_base, xi, targetsi, tol_adj]

    def _merge_on_weights(
        self, m: Moment, weights: IntoFrameT, sequential: bool = False
    ):
        temp_name = self.final_weight + "___temp___"
        weights = nw.from_native(weights)
        weights = safe_sum_cast(
            weights.rename({weights.lazy().collect_schema().names()[0]: temp_name}),
            columns=temp_name,
        )

        #   Adjust to share in this group (relative to total weight in this group vs. overall)
        #       relative to total weights that sum to the n in the sample
        n_obs = safe_height(weights)
        weights = (
            nw.from_native(weights)
            .with_columns(
                (nw.col(temp_name) / nw.col(temp_name).sum() * n_obs).alias(temp_name)
            )
            .to_native()
        )
        #   .rename({wOut.columns[0]:self.final_weight}

        if sequential:
            weights = nw.from_native(weights).with_columns(
                nw.all() * m.by_share / n_obs / safe_height(self.df)
            )

            #   Merge by index
            weights = pl.concat(
                [
                    nw.from_native(m.model_matrix)
                    .select(self.index)
                    .collect()
                    .to_native(),
                    weights,
                ],
                how="horizontal",
            )
            self.df = (
                nw.from_native(
                    join_wrapper(self.df, weights, on=self.index, how="left")
                )
                .with_columns(
                    nw.when(nw.col(temp_name).is_not_missing())
                    .then(nw.col(temp_name))
                    .otherwise(nw.col(self.final_weight))
                    .alias(self.final_weight)
                )
                .drop(temp_name)
                .lazy_backend(self.nw_type)
                .to_native()
            )
        else:
            #   Concatenate, it's faster (since the data's sorted and the same length)
            self.df = (
                concat_wrapper(
                    [
                        nw.from_native(self.df)
                        .drop(self.final_weight)
                        .lazy()
                        .collect(),
                        nw.from_native(weights).rename({temp_name: self.final_weight}),
                    ],
                    how="horizontal",
                )
                .lazy_backend(self.nw_type)
                .to_native()
            )

        # diagi = self._diagnostics_moment(m=m,
        #                                  with_by_prefix=False,
        #                                  with_count_prefix=False,
        #                                  count_value=False)

        # logger.info(diagi)

    def diagnostics(self):
        diag = None

        with_count_prefix = len(self.moments) > 1
        for i_moment, m in enumerate(self.moments):
            if m.targets is not None:
                diagi = self._diagnostics_moment(
                    m=m,
                    with_by_prefix=False,
                    with_count_prefix=with_count_prefix,
                    count_value=i_moment,
                )

                if diag is None and diagi is not None:
                    diag = diagi
                else:
                    diag = concat_wrapper([diag, diagi], how="vertical")

            for subi in m.sub_moments:
                diagi = self._diagnostics_moment(
                    m=subi,
                    with_by_prefix=True,
                    with_count_prefix=with_count_prefix,
                    count_value=i_moment,
                )

                if diag is None and diagi is not None:
                    diag = diagi
                else:
                    diag = concat_wrapper([diag, diagi], how="vertical")

        diag = (
            nw.from_native(diag)
            .filter((nw.col("Estimates").is_not_missing()))
            .with_columns((nw.col("Estimates") - nw.col("Targets")).alias("diff"))
            .with_columns(
                (100 * (nw.col("Estimates") / nw.col("Targets") - 1)).alias("percent")
            )
            .to_native()
        )

        diag = compress_df(diag, no_boolean=True)
        max_diff = (
            nw.from_native(diag)
            .filter(nw.col("Calibrated") == 1)
            .with_columns((nw.col("percent").abs() / 100).alias("abs_diff"))
            .select(nw.col("abs_diff").max())
            .lazy()
            .collect()
            .item(0, 0)
        )

        return {"diagnostics": diag, "max_diff": max_diff}

    def _diagnostics_moment(
        self,
        m: Moment,
        with_by_prefix: bool = False,
        with_count_prefix: bool = False,
        count_value: int = 0,
    ):
        if m.model_matrix is None:
            #   Do nothing
            return None

        #   Calculate the stats on model matrix
        #       Rescale the data (if necessary)
        cols_df = []
        cols_df.extend(self.index)
        cols_df.append(self.weight)
        cols_df.append(self.final_weight)

        df_statsin = join_wrapper(
            m.rescaled_model_matrix(), self.df, on=m.index, how="left"
        )

        #       What are we calculating
        c_exclude = [m.weight, self.final_weight]
        c_exclude.extend(m.index)

        cols_in_mm = [
            coli
            for coli in nw.from_native(m.model_matrix).lazy().collect_schema().names()
            if coli not in m.index
        ]
        column_stats = column_stats_builder(
            cols_include=cols_in_mm,
            df=m.model_matrix,
            cols_exclude=c_exclude,
            stat="mean",
        )
        #       Final weighted estimates
        df_estimates = (
            nw.from_native(
                calculate_by(
                    df=df_statsin,
                    column_stats=column_stats,
                    weight=self.final_weight,
                    no_suffix=True,
                )
            )
            .with_columns(cs.numeric() * m.by_share)
            .with_columns(nw.lit("Estimates").alias("Column"))
            .to_native()
        )
        df_estimates = df_estimates

        #       Initial values
        df_initial = (
            nw.from_native(
                calculate_by(
                    df=df_statsin,
                    column_stats=column_stats,
                    weight=self.weight,
                    no_suffix=True,
                )
            )
            .with_columns(cs.numeric() * m.by_share)
            .with_columns(nw.lit("Initial").alias("Column"))
            .to_native()
        )

        #   Return some basic info with the diagnostics
        df_targets = (
            nw.from_native(m.targets)
            .with_columns(nw.lit("Targets").alias("Column"))
            .to_native()
        )

        #   Non-zero counts in the weighted data
        df_non_zero = (
            nw.from_native(m.non_zero)
            .with_columns(nw.lit("NonZero").alias("Column"))
            .to_native()
        )

        #   Non-zero counts in the target data
        df_non_zero_target = (
            nw.from_native(m.non_zero_target)
            .with_columns(nw.lit("NonZero_Target").alias("Column"))
            .to_native()
        )

        #   What variables were actually used in the calibration?
        #       It seems easiest to use the one row NonZero vector as the basis
        cols_m_no_index = [ci for ci in m.columns if ci not in m.index]
        c_ones = [nw.lit(1).alias(coli) for coli in cols_m_no_index]
        c_zeros = [
            nw.lit(0).alias(coli) for coli in cols_in_mm if coli not in cols_m_no_index
        ]

        df_calibrated = (
            nw.from_native(m.non_zero)
            .select(cols_m_no_index)
            .with_columns(c_ones)
            .with_columns(c_zeros)
            .with_columns(nw.lit("Calibrated").alias("Column"))
            .to_native()
        )

        #   Stack the results
        df_diagnostics = concat_wrapper(
            [
                df_initial,
                df_targets,
                df_estimates,
                df_calibrated,
                df_non_zero,
                df_non_zero_target,
            ],
            how="diagonal",
        )

        #   Add the prefix?
        prefix = self._combine_prefix(
            with_by_prefix=with_by_prefix,
            with_count_prefix=with_count_prefix,
            count_value=count_value,
            m=m,
        )
        if prefix != "":
            df_diagnostics = rename_with_prefix_suffix(
                df=df_diagnostics, prefix=prefix, ExcludeList=["Column"]
            )

        #   Reorder and transpose the diagnostics to be easier to read
        colorder = (
            nw.from_native(df_diagnostics)
            .drop("Column")
            .lazy()
            .collect_schema()
            .names()
        )
        nw_type = NarwhalsType(df_diagnostics)
        df_diagnostics = nw_type.to_polars().lazy().collect()
        df_diagnostics = nw_type.from_polars(
            df_diagnostics.select(colorder)
            .transpose(
                include_header=True, column_names=df_diagnostics["Column"].to_list()
            )
            .rename({"column": "Variable"})
        )

        return nw.from_native(df_diagnostics).lazy_backend(nw_type).to_native()

    def print_diagnostics(
        self,
        max_columns: int = 100,
        calibrated_only: bool = False,
        min_non_zero: int = 10,
        sort: list = None,
        descending: list = None,
    ):
        """
        Print formatted calibration diagnostics to the logger.

        Parameters
        ----------
        max_columns : int, optional
            Maximum number of rows to display. Default is 100.
        calibrated_only : bool, optional
            Only show moments that were actually calibrated to. Default is False.
        min_non_zero : int, optional
            Minimum non-zero observations required to display a moment. Default is 10.
        sort : list, optional
            Column names to sort by. Default is ["Calibrated", "abs", "NonZero"].
        descending : list, optional
            Sort order for each column in sort. Default is [True, True, True].
        """

        diag = self.diagnostics_out

        if sort is None and descending is None:
            sort = ["Calibrated", "abs", "NonZero"]
            descending = [True, True, True]

        #   Some polars config code to make the diagnostic readable
        logger.info("Converged:          " + str(diag["converged"]))
        logger.info("Maximum Difference: " + str(diag["max_diff"]))

        diagnostics = diag["diagnostics"]
        diagnostics = NarwhalsType(diagnostics).to_polars()

        if calibrated_only:
            diagnostics = diagnostics.filter(pl.col.Calibrated == 1)

        diagnostics = diagnostics.lazy().collect()
        #   Only show NonZero_Target if it not identical to NonZero
        if diagnostics.select("NonZero").equals(
            diagnostics.select("NonZero_Target").rename({"NonZero_Target": "NonZero"})
        ):
            diagnostics = diagnostics.drop("NonZero_Target")

        with pl.Config(fmt_str_lengths=50) as cfg:
            #   Basic formatting
            cfg.set_tbl_cell_alignment("RIGHT")
            cfg.set_tbl_hide_column_data_types(True)
            cfg.set_tbl_width_chars(600)
            cfg.set_tbl_cols(len(diagnostics.lazy().collect_schema().names()))

            #   Show all the rows (up to 100)
            cfg.set_tbl_rows(min(max_columns, diagnostics.height))

            #   Don't show tiny moments

            if descending is not None:
                d_descending = {"descending": descending}
            else:
                d_descending = {}

            logger.info(
                diagnostics.filter(pl.col.NonZero >= min_non_zero)
                .with_columns((pl.col("diff").abs().alias("abs")))
                .sort(sort, **d_descending)
                .drop("abs")
            )

    def get_final_weights(
        self,
        df_merge_to: IntoFrameT | None = None,
        truncate_low: float | None = None,
        truncate_high: float | None = None,
    ) -> IntoFrameT:
        """
        Get the final calibrated weights, optionally merged to another dataframe.

        Parameters
        ----------
        df_merge_to : IntoFrameT | None, optional
            Dataframe to merge weights to. If None, returns weights with index only.
            Default is None.
        truncate_low : float | None, optional
            Truncate weights below this value. Default is None.
        truncate_high : float | None, optional
            Truncate weights above this value. Default is None.

        Returns
        -------
        IntoFrameT
            Dataframe with calibrated weights. If df_merge_to provided, returns that
            dataframe with weights merged on index. Otherwise returns index columns
            and final weight column only.
        """

        col_weights = []
        col_weights.extend(self.index)
        col_weights.append(self.final_weight)
        df_weights = nw.from_native(self.df).select(col_weights).to_native()

        c_final_weight = nw.col(self.final_weight)
        truncate = None
        if truncate_low is not None:
            #   N to truncate?
            expr_low = c_final_weight.lt(truncate_low)
            n_truncate_low = safe_height(
                nw.from_native(df_weights).filter(expr_low).to_native()
            )
            logger.info(f"     n truncated at {truncate_low} = {n_truncate_low}")

            if n_truncate_low:
                truncate = nw.when(expr_low).then(nw.lit(truncate_low))
        if truncate_high is not None:
            #   N to truncate?
            expr_high = c_final_weight.gt(truncate_high)
            n_truncate_high = safe_height(
                nw.from_native(df_weights).filter(expr_high).to_native()
            )

            logger.info(f"     n truncated at {truncate_high} = {n_truncate_high}")

            if n_truncate_high:
                if truncate is not None:
                    base = truncate
                else:
                    base = nw

                truncate = base.when(expr_high).then(nw.lit(truncate_high))

        if truncate is not None:
            truncate = truncate.otherwise(nw.col(self.final_weight)).alias(
                self.final_weight
            )

            df_weights = nw.from_native(df_weights).with_columns(truncate).to_native()

        if df_merge_to is not None:
            if self.index == ["___rownumber"]:
                #   Just concatenate, we made the index internally
                df_weights = concat_wrapper(
                    [
                        df_merge_to,
                        (
                            nw.from_native(df_weights)
                            .sort(self.index)
                            .select(self.final_weight)
                            .to_native()
                        ),
                    ],
                    how="horizontal",
                )
            else:
                #   Merge on the index (safer - doesn't require assumption neither have changed)
                df_weights = join_wrapper(
                    df_merge_to, df_weights, on=self.index, how="left"
                )

        return df_weights
