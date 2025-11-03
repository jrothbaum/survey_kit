from __future__ import annotations
from typing import TYPE_CHECKING

import narwhals as nw

from ..utilities.dataframe import safe_height
from ..serializable import Serializable

from .. import logger

if TYPE_CHECKING:
    from .calibration import Calibration


class Trim(Serializable):
    _save_suffix = "calibration_trim"

    def __init__(
        self,
        trim: bool = False,
        min_val: float = 0.05,
        max_val: float = 5.0,
        step: float = 0.02,
        tolerance_step: float = 0.01,
        ignore_n: int = 0,
    ):
        self.trim = trim
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.tolerance_step = tolerance_step
        self.ignore_n = ignore_n

    def trim_in_loop(self, c: Calibration, iLoop: int, nLoops: int):
        # Complete unless set to false later
        bComplete = True

        #   Check if the weights need trimming (not on last loop)
        if self.trim and (iLoop - 1) < nLoops:
            # Check min and max weight
            max_weight = (
                nw.from_native(c.df)
                .select(nw.col(c.final_weight).max())
                .collect()
                .item(0, 0)
            )
            min_weight = (
                nw.from_native(c.df)
                .select(nw.col(c.final_weight).min())
                .collect()
                .item(0, 0)
            )

            trim_limit_max = self.max_val * (1 + iLoop * self.tolerance_step)
            trim_limit_min = self.min_val * (1 - iLoop * self.tolerance_step)
            btrim_max = False
            btrim_min = False
            n_to_trim_max = 0
            n_to_trim_min = 0

            if max_weight > trim_limit_max:
                if self.ignore_n > 0:
                    n_to_trim_max = safe_height(
                        nw.from_native(c.df).filter(
                            nw.col(c.final_weight) > trim_limit_max
                        )
                    )

                    btrim_max = n_to_trim_max > self.ignore_n

                    if not btrim_max:
                        logger.info(
                            f"Max weight ({max_weight}) is greater than trim_max({trim_limit_max}), but NOT TRIMMING because only {n_to_trim_max}  need trimming against a passed limit of {self.ignore_n}"
                        )
            else:
                btrim_max = True

            if min_weight < trim_limit_min:
                if self.ignore_n > 0:
                    n_to_trim_max = safe_height(
                        nw.from_native(c.df).filter(
                            nw.col(c.final_weight) < trim_limit_min
                        )
                    )

                    btrim_min = n_to_trim_min > self.ignore_n

                    if not btrim_min:
                        logger.info(
                            f"Min weight ({min_weight}) is greater than trim_max({trim_limit_min}), but NOT TRIMMING because only {n_to_trim_min}  need trimming against a passed limit of {self.ignore_n}"
                        )
                    else:
                        btrim_min = True

            if btrim_max:
                bComplete = False

                trim_at = self.max_val * (1 - iLoop * self.step)

                logger.info(
                    f"Max weight ({max_weight}) is greater than trim_max({trim_limit_max}), trimming to {trim_at})"
                )

                if n_to_trim_max > 0:
                    logger.info(f"     {n_to_trim_max} need trimming")

                c.df = (
                    nw.from_native(c.df)
                    .with_columns(
                        (
                            nw.when(nw.col(c.final_weight) > trim_at)
                            .then(nw.lit(trim_at))
                            .otherwise(nw.col(c.final_weight))
                            .alias(c.final_weight)
                        )
                    )
                    .to_native()
                )

            if btrim_min:
                bComplete = False
                trim_at = self.min_val * (1 + iLoop * self.step)

                logger.info(
                    f"Min weight ({min_weight}) is greater than trim_max({trim_limit_min}), trimming to {trim_at})"
                )

                if n_to_trim_min > 0:
                    logger.info(f"     {n_to_trim_min} need trimming")

                c.df = (
                    nw.from_native(c.df)
                    .with_columns(
                        (
                            nw.when(nw.col(c.final_weight) < trim_at)
                            .then(nw.lit(trim_at))
                            .otherwise(nw.col(c.final_weight))
                            .alias(c.final_weight)
                        )
                    )
                    .to_native()
                )

        return bComplete

    def _str__(self):
        return (
            f"trim           = {self.trim}"
            + "\n"
            + f"min_val        = {self.min_val}"
            + "\n"
            + f"min_val        = {self.min_val}"
            + "\n"
            + f"max_val        = {self.max_val}"
            + "\n"
            + f"step           = {self.step}"
            + "\n"
            + f"tolerance_step = {self.tolerance_step}"
            + "\n"
            + f"ignore_n       = {self.ignore_n}"
        )
