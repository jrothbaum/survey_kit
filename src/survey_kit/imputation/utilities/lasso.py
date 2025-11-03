from __future__ import annotations
from typing import Optional

import os
import numpy as np
import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT
from enum import Enum
import lightgbm as lgb
import formulaic

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso as sk_lasso
from sklearn.model_selection import train_test_split

from copy import deepcopy

from ...utilities.random import set_seed, generate_seed
from ...utilities.dataframe import (
    safe_sum_cast,
    columns_from_list,
    concat_wrapper,
    NarwhalsType,
)
from ...utilities.formula_builder import FormulaBuilder
from ...statistics.basic_calculations import _mean, _std


class Lasso:
    def __init__(
        self,
        df: IntoFrameT,
        y: str = "",
        x: list | str | None = None,
        weight: str = "",
        formula: str | list[str] = "",
        nfolds: int = 5,
        optimal_lambda: float | None = None,
        max_iter: int = 10_000,
        seed: int = 0,
        #  check_rank:bool=True
    ):
        if x is None:
            x = []
        if type(x) is str:
            x = [x]

        self.df = nw.from_native(df).filter(nw.col(y).is_not_missing()).to_native()

        self.y = y
        self.x = x
        self.weight = weight
        self.formula = formula
        self.nfolds = nfolds
        self.max_iter = max_iter
        self.optimal_lambda = optimal_lambda

        if seed == 0:
            seed = generate_seed()
        self.seed = seed
        # self.check_rank = check_rank

        self.process_formula()

    def process_formula(self):
        #   Only do the processing, if it's needed
        if self.formula != "":
            if type(self.formula) is str:
                self._process_formula_string()
            elif type(self.formula) is list:
                self._process_formula_list()

        with_standardize = []
        if self.weight != "":
            self.df = safe_sum_cast(df=self.df, columns=[self.weight])
        for vari in self.x:
            c_vari = nw.col(vari)

            with_standardize.append(
                (
                    (c_vari - _mean(column=vari, c_filter=None, weight=self.weight))
                    / _std(column=vari, c_filter=None, weight=self.weight)
                ).alias(vari)
            )

        self.df = (
            nw.from_native(self.df)
            .with_columns(with_standardize)
            .lazy()
            .collect()
            .to_native()
        )

        #   Drop variables with all missing values
        all_missing = (
            nw.from_native(self.df)
            .select([nw.col(coli).is_missing().all() for coli in self.x])
            .to_polars()
            .to_dicts()[0]
        )

        self.x = [coli for coli, col_missing in all_missing.items() if not col_missing]

    def _process_formula_string(self):
        #       Parse the formula and see if we need to get a
        #   Simple proxy for needing to go to R and get the model matrix
        #       Does it have a "(" indicating some kind of transformation
        fb = FormulaBuilder(df=self.df, formula=self.formula)

        #   Do we need to get the model matrix?
        b_need_mm = fb.formula.find("(") > 0 or fb.formula.find(":") > 0

        if b_need_mm:
            #       Get analysis dataset from formulaic (the model matrix)
            fb.remove_constant()
            nw_type = NarwhalsType(self.df)

            df_mm = formulaic.Formula(fb.formula).get_model_matrix(
                nw.from_native(self.df).lazy().collect()
            )

            if hasattr(df_mm, "rhs"):
                df_mm = df_mm.rhs
            # df_mm = (
            #     formulaic.Formula(fb.formula)
            #     .get_model_matrix(nw.from_native(self.df).lazy().collect())
            #     .rhs
            # )

            self.x = nw.from_native(df_mm).lazy().collect_schema().names()

            if self.y == "":
                self.y = fb.lhs()

            y_weight = []
            if self.y != "":
                y_weight.append(self.y)

            if self.weight != "":
                if self.weight not in df_mm.columns:
                    y_weight.append(self.weight)

            #   Replace the dataframe with the model matrix
            self.df = concat_wrapper(
                [
                    (
                        nw.from_native(self.df)
                        .select(y_weight)
                        .lazy()
                        .collect()
                        .to_native()
                    ),
                    (nw.from_native(df_mm).lazy().collect().to_native()),
                ],
                how="horizontal",
            )
        else:
            #   No need to go to R, just set x from the formula
            self.x = fb.columns_rhs

    def _process_formula_list(self):
        self.x = columns_from_list(df=self.df, columns=self.formula)

    def find_optimal_lambda(self):
        self.df = nw.from_native(self.df).lazy().collect().to_native()
        set_seed(self.seed)
        random_state = generate_seed()

        #   Update seed
        self.seed = generate_seed()

        lasso_cv = LassoCV(
            cv=self.nfolds,
            random_state=random_state,
            max_iter=self.max_iter,
            alphas=np.logspace(-4, 0, 100),
        )

        lasso_cv.fit(
            nw.from_native(self.df).select(self.x).to_numpy(),
            nw.from_native(self.df).select(self.y).to_numpy().ravel(),
        )

        self.optimal_lambda = lasso_cv.alpha_

    def run(self) -> list[str]:
        self.df = nw.from_native(self.df).lazy().collect().to_native()
        if self.optimal_lambda is None:
            self.find_optimal_lambda()

        set_seed(self.seed)
        random_state = generate_seed()
        #   Update seed
        self.seed = generate_seed()

        lasso = sk_lasso(
            alpha=self.optimal_lambda, random_state=random_state, max_iter=self.max_iter
        )

        lasso.fit(
            nw.from_native(self.df).select(self.x).to_numpy(),
            nw.from_native(self.df).select(self.y).to_numpy().ravel(),
        )

        coef = lasso.coef_
        print(coef)
        vars_kept = []

        for i in range(0, len(self.x)):
            if coef[i] != 0:
                vars_kept.append(self.x[i])

        return vars_kept
