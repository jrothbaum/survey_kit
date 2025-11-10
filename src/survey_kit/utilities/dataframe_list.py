from __future__ import annotations


import os
import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT

from ..utilities.dataframe import (
    join_wrapper,
    concat_wrapper,
    join_list,
    columns_from_list,
)

from ..utilities.inputs import list_input

from ..statistics.statistics import Statistics
from ..statistics.calculator import StatCalculator

from ..serializable import Serializable

from .. import logger


class DataFrameList(Serializable):
    """
    A light wrapper around Lazy/DataFrames
    that can be used like a list, mostly, (append, extend, +)
    and also can be used like the underlying LazyFrame or DataFrame
    where a narwhals operation is applied to all items in the list.

    Examples
    --------
    >>> df_list = DataFrameList([df1, df2])
    >>> df_list = df_list.filter(nw.col("a") == 1)

    This would apply the filter to df1 and df2 and return a DataFrameList with
    the filtered data.

    """

    _save_suffix = "df_list"

    def __init__(self, df_list: list[IntoFrameT]):
        self._df_list = df_list

    def __getitem__(self, index):
        return self._df_list[index]

    def __setitem__(self, index, value):
        self._df_list[index] = value

    def __len__(self):
        return len(self._df_list)

    def append(self, df: IntoFrameT):
        self._df_list.append(df)

    def extend(self, iterable: DataFrameList | list[IntoFrameT]):
        if type(iterable) is DataFrameList:
            self._df_list.extend(iterable._df_list)
        else:
            self._df_list.extend(iterable)

    def __add__(self, other: DataFrameList | list[IntoFrameT]):
        if type(other) is DataFrameList:
            self._df_list = self._df_list + other._df_list
            return self
        else:
            self._df_list = self._df_list + other
            return self

    def __repr__(self):
        #   Mimic how lists of polars tables are displayed
        return ",\n".join([dfi.__repr__() for dfi in self._df_list])

    def __iter__(self):
        return iter(self._df_list)

    def __getattr__(self, attr):
        """
        Delegate attribute to polars LazyFrame | DataFrame
            Determines lazy/data based on df_list object 0

        """

        if hasattr(nw.from_native(self._df_list[0]), attr):
            this_attr = getattr(nw.from_native(self._df_list[0]), attr)
            if callable(this_attr):

                def wrapper(*args, **kwargs):
                    output = [
                        getattr(nw.from_native(dfi), attr)(*args, **kwargs)
                        for dfi in self._df_list
                    ]
                    if isinstance(output[0], (nw.LazyFrame, nw.DataFrame)):
                        return DataFrameList([dfi.to_native() for dfi in output])
                    else:
                        return DataFrameList([itemi for itemi in output])

                return wrapper
            else:
                output = [getattr(nw.from_native(dfi), attr) for dfi in self._df_list]
                if isinstance(output[0], (nw.LazyFrame, nw.DataFrame)):
                    return DataFrameList([dfi.to_native() for dfi in output])
                else:
                    return DataFrameList([itemi for itemi in output])

        else:
            raise AttributeError(
                f"{type(nw.from_native(self._df_list[0]))} has no attribute '{attr}'"
            )

    def join_to_list(
        self,
        df_join: list[IntoFrameT],
        on: list[str],
        how: str,
        suffixes: list[str] | None = None,
        prefixes: list[str] | None = None,
    ) -> DataFrameList:
        prefixes = list_input(prefixes)
        suffixes = list_input(suffixes)

        if len(prefixes) and len(prefixes) == len(df_join):
            prefixes = [""] + prefixes
        if len(suffixes) and len(suffixes) == len(df_join):
            suffixes = [""] + suffixes

        for i in range(0, len(self._df_list)):
            self._df_list[i] = join_list(
                [self._df_list[i]] + df_join,
                on=on,
                how=how,
                prefixes=prefixes,
                suffixes=suffixes,
            )
        return self

    def append_list(self, df_append: list[IntoFrameT]) -> DataFrameList:
        for i in range(0, len(self._df_list)):
            self._df_list[i] = concat_wrapper(
                df_list=[self._df_list[i]] + df_append, how="diagonal"
            )

        return self

    def calculate_stats(
        self,
        statistics: list[Statistics] | Statistics | None = None,
        weight: str = "",
        scale_wgts_to: float = 0.0,
        summarize_by: dict[str, list[str]] | None = None,
        display: bool = True,
        display_all_vars: bool = True,
        display_max_vars: int = 20,
        round_output: bool | int = True,
    ) -> DataFrameList:
        def stats_for_one(df: IntoFrameT):
            sc = StatCalculator(
                df=df,
                statistics=statistics,
                weight=weight,
                scale_wgts_to=scale_wgts_to,
                summarize_by=summarize_by,
                display=display,
                display_all_vars=display_all_vars,
                display_max_vars=display_max_vars,
                round_output=round_output,
            )

            return sc.df_estimates

        return self.pipe(stats_for_one)

    def calculate_stats_average(
        self,
        statistics: list[Statistics] | Statistics | None = None,
        weight: str = "",
        scale_wgts_to: float = 0.0,
        summarize_by: dict[str, list[str]] | None = None,
        display: bool = True,
        display_all_vars: bool = True,
        display_max_vars: int = 20,
        round_output: bool | int = True,
    ) -> StatCalculator:
        df_list = self.calculate_stats(
            statistics=statistics,
            weight=weight,
            scale_wgts_to=scale_wgts_to,
            summarize_by=summarize_by,
            display=False,
            display_all_vars=display_all_vars,
            display_max_vars=display_max_vars,
            round_output=round_output,
        )

        sc = StatCalculator(
            df=self[0],
            statistics=statistics,
            weight=weight,
            scale_wgts_to=scale_wgts_to,
            summarize_by=summarize_by,
            display=False,
            display_all_vars=display_all_vars,
            display_max_vars=display_max_vars,
            round_output=round_output,
            calculate=False,
        )
        sc.df_estimates = df_list.average()

        if display:
            sc.print()

        return sc

    def average(self, order_by: list[str] | str | None = None) -> IntoFrameT:
        index = "__df_average_row_index__"
        order_by = list_input(order_by)
        if len(order_by) == 0:
            df_list = [
                nw.from_native(dfi).lazy().collect().with_row_index(name=index)
                for dfi in self._df_list
            ]
        else:
            df_list = [
                nw.from_native(dfi).lazy().with_row_index(name=index, order_by=order_by)
                for dfi in self._df_list
            ]
        df = concat_wrapper(df_list, how="diagonal")

        all_cols = df.collect_schema().names()

        order_by_full = order_by + [index]
        cols_numeric = (
            df.select(cs.numeric() | cs.boolean())
            .drop(order_by + [index])
            .collect_schema()
            .names()
        )
        cols_first = columns_from_list(
            df, columns="*", exclude=cols_numeric + order_by_full
        )

        with_agg = []
        if len(cols_first):
            with_agg.append(nw.col(cols_first).first())
        with_agg.append(nw.col(cols_numeric).mean())
        df_out = nw.from_native(df).lazy().group_by(order_by_full).agg(with_agg)

        return df_out.select(all_cols).sort(order_by_full).drop(index).to_native()
