from __future__ import annotations

import re
import logging
import narwhals as nw
import narwhals.selectors as cs
from narwhals.typing import IntoFrameT
from narwhals import (
    Implementation,
)
from narwhals._utils import is_eager_allowed, is_lazy_allowed
import polars as pl
from .inputs import list_input

from ..serializable import Serializable

from .. import logger


@nw.narwhalify
def fill_missing(
    df: IntoFrameT, columns: list[str] | str | None = None, value=0
) -> IntoFrameT:
    nw_type = NarwhalsType(df)
    if columns is None:
        c_missing = nw.all()
        c_numeric = cs.numeric()
    else:
        c_missing = nw.col(columns_from_list(columns))
        c_numeric = None

    if value is not None:
        try:
            df = df.with_columns(c_missing.fill_null(value))
        except:
            df = df.with_columns(c_numeric.fill_null(value))
    if c_numeric is not None:
        df = df.with_columns(c_numeric.fill_nan(value))

    return df.lazy_backend(nw_type)


@nw.narwhalify
def safe_sum_cast(df: IntoFrameT, columns: list[str] | str | None = None) -> IntoFrameT:
    columns = list_input(columns)

    schema = df.lazy().collect_schema()

    castlist = []
    for coli in columns:
        if schema[coli] == nw.Int32:
            castlist.append(nw.col(coli).cast(nw.Int64).alias(coli))

    if len(castlist) > 0:
        df = df.with_columns(castlist)

    return df


def columns_if_present(columns: list[str], colums_to_check: list[str]) -> list[str]:
    columns_present = set(columns).intersection(colums_to_check)

    return _columns_original_order(
        columns_unordered=columns_present, columns_ordered=colums_to_check
    )


@nw.narwhalify
def safe_columns(df: IntoFrameT) -> list[str]:
    return df.lazy().collect_schema().names()


@nw.narwhalify
def columns_from_list(
    df: IntoFrameT,
    columns: list[str] | str,
    exclude: list[str] | str | None = None,
    case_insensitive: bool = False,
) -> list[str]:
    columns = list_input(columns)
    exclude = list_input(exclude)
    all_cols = []
    for coli in columns:
        if coli == "*":
            all_cols.extend(df.lazy().collect_schema().names())
        elif "*" in coli:
            all_cols.extend(
                _by_name_asterisk(
                    df=df, pattern=coli, case_insensitive=case_insensitive
                )
            )
        else:
            all_cols.append(coli)

    if len(exclude):
        cols_exclude = columns_from_list(
            df=df, columns=exclude, case_insensitive=case_insensitive
        )

        if len(cols_exclude):
            all_cols = [coli for coli in all_cols if coli not in cols_exclude]
    if df is not None:
        return _columns_original_order(
            list(set(all_cols)), columns_ordered=df.lazy().collect_schema().names()
        )
    else:
        return list(dict.fromkeys(all_cols))


def _columns_original_order(
    columns_unordered: list[str], columns_ordered: list[str]
) -> list[str]:
    if len(set(columns_unordered).intersection(columns_ordered)) == len(
        columns_unordered
    ):
        d_cols = {}
        for coli in columns_unordered:
            index = columns_ordered.index(coli)
            d_cols[index] = coli

        return list(dict(sorted(d_cols.items())).values())
    else:
        return columns_unordered


@nw.narwhalify
def _by_name_asterisk(
    df: IntoFrameT, pattern: str = "", case_insensitive: bool = False
) -> list[str]:
    if case_insensitive:
        casei_regex = "(?i)"
    else:
        casei_regex = ""

    pattern_regex = casei_regex + "^" + pattern.replace("*", ".*") + "$"

    regex = re.compile(pattern_regex)
    return [col for col in df.lazy().collect_schema().names() if regex.match(col)]


def join_list(
    df_list: list[IntoFrameT | nw.LazyFrame | nw.DataFrame],
    on: list[list[str]] | list[str] | str,
    how: str,
    suffixes: list[str] | None = None,
    prefixes: list[str] | None = None,
) -> IntoFrameT | nw.LazyFrame | nw.DataFrame:
    nw_list = [NarwhalsType(dfi) for dfi in df_list]
    on = list_input(on)

    on_left_right = all((type(oni) is list) for oni in on) and len(on) == len(df_list)
    prefixes = list_input(prefixes)
    suffixes = list_input(suffixes)

    if len(prefixes) or len(suffixes):
        columns_to_rename_list = []
        for dfi in df_list:
            columnsi = nw.from_native(dfi).lazy().collect_schema().names()
            columns_to_rename_list.append(list(set(columnsi).difference(on)))

        for i in range(len(df_list)):
            if len(prefixes):
                prefixi = prefixes[i]
            else:
                prefixi = ""

            if len(suffixes):
                suffixi = suffixes[i]
            else:
                suffixi = ""
            renamei = {
                coli: f"{prefixi}{coli}{suffixi}" for coli in columns_to_rename_list[i]
            }

            if len(renamei):
                df_list[i] = nw.from_native(df_list[i]).rename(renamei)

    df_out = df_list[0]
    df_others = df_list[1:]

    for i, dfi in enumerate(df_others):
        if on_left_right:
            oni = None
            left_on = on[0]
            right_on = on[i]
        else:
            oni = on
            left_on = None
            right_on = None
        df_out = join_wrapper(
            df=df_out, df_to=dfi, on=oni, how=how, left_on=left_on, right_on=right_on
        )

    return NarwhalsType.return_df(df=df_out, nw_types=nw_list)


def join_wrapper(
    df: IntoFrameT | nw.LazyFrame | nw.DataFrame,
    df_to: IntoFrameT | nw.LazyFrame | nw.DataFrame,
    on: list[str] | str,
    how: str,
    left_on: list[str] | None = None,
    right_on: list[str] | None = None,
) -> IntoFrameT | nw.LazyFrame | nw.DataFrame:
    (df, df_to) = safe_upcast_list([df, df_to])
    nw_type_df = NarwhalsType(df)
    nw_type_df_to = NarwhalsType(df_to)

    if left_on is not None and right_on is not None:
        on = None
    else:
        left_on = None
        right_on = None

    if nw_type_df.backend != nw_type_df_to.backend:
        df_to = convert_to_backend(df_to, nw_type_df)
        nw_type_df_to = NarwhalsType(df_to)

    df_out = (
        nw_type_df.safe_to_narwhals()
        .lazy()
        .join(
            nw_type_df_to.safe_to_narwhals().lazy(),
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
        )
    )

    cols_final = df_out.lazy().collect_schema().names()
    c_coalesce = []
    c_drop = []

    if on is None:
        for i in range(len(left_on)):
            left_oni = left_on[i]
            right_oni = right_on[i]

            if left_oni == right_oni:
                coli = left_oni
                coli_right = f"{coli}_right"

                if coli_right in cols_final:
                    c_coalesce.append(
                        nw.coalesce(nw.col(coli), nw.col(coli_right)).alias(coli)
                    )
                    c_drop.append(coli_right)
    else:
        for coli in on:
            coli_right = f"{coli}_right"

            if coli_right in cols_final:
                c_coalesce.append(
                    nw.coalesce(nw.col(coli), nw.col(coli_right)).alias(coli)
                )
                c_drop.append(coli_right)
    if len(c_coalesce):
        df_out = df_out.with_columns(c_coalesce).drop(c_drop)

    return NarwhalsType.return_df(df_out, nw_types=[nw_type_df, nw_type_df_to])


def convert_to_backend(df: IntoFrameT | None, nw_type: NarwhalsType) -> IntoFrameT:
    try:
        if NarwhalsType(df).backend != nw_type.backend:
            return nw.from_native(
                nw.from_native(df).lazy().collect().to_arrow()
            ).lazy_backend(nw_type)
    except:
        pass
    return df


def concat_wrapper(
    df_list: list[IntoFrameT | nw.LazyFrame | nw.DataFrame],
    how: str,
    upcast: bool = True,
) -> IntoFrameT | nw.LazyFrame | nw.DataFrame:
    df_list = [dfi for dfi in df_list if dfi is not None]
    if upcast:
        df_list = safe_upcast_list(df_list)

    nw_type_list = [NarwhalsType(dfi) for dfi in df_list]

    for i in range(len(df_list)):
        if i == 0:
            nw_type_0 = nw_type_list[0]
        if i > 0:
            nw_typei = nw_type_list[i]
            if nw_typei.backend != nw_type_0.backend:
                df_list[i] = convert_to_backend(df_list[i], nw_type_0)
                nw_type_list[i] = NarwhalsType(df_list[i])

    if how == "horizontal":
        return NarwhalsType.return_df(
            nw.concat(
                [
                    nw.maybe_reset_index(nw_typei.safe_to_narwhals()).collect()
                    for nw_typei in nw_type_list
                ],
                how=how,
            ),
            nw_types=nw_type_list,
        )
    else:
        return NarwhalsType.return_df(
            nw.concat(
                [nw_typei.safe_to_narwhals() for nw_typei in nw_type_list], how=how
            ),
            nw_types=nw_type_list,
        )


class NarwhalsType(Serializable):
    _save_suffix = "nw_type"
    _save_exclude_items = ["df"]

    def __init__(
        self,
        df: IntoFrameT | nw.LazyFrame | nw.DataFrame | None = None,
        is_narwhals: bool | None = None,
        is_lazy: bool | None = None,
        backend: str | None = None,
    ):
        if df is not None:
            self.is_narwhals = isinstance(df, (nw.LazyFrame, nw.DataFrame))
            self.is_lazy = type(df) is nw.LazyFrame
            if self.is_narwhals:
                df = df.to_native()
            self.df = df
            self.backend = nw.get_native_namespace(df).__name__
        else:
            self.is_narwhals = is_narwhals
            self.is_lazy = is_lazy
            self.backend = backend
            self.df = None

    def safe_to_narwhals(self) -> nw.LazyFrame:
        if isinstance(self.df, (nw.LazyFrame, nw.DataFrame)):
            return self.df.lazy()
        else:
            return nw.from_native(self.df).lazy()

    def is_polars(self) -> bool:
        return nw.from_native(self.df).implementation.is_polars()

    def to_polars(self) -> pl.LazyFrame | pl.DataFrame:
        if not self.is_polars():
            return nw.from_native(self.df).lazy().collect().to_polars()
        else:
            return self.df

    def from_polars(self, df: pl.LazyFrame | pl.DataFrame) -> IntoFrameT:
        if not self.is_polars():
            return nw.from_native(df.lazy().collect().to_arrow()).lazy_backend(
                NarwhalsType(backend=self.backend)
            )
        else:
            return df

    @staticmethod
    def _return_as_narwhals(nw_types: list[NarwhalsType]) -> bool:
        return all([nw_typei.is_narwhals for nw_typei in nw_types])

    @staticmethod
    def _return_as_lazy(nw_types: list[NarwhalsType]) -> bool:
        return all([nw_typei.is_lazy for nw_typei in nw_types])

    @staticmethod
    def return_df(
        df: nw.LazyFrame | nw.DataFrame, nw_types: list[NarwhalsType]
    ) -> nw.LazyFrame | nw.DataFrame | IntoFrameT:
        nw_types = list_input(nw_types)
        if NarwhalsType._return_as_narwhals(nw_types):
            if NarwhalsType._return_as_lazy(nw_types):
                if type(df) is nw.LazyFrame:
                    return df
                else:
                    return nw.from_native(df).lazy_backend(
                        NarwhalsType(backend=nw_types[0].backend)
                    )
            else:
                return df.lazy().collect()
        else:
            return nw.from_native(df).to_native()

    @staticmethod
    def lazy(
        df: IntoFrameT | nw.LazyFrame | nw.DataFrame, nw_type: NarwhalsType | None
    ) -> nw.LazyFrame:
        if not isinstance(df, (nw.LazyFrame, nw.DataFrame)):
            df = nw.from_native(df)

        if isinstance(df, nw.LazyFrame):
            try:
                return df.lazy(nw_type.backend)
            except:
                return df
        elif isinstance(df, nw.DataFrame):
            if nw_type is not None:
                if not is_lazy_allowed(Implementation(nw_type.backend)):
                    return (
                        nw.from_native(df.to_arrow())
                        .lazy()
                        .collect(backend=nw_type.backend)
                        .lazy()
                    )
                else:
                    try:
                        return df.lazy(backend=nw_type.backend)
                    except:
                        return df.lazy()
            else:
                return df.lazy()
        else:
            return df


def safe_upcast_list(
    dfs: list[IntoFrameT | nw.LazyFrame | nw.DataFrame],
) -> list[IntoFrameT | nw.LazyFrame | nw.DataFrame]:
    d_castordering = _cast_ordering(False)

    schemas = [nw.from_native(dfi).lazy().collect_schema() for dfi in dfs]

    schema_superset = {}
    for i, schemai in enumerate(schemas):
        for coli, typei in schemai.items():
            if coli not in schema_superset:
                schema_superset[coli] = type(typei)
            else:
                type1 = schema_superset[coli]
                type2 = typei

                type_to_check = _CastOrderingItem(type1, type(type2))
                if type_to_check in d_castordering:
                    cast_to = d_castordering[type_to_check]

                    if cast_to.type1 is not None:
                        schema_superset[coli] = cast_to.type1

    for i in range(len(dfs)):
        schemai = schemas[i]
        schemai_to = {
            coli: schema_superset[coli]
            for coli, typei in schemai.items()
            if type(typei) != schema_superset[coli]
        }

        if len(schemai_to):
            dfs[i] = NarwhalsType.return_df(
                nw.from_native(dfs[i]).with_columns(
                    [nw.col(coli).cast(typei) for coli, typei in schemai_to.items()]
                ),
                nw_types=[NarwhalsType(dfs[i])],
            )

    return dfs


class _CastOrderingItem:
    def __init__(self, type1: nw.dtypes.DType | None, type2: nw.dtypes.DType | None):
        self.type1 = type1
        self.type2 = type2

    def __hash__(self):
        return hash(f"type1={self.type1},type2={self.type2}")

    def __eq__(self, other):
        if not isinstance(other, _CastOrderingItem):
            return False
        return self.type1 == other.type1 and self.type2 == other.type2


def _cast_ordering(
    binary_to_string: bool = False,
) -> dict[_CastOrderingItem, _CastOrderingItem]:
    """
    Generate DataFrame with Polars type casting rules for safe upcasting.

    Creates a lookup table that defines how to safely cast between Polars data types
    when joining DataFrames. Ensures no data loss occurs during type conversions.

    Returns:
        dict of _CastOrderingItem mapping pairs of types->casts:
    Note:
        Casting hierarchy follows these principles:
        - Smaller integer types cast to larger ones
        - Signed/unsigned integer conflicts resolved by finding common safe type
        - Boolean casts to any numeric type
        - Float32 can upcast to Float64
        - No downcasting rules to prevent data loss
    """
    thisItem = nw.Boolean
    ordering = [
        (thisItem, nw.Boolean, None, None),
        (thisItem, nw.Int8, nw.Int8, None),
        (thisItem, nw.Int16, nw.Int16, None),
        (thisItem, nw.Int32, nw.Int32, None),
        (thisItem, nw.Int64, nw.Int64, None),
        (thisItem, nw.UInt8, nw.UInt8, None),
        (thisItem, nw.UInt16, nw.UInt16, None),
        (thisItem, nw.UInt32, nw.UInt32, None),
        (thisItem, nw.UInt64, nw.UInt64, None),
        (thisItem, nw.Float32, nw.Float32, None),
        (thisItem, nw.Float64, nw.Float64, None),
    ]

    thisItem = nw.Int8
    ordering.extend(
        [
            (thisItem, nw.Int8, None, None),
            (thisItem, nw.Int16, nw.Int16, None),
            (thisItem, nw.Int32, nw.Int32, None),
            (thisItem, nw.Int64, nw.Int64, None),
            (thisItem, nw.UInt8, nw.Int16, nw.Int16),
            (thisItem, nw.UInt16, nw.Int32, nw.Int32),
            (thisItem, nw.UInt32, nw.Int64, nw.Int64),
            (thisItem, nw.UInt64, nw.Float64, nw.Float64),
            (thisItem, nw.Float32, nw.Float32, None),
            (thisItem, nw.Float64, nw.Float64, None),
        ]
    )

    thisItem = nw.Int16
    ordering.extend(
        [
            (thisItem, nw.Int16, None, None),
            (thisItem, nw.Int32, nw.Int32, None),
            (thisItem, nw.Int64, nw.Int64, None),
            (thisItem, nw.UInt8, None, nw.Int16),
            (thisItem, nw.UInt16, nw.Int32, nw.Int32),
            (thisItem, nw.UInt32, nw.Int64, nw.Int64),
            (thisItem, nw.UInt64, nw.Float64, nw.Float64),
            (thisItem, nw.Float32, nw.Float32, None),
            (thisItem, nw.Float64, nw.Float64, None),
        ]
    )

    thisItem = nw.Int32
    ordering.extend(
        [
            (thisItem, nw.Int32, None, None),
            (thisItem, nw.Int64, nw.Int64, None),
            (thisItem, nw.UInt8, None, nw.Int32),
            (thisItem, nw.UInt16, None, nw.Int32),
            (thisItem, nw.UInt32, nw.Int64, nw.Int64),
            (thisItem, nw.UInt64, nw.Float64, nw.Float64),
            (thisItem, nw.Float32, nw.Float64, nw.Float64),
            (thisItem, nw.Float64, nw.Float64, None),
        ]
    )

    thisItem = nw.Int64
    ordering.extend(
        [
            (thisItem, nw.Int64, None, None),
            (thisItem, nw.UInt8, None, nw.Int64),
            (thisItem, nw.UInt16, None, nw.Int64),
            (thisItem, nw.UInt32, None, nw.Int64),
            (thisItem, nw.UInt64, nw.Float64, nw.Float64),
            (thisItem, nw.Float32, nw.Float64, nw.Float64),
            (thisItem, nw.Float64, nw.Float64, None),
        ]
    )

    thisItem = nw.UInt8
    ordering.extend(
        [
            (thisItem, nw.UInt8, None, None),
            (thisItem, nw.UInt16, nw.UInt16, None),
            (thisItem, nw.UInt32, nw.UInt32, None),
            (thisItem, nw.UInt64, nw.UInt64, None),
            (thisItem, nw.Float32, nw.Float32, None),
            (thisItem, nw.Float64, nw.Float64, None),
        ]
    )

    thisItem = nw.UInt16
    ordering.extend(
        [
            (thisItem, nw.UInt16, None, None),
            (thisItem, nw.UInt32, nw.UInt32, None),
            (thisItem, nw.UInt64, nw.UInt64, None),
            (thisItem, nw.Float32, nw.Float32, None),
            (thisItem, nw.Float64, nw.Float64, None),
        ]
    )

    thisItem = nw.UInt32
    ordering.extend(
        [
            (thisItem, nw.UInt32, None, None),
            (thisItem, nw.UInt64, nw.UInt64, None),
            (thisItem, nw.Float32, nw.Float32, None),
            (thisItem, nw.Float64, nw.Float64, None),
        ]
    )

    thisItem = nw.UInt64
    ordering.extend(
        [
            (thisItem, nw.UInt64, None, None),
            (thisItem, nw.Float32, nw.Float64, nw.Float64),
            (thisItem, nw.Float64, nw.Float64, None),
        ]
    )

    thisItem = nw.Float32
    ordering.extend(
        [
            (thisItem, nw.Float32, None, None),
            (thisItem, nw.Float64, nw.Float64, None),
        ]
    )

    if binary_to_string:
        thisItem = nw.Binary
        ordering.extend(
            [
                (thisItem, nw.Boolean, nw.String, nw.String),
                (thisItem, nw.Int8, nw.String, nw.String),
                (thisItem, nw.Int16, nw.String, nw.String),
                (thisItem, nw.Int32, nw.String, nw.String),
                (thisItem, nw.Int64, nw.String, nw.String),
                (thisItem, nw.UInt8, nw.String, nw.String),
                (thisItem, nw.UInt16, nw.String, nw.String),
                (thisItem, nw.UInt32, nw.String, nw.String),
                (thisItem, nw.UInt64, nw.String, nw.String),
                (thisItem, nw.Float32, nw.String, nw.String),
                (thisItem, nw.Float64, nw.String, nw.String),
                (thisItem, nw.String, nw.String, None),
                (thisItem, nw.Binary, nw.Binary, nw.String),
            ]
        )

    d_ordering = {}
    for type1, type2, cast1, cast2 in ordering:
        d_ordering[_CastOrderingItem(type1, type2)] = _CastOrderingItem(cast1, cast2)
        d_ordering[_CastOrderingItem(type2, type1)] = _CastOrderingItem(cast2, cast1)
    return d_ordering


def safe_height(df: IntoFrameT) -> int:
    return nw.from_native(df).lazy().select(nw.len()).collect().item(0, 0)


#   Monkey patch lazy
def lazy_backend(
    self: nw.LazyFrame | nw.DataFrame, nw_type: NarwhalsType
) -> nw.LazyFrame:
    return NarwhalsType.lazy(self, nw_type)


nw.LazyFrame.lazy_backend = lazy_backend
nw.DataFrame.lazy_backend = lazy_backend


def backend_eager(backend: str):
    if not is_eager_allowed(Implementation(backend)):
        return "pyarrow"
    else:
        return backend


@nw.narwhalify
def rename_with_prefix_suffix(
    df: IntoFrameT,
    prefix: str = "",
    suffix: str = "",
    exclude_list: list[str] | str | None = None,
) -> IntoFrameT:
    """
    Add prefix and/or suffix to column names with optional exclusions.

    Renames columns by adding specified prefix and suffix, useful for
    avoiding name conflicts when joining datasets.

    Args:
        df (IntoFrameT):
            Input data
        prefix (str):
            Text to add before column names
        suffix (str):
            Text to add after column names
        exclude_list (list[str] | str, optional):
            Column names to exclude from renaming

    Returns:
        IntoFrameT: Data with renamed columns

    Examples:
        Add prefix to all columns:
            >>> df_renamed = rename_with_prefix_suffix(df, prefix="old_")

        Add suffix except for ID columns:
            >>> df_renamed = rename_with_prefix_suffix(df,
            ...                                        suffix="_2023",
            ...                                        exclude_list=["h_seq",
            ...                                                      "pppos"])

        Combine prefix and suffix:
            >>> df_renamed = rename_with_prefix_suffix(df,
            ...                                        prefix="temp_",
            ...                                        suffix="_backup")
    """

    exclude_list = list_input(exclude_list)
    renames = {}
    columns = df.lazy().collect_schema().names()

    if len(exclude_list):
        exclude_list_l = list(map(str.lower, exclude_list))

        for coli in columns:
            bFound = coli.lower() in exclude_list_l

            if not bFound:
                renames[coli] = f"{prefix}{coli}{suffix}"
    else:
        renames = {coli: f"{prefix}{coli}{suffix}" for coli in columns}

    if len(renames):
        df = df.rename(renames)

    return df


@nw.narwhalify
def drop_if_exists(
    df: IntoFrameT,
    columns: list[str] | str | None = None,
) -> IntoFrameT:
    columns = list_input(columns)
    columns_exist = df.lazy().collect_schema().names()
    drop_list = set(columns).intersection(columns_exist)

    if len(drop_list):
        return df.drop(drop_list)
    else:
        return df


def asterisk_matched_substring(
    pattern: str = "", input_list: list = None
) -> dict[str, str]:
    """
    Extract substrings from a list of strings based on a pattern with asterisk wildcards.

    Uses regex to match a pattern containing '*' wildcards and extracts the
    portions that match the wildcard locations.

    Args:
        pattern (str): Pattern string with '*' as wildcards
        input_list (list[str], optional): List of strings to match against

    Returns:
        dict: Mapping of matched strings to their extracted wildcard portions

    Example:
        >>> AsteriskMatchedSubstring("data_*.parquet", ["data_2023.parquet", "data_2024.parquet"])
        {'data_2023.parquet': '2023', 'data_2024.parquet': '2024'}
    """
    if input_list is None:
        input_list = []

    escaped_pattern = re.escape(pattern).replace(r"\*", "(.+)")

    regex_pattern = re.compile(escaped_pattern)

    result = {}

    for itemi in input_list:
        matched = regex_pattern.match(itemi)

        if matched:
            result[itemi] = matched.group(1)

    return result


def print_longer_table(
    df: IntoFrameT, max_rows: int = 100, drb_round: bool = True, logging: logging = None
) -> None:
    """
    Print DataFrame with more rows than default Polars displays.

    Utility for displaying larger tables with optional disclosure-safe rounding
    and customizable row limits.

    Args:
        df (pl.LazyFrame | pl.DataFrame):
            Data to display
        max_rows (int):
            Maximum rows to display (0 = no limit)
        drb_round (bool):
            Apply disclosure-safe rounding
        logger (logging,
                optional): Custom logger for output

    Returns:
        None: Prints formatted table

    Example:
        >>> print_longer_table(df,
        ...                    max_rows=200,
        ...                    drb_round=False)

    Note:
        Useful for examining larger datasets while maintaining readable output.
        Boolean columns converted to Int8 for display consistency.
    """

    from .rounding import drb_round_table

    if logging is None:
        logging = logger

    nw_type = NarwhalsType(df)
    df = nw_type.to_polars().with_columns(pl.col(pl.Boolean).cast(pl.Int8))
    if drb_round:
        df = drb_round_table(df)

    if max_rows:
        n_rows = min(max_rows, safe_height(df))
    else:
        n_rows = safe_height(df)

    with pl.Config(fmt_str_lengths=50) as cfg:
        #   Basic formatting
        cfg.set_tbl_cell_alignment("RIGHT")
        cfg.set_tbl_hide_column_data_types(True)
        cfg.set_tbl_hide_dataframe_shape(True)
        cfg.set_thousands_separator(True)
        cfg.set_tbl_width_chars(600)
        cfg.set_tbl_cols(len(df.lazy().collect_schema().names()))

        cfg.set_tbl_rows(n_rows)

        logging.info(df.lazy().collect())


@nw.narwhalify
def summary(
    df: IntoFrameT,
    columns: list[str] | str | None = None,
    weight: str = "",
    print: bool = True,
    stats: list[str] | str | None = None,
    detailed: bool = False,
    additional_stats: list[str] | str | None = None,
    by: list[str] | str | None = None,
    quantile_interpolated: bool = False,
    drb_round: bool = False,
) -> IntoFrameT:
    """
    Generate summary statistics for a dataframe.

    A convenience function for quickly exploring data. Calculates common summary
    statistics (mean, std, min, max, etc.) with optional weighting and grouping.
    Works with any dataframe backend (Polars, Pandas, Arrow, DuckDB) via Narwhals.

    Parameters
    ----------
    df : IntoFrameT
        Input dataframe to summarize.
    columns : list[str] | str | None, optional
        Columns to summarize. Supports wildcards (e.g., "income_*").
        If None, summarizes all columns. Default is None.
    weight : str, optional
        Column name for weights. If provided, calculates weighted statistics.
        Default is "" (unweighted).
    print : bool, optional
        Print the summary table. Default is True.
    stats : list[str] | str | None, optional
        Statistics to calculate. If None, uses default set.
        See Statistics.available_stats() for options. Default is None.
    detailed : bool, optional
        Use detailed statistics (includes quartiles).
        Overrides stats parameter. Default is False.
    additional_stats : list[str] | str | None, optional
        Additional statistics beyond the default/detailed set.
        Examples: ["q10", "q90", "n|not0", "share|not0"]. Default is None.
    by : list[str] | str | None, optional
        Column(s) to group by before calculating statistics. Default is None.
    quantile_interpolated : bool, optional
        Use interpolated quantiles (vs exact values from data). Default is False.
    drb_round : bool, optional
        Apply DRB (Disclosure Review Board) rounding rules for 4 significant digits.
        Useful for publication-ready output. Default is False.

    Returns
    -------
    IntoFrameT
        Dataframe of summary statistics (same type as input df).

    Examples
    --------
    Basic unweighted summary:

    >>> from survey_kit.utilities.dataframe import summary
    >>> from survey_kit.utilities.random import RandomData
    >>>
    >>> df = RandomData(n_rows=1000, seed=123).integer("income", 0, 100_000).to_df()
    >>> summary(df)

    Weighted summary:

    >>> summary(df, weight="survey_weight")

    By groups:

    >>> summary(df, weight="survey_weight", by="year")

    Detailed statistics with rounding:

    >>> summary(df, weight="survey_weight", detailed=True, drb_round=True)

    Custom statistics:

    >>> from survey_kit.statistics.statistics import Statistics
    >>> Statistics.available_stats()  # See options
    >>> summary(df, additional_stats=["q10", "q90", "n|not0", "share|not0"])

    Specific columns with wildcards:

    >>> summary(df, columns=["income_*", "age"], weight="survey_weight")

    Get results without printing:

    >>> df_stats = summary(df, weight="survey_weight", print=False)
    >>> print(df_stats.collect())

    Notes
    -----
    Default statistics (if stats=None and detailed=False):
    - n: Count of non-missing values
    - n|missing: Count of missing values
    - mean: Average
    - std: Standard deviation
    - min: Minimum
    - max: Maximum

    Detailed statistics (if detailed=True):
    - Adds: q25, q50 (median), q75

    The "|not0" suffix excludes zeros: "n|not0" counts non-zero values,
    "share|not0" calculates proportion among non-zero observations.

    See Also
    --------
    StatCalculator : For standard errors with replicate weights
    Statistics : For defining custom statistics
    """

    from ..statistics.calculator import StatCalculator
    from ..statistics.statistics import Statistics

    columns = list_input(columns)
    stats = list_input(stats)
    additional_stats = list_input(additional_stats)
    by = list_input(by)
    if len(by) > 1:
        by = [by]

    if len(columns):
        columns = columns_from_list(df=df, columns=columns)
    else:
        columns = df.lazy().collect_schema().names()

    if len(stats) == 0:
        if detailed:
            stats = [
                "n",
                "n|missing",
                "mean",
                "std",
                "min",
                "q25",
                "q50",
                "q75",
                "max",
            ] + additional_stats

        else:
            stats = ["n", "n|missing", "mean", "std", "min", "max"] + additional_stats

    if weight != "" and weight in columns:
        columns.remove(weight)
    stats = Statistics(
        stats=stats, columns=columns, quantile_interpolated=quantile_interpolated
    )

    sc = StatCalculator(
        df=df,
        statistics=stats,
        display=print,
        round_output=drb_round,
        weight=weight,
        by=by,
    )

    return sc.df_estimates


def winsorize_by_percentiles(
    df: IntoFrameT,
    percentiles: tuple[float, float],
    columns: list | str,
    weight: str = "",
) -> IntoFrameT:
    """
    Winsorize (truncate) column values to a specific percentile

    Parameters
    ----------
    df : pl.IntoFrameT
        Data
    percentiles : tuple[float,float]
        Lower and upper bound
    columns : list|str
        Columns to winsorize
    weight : str, optional
        If weighted, weight column. The default is "".

    Returns
    -------
    df : IntoFrameT

    """

    from ..statistics.basic_calculations import calculate_by

    if type(columns) is str:
        columns = [columns]

    multiply_ptiles = not any([pi >= 1 for pi in percentiles])
    if multiply_ptiles:
        percentiles = [pi * 100 for pi in percentiles]

    column_stats = {}
    for coli in columns:
        column_stats[coli] = [f"q{pi}" for pi in percentiles]

    df_winsor = calculate_by(df=df, column_stats=column_stats, weight=weight)

    quantile_low = f"{percentiles[0] / 100}".replace(".", "_")
    quantile_high = f"{percentiles[1] / 100}".replace(".", "_")
    clip_list = []
    for coli in columns:
        clip_low = (
            nw.from_native(df_winsor)
            .select(nw.col(f"{coli}_q{quantile_low}"))
            .item(0, 0)
        )
        clip_high = (
            nw.from_native(df_winsor)
            .select(nw.col(f"{coli}_q{quantile_high}"))
            .item(0, 0)
        )

        clip_list.append(nw.col(coli).clip(lower_bound=clip_low, upper_bound=clip_high))

    df = nw.from_native(df).with_columns(clip_list).to_native()
    return df
