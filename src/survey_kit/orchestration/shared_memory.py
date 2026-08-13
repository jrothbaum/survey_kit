from __future__ import annotations
from typing import Optional

import narwhals as nw
from narwhals.typing import IntoFrameT
import multiprocessing
import multiprocessing.shared_memory
from multiprocessing.managers import SharedMemoryManager
import pyarrow as pa
import polars as pl


class SharedMemoryParameters:
    def __init__(
        self, shm: multiprocessing.shared_memory.SharedMemory, size: int, name: str
    ):
        self.shm = shm
        self.size = size
        self.name = name

    def to_dict(self):
        return {"shm": self.shm, "size": self.size, "name": self.name}

    def from_dict(d: dict):
        return SharedMemoryParameters(**d)


class SharedMemoryUtility:
    def arrow_shm_list_to_dict_df(
        shm_param_list: list[SharedMemoryParameters] | list[dict] | None,
    ):
        #   Get a dictionary of the passed items as polars dataframes
        if shm_param_list is None:
            return {}

        elif len(shm_param_list) == 0:
            return {}

        else:
            d_out = {}
            for shm_parami in shm_param_list:
                if type(shm_parami) is dict:
                    shm_parami = SharedMemoryParameters.from_dict(shm_parami)

                d_out[shm_parami.name] = SharedMemoryUtility.arrow_shm_to_polars(
                    shm_parami
                )

            return d_out

    def df_to_arrow_shm(
        df: IntoFrameT | str,
        smm: SharedMemoryManager,
        name: str = "",
        backend: str = "",
    ):
        if type(df) is str:
            if backend != "":
                df = nw.scan_parquet(df, backend=backend)
            else:
                df = nw.scan_parquet(df)

            df_a = df.collect()
        else:
            df_a = nw.from_native(df).lazy().collect().to_arrow()

        #   From github.com/wjones127/arrow-ipc-bench
        #       Get size
        mock_sink = pa.MockOutputStream()
        with pa.ipc.new_stream(mock_sink, df_a.schema) as writer:
            writer.write_table(df_a)

        size = mock_sink.size()

        #   Convert table to buffer

        shm = smm.SharedMemory(size=size)
        stream = pa.FixedSizeBufferWriter(pa.py_buffer(shm.buf))

        with pa.RecordBatchStreamWriter(stream, df_a.schema) as writer:
            writer.write_table(df_a)

        return SharedMemoryParameters(shm=shm, size=size, name=name)

    def arrow_shm_to_pandas(shm_params: SharedMemoryParameters | dict):
        if type(shm_params) is dict:
            shm_params = SharedMemoryParameters.from_dict(shm_params)

        return pa.ipc.open_stream(shm_params.shm.buf).read_all().to_pandas()

    def arrow_shm_to_polars(shm_params: SharedMemoryParameters | dict):
        if type(shm_params) is dict:
            shm_params = SharedMemoryParameters.from_dict(shm_params)

        return pl.from_arrow(pa.ipc.open_stream(shm_params.shm.buf).read_all())
