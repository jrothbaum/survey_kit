import sys
import os
from pathlib import Path
import polars as pl
import pandas as pd

from survey_kit.utilities.random import RandomData
from survey_kit.serializable import SerializableList, Serializable
from survey_kit.utilities.dataframe import NarwhalsType


from survey_kit import logger, config

path_scratch = config.path_temp_files


n_rows = 1_000
df = (
    RandomData(n_rows=n_rows, seed=12332151)
    .index("index")
    .integer("v_1", 0, 10)
    .boolean("v_bool")
    .to_df()
    .lazy()
)


l = SerializableList(["a", 1, 2, df])


# Save to the scratch directory
path_save = path_scratch
print(path_save)


nw_type = NarwhalsType(df)
nw_type.save(path_save)

l.save(path_save)
l_replaced = SerializableList.load(path_save)

for i in range(len(l)):
    if isinstance(l[i], pl.LazyFrame):
        l[i] = l[i].collect()
    if isinstance(l_replaced[i], pl.LazyFrame):
        l_replaced[i] = l_replaced[i].collect()

    if isinstance(l[i], pl.DataFrame):
        assert l[i].equals(l_replaced[i])
    else:
        assert l[i] == l_replaced[i]

l_delete = SerializableList.load(path_save, delete=True)
for i in range(len(l)):
    if isinstance(l[i], pl.DataFrame):
        assert l[i].equals(l_replaced[i])
    else:
        assert l[i] == l_replaced[i]


l_pandas = SerializableList(["a", 1, 2, df.collect().to_pandas()])

l_pandas.save(path_save)
l_replaced_pandas = SerializableList.load(path_save, delete=True)

for i in range(len(l_pandas)):
    if isinstance(l_pandas[i], pd.DataFrame):
        assert l_pandas[i].equals(l_replaced_pandas[i])
    else:
        assert l_pandas[i] == l_replaced_pandas[i]


logger.info(Serializable._registry)
