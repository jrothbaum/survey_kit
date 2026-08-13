import narwhals as nw
import polars as pl
from survey_kit.utilities.random import RandomData, generate_seed, set_seed
from survey_kit import logger
from survey_kit.utilities.dataframe import summary
from survey_kit.utilities.dataframe_list import DataFrameList
from survey_kit.utilities.compress import compress_df


def random_data(n_rows: int = 100, n_datasets: int = 2, compress: bool = True):
    return DataFrameList(
        [
            (
                RandomData(n_rows=n_rows, seed=generate_seed())
                .index("index")
                .integer("var1", 0, 100_000)
                .float("var2", 0, (i + 1) * 500)
                .to_df(compress=compress)
            )
            for i in range(n_datasets)
        ]
    )


def main_test():
    n_rows = 100
    n_datasets = 2
    seed = 654951
    set_seed(seed)
    #   Fake data
    df_list = random_data(n_rows=n_rows, n_datasets=n_datasets, compress=False)

    df_append = (
        RandomData(n_rows=n_rows, seed=generate_seed())
        .index("index")
        .integer("var1", 0, 100_000)
        .float("var2", 0, 100_000)
        .to_df()
    )

    df_join = (
        RandomData(n_rows=n_rows, seed=generate_seed())
        .index("index")
        .integer("var3", 0, 100_000)
        .float("var4", 0, 100_000)
        .to_df()
    )

    logger.info(df_list.schema)
    df_list = df_list.pipe(compress_df)
    logger.info(df_list.schema)

    df_list = df_list.lazy()
    df_list = df_list.filter(nw.col("index") > 50).collect()

    logger.info(len(df_list))

    for dfi in df_list:
        logger.info(dfi.height)

    for dfi in df_list:
        summary(dfi)

    df_list.append_list([df_append.with_columns(pl.lit(True).alias("_new_"))])
    for dfi in df_list:
        summary(dfi)

    df_list.join_to_list([df_join], on=["index"], how="left")
    for dfi in df_list:
        summary(dfi)

    logger.info(df_list.average(order_by=["index"]).lazy().collect())


main_test()
