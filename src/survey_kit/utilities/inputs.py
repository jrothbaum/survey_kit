from __future__ import annotations

import os
import polars as pl

from .. import logger


def list_input(l: list | object | None = None) -> list:
    if l is None:
        l = []
    elif type(l) is not list:
        l = [l]

    return l


def is_polars_frame(df: object) -> bool:
    return isinstance(df, (pl.DataFrame, pl.LazyFrame))


def create_folders_if_needed(paths: list[str] | str, quietly: bool = True):
    paths = list_input(paths)

    #   Loop over the list and check if the folders exist
    for pathi in paths:
        pathExists = os.path.isdir(pathi)

        if pathExists:
            if not quietly:
                logger.info(pathi + " already exists")
        else:
            if not quietly:
                logger.info(pathi + " does not exist")
                logger.info("     CREATING NOW")
            os.makedirs(pathi)
