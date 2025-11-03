from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import sys
import os

from .config import Config

from .utilities import LINEBREAK, Languages


def remove_useless_errors(
    log: str, language: Languages, remove_stubs: bool = False
) -> str:
    #   Ignore the language because many sas calls or nested within a python call
    lines_to_remove = [
        "ERROR: Expecting page 1, got page -1 instead.",
        "ERROR: Page validation error while reading SASUSER.PROFILE.CATALOG.",
        "ERROR: Errors printed on page 1",
        # "NOTE: Unable to open SASUSER.PROFILE. WORK.PROFILE will be opened instead.",
        # "NOTE: All profile changes will be lost at the end of the session.",
        # "NOTE: Unable to open SASUSER.REGSTRY. WORK.REGSTRY will be opened instead.",
        # "NOTE: All registry changes will be lost at the end of the session.",
        # "WARNING: Unable to copy SASUSER registry to WORK registry. Because of this, you will not see registry customizations during this ",
        # "         session."
    ]

    stubs_to_remove = ["DEBUG:"]

    if language == Languages.Python:
        stubs_to_remove.extend(
            ["Parquet SCAN [", f"{Config().data_root}/TempFiles", " object at 0x"]
        )
    #   Python can call anything
    if language == Languages.R or language == Languages.Python:
        stubs_to_remove.extend(
            [
                "R home found:",
                "R library path:",
                "LD_LIBRARY_PATH:",
                "Default options to initialize R:",
                "R was initialized outside of rpy2",
                "warnings.warn(msg)",
                "R is already initialized.",
                "cffi mode is CFFI_MODE.ANY",
            ]
        )
    if language == Languages.SAS or language == Languages.Python:
        stubs_to_remove.extend(
            [
                "      real time",
                "      cpu time",
                "      Last Modified=",
                "      Filename=",
                "      Pipe command=",
                "records were read from the infile",
                "      The minimum record length was",
                "      The maximum record length was",
                "The SAS System",
                "path_temp = ",
                "DATE:      ",
                "    FILE:",
            ]
        )
    if language == Languages.Stata or language == Languages.Python:
        stubs_to_remove.extend(
            [
                "file /tmp",
                "opened on:  ",
                'using "/tmp/',
            ]
        )
    # if language == Languages.SAS:
    #     lines_to_remove = ["ERROR: Expecting page 1, got page -1 instead.",
    #                        "ERROR: Page validation error while reading SASUSER.PROFILE.CATALOG.",
    #                        "ERROR: Errors printed on page 1",
    #                        # "NOTE: Unable to open SASUSER.PROFILE. WORK.PROFILE will be opened instead.",
    #                        # "NOTE: All profile changes will be lost at the end of the session.",
    #                        # "NOTE: Unable to open SASUSER.REGSTRY. WORK.REGSTRY will be opened instead.",
    #                        # "NOTE: All registry changes will be lost at the end of the session.",
    #                        # "WARNING: Unable to copy SASUSER registry to WORK registry. Because of this, you will not see registry customizations during this ",
    #                        # "         session."
    #                        ]

    # else:
    #     lines_to_remove = []

    if len(lines_to_remove) or len(stubs_to_remove):
        lines = []
        for linei in log.splitlines(False):
            if linei not in lines_to_remove:
                if remove_stubs:
                    if not any([stubi for stubi in stubs_to_remove if stubi in linei]):
                        lines.append(linei)
                else:
                    lines.append(linei)

        return LINEBREAK.join(lines)
    else:
        return log
