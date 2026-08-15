import os
import logging
from pathlib import Path
from .utilities.logging import set_logging
from survey_kit.orchestration.config import Config

logger = set_logging(name=__name__, level=logging.INFO)
config = Config()
config.code_root = os.path.dirname(__file__)
config._set_thread_limits()

if config.data_root == "":
    config.data_root = (
        Path(config.code_root).as_posix().replace("/src/survey_kit", "") + "/.scratch"
    )
