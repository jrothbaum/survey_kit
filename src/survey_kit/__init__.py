import os
import sys
import logging
from pathlib import Path
from .utilities.logging import set_logging
from survey_kit.orchestration.config import Config

#   On Windows, stdout/stderr fall back to the locale codepage (usually
#   cp1252) instead of UTF-8 whenever output isn't an interactive console
#   (redirected to a file, piped, captured by a test runner, ...) - cp1252
#   can't represent the box-drawing characters polars uses for table
#   output, so printing a result table crashes with UnicodeEncodeError in
#   exactly that situation. reconfigure() is process-local (no env vars
#   touched); guarded since some hosts (e.g. pytest's capsys, a Jupyter
#   kernel) replace stdout/stderr with an object that doesn't have it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
del _stream

logger = set_logging(name=__name__, level=logging.INFO)
config = Config()
config.code_root = os.path.dirname(__file__)
config._set_thread_limits()

if config.data_root == "":
    config.data_root = (
        Path(config.code_root).as_posix().replace("/src/survey_kit", "") + "/.scratch"
    )
