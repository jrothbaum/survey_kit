#   Default Logging setup and basic logging function to call
import sys
import os
import logging
from copy import deepcopy
from contextlib import contextmanager


def set_logging(
    path_log: str = "",
    force: bool = True,
    to_console: bool = True,
    append_to_file: bool = False,
    level=logging.NOTSET,
    name: str = "",
):
    if name == "":
        globals()["_survey_kit_log_params"] = deepcopy(locals())
    handlers = []
    Format = "%(levelname)10s:\t\t%(message)s"

    if path_log != "":
        if not append_to_file:
            if os.path.isfile(path_log):
                os.remove(path_log)
        mode = "a"

        hFile = SurveyKitLoggingFileHandler(path_log, mode=mode, encoding="utf-8")
        hFile.setFormatter(SurveyKitLoggingFormatting(Format))
        handlers.append(hFile)

    if to_console:
        hConsole = SurveyKitLoggingStreamHandler()
        hConsole.setFormatter(SurveyKitLoggingFormatting(Format))
        handlers.append(hConsole)

    if name != "":
        log_out = logging.getLogger(name)
        for handi in handlers:
            log_out.addHandler(handi)

        log_out.setLevel(level)
        log_out.propagate = False

        return log_out
    else:
        logging.basicConfig(
            format=Format,
            force=force,
            handlers=handlers,
            level=level,
        )


class SurveyKitLoggingFileHandler(logging.FileHandler):
    NoCarriageReturn = "[!n]"

    def emit(self, record) -> None:
        if self.NoCarriageReturn in str(record.msg):
            newrecord = deepcopy(record)
            newrecord.msg = str(newrecord.msg).replace(self.NoCarriageReturn, "")

            return super().emit(record=newrecord)
        else:
            self.terminator = "\n"
            return super().emit(record=record)


class SurveyKitLoggingStreamHandler(logging.StreamHandler):
    NoCarriageReturn = "[!n]"

    def emit(self, record) -> None:
        if self.NoCarriageReturn in str(record.msg):
            newrecord = deepcopy(record)
            newrecord.msg = str(newrecord.msg).replace(self.NoCarriageReturn, "")
            self.terminator = ""
            return super().emit(record=newrecord)
        else:
            self.terminator = "\n"
            return super().emit(record=record)


class SurveyKitLoggingFormatting(logging.Formatter):
    info_fmt = logging.Formatter("%(message)s")
    default_fmt = logging.Formatter("%(levelname)10s:\t\t%(message)s")

    def format(self, record):
        if record.levelno == logging.INFO:
            return self.info_fmt.format(record)
        else:
            return self.default_fmt.format(record)


class PrintLogger(object):
    #   From stack overflow q 216517/616645
    #   Capture print to log file (with logging calls)
    #   Used in asynchronous/separate thread calls from FunctionCall
    def __init__(
        self, filepath: str = "", to_console: bool = True, append_to_file: bool = True
    ):
        if append_to_file:
            mode = "a"
        else:
            mode = "w"

        self.stdout = sys.stdout
        self.file = open(filepath, mode, encoding="utf-8")
        self.to_console = to_console

        #   Set the target of the print command to self
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        if self.to_console:
            self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        if self.to_console:
            self.stdout.flush()

        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None


@contextmanager
def run_with_temporary_logging():
    #   Save current state
    existing_logger = logging.getLogger()
    original_level = existing_logger.level
    original_handlers = existing_logger.handlers.copy()
    original_disabled = existing_logger.disabled

    #   Do whatever happens in the function
    yield

    existing_logger.setLevel(original_level)
    existing_logger.handlers = original_handlers
    existing_logger.disabled = original_disabled
