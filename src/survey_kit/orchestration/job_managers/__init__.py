from ..utilities import CallTypes
from .in_process import Executor as _in_process_executor
from .shell import Executor as _shell_executor
from .pbs import Executor as _pbs_executor
from .multiprocessing import Executor as _multiprocessing_executor

from ... import logger

EXECUTORS = {
    CallTypes.in_process: _in_process_executor,
    CallTypes.PBS: _pbs_executor,
    CallTypes.shell: _shell_executor,
    CallTypes.multiprocessing: _multiprocessing_executor,
}


def get_executor(call_type: CallTypes):
    if call_type not in EXECUTORS:
        message = f"No executor for {call_type}"
        logger.error(message)
        raise Exception(message)

    return EXECUTORS[call_type]
