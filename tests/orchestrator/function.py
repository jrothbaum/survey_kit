import os
import sys
from pathlib import Path

from survey_kit.orchestration.config import Config
from survey_kit.orchestration.from_decorator import as_function
from survey_kit.orchestration.callers import run_function_list
from survey_kit.orchestration.utilities import CallInputs, CallTypes
from survey_kit import logger, config


@as_function(inputs_as_dictionary=False)
def do_something(a: int, b: int) -> None:
    print(f"printed: {a}*{b} = {a * b}")
    logger.info(f"logged: {a}*{b} = {a * b}")


if __name__ == "__main__":
    f = do_something.as_function(a=1, b=2)

    print(type(f))

    run_function_list(
        f,
        run_all=True,
        testing=False,
        call_input=CallInputs(call_type=CallTypes.shell, n_cpu=4),
        use_function_call_inputs=False,
    )
