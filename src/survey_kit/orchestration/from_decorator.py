#   Create a decorator that sets the parameters to turn a function handle into
#       a Function object
from __future__ import annotations
from typing import Optional, Callable, Any, TYPE_CHECKING, TypeVar, Generic
from typing_extensions import ParamSpec

import os
import logging
import inspect
from copy import deepcopy
import re
import polars as pl

from functools import update_wrapper

from ..serializable import SerializableDictionary
from .config import Config
from .from_python import FunctionFromPython
from .utilities import CallInputs, FileLoaderUtilities

# from NEWS.CodeUtilities.Python.FileLoaderUtilities import FileLoaderUtilities
if TYPE_CHECKING:
    from .function import Function


R = TypeVar("R")
P = ParamSpec("P")
F = TypeVar("F", bound=Callable[..., Any])


class f_arg:
    def __init__(self, name: str):
        self.name = name


class FunctionWrapper(Generic[P, R]):
    def __init__(
        self,
        func: Callable[P, R],
        inputs: list[str] | Callable | None = None,
        args_inputs: dict | list | Any | None = None,
        outputs: list[str] | Callable | None = None,
        args_outputs: dict | None = None,
        call_input: CallInputs | None = None,
        inputs_parameters: list[str] | str | None = None,
        outputs_parameters: list[str] | str | None = None,
        inputs_as_dictionary: bool = False,
    ):
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []

        self.func = func

        self.inputs = inputs
        self.args_inputs = args_inputs
        self.outputs = outputs
        self.args_outputs = args_outputs
        self.call_input = call_input
        self.inputs_parameters = inputs_parameters
        self.outputs_parameters = outputs_parameters
        self.inputs_as_dictionary = inputs_as_dictionary

        update_wrapper(self, func)
        self.__wrapped__ = func
        self.__signature__ = inspect.signature(func)

    def __call__(self, *args, **kwargs) -> R:
        return self.func(*args, **kwargs)

    @classmethod
    def _multiple_slashes_to_one(cls, value: str) -> str:
        return re.sub(r"/+", "/", value)

    def _multiple_dots_to_one(cls, value: str) -> str:
        return re.sub(r"\.+", ".", value)

    def as_function(
        self,
        loadutils: FileLoaderUtilities | None = None,
        call_input: CallInputs | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Function:
        try:
            f_original = inspect.unwrap(self.func)
        except:
            f_original = self.func
        source_file = inspect.getfile(f_original)

        load_from_file = source_file
        namespace = ""

        sig = inspect.signature(self.func)
        args = sig.bind(*args, **kwargs)
        args.apply_defaults()
        d_args = args.arguments
        del args

        if call_input is None:
            call_input = self.call_input

        if self.inputs_as_dictionary:
            d_args = SerializableDictionary(d_args)

        f = FunctionFromPython(
            self.func,
            namespace=namespace,
            load_from_file=load_from_file,
            parameters=d_args,
            inputs=self._get_inputs_from_params(
                self.inputs, self.args_inputs, d_args, self.inputs_parameters
            ),
            outputs=self._get_inputs_from_params(
                self.outputs, self.args_outputs, d_args, self.outputs_parameters
            ),
            call_input=call_input,
        )

        return f

    def _get_inputs_from_params(
        self,
        inputs: list[str | f_arg] | Callable | None,
        args_inputs: dict | list | Any | None,
        parameters: dict | SerializableDictionary | None,
        inputs_as_arg: list[str] | None,
    ) -> list[str]:
        if type(parameters) is SerializableDictionary:
            parameters = parameters._d
        final_inputs = []
        if type(inputs) is list:
            final_inputs = inputs.copy()
        elif type(inputs) is str:
            final_inputs = [inputs]
        elif callable(inputs):
            if args_inputs is None:
                args_inputs = {}

            if type(args_inputs) is dict:
                for keyi, valuei in args_inputs.items():
                    if type(valuei) is f_arg:
                        args_inputs[keyi] = parameters[valuei.name]
                    elif type(valuei) is str:
                        args_inputs[keyi] = parameters[valuei]

                final_inputs = self._process_input_values(inputs(**args_inputs))

            elif type(args_inputs) is list:
                args_inputs = [parameters[valuei] for valuei in args_inputs]
                final_inputs = self._process_input_values(inputs(*args_inputs))
            else:
                final_inputs = self._process_input_values(
                    inputs(parameters[args_inputs])
                )

        if inputs_as_arg is not None:
            if type(inputs_as_arg) is str:
                final_inputs.append(parameters.get(inputs_as_arg))
            else:
                for argi in inputs_as_arg:
                    if argi is not None and type(argi) is str:
                        final_inputs.append(parameters.get(argi))
        return [ini for ini in final_inputs if ini is not None and ini != ""]

    def _process_input_values(
        self, values: dict | list | str | pl.DataFrame
    ) -> list[str]:
        if type(values) is dict:
            values = list(values.values())
        elif type(values) is str:
            values = [values]
        elif type(values) is pl.DataFrame:
            if "FullPath" in values.columns:
                values = values["FullPath"].to_list()

        return values


def as_function(
    inputs: list[str] | Callable | None = None,
    args_inputs: dict | list | Any | None = None,
    outputs: list[str] | Callable | None = None,
    args_outputs: dict | None = None,
    call_input: CallInputs | None = None,
    inputs_parameters: list[str] | str | None = None,
    outputs_parameters: list[str] | str | None = None,
    inputs_as_dictionary: bool = False,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        return FunctionWrapper(
            func=func,
            inputs=deepcopy(inputs),
            args_inputs=deepcopy(args_inputs),
            outputs=deepcopy(outputs),
            args_outputs=deepcopy(args_outputs),
            call_input=deepcopy(call_input),
            inputs_parameters=deepcopy(inputs_parameters),
            outputs_parameters=deepcopy(outputs_parameters),
            inputs_as_dictionary=inputs_as_dictionary,
        )

    return decorator
