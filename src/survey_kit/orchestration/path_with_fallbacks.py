from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import os
import psutil
from copy import deepcopy

from .config import Config
from .utilities import UpdateParams

from ..serializable import Serializable
from .. import logger

if TYPE_CHECKING:
    from .function import Function


class CheckInputs(Serializable):
    def __init__(
        self,
        key_values: dict[str, str],
        inputs: list[str],
        update: UpdateParams | None = None,
    ):
        self.key_values = key_values
        self.inputs = inputs

        if update is None:
            update = UpdateParams(update_by_date=True, Update_by_used_file_list=True)

        self.update = update

    def need_check(self, d: dict[str, str]):
        b_check_inputs = True
        for keyi, valuei in self.key_values.items():
            b_check_inputs = (
                b_check_inputs and keyi in d.keys() and d.get(keyi, None) == valuei
            )

        return b_check_inputs

    def check(self, path: str, version_to_use: str = "") -> bool:
        from .dependency_order import FunctionDependencyOrder

        if version_to_use != "":
            inputs = self._replace_version(version_to_use)
        else:
            inputs = self.inputs

        f = Function()
        f.inputs = inputs
        f.outputs = [path]
        fdo = FunctionDependencyOrder([f], update=self.update)

        return not f.run

    def _replace_version(self, version_to_use: str) -> list[str]:
        old_version = Config().data_with_version
        new_version = f"{Config().data_root}/{version_to_use}"

        if old_version == new_version:
            #   Nothing to do
            return self.inputs
        else:
            adjusted_inputs = []
            for filei in self.inputs:
                if filei.startswith(old_version):
                    filei = new_version + filei[len(old_version) :]

                adjusted_inputs.append(filei)

            return adjusted_inputs

    # def inputs_exist(self,
    #                  version_to_use:str) -> list[str]:
    #     pass


class PathWithFallbacks(Serializable):
    def __init__(
        self,
        input_path: str = "",
        fallback_options: dict[str, list[str]] | None = None,
        with_data_root: bool = False,
        no_version: bool = False,
        options_order: list[dict[str, str]] | None = None,
        inputs_by_options: list[CheckInputs] | None = None,
        quietly: bool = True,
    ):
        """
        PathWithFallbacks - pass a path with {field} as placeholders
            and a list of fields that need replacing
            with an ordered list of options for each and then
            check if the file exists or could exist...

        Parameters
        ----------
        input_path : str, optional
            The path of form "/my_path/{fieldi}/{fieldj}"
            where fieldi and j are keys in fallback_options
            The default is ""
        fallback_options : dict[str,list[str]], Optional
            Keys: list of fields in the path to check
            Values: options for each key where the options are ordered
                by preference
            An example:
                path = "/root/CPS_ASEC/{year}.parquet"
                fallback_options["year"] = ["2024",
                                            "2023",
                                            "2022"]
                This would say use any of the 3 years of files, preferring
                2024, then 2023, then 2022.  If 2024 doesn't exist, but 2023 does,
                it would resolve to:
                path = "/root/CPS_ASEC/2023.parquet"

            The default is None
        with_data_root, bool, optional
            Add the data root, with version # at the start of path
            If "Version" is in fallback_options, will check them,
            if not, will use current version in environment variable
        no_version, bool, optional
            Include the version in the data root
            The default is True
        options_order, list[dict[str,str]], optional
            Prespecify the preference ordering
            An example:
                Suppose you have Version and Year in fallback_options.keys()
                and you want the order to be something other than just
                higher version > lower version and within version, higher year > lower year,
                you can pass an options order to set it to any ordering:
                {Version:V3,Year:2023} > {Version:V3,Year:2021} > {Version:V2,Year:2021} > {Version:V2,Year:2022}

                This ordering seems weird and I don't know why you'd want it
                but it's costless to make this available as an option
        inputs_by_options, list[CheckInputs], optional
            Only use certain options if the inputs match
            The idea here, is only use an old version file
            if it's the right file
        quietly, bool, optional
            Suppress log messages?
            The default is True
        Returns
        -------
        None.

        """

        if with_data_root:
            #   Make sure it starts with a slash
            if not input_path.startswith("/"):
                input_path = "/" + input_path

            if "Version" in fallback_options.keys():
                #   Version is a choice, then put in the {Version}
                input_path = Config().data_root + "/{Version}" + input_path
            else:
                #   Otherwise, stick to the current version
                if no_version:
                    input_path = Config().data_root + input_path
                else:
                    input_path = Config().data_with_version + input_path

        self.input_path = input_path
        self.fallback_options = fallback_options
        self.path = ""
        self.options_order = options_order
        self.inputs_by_options = inputs_by_options
        self.quietly = quietly

        #   Get the ordering if needed
        if self.options_order is None and self.fallback_options is not None:
            for keyi, valuei in self.fallback_options.items():
                self.options_order = self._get_ordering(
                    keyi, valuei, self.options_order
                )

        self.checked_order = []
        self._get_path_list()
        #   self.resolve_path()

    def _get_path_list(self) -> list[str]:
        if self.options_order is not None:
            for optioni in self.options_order:
                path_to_check = self.input_path.format_map(optioni)

                self.checked_order.append(path_to_check)

    def resolve_path(self) -> PathWithFallbacks:
        found_path = False
        for indexi, path_to_check in enumerate(self.checked_order):
            if not found_path:
                d_replace = self.options_order[indexi]
                file_exists = os.path.isfile(path_to_check) or os.path.isdir(
                    path_to_check
                )

                if self.inputs_by_options is not None:
                    for checki in self.inputs_by_options:
                        if checki.need_check(d_replace):
                            found_path = checki.check(
                                path_to_check,
                                version_to_use=d_replace.get("Version", ""),
                            )
                else:
                    found_path = file_exists

                if found_path:
                    self.path = os.path.normpath(path_to_check)

        return self

    def _get_ordering(
        self,
        key: str,
        values: list[str],
        existing_ordering: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        #   Do nothing if the key isn't in the path to be replaced
        if f"{{{key}}}" not in self.input_path:
            if not self.quietly:
                logger.info(
                    f"     Dropping {{{key}}} from resolve_path: " + self.input_path
                )
            return existing_ordering

        this_ordering = [{key: valuei} for valuei in values]

        if existing_ordering is None:
            return this_ordering
        else:
            final_ordering = []
            for existingi in existing_ordering:
                for newi in this_ordering:
                    combinedi = existingi.copy()
                    combinedi.update(newi)
                    final_ordering.append(combinedi)

            return final_ordering

    def __eq__(self, other: object) -> bool:
        if type(other) is str:
            return self.path_main == other
        elif type(other) is PathWithFallbacks:
            if len(self.options_order) != len(other.options_order):
                return False

            return set(self.checked_order) == set(other.checked_order)
        else:
            return False

    #   For methods that expect something path-like
    def __fspath__(self):
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> str:
        return hash(self.__str__())

    def __str__(self) -> str:
        if self.path == "":
            return os.path.normpath(self.path_main)
        else:
            return os.path.normpath(self.path)

    def __add__(self, other) -> str:
        return self._apply_string_method("__add__", other)

    def __radd__(self, other) -> str:
        self = deepcopy(self)
        d_properties = vars(self)

        for keyi, valuei in d_properties.items():
            if type(valuei) is str:
                d_properties[keyi] = other + valuei
            elif type(valuei) is list:
                for i in range(0, len(valuei)):
                    if type(valuei[i]) is str:
                        valuei[i] = other + valuei[i]

        return self

    def __copy__(self) -> PathWithFallbacks:
        d_properties = vars(self)
        out = PathWithFallbacks()
        for keyi, valuei in d_properties.items():
            setattr(out, keyi, valuei)

        return out

    def __deepcopy__(self, memo) -> PathWithFallbacks:
        d_properties = deepcopy(vars(self))
        out = PathWithFallbacks()
        for keyi, valuei in d_properties.items():
            setattr(out, keyi, valuei)

        return out

    def __getattr__(self, name):
        """
        Handle string methods that modify strings
        """

        if hasattr(str, name) and callable(getattr(str, name)):
            allowed_methods_class = [
                "lower",
                "upper",
                "strip",
                "lstrip",
                "rstrip",
                "replace",
            ]

            def wrapper(*args, **kwargs):
                if name in allowed_methods_class:
                    return self._apply_string_method(name, *args, **kwargs)
                else:
                    #   Doesn't return a class or do on the main string...
                    method = getattr(self.__str__(), name)
                    return method(*args, **kwargs)

            return wrapper

    def _apply_string_method(self, method: str, *args, **kwargs) -> PathWithFallbacks:
        self = deepcopy(self)

        d_properties = vars(self)

        for keyi, valuei in d_properties.items():
            if type(valuei) is str:
                if keyi == "path" and valuei == "":
                    # do nothing
                    pass
                else:
                    d_properties[keyi] = getattr(valuei, method)(*args, **kwargs)
            elif type(valuei) is list:
                for i in range(0, len(valuei)):
                    if type(valuei[i]) is str:
                        valuei[i] = getattr(valuei[i], method)(*args, **kwargs)

        return self

    def _prepend(self, prepend_str: str) -> PathWithFallbacks:
        self = deepcopy(self)
        d_properties = vars(self)

        for keyi, valuei in d_properties.items():
            if type(valuei) is str:
                d_properties[keyi] = prepend_str + valuei
            elif type(valuei) is list:
                for i in range(0, len(valuei)):
                    if type(valuei[i]) is str:
                        valuei[i] = prepend_str + valuei[i]

        return self

    @property
    def path_main(self) -> str:
        if len(self.checked_order):
            return os.path.normpath(self.checked_order[0])
        else:
            return ""


def version_list() -> list[str]:
    return Config().versions


def as_output_path(path: str | PathWithFallbacks) -> str:
    if type(path) is PathWithFallbacks:
        return path.path_main
    else:
        return path
