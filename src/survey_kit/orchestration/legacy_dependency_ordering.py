#   Kept around for things like R package dependency resolver
#       where I don't care about duplicate c's in outputs (like a->c, b->c )
#       and just want to order packages
#       (duplicate outputs cause the faster method to throw an error, as it should)
from __future__ import annotations
from typing import TYPE_CHECKING

import os
import logging

if TYPE_CHECKING:
    from .function import Function


class LegacyOrdering:
    def __init__(self, function_list: list = None):
        if function_list is None:
            function_list = []

        self.function_list = function_list
        self.d_output_source_dict = self._build_dependency_dict_maps()
        for thisFunction in self.function_list:
            thisFunction.parent_functions.extend(
                self.find_parents(function=thisFunction)
            )

        self.loadOrder = self.order_dependencies()

    def _build_dependency_dict_maps(self) -> dict[str, list[Function]]:
        output_to_function = {}
        for f in self.function_list:
            for output in f.outputs:
                if output not in output_to_function:
                    output_to_function[output] = []

                output_to_function[output].append(f)

        return output_to_function

    def order_dependencies(self):
        """
        With each function's dependencies, set the sequence of functions to be run
            Populates a list of list (loadOrder) where the first list is
            a list of the functions to be run sequentially
            i.e. run loadOrder[0], then run loadOrder[1]
        The nested lists can be run simultaneously
            i.e. loadOrder[1][0] and loadOrder[1][1] as each item in
            loadOrder[1] depends on the items run in loadOrder[0]

        """

        loadOrder = []
        alread_loaded = []
        pending = self.function_list
        while pending:
            (alread_loaded, loadlist, pending) = self.dependencies_parent_set_to_load(
                alread_loaded=alread_loaded, pending=pending
            )
            loadOrder.append(loadlist.copy())

        return loadOrder

    def dependencies_parent_set_to_load(
        self, alread_loaded: list = None, pending: list = None
    ):
        if alread_loaded is None:
            alread_loaded = []
        if pending is None:
            pending = []

        remaining = []
        loadlist = []

        for functioni in pending:
            bLoad = True
            #   If all parents are loaded, then load this in this group
            for parenti in functioni.parent_functions:
                bThisParentLoaded = False
                for loadedi in alread_loaded:
                    bThisParentLoaded = bThisParentLoaded or parenti == loadedi
                    if bThisParentLoaded:
                        break

                bLoad = bLoad and bThisParentLoaded
            if bLoad:
                #   All parents loaded, set to load in this group
                loadlist.append(functioni)
            else:
                #   Not all parents loaded, leave in "remaining" list
                remaining.append(functioni)

        #   Update the list of alread_loaded functions after this group
        if len(alread_loaded) == 0:
            alread_loaded = loadlist
        else:
            alread_loaded.extend(loadlist)

        return (alread_loaded, loadlist, remaining)

    def find_parents(self, function: Function = None):
        parents = set()

        for inputi in function.inputs:
            if inputi in self.d_output_source_dict:
                for parenti in self.d_output_source_dict[inputi]:
                    if parenti != function:
                        parents.add(parenti)

        return list(parents)
