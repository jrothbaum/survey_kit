from __future__ import annotations
from typing import Optional

import igraph as ig

from .utilities import UpdateParams
from .function import Function
from .need_to_run import update_need_to_run, RunBecause
from .path_with_fallbacks import PathWithFallbacks

from .. import logger


class FunctionDependencyOrder:
    """
    Analyzes function dependencies and determines execution order.

    FunctionDependencyOrder takes a list of functions and creates an execution
    plan that respects dependencies while maximizing parallelism. It builds
    dependency graphs and groups functions that can run simultaneously.

    Parameters
    ----------
    function_list : list, optional
        List of Function objects to analyze. Default is None (empty list).
    update : UpdateParams, optional
        Configuration for determining which functions need execution.
        Default creates new UpdateParams.

    Attributes
    ----------
    function_list : list
        List of functions being analyzed.
    update : UpdateParams
        Update checking configuration.
    loadOrder : list
        Computed execution order as list of function groups.

    Examples
    --------
    >>> functions = [func1, func2, func3]  # Function objects
    >>> ordering = FunctionDependencyOrder(
    ...     function_list=functions,
    ...     update=UpdateParams(update_by_date=True)
    ... )

    Notes
    -----
    The object orders the functions into two groups:
        1. Those the can be run immediately as they do not have any upstream dependencies
            (i.e. no parent function whose output is that function's input)
        2. Those that must wait for another function to complete before they can start
    """

    def __init__(self, function_list: list = None, update: UpdateParams = None):
        if function_list is None:
            function_list = []

        if update is None:
            update = UpdateParams()

        #   Deduplicate functions but keep in original order, more or less
        function_list = list(dict.fromkeys(function_list))
        self.function_list = function_list
        self.update = update

        #   Run the dependency ordering functions
        # self.d_output_source_dict = self._build_dependency_dict_maps()
        # self.DetermineDependencies()

        (self.run_initial, self.run_on_parent_complete) = self._order_dependencies()

        self._set_functions_to_run()

    def _order_dependencies(self) -> tuple[list[Function], list[Function]]:
        function_list = self.function_list

        producers = {}
        producer_indices = {}
        consumers = {}
        output_inputs = {}

        for indexi, fi in enumerate(function_list):
            for outputi in fi.outputs:
                if type(outputi) is PathWithFallbacks:
                    outputi = outputi.path_main

                if outputi in producers:
                    message = f"Multiple functions produce {outputi}: {producers[outputi]} and {fi}"
                    logger.error(message)
                    raise ValueError(message)

                producers[outputi] = fi
                producer_indices[outputi] = indexi
                output_inputs[outputi] = fi.inputs.copy()

            for inputi in fi.inputs:
                if type(inputi) is PathWithFallbacks:
                    inputi = inputi.path_main
                if inputi not in consumers:
                    consumers[inputi] = []

                consumers[inputi].append(fi)

        edges = []

        for indexi, fi in enumerate(function_list):
            for inputi in fi.inputs:
                if inputi in producers:
                    producer_index = producer_indices[inputi]
                    produceri = producers[inputi]

                    if producer_index != indexi:
                        edges.append((producer_index, indexi))

                        if fi.parent_functions is None:
                            fi.parent_functions = []

                        fi.parent_functions.append(produceri)

            for outputi in fi.outputs:
                if outputi in consumers:
                    fi.child_functions.extend(consumers[outputi])

        g = ig.Graph(directed=True)
        g.add_vertices(function_list)
        g.add_edges(edges)

        topo_order = g.topological_sorting()

        no_parents = []
        with_parents = []
        for indexi in topo_order:
            fi = function_list[indexi]
            if len(fi.parent_functions):
                with_parents.append(fi)
            else:
                no_parents.append(fi)

        return (no_parents, with_parents)

    def _set_functions_to_run(self):
        for fi in self.function_list:
            if not fi.run:
                update_need_to_run(fi, self.update)

        for fi in self.function_list:
            if not fi.run:
                fi.run = any([f_parent.run for f_parent in fi.parent_functions])
                if fi.run:
                    fi.run_because = RunBecause.ParentSetToRun
