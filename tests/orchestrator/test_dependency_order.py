"""Regression test: "parent is set to run" must propagate through whole chains
regardless of the order functions are passed in.

Run with:  pytest tests/orchestrator/test_dependency_order.py
     or:  python tests/orchestrator/test_dependency_order.py
"""

from itertools import permutations

import survey_kit.orchestration.dependency_order as dependency_order
from survey_kit.orchestration.dependency_order import FunctionDependencyOrder
from survey_kit.orchestration.need_to_run import RunBecause


class _StubFunction:
    """Just the attributes FunctionDependencyOrder reads."""

    def __init__(self, name: str, inputs: list[str], outputs: list[str]):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.parent_functions = []
        self.child_functions = []
        self.run = False
        self.run_because = RunBecause.Not

    def __repr__(self) -> str:
        return self.name


def _only_out_of_date(*names: str):
    """Replacement for update_need_to_run: only `names` look stale on disk."""

    def fake(function, update) -> None:
        function.run = function.name in names
        function.run_because = (
            RunBecause.UpdateByDate if function.run else RunBecause.Not
        )

    return fake


def _chain() -> dict[str, _StubFunction]:
    #   A -> B -> C, plus D, which joins B with an independent E
    return {
        "A": _StubFunction("A", ["raw_a"], ["a"]),
        "B": _StubFunction("B", ["a"], ["b"]),
        "C": _StubFunction("C", ["b"], ["c"]),
        "E": _StubFunction("E", ["raw_e"], ["e"]),
        "D": _StubFunction("D", ["b", "e"], ["d"]),
    }


def _run_flags(order: tuple[str, ...], stale: tuple[str, ...]) -> dict[str, bool]:
    original = dependency_order.update_need_to_run
    dependency_order.update_need_to_run = _only_out_of_date(*stale)
    try:
        functions = _chain()
        FunctionDependencyOrder(function_list=[functions[n] for n in order])
        return {name: f.run for name, f in functions.items()}
    finally:
        dependency_order.update_need_to_run = original


def test_rerun_propagates_in_any_input_order():
    #   A is stale, so everything downstream of A (B, C, D) must rerun; E must not
    expected = {"A": True, "B": True, "C": True, "D": True, "E": False}
    for order in permutations("ABCDE"):
        assert _run_flags(order, stale=("A",)) == expected, order


def test_parent_set_to_run_reason():
    original = dependency_order.update_need_to_run
    dependency_order.update_need_to_run = _only_out_of_date("A")
    try:
        functions = _chain()
        FunctionDependencyOrder(
            function_list=[functions[n] for n in ("C", "B", "A", "D", "E")]
        )
        assert functions["A"].run_because == RunBecause.UpdateByDate
        for name in ("B", "C", "D"):
            assert functions[name].run_because == RunBecause.ParentSetToRun, name
        assert functions["E"].run_because == RunBecause.Not
    finally:
        dependency_order.update_need_to_run = original


def test_nothing_stale_runs_nothing():
    flags = _run_flags(("E", "D", "C", "B", "A"), stale=())
    assert not any(flags.values())


if __name__ == "__main__":
    test_rerun_propagates_in_any_input_order()
    test_parent_set_to_run_reason()
    test_nothing_stale_runs_nothing()
    print("ok")
