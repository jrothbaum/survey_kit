import os
import logging
import narwhals as nw
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


def nw_monkey_patch():
    if not hasattr(nw.Expr, "mod"):

        def mod(self, other):
            return self - (self.floordiv(other) * other)

        # Monkey-patch it onto the Expr class
        nw.Expr.mod = mod

    if not hasattr(nw.Expr, "floordiv"):

        def floordiv(self, other):
            return (self / other).cast(nw.Int64)

        # Monkey-patch it onto the Expr class
        nw.Expr.floordiv = floordiv

    if not hasattr(nw.Expr, "is_missing"):

        def is_missing(self):
            return self.is_null()  # | self.is_nan()

        nw.Expr.is_missing = is_missing

        def is_not_missing(self):
            return ~self.is_missing()

        nw.Expr.is_not_missing = is_not_missing

    if not hasattr(nw.Expr, "ne"):

        def ne(self, other):
            return self != other

        nw.Expr.ne = ne

    if not hasattr(nw.Expr, "eq"):

        def eq(self, other):
            return self == other

        nw.Expr.eq = eq

    if not hasattr(nw.Expr, "gt"):

        def gt(self, other):
            return self > other

        nw.Expr.gt = gt

    if not hasattr(nw.Expr, "ge"):

        def ge(self, other):
            return self >= other

        nw.Expr.ge = ge

    if not hasattr(nw.Expr, "lt"):

        def lt(self, other):
            return self < other

        nw.Expr.lt = lt

    if not hasattr(nw.Expr, "le"):

        def le(self, other):
            return self <= other

        nw.Expr.le = le

    if not hasattr(nw.Expr, "pow"):

        def pow(self, other):
            return self**other

        nw.Expr.pow = pow

    if not hasattr(nw.Expr, "is_not_null"):

        def is_not_null(self):
            return ~self.is_null()

        nw.Expr.is_not_null = is_not_null


nw_monkey_patch()
