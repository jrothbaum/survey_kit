"""MKL-accelerated linear algebra, with fallbacks for CPUs/platforms MKL doesn't support.

pypardiso and sparse_dot_mkl both require Intel's MKL runtime, which only ships
wheels for x86_64 and can still fail to load there (e.g. the shared library
isn't on the loader's search path inside a venv). Rather than let that crash
the whole subpackage on import, everything here degrades gracefully:

    dot products / gram matrix : sparse_dot_mkl -> scipy/numpy
    SPD sparse solve           : pypardiso -> scikit-sparse (CHOLMOD) -> scipy
    general sparse solve       : pypardiso -> scipy
    (CHOLMOD assumes real symmetric positive definite input, so it's only
    used for the SPD case.)
"""

from __future__ import annotations

import ctypes.util as _ctypes_util
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .. import logger
from .typing import Any, AnyArray, FArr


def _make_mkl_rt_discoverable() -> None:
    """Point MKL_RT at the venv's bundled library if the loader can't find it.

    pip's ``mkl`` package drops ``libmkl_rt.so.<N>`` into ``<venv>/lib``, which
    isn't on the dynamic linker's search path, so ``ctypes.util.find_library``
    fails even though the library is present right there.
    """
    if os.environ.get("MKL_RT") or _ctypes_util.find_library("mkl_rt"):
        return
    lib_dir = os.path.join(sys.prefix, "lib")
    if not os.path.isdir(lib_dir):
        return
    for name in sorted(os.listdir(lib_dir)):
        if name.startswith(("libmkl_rt.so", "libmkl_rt.dylib")):
            os.environ["MKL_RT"] = os.path.join(lib_dir, name)
            return


_make_mkl_rt_discoverable()

try:
    import pypardiso as _pypardiso

    HAS_PYPARDISO = True
except ImportError:
    _pypardiso = None
    HAS_PYPARDISO = False
    logger.warning(
        "pypardiso is unavailable (no usable MKL runtime on this CPU/platform); "
        "sparse solves will fall back to scikit-sparse/scipy, which are slower."
    )

try:
    import sparse_dot_mkl as _sdmkl

    HAS_SPARSE_DOT_MKL = True
except ImportError:
    _sdmkl = None
    HAS_SPARSE_DOT_MKL = False
    logger.warning(
        "sparse_dot_mkl is unavailable (no usable MKL runtime on this CPU/platform); "
        "falling back to scipy/numpy matrix products, which are slower."
    )

try:
    from sksparse.cholmod import CholmodError as _CholmodError
    from sksparse.cholmod import cholesky as _cholmod_cholesky

    HAS_CHOLMOD = True
except ImportError:
    _cholmod_cholesky = None

    class _CholmodError(Exception):  # type: ignore[no-redef]
        """Placeholder so `except _CholmodError` is always valid when unavailable."""

    HAS_CHOLMOD = False
    logger.warning(
        "scikit-sparse is unavailable (missing the package or the system "
        "SuiteSparse/CHOLMOD library it builds against); the SPD sparse solve "
        "will fall back further to scipy, which is slower."
    )


def dot_product_mkl(a: AnyArray, b: AnyArray, *, cast: bool = False) -> AnyArray:
    """Matrix product, via MKL when available, else the plain ``@`` operator."""
    if HAS_SPARSE_DOT_MKL:
        return _sdmkl.dot_product_mkl(a, b, cast=cast)
    return a @ b


def gram_matrix_mkl(a: AnyArray, *, cast: bool = False) -> AnyArray:
    """``A.T @ A``, upper-triangular only, via MKL when available, else scipy."""
    if HAS_SPARSE_DOT_MKL:
        return _sdmkl.gram_matrix_mkl(a, cast=cast)
    gram = a.T @ a
    return sp.triu(gram, format="csr") if sp.issparse(gram) else np.triu(gram)


class SparseLinearSolver:
    """
    Repeatedly solve ``(matrix + regularizer * I) @ x = b`` for a fixed size,
    where ``matrix`` is rebuilt fresh on every call (e.g. a Newton step's
    Hessian, or a Woodbury-identity Schur complement) and ``regularizer`` is a
    caller-supplied Tikhonov term.

    Backend order (fastest first, picked once at construction):
        spd=True:  PARDISO (SPD mode) -> CHOLMOD -> scipy (SuperLU)
        spd=False: PARDISO (general)  -> scipy (SuperLU)

    Only pass ``spd=True`` when ``matrix`` is guaranteed real symmetric
    positive (semi)definite -- CHOLMOD assumes that and will silently use only
    half the matrix, or hard-error, if it isn't true.

    On a solve failure or a non-finite result (matrix effectively
    rank-deficient), ``regularizer`` is increased and the same backend is
    retried; after repeated failures it falls through to the next backend.
    Reproduces the "bump the penalty and retry" behavior the dense code path
    already used, so callers get the same rank-deficiency robustness
    regardless of which backend ends up solving the system.
    """

    _MAX_RETRIES_PER_BACKEND = 8
    _RETRY_GROWTH = 10.0

    def __init__(self, size: int, *, spd: bool):
        self._eye = sp.eye_array(size, format="csc")
        self._spd = spd
        self._backend_names = self._select_backends()
        self._warned: set[tuple[str, str]] = set()
        logger.info(
            f"SparseLinearSolver(size={size}, spd={spd}) using: "
            f"{self._backend_names[0]}"
        )

    def _select_backends(self) -> list[str]:
        backends = []
        if HAS_PYPARDISO:
            backends.append("pypardiso")
        if self._spd and HAS_CHOLMOD:
            backends.append("cholmod")
        backends.append("scipy")  # always available: scipy is a hard dependency
        return backends

    def _solve_with_backend(self, name: str, matrix: AnyArray, rhs: FArr) -> FArr:
        if name == "pypardiso":
            kwargs: dict[str, Any] = {"set_matrix_type": 2} if self._spd else {}
            return _pypardiso.spsolve(matrix, rhs, **kwargs)
        if name == "cholmod":
            factor = _cholmod_cholesky(matrix.tocsc())
            return factor(rhs)
        return spla.spsolve(matrix.tocsc(), rhs)

    def _note_fallback(self, from_name: str, to_name: str, reason: str) -> None:
        key = (from_name, to_name)
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(
                f"Sparse solve backend '{from_name}' failed ({reason}); "
                f"falling back to '{to_name}'. (Won't repeat this message "
                "for this solver instance.)"
            )

    def solve(self, matrix: AnyArray, rhs: FArr, regularizer: float = 0.0) -> FArr:
        """Solve ``(matrix + regularizer * I) @ x = b``."""
        lhs = (matrix + regularizer * self._eye).tocsc()
        current_regularizer = regularizer
        last_reason = "unknown error"
        for i, name in enumerate(self._backend_names):
            for _ in range(self._MAX_RETRIES_PER_BACKEND):
                try:
                    x = self._solve_with_backend(name, lhs, rhs)
                    if np.all(np.isfinite(x)):
                        return x
                    last_reason = "non-finite solution (near-singular matrix)"
                except (RuntimeError, ValueError, _CholmodError) as err:
                    last_reason = str(err)
                current_regularizer = max(current_regularizer, 1e-10) * self._RETRY_GROWTH
                lhs = (matrix + current_regularizer * self._eye).tocsc()
            if i + 1 < len(self._backend_names):
                self._note_fallback(name, self._backend_names[i + 1], last_reason)
        raise np.linalg.LinAlgError(
            "All available sparse solvers failed even after regularization retries: "
            f"{last_reason}"
        )
