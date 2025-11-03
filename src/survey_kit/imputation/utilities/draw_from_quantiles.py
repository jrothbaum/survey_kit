"""
Vectorized quantile distribution using Numba JIT compilation with smart parallelization.

This version automatically uses:
- Numba parallel (zero overhead) for Gaussian/Exponential/Lognormal tails
"""

import polars as pl
import narwhals as nw
from narwhals.typing import IntoFrameT
import numpy as np
from typing import Tuple
from scipy.optimize import minimize
from numba import jit, prange

from ...utilities.dataframe import NarwhalsType
from ...utilities.random import set_seed, RandomNumberGenerator
from ... import logger

TAIL_TYPES = {
    "gaussian": 0,
    "exponential": 1,
    "lognormal": 2,
    "pareto": 3,
}


@jit(nopython=True, cache=True)
def _norm_ppf_approx(p: float) -> float:
    """
    Fast approximation of the standard normal inverse CDF (probit function).

    Computes the inverse of the cumulative distribution function for the
    standard normal distribution N(0,1). This answers the question: "What
    z-score has p% of the distribution below it?"

    Uses Acklam's algorithm, which is accurate to approximately 1.15e-9
    relative error across the full range (0, 1).

    Parameters
    ----------
    p : float
        Probability value in (0, 1). Values <= 0 or >= 1 return extreme values.

    Returns
    -------
    float
        The z-score such that P(Z <= z) = p for Z ~ N(0, 1)
        Returns -1e10 for p <= 0 and 1e10 for p >= 1

    Examples
    --------
    >>> _norm_ppf_approx(0.5)
    0.0  # median
    >>> _norm_ppf_approx(0.975)
    1.959963984540054  # ~1.96 for 95% CI
    >>> _norm_ppf_approx(0.025)
    -1.959963984540054  # ~-1.96

    Notes
    -----
    This implementation is needed because scipy.stats.norm.ppf cannot be
    called from Numba's nopython mode. It uses different rational function
    approximations for three regions:

    - Lower tail (p < 0.02425): Special approximation for extreme values
    - Central region (0.02425 <= p <= 0.97575): Highest accuracy region
    - Upper tail (p > 0.97575): Mirror of lower tail

    Algorithm from:
    Peter John Acklam (2003). "An algorithm for computing the inverse normal
    cumulative distribution function." https://web.archive.org/web/20151030215612/
    http://home.online.no/~pjacklam/notes/invnorm/,
    see https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/

    The coefficients a, b, c, d are hardcoded for maximum performance.
    """
    # Acklam's algorithm - accurate to about 1e-9
    if p <= 0.0:
        return -1e10
    if p >= 1.0:
        return 1e10

    # Coefficients in rational approximations
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )

    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )

    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )

    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        # Rational approximation for lower region
        q = np.sqrt(-2 * np.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    elif p <= p_high:
        # Rational approximation for central region
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    else:
        # Rational approximation for upper region
        q = np.sqrt(-2 * np.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )

    return x


@jit(nopython=True, cache=True)
def _fit_tail_params_pareto_upper(
    q0: float, q1: float, alpha0: float, alpha1: float
) -> Tuple[float, float]:
    """
    Fit Upper Pareto parameters (xm, alpha) from two quantiles in Numba.

    Upper Tail Quantile: Q(p) = xm / (1-p)^(1/alpha)

    Uses two-point method (algebraic solution), not optimization.
    """
    xm_fallback = q0 * 0.5
    alpha_fallback = 2.0

    if (
        q0 <= 0
        or q1 <= 0
        or q0 >= q1
        or alpha0 >= alpha1
        or alpha0 >= 1.0
        or alpha1 >= 1.0
    ):
        return (xm_fallback, alpha_fallback)

    # Upper tail ratio: R_q = q1 / q0
    # Tail probability ratio: R_t = (1-alpha0) / (1-alpha1)

    log_q_ratio = np.log(q1 / q0)
    log_tail_ratio = np.log((1 - alpha0) / (1 - alpha1))

    if log_q_ratio == 0 or log_tail_ratio == 0:
        return (xm_fallback, alpha_fallback)

    # alpha = log(R_t) / log(R_q)
    alpha_est = log_tail_ratio / log_q_ratio

    # Back out xm from: q0 = xm / (1-alpha0)^(1/alpha_est)
    # xm = q0 * (1-alpha0)^(1/alpha_est)
    xm_est = q0 * ((1.0 - alpha0) ** (1.0 / alpha_est))

    return (max(xm_est, 1e-6), max(alpha_est, 1e-6))


@jit(nopython=True, cache=True)
def _fit_tail_params_pareto_lower(
    q0: float, q1: float, alpha0: float, alpha1: float
) -> Tuple[float, float]:
    """
    Fit Lower Pareto parameters (xm, alpha) from two quantiles in Numba.

    Lower Tail Quantile: Q(p) = xm * p^(1/alpha)

    Uses two-point method (algebraic solution), not optimization.
    """
    xm_fallback = q1 * 1.5
    alpha_fallback = 2.0

    if (
        q0 <= 0
        or q1 <= 0
        or q0 >= q1
        or alpha0 >= alpha1
        or alpha0 <= 0.0
        or alpha1 <= 0.0
    ):
        return (xm_fallback, alpha_fallback)

    # Quantile ratio: R_q = q1 / q0
    # Probability ratio: R_p = alpha1 / alpha0

    log_q_ratio = np.log(q1 / q0)
    log_p_ratio = np.log(alpha1 / alpha0)

    if log_q_ratio == 0 or log_p_ratio == 0:
        return (xm_fallback, alpha_fallback)

    # alpha = log(R_p) / log(R_q)
    alpha_est = log_p_ratio / log_q_ratio

    # Back out xm from: q0 = xm * alpha0^(1/alpha_est)
    # xm = q0 / alpha0^(1/alpha_est)
    xm_est = q0 / (alpha0 ** (1.0 / alpha_est))

    return (max(xm_est, 1e-6), max(alpha_est, 1e-6))


@jit(nopython=True, parallel=True, cache=True)
def _compute_spline_params_fast(
    quantiles: np.ndarray, alphas: np.ndarray, tail_lower: int, tail_upper: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast Numba-parallel computation for Gaussian/Exponential/Lognormal tails.

    Zero-overhead parallelization using prange. Only works for tail types
    with closed-form solutions (no scipy.optimize needed).

    This is the FAST PATH - automatically selected for:
    - Gaussian tails (0)
    - Exponential tails (1)
    - Lognormal tails (2)

    Parameters
    ----------
    quantiles : np.ndarray
        Shape (n_obs, n_quantiles)
    alphas : np.ndarray
        Shape (n_quantiles,)
    tail_lower : int
        0=Gaussian, 1=Exponential, 2=Lognormal
    tail_upper : int
        0=Gaussian, 1=Exponential, 2=Lognormal

    Returns
    -------
    (y2, tail_param_l, tail_param_u) : tuple of arrays
        y2: shape (n_obs, n_quantiles)
        tail_param_l: shape (n_obs, 2)
        tail_param_u: shape (n_obs, 2)
    """
    n_obs, n_q = quantiles.shape
    y2 = np.zeros((n_obs, n_q))
    tail_param_l = np.zeros((n_obs, 2))
    tail_param_u = np.zeros((n_obs, 2))

    # Parallel loop over observations - automatic multi-core distribution
    for i in prange(n_obs):
        q = quantiles[i, :]

        # Fit lower tail - all logic inlined for Numba
        if tail_lower == 0:  # Gaussian
            z0 = _norm_ppf_approx(alphas[0])
            z1 = _norm_ppf_approx(alphas[1])
            sigma_l = (q[1] - q[0]) / (z1 - z0)
            mu_l = q[0] - sigma_l * z0
            tail_param_l[i, 0] = mu_l
            tail_param_l[i, 1] = sigma_l

        elif tail_lower == 1:  # Exponential
            log_a0 = np.log(alphas[0])
            log_a1 = np.log(alphas[1])
            scale_l = (q[1] - q[0]) / (log_a1 - log_a0)
            loc_l = q[0] - scale_l * log_a0
            tail_param_l[i, 0] = loc_l
            tail_param_l[i, 1] = scale_l

        elif tail_lower == 2:  # Lognormal
            if q[0] > 0 and q[1] > 0:
                log_q0 = np.log(q[0])
                log_q1 = np.log(q[1])
                z0 = _norm_ppf_approx(alphas[0])
                z1 = _norm_ppf_approx(alphas[1])
                sigma_l = (log_q1 - log_q0) / (z1 - z0)
                mu_l = log_q0 - sigma_l * z0
                tail_param_l[i, 0] = mu_l
                tail_param_l[i, 1] = sigma_l
        elif tail_lower == 3:  # Pareto (using Numba-compatible two-point fit)
            xm, alpha = _fit_tail_params_pareto_lower(q[0], q[1], alphas[0], alphas[1])
            tail_param_l[i, 0] = xm
            tail_param_l[i, 1] = alpha

            # (tail_param_l[i, 0], tail_param_l[i, 1]) = _fit_pareto_lower_numba(quantiles,alphas)

        # Fit upper tail - all logic inlined for Numba
        if tail_upper == 0:  # Gaussian
            z0 = _norm_ppf_approx(alphas[n_q - 2])
            z1 = _norm_ppf_approx(alphas[n_q - 1])
            sigma_u = (q[n_q - 1] - q[n_q - 2]) / (z1 - z0)
            mu_u = q[n_q - 2] - sigma_u * z0
            tail_param_u[i, 0] = mu_u
            tail_param_u[i, 1] = sigma_u

        elif tail_upper == 1:  # Exponential
            log_a0 = np.log(1 - alphas[n_q - 2])
            log_a1 = np.log(1 - alphas[n_q - 1])
            scale_u = (q[n_q - 2] - q[n_q - 1]) / (log_a1 - log_a0)
            loc_u = q[n_q - 2] - scale_u * log_a0
            tail_param_u[i, 0] = loc_u
            tail_param_u[i, 1] = scale_u

        elif tail_upper == 2:  # Lognormal
            if q[n_q - 2] > 0 and q[n_q - 1] > 0:
                log_q0 = np.log(q[n_q - 2])
                log_q1 = np.log(q[n_q - 1])
                z0 = _norm_ppf_approx(alphas[n_q - 2])
                z1 = _norm_ppf_approx(alphas[n_q - 1])
                sigma_u = (log_q1 - log_q0) / (z1 - z0)
                mu_u = log_q0 - sigma_u * z0
                tail_param_u[i, 0] = mu_u
                tail_param_u[i, 1] = sigma_u
        elif tail_upper == 3:  # Pareto (using Numba-compatible two-point fit)
            xm, alpha = _fit_tail_params_pareto_upper(
                q[n_q - 2], q[n_q - 1], alphas[n_q - 2], alphas[n_q - 1]
            )
            tail_param_u[i, 0] = xm
            tail_param_u[i, 1] = alpha
            # (tail_param_l[i, 0], tail_param_l[i, 1]) = _fit_pareto_upper_numba(quantiles,alphas)

        # Compute spline second derivatives (natural cubic spline)
        if n_q > 2:
            # Build tridiagonal system for this observation
            A = np.zeros((n_q, n_q))
            b = np.zeros(n_q)

            A[0, 0] = 1.0
            A[n_q - 1, n_q - 1] = 1.0

            for j in range(1, n_q - 1):
                h_left = alphas[j] - alphas[j - 1]
                h_right = alphas[j + 1] - alphas[j]

                A[j, j - 1] = h_left
                A[j, j] = 2.0 * (h_left + h_right)
                A[j, j + 1] = h_right

                b[j] = 6.0 * ((q[j + 1] - q[j]) / h_right - (q[j] - q[j - 1]) / h_left)

            # Solve tridiagonal system for this observation
            y2[i, :] = np.linalg.solve(A, b)

    return y2, tail_param_l, tail_param_u


@jit(nopython=True, cache=True)
def _eval_quantile(
    p: float,
    quantiles: np.ndarray,
    alphas: np.ndarray,
    tail_param_l: np.ndarray,
    tail_param_u: np.ndarray,
    y2: np.ndarray,
    tail_lower: int,
    tail_upper: int,
) -> float:
    """
    JIT-compiled quantile evaluation for a single observation.

    Evaluates the quantile function (inverse CDF) at probability p using
    cubic spline interpolation between observed quantiles, with tail
    extrapolation for probabilities outside the observed range.

    Parameters
    ----------
    p : float
        Probability value in (0, 1) at which to evaluate the quantile
    quantiles : np.ndarray
        Array of observed quantile values for this observation
    alphas : np.ndarray
        Probability levels corresponding to the quantiles
    tail_param_l : np.ndarray
        Lower tail parameters [param1, param2]
    tail_param_u : np.ndarray
        Upper tail parameters [param1, param2]
    y2 : np.ndarray
        Second derivatives for cubic spline interpolation
    tail_lower : int
        Integer value from TAIL_TYPES
    tail_upper : int
        Integer value from TAIL_TYPES

    Returns
    -------
    float
        The quantile value corresponding to probability p

    Notes
    -----
    This function is compiled by Numba to machine code for performance.
    It handles three regions:
    - Lower tail (p < alphas[0]): Parametric or linear extrapolation
    - Interior (alphas[0] <= p <= alphas[-1]): Cubic spline interpolation
    - Upper tail (p > alphas[-1]): Parametric or linear extrapolation
    """
    n_q = len(alphas)

    # Lower tail
    if p < alphas[0]:
        if tail_lower == 0:  # Gaussian - use fitted parameters
            mu = tail_param_l[0]
            sigma = tail_param_l[1]
            z = _norm_ppf_approx(p)
            return mu + sigma * z

        elif tail_lower == 1:  # Exponential - use fitted parameters
            loc = tail_param_l[0]
            scale = tail_param_l[1]
            return loc + scale * np.log(p)

        elif tail_lower == 2:  # Lognormal - use fitted parameters
            mu = tail_param_l[0]
            sigma = tail_param_l[1]
            z = _norm_ppf_approx(p)
            return np.exp(mu + sigma * z)

        elif tail_lower == 3:  # Pareto - use fitted parameters
            xm = tail_param_l[0]
            alpha = tail_param_l[1]
            return xm * p ** (1.0 / alpha)

    # Upper tail
    if p > alphas[n_q - 1]:
        if tail_upper == 0:  # Gaussian - use fitted parameters
            mu = tail_param_u[0]
            sigma = tail_param_u[1]
            z = _norm_ppf_approx(p)
            return mu + sigma * z

        elif tail_upper == 1:  # Exponential - use fitted parameters
            loc = tail_param_u[0]
            scale = tail_param_u[1]
            return loc - scale * np.log(1 - p)

        elif tail_upper == 2:  # Lognormal - use fitted parameters
            mu = tail_param_u[0]
            sigma = tail_param_u[1]
            z = _norm_ppf_approx(p)
            return np.exp(mu + sigma * z)

        elif tail_upper == 3:  # Pareto - use fitted parameters
            xm = tail_param_u[0]
            alpha = tail_param_u[1]
            return xm / (1.0 - p) ** (1.0 / alpha)

    # Interior: find interval
    idx = 0
    for i in range(n_q - 1):
        if alphas[i] <= p <= alphas[i + 1]:
            idx = i
            break

    # Cubic spline interpolation
    h = alphas[idx + 1] - alphas[idx]
    t = (p - alphas[idx]) / h

    q0 = quantiles[idx]
    q1 = quantiles[idx + 1]

    a_coef = q0
    b_coef = (q1 - q0) / h - h * (2.0 * y2[idx] + y2[idx + 1]) / 6.0
    c_coef = y2[idx] / 2.0
    d_coef = (y2[idx + 1] - y2[idx]) / (6.0 * h)

    th = t * h
    return a_coef + b_coef * th + c_coef * th * th + d_coef * th * th * th


@jit(nopython=True, parallel=True, cache=True)
def _draw_random_values_vectorized(
    p: np.ndarray,
    quantiles: np.ndarray,
    alphas: np.ndarray,
    y2: np.ndarray,
    tail_param_l: np.ndarray,
    tail_param_u: np.ndarray,
    tail_lower: int,
    tail_upper: int,
    n_obs: int,
    n_samples: int,
) -> np.ndarray:
    """
    Truly vectorized random sampling using parallel Numba.

    Generates random samples from multiple quantile-defined distributions
    in parallel. This is where the major performance gains come from -
    the entire sampling process runs in compiled machine code with
    parallel execution across CPU cores.

    Parameters
    ----------
    p : np.ndarray
       Uniform random values in [0,1] to convert to quantiles, shape (n_obs, n_samples)
    quantiles : np.ndarray
        Quantile values for all observations, shape (n_obs, n_quantiles)
    alphas : np.ndarray
        Probability levels corresponding to quantiles, shape (n_quantiles,)
    y2 : np.ndarray
        Second derivatives for cubic splines, shape (n_obs, n_quantiles)
    tail_param_l : np.ndarray
        Lower tail parameters for each observation, shape (n_obs, 2)
    tail_param_u : np.ndarray
        Upper tail parameters for each observation, shape (n_obs, 2)
    tail_lower : int
        integer from TAIL_TYPES
    tail_upper : int
        integer from TAIL_TYPES
    n_obs : int
        Number of observations (distributions)
    n_samples : int
        Number of samples to generate per observation

    Returns
    -------
    np.ndarray
        Random samples, shape (n_obs, n_samples)
    """
    samples = np.empty((n_obs, n_samples))

    # Parallel loop over observations
    for i in prange(n_obs):
        for j in range(n_samples):
            pi = p[i, j]
            samples[i, j] = _eval_quantile(
                pi,
                quantiles[i, :],
                alphas,
                tail_param_l[i, :],
                tail_param_u[i, :],
                y2[i, :],
                tail_lower,
                tail_upper,
            )

    return samples


class DrawFromQuantileVectors:
    """
    High-performance vectorized quantile distribution with smart parallelization.

    Parameters
    ----------
    quantiles : IntoFrameT
        Dataframe of quantile values, shape (n_obs, n_quantiles)
    alphas : np.ndarray or list[float]
        Probability levels (must be sorted, in (0,1))
    tails : str or tuple[str, str], default='gaussian'
        Tail extrapolation method. Can be:
        - Single string: same tail for both sides
        - Tuple: (lower_tail, upper_tail) for asymmetric tails
        Options: 'gaussian', 'exponential', 'lognormal', 'pareto'
    seed : int, default=0
        Random seed for reproducibility (0 means no seed set)

    Examples
    --------
    >>> # Symmetric Gaussian tails (fast Numba parallel)
    >>> quantiles = np.array([[10, 25, 50, 75, 150]])
    >>> alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
    >>> dist = DrawFromQuantileVectors(quantiles, alphas, tails='gaussian')
    >>> df = dist.draw_random_values(n_draws=2)

    >>> # Asymmetric tails for income data
    >>> dist = DrawFromQuantileVectors(quantiles, alphas,
    ...                                tails=('gaussian', 'lognormal'))

    >>> # Pareto with parallel for large dataset
    >>> large_quantiles = np.random.randn(500_000, 5).cumsum(axis=1)
    >>> dist = DrawFromQuantileVectors(large_quantiles, alphas,
    ...                                tails='pareto')
    """

    def __init__(
        self,
        df_quantiles: IntoFrameT,
        alphas: np.ndarray | list[float],
        tails: str | tuple[str, str] = "gaussian",
        seed: int = 0,
    ) -> None:
        self.nw_type = NarwhalsType(df_quantiles)

        columns = nw.from_native(df_quantiles).lazy().collect_schema().names()
        df_quantiles = (
            self.nw_type.to_polars()
            .select(pl.concat_list(pl.all()).list.sort().list.to_struct(fields=columns))
            .unnest(columns[0])
        )

        self.quantiles = np.atleast_2d(nw.from_native(df_quantiles).to_numpy()).astype(
            np.float64
        )
        self.alphas = np.asarray(alphas, dtype=np.float64)

        if type(tails) is str:
            tail_lower = tails
            tail_upper = tails
        else:
            tail_lower = tails[0]
            tail_upper = tails[1]

        try:
            self.tail_lower = TAIL_TYPES[tail_lower]
        except:
            message = f"{tail_lower} is not an acceptable tail value.  Acceptable tails are {list(TAIL_TYPES.keys())}"
            logger.error(message)
            raise Exception(message)

        try:
            self.tail_upper = TAIL_TYPES[tail_upper]
        except:
            message = f"{tail_upper} is not an acceptable tail value.  Acceptable tails are {list(TAIL_TYPES.keys())}"
            logger.error(message)
            raise Exception(message)

        self.n_obs = self.quantiles.shape[0]
        self.n_quantiles = self.quantiles.shape[1]
        self.seed = seed

        # Validate
        if len(self.alphas) != self.n_quantiles:
            raise ValueError("Length of alphas must match number of quantiles")
        if not np.all(np.diff(self.alphas) > 0):
            raise ValueError("alphas must be strictly increasing")
        if np.any((self.alphas <= 0) | (self.alphas >= 1)):
            raise ValueError("alphas must be in (0, 1)")

        # Compute spline parameters
        self._compute_spline_params()

    def _compute_spline_params(self) -> None:
        # Fast path: Numba parallel (near-zero overhead)
        # logger.info(f"Using fast Numba parallel for {n_obs} observations")
        self.y2, self.tail_param_l, self.tail_param_u = _compute_spline_params_fast(
            self.quantiles, self.alphas, self.tail_lower, self.tail_upper
        )

    def draw_random_values(self, n_draws: int = 1) -> IntoFrameT:
        """
        Generate random samples from the quantile-defined distribution(s).

        This is the main sampling method and where the major performance gains
        occur. Uses JIT-compiled parallel code to generate samples at extremely
        high speed.

        Parameters
        ----------
        n_draws : int, default=1
            Number of samples to generate per distribution

        Returns
        -------
        IntoFrameT with columns ["p", "values"]
            - p: Uniform random values in [0, 1] used for sampling
            - values: The quantile values corresponding to those probabilities

        Notes
        -----
        **First Call Overhead:**
        The first call to this method will be slower (~10 seconds, maybe) because
        Numba needs to JIT-compile the sampling function. Subsequent calls
        are extremely fast.


        Examples
        --------
        Generate samples from a single distribution:

        >>> quantiles = np.array([[1, 2, 3, 4, 5]])
        >>> alphas = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> dist = DrawFromQuantileVectors(quantiles, alphas)
        >>> df = dist.draw_random_values(10)

        Generate samples from multiple distributions:

        >>> quantiles = np.random.randn(100, 9).cumsum(axis=1)
        >>> alphas = np.linspace(0.1, 0.9, 9)
        >>> dist = DrawFromQuantileVectors(quantiles, alphas)
        >>> df = dist.draw_random_values(10)
        """

        if self.seed > 0:
            set_seed(self.seed)

        p = RandomNumberGenerator().uniform(low=0, high=1, size=(self.n_obs, n_draws))
        samples = _draw_random_values_vectorized(
            p=p,
            quantiles=self.quantiles,
            alphas=self.alphas,
            y2=self.y2,
            tail_param_l=self.tail_param_l,
            tail_param_u=self.tail_param_u,
            tail_lower=self.tail_lower,
            tail_upper=self.tail_upper,
            n_obs=self.n_obs,
            n_samples=n_draws,
        )

        if n_draws == 1:
            rename_p = {"column_0": "p"}
            rename_values = {"column_0": "values"}
        else:
            rename_p = {f"column_{i}": f"p_{i}" for i in range(n_draws)}
            rename_values = {f"column_{i}": f"values_{i}" for i in range(n_draws)}

        df_p = pl.DataFrame(p).rename(rename_p)
        df_values = pl.DataFrame(samples).rename(rename_values)

        return (
            nw.from_native(
                self.nw_type.from_polars(pl.concat([df_p, df_values], how="horizontal"))
            )
            .lazy_backend(self.nw_type)
            .to_native()
        )

    def __repr__(self):
        tail_names = {v: k for k, v in TAIL_TYPES.items()}
        lower_name = tail_names.get(self.tail_lower, "unknown")
        upper_name = tail_names.get(self.tail_upper, "unknown")
        tails_str = (
            f"({lower_name}, {upper_name})" if lower_name != upper_name else lower_name
        )
        return f"DrawFromQuantileVectors(n_obs={self.n_obs:,}, tails={tails_str})"
