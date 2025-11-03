from __future__ import annotations

import numpy as np

#   from scipy.optimize import minimize, nnls, lsq_linear, least_squares
#   import scipy.linalg.interpolative
from timeit import default_timer as timer

from .. import logger


def survey_calibration(
    *,
    x_sample: np.ndarray,
    weights0: np.ndarray,
    mean_population_moments: np.ndarray,
    penalty_fn: str = "log_diff",
    step_tol: float = 1e-6,
    n_iterations: int = 250,
    n_iterations_linestep: int = 500,
    check_step_tol_only: bool = True,
    rank_regularization_term: float = 1e-8,
    initial_guess: np.ndarray | None = None,
) -> (np.array, str):
    """
    A function to perform survey calibration, transforming an initial set of survey weights to
    a final set of weights by choosing the ratios (r) of old to new weights using the criterion

    w_new = weights0 * r_max = weights0 * [arg min(r) sum(weights0 * phi(r))]
    s.t. (weights0[:,None] * x_sample) @ r = mean_population_moments

    phi defualts to the "raking" function r * log(r) - r + 1, which also has the fastest and most
    stable solving behavior. Previously other criterions were
    implemented but showed slow convergence and issues with finding a feasible point in practice and
    so are no longer available.
    Weights are allowed to be anything that is proportially correct and are ideally 1-ish, but population moments
    must be in means.


    Parameters
    ----------
    x_sample :
        The n x m sample of observable values. Can be rank-deficient but should not contain NaNs
        or other invalid values, so missings must be removed ahead of time.
    agg_population_moments :
        The n-vector or n x 1 population moments to be matched.
        Need to be aggregated, not the sample averages. The new weights will satisfy
        sum(w[i] * x[i,j] = moment[j], so if you want averages adjust your weights or moments manually.
    weights0 :
        Initial n-vector set of sampling weights to be improved.
    penalty_fn : optional
        The criterion function above. Anything but "log_diff" is not suggested at this time. The default is "log_diff".
    step_tol :
        The size of the weight vector step that will define convergence. The default is 1e-6.
    rank_regularization_term : optional
        The term that enforces a unique next step across the primal/dual pair. If the constraints are
        full rank, this can be 0, but if they are not, 0 will error out with a singular matrix error.
        A larger term should theoretically still get the same result but take longer, and in
        practice may fail. Numerical experiements have shown stability for values of at least 1e-1.
        The default is 1e-8.

    Algorithm:
    ----------
    The algorithm is a slight adjustment of a SQP method as outlined
    in section 18.1 of Nocedal and Wright "Numerical Optimization" (2006), based on a regularization
    idea see in Armand and Omheni (2015)'s Optimization Methods and Software paper to simply add a
    small multiple of the identity matrix to enforce invertibility in the SQP step calculation regardless
    of the rank of the observable matrix.
    I construct primal and dual steps p_k and p_l from
    [ HessLagranian     A.T ][p_k]  = [- Jac_f_k + A.T * lambda_k]
    [      A         eps * I][p_l]    [- c]

    where A is the constraint LHS matrix, HessLagrangian is the diagonal matrix of the second
    deriviatives of the crterion function f, eps is a small parameter, J is the Jacobian operator,
    lambda_k are the current Lagrange multipliers, and c is the current constraint violation.
    Given a large number of parameters, direct construction and solution of the system is infeasible.
    Instead, I use the diagonal nature of the Hessian of the Lagrangian along with the Schur matrix
    block-inversion formula to reduce the memory requirements and skip construction/inversion of any sample-size x sample-size matrices.

    Due to this alternative approach, a large number of population moments and colinear observables are feasible.
    Colinearity may break feasibility if the corresponding population moments are not colinear themselves; that is the
    end user's responsiblity.

    In the case when the system is inconsistent (rank deficient on LHS, augmented matrix's rank increased with RHS column added)
    we first solve the problem

    delta_b = arg min(b) (Ax - b)'(Ax-b) s.t. x >= 0

    where b are the initial population mean moments.
    delta_b is the least-square smallest adjustment to b that will allow some set of weights to satisfy the constraint.
    We then use b + delta_b as the new RHS moments to match, return a message noting this, and return the
    adjustment vector. A situation like this really requires another model that allows for imperfect measurement
    of constraints.

    Returns
    -------
    (new_weights, return_code): (np.array, str, np.array)
        The set of new weights that equate the population moments and the weighted moments while minimizing
        the information distance between the old and new weights.

    """
    #   logger.info("x_sample hash:" + str(zlib.crc32(x_sample.data.tobytes())))
    #   logger.info("mean_population_moments hash:" + str(zlib.crc32(mean_population_moments.data.tobytes())))
    agg_population_moments, x_sample, weights0 = check_types_and_shapes(
        mean_population_moments, x_sample, weights0
    )
    n, k = x_sample.shape
    lambda0 = np.ones(k)
    multipliers = lambda0
    cap_a_mat = weights0[:, None] * x_sample
    #   ratio, moment_adjustment = initial_guess(cap_a_mat, agg_population_moments, feasible_initial_calculation=False)
    #   agg_population_moments = agg_population_moments + moment_adjustment

    if initial_guess is None:
        ratio = np.ones(n)
    else:
        ratio = initial_guess / weights0

    candidate_step = np.full(n, np.inf)

    logger.info(
        f"{'#':^5}{'Criterion':^14}{'Cons. Violation':^18}{'Step Size':^14}{'Merit':^14}"
    )

    current_iteration = 0

    continue_weighting = True

    #   Continue dummy set at the end of each loop
    while continue_weighting:
        last_candidate = candidate_step
        current_iteration += 1
        merit_value = merit(
            ratio,
            agg_population_moments=agg_population_moments,
            cap_a_mat=cap_a_mat,
            weights0=weights0,
            penalty_function=penalty_fn,
        )

        if current_iteration <= 25:
            b_print = True
        elif current_iteration <= 50:
            b_print = (current_iteration % 5) == 0
        else:
            b_print = (current_iteration % 10) == 0

        if b_print:
            criterion_value = criterion(ratio, weights0, penalty_fn)[0]
            constraint_violation = np.linalg.norm(
                cap_a_mat.T @ ratio - agg_population_moments
            )
            step_size = np.linalg.norm(candidate_step)
            logger.info(
                f"{current_iteration:^5}{criterion_value:^14.6f}{constraint_violation:^18.6f}{step_size:^14.8f}{merit_value:^14.4f}"
            )
        # print(np.linalg.norm(candidate_step),
        #       criterion(ratio, weights0, penalty_fn)[0],
        #       np.linalg.norm(cap_a_mat.T @ ratio - agg_population_moments),
        #       merit_value)
        func, objfunc_grad, inverse_hess_diag = criterion(ratio, weights0, penalty_fn)
        cap_a_mat_sqrtL = np.sqrt(inverse_hess_diag[:, None]) * cap_a_mat
        #   TODO - This is the slow step that might benefit from a sparse calculation
        a_delta2L_aT = np.dot(cap_a_mat_sqrtL.T, cap_a_mat_sqrtL)
        del cap_a_mat_sqrtL

        proj_newtonstep = np.dot(cap_a_mat.T, inverse_hess_diag * objfunc_grad)
        b = agg_population_moments
        c = cap_a_mat.T @ ratio - b
        rhs = proj_newtonstep - a_delta2L_aT @ multipliers - c
        lambda_step = np.linalg.solve(
            a_delta2L_aT - rank_regularization_term * np.eye(a_delta2L_aT.shape[0]), rhs
        )
        proj_lambda_steps = cap_a_mat @ lambda_step
        lagrangian_descent_dir = -objfunc_grad + cap_a_mat @ multipliers
        ratio_step_unit = (
            proj_lambda_steps + lagrangian_descent_dir
        ) * inverse_hess_diag
        backtrack = 1.0
        candidate_step = ratio_step_unit.ravel()
        iters = 0
        while (
            merit(
                ratio + candidate_step,
                agg_population_moments=agg_population_moments,
                cap_a_mat=cap_a_mat,
                weights0=weights0,
                penalty_function=penalty_fn,
            )
            >= merit_value
        ):
            backtrack *= 0.8
            candidate_step = backtrack * ratio_step_unit
            iters += 1
            if iters == n_iterations_linestep:
                if np.linalg.norm(last_candidate) < step_tol:
                    return_msg = "success"
                else:
                    return_msg = "failure"

                logger.info(f"Finished in line step improvement: {return_msg}")

                return (ratio * weights0, return_msg)
        ratio += candidate_step.ravel()
        multipliers += lambda_step.ravel() * backtrack
        return_msg = "success"

        #   Too many iterations?
        if n_iterations < current_iteration:
            return_msg = "failure"
            logger.info(f"Did not converge by step {n_iterations}: {return_msg}")
            return (ratio * weights0, return_msg)

        if check_step_tol_only:
            continue_weighting = np.linalg.norm(candidate_step) > step_tol
        else:
            continue_weighting = (
                np.linalg.norm(candidate_step) > step_tol
                and constraint_violation > step_tol
            )

    logger.info(return_msg)
    return (ratio * weights0, return_msg)


def criterion(g, weights0, penalty_function, logistic_bounds=(0.6, 1.4)):
    if np.any(g <= 0.0):
        return (np.inf, np.inf, np.inf)
    if penalty_function == "log_diff":
        out = (
            np.sum(weights0 * (g * np.log(g) - g + 1)),
            weights0 * np.log(g),
            g / weights0,
        )
    else:
        raise NotImplementedError(
            "All other attempted criterions functions performed very poorly. If you want a new one, the only function needing editing is this one."
        )
    return out


def merit(ratio, *, agg_population_moments, cap_a_mat, weights0, penalty_function):
    if np.any(ratio <= 0.0):
        return np.inf
    crit, _, _ = criterion(ratio, weights0, penalty_function)
    cv = np.abs(cap_a_mat.T @ ratio - agg_population_moments)
    return crit + 3.0 * np.sum(np.abs(cv))
    #   return crit + 1000.0 * np.sum(np.abs(cv))


def constraint_violation(x_sample, weights, agg_population_moments):
    return x_sample.T @ weights / np.sum(weights) - agg_population_moments


def check_types_and_shapes(mean_population_moments, x_sample, weights0):
    """
    A variety of type and shape checks. This is quite un-Pythonic,
    but this code may be called from other languages and so
    needs to be a) protected from shape-change issues and
    b) be extremely clear in its error messages.
    """
    agg_population_moments = mean_population_moments * np.sum(weights0)
    try:
        if type(agg_population_moments) != np.ndarray:
            raise TypeError("agg_population_moments must be ndarray")
        agg_population_moments = np.squeeze(np.array(agg_population_moments))
    except IndexError:
        logger.info("\nPopulation_moments not usable as ndarray vector\n")
        raise
    try:
        if type(x_sample) != np.ndarray:
            raise TypeError("x_sample must be ndarray")
        x_sample = np.array(x_sample)
    except IndexError:
        logger.info("x_sample not usable as 2-d array")
        raise
    if type(weights0) != np.ndarray:
        raise TypeError
    weights0 = np.squeeze(np.array(weights0))
    if agg_population_moments.ndim > 1:
        raise IndexError("agg_population_moments can't be multidimensional")
    if x_sample.ndim != 2:
        x_sample = np.squeeze(x_sample)
        if x_sample.ndim == 1:
            x_sample = x_sample[:, None]
        elif x_sample.ndim != 2:
            raise IndexError(
                "x_sample needs to be either a vector (for one covariate) or two-dim"
            )
    if weights0.ndim > 1:
        raise IndexError("weights0 can't be multidimensional")
    return agg_population_moments, x_sample, weights0


if __name__ == "__main__":
    import polars as pl

    weights0 = (
        pl.read_csv("weights0.csv", infer_schema_length=None)
        .select("wgt_na_march")
        .to_numpy()
        .ravel()
    )
    x_sample = pl.read_csv("x_sample.csv", infer_schema_length=None).drop("").to_numpy()
    mean_population_moments = (
        pl.read_csv("population_moments.csv", infer_schema_length=None)
        .select(pl.col("^Target.*$"))
        .to_numpy()
        .ravel()
    )
    ebw_weights = (
        pl.read_csv("ebw_weights.csv", infer_schema_length=None).to_numpy().ravel()
    )

    z = timer()
    rowreps = 1
    colreps = 1
    weights0 = np.tile(weights0, rowreps)
    x_sample = np.tile(x_sample, (rowreps, colreps))
    logger.info("Sample size, covariates:", x_sample.shape)

    mean_population_moments = np.tile(mean_population_moments, colreps)

    ebw_weights = np.tile(ebw_weights, rowreps) / rowreps

    new_weights, return_msg, weight_adjustments = survey_calibration(
        x_sample=x_sample,
        mean_population_moments=mean_population_moments,
        weights0=weights0,
        penalty_fn="log_diff",
        rank_regularization_term=1e-3,
    )
    logger.info(new_weights)
    logger.info("Seconds to generate reweights:", timer() - z)
