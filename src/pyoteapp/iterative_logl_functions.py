"""
A collection of functions for fast MLE fits to light curves.

MLE: Maximum Likelihood Estimation

The 'fit' is to an array of intensities (y[]: float) that comprise the light
curve.
"""

import numpy as np
import sys
from math import log, pi, sqrt, exp
from typing import Tuple
from pyoteapp.solverUtils import calcNumCandidatesFromEventSize
from pyoteapp.solverUtils import calcNumCandidatesFromDandRlimits
from pyoteapp.solverUtils import model, logLikelihoodLine
from pyoteapp.likelihood_calculations import cum_loglikelihood, aicc
from numba import njit, jit

MIN_FLOAT: float = sys.float_info.min


@jit
def add_entry(ynew: float, s: float, s2: float, n: int, calc_var: bool):
    """Adds an entry to the metrics, s, s2, and n.

    s:  previous value of sum of y[]
    s2: previous value of sum of y[]*y[]
    n:  previous number of entries in the metric
    """

    n = n + 1
    s = s + ynew
    s2 = s2 + ynew * ynew

    if calc_var:
        var = (s2 - s * s / n) / n  # This is sigma**2
    else:
        var = None

    return s, s2, n, var


@jit
def sub_entry(ynew: float, s: float, s2: float, n: int, calc_var: bool):
    """Subtracts an entry from the metrics, s, s2, and n.

    s:  previous value of sum of y[]
    s2: previous value of sum of y[]*y[]
    n:  previous number of entries in the metric
    """

    n = n - 1
    s = s - ynew
    s2 = s2 - ynew * ynew

    if calc_var:
        var = (s2 - s * s / n) / n  # This is sigma**2
    else:
        var = None

    return s, s2, n, var


def calc_metric_iteratively(y: np.ndarray) -> Tuple[float, float, int, float]:
    """Calculates a metric iteratively (term by term) for test purposes only.

    This is expected to be very slow compared to simply using numpy and on an
    array of size 1000, the numpy version was 600 times faster.

    y: array of floats
    """

    # Set initial values for iteration
    s = 0.0
    s2 = 0.0
    n = 0
    var = None

    for ynew in np.nditer(y):
        s, s2, n, var = add_entry(ynew, s, s2, n, calc_var=True)

    return s, s2, n, var


@jit
def calc_metric_numpy(y: np.ndarray):
    """Used for timing comparisons and initializing a metric from a large y[].

    It calculates the metrics using fast numpy operations.
    """

    n = y.size
    s2 = np.sum(y * y)
    s = y.sum()
    var = (s2 - s * s / n) / n  # This is sigma**2

    return s, s2, n, var


StdAnswer = Tuple[int, int, float, float, float, float, float]
"""StdAnswer is: d, r, b, a, sigmaB, sigmaA, metric """


# Because njit on locate_fixed_event_position() gave Win11 assert errors, we don't
# njit that function, so we can't njit this function either because it uses locate_fixed_event_position()
# @njit  # cache=True did not work for this function --- gave a pickling error
def find_best_event_from_min_max_size(
        y: np.ndarray, left: int, right: int, min_event: int, max_event: int):
    """Finds the best size and location for an event >=  min and <=  max"""

    # The only time this function is called with a y containing 1 element
    # is during the import of this module where the call is made to force the jit
    # compiler into action --- a work-around to the pickle error problem.
    if y.size == 1:
        yield -1.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0

    max_metric = 0.0
    d_best = 0
    r_best = 0
    b_best = 0.0
    a_best = 0.0
    sigma_b_best = 0.0
    sigma_a_best = 0.0
    not_started = True

    num_candidates = calcNumCandidatesFromEventSize(eventType="DandR",
                                                    left=left, right=right, minSize=min_event, maxSize=max_event)

    solution_counter = 0

    for event in range(min_event, max_event + 1):
        d, r, b, a, sigma_b, sigma_a, metric, sol_count = \
            locate_fixed_event_position(y, left, right, event)

        # Initialize the 'best' values
        if not_started:
            max_metric = metric
            d_best = d
            r_best = r
            b_best = b
            a_best = a
            sigma_b_best = sigma_b
            sigma_a_best = sigma_a
            not_started = False

        if metric >= max_metric and b > a:
            max_metric = metric
            d_best = d
            r_best = r
            b_best = b
            a_best = a
            sigma_b_best = sigma_b
            sigma_a_best = sigma_a

        solution_counter += sol_count

        # if not sigma_b_best >= 0.0:
        #     print(f'locate_fixed_event_position() returned sigma_b_best == {sigma_b_best} when looking at event size: {event}')
        #     print(f"d: {d}  r: {r}  d_best: {d_best}  r_best: {r_best}  left: {left}  right: {right}")
        #     print(f"y[d]: {y[d]}   y[r]: {y[r]}")
        #     print(f"y[d_best]: {y[d_best]}   y[r_best]: {y[r_best]}\n")

        yield 1.0, solution_counter / num_candidates, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Here we test for solution being better than straight line
    # if not (sigma_a_best >= 0.0 and sigma_b_best >= 0.0):
    #     print(f"sigma_b_best: {sigma_b_best}  sigma_a_best: {sigma_a_best}  d_best: {d_best}  r_best: {r_best}")
    assert(sigma_a_best >= 0.0)
    assert(sigma_b_best >= 0.0)
    if not solution_is_better_than_straight_line(
            y, left, right, d_best, r_best, b_best, a_best, sigma_b_best, sigma_a_best):
        # yield 'no event present', solution_counter / num_candidates
        yield -1.0, solution_counter / num_candidates, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0

    # yield d_best, r_best, b_best, a_best, sigma_b_best, sigma_a_best, max_metric
    yield 0.0, 1.0, d_best, r_best, b_best, a_best, sigma_b_best, sigma_a_best, max_metric


def find_best_r_only_from_min_max_size(
        y: np.ndarray, left: int, right: int, min_event: int, max_event: int):
    """Finds the best r-only location for r >=  min_event and <=  max_event"""

    assert min_event >= 1
    assert max_event <= right - left

    def update_best_solution():
        nonlocal max_metric, b_best, a_best, sigma_b, sigma_a
        nonlocal r_best

        max_metric = metric
        b_best = b
        a_best = a
        sigma_b = sqrt(b_var)
        sigma_a = sqrt(a_var)
        r_best = r

    def calc_metric():
        nonlocal a_var, b_var
        max_var = max(a_var, b_var, sys.float_info.min)

        if a_var <= 0.0:
            a_var = max_var
        if b_var <= 0.0:
            b_var = max_var
        return -b_n * log(b_var) - a_n * log(a_var)

    # These get changed by the first call to update_best_solution but
    # have the be set to proper type to satisfy type checking.
    metric = 0.0
    max_metric = 0.0
    r_best = 0
    b_best = 0.0
    a_best = 0.0
    sigma_b = 0.0
    sigma_a = 0.0

    r = left + min_event

    # Use numpy version of metric calculator to initialize iteration variables

    # changed in 4.4.7
    # b_s, b_s2, b_n, b_var = calc_metric_numpy(y[r + 1:right + 1])
    b_s, b_s2, b_n, b_var = calc_metric_numpy(y[r:right + 1])

    a_s, a_s2, a_n, a_var = calc_metric_numpy(y[left:r])

    b = b_s / b_n
    a = a_s / a_n

    # Calculate metric for initial position of r
    metric = calc_metric()
    update_best_solution()

    r_final = left + max_event

    while r < r_final:
        # calc metric for next r position from current position

        # changed in 4.4.7
        # b_s, b_s2, b_n, b_var = sub_entry(y[r+1], b_s, b_s2, b_n, True)
        b_s, b_s2, b_n, b_var = sub_entry(y[r], b_s, b_s2, b_n, True)

        a_s, a_s2, a_n, a_var = add_entry(y[r], a_s, a_s2, a_n, True)
        r += 1

        metric = calc_metric()
        b = b_s / b_n
        a = a_s / a_n

        goodSolution = solution_is_better_than_straight_line(
                y, left, right, -1, r, b, a, sqrt(b_var), sqrt(a_var), k=3)

        if metric > max_metric and b > a and goodSolution:
            update_best_solution()

    if b_best <= a_best:
        # yield 'no event present', 1.0
        yield -1.0, 1.0, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0

    event_size_found = r_best - left
    if event_size_found == max_event or event_size_found == min_event:
        # Invalid event size --- invalid limit
        yield -1.0, 1.0, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0

    # Here we test for the best solution being better than straight line
    if not solution_is_better_than_straight_line(
            y, left, right, -1, r_best, b, a, sigma_b, sigma_a, k=3):
        # yield 'no event present', 1.0
        yield -1.0, 1.0, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0

    # yield None, r_best, b_best, a_best, sigma_b, sigma_a, max_metric
    yield 0.0, 1.0, -1, r_best, b_best, a_best, sigma_b, sigma_a, max_metric


def find_best_d_only_from_min_max_size(
        y: np.ndarray, left: int, right: int, min_event: int, max_event: int):
    """Finds the best d-only location for max_event >= event >=  min_event """

    assert min_event >= 1
    assert max_event <= right - left

    def update_best_solution():
        nonlocal max_metric, b_best, a_best, sigma_b, sigma_a
        nonlocal d_best

        max_metric = metric
        b_best = b
        a_best = a
        sigma_b = sqrt(b_var)
        sigma_a = sqrt(a_var)
        d_best = d

    def calc_metric():
        nonlocal a_var, b_var
        max_var = max(a_var, b_var, sys.float_info.min)

        if a_var <= 0.0:
            a_var = max_var
        if b_var <= 0.0:
            b_var = max_var
        return -b_n * log(b_var) - a_n * log(a_var)

    # These get changed by the first call to update_best_solution but
    # have to be set to proper type to satisfy type checking.
    metric = 0.0
    max_metric = 0.0
    d_best = 0
    b_best = 0.0
    a_best = 0.0
    sigma_b = 0.0
    sigma_a = 0.0

    d = right - max_event  # Initial d position

    # Use numpy version of metric calculator to initialize iteration variables
    b_s, b_s2, b_n, b_var = calc_metric_numpy(y[left:d])

    # changed in 4.4.7
    # a_s, a_s2, a_n, a_var = calc_metric_numpy(y[d+1:right+1])
    # a_s, a_s2, a_n, a_var = calc_metric_numpy(y[d:right+1])
    # changed in 4.4.8
    a_s, a_s2, a_n, a_var = calc_metric_numpy(y[d:right])

    b = b_s / b_n
    a = a_s / a_n

    # print(b, a, b_n, a_n)

    # Calculate metric for initial position of d
    metric = calc_metric()
    update_best_solution()

    # d_final = right - min_event
    # changed in 4.4.7
    d_final = right - min_event + 1

    while d < d_final:
        # calc metric for next d position from current position
        b_s, b_s2, b_n, b_var = add_entry(y[d], b_s, b_s2, b_n, True)

        # changed in 4.4.7
        # a_s, a_s2, a_n, a_var = sub_entry(y[d+1], a_s, a_s2, a_n, True)
        a_s, a_s2, a_n, a_var = sub_entry(y[d], a_s, a_s2, a_n, True)

        d += 1

        metric = calc_metric()
        b = b_s / b_n
        a = a_s / a_n

        goodSolution = solution_is_better_than_straight_line(
            y, left, right, d, -1, b, a, sqrt(b_var), sqrt(a_var), k=3)

        if metric > max_metric and b > a and goodSolution:
            update_best_solution()

    if b_best <= a_best:
        # yield 'no event present', 1.0
        yield -1.0, 1.0, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0

    event_size_found = right - d_best
    if event_size_found == max_event or event_size_found == min_event:
        # Invalid event size --- invalid limit
        yield -1.0, 1.0, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0

    if not solution_is_better_than_straight_line(
            y, left, right, d_best, -1, b, a, sigma_b, sigma_a, k=3):
        # yield 'no event present', 1.0
        yield -1.0, 1.0, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0

    # yield d_best, None, b_best, a_best, sigma_b, sigma_a, max_metric
    yield 0.0, 1.0, d_best, -1, b_best, a_best, sigma_b, sigma_a, max_metric


# The njit operator has been removed because it seems to have been the culprit in
# assert errors occuring on Win11 machines but not Win10
# @njit(cache=True)
def locate_fixed_event_position(
        y: np.ndarray, left: int, right: int,
        event_size: int) -> Tuple[int, int, float, float, float, float,
                                  float, int]:
    """Finds the best location for a fixed size event"""

    d = left

    # changed in 4.4.7
    # r = d + event_size + 1
    r = d + event_size

    # assert(r < right)

    # Use numpy version of metric calculator to initialize iteration variables

    # changed in 4.4.8
    # b_s, b_s2, b_n, b_var = calc_metric_numpy(y[r+1:right+1])
    b_s, b_s2, b_n, b_var = calc_metric_numpy(y[r:right+1])

    # changed in 4.4.8
    # a_s, a_s2, a_n, a_var = calc_metric_numpy(y[left+1:r])
    a_s, a_s2, a_n, a_var = calc_metric_numpy(y[left:r])

    b = b_s / b_n
    a = a_s / a_n

    # Calculate metric for initial position of event at extreme left
    # ========== calc_metric() ===========
    max_var = max(a_var, b_var, MIN_FLOAT)
    if a_var <= 0.0:
        a_var = max_var
    if b_var <= 0.0:
        b_var = max_var
    metric = - b_n * log(b_var) - a_n * log(a_var)
    # ========== calc_metric() ===========

    # ======= update_best_solution() ========
    max_metric = metric
    b_max = b
    a_max = a
    sigma_b = sqrt(b_var)
    sigma_a = sqrt(a_var)
    d_max = d
    r_max = r
    # ======= update_best_solution() ========

    # The metric used is the variable part of logL(D,R), droping the constant
    # part and ignoring a factor of 2.  The full logL(D,R) would have been:
    #
    # -0.5 * (b_n*log(b_var) + a_n*log(a_var) + (b_n + a_n) * (1 + log(2*pi))
    #
    # We use the reduced form to speed the calculation yet achieve a MLE
    # solution

    # metrics = [metric]  # For use during development

    solution_count = 0

    while r < right - 1:
        # calc metric for next event position from current position
        b_s, b_s2, b_n, b_var = add_entry(y[d], b_s, b_s2, b_n, False)

        # changed in 4.4.8
        # b_s, b_s2, b_n, b_var = sub_entry(y[r+1], b_s, b_s2, b_n, True)
        b_s, b_s2, b_n, b_var = sub_entry(y[r], b_s, b_s2, b_n, True)

        a_s, a_s2, a_n, a_var = add_entry(y[r], a_s, a_s2, a_n, False)

        # changed in 4.4.8
        # a_s, a_s2, a_n, a_var = sub_entry(y[d+1], a_s, a_s2, a_n, True)
        a_s, a_s2, a_n, a_var = sub_entry(y[d], a_s, a_s2, a_n, True)

        b = b_s / b_n
        a = a_s / a_n

        # ========== calc_metric() ===========
        max_var = max(a_var, b_var, MIN_FLOAT)
        if a_var <= 0.0:
            a_var = max_var
        if b_var <= 0.0:
            b_var = max_var
        metric = - b_n * log(b_var) - a_n * log(a_var)
        # ========== calc_metric() ===========

        # The calculation above looks ahead 1 position. Before calculating goodSolution,
        # we therefore advance d and r to correctly refer to the event position calculated.
        # Move to next position
        d += 1
        r += 1

        goodSolution = solution_is_better_than_straight_line(
            y, left, right, d, r, b, a, sqrt(b_var), sqrt(a_var), k=4)

        if metric > max_metric and b > a and goodSolution:
            # ======= update_best_solution() ========
            max_metric = metric
            b_max = b
            a_max = a
            sigma_b = sqrt(b_var)
            sigma_a = sqrt(a_var)
            d_max = d
            r_max = r
            # ======= update_best_solution() ========

        solution_count += 1

    return d_max, r_max, b_max, a_max, sigma_b, sigma_a, max_metric, solution_count


@njit  # cache=True gave pickling error
def locate_event_from_d_and_r_ranges(
        y: np.ndarray, left: int, right: int, d_start: int, d_end: int,
        r_start: int,  r_end: int):
    """Finds the best size and location for event specified by d & r  ranges"""

    # The only time this function is called with a y containing 1 element
    # is during the import of this module where the call is made to force the jit
    # compiler into action --- a work-around to the pickle error problem.
    if y.size == 1:
        yield -1.0, 0.0, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0

    num_candidates = calcNumCandidatesFromDandRlimits(
        eventType='DandR',
        d_start=d_start, d_end=d_end, r_start=r_start, r_end=r_end)

    clump_size = np.ceil(num_candidates / 50)
    solution_counter = 0

    d = d_start

    max_metric = 0.0
    d_best = 0
    r_best = 0
    b_s_best = 0.0
    a_s_best = 0.0
    b_var_best = 0.0
    a_var_best = 0.0
    b_n_best = 0
    a_n_best = 0
    not_started = True

    while d <= d_end:
        # Use numpy version of metric calculator to initialize iteration
        # variables for current d and initial r_start
        r = r_start

        if d > left:
            b_sl, b_s2l, b_nl, b_varl = calc_metric_numpy(y[left:d])
            # Lefthand wing
        else:
            b_sl = 0.0
            b_s2l = 0.0
            b_nl = 0
            b_varl = 0.0

        # changed in 4.4.7
        # b_sr, b_s2r, b_nr, b_varr = calc_metric_numpy(y[r+1:right+1])
        b_sr, b_s2r, b_nr, b_varr = calc_metric_numpy(y[r:right+1])

        # Righthand wing

        b_s = b_sl + b_sr
        b_s2 = b_s2l + b_s2r
        b_n = b_nl + b_nr
        b_var = b_varl + b_varr

        # changed in 4.4.7
        # a_s, a_s2, a_n, a_var = calc_metric_numpy(y[d+1:r])
        a_s, a_s2, a_n, a_var = calc_metric_numpy(y[d:r])

        # ============== calc_metric() =================
        max_var = max(a_var, b_var, MIN_FLOAT)

        if a_var <= 0.0:
            a_var = max_var
        if b_var <= 0.0:
            b_var = max_var
        metric = - b_n * log(b_var) - a_n * log(a_var)
        # ============== calc_metric() =================

        if not_started:
            # =========== update_best_solution() =======
            max_metric = metric
            d_best = d
            r_best = r
            b_s_best = b_s
            a_s_best = a_s
            b_var_best = b_var
            a_var_best = a_var
            b_n_best = b_n
            a_n_best = a_n
            # =========== update_best_solution() =======
            not_started = False

        b = b_s / b_n
        a = a_s / a_n
        if metric >= max_metric and b > a:
            # =========== update_best_solution() =======
            max_metric = metric
            d_best = d
            r_best = r
            b_s_best = b_s
            a_s_best = a_s
            b_var_best = b_var
            a_var_best = a_var
            b_n_best = b_n
            a_n_best = a_n
            # =========== update_best_solution() =======

        while r < r_end:
            b_s, b_s2, b_n, b_var = sub_entry(y[r], b_s, b_s2, b_n, True)
            # changed in 4.4.8
            # a_s, a_s2, a_n, a_var = add_entry(y[r-1], a_s, a_s2, a_n, True)
            a_s, a_s2, a_n, a_var = add_entry(y[r], a_s, a_s2, a_n, True)

            r += 1
            # ============== calc_metric() =================
            max_var = max(a_var, b_var, MIN_FLOAT)

            if a_var <= 0.0:
                a_var = max_var
            if b_var <= 0.0:
                b_var = max_var
            metric = - b_n * log(b_var) - a_n * log(a_var)
            # ============== calc_metric() =================

            b = b_s / b_n
            a = a_s / a_n

            goodSolution = solution_is_better_than_straight_line(
                y, left, right, d, r, b, a, sqrt(b_var), sqrt(a_var), k=4)

            if metric >= max_metric and b > a and goodSolution:
                # =========== update_best_solution() =======
                max_metric = metric
                d_best = d
                r_best = r
                b_s_best = b_s
                a_s_best = a_s
                b_var_best = b_var
                a_var_best = a_var
                b_n_best = b_n
                a_n_best = a_n
                # =========== update_best_solution() =======

            solution_counter += 1
            if solution_counter % clump_size == 0:
                # yield 'fractionDone', solution_counter / num_candidates
                yield 1.0, solution_counter / num_candidates, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0
        d += 1

    b = b_s_best / b_n_best
    a = a_s_best / a_n_best
    sigma_b = sqrt(b_var_best)
    sigma_a = sqrt(a_var_best)

    # Here we test for solution being better than straight line
    if not solution_is_better_than_straight_line(
            y, left, right, d_best, r_best, b, a, sigma_b, sigma_a, k=4):
        # yield 'no event present', solution_counter / num_candidates
        yield -1.0, solution_counter / num_candidates, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0

    yield 0.0, 1.0, d_best, r_best, b, a, sigma_b, sigma_a, max_metric


@njit(cache=True)
def solution_is_better_than_straight_line(y, left=-1, right=-1, d=-1, r=-1, b=0.0, a=0.0, sigma_b=0.0,
                                          sigma_a=0.0, k=4):

    # The only time that the result of this routine is important is for very
    # low snr signals.  In that case, sigma_b and sigma_a are approximately
    # equal anyway.  For other cases, we want to 'score' the straight line
    # against the signal in as equal a manner as possible, so we will use a
    # common noise value for all points.  Here we calculate that value...

    big_sigma = np.float64(max(sigma_b, sigma_a))

    # And here we make sure it never gets too small...

    if big_sigma < (b - a) / 100.0:  # 100 == max snr
        big_sigma = (b - a) / 100.0

    # If the current snr is greater than 3, the solution is always better
    # than a straight line, so we skip the calculations.

    if (b - a) / big_sigma > 3.0:
        return True

    # If this point is reached, a valid scoring needs to be performed.

    num_pts = y.size
    m, sigma = model(
        B=b, A=a, D=d, R=r, sigmaB=big_sigma, sigmaA=big_sigma,
        numPts=num_pts)

    solution_logl = cum_loglikelihood(y, m, sigma, left, right)

    # lineScore = logLikelihoodLine(y, sigmaB=big_sigma, left=left, right=right)
    lineScore = logLikelihoodLine(y, sigmaB=np.sqrt(np.var(y)), left=left, right=right)

    aiccSol = aicc(solution_logl, right - left + 1, k)
    aiccLine = aicc(lineScore, right - left + 1, 1)

    if aiccSol < aiccLine:
        pLine = exp(-(aiccLine - aiccSol) / 2)
    else:
        pLine = 1.00
    if pLine > 0.001:
        return False
    else:
        return True


def calc_logl_from_metric(s: float, s2: float, n: int) -> Tuple[float, float]:

    sigma2 = (s2 / n - (s / n) * (s / n))

    # -log(sqrt(2*pi)) = -0.9189385332046727

    return -n * 0.9189385332046727 - n / 2.0 - n * log(sigma2) / 2.0, sigma2


def cum_loglikelihood_raw(y, m, sigma):
    """ numpy accelerated sum of loglikelihoods --- for test purposes

        Args:
            y (ndarray):     measured values
            m (ndarray):     associated mean values (the 'model')
            sigma (ndarray): associated stdev values
    """

    n = len(y)

    ans = -n * np.log(np.sqrt(2*pi))

    ans -= np.sum(np.log(sigma))

    ans -= (np.sum((y - m) ** 2 / sigma ** 2) / 2.0)

    return ans, np.var(y)


def loglikelihood(y, m, sigma):
    """ calculate ln(likelihood) of a single point from a gaussian distribution

    Args:
        y (float):     measured value
        m (float):     mean (expected model value)
        sigma (float): stdev of measurements

    Returns:
        natural logarithm of un-normalized probability based on Gaussian
        distribution

    """
    # log(x) is natural log (base e)
    # -log(sqrt(2*pi)) = -0.9189385332046727 = -log(2*pi)/2
    # t1 = -log(sqrt(2*pi))

    t1 = -0.9189385332046727
    t2 = -log(sigma)
    t3 = -(y - m) ** 2 / (2 * sigma ** 2)
    return t1 + t2 + t3


@njit
def bob():
    print("Hello from Bob")
    num_pts = 30
    m, sigma = model(
        B=4.0, A=1.0, D=19, R=29, sigmaB=1.0, sigmaA=1.0,
        numPts=num_pts)

    # m += np.random.normal(0.0, 0.01, 30)
    # print(m.size, sigma.size)
    # model.inspect_types()
    # ans = solution_is_better_than_straight_line(m, 0, 29, 10, 20, 4.0, 1.0, 1.0, 1.0)
    # print(ans)
    # solution_is_better_than_straight_line.inspect_types()

    left = 1
    right = num_pts
    evt = 9
    noise_sigma = 0.01

    noise = np.random.normal(0.0, noise_sigma, num_pts)
    ans = locate_fixed_event_position(m + noise, left, right, evt)
    print(ans)

    noise = np.random.normal(0.0, noise_sigma, num_pts)
    ans = locate_fixed_event_position(m + noise, left, right, evt)
    print(ans)

    noise = np.random.normal(0.0, noise_sigma, num_pts)
    ans = locate_fixed_event_position(m + noise, left, right, evt)
    print(ans)

    print(m)

    # ans = locate_fixed_event_position(m, 0, 29, 10)
    # print(ans)

    # ans = locate_fixed_event_position(m, 0, 29, 11)
    # locate_fixed_event_position.inspect_types()
    # print(ans)

    # ans = calc_metric_numpy(m)
    # calc_metric_numpy.inspect_types()
    # print(ans)


# We perform the following calls to force the njit of the functions.  This hides the
# compile time from the user (extends the load time a bit) and thus eliminates
# the slightly disconcerting 1 or 2 second delay before this functions start to
# operate if we wait until the user first invokes them after starting ppyote.  We
# do this as a work-around to the pickle problem that keeps normal caching from working.

_ = find_best_event_from_min_max_size(np.zeros(1), 0, 0, 0, 0)
_ = locate_event_from_d_and_r_ranges(np.zeros(1), 0, 0, 0, 0, 0, 0)


if __name__ == "__main__":
    bob()
    # bob.inspect_types()
    pass
