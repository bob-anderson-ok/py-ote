"""
Provides tests for the iterative_logl_functions module.

To make the import statements work, before calling pytest at a command line,
you must create an entry in the PYTHONPATH environment variable.  With the
current directory structure in pyote-repo,
the entry would be:

  export PYTHONPATH=src/pyoteapp

If PYTHONPATH already has entries, you should do:

  export PYTHONPATH=$PYTHONPATH:src/pyoteapp

Then do:

    pytest -v
"""

from iterative_logl_functions import add_entry
from iterative_logl_functions import sub_entry
from iterative_logl_functions import calc_metric_numpy
from iterative_logl_functions import calc_metric_iteratively
from iterative_logl_functions import calc_logl_from_metric
from iterative_logl_functions import cum_loglikelihood
from iterative_logl_functions import locate_fixed_event_position
from iterative_logl_functions import find_best_event_from_min_max_size
from iterative_logl_functions import locate_event_from_d_and_r_ranges
from iterative_logl_functions import find_best_r_only_from_min_max_size
from iterative_logl_functions import find_best_d_only_from_min_max_size

import numpy as np
import pytest
from math import isclose


def test_add_entry():
    assert (add_entry(ynew=1.0, s=0.0, s2=0.0, n=0, calc_var=True) ==
            (1.0, 1.0, 1, 0.0))


def test_sub_entry():
    assert (sub_entry(ynew=1.0, s=2.0, s2=2.0, n=2, calc_var=True) ==
            (1.0, 1.0, 1, 0.0))


def test_numpy_versus_iterative_algorithm():
    numpts = 10000
    a = np.random.randn(numpts)  # Generate numpts random normal numbers
    sn, sn2, n_numpy, var_numpy = calc_metric_numpy(a)
    si, si2, n_itera, var_itera = calc_metric_iteratively(a)
    assert n_numpy == n_itera
    assert isclose(si, sn, rel_tol=1e-12)
    assert isclose(si2, sn2, rel_tol=1e-12)
    assert isclose(var_numpy, var_itera, rel_tol=1e-12)


def test_calc_logl_from_metric():
    numpts = 10000
    a = np.random.randn(numpts)  # Generate 10,000 random normal numbers

    s, s2, n, var = calc_metric_iteratively(a)

    mean = np.repeat([s / n], numpts)
    sigma = np.repeat(np.sqrt(var), numpts)

    logl_iter, sigma2_iter = calc_logl_from_metric(s=s, s2=s2, n=n)
    logl_numpy, sigma2_numpy = cum_loglikelihood(y=a, m=mean, sigma=sigma)
    assert isclose(logl_iter, logl_numpy, rel_tol=1e-12)
    assert isclose(sigma2_iter, sigma2_numpy, rel_tol=1e-12)


def test_locate_fixed_event_position():
    y = np.repeat([10.0], 1000)
    y[500:600] = 9.0
    d, r, b, a, sigma_b, sigma_a, metric, sol_count = \
        locate_fixed_event_position(y, 0, 999, 100)
    assert d == 499
    assert r == 600
    assert b == 10.0
    assert a == 9.0
    assert sigma_b == pytest.approx(0.0)
    assert sigma_a == pytest.approx(0.0)

    np.random.seed(1)
    y = np.random.normal(10.0, 3.0, 1000)
    y[500:600] -= 5
    d, r, b, a, sigma_b, sigma_a, metric, sol_count = \
        locate_fixed_event_position(y, 0, 999, 100)
    assert d == 497
    assert r == 598
    assert isclose(b, 10.097998878345342)
    assert isclose(a, 5.4154415982851232)
    assert isclose(sigma_b, 2.912670678784627)
    assert isclose(sigma_a, 3.135871168854765)
    assert isclose(metric, -2148.6318771781766)


def test_find_best_event_from_min_max_size():

    d = None
    r = None
    b = None
    a = None
    sigma_b = None
    sigma_a = None
    metric = None

    y = np.repeat([10.0], 1000)
    y[500:602] = 5.0
    y[500] = 6.0
    y[601] = 7.0

    solverGen = find_best_event_from_min_max_size(y, 0, 999, 80, 110)
    for item in solverGen:
        if item[0] == 'fractionDone' or item[0] == 'event not found':
            pass
        else:
            d, r, b, a, sigma_b, sigma_a, metric = item

    assert d == 500
    assert r == 601
    assert isclose(b, 10.0)
    assert isclose(a, 5.0)
    assert sigma_b == pytest.approx(0.0)
    assert sigma_a == pytest.approx(0.0)
    assert isclose(metric, 706979.6256951995)

    # This is the light curve from 20171019_Gimple_MeyerCSVnew_fields.csv
    y_list = [
        234.369, 236.659, 256.550, 222.844, 260.800, 237.981, 265.813, 270.387,
        238.722, 247.669,
        232.009, 241.512, 251.753, 232.209, 248.812, 263.184, 274.191, 250.641,
        275.016, 243.750,
        271.359, 248.431, 231.219, 282.344, 189.022, 16.5219, 20.1562, 19.0281,
        196.997, 255.822,
        269.656, 266.569, 227.109, 232.247, 236.169, 242.925, 268.988, 214.419,
        242.812, 238.772,
        254.209, 255.647, 238.491, 232.959, 222.750, 210.197, 226.834, 259.947,
        239.297, 240.012,
        215.669, 273.975]

    y = np.array(y_list)

    solverGen = find_best_event_from_min_max_size(y, 0, 51, 1, 10)
    for item in solverGen:
        if item[0] == 'fractionDone' or item[0] == 'event not found':
            pass
        else:
            d, r, b, a, sigma_b, sigma_a, metric = item

    assert d == 24
    assert r == 28
    assert isclose(b, 246.46168085106385,)
    assert isclose(a, 18.568733333333267)
    assert isclose(sigma_b, 17.657750715901955)
    assert isclose(sigma_a, 1.5188367090842694)
    assert isclose(metric, -272.39810149042796)

    solverGen = find_best_event_from_min_max_size(y, 10, 41, 1, 10)
    for item in solverGen:
        if item[0] == 'fractionDone' or item[0] == 'event not found':
            pass
        else:
            d, r, b, a, sigma_b, sigma_a, metric = item

    assert d == 25
    assert r == 27
    assert isclose(b, 246.13079310344824)
    assert isclose(a, 20.156200000000101)
    assert isclose(sigma_b, 21.751916084256006)
    assert isclose(sigma_a, 3.5683225505580343e-06)
    assert isclose(metric, -153.5358773771935)


def test_locate_event_from_d_and_r_ranges():
    d = None
    r = None
    b = None
    a = None
    sigma_b = None
    sigma_a = None
    metric = None

    y = np.repeat([10.0], 1000)
    y[500:602] = 5.0
    y[500] = 6.0
    y[601] = 7.0

    # d, r, b, a, sigma_b, sigma_a, metric = \
    #     locate_event_from_d_and_r_ranges(y, 0, 999, 1, 520, 522, 999)
    solverGen = locate_event_from_d_and_r_ranges(y, 0, 999, 1, 520, 522, 999)
    for item in solverGen:
        if item[0] == 'fractionDone' or item[0] == 'event not found':
            pass
        else:
            d, r, b, a, sigma_b, sigma_a, metric = item

    assert d == 500
    assert r == 601
    assert isclose(b, 10.0)
    assert isclose(a, 5.0)
    assert sigma_b == pytest.approx(0.0)
    assert sigma_a == pytest.approx(0.0)
    assert isclose(metric, 706979.6256951995)

    # d, r, b, a, sigma_b, sigma_a, metric = \
    #     locate_event_from_d_and_r_ranges(y, 10, 990, 10, 520, 522, 989)
    solverGen = locate_event_from_d_and_r_ranges(y, 10, 990, 10, 520, 522, 989)
    for item in solverGen:
        if item[0] == 'fractionDone' or item[0] == 'event not found':
            pass
        else:
            d, r, b, a, sigma_b, sigma_a, metric = item

    assert d == 500
    assert r == 601
    assert isclose(b, 10.0)
    assert isclose(a, 5.0)
    assert sigma_b == pytest.approx(0.0)
    assert sigma_a == pytest.approx(0.0)
    assert isclose(metric, 693520.0937430866)


def test_find_best_r_only_from_min_max_size():
    y = np.repeat([10.0], 1000)
    y[:500] -= 5.0
    y[500] = 6.0

    # Test with left=10 and right=900 (trimmed light curve)
    # d, r, b, a, sigma_b, sigma_a, metric = \
    #     find_best_r_only_from_min_max_size(
    #         y, 10, 900, 400, 600)

    solverGen = find_best_r_only_from_min_max_size( y, 10, 900, 400, 600)
    d, r, b, a, sigma_b, sigma_a, metric = next(solverGen)

    assert d is None
    assert r == 500
    assert b == 10.0
    assert a == 5.0
    assert sigma_b == pytest.approx(0.0)
    assert sigma_a == pytest.approx(0.0)
    assert isclose(metric, 630472.812493715)

    # Test with no trimming and smallest event size in the mix
    solverGen = find_best_r_only_from_min_max_size(y, 0, 999, 1, 600)
    d, r, b, a, sigma_b, sigma_a, metric = next(solverGen)

    assert d is None
    assert r == 500
    assert b == 10.0
    assert a == 5.0
    assert sigma_b == pytest.approx(0.0)
    assert sigma_a == pytest.approx(0.0)
    assert isclose(metric, 707688.0221137318)

    # Test an edge case
    y = np.repeat([10.0], 1000)
    y[0] = 5.0
    y[1] = 6.0

    solverGen = find_best_r_only_from_min_max_size(y, 0, 999, 1, 600)
    d, r, b, a, sigma_b, sigma_a, metric = next(solverGen)

    assert d is None
    assert r == 1
    assert b == 10.0
    assert a == 5.0
    assert sigma_b == pytest.approx(0.0)
    assert sigma_a == pytest.approx(0.0)
    assert isclose(metric, 707688.0221137318)

    # Test another edge case
    np.random.seed(10)
    y = np.random.normal(size=1000)

    y[999] += 9.0
    y[998] += 6.0

    solverGen = find_best_r_only_from_min_max_size(y, 0, 999, 800, 998)
    d, r, b, a, sigma_b, sigma_a, metric = next(solverGen)

    assert d is None
    assert r == 998
    assert b == 9.6424506244340176
    assert a == -0.014838545466908208
    assert isclose(sigma_b, 0.9385913928208224)
    assert isclose(sigma_a, 0.9385913928208224)
    assert isclose(metric, 126.62334169871188,)


def test_find_best_d_only_from_min_max_size():

    # Test edge case bdaaaaaa.....
    y = np.repeat([0.0], 1000)
    y[0] = 10.0
    y[1] = 6.0

    solverGen = find_best_d_only_from_min_max_size(y, 0, 999, 1, 998)
    d, r, b, a, sigma_b, sigma_a, metric = next(solverGen)

    assert d == 1
    assert r is None
    assert b == 10.0
    assert a == 0.0
    assert sigma_b == pytest.approx(0.0)
    assert sigma_a == pytest.approx(0.0)
    assert isclose(metric, 707688.0221137318)

    # Test edge case bbb.....da
    y = np.repeat([10.0], 1000)
    y[998] = 5.0
    y[999] = 0.0

    solverGen = find_best_d_only_from_min_max_size(y, 0, 999, 1, 998)
    d, r, b, a, sigma_b, sigma_a, metric = next(solverGen)

    assert d == 998
    assert r is None
    assert b == 10.0
    assert a == 0.0
    assert sigma_b == pytest.approx(0.0)
    assert sigma_a == pytest.approx(0.0)
    assert isclose(metric, 707688.0221137318)

    np.random.seed(10)
    y = np.random.normal(loc=10.0, size=100)
    y[99] = 0.0
    # y[98] = 0.0

    solverGen = find_best_d_only_from_min_max_size(y, 0, 99, 1, 98)
    d, r, b, a, sigma_b, sigma_a, metric = next(solverGen)

    assert d == 98
    assert r is None
    assert b == 10.092542650928841
    assert a == pytest.approx(0.0)
    assert isclose(sigma_b, 0.9519255746939)
    assert isclose(sigma_a, 1.6858739404357607e-06)
    assert isclose(metric, 36.243064256724104)
