""" This module provides several functions for calculating likelihood values.

    A 'likelihood' is an un-normalized probability, calculated from the pdf of a
    distribution.  In this module, the pdf that is used is the Gaussian pdf.

    The main function is cum_corr_loglikelihood() in that it can stand in for any
    of the other functions, in particular by setting the correlation coefficient
    to zero and possibly giving only a single point.
"""

# coding: utf-8

from math import log, exp, sqrt, pi
import numpy as np

__all__ = ['likelihood', 'loglikelihood', 'conditional_likelihood',
           'cum_loglikelihood', 'cum_corr_loglikelihood', 'aicc']


def aicc(logLikelihood, n, k):
    """
    Akaike information criterion corrected for small sample size
    n = number of samples
    k = number of degrees of freedom
    """
    assert(n > k + 2)
    return -2 * logLikelihood + 2 * k * n / (n - k - 1)


def likelihood(y, m, sigma):
    """ calculates likelihood given Gaussian statistics

    Args:
        y (float):     measured value
        m (float):     mean (expected model value)
        sigma (float): stdev of measurements

    Returns:
        un-normalized probability based on Gaussian distribution
    """

    t1 = 1 / sqrt(2*pi*sigma**2)
    t2 = -(y - m) ** 2
    t3 = t2 / (2*sigma**2)
    return t1 * exp(t3)


def loglikelihood(y, m, sigma):
    """ calculate ln(likelihood) given Gaussian statistics

    Args:
        y (float):     measured value
        m (float):     mean (expected model value)
        sigma (float): stdev of measurements

    Returns:
        natural logarithm of un-normalized probability based on Gaussian distribution

    """
    # log(x) is natural log (base e)
    t1 = -log(sqrt(2*pi))
    t2 = -log(sigma)     
    t3 = -(y - m) ** 2 / (2 * sigma ** 2)
    return t1 + t2 + t3


def logLikelihoodLine(y,  sigmaB=None, left=None, right=None):
    """ log likelihood of a straight line through the readings"""
    
    B = np.mean(y[left:right+1])
    n = right - left + 1
    
    ans = -n * np.log(np.sqrt(2*pi))
    ans -= n * np.log(sigmaB)
    ans -= np.sum((y[left:right+1] - B) ** 2 / sigmaB ** 2) / 2.0
    
    return ans


def cum_loglikelihood(y, m, sigma, left, right):
    """ numpy accelerated sum of loglikelihoods

    ARGS:
        y (ndarray):     measured values
        m (ndarray):     associated mean values (the 'model')
        sigma (ndarray): associated stdev values
        left             index of first y to include
        right            index of last  y to include
    """

    assert(len(y) == len(m) == len(sigma))

    n = right - left + 1

    ans = -n * np.log(np.sqrt(2*pi))

    ans -= np.sum(np.log(sigma[left:right+1]))

    ans -= np.sum((y[left:right+1] - m[left:right+1]) ** 2 / sigma[left:right+1] ** 2) / 2.0

    return ans


def conditional_likelihood(rho, y1, m1, sigma1, y0, m0, sigma0):
    """ Computes the conditional likelihood p(y1|y0) which should be read as:
        ...the probability of y1, given y0, when the values are
        partially correlated.

        All arguments are standard floats

        Args:
            rho:    correlation coefficient (0 <= rho < 1)
            y1:     measured value at position 1
            y0:     measured value at position 0
            m1:     model value at position 1
            m0:     model value at position 0
            sigma1: noise at position 1
            sigma0: noise at position 0
    """

    if rho < 0.0 or rho >= 1.0:
        raise Exception("rho must be non-negative and less than 1")

    if sigma1 <= 0.0 or sigma0 <= 0:
        raise Exception("sigma values must be greater than zero")

    t1 = (y1 - m1)**2 / sigma1**2
    t0 = (y0 - m0)**2 / sigma0**2
    t2 = 2.0 * rho * (y1 - m1) * (y0 - m0) / (sigma1 * sigma0)

    ctop = 1.0 / (2 * pi * sigma1 * sigma0 * sqrt(1 - rho**2))
    cbot = 1.0 / (sigma0 * sqrt(2*pi))

    numerator = ctop * exp(-(1.0/(2.0 * (1 - rho**2))) * (t1 - t2 + t0))
    denominator = cbot * exp(-(1.0/2.0) * t0)

    return numerator / denominator


# noinspection PyTypeChecker
def cum_corr_loglikelihood(rho, y, m, sigma):
    """ calculates the sum of correlated loglikelihoods of a measurement array
        using numpy acceleration

        Args:
            rho (float):     average nearest neighbor correlation coeffiecient
            y (ndarray):     measurements
            m (ndarray):     means (model values)
            sigma (ndarray): stdev associated with each y

        Returns:
            sum of natural logarithms of nearest neighbor correlated measurements
            assuming measurements have Gaussian distributions
    """
    assert (len(y) == len(m) == len(sigma))
    n = len(y)
    z = np.ndarray(shape=[n])
    p = np.ndarray(shape=[n])
    alpha = np.ndarray(shape=[n])
    z[0] = y[0]
    z[1:] = y[:-1]
    p[0] = m[0]
    p[1:] = m[:-1]
    alpha[0] = sigma[0]
    alpha[1:] = sigma[:-1]
    term0 = -n * log(sqrt(2 * pi * (1 - rho ** 2))) - np.sum(np.log(sigma))
    factor = 1 / (2 * (1 - rho ** 2))
    term1 = np.sum(((y - m) ** 2) / sigma ** 2)
    term2 = 2 * rho * np.sum(((y - m) * (z - p)) / (sigma * alpha))
    term3 = rho * rho * np.sum(((z - p) ** 2) / alpha ** 2)
    return term0 - factor * (term1 - term2 + term3)
