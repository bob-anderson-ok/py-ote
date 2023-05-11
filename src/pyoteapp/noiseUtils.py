#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 20:11:13 2017

@author: bob
"""

import numpy as np
from scipy.signal import savgol_filter as savgol


def laggedCoef(x, lag):
    if lag == 0:
        return 1.0
    corMatrix = np.corrcoef(x[lag:], x[:-lag])
    return corMatrix[0, 1]


def savgolTrendLine(y, window=101, degree=3):

    if window > len(y):
        window = len(y)

    # savgol requires an odd number for the window --- enforce that here
    if window % 2 == 0:
        window -= 1
        
    stage1trend = savgol(np.array(y), window, degree)
    stage2trend = savgol(stage1trend, window, degree)

    return stage2trend  # This is a numpy.ndarray


def polyTrendLine(x, y, degree=3):
    poly = np.polyfit(x, y, degree)
    return np.polyval(poly, x)


def getCorCoefs(x, y):
    combo = list(zip(x, y))
    combo.sort()
    yvalsConcat = np.array([item[1] for item in combo])

    # 5.2.4 We can't calculate sigmaA from just two points - we return None and this causes
    # sigmaA = sigmaB to be executed when the caller finds no sigmaA was returned
    if len(yvalsConcat) < 3:
        return np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), len(x), None

    try:
        trend = savgolTrendLine(yvalsConcat, window=301, degree=1)
        residuals = yvalsConcat - trend
    except ValueError:
        return np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), len(x), 0.0

    # We only compute correlation coefficients if there at least 14 data points
    if len(residuals) < 14:
        ans = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        ans = []
        for lag in range(11):
            ans.append(laggedCoef(residuals, lag))

    # This deals with no-noise test sets.
    sigma = np.std(np.diff(np.array(y))) / np.sqrt(2.0)  # In place of: np.std(residuals)
    if sigma == 0:
        ans = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return np.array(ans), len(x), sigma
