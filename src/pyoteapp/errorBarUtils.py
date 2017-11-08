#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 09:37:59 2017

@author: Bob Anderson
"""

from typing import List, Iterator, Union, Tuple
from scipy.signal import savgol_filter as savgol
from pyoteapp.solverUtils import model
from pyoteapp import autocorrtools
import numpy as np

import pyximport
pyximport.install()
from pyoteapp.c_functions import find_Dedge_logl  # Finds D using a subframe model


def edgeDistributionGenerator(*, ntrials: int=10000, numPts: int=None, D: int=None, acfcoeffs: List[float]=None,
                              B: float=None, A: float=None, sigmaB: float=None, sigmaA: float=None
                              ) -> Iterator[Union[float, List[float]]]:
    my_noise_gen = None
    try:
        my_noise_gen = autocorrtools.CorrelatedNoiseGenerator(acfcoeffs)
    except np.linalg.LinAlgError:
        yield -1.0  # This is a flag value that is only returned when a LinAlgError exception occurs

    mb = np.ndarray(shape=(numPts,), dtype=np.double)  # 'model' B value
    ma = np.ndarray(shape=(numPts,), dtype=np.double)  # 'model' A value
    mm = np.ndarray(shape=(numPts,), dtype=np.double)  # 'model' intermediate value

    edgePos = np.zeros(shape=ntrials, dtype=np.float)

    m, sigma = model(B=B, A=A, edgeTuple=(D, None), sigmaB=sigmaB, sigmaA=sigmaA, numPts=numPts)
    # m[:D] == B  m[D:] == A  sigma[:D] == sigmaB  sigma[:D] == sigmaA

    for i in range(ntrials):
        if i % 2000 == 0:
            yield i / ntrials
        noise = my_noise_gen.corr_normal(numPts)    
        y = m + noise * sigma
        
        centerValue = np.random.uniform(A, B)
        spanFrac = (B - centerValue) / (B - A)
        noise = sigmaB - spanFrac * (sigmaB - sigmaA)
        y[D] = np.random.normal(centerValue, noise)
        deltaPos = 1 - spanFrac
        edge_pos = find_Dedge_logl(numPts, y, mb, ma, mm, B, A, sigmaB, sigmaA)

        edgePos[i] = edge_pos - deltaPos
        
    yield edgePos


def ciBars(*, dist: np.ndarray = None, ci: float = None) -> Tuple[float, ...]:
    assert(ci >= 0)
    
    ecdfX = np.sort(dist)
    ecdfY = np.arange(ecdfX.size) / ecdfX.size
    
    indices = np.where(ecdfY >= 0.5 + ci / 2)[0]
    if indices.size == 0:
        hiBar = ecdfX[-1]
    else:
        hiBar = ecdfX[np.where(ecdfY >= 0.5 + ci / 2)[0][0]]
     
    indices = np.where(ecdfY >= 0.5 - ci / 2)[0]
    if indices.size == 0:
        loBar = 0.0
    else:
        loBar = ecdfX[np.where(ecdfY >= 0.5 - ci / 2)[0][0]]
    
    midBar = ecdfX[np.where(ecdfY >= 0.5)[0][0]]
    
    return loBar, midBar, hiBar, loBar - midBar, hiBar - midBar


def sampledCdf(dist: np.ndarray, nsamples: int) -> Tuple[np.ndarray, np.ndarray]:
    sdist = np.sort(dist)
    
    # spt stands for 'sample point'
    spt = sdist[0]
    
    delta = (sdist[-1] - sdist[0]) / nsamples
    
    y = []
    x = []

    i = 0
    while spt <= sdist[-1]:
        x.append(spt)
        y.append(i/sdist.size)
        spt += delta
        while spt > sdist[i]:
            i += 1
            if i == sdist.size:
                break
           
    return np.array(x), np.array(y)


def smoothedSolutionDistribution(dist: np.ndarray, nsamples: int) -> Tuple[np.ndarray, np.ndarray]:
    sampledX, sampledY = sampledCdf(dist, nsamples)
    smoothedY = savgol(sampledY, 21, 3)
    
    return np.array(sampledX[:-1]), np.diff(smoothedY)


def createDurDistribution(dist: np.ndarray) -> np.ndarray:
    maxIndex = dist.size - 1
    
    def sample():
        return dist[np.random.randint(0, maxIndex)]
    
    durDist = np.ndarray(shape=(dist.size,))
    
    for i in range(dist.size):
        durDist[i] = sample() - sample()
        
    return durDist
