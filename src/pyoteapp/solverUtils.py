#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 07:00:33 2017

@author: Bob Anderson
"""
MIN_SIGMA = 0.1
MAX_PLINE = 0.1

from math import exp
import numpy as np
from likelihood_calculations import cum_loglikelihood, aicc, logLikelihoodLine
from likelihood_calculations import loglikelihood
from numba import njit


def aicModelValue(*, obsValue=None, B=None, A=None, sigmaB=None, sigmaA=None):
    assert B >= A
    # assert sigmaA > 0.0
    # assert sigmaB > 0.0

    # This function determines if an observation point should categorized as a baseline (B)
    # point, an event (A) point, or a valid intermediate point using the Akaike Information Criterion
    # An intermediate point reflects a more complex model (higher dimension model)
    if obsValue >= B:
        return B  # Categorize as baseline point
    if obsValue <= A:
        return A  # Categorize as event point
    if B == A:
        return B

    # We do this to allow test files with zero noise to be processed
    if sigmaA == 0:
        if sigmaB == 0:
            return obsValue
        else:
            assert sigmaA > 0.0

    sigmaM = sigmaA + (sigmaB - sigmaA) * ((obsValue - A) / (B - A))
    loglB = loglikelihood(obsValue, B, sigmaB)
    loglM = loglikelihood(B, B, sigmaM) - 1.0  # The -1 is the aic model complexity 'penalty'
    loglA = loglikelihood(obsValue, A, sigmaA)

    if loglM > loglA and loglM > loglB:
        return obsValue  # Categorize as valid intermediate value
    elif loglB > loglA:
        return B  # Categorize as baseline point
    else:
        return A  # Categorize as event point


@njit(cache=True)
def model(B=0.0, A=0.0,
          D=-1, R=-1,
          sigmaB=0.0, sigmaA=0.0, numPts=0):
    
    # D, R = edgeTuple
    assert(numPts > 0 and sigmaA >= 0 and sigmaB >= 0)
    # TODOnot Remove this experimental code and uncomment the line above
    # if sigmaA < 0:
    #     sigmaA = 0.0
    # if sigmaB < 0:
    #     sigmaB = 0.0
    # assert(numPts > 0)
    # # assert(sigmaA >= 0.0)
    # # assert(sigmaB >= 0.0)
    m = np.zeros(numPts)
    sigma = np.zeros(numPts)

    if D == -1 and R == -1:  # -1 means None
        m[:] = B
        sigma[:] = sigmaB
        
        return m, sigma
    
    if not D == -1 and not R == -1:  # D and R are not None
        assert((D >= 0) and (R > D))
        assert((D < numPts) and (R < numPts))
        assert(R > D)
        
        m[:D] = B
        m[D:R] = A
        m[R:] = B
        
        sigma[:D] = sigmaB
        sigma[D:R] = sigmaA
        sigma[R:] = sigmaB
        
        return m, sigma
    
    if not D == -1 and R == -1:  # R is None
        assert((D >= 0) and (D < numPts))
        
        m[:D] = B
        m[D:] = A
        
        sigma[:D] = sigmaB
        sigma[D:] = sigmaA
        
        return m, sigma
    
    if not R == -1 and D == -1:  # D is None
        assert((R >= 1) and (R < numPts))
        
        m[:R] = A
        m[R:] = B
        
        sigma[:R] = sigmaA
        sigma[R:] = sigmaB
        
        return m, sigma
            
    raise Exception("Unexpected condition.")
    

def candidatesFromEventSize(*, eventType='DandR',
                            left=None, right=None, 
                            minSize=None, maxSize=None):
    """
    This is implemented as a generator because if numPts is large, it is
    easy to get a candidate list in the many millions, so it's best to
    create the candidates one at a time.
    """

    assert(right > left)
    assert(minSize >= 1)
    assert(maxSize >= minSize)
    assert(maxSize < right - left)
    
    if eventType == 'Donly':
        for D in range(right-maxSize+1, right-minSize+2):
            yield D, None
    
    elif eventType == 'Ronly':
        for R in range(left+minSize, left+maxSize+1):
            yield None, R
    
    elif eventType == 'DandR':
        for size in range(minSize, maxSize+1):
            for pos in range(left+1, right+1-size):
                yield pos, pos+size
        
    else:
        raise Exception("Unrecognized event type")
        

def candidatesFromDandRlimits(*, eventType='DandR',
                              dLimits=None, rLimits=None):
    """
    This is implemented as a generator because if large limits are given, it is
    easy to get a very large candidate list, so it's best to
    create the candidates one at a time.
    """
    # The D and R limits are assumed valid as input and are non-overlapping
    
    if eventType == 'Donly':
        for D in range(dLimits[0], dLimits[1] + 1):
            yield D, None
    
    elif eventType == 'Ronly':
        for R in range(rLimits[0], rLimits[1] + 1):
            yield None, R
    
    elif eventType == 'DandR':
        for D in range(dLimits[0], dLimits[1] + 1):
            for R in range(rLimits[0], rLimits[1] + 1):
                yield D, R
        
    else:
        raise Exception("Unrecognized edvent type")
        

@njit(cache=True)
def calcNumCandidatesFromDandRlimits(eventType='DandR', d_start=-1, d_end=-1,
                                     r_start=-1, r_end=-1):
    
    # The D and R limits are assumed valid as input and are non-overlapping
    
    if eventType == 'Donly':
        return d_end - d_start + 1
    elif eventType == 'Ronly':
        return r_end - r_start + 1
    elif eventType == 'DandR':
        return (d_end - d_start + 1) * (r_end - r_start + 1)
    else:
        raise Exception("Unrecognized event type")


@njit(cache=True)
def calcNumCandidatesFromEventSize(eventType='DandR',
                                   left=None, right=None,
                                   minSize=None, maxSize=None):
    
    numPts = right - left + 1
    assert(numPts >= 0)
    assert(minSize >= 1)
    assert(maxSize >= minSize)
    assert(maxSize < numPts - 1)

    if eventType == 'DandR':
        c1 = maxSize - minSize + 1
        c2 = 2 * numPts - 2 - minSize - maxSize
        return int(c1 * c2 / 2)
    elif eventType == 'Donly':
        return maxSize - minSize + 1
    elif eventType == 'Ronly':
        return maxSize - minSize + 1
    else:
        raise Exception("Unrecognized edge specifier")


def newCalcBandA(*, yValues=None, tpList=None, left=None, right=None, cand=None):
    assert (right > left)

    sigmaA = 0.0

    def valuesToBeUsed(lowRange, highRange):
        valList = []
        for i in range(lowRange, highRange):
            if i not in tpList:
                valList.append(yValues[i])
        return valList

    D, R = cand  # Extract D and R from the tuple

    if R is None:
        assert (D >= left)
        # This is a 'Donly' candidate
        # Note that the yValue at D is not included in the B calculation
        # because that point is in the event bottom.
        valuesToUse = valuesToBeUsed(left, D)
        B = np.mean(valuesToUse)
        sigmaB = np.std(valuesToUse)
        # We have to deal with a D at the right edge.  There is no value to
        # the right to use to calculate A so we simply return the value at D
        # as the best estimate of A
        if D == right:
            A = yValues[D]
            sigmaA = 0.0
        else:
            # changed in 4.4.7
            # A = np.mean(yValues[D+1:right+1])
            valuesToUse = valuesToBeUsed(D, right + 1)
            A = np.mean(valuesToUse)
            sigmaA = np.std(valuesToUse)
        if A >= B:
            A = B * 0.999
        return B, A, sigmaB, sigmaA

    elif D is None:
        assert (R <= right)
        # This is an 'Ronly' candidate
        # We have to deal with a R at the right edge.  There is no value to
        # the right to use to calculate B so we simply return the value at R
        # as the best estimate of B
        if R == right:
            B = yValues[R]
            sigmaB = 0.0
        else:
            # changed in 4.4.7
            # B = np.mean(yValues[R+1:right+1])
            # Changed i to i + R in 5.2.3 to solve flash edge time problem
            valuesToUse = [val for i, val in enumerate(yValues[R:right + 1]) if i + R not in tpList]
            B = np.mean(valuesToUse)
            sigmaB = np.std(valuesToUse)
        # Changed to R - 1 in 5.2.3 to solve flash edge time problem
        valuesToUse = valuesToBeUsed(left, R - 1)
        A = np.mean(valuesToUse)
        sigmaA = np.std(valuesToUse)
        if A >= B:
            A = B * 0.999
        return B, A, sigmaB, sigmaA

    else:
        assert ((D >= left) and (R <= right) and (R > D))
        # We have a 'DandR' candidate
        leftBvals = valuesToBeUsed(left, D)

        if R == right:
            rightBvals = yValues[right]
        else:
            # changed in 4.4.8
            rightBvals = valuesToBeUsed(R, right + 1)
        B = (np.sum(leftBvals) + np.sum(rightBvals)) / (len(leftBvals) + len(rightBvals))
        sigmaB = np.std(leftBvals + rightBvals)

        if R - D == 1:  # Event size of 1 has no valid A --- we choose the value at D
            A = yValues[D]
        else:
            valuesToUse = valuesToBeUsed(D, R)
            A = np.mean(valuesToUse)
            sigmaA = np.std(valuesToUse)
        if A >= B:
            A = B * 0.999
        return B, A, sigmaB, sigmaA


def calcBandA(*, yValues=None, left=None, right=None, cand=None):
    
    assert(right > left)
    
    D, R = cand  # Extract D and R from the tuple
        
    if R is None:
        assert(D >= left)
        # This is a 'Donly' candidate
        # Note that the yValue at D is not included in the B calculation
        # because that point is in the event bottom.
        B = np.mean(yValues[left:D])
        # We have to deal with a D at the right edge.  There is no value to
        # the right to use to calculate A so we simply return the value at D
        # as the best estimate of A
        if D == right:
            A = yValues[D]
        else:
            # changed in 4.4.7
            # A = np.mean(yValues[D+1:right+1])
            A = np.mean(yValues[D:right+1])
        if A >= B:
            A = B * 0.999
        return B, A
    
    elif D is None:
        assert(R <= right)
        # This is an 'Ronly' candidate
        # We have to deal with a R at the right edge.  There is no value to
        # the right to use to calculate B so we simply return the value at R
        # as the best estimate of B
        if R == right:
            B = yValues[R]
        else:
            # changed in 4.4.7
            # B = np.mean(yValues[R+1:right+1])
            B = np.mean(yValues[R:right+1])
        A = np.mean(yValues[left:R])  # smallest R is left + 1
        if A >= B:
            A = B * 0.999
        return B, A
    
    else:
        assert((D >= left) and (R <= right) and (R > D))
        # We have a 'DandR' candidate
        leftBvals = yValues[left:D]  # Smallest D will = left
        if R == right:
            rightBvals = yValues[right]
        else:
            # changed in 4.4.8
            rightBvals = yValues[R:right+1]
        B = (np.sum(leftBvals) + np.sum(rightBvals)) / (leftBvals.size + rightBvals.size)

        if R - D == 1:  # Event size of 1 has no valid A --- we choose the value at D
            A = yValues[D]
        else:
            A = np.mean(yValues[D:R])
        if A >= B:
            A = B * 0.999
        return B, A


# def scoreCandidate(yValues, left, right, cand, sigmaB, sigmaA):
#     B, A = calcBandA(yValues=yValues, left=left, right=right, cand=cand)
#     m, sigma = model(B=B, A=A, D=cand[0], R=cand[1],
#                      sigmaB=sigmaB, sigmaA=sigmaA, numPts=yValues.size)
#     return cum_loglikelihood(yValues, m, sigma, left, right), B, A


def scoreSubFrame(yValues, left, right, cand, sigmaB, sigmaA):
    B, A = calcBandA(yValues=yValues, left=left, right=right, cand=cand)
    m, sigma = model(B=B, A=A, D=cand[0], R=cand[1],
                     sigmaB=sigmaB, sigmaA=sigmaA, numPts=yValues.size)
    D, R = cand
    if D is not None:
        if (yValues[D] < B) and (yValues[D] > A):
            m[D] = yValues[D]
    if R is not None:
        if (yValues[R] < B) and (yValues[R] > A):
            m[R] = yValues[R]
    return cum_loglikelihood(yValues, m, sigma, left, right), B, A
    

def candidateCounter(*, eventType='DandR',
                     dLimits=None, rLimits=None, 
                     left=None, right=None,
                     numPts=None, minSize=None, maxSize=None):

    def minMaxOk():
        if numPts is None:
            return False
        if minSize is None:
            return False
        if maxSize is None:
            return False
        if minSize < 1:
            return False
        if maxSize > (numPts - 2):
            return False
        if maxSize < minSize:
            return False
        return True
    
    # D and R limits trumps event size as candidate generator/counter
    if eventType == 'Donly':
        if dLimits:
            return ('usedLimits', calcNumCandidatesFromDandRlimits(eventType=eventType,
                    d_start=dLimits[0], d_end=dLimits[1], r_start=-1, r_end=-1))
        else:
            if minMaxOk():
                return ('usedSize', calcNumCandidatesFromEventSize(eventType=eventType, 
                        left=left, right=right, minSize=minSize, maxSize=maxSize))
            else:
                return 'error', -1
            
    elif eventType == 'Ronly':
        if rLimits:
            return ('usedLimits', calcNumCandidatesFromDandRlimits(eventType=eventType,
                    d_start=-1, d_end=-1,  r_start=rLimits[0], r_end=rLimits[1]))
        else:
            if minMaxOk():
                return ('usedSize', calcNumCandidatesFromEventSize(eventType=eventType, 
                        left=left, right=right, minSize=minSize, maxSize=maxSize))
            else:
                return 'error', -1
        
    elif eventType == 'DandR':
        if rLimits and dLimits:
            return ('usedLimits', calcNumCandidatesFromDandRlimits(eventType=eventType,
                    d_start=dLimits[0], d_end=dLimits[1], r_start=rLimits[0], r_end=rLimits[1]))
        else:
            if minMaxOk():
                return ('usedSize', calcNumCandidatesFromEventSize(eventType=eventType, 
                        left=left, right=right, minSize=minSize, maxSize=maxSize))
            else:
                return 'error', -1
            
    else:
        raise Exception("Unrecognized event type")


def subFrameAdjusted(*, eventType=None, cand=None, B=None, A=None,
                     sigmaA=None, sigmaB=None,
                     yValues=None, left=None, right=None):

    def adjustR():
        value = yValues[R]
        adj = (B - value) / (B - A)
        return R + adj

    def adjustD():
        value = yValues[D]
        adj = (value - A) / (B - A)
        return D + adj

    transitionPoints = []

    D, R = cand
    adjD = D
    adjR = R

    # Here we add code so we can analyze light curves that may have sigmaB or
    #  sigmaA values of zero.  This happens when testing with artificial data
    #  but can also result from real light curves that may be clipped so that
    #  all B pixels have a constant value.  Limovie can produce a sigmaA=0
    # when a rectangular aperture is in use as well

    if eventType == 'Donly':
        if aicModelValue(obsValue=yValues[D], B=B, A=A, sigmaB=sigmaB, sigmaA=sigmaA) == yValues[D]:
            # If point at D categorizes as M (valid mid-point), do sub-frame
            # adjustment and exit
            adjD = adjustD()
            transitionPoints.append(D)
            B, A, sigmaB, sigmaA = newCalcBandA(yValues=yValues, tpList=transitionPoints,
                                                left=left, right=right, cand=(D, R))
        elif aicModelValue(obsValue=yValues[D], B=B, A=A, sigmaB=sigmaB, sigmaA=sigmaA) == B:
            # else if point at D categorizes as B, set D to D+1 and recalculate B and A
            D = D + 1
            adjD = D
            B, A, sigmaB, sigmaA = newCalcBandA(yValues=yValues, tpList=transitionPoints,
                                                left=left, right=right, cand=(D, R))
            # It's possible that this new point qualifies as M --- so we check:
            if aicModelValue(
                    obsValue=yValues[D], B=B, A=A, sigmaB=sigmaB,
                    sigmaA=sigmaA) == yValues[D]:
                adjD = adjustD()
                transitionPoints.append(D)
                B, A, sigmaB, sigmaA = newCalcBandA(yValues=yValues, tpList=transitionPoints,
                                                    left=left, right=right, cand=(D, R))
        # else (point at D categorizes as A) --- nothing to do
        return [adjD, adjR], B, A, sigmaB, sigmaA

    elif eventType == 'Ronly':
        if aicModelValue(obsValue=yValues[R], B=B, A=A, sigmaB=sigmaB, sigmaA=sigmaA) == yValues[R]:
            # If point at R categorizes as M, do sub-frame adjustment
            adjR = adjustR()
            transitionPoints.append(R)
            B, A, sigmaB, sigmaA = newCalcBandA(yValues=yValues, tpList=transitionPoints,
                                                left=left, right=right, cand=(D, R))
        elif aicModelValue(obsValue=yValues[R], B=B, A=A, sigmaB=sigmaB, sigmaA=sigmaA) == A:
            # else if point at R categorizes as A, set R to R + 1 and recalculate B and A
            R = R + 1
            adjR = R
            B, A, sigmaB, sigmaA = newCalcBandA(yValues=yValues, tpList=transitionPoints,
                                                left=left, right=right, cand=(D, R))
            # It's possible that this new point qualifies as M --- so we check
            if aicModelValue(
                    obsValue=yValues[R], B=B, A=A, sigmaB=sigmaB,
                    sigmaA=sigmaA) == yValues[R]:
                adjR = adjustR()
                transitionPoints.append(R)
                B, A, sigmaB, sigmaA = newCalcBandA(yValues=yValues, tpList=transitionPoints,
                                                    left=left, right=right, cand=(D, R))
        elif aicModelValue(obsValue=yValues[R - 1], B=B, A=A, sigmaB=sigmaB,
                           sigmaA=sigmaA) == yValues[R - 1]:
            # The point at R categorizes as B, and we have found
            # that the point at R-1 categorizes as M, so set R to R-1 and
            # recalculate B and A
            R = R - 1
            adjR = adjustR()
            transitionPoints.append(R)
            B, A, sigmaB, sigmaA = newCalcBandA(yValues=yValues, tpList=transitionPoints,
                                                left=left, right=right, cand=(D, R))
        return [adjD, adjR], B, A, sigmaB, sigmaA

    elif eventType == 'DandR':
        if not R - D > 2:
            return [D, R], B, A, sigmaB, sigmaA
        if aicModelValue(
                obsValue=yValues[D], B=B, A=A, sigmaB=sigmaB, sigmaA=sigmaA) == yValues[D]:
            # The point at D categorizes as M, do sub-frame adjustment; this
            # (finishes D)
            adjD = adjustD()
            transitionPoints.append(D)
        elif aicModelValue(obsValue=yValues[D], B=B, A=A, sigmaB=sigmaB, sigmaA=sigmaA) == B:
            # The point at D categorizes as B, set D to D+1 and recalculate B and A
            D = D + 1
            adjD = D
            # It's possible that this new point qualifies as M --- so we check
            if aicModelValue(
                    obsValue=yValues[D], B=B, A=A, sigmaB=sigmaB,
                    sigmaA=sigmaA) == yValues[D]:
                adjD = adjustD()
                transitionPoints.append(D)
        elif aicModelValue(obsValue=yValues[D-1], B=B, A=A, sigmaB=sigmaB,
                           sigmaA=sigmaA) == yValues[D-1]:
            # The point at D categorizes as A, and we have found
            # that the point at D-1 categorizes as M, so set D to D-1 and
            # recalculate B and A
            D = D - 1
            adjD = adjustD()
            transitionPoints.append(D)

        if aicModelValue(
                obsValue=yValues[R], B=B, A=A, sigmaB=sigmaB, sigmaA=sigmaA) == yValues[R]:
            # The point at R categorizes as M, do sub-frame adjustment
            adjR = adjustR()
            transitionPoints.append(R)
        elif aicModelValue(obsValue=yValues[R], B=B, A=A, sigmaB=sigmaB, sigmaA=sigmaA) == A:
            # The point at R categorizes as A, set R to R + 1 and recalculate B and A
            R = R + 1
            adjR = R
            # It's possible that this new point qualifies as M --- so we check
            if aicModelValue(
                    obsValue=yValues[R], B=B, A=A, sigmaB=sigmaB,
                    sigmaA=sigmaA) == yValues[R]:
                adjR = adjustR()
                transitionPoints.append(R)
        elif aicModelValue(obsValue=yValues[R - 1], B=B, A=A, sigmaB=sigmaB,
                           sigmaA=sigmaA) == yValues[R - 1]:
            # The point at R categorizes as B, and we have found
            # that the point at R-1 categorizes as M, so set R to R-1 and
            # recalculate B and A
            R = R - 1
            adjR = adjustR()
            transitionPoints.append(R)
        return [adjD, adjR], B, A, sigmaB, sigmaA

    else:
        raise Exception('Unrecognized event type')


def solver(*, eventType=None, yValues=None,
           left=None, right=None,
           sigmaB=None, sigmaA=None, 
           dLimits=None, rLimits=None,
           minSize=None, maxSize=None):
    
    bestCand = None
    bestScore = float('-inf')
    bestB = None
    bestA = None
    
    mode, numCandidates = candidateCounter(eventType=eventType, dLimits=dLimits, rLimits=rLimits,
                                           left=left, right=right, numPts=yValues.size,
                                           minSize=minSize, maxSize=maxSize)
    
    if mode == 'error':
        return bestCand, bestB, bestA
    
    if mode == 'usedLimits':
        candGen = candidatesFromDandRlimits(eventType=eventType, dLimits=dLimits,
                                            rLimits=rLimits)
    elif mode == 'usedSize':
        candGen = candidatesFromEventSize(eventType=eventType, left=left, right=right, 
                                          minSize=minSize, maxSize=maxSize)
    else:
        raise Exception('candidateCounter() returned unexpected "mode" ')
     
    counter = 0
    for cand in candGen:
        score, B, A = scoreSubFrame(yValues, left, right, cand, sigmaB, sigmaA)
        if score > bestScore:
            bestScore = score
            bestB = B
            bestA = A
            bestCand = cand
        counter += 1
        if counter % 1000 == 0:
            yield 'fractionDone', counter/numCandidates
           
    if eventType == 'DandR':
        k = 4
    else:
        k = 3
        
    # lineScore = logLikelihoodLine(yValues, sigmaB=sigmaB, left=left, right=right)
    lineScore = logLikelihoodLine(yValues, sigmaB=np.sqrt(np.var(yValues)), left=left, right=right)
    aiccSol = aicc(bestScore, right-left+1, k)
    aiccLine = aicc(lineScore, right-left+1, 1)
    if aiccSol < aiccLine:
        pLine = exp(-(aiccLine - aiccSol)/2)
    else:
        pLine = 1.00
    if pLine > MAX_PLINE:
        yield 'no event present', counter/numCandidates

    yield subFrameAdjusted(eventType=eventType, cand=bestCand, 
                           B=bestB, A=bestA, sigmaB=sigmaB, sigmaA=sigmaA,
                           yValues=yValues, left=left, right=right)
