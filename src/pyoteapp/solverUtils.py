#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 07:00:33 2017

@author: Bob Anderson
"""

from math import exp

import numpy as np

from pyoteapp.likelihood_calculations import cum_loglikelihood, aicc, logLikelihoodLine


def model(*, B=None, A=None,
          edgeTuple=None, 
          sigmaB=None, sigmaA=None, numPts=None):
    
    D, R = edgeTuple
    assert(numPts > 0 and sigmaA >= 0 and sigmaB >= 0)
    m = np.ndarray(shape=[numPts])
    sigma = np.ndarray(shape=[numPts])
    
    if D is None and R is None:
        m[:] = B
        sigma[:] = sigmaB
        
        return m, sigma
    
    if D is not None and R is not None:
        assert((D >= 1) and (R > D))
        assert((D < numPts) and (R < numPts))
        assert(R > D)
        
        m[:D] = B
        m[D:R] = A
        m[R:] = B
        
        sigma[:D] = sigmaB
        sigma[D:R] = sigmaA
        sigma[R:] = sigmaB
        
        return m, sigma
    
    if D is not None and R is None:
        assert((D >= 1) and (D < numPts))
        
        m[:D] = B
        m[D:] = A
        
        sigma[:D] = sigmaB
        sigma[D:] = sigmaA
        
        return m, sigma
    
    if R is not None and D is None:
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
            yield (D, None)
    
    elif eventType == 'Ronly':
        for R in range(left+minSize, left+maxSize+1):
            yield (None, R)
    
    elif eventType == 'DandR':
        for size in range(minSize, maxSize+1):
            for pos in range(left+1, right+1-size):
                yield (pos, pos+size)
        
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
            yield (D, None)
    
    elif eventType == 'Ronly':
        for R in range(rLimits[0], rLimits[1] + 1):
            yield (None, R)
    
    elif eventType == 'DandR':
        for D in range(dLimits[0], dLimits[1] + 1):
            for R in range(rLimits[0], rLimits[1] + 1):
                yield (D, R)
        
    else:
        raise Exception("Unrecognized edvent type")
        

def calcNumCandidatesFromDandRlimits(*, eventType='DandR', dLimits=None,
                                     rLimits=None):
    
    # The D and R limits are assumed valid as input and are non-overlapping
    
    if eventType == 'Donly':
        return dLimits[1] - dLimits[0] + 1
    elif eventType == 'Ronly':
        return rLimits[1] - rLimits[0] + 1
    elif eventType == 'DandR':
        return (dLimits[1] - dLimits[0] + 1) * (rLimits[1] - rLimits[0] + 1)
    else:
        raise Exception("Unrecognized event type")
 

def calcNumCandidatesFromEventSize(*, eventType='DandR',
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
    

def calcBandA(*, yValues=None, left=None, right=None, cand=None):
    
    assert(right > left)
    
    D, R = cand  # Extract D and R from the tuple
        
    if R is None:
        assert(D > left)
        # This is a 'Donly' candidate
        # Note that the yValue at D is not included in the B calculation
        B = np.mean(yValues[left:D])
        # We have to deal with a D at the right edge.  There is no value to
        # the right to use to calculate A so we simply return the value at D
        # as the best estimate of A
        if D == right:
            A = yValues[D]
        else:
            A = np.mean(yValues[D+1:right+1])
        if A > B:
            A = B - 1
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
            B = np.mean(yValues[R+1:right+1])
        A = np.mean(yValues[left:R])  # smallest R is left + 1
        if A > B:
            A = B - 1
        return B, A
    
    else:
        assert((D > left) and (R <= right) and (R > D))
        # We have a 'DandR' candidate
        leftBvals = yValues[left:D]  # Smallest D will 1
        if R == right:
            rightBvals = yValues[right]
        else:
            rightBvals = yValues[R+1:right+1]
        B = (np.sum(leftBvals) + np.sum(rightBvals)) / (leftBvals.size + rightBvals.size)
        
        if R - D == 1:  # Event size of 1 has no valid A --- we choose the value at D
            A = yValues[D]
        else:    
            A = np.mean(yValues[D+1:R])
        if A > B:
            A = B - 1
        return B, A


def scoreCandidate(yValues, left, right, cand, sigmaB, sigmaA):
    B, A = calcBandA(yValues=yValues, left=left, right=right, cand=cand)
    m, sigma = model(B=B, A=A, edgeTuple=cand, 
                     sigmaB=sigmaB, sigmaA=sigmaA, numPts=yValues.size)
    return cum_loglikelihood(yValues, m, sigma, left, right), B, A


def scoreSubFrame(yValues, left, right, cand, sigmaB, sigmaA):
    B, A = calcBandA(yValues=yValues, left=left, right=right, cand=cand)
    m, sigma = model(B=B, A=A, edgeTuple=cand, 
                     sigmaB=sigmaB, sigmaA=sigmaA, numPts=yValues.size)
    D, R = cand
    if D is not None:
        if (yValues[D] < B) and (yValues[D] > A):
            m[D] = yValues[D]
    if R is not None:
        if (yValues[R] < B) and (yValues[R] > A):
            m[R] = yValues[R]
    return cum_loglikelihood(yValues, m, sigma, left, right), B, A


def getAssociationForEntry(*, yValues=None, entry=None,
                           B=None, A=None,
                           sigmaB=None, sigmaA=None):
    eVal = yValues[entry]
    if (eVal > (A + 3 * sigmaA)) and (eVal < (B - 3 * sigmaB)):
        return 'subFrameValue'
    elif abs(eVal - B) < abs(eVal-A):
        return 'B'
    else:
        return 'A'
    

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
                    dLimits=dLimits, rLimits=rLimits))
        else:
            if minMaxOk():
                return ('usedSize', calcNumCandidatesFromEventSize(eventType=eventType, 
                        left=left, right=right, minSize=minSize, maxSize=maxSize))
            else:
                return 'error', -1
            
    elif eventType == 'Ronly':
        if rLimits:
            return ('usedLimits', calcNumCandidatesFromDandRlimits(eventType=eventType, 
                    dLimits=dLimits, rLimits=rLimits))
        else:
            if minMaxOk():
                return ('usedSize', calcNumCandidatesFromEventSize(eventType=eventType, 
                        left=left, right=right, minSize=minSize, maxSize=maxSize))
            else:
                return 'error', -1
        
    elif eventType == 'DandR':
        if rLimits and dLimits:
            return ('usedLimits', calcNumCandidatesFromDandRlimits(eventType=eventType,
                    dLimits=dLimits, rLimits=rLimits))
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

    # Unlike R-OTE, where an AIC determination is made to qualify for subframe timing,
    # here we simply compare against sigmaB and sigmaB
    sigFactor = 1.5

    def adjusted(transitionPoint):
        value = yValues[transitionPoint]
        if (value > (A + sigFactor * sigmaA)) and (value < (B - sigFactor * sigmaB)):
            if transitionPoint == R:
                adj = (B - value) / (B - A)
            else:
                adj = (value - A) / (B - A)
            return transitionPoint + adj
        else:
            return transitionPoint

    D, R = cand
    
    if eventType == 'Donly':
        return (adjusted(D), R), B, A

    elif eventType == 'Ronly':
        return (D, adjusted(R)), B, A

    elif eventType == 'DandR':
        return (adjusted(D), adjusted(R)), B, A
                       
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
        # score, B, A = scoreCandidate(yValues, left, right, cand, sigmaB, sigmaA)
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
        
    lineScore = logLikelihoodLine(yValues, sigmaB=sigmaB, left=left, right=right)
    aiccSol = aicc(bestScore, right-left+1, k)
    aiccLine = aicc(lineScore, right-left+1, 1)
    if aiccSol < aiccLine:
        pLine = exp(-(aiccLine - aiccSol)/2)
    else:
        pLine = 1.00
    if pLine > 0.001:
        yield 'no event present', counter/numCandidates

    yield subFrameAdjusted(eventType=eventType, cand=bestCand, 
                           B=bestB, A=bestA, sigmaB=sigmaB, sigmaA=sigmaA,
                           yValues=yValues, left=left, right=right)
