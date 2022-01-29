import numpy as np
import warnings

# suppress warnings
warnings.filterwarnings('ignore')


def expRedge(B, A, n, offset, N):
    vstart = A
    vend = B
    n -= offset
    vout = vstart - (vstart - vend) * (1.0 - np.exp(-n / N))
    if vend < vstart < vout:
        vout = vstart
    if vend > vstart > vout:
        vout = vstart
    return vout


def expDedge(B, A, n, offset, N):
    vstart = B
    vend = A
    n -= offset
    vout = vstart - (vstart - vend) * (1.0 - np.exp(-n / N))
    if vend < vstart < vout:
        vout = vstart
    if vend > vstart > vout:
        vout = vstart
    return vout


def getRedgePoints(numPoints, B, A, offset, N):
    # Returns 2 * numPoints + 1 values

    # Generate points to the left of the 'edge'
    edgePoints = [A for _ in range(numPoints)]

    # Now generate numPoints + 1 to the right of the 'edge'
    n = 0
    for i in range(numPoints + 1):
        vn = expRedge(B, A, n, offset, N)
        edgePoints.append(vn)
        n += 1
    return edgePoints


def getDedgePoints(numPoints, B, A, offset, N):
    # Returns 2 * numPoints + 1 values

    # Generate points to the left of the 'edge'
    edgePoints = [B for _ in range(numPoints)]

    # Now generate numPoints + 1 to the right of the 'edge'
    n = 0
    for i in range(numPoints + 1):
        vn = expDedge(B, A, n, offset, N)
        edgePoints.append(vn)
        n += 1
    return edgePoints


def scoreDedge(numTheoryPts, B, A, offset, N, actual, matchPoint):
    theory = getDedgePoints(numTheoryPts, B, A, offset, N)
    nTheory = len(theory)
    assert len(actual) >= nTheory
    assert matchPoint + numTheoryPts <= len(actual)
    metric = 0.0
    for i in range(nTheory):
        metric += (theory[i] - actual[i + matchPoint]) ** 2
    return metric


def scoreRedge(numTheoryPts, B, A, offset, N, actual, matchPoint):
    theory = getRedgePoints(numTheoryPts, B, A, offset, N)
    nTheory = len(theory)
    assert len(actual) >= nTheory
    assert matchPoint + numTheoryPts <= len(actual)
    metric = 0.0
    for i in range(nTheory):
        metric += (theory[i] - actual[i + matchPoint]) ** 2
    return metric


def locateIndexOfBestMatchPoint(numTheoryPts, B, A, offset, N, actual, edge='D'):
    metricVec = []
    for k in range(len(actual) - numTheoryPts * 2):
        if edge == 'D':
            metricVec.append(scoreDedge(numTheoryPts, B, A, offset, N, actual, k))
        else:
            metricVec.append(scoreRedge(numTheoryPts, B, A, offset, N, actual, k))

    return np.where(metricVec == np.min(metricVec))[0][0]


def locateBestOffset(numTheoryPts, B, A, N0, actual, matchPoint, edge='D'):
    metricVec = []
    offsetVec = []
    for offset in np.linspace(-2, 2, 81):
        if edge == 'D':
            metricVec.append(scoreDedge(numTheoryPts, B, A, offset, N0, actual, matchPoint))
        else:
            metricVec.append(scoreRedge(numTheoryPts, B, A, offset, N0, actual, matchPoint))
        offsetVec.append(offset)

    indexToBestOffset = np.where(metricVec == np.min(metricVec))[0][0]
    return offsetVec[indexToBestOffset]


def locateBestN(numTheoryPts, B, A, offset, N0, actual, matchPoint, edge='D'):
    metricVec = []
    NVec = []
    if N0 - 1 <= 0:
        lowN = 0.1
        hiN = lowN + 2.0
    else:
        lowN = N0 - 1.0
        hiN = lowN + 1.0
    for N in np.linspace(lowN, hiN, 41):
        if edge == 'D':
            metricVec.append(scoreDedge(numTheoryPts, B, A, offset, N, actual, matchPoint))
        else:
            metricVec.append(scoreRedge(numTheoryPts, B, A, offset, N, actual, matchPoint))
        NVec.append(N)

    indexToBestN = np.where(metricVec == np.min(metricVec))[0][0]
    return NVec[indexToBestN]
