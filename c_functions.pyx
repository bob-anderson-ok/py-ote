#cython: language_level=3, boundscheck=False
#cython: language_level=3, wraparound=False
#cython: language_level=3, cdivision=True

cdef extern from "math.h" nogil:
    double log(double x)

cdef extern from "math.h" nogil:
    double sqrt(double x)

cdef double c_loglikelihood(double y, double m, double sigma) nogil:
    """ calculate ln(likelihood) given Gaussian statistics

    Args:
        y (double):     measured value
        m (double):     mean (expected model value)
        sigma (double): stdev of measurements

    Returns:
        natural logarithm of un-normalized probability based on Gaussian distribution
    """

    # -log(sqrt(2*pi)) = -0.9189385332046727

    return -0.9189385332046727 - log(sigma) - (y-m)*(y-m)/(2*sigma*sigma)

def wrap_logl(y, m, sigma):
    return c_loglikelihood(y, m, sigma)

# The following routine finds an R edge using a square wave model
cdef int c_find_Redge_logl(int n, double[:] y, double[:] metric,
                   double[:] mb, double[:] ma,
                   double b, double a, double sigma_b, double sigma_a) nogil:
    cdef:
        int i, edge_pos = 0
        double mcur = 0
        double mmax
        double noise

    for i in range(n):
        mb[i] = c_loglikelihood(y[i], b, sigma_b)
        ma[i] = c_loglikelihood(y[i], a, sigma_a)
        mcur += mb[i]

    mmax = mcur

    for i in range(n-1):
        metric[i] = mcur
        mcur += ma[i] - mb[i]
        if mcur > mmax:
            mmax = mcur
            edge_pos = i + 1

    metric[n-1] = mcur

    return edge_pos

def find_Redge_logl(n, y, metric, mb, ma, b, a, sigma_b, sigma_a):
    return c_find_Redge_logl(n, y, metric, mb, ma, b, a, sigma_b, sigma_a)

# The following routine finds an D edge using a subframe model
cdef double c_find_Dedge_logl(int n, double[:] y,
                   double[:] mb, double[:] ma, double[:] mm,
                   double b, double a, double sigma_b, double sigma_a) nogil:
    cdef:
        int i 
        int edge_pos
        double mcur = 0.0
        double mmax
        double noise
        double adj

    for i in range(n):
        mb[i] = c_loglikelihood(y[i], b, sigma_b)
        ma[i] = c_loglikelihood(y[i], a, sigma_a)
        if y[i] > b:
            mm[i] = mb[i]
        elif y[i] < a:
            mm[i] = ma[i]
        else:
            noise = sigma_b - ((b - y[i])/(b - a)) * (sigma_b - sigma_a)
            mm[i] = c_loglikelihood(y[i], y[i], noise)
        mcur += ma[i]
        
    
    mcur = mcur - ma[0] + mm[0]
    edge_pos = 0
    
    mmax = mcur
    for i in range(n-1):
        mcur = mcur - mm[i] + mb[i] - ma[i+1] + mm[i+1]
        if mcur > mmax:
            mmax = mcur
            edge_pos = i + 1
            
    if y[edge_pos] >= a and y[edge_pos] <= b:
        adj = 1.0 - ((b-y[edge_pos]) / (b - a))
    elif y[edge_pos] > b:
        adj = 1.0
    else: # y[edge_pos] < a
        adj = 0.0
        
    return float(edge_pos) + adj
        
        
def find_Dedge_logl(n, y, mb, ma, mm, b, a, sigma_b, sigma_a):
    return c_find_Dedge_logl(n, y, mb, ma, mm, b, a, sigma_b, sigma_a)    
 

cdef int c_find_edge_wlsq(int n, double[:] y, double[:] metric,
                        double[:] numeratorb, double[:] numeratora,
                        double[:] denominatorb, double[:] denominatora,
                        double b, double a, double sigma_b, double sigma_a) nogil:
    cdef:
        int i, edge_pos = 0
        double mcur = 0
        double cumnumerator = 0
        double cumdenominator = 0
        double mmin

    for i in range(n):
        denominatorb[i] = 1 / (sigma_b * sigma_b)
        denominatora[i] = 1 / (sigma_a * sigma_a)
        numeratorb[i]   = (y[i]-b) * (y[i]-b) * denominatorb[i]
        numeratora[i]   = (y[i]-a) * (y[i]-a) * denominatora[i]
        cumnumerator   += numeratorb[i]
        cumdenominator += denominatorb[i]

    mcur = cumnumerator / cumdenominator
    mmin = mcur

    for i in range(n-1):
        metric[i] = mcur
        cumnumerator   += numeratora[i]   - numeratorb[i]
        cumdenominator += denominatora[i] - denominatorb[i]
        mcur = cumnumerator / cumdenominator
        if mcur < mmin:
            mmin = mcur
            edge_pos = i + 1

    metric[n-1] = mcur

    return edge_pos

def find_edge_wlsq(n, y, metric, numeratorb, numeratora, denominatorb, denominatora, b, a, sigma_b, sigma_a):
    return c_find_edge_wlsq(n, y, metric, numeratorb, numeratora, denominatorb, denominatora, b, a, sigma_b, sigma_a)

cdef int c_find_edge_lsq(int n, double[:] y, double[:] metric, double[:] mb, double[:] ma,
                        double b, double a) nogil:
    cdef:
        int i, edge_pos = 0
        double mcur = 0
        double mmin

    for i in range(n):
        mb[i] = (y[i]-b) * (y[i]-b)
        ma[i] = (y[i]-a) * (y[i]-a)
        mcur += mb[i]

    mmin = mcur

    for i in range(n-1):
        metric[i] = mcur
        mcur += ma[i] - mb[i]
        if mcur < mmin:
            mmin = mcur
            edge_pos = i + 1

    metric[n-1] = mcur

    return edge_pos

def find_edge_lsq(n, y, metric, mb, ma, b, a):
    return c_find_edge_lsq(n, y, metric, mb, ma, b, a)



cdef double c_corr_logl(double rho, double y1, double m1, double sigma1, double y0, double m0, double sigma0) nogil:

    cdef:
        double term0
        double term1
        double term2
        double term3
        double term4

    term0 = 2 * (1 - rho * rho)
    term1 = -log(sqrt(6.283185307179586 * (1-rho*rho))) - log(sigma1)
    term2 = -(y1-m1) * (y1-m1) / (sigma1 * sigma1) / term0
    term3 = 2 * rho * (y1-m1) * (y0-m0) / (sigma1 * sigma0) / term0
    term4 = - rho * rho * (y0-m0) * (y0-m0) / (sigma0 * sigma0) / term0

    return term1 + term2 + term3 + term4

def corr_logl(rho, y1, m1, sigma1, y0, m0, sigma0):
    return c_corr_logl(rho, y1, m1, sigma1, y0, m0, sigma0)

cdef int c_find_edge_corr_logl(int n, double rho, double[:] y, double[:] metric,
                        double[:] paa, double[:] pbb, double[:] pba,
                        double b, double a, double sigma_b, double sigma_a) nogil:
    cdef:
        int i, edge_pos = 0
        double mcur = 0
        double mmax

    paa[0] = c_corr_logl(rho, y[0], a, sigma_a, y[0], a, sigma_a)
    pbb[0] = c_corr_logl(rho, y[0], b, sigma_b, y[0], b, sigma_b)
    pba[0] = pbb[0]

    mcur = pba[0]

    for i in range(1, n):
        paa[i] = c_corr_logl(rho, y[i], a, sigma_a, y[i-1], a, sigma_a)
        pbb[i] = c_corr_logl(rho, y[i], b, sigma_b, y[i-1], b, sigma_b)
        pba[i] = c_corr_logl(rho, y[i], b, sigma_b, y[i-1], a, sigma_a)
        mcur += pbb[i]

    mmax = mcur

    for i in range(1, n):
        metric[i-1] = mcur
        mcur += pba[i] - pbb[i]      # replace pbb[i]   with pba[i]   in current sum
        mcur += paa[i-1] - pba[i-1]  # replace pba[i-1] with paa[i-1] in current sum
        if mcur > mmax:
            mmax = mcur
            edge_pos = i

    metric[n-1] = mcur

    return edge_pos

def find_edge_corr_logl(n, rho, y, metric, paa, pbb, pba, b, a, sigma_b, sigma_a):
    return c_find_edge_corr_logl(n, rho, y, metric, paa, pbb, pba, b, a, sigma_b, sigma_a)
