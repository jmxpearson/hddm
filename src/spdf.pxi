
#cython: embedsignature=True
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False

cimport cython

cdef extern from "math.h" nogil:
    double sin(double)
    double cos(double)
    double log(double)
    double exp(double)
    double sqrt(double)
    double M_PI

# from libcpp.vector cimport vector
#
# cdef extern from "<algorithm>" namespace "std" nogil:
#     F upper_bound[F, T](F f, F l, T val)

cdef double B(int k, int j, double dv, bint sign) nogil:
    """
    Calculate B coefficient (i, j) for drift rate difference dv.
    Use sign = 1 for B+, sign = 0 for B-
    """
    cdef double alpha = (-1)**(sign + 1) * dv

    cdef double B = j * k * 4 * M_PI**2 * alpha;

    B *= exp(alpha) * (-1)**(j + k) - 1

    B /= (alpha**4 + 2 * M_PI**2 * alpha**2 * (j**2 + k**2) +
          M_PI**4 * (j**2 - k**2))

    return B

def BB(k, j, dv, s):
    return B(k, j, dv, s)

cdef inline double A0(int k, double v1, double z) nogil:
    """
    Returns A_{k,1}(v, z)
    """
    return sqrt(2) * sin(k * M_PI * z) * exp(-v1 * z/2)

cdef double A(int k, int n, double v1, double v2, double z, double[:] s) nogil:
    """
    Recursively calculate coefficent A_{k, n}(z)
    TODO: Make sure n <= len(s)
    """
    if n == 0:
        return A0(k, v1, z)

    cdef int J = 4  # TODO: replace with principled bound

    cdef double ds = 0
    if n == 1:
        ds = s[0]
    else:
        ds = s[n - 1] - s[n - 2]

    cdef double dv = v1 - v2
    cdef double acc = 0
    cdef int j

    for j from 1 <= j <= J:
        acc += exp(-(j*M_PI)**2 * ds) * A(j, n - 1, v1, v2, z, s) * B(k, j, dv, n % 2)

    return acc

cdef double pdf_kernel(double tt, double w, double v1, double v2, double[:] s, int n, double err) nogil:
    """
    Compute the (infinite) sum piece of the pdf using dimensionless
    variables.
    """

    # TODO: compute number of terms needed...
    cdef int K = 4

    cdef double p
    cdef int k
    for k from 1 <= k <= K:
        p += k * exp(-(k * M_PI)**2 * (tt - s[n])) * A(k, n, v1, v2, w, s)
    p *= M_PI

    return p

cdef double pdf(double t, double v1, double v2, double a, double z, double[:] s, double err) nogil:
    """
    Compute the likelihood of the switching drift diffusion model
    f(t|v, a, z, s) with switch times given by the array s.
    v1 is the drift velocity for the first time segment, after which
    diffusion alternates between v1 and v2.
    """
    if t <= 0:
        return 0

    # convert to normalized values
    cdef double asq = a**2
    cdef double tt = t/asq
    cdef double w = z/a
    cdef double vv1 = a * v1
    cdef double vv2 = a * v2
    cdef double[:] ss = s
    cdef int ii
    for ii in range(ss.shape[0]):
        ss[ii] /= asq

    cdef int n
    # find n(tt)
    # this is okay for short lists s, but is terrible for long ones
    # eventually, should replace this with binary search, since s is sorted
    for n in range(s.shape[0]):
        if s[n] > tt:
            break
    n -= 1

    cdef double p = pdf_kernel(tt, w, vv1, vv2, ss, n, err)

    cdef double vv
    cdef double sum_v2 = 0
    for ii from 1 <= ii <= n + 1:
        vv = v1 if ii % 2 else v2
        if ii <= n:
            sum_v2 += (s[ii] - s[ii - 1]) * vv**2
        else:
            sum_v2 += (tt - s[n]) * vv**2

    p *= exp(-sum_v2/2)
    p *= p/asq

    return p


def AA(k, n, v1, v2, z, ds):
    return A(k, n, v1, v2, z, ds)
