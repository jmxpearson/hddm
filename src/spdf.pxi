
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

cdef double B(int k, int j, double dv, bint sign) nogil:
    """
    Calculate B coefficient (i, j) for drift rate difference dv.
    Use sign = 1 for B+, sign = 0 for B-
    """
    cdef double alpha = (-1)**(sign + 1) * dv/2

    cdef double B = j * k * 4 * M_PI**2 * alpha;

    B *= exp(alpha) * (-1)**(j + k) - 1

    B /= (alpha**4 + 2 * M_PI**2 * alpha**2 * (j**2 + k**2) +
          M_PI**4 * (j**2 - k**2))

    return B

def BB(k, j, dv, s):
    return B(k, j, dv, s)

cdef inline double A1(int k, double v1, double z) nogil:
    """
    Returns A_{k,1}(v, z)
    """
    return sqrt(2) * sin(k * M_PI * z) * exp(-v1 * z/2)

cdef double A(int k, int n, double v1, double v2, double z, double[:] s) nogil:
    """
    Recursively calculate coefficent A_{k, n}(z)
    TODO: Make sure n <= len(s) + 1
    """
    if n == 1:
        return A1(k, v1, z)

    cdef int J = 4

    cdef double ds = 0
    if n == 2:
        ds = s[0]
    else:
        ds = s[n - 2] - s[n - 3]

    cdef double dv = v1 - v2
    cdef double acc = 0
    cdef int j

    for j in range(1, J + 1):
        acc += exp(-(j*M_PI)**2 * ds) * A(j, n - 1, v1, v2, z, s) * B(k, j, dv, n % 2 + 1)

    return acc



def AA(k, n, v1, v2, z, ds):
    return A(k, n, v1, v2, z, ds)
