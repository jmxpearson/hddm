
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
    double fabs(double)
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

    cdef double B = 4 * M_PI**2 * j * k * alpha;

    B *= (-1)**(j + k) * exp(alpha) - 1

    B /= (alpha**4 + 2 * M_PI**2 * alpha**2 * (j**2 + k**2) +
          M_PI**4 * (j**2 - k**2))

    return B

def BB(k, j, dv, s):
    return B(k, j, dv, s)

cdef inline bint parity(int n) nogil:
    """
    Return 1 if n even, 0 if odd.
    """
    return n % 2 == 0

cdef inline double A0(int k, double v1, double z) nogil:
    """
    Returns A_{k,1}(v, z)
    """
    return sqrt(2) * sin(k * M_PI * z) * exp(-v1 * z)

cdef double A(int k, int n, double v1, double v2, double z, double[:] s) nogil:
    """
    Recursively calculate coefficent A_{k, n}(z)
    TODO: Make sure n <= len(s)
    """
    if n == 0:
        return A0(k, v1, z)

    cdef int J = 4  # TODO: replace with principled bound
    cdef double dv = v1 - v2

    cdef double acc = 0
    cdef Py_ssize_t j

    for j in range(1, J + 1):
        acc += exp(-(j*M_PI)**2 * (s[n] - s[n - 1])/2) * A(j, n - 1, v1, v2, z, s) * B(k, j, dv, parity(n - 1))

    return acc

cdef double pdf_kernel(double tt, double w, double v1, double v2, double[:] s, int n, double err) nogil:
    """
    Compute the (infinite) sum piece of the pdf using dimensionless
    variables.
    """

    # TODO: compute number of terms needed...
    cdef int K = 4

    cdef double p
    cdef Py_ssize_t k
    for k in range(1, K + 1):
        p += k * exp(-(k * M_PI)**2 * (tt - s[n])/2) * A(k, n, v1, v2, w, s)

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
    cdef Py_ssize_t ii
    for ii in range(ss.shape[0]):
        ss[ii] /= asq

    # find n(tt)
    # this is okay for short lists s, but is terrible for long ones
    # eventually, should replace this with binary search, since s is sorted
    cdef Py_ssize_t n = ss.shape[0] - 1
    for ii in range(ss.shape[0]):
        if ss[ii] > tt:
            n = ii - 1
            break

    cdef double p = pdf_kernel(tt, w, vv1, vv2, ss, n, err)

    cdef double vv
    cdef double sum_v2 = 0
    for ii in range(1, n + 1):
        vv = v1 if ii % 2 else v2
        if ii <= n:
            sum_v2 += (s[ii] - s[ii - 1]) * vv**2
        else:
            sum_v2 += (tt - s[n]) * vv**2

    p *= exp(-sum_v2/2)
    p *= M_PI/(sqrt(2) * asq)

    return p

# cdef double pdf_sv(double t, double v1, double v2, double sv, double a, double z, double[:] s, double err) nogil:
#     """
#     Compute the pdf after integrating over trial-to-trial noise (sv) in
#     the diffiusion rate (v). This can be done analytically.
#     """
#     if t <= 0:
#         return 0
#
#     if sv == 0:
#         return pdf(t, v1, v2, a, z, s, err)
#
#     # fix later
#     return pdf(t, v1, v2, a, z, s, err)

cpdef double full_pdf(double x, double v1, double v2, double sv, double a, double z, double sz, double[:] s, double t, double st, double err, int n_st=2, int n_sz=2, bint use_adaptive=1, double simps_err=1e-3) nogil:
    """full pdf"""

    # Check if parpameters are valid
    if (z<0) or (z>1) or (a<0) or (t<0) or (st<0) or (sv<0) or (sz<0) or (sz>1) or \
       ((fabs(x)-(t-st/2.))<0) or (z+sz/2.>1) or (z-sz/2.<0) or (t-st/2.<0):
       return 0

    cdef Py_ssize_t ii
    for ii in range(1, s.shape[0]):
        if s[ii] <= s[ii - 1]:
            return 0

    return pdf(x, v1, v2, a, z, s, err)


def AA(k, n, v1, v2, z, ds):
    return A(k, n, v1, v2, z, ds)
