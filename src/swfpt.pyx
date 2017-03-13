#cython: embedsignature=True
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False
#
# Cython version of the first passage time distribution for a switching DDM

import hddm

import numpy as np

cimport numpy as np
cimport cython

from cython.parallel import *

include "spdf.pxi"

def pdf_array(np.ndarray[double, ndim=1] x, double v1, double v2, double sv, double a, double z, double sz, double[:] s, double t, double st, double err=1e-4, bint logp=0, int n_st=2, int n_sz=2, bint use_adaptive=1, double simps_err=1e-3, double p_outlier=0, double w_outlier=0):

    cdef Py_ssize_t size = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim=1] y = np.empty(size, dtype=np.double)

    # for i in prange(size, nogil=True):
    for i in range(size):
        print(i)
        y[i] = full_pdf(x[i], v1, v2, sv, a, z, sz, s, t, st, err, n_st, n_sz, use_adaptive, simps_err)

    y = y * (1 - p_outlier) + (w_outlier * p_outlier)
    if logp==1:
        return np.log(y)
    else:
        return y

cdef inline bint p_outlier_in_range(double p_outlier): return (p_outlier >= 0) & (p_outlier <= 1)
