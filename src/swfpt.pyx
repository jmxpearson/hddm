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

include "spdf.pxi"
