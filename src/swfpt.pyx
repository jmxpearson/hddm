#cython: embedsignature=True
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False
#
# Cython version of the first passage time distribution for a switching DDM

import hddm

import scipy.integrate as integrate
from copy import copy
import numpy as np

cimport numpy as np
cimport cython

from cython.parallel import *
#cimport openmp

include "spdf.pxi"
