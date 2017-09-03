import unittest
import numpy as np
import matplotlib.pyplot as plt
import time

import swfpt

class TestABParams(unittest.TestCase):
    def runTest(self):
        pass

    def getExpectedValueB(self, j, k, dv, sign):
        temp = 1.0 * (-1) ** (sign + 1)
        B = temp * 4.0 * np.pi**2 * j * k * dv * (((-1) ** (j + k) * np.exp(temp*dv) - 1)) / (2.0 * np.pi**2 * (j**2 + k**2) * dv**2 + 1.0*np.pi**4 * (j**2 - k**2)**2 + dv**4)
        return B

    def getExpectedValueAk0(self, k, v, y):
        A = np.exp(-v * y) * np.sqrt(2) * np.sin(k* np.pi * y)
        return A

    def getExpectedValueAkj(self, k, v, y):
        A = 0
        return A

    def test_check_B(self, size=100):
        j = 1
        k = 0
        dv = 0
        sign = 0
        m = swfpt.BB(k, j, dv, sign)
        np.testing.assert_almost_equal(m,0);

        #Sign is negative
        j = 1
        k = 1
        dv = 1
        sign = 1
        expected = 1.675832495
        m = swfpt.BB(k, j, dv, sign)
        np.testing.assert_almost_equal(m, expected)

        #Sign is positive
        sign = 0
        expected = 0.61650432
        m = swfpt.BB(k, j, dv, sign)
        B = self.getExpectedValueB(k, j, dv, sign)
        np.testing.assert_almost_equal(m, expected)
        np.testing.assert_almost_equal(B, expected)
        np.testing.assert_almost_equal(m, B)

        #check time for matrix version
        #t1_begin = time.time()
        nj, nk, ndv, nsign = (10, 10, 10, 2)
        k = np.arange(1, nk)
        j = np.arange(1, nj)
        dv = np.linspace(0.0, 5.0, num=ndv)
        sign = np.arange(0, nsign)
        vfunc = np.vectorize(swfpt.BB)
        xk, xj, xdv, xsign = np.meshgrid(k, j, dv, sign)
        expValue = self.getExpectedValueB(xk, xj, xdv, xsign)
        m = vfunc(xk, xj, xdv, xsign)
        np.testing.assert_almost_equal(m, expValue)
        #t1_end = time.time()

        #print ("matrix-time:", t1_end - t1_begin, " loop-time:", t2_end-t2_begin)
        #Values are around : matrix-time: 0.0009610652923583984  loop-time: 0.05646038055419922


    def test_check_Ak0(self, size=100):
        k = 1
        v = 0
        y = 0
        A_k_0 = self.getExpectedValueAk0(k, v, y)
        A_k_0_cy = swfpt.AA0(k, v, y)
        np.testing.assert_almost_equal(A_k_0, 0)
        np.testing.assert_almost_equal(A_k_0_cy, 0)

        k = 1
        v = 0
        y = 1
        A_k_0 = self.getExpectedValueAk0(k, v, y)
        A_k_0_cy = swfpt.AA0(k, v, y)
        np.testing.assert_almost_equal(A_k_0_cy, A_k_0)


        nk, nv, ny = (10, 10, 10)
        k = np.arange(1, nk+1)
        v = np.linspace(0.0, 5.0, num=nv)
        y = np.linspace(0.0, 5.0, num=ny)
        kx, vx, yx = np.meshgrid(k, v, y)
        A_k_0 = self.getExpectedValueAk0(kx, vx, yx)
        vfunc = np.vectorize(swfpt.AA0)
        A_k_0_cy = vfunc(kx, vx, yx)
        np.testing.assert_almost_equal(A_k_0_cy, A_k_0)

    # tt - time for which pdf is wanted
    # w - z/a normalized
    # v1 - diffusion rate 1
    # v2 - diffusion rate 2
    # s - Switching times
    # n - number of elements in s before time t

    def calculate_Ak(self, tt, w, v1, v2, s, n, err):
        K = 2
        A = np.array(np.zeros(K), dtype=np.double)
        tmp = np.array(np.zeros(K), dtype=np.double)
        k_val = np.arange(1, K+1)
        tmp = np.zeros_like(tmp)
        if (s[0] != 0):
           s = np.insert(s, 0, 0)
        diff = np.ediff1d(s)

        #get dv
        dv = v1 - v2
        #starting value of A
        A = self.getExpectedValueAk0(k_val, v1, w)
        print ("A:", A)

        #Calculate B matrix to be used
        nj, nk, nsign = (2, 2, 2)
        k = np.arange(1, nk+1)
        j = np.arange(1, nj+1)
        sign = np.arange(0, nsign)
        xk, xj, xsign = np.meshgrid(k, j, sign)
        B = self.getExpectedValueB(xk, xj, dv, xsign)
        print ("B:", B)

        #calculate lambda in array format
        lambda_arr = (k**2) * (np.pi**2) / 2

        #Array with one row for each lambda, one column for each time itertion tau
        mul_factor = np.exp(-np.outer(lambda_arr, diff))


        #Start with sign 0 and flip back and forth in every iteration
        sign = 0
        #going from 1 to n, calculate A-s iteratively
        for i in range(n):
            tau = diff[i-1] #difference from previous switching value
            A = A * mul_factor[:, i-1]
            #Multiply with B matrix for particular sign
            A = np.dot(B[:,:,sign], A)
            print ("iter:", i)
            print (A)
            sign = 1-sign





    def test_check_pdf_kernel(self, size=100):

        x = np.linspace(0, 10, 1000)
        x_size = x.shape[0]
        v1 = 2
        v2 = 1
        sv = 0
        a = 1
        z = 0.5
        sz = 0
        s = np.arange(0, 2, 0.1)
        t = 0
        st = 0
        err = 1e-4
        logp = 0
        s_size = s.shape[0]
        swfpt.pdf_array(x, v1, v2, sv, a, z, sz, s, t, st, err, logp)
        self.calculate_Ak(x, z, v1, v2, s, s_size, err)



if __name__ == "__main__":
    test = TestABParams()
    test.test_check_B()
    test.test_check_Ak0()
    test.test_check_pdf_kernel()

