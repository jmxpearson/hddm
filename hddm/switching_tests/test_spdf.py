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
        t1_begin = time.time()
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
        t1_end = time.time()


        #check time for loop
        t2_begin = time.time()
        for j1 in j:
            for k1 in k:
                for dv1 in dv:
                    for sign1 in sign:
                        expValue = self.getExpectedValueB(k1, j1, dv1, sign1)
                        m = swfpt.BB(k1, j1, dv1, sign1)
                        np.testing.assert_almost_equal(m, expValue)
        t2_end = time.time()

        print ("matrix-time:", t1_end - t1_begin, " loop-time:", t2_end-t2_begin)
        #Values are around : matrix-time: 0.0009610652923583984  loop-time: 0.05646038055419922


    def test_check_A(self, size=100):
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
        k = np.arange(1, nk)
        v = np.linspace(0.0, 5.0, num=nv)
        y = np.linspace(0.0, 5.0, num=ny)
        kx, vx, yx = np.meshgrid(k, v, y)
        A_k_0 = self.getExpectedValueAk0(kx, vx, yx)
        #A_k_0_cy = swfpt.AA0(kx, vx, yx)
        #np.testing.assert_almost_equal(A_k_0_cy, A_k_0)

if __name__ == "__main__":
    test = TestABParams()
    test.test_check_B()

