import unittest
import numpy as np
import matplotlib.pyplot as plt

import swfpt

class TestABParams(unittest.TestCase):
    def runTest(self):
        pass

    def getExpectedValueB(self, j, k, dv, sign):
        temp = 1.0 * (-1) ** (sign + 1)
        B = temp * 4.0 * np.pi**2 * j * k * dv * (((-1) ** (j + k) * np.exp(temp*dv) - 1)) / (2.0 * np.pi**2 * (j**2 + k**2) * dv**2 + 1.0*np.pi**4 * (j**2 - k**2)**2 + dv**4)
        return B



    def test_check_B(self, size=100):
        j = 1
        k = 0
        dv = 0
        sign = 0
        m = swfpt.BB(k, j, dv, sign)
        np.testing.assert_almost_equal(m,0);

        #Sign is positive
        j = 1
        k = 1
        dv = 1
        sign = 1
        expected = 1.675832495
        m = swfpt.BB(k, j, dv, sign)
        np.testing.assert_almost_equal(m, expected)

        #Sign is negative
        sign = 0
        expected = 0.61650432
        m = swfpt.BB(k, j, dv, sign)
        B = self.getExpectedValueB(k, j, dv, sign)
        np.testing.assert_almost_equal(m, expected)
        np.testing.assert_almost_equal(B, expected)
        np.testing.assert_almost_equal(m, B)

        for j in range(1, 5):
            for k in range(1, 5):
                for dv in np.linspace(0.0, 3.0, num=7):
                    for sign in range(0, 2):
                        expValue = self.getExpectedValueB(k, j, dv, sign)
                        m = swfpt.BB(k, j, dv, sign)
                        print ("k= ", k, " j = ", j, " dv = ", dv, " sign = ", sign)
                        np.testing.assert_almost_equal(m, expValue)
