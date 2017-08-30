import unittest
import numpy as np
import matplotlib.pyplot as plt

import swfpt

class TestABParams(unittest.TestCase):
    def runTest(self):
        pass

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
        np.testing.assert_almost_equal(m, expected)
