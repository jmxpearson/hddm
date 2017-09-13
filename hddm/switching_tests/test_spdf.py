import unittest
import numpy as np
# import matplotlib.pyplot as plt
# import time

import swfpt
import bisect


class TestABParams(unittest.TestCase):
    def runTest(self):
        pass

    def getExpectedValueB(self, j, k, dv, sign):
        temp = 1.0 * (-1) ** (sign + 1)
        B = temp * 4.0 * np.pi**2 * j * k * dv * (((-1) ** (j + k) * np.exp(temp*dv) - 1)) / (2.0 * np.pi**2 * (j**2 + k**2) * dv**2 + 1.0*np.pi**4 * (j**2 - k**2)**2 + dv**4)
        return B

    def getExpectedValueAk0(self, k, v, y):
        A = np.exp(-v * y) * np.sqrt(2) * np.sin(k * np.pi * y)
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
        np.testing.assert_almost_equal(m, 0)

        # Sign is negative
        j = 1
        k = 1
        dv = 1
        sign = 1
        expected = 1.675832495
        m = swfpt.BB(k, j, dv, sign)
        np.testing.assert_almost_equal(m, expected)

        # Sign is positive
        sign = 0
        expected = 0.61650432
        m = swfpt.BB(k, j, dv, sign)
        B = self.getExpectedValueB(k, j, dv, sign)
        np.testing.assert_almost_equal(m, expected)
        np.testing.assert_almost_equal(B, expected)
        np.testing.assert_almost_equal(m, B)

        # check time for matrix version
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

    #  tt - time for which pdf is wanted
    #  w - z/a normalized
    #  v1 - diffusion rate 1
    #  v2 - diffusion rate 2
    #  s - Switching times
    #  n - number of elements in s before time t

    def calculate_Ak(self, tt, w, v1, v2, s, n, err):
        K = 4
        A = np.array(np.zeros(K), dtype=np.double)
        tmp = np.array(np.zeros(K), dtype=np.double)
        k_val = np.arange(1, K+1)
        tmp = np.zeros_like(tmp)
        diff = np.diff(s)

        # get dv
        dv = v1 - v2
        # starting value of A
        A = self.getExpectedValueAk0(k_val, v1, w)

        # Calculate B matrix to be used
        nj, nk, nsign = (K, K, 2)
        k = np.arange(1, nk+1)
        j = np.arange(1, nj+1)
        sign = np.arange(0, nsign)
        xk, xj, xsign = np.meshgrid(k, j, sign)
        B = self.getExpectedValueB(xk, xj, dv, xsign)
        vfunc = np.vectorize(swfpt.BB)
        m = vfunc(xk, xj, dv, xsign)
        np.testing.assert_almost_equal(m, B)

        # starting value of A
        A = self.getExpectedValueAk0(k_val, v1, w)
        vfunc1 = np.vectorize(swfpt.AA0)
        A_k_0_cy = vfunc1(k_val, v1, w)
        np.testing.assert_almost_equal(A_k_0_cy, A)

        # Loop version
        sign = 1
        for i in range(0, n):
            tau = diff[i]  # difference from previous switching value
            for k in range(1, K+1):
                # Calculate A(k, nn+1)
                for j in range(1, K+1):
                    tmp[k-1] += np.exp(-(j*np.pi)**2 * tau/2) * B[k-1, j-1, sign] * A[j-1]
            for k in range(K):
                A[k] = tmp[k]
                tmp[k] = 0
            sign = 1-sign

        p = 0
        for k in range(1, K+1):
            p += k * np.exp(-(k * np.pi)**2 * (tt - s[n])/2) * A[k-1]

        return p

    def calculate_full_pdf(self, x, a, z, v1, v2, s, NS, err):

        if (s[0] != 0):
            s = np.insert(s, 0, 0)

        # Normalize values
        a_sqr = a**2
        w = z/a
        tt = x/a_sqr
        vv1 = v1*a
        vv2 = v2*a
        ss = s/a_sqr

        n = bisect.bisect(ss, tt) - 1
        p = self.calculate_Ak(tt, w, vv1, vv2, ss, n, err)
        idx = 1
        v = []
        v.append(vv1)
        v.append(vv2)
        v_iter = v[idx]**2
        sum_sqr_v = 0
        for i in range(n+1):
            v_iter = v[idx]**2
            if i < n:
                tau = (s[i+1] - s[i])
            else:
                tau = (tt - s[i])
            sum_sqr_v += tau * v_iter
            idx = 1-idx
        pdf = p * np.pi * np.exp(-sum_sqr_v / 2) / (np.sqrt(2) * a_sqr)
        return pdf

    def calculate_pdf_array(self, x, a, z, v1, v2, s, NS, logp, err):
        y = []
        for i in range(x.shape[0]):
            y.append(self.calculate_full_pdf(x[i], a, z, v1, v2, s, NS, err))
        if logp == 1:
            return np.log(y)
        else:
            return y

    def test_check_pdf_kernel(self, size=100):

        x = np.linspace(0.225, 0.225, 1)
        v1 = 2
        v2 = 1
        sv = 0
        a = 1
        z = 0.5
        sz = 0
        s = np.arange(0.1, 0.3, 0.05)
        t = 0
        st = 0
        err = 1e-4
        logp = 0
        s_size = s.shape[0]
        pdf_expected = self.calculate_pdf_array(x, a, z, v1, v2, s, s_size, logp, err)
        pdf_cython = swfpt.pdf_array(x, v1, v2, sv, a, z, sz, s, t, st, err, logp)
        np.testing.assert_almost_equal(pdf_cython, pdf_expected)

if __name__ == "__main__":
    test = TestABParams()
    test.test_check_B()
    test.test_check_Ak0()
    test.test_check_pdf_kernel()
