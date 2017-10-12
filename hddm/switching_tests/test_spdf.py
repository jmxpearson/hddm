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
        """
        Numpy version of calculation of B coefficient (k, j) for drift rate
        difference dv
        Use sign = 1 for B+, sign = 0 for B-
        """
        temp = 1.0 * (-1) ** (sign + 1)
        B = temp * 4.0 * np.pi**2 * j * k * dv * (((-1) ** (j + k) * np.exp(temp*dv) - 1)) / (2.0 * np.pi**2 * (j**2 + k**2) * dv**2 + 1.0*np.pi**4 * (j**2 - k**2)**2 + dv**4)
        return B

    def getExpectedValueAk0(self, k, v, y):
        """
        Numpy version of calculation of A_k{k, 0}(v, y)
        """
        A = np.exp(-v * y) * np.sqrt(2) * np.sin(k * np.pi * y)
        return A

    def test_check_B(self, size=100):
        j = 1
        k = 0
        dv = 0
        sign = 0
        m = swfpt.BB(k, j, dv, sign)
        np.testing.assert_almost_equal(m, 0)

        # Sign is positive
        j = 1
        k = 1
        dv = 1
        sign = 1
        expected = 1.675832495
        m = swfpt.BB(k, j, dv, sign)
        np.testing.assert_almost_equal(m, expected)

        # Sign is negative
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

    def calculate_Ak(self, K, tt, w, v1, v2, s, n, err):
        """
        Numpy version of calculation of pdf kernel
        Compute the sum portion of the pdf using normalized variables
        """
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

    def calculate_full_pdf(self, K, x, a, z, v1, v2, s, NS, err):
        """
        Numpy version of calculation of full pdf
        Compute the likelihood of the switching drift diffusion model
        f(t|v1, v2, a, z, s) with switch times given by array s.
        Normalize al lthe variables
        v2 is the drift velocity for the first time segment, after which
        diffusion alternates between v1 and v2 
        """

        if (z<0) or (z>a) or (a<0):
            return 0

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
        p = self.calculate_Ak(K, tt, w, vv1, vv2, ss, n, err)
        idx = 1
        v = []
        v.append(vv1)
        v.append(vv2)
        v_iter = v[idx]**2
        sum_sqr_v = 0
        for i in range(n+1):
            v_iter = v[idx]**2
            if i < n:
                tau = (ss[i+1] - ss[i])
            else:
                tau = (tt - ss[i])
            sum_sqr_v += tau * v_iter
            idx = 1-idx
        pdf = p * np.pi * np.exp(-sum_sqr_v / 2) / (np.sqrt(2) * a_sqr)
        return pdf

    def calculate_pdf_array(self, K, x, a, z, v1, v2, s, NS, logp, err):
        """
        Numpy version of calculating pdf array
        """
        y = []
        for i in range(x.shape[0]):
            y.append(self.calculate_full_pdf(K, x[i], a, z, v1, v2, s, NS, err))
        if logp == 1:
            return np.array(np.log(y))
        else:
            return np.array(y)

    def get_lambda(self, k):
        return ((k**2) * (np.pi**2)) / 2

    def calculate_crude_error_bound(self, K, x, a, z, v1, v2, s):
        """
        Numpy calculation of crude error bounding term
        """
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
        V0 = (vv1/2)**2
        V1 = np.abs(vv2)/2
        if (vv1**2 > vv2**2):
            V0 = (vv2/2)**2
            V1 = np.abs(vv1)/2
        diff = np.diff(ss)
        tau_min = np.amin(diff)
        const = np.sqrt(2*(np.pi**3)) * a_sqr

        error = (n+1) * np.exp(V1*n - n*V0*tau_min) * (tau_min ** (-3/2))
        error *= np.exp(-np.pi**2 * K**2 * tau_min)/const
        return error


    def calculate_crude_error_bound_array(self, K, x, a, z, v1, v2, s):
        """
        Numpy version of calculating crude error for pdf array
        """
        y = []
        for i in range(x.shape[0]):
            y.append(self.calculate_crude_error_bound(K, x[i], a, z, v1, v2, s))
        return y

    def calculate_strict_error_bound(self, K, x, a, z, v1, v2, s):
        """
        Numpy calculation of fine error bounding term
        """
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
        V0 = (vv1/2)**2
        V1 = np.abs(vv2)/2
        if (vv1**2 > vv2**2):
            V0 = (vv2/2)**2
            V1 = np.abs(vv1)/2
        diff = np.diff(ss)
        tau_last = (tt - ss[n])
        sum_exp = 0
        for i in range(0, n):
            sum_exp += np.exp(-K**2 * np.pi**2 * diff[i])
        const = np.pi * np.sqrt(2*np.pi*diff[0]) * a_sqr

        error = np.exp(V1*n - V0*ss[n] - self.get_lambda(1)*tau_last)
        error *= sum_exp
        error /= (const * tau_last)
        return error


    def calculate_strict_error_bound_array(self, K, x, a, z, v1, v2, s):
        """
        Numpy version of calculating fine error for pdf array
        """
        y = []
        for i in range(x.shape[0]):
            y.append(self.calculate_strict_error_bound(K, x[i], a, z, v1, v2, s))
        return y



    def test_check_pdf_kernel(self, size=100):
        """
        Test to check pdf values
        """
        sv = 0
        sz = 0
        st = 0
        t = 0.1
        err = 1e-4
        x = np.linspace(0.1, 1.0, 10)
        s = np.linspace(0.1, 1.0, 10)
        s_size = s.shape[0]
        K = 6
        for v1 in range(1, 5):
            for v2 in range(1, 5):
                for a in range(1, 5):
                    for z in range(1, 5):
                        for logp in (0, 1):
                            pdf_expected = self.calculate_pdf_array(K, x, a, z, v1, v2, s, s_size, logp, err)
                            pdf_cython = swfpt.pdf_array(x, v1, v2, sv, a, z, sz, s, t, st, err, logp)
                            np.testing.assert_almost_equal(pdf_cython, pdf_expected)

    def test_check_pdf_kernel_error(self, size=100):
        """
        Test to check error bound pdf values
        """
        sv = 0
        sz = 0
        st = 0
        t = 0.1
        err = 1e-4
        x = np.linspace(0.15, 1.05, 10)
        s = np.linspace(0.1, 1.0, 10)
        s_size = s.shape[0]
        K = 6
        logp = 0
        for v1 in range(1, 5):
            for v2 in range(1, 5):
                for a in range(1, 5):
                    for z in range(1, 5):
                        pdf_K = self.calculate_pdf_array(K, x, a, z, v1, v2, s, s_size, logp, err)
                        pdf_expected = self.calculate_pdf_array(10, x, a, z, v1, v2, s, s_size, logp, err)
                        error_exp_crude = self.calculate_crude_error_bound_array(K, x, a, z, v1, v2, s)
                        error_exp_strict = self.calculate_strict_error_bound_array(K, x, a, z, v1, v2, s)
                        error_arr = pdf_expected - pdf_K
                        error = np.abs(error_arr)
                        if (not np.isnan(error).all()):
                            np.testing.assert_array_less(error, error_exp_crude, "Error is not bounded")
                            np.testing.assert_array_less(error, error_exp_strict, "Error is not bounded strictly")
                            #np.testing.assert_array_less(error_exp_crude, error_exp_strict, "Crude error is less than strict error")

if __name__ == "__main__":
    test = TestABParams()
    test.test_check_B()
    test.test_check_Ak0()
    test.test_check_pdf_kernel()
    test.test_check_pdf_kernel_error()
