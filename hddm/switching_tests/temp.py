import numpy as np
import matplotlib.pyplot as plt

import swfpt


j = 1
k = 1
dv = 1
sign = 1
expected = 1.675832495
alpha = (-1)**(sign + 1) * dv
print (alpha)

B = 4 * np.pi**2 * j * k * alpha;
print (B)


B *= (-1)**(j + k) * np.exp(alpha) - 1
print (B)

B /= (alpha**4 + 2 * np.pi**2 * alpha**2 * (j**2 + k**2) +
    np.pi**4 * (j**2 - k**2)**2)
print (B)

m = swfpt.BB(k, j, dv, sign)
print (m);
print (B);
