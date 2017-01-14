from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def theta(s):
    return 1.0 / (1.0 + np.exp((-1.0) * s))
def err_a (x):
    x = 1-x
    return np.max([0, x], axis=0)
def err_b (x):
    x = 1-x
    return np.max([0, x])**2
def err_c (x):
    x = 1-x
    return np.max([0, x])
def err_d (x):
    return theta(x)
def err_e (x):
    return np.exp(-x)
    
wx = np.linspace(-1,1,10)

a = []
b = []
c = []
d = []
e = []
for x in wx:
    a.append(err_a(x))
    b.append(err_b(x))
    c.append(err_c(x))
    d.append(err_d(x))
    e.append(err_e(x))

plt.step([0,1], [0,1], linewidth=2)
plt.plot(wx, a, '-^', linewidth=2)
plt.plot(wx, b, '-v',linewidth=2)
plt.plot(wx, c, '->', linewidth=2)
plt.plot(wx, d, '-<', linewidth=2)
plt.plot(wx, e, '-*', linewidth=2)

plt.grid()
plt.legend(['sign function', 'error A', 'error B', 'error C', 'error D', 'error E'])
plt.ylabel('err')
plt.xlabel('$w^Tx$')
plt.title('Uppder Bounds')
plt.savefig('results/figure_2.png')
plt.show()