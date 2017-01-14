from __future__ import division
import numpy as np

def gradient(x, eta):
    u = x[0]
    v = x[1]
    du = np.exp(u) + v*np.exp(u*v) +2*u-2*v-3
    dv = 2*np.exp(2*v) + u*np.exp(u*v) -2*u+4*v-2
    grad = np.asarray([eta* du, eta*dv])
    return grad

def update(x, eta):
    grad = gradient(x, eta)
    newx = x - grad
    return newx

eta = 0.01
x = [0,0]
for i in range(5):
    print ('step %d' % (i +1))
    x = update(x, eta)
    print (x)