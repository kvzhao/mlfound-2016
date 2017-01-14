from __future__ import division
import numpy as np
def Ed(N, sigma, d):
    return sigma**2 * (1- (d+1)/N)
for N in range(10,80):
    print ('N = %d, Ein = %f' % (N, Ed(N, 0.1, 8)))