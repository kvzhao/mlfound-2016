from __future__ import division
import numpy as np

def VC(N, eps, dvc):
    growth = 4*(2*N)**dvc
    decay = np.exp(-0.125*(eps**2) *N)
    return growth * decay

eps = 0.05
dvc = 10

N = 400000
print ('N = %d, Bound <= %f ' % (N, VC(N=N, eps=eps, dvc=dvc)))
N = 420000
print ('N = %d, Bound <= %f ' % (N, VC(N=N, eps=eps, dvc=dvc)))
N = 440000
print ('N = %d, Bound <= %f ' % (N, VC(N=N, eps=eps, dvc=dvc)))
N = 460000
print ('N = %d, Bound <= %f ' % (N, VC(N=N, eps=eps, dvc=dvc)))
N = 500000
print ('N = %d, Bound <= %f ' % (N, VC(N=N, eps=eps, dvc=dvc)))
N = 600000
print ('N = %d, Bound <= %f ' % (N, VC(N=N, eps=eps, dvc=dvc)))
N = 700000
print ('N = %d, Bound <= %f ' % (N, VC(N=N, eps=eps, dvc=dvc)))
N = 800000
print ('N = %d, Bound <= %f ' % (N, VC(N=N, eps=eps, dvc=dvc)))