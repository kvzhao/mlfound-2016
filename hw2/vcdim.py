
import numpy as np
import matplotlib.pyplot as plt

def VC(N, delta, dvc):
    return np.sqrt( (8.0/N)*np.log(4.0*((2.0*N) ** dvc +1)/delta))

def VariantVC(N, delta, dvc):
    return np.sqrt( (16.0/N)*np.log(2.0*((N) ** dvc +1)/np.sqrt(delta)))

def RP(N, delta, dvc):
    return np.sqrt(2.0*np.log(2.0*N*( N**dvc +1 ))/N) + np.sqrt(2.0/N * np.log(1.0/delta)) + 1.0/N

def PVB(N, delta, dvc):
    eps = 5.1
    return np.sqrt( 1.0 * ((2.0 * eps) + np.log((6.0*(2*N)**dvc+1)/delta))/N )

def Dvy(N, delta, dvc):
    eps = 5
    return np.sqrt((4.0*eps*(1+eps) + np.log(4.0*(N*N)**dvc+1)/delta)/(2*N))

dvc = 50
delta = 0.05
N = 10

bound = VC(N, delta, dvc)
print ('VC Bound %f' % bound)
bound = VariantVC(N, delta, dvc)
print ('Variant VC Bound %f' % bound)
bound = RP(N, delta, dvc)
print ('RP Bound %f' % bound)

bound = PVB(N, delta, dvc)
print ('PVB Bound %f' % bound)

bound = Dvy(N, delta, dvc)
print ('Dvy Bound %f' % bound)
