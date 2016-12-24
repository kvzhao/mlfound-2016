from __future__ import division
import numpy as np
import random
import matplotlib.pyplot as plt

from decision_stump import *
print ('hw 2-19')

data = np.loadtxt('data/hw2_train.dat', dtype='float')
train_X = data[:,:-1]
train_y = data[:,-1]

data = np.loadtxt('data/hw2_test.dat', dtype='float')
test_X = data[:,:-1]
test_y = data[:,-1]

Ein, s, idx, theta = multiDecisionStump(train_X, train_y)
print ('Ein: %f, with s = %d and theta = %f' % (Ein, s, theta))

print ('hw 2-20')
tx = test_X[:, idx]
ty = test_y
pred = s * np.sign(tx - theta)
Eout = np.sum(pred != ty)/len(ty)
print ('Eout: %f' % Eout)
