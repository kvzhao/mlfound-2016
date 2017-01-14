from __future__ import division
import numpy as np

def generate_fake_data(size):
    x = np.random.uniform(-1, 1, size) 
    noise = np.random.uniform(0, 1, size)
    y = [-1 if (i<0 and n>0.2) or (i>=0 and n<=0.2) else 1 for i, n in zip(x, noise)]
    return x, y

def findThreshold(x, y, s):
    N = len(y)
    errors = np.zeros(N+1)

    for t in range(N):
        h = s * np.concatenate([-np.ones(t),  np.ones(N-t)])
        errors[t+1] = np.sum(y != h)

    err = np.min(errors[1:])
    idx = np.argmin(errors)

    x = [x[1], x, x[-1]]
    threshold = np.mean(x[idx:idx+1])
    return err, threshold

def decisionStump(x, y):
    N = len(y)
    errors = []
    threshold = []

    err, t = findThreshold(x, y, -1)
    errors.append(err)
    threshold.append(t)
    err, t = findThreshold(x, y, 1)
    errors.append(err)
    threshold.append(t)

    idx = np.argmin(errors)

    Ein = errors[idx] / float(N)
    theta = threshold[idx]
    s = np.sign(idx - 1.5)
    
    Eout = 0.5 + 0.3 * s * (abs(theta) -1.0)

    return Ein, Eout, s, theta

def multiDecisionStump(x, y):
    dim = x.shape[1]
    #Ein = np.zeros([1, dim])
    #s = np.zeros([1, dim])
    #theta = np.zeros([1, dim])
    Ein = np.zeros(dim)
    s = np.zeros(dim)
    theta = np.zeros(dim)

    for d in range(0, dim):
        x_sorted = np.sort(x[:, d])
        idx = np.argsort(x[:, d])
        y_sorted = y[idx]
        Ein[d], _, s[d], theta[d] = decisionStump(x_sorted, y_sorted)

    Emin = np.min(Ein)
    idx = np.argmin(Ein)
    sign = s[idx]
    theta = theta[idx]

    return Emin, sign, idx, theta
