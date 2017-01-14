import numpy as np
import random
import matplotlib.pyplot as plt

from decision_stump import *
print ('hw 2-17')

rounds = 5000
N = 10

EinList = []
EoutList = []
for i in range(rounds):
    x, y = generate_fake_data(N)
    Ein, Eout, s, theta = decisionStump(x, y)
    EinList.append(Ein)
    EoutList.append(Eout)

Ein = np.mean(EinList)
Eout = np.mean(EoutList)
print ('Ein: %f, Eout: %f' % (Ein, Eout))


bins= 5
hist, bin_edges = np.histogram(EinList, bins=bins)
plt.hist(EinList, bin_edges)

plt.xlabel('Ein')
plt.ylabel('Frequency')
plt.title('Histrogram of Ein Distribution')
plt.grid(True)
plt.savefig('results/OneDimEinDist.png')
plt.show()

