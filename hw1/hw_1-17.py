from mltools.PLA import PLA
import numpy as np
import matplotlib.pyplot as plt

print ('Prob 1-17')
data_path = 'data/hw1_15_train.dat'
pla = PLA()

pla.load_train_data(data_path)
mode = 'zero'
pla.init_weight(mode)
print ('Initialization method: ' + mode)

loop_mode = 'rand_cycle'
print ('We run experiments 2,000 times with random cycle, eta = 0.25')

updates_list = []
for i in range(2000):
	num_of_updates = pla.train(loop_mode, eta = 0.25)
	updates_list.append(num_of_updates)
	pla.init_weight(mode)

print ('Average number of updates: %f' % np.mean(updates_list))