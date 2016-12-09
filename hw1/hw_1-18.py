from mltools.PocketPLA import PocketPLA
import numpy as np

print ('Prob 1-18')
train_data_path = 'data/hw1_18_train.dat'
test_data_path = 'data/hw1_18_test.dat'

pla = PocketPLA()

pla.load_train_data(train_data_path)
mode = 'zero'
pla.init_weight(mode)
print ('Initialization method: ' + mode)

num_of_iters = 50
num_of_updates = pla.train(num_of_iters)

print('Pocket Updates %d times within %d iterations.' % (num_of_updates, num_of_iters))

pla.load_test_data(test_data_path)
testnum = pla.N_test
errs = pla.test()
print ('Get %d errors on %d testing dataset, accuracy: %f percents' % (errs, testnum, 1.0-errs/float(testnum)))

updates_list = []
for i in range(2000):
	num_of_updates = pla.train(num_of_iters)
	updates_list.append(num_of_updates)
	pla.init_weight(mode)

print ('Average number of updates: %f' % np.mean(updates_list))
