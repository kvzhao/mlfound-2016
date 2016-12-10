from mltools.PocketPLA import PocketPLA
import numpy as np
import matplotlib.pyplot as plt

print ('Prob 1-19')
train_data_path = 'data/hw1_18_train.dat'
test_data_path = 'data/hw1_18_test.dat'

pla = PocketPLA()

pla.load_train_data(train_data_path)
mode = 'zero'
pla.init_weight(mode)
print ('Initialization method: ' + mode)

num_of_iters = 100
num_of_updates = pla.train(num_of_iters)

print('Pocket Updates %d times within %d iterations.' % (num_of_updates, num_of_iters))

pla.load_test_data(test_data_path)
testnum = pla.N_test
errs = pla.test()
print ('Get %d errors on %d testing dataset, accuracy: %f percents' % (errs, testnum, 1.0-errs/float(testnum)))

errs_list = []
for i in range(2000):
	num_of_updates = pla.train(num_of_iters)
	errs = pla.test()
	errs /= float(testnum)
	errs = int(errs*100)
	errs_list.append(errs)
	pla.init_weight(mode)

binwidth = 5
plt.hist(errs_list, bins=range(min(errs_list), max(errs_list) + binwidth, binwidth), color='g')
plt.xlabel('Error Rate')
plt.ylabel('Frequency')
plt.title('Histrogram of Pocket-PLA Average Error-Rate (Updates: ' + str(num_of_iters) +' $\eta=1.0$)')
plt.grid(True)
plt.savefig('results/hist-errate-pocketpla-itr='+str(num_of_iters)+'-eta=1.png')

print ('Average Error Rate after 2000 experiments: %d' % np.mean(errs_list))
