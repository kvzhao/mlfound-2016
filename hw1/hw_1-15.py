from mltools.PLA import PLA
import numpy as np

print ('Prob 1-15')
data_path = 'data/hw1_15_train.dat'
pla = PLA()

pla.load_train_data(data_path)
mode = 'zero'
pla.init_weight(mode)
print ('Initialization method: ' + mode)

num_of_updates = pla.train()
print ('Updates Number: %d' % num_of_updates) 
blame_index = np.argmax(pla.data_flawness)
print ('Most frequent update: %d' % blame_index)