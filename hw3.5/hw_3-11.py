from mltools.LogisticRegressionModel import LogisticRegressionModel
import numpy as np

def main():
	pass

if __name__ == '__main__':
	train_data_path = 'data/hw3_train.dat'
	test_data_path = 'data/hw3_test.dat'

	LR = LogisticRegressionModel()
	LR.load_train_data(train_data_path)

	mode = 'zero'
        solver = 'vallina'
	LR.init_weight(mode)
	print ('Initialization method: ' + mode)
	print ('Solver type: ' + solver)

	num_of_updates = LR.train(epoch=2000, eta=0.001)

	LR.load_test_data(test_data_path)
	error = LR.eval()
	print ('Error: %f' % error)
        print (LR.W)