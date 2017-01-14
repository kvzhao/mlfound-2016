from mltools.LinearRegressionModel import LinearRegressionModel
import numpy as np

def main():
	pass

if __name__ == '__main__':
	train_data_path = 'data/hw4_train.dat'
	test_data_path = 'data/hw4_test.dat'

	LR = LinearRegressionModel()
	LR.load_train_data(train_data_path)

	mode = 'zero'
	LR.init_weight(mode)
	print ('Initialization method: ' + mode)

        lambda_val = 1.126
	ein = LR.train(lambda_val= lambda_val)
	LR.load_test_data(test_data_path)
	eout = LR.Eout()

	print ('Ein: %f, Eout: %f' % (ein, eout))