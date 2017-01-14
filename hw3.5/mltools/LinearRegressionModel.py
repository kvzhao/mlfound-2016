import numpy as np
import random
import os

class LinearRegressionModel():
	def __init__(self):
		self.W = []
		self.eta = 0.0

		self.N = 0
		self.train_X = []
		self.train_y = []
		self.test_X  = []
		self.test_y  = []

	def load_train_data(self, input_data_file):
		self.status = 'load_train_data'

		if (os.path.isfile(input_data_file) is not True):
			print ('Please check the train data path')
			return self.train_X, self.train_y
		data = np.loadtxt(input_data_file, dtype='float')
		self.train_X = np.ones(data.shape)
		self.train_X[:,1:] = data[:,:-1]
		self.train_y = data[:, -1]
		self.N = len(self.train_X)
		self.data_flawness = np.zeros(self.N)
		return self.train_X, self.train_y

	def load_test_data(self, input_data_file):
		self.status = 'load_test_data'

		if (os.path.isfile(input_data_file) is not True):
			print ('Please check the test data path')
			return self.test_X, self.test_y

		data = np.loadtxt(input_data_file, dtype='float')
		self.test_X= np.ones(data.shape)
		self.test_X[:,1:] = data[:,:-1]
		self.test_y = data[:, -1]
		return self.test_X, self.test_y

	def init_weight(self, mode='normal'):
		if (self.status != 'load_train_data') and (self.status != 'train'):
			print ('Please load training data first')
			return self.W

		self.updates = 0
		if mode == 'normal':
			self.W = np.random.randn(self.train_X.shape[1])
		elif mode == 'uniform':
			self.W = np.random.rand(self.train_X.shape[1])
		elif mode == 'zero':
			self.W = np.zeros(self.train_X.shape[1])

		self.status = 'inited'

        def compute_wreg(self, X, y, lambda_val):
            inverse_part = np.linalg.inv(np.dot(X.transpose(), X) + lambda_val * np.eye(X.shape[1]))
            Wreg = np.dot(np.dot(inverse_part, X.transpose()), y)
            return Wreg

	def train(self, lambda_val = 0.1):
		if (self.status != 'inited'):
			print ('Plesae initialize weights first')
			return 
		self.status = 'train'
                
                inverse_part = np.linalg.inv(np.dot(self.train_X.transpose(), self.train_X) + lambda_val * np.eye(self.train_X.shape[1]))
                Wreg = np.dot(np.dot(inverse_part, self.train_X.transpose()), self.train_y)
                self.W = Wreg

                err = self.eval(len(self.train_y), self.train_X, self.train_y)
                return err

        def train_on(self, split_range, lambda_val = 0.1):
            X = self.train_X[split_range,:]
            y = self.train_y[split_range]

            self.W = self.compute_wreg(X, y, lambda_val)
            err = self.eval(len(y), X, y)

            return err

        def cross_validate_train(self, folds, lambda_val):
            valset_X=[[] for i in  range(folds)]
            valset_y=[[] for i in range(folds)]
            dtotal = len(self.train_y)
            dnum = dtotal/folds

            for v in range(folds):
                valset_X[v] = (self.train_X[v*dnum:(v+1)*dnum,:])
                valset_y[v] = (self.train_y[v*(dnum):(v+1)*dnum])

            ecv = 0.0
            for v in range(folds):
                X = []
                y = []
                for r in range(folds):
                    if r != v:
                        X.extend(valset_X[r])
                        y.extend(valset_y[r])
                X = np.array(X)
                y = np.asarray(y)
                self.W = self.compute_wreg(X, y, lambda_val)
                ecv += self.eval(dnum, valset_X[v], valset_y[v])
            ecv /= folds
            return ecv

        # general evaluation function for 0/1 error
	def eval(self, data_num, X, y):
		error_num = 0
		for i in range(data_num):
                        yhat = np.dot(self.W, X[i])
                        score = np.sign(yhat)
			error_num = error_num + self.error_(score, y[i])
		error = error_num / float(data_num)
		return error

        def eval_on(self, split_range):
            X = self.train_X[split_range,:]
            y = self.train_y[split_range]
            data_num = len(y)
            return self.eval(data_num, X, y)

        # Evaluate on testing dataset
        def Eout(self):
            dnum = len(self.test_y)
            err = self.eval(dnum, self.test_X, self.test_y)
            return err

        # 0/1 error
	def error_(self, y_prediction, y_truth):
		if y_prediction != y_truth:
			return 1
		else:
			return 0


