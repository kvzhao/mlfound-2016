import numpy as np
import random
import os

class LogisticRegressionModel():
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

	def theta(self, s):
		return 1 / (1 + np.exp((-1) * s))

	def compute_gradient(self, X, Y, W):
		if type(Y) is np.ndarray:
			data_num = len(Y)
		else:
			data_num = 1
		a = self.theta((-1) * Y * np.dot(W, X.transpose())) *(-1)*Y
		b = np.inner(a, X.transpose())
		grad = b/data_num
		return grad


	def train(self, epoch=2000, eta = 0.01, solver_type='vanilla'):
		if (self.status != 'inited'):
			print ('Plesae initialize weights first')
			return 
		self.status = 'train'
		self.eta = eta

		for i in range(epoch):
			grad = 0.0
			if solver_type == 'stochastic':
                                r = np.random.randint(self.N)
				x = self.train_X[r]
				y = self.train_y[r]
				grad = self.compute_gradient(x, y, self.W)
			elif solver_type == 'vallina':
				grad = self.compute_gradient(self.train_X, self.train_y, self.W)
				# Update in the end
			self.W = self.W - (self.eta * grad)

	def eval(self):
		data_num = len(self.test_y)
		error_num = 0
		for i in range(data_num):
			score = self.theta(np.inner(self.test_X[i], self.W))
			if score >= 0.5:
				score = 1.0
			else:
				score = -1.0
			error_num = error_num + self.error_(score, self.test_y[i])
		error = error_num / float(data_num)
		return error

	def error_(self, y_prediction, y_truth):
		if y_prediction != y_truth:
			return 1
		else:
			return 0



