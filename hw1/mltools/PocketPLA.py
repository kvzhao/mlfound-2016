import numpy as np
import random
import os

class PocketPLA():
	def __init__(self):
		self.status = 'empty'
		self.loop_mode = 'rand_pick'

		self.W = []
		self.pocket_W = []
		self.eta = 1.0
		self.updates = 0
		self.data_flawness = []
		self.put_in_pocket_times = 0

		self.N_train = 0
		self.dim = 0
		is_train_loaded = False
		self.train_X = []
		self.train_y = []

		is_test_loaded = False
		self.N_test = 0
		self.test_X  = []
		self.test_y  = []

	def _error(self, W, X, y):
		errs = 0
		for i, x in enumerate(X):
			if (np.sign(np.dot(W,x)) != y[i]):
				errs += 1
		return errs

	def load_train_data(self, input_data_file):
		self.status = 'load_train_data'

		if (os.path.isfile(input_data_file) is not True):
			print ('Please check the train data path')
			return self.train_X, self.train_y
		data = np.loadtxt(input_data_file, dtype='float')
		self.train_X = np.ones(data.shape)
		self.train_X[:,1:] = data[:,:4]
		self.train_y = data[:, 4]
		self.N_train = len(self.train_X)
		self.dim = self.train_X.shape[1]
		self.data_flawness = np.zeros(self.N_train)
		self.is_train_loaded = True
		return self.train_X, self.train_y

	def load_test_data(self, input_data_file):
		self.status = 'load_test_data'

		if (os.path.isfile(input_data_file) is not True):
			print ('Please check the test data path')
			return self.test_X, self.test_y

		data = np.loadtxt(input_data_file, dtype='float')
		self.test_X= np.ones(data.shape)
		self.test_X[:,1:] = data[:,:4]
		self.test_y = data[:, 4]
		self.N_test = len(self.test_X)
		self.is_test_loaded = True
		return self.test_X, self.test_y

	def init_weight(self, mode='normal'):
		if not self.is_train_loaded:
			print ('Please load training data first')
			return self.W

		self.updates = 0
		self.put_in_pocket_times = 0
		if mode == 'normal':
			self.W = np.random.randn(self.dim)
			self.pocket_W = np.random.randn(self.dim)
		elif mode == 'uniform':
			self.W = np.random.rand(self.dim)
			self.pocket_W = np.random.rand(self.dim)
		elif mode == 'zero':
			self.W = np.zeros(self.dim)
			self.pocket_W = np.zeros(self.dim)

		self.status = 'inited'

	def train(self, itererations, loop_mode = 'rand_pick', eta = 1.0):

		self.status = 'train'
		self.loop_mode = loop_mode

		self.eta = eta

		for it in range(itererations):
			r = np.random.randint(self.N_train)
			x = self.train_X[r]
			y = self.train_y[r]
			if np.sign(np.dot(self.W, x)) != y:
				self.updates += 1
				self.W = self.W + self.eta * x * y
				self.data_flawness[r] += 1
				# pocket algorithm
				if self._error(self.W, self.train_X, self.train_y) < self._error(self.pocket_W, self.train_X, self.train_y):
					self.pocket_W = self.W
					self.put_in_pocket_times += 1
		return self.put_in_pocket_times

	def test(self, weight_type='pocket'):
		if self.is_test_loaded:
			if weight_type == 'pocket':
				errs = self._error(self.pocket_W, self.test_X, self.test_y) 
			elif weight_type == 'onhand':
				errs = self._error(self.W, self.test_X, self.test_y) 
			return errs