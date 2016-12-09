import numpy as np
import random
import os

class PLA():
	def __init__(self):
		self.status = 'empty'
		self.loop_mode = 'naive'

		self.W = []
		self.eta = 1.0
		self.updates = 0
		self.data_flawness = []

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
		self.train_X[:,1:] = data[:,:4]
		self.train_y = data[:, 4]
		self.data_flawness = np.zeros(len(self.train_X))
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
		return self.test_X, self.test_y

	def init_weight(self, mode='normal'):
		if (self.status != 'load_train_data') and (self.status != 'train'):
			print ('Please load training data first')
			return self.W

		if mode == 'normal':
			self.W = np.random.randn(self.train_X.shape[1])
		elif mode == 'uniform':
			self.W = np.random.rand(self.train_X.shape[1])
		elif mode == 'zero':
			self.W = np.zeros(self.train_X.shape[1])

		self.status = 'inited'

	def train(self, loop_mode = 'naive'):
		if (self.status != 'inited'):
			print ('Plesae initialize weights first')
			return 

		self.loop_mode = loop_mode

		while True:
			need_update = False
			if self.loop_mode == 'naive':
				for i, x in enumerate(self.train_X):
					y = self.train_y[i]
					if np.sign(np.dot(self.W, x)) != y:
						self.updates += 1
						self.W = self.W + self.eta * x * y
						self.data_flawness[i] += 1
						need_update = True
				if not need_update:
					break
		return self.updates