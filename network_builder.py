import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, activation
from tflearn.layers.conv import conv_2d, max_pool_2d, residual_block, global_avg_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum
from constants import *

class NetworkBuilder:
	def __init__(self):
		pass

	def build_alexnet(self):
		# Smaller Alexnet
		img_aug = tflearn.ImageAugmentation()
		img_aug.add_random_flip_leftright()
		img_aug.add_random_crop([SIZE_FACE, SIZE_FACE], padding = 4)
		print('[+] Building Alexnet')
		self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1], 
			data_augmentation = img_aug)
		self.network = conv_2d(self.network, 64, 5, activation = 'relu')
		self.network = max_pool_2d(self.network, 3, strides = 2)
		self.network = conv_2d(self.network, 64, 5, activation = 'relu')
		self.network = max_pool_2d(self.network, 3, strides = 2)
		self.network = conv_2d(self.network, 128, 4, activation = 'relu')
		self.network = dropout(self.network, 0.3)
		self.network = fully_connected(self.network, 3072, activation = 'relu')
		self.network = fully_connected(self.network, len(EMOTIONS), 
			activation = 'softmax')
		self.network = regression(self.network, optimizer = 'momentum', 
			loss = 'categorical_crossentropy')
		self.model = tflearn.DNN(self.network, 
			checkpoint_path = SAVE_DIRECTORY + RUN_NAME, 
			max_checkpoints = 1, tensorboard_verbose = 0)
		return self.model

	def build_vgg(self):
		# Smaller VGG
		img_aug = tflearn.ImageAugmentation()
		img_aug.add_random_flip_leftright()
		img_aug.add_random_crop([SIZE_FACE, SIZE_FACE], padding = 4)
		print('[+] Building VGG')
		self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1], 
			data_augmentation = img_aug)
		self.network = conv_2d(self.network, 64, 3, activation = 'relu')
		self.network = conv_2d(self.network, 64, 3, activation = 'relu')
		self.network = max_pool_2d(self.network, 2, strides = 2)

		self.network = conv_2d(self.network, 128, 3, activation = 'relu')
		self.network = conv_2d(self.network, 128, 3, activation = 'relu')
		self.network = max_pool_2d(self.network, 2, strides = 2)

		self.network = conv_2d(self.network, 256, 3, activation = 'relu')
		self.network = conv_2d(self.network, 256, 3, activation = 'relu')
		self.network = conv_2d(self.network, 256, 3, activation = 'relu')
		self.network = max_pool_2d(self.network, 2, strides = 2)

		self.network = conv_2d(self.network, 512, 3, activation = 'relu')
		self.network = conv_2d(self.network, 512, 3, activation = 'relu')
		self.network = conv_2d(self.network, 512, 3, activation = 'relu')
		self.network = max_pool_2d(self.network, 2, strides = 2)

		self.network = conv_2d(self.network, 512, 3, activation = 'relu')
		self.network = conv_2d(self.network, 512, 3, activation = 'relu')
		self.network = conv_2d(self.network, 512, 3, activation = 'relu')
		self.network = max_pool_2d(self.network, 2, strides = 2)

		self.network = dropout(self.network, 0.3)
		self.network = fully_connected(self.network, 3072, activation = 'relu')
		self.network = fully_connected(self.network, len(EMOTIONS), 
			activation = 'softmax')
		self.network = regression(self.network, optimizer = 'rmsprop',
			loss = 'categorical_crossentropy', learning_rate = 0.0001)
		self.model = tflearn.DNN(self.network, 
			checkpoint_path = SAVE_DIRECTORY + RUN_NAME, 
			max_checkpoints = 1, tensorboard_verbose = 0)
		return self.model

	def build_resnet(self):
		# Adapted resnet
		# Adaptation did not work
		print('[+] Building Resnet')
		# Residual blocks
		n = 5
		self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1])
		self.network = conv_2d(self.network, SIZE_FACE / 2, 3, 
			regularizer = 'L2', weight_decay = 0.0001)
		self.network = residual_block(self.network, n, SIZE_FACE / 2)
		self.network = residual_block(self.network, 1, SIZE_FACE, 
			downsample = True)
		self.network = residual_block(self.network, n-1, SIZE_FACE)
		self.network = residual_block(self.network, 1, SIZE_FACE * 2, 
			downsample = True)
		self.network = residual_block(self.network, n-1, SIZE_FACE * 2)
		self.network = batch_normalization(self.network)
		self.network = activation(self.network, 'relu')
		self.network = global_avg_pool(self.network)
		self.network = fully_connected(self.network, len(EMOTIONS), 
			activation = 'softmax')
		self.momentum = Momentum(0.1, lr_decay = 0.1, 
			decay_step = 32000, staircase = True)
		self.network = regression(self.network, optimizer = self.momentum, 
			loss = 'categorical_crossentropy')
		self.model = tflearn.DNN(self.network,
			checkpoint_path = SAVE_DIRECTORY + RUN_NAME, 
			max_checkpoints = 1, tensorboard_verbose = 0)
		return self.model
