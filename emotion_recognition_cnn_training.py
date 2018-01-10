from __future__ import division, absolute_import
import numpy as np
from dataset_loader import DatasetLoader
from network_builder import NetworkBuilder
from constants import *
from os.path import isfile, join
import random
import sys

class EmotionRecognition:

	def __init__(self):
		self.dataset = DatasetLoader()
		self.networkbuilder = NetworkBuilder()

	def build_network(self):
		self.model = self.networkbuilder.build_vgg()
		# self.load_model()

	def load_saved_dataset(self):
		self.dataset.load_from_save()
		print('[+] Dataset found and loaded')

	def start_training(self):
		self.load_saved_dataset()
		self.build_network()
		if self.dataset is None:
			self.load_saved_dataset()
		print('[+] Training network')
		self.model.fit(
			self.dataset.images, self.dataset.labels, 
			validation_set = (self.dataset.images_test, self.dataset.labels_test),
			n_epoch = 100, batch_size = 100, shuffle = True, show_metric = True, 
			snapshot_step = 200, snapshot_epoch = True, 
			run_id = RUN_NAME)

	def predict(self, image):
		if image is None:
			return None
		image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
		return self.model.predict(image)

	def save_model(self):
		self.model.save(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
		print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)

	def load_model(self):
		self.model.load(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
		print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)

def show_usage():
	print('[!] Usage: python emotion_recognition.py')
	print('\t emotion_recognition.py train \t Trains and saves model with saved dataset')
	print('\t emotion_recognition.py poc \t Launch the proof of concept')

if __name__ == '__main__':
	if len(sys.argv) <= 1:
		show_usage()
		exit()

	network = EmotionRecognition()
	if sys.argv[1] == 'train':
		network.start_training()
		network.save_model()
	elif sys.argv[1] == 'poc':
		import poc
	else:
		show_usage()