'''
Created on Feb 19, 2019

@author: sirjwhite

Adapted from dataset/explanations found at:
https://www.getdrip.com/forms/798345213/submissions

'''

from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# File to build the pictures dataset and supply its features to testknn.py.
# Split into colors.
def extract_colors(image):
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
		np.std(G), np.std(B)]
	return features

# Loads the dataset of the scenes.
def load_scenes():

	# Parse the arguments based on the name of the folder.
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", type=str, default="3scenes",
		help="path to directory containing the '3scenes' dataset")
	ap.add_argument("-m", "--model", type=str, default="knn",
		help="type of python machine learning model to use")
	args = vars(ap.parse_args())

	#Get paths of image.
	imagePaths = paths.list_images(args["dataset"])
	data = []
	labels = []

	#Build 
	for imagePath in imagePaths:
		image = Image.open(imagePath)
		features = extract_colors(image)
		data.append(features)

		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

	le = LabelEncoder()
	labels = le.fit_transform(labels)

	return (data,labels)