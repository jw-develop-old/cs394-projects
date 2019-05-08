'''
Created on Feb 13, 2019

@author: sirjwhite
'''

import knn
import csv
import numpy as np
import pandas as pd
import random as r

from sklearn.model_selection import train_test_split

# Module to divide data and house tests called by testknn.py, delegated to knn.py.
# runTest is for the test itself, returning the percent success rate as calculated by knn.py.
def runTest(ts,r,dataset):
	data, inputs, targets, y_test = train_test_split(
	dataset[0], dataset[1], test_size=ts,random_state=r)

	# Parameters that could be varied in other tests.
	k = 2
	model = knn.euclidean

	# The predicting done by the actual model.
	y_pred = knn.knn(data, targets,k,model, inputs)

	percent = 1 - np.mean(y_pred != y_test)
	print("{:.2f}".format(ts))

	return percent

# Calls runTest() several times at test_splits between .05 and .90.
# It also writes the data to .csv files in the directory.
def multiTest(dataset,file):
	val = .01

	# Headings and instructions for writing in.
	fnames = ['test_size','percent']
	w = csv.DictWriter(open(file, 'w', newline=''), delimiter=',',
								quotechar='|', quoting=csv.QUOTE_NONE,fieldnames=fnames)
	w.writeheader()

	#Iterative call to knn function, complete with a randomness factor.
	for ts in np.arange(.95,.98,val):
		w.writerow({'test_size' : "{:.4f}".format(ts),
					'percent' : "{:.5f}".format(
						runTest(ts,0,dataset))})