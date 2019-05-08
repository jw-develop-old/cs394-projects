'''
Created on Mar 18, 2019

@author: James White
'''

import mlp
import csv
import numpy as np
import pandas as pd
import random as r

from sklearn.model_selection import train_test_split

def runTest(dataset,file):
#	for r in range(0,10):
	test(.2,0,dataset)

# Module to divide data and house tests called by testknn.py, delegated to knn.py.
# runTest is for the test itself, returning the percent success rate as calculated by knn.py.
def test(ts,r,dataset):
	X_train, X_test, y_train, y_test = train_test_split(
	dataset[0], dataset[1], test_size=ts,random_state=r)

	# Perceptron count
	p_count = 1

	illegal = mlp.illegal_train(p_count, X_train, y_train,r)
	y_pred = mlp.illegal_classify(illegal, X_test)
	percent = 1 - np.mean(y_pred != y_test)
	print("\n{:.2f}".format(percent))

	# Training the classifier.
	classifier = mlp.train(p_count, X_train, y_train)
	y_pred = mlp.classify(classifier, X_test)

	percent = 1 - np.mean(y_pred != y_test)
	print("\n{:.2f}\n".format(percent))

	return percent

"""
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
"""