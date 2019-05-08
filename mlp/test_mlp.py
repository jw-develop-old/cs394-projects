'''
Created on Mar 18, 2019

@author: James White
'''

from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
import mlp
import csv
import numpy as np
import pandas as pd
import random as r
from sklearn.model_selection import train_test_split

# Module to run tests and simulate the data.
def test(r,data,target,p_count):
	# X_train, X_test, y_train, y_test = train_test_split(
	# dataset[0], dataset[1], test_size=ts,random_state=r)

	X_train = data
	X_test = data
	y_train = target
	y_test = target

	# Train and predict values.
	classifier = mlp.train(p_count, X_train, y_train)
	y_pred = mlp.classify(classifier, X_test)

	# Print the results.
	print(y_pred)
	print(y_test.T)

	# Print percentage success.
	percent = 1 - np.mean(y_pred != y_test.T)
	print("\n{:.2f}\n".format(percent))

	return percent

def runTest(dataset,target,a,b):

	test(0,dataset,target,1)
	test(0,dataset,target,5)

# Main method. Imports data and sets up for the tests.
if __name__ == '__main__':

	t_d = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
	t_t = np.array([[0,0,1,1]]).T

	print(t_d.shape)
	print(t_t.shape)

	a_d = 'triples_results.txt'
	a_i = 'triples_iterations.txt'

	print("--Test 1--")
	runTest(t_d,t_t,a_d,a_i)

	and_d = np.array([ [1,1],[0,0],[0,1],[1,0] ])
	and_t = np.array([[1,0,0,0]]).T

	b_d = 'and_results.txt'
	b_i = 'and_iterations.txt'

	print("--Test 2--")
	runTest(and_d,and_t,b_d,b_i)

	xor_d = np.array([ [0,0],[1,1],[0,1],[1,0] ])
	xor_t = np.array([[0,0,1,1]]).T

	c_d = 'xor_results.txt'
	c_i = 'xor_iterations.txt'

	print("--Test 3--")
	runTest(xor_d,xor_t,c_d,c_i)