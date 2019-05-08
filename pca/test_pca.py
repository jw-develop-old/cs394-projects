'''
Created on Apr 27, 2019

@author: James White
'''

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import pca
import svm
import info_svm
import multi_test
import kernel
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from termcolor import colored

def printResults(sums,y_test,y_pred):
	# Display sums computed for each value before comparing using step function.
	print("\nSums: Blue -1 | Green 1 | Red Wrong ")
	for j in range(len(sums)):
		sum = sums[j]
		if sum > 0 and y_test[j] == 1:
			print(colored("{:.2f}".format(sum),'green'),end=' ')
		elif sum < 0 and y_test[j] == -1:
			print(colored("{:.2f}".format(sum),'blue'),end=' ')
		else:
			print(colored("{:.2f}".format(sum),'red'),end=' ')
		if j % 5 == 4:
			print()

	# Print percentage success
	percent = 1 - np.mean(y_pred != y_test.T)
	print("\n{:.2f}\n".format(percent))

# Similar to recommended data, easily seperable example data.
def testOne():
	t_d = np.array([ [2,1,4,3],[3,2,5,7],[13,3,9,25],[-1,-3,-2,-1],[-2,-4,-5,-7],[-3,-5,-7,-9]])
	t_t = np.array([[1,1,1,-1,-1,-1]]).T

# Primary testing template function.
def mainTest(X_train, X_test, y_train, y_test,k):
	print("--Test 1--")

	M = 3

	# PCA Work
	print("\nTraining data:")
	comp_1 = pca.pca(X_train,M)
	X_train_t = pca.transform(X_train, comp_1)

	print("\nTesting data:")
	comp_2 = pca.pca(X_test,M)
	X_test_t = pca.transform(X_test, comp_2)

	# Print base results.
	print("\nBefore PCA - Dim ",len(X_train[0]))

	classifier = svm.train(X_train,y_train,k,C=None)
	info = svm.classify(classifier, X_test,return_sums=True)

	printResults(info[1],y_test,info[0])

	# Print transformed results.
	print("After PCA - Dim ",M)
	X_train = X_train_t
	X_test = X_test_t

	classifier = svm.train(X_train,y_train,k,C=None)
	info = svm.classify(classifier, X_test,return_sums=True)

	printResults(info[1],y_test,info[0])

def advTest(data,n,k):

	# Trimming out iris objects that correspond to greater than 1 and converting data.
	targets = np.array([data['target'].tolist()[:n]]).T
	inputs = np.array(data['data'].tolist()[:n])

	# Turning 0's into -1's.
	for i in range(len(targets)):
		if targets[i][0] == 0:
			targets[i][0] = -1

	X_train, X_test, y_train, y_test = train_test_split(inputs,targets)
	mainTest(X_train, X_test, y_train, y_test,k)

def irisTest():
	advTest(load_iris(),100,kernel.make_poly_kernel(3))

if __name__ == '__main__':
	irisTest()