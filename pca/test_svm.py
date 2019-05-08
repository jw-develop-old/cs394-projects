'''
Created on Apr 5, 2019

@author: James White
'''

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import svm
import info_svm
import multi_test
import kernel
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from termcolor import colored

# Primary testing template function.
def mainTest(X_train, X_test, y_train, y_test,k):

	# If "info" argv was passed in.
	if "info" in sys.argv:
		s = info_svm
	else:
		s = svm

	# Train and predict values.
	classifier = s.train(X_train,y_train,k,C=None)
	info = s.classify(classifier, X_test,return_sums=True)

	y_pred = info[0]
	sums = info[1]

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

	return percent

# Helper function for 3 simple tests.
def basicTest(data,targets):
	mainTest(data,data,targets,targets,kernel.linear)

# Similar to recommended data, easily seperable example data.
def testOne():
	t_d = np.array([ [2,1,4],[3,2,5],[4,3,9],[-1,-3,-2],[-2,-4,-5],[-3,-5,-7]])
	t_t = np.array([[1,1,1,-1,-1,-1]]).T

	print("--Test 1--")
	basicTest(t_d,t_t)

def testTwo():
	and_d = np.array([[2,1],[1,2],[-1,-1],[-2,-2]])
	and_t = np.array([[1,1,-1,-1]]).T

	print("--Test 2--")
	basicTest(and_d,and_t)

def testThree():
	xor_d = np.array([ [0,0],[1,1],[0,1],[1,0] ])
	xor_t = np.array([[-1,-1,1,1]]).T

	print("--Test 3--")
	basicTest(xor_d,xor_t)

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

def wineTest():
	advTest(load_wine(),130,kernel.linear)

def irisTest():
	advTest(load_iris(),100,kernel.make_poly_kernel(3))


# Main method. Makes calls based on parameters.
if __name__ == '__main__':

	print("Args:",str(sys.argv[1:]))

	if "multi" in sys.argv:
		if "wine" in sys.argv:
			multi_test.run(load_wine(),130)
		else:
			multi_test.run(load_iris(),100)
	else:
		if "1" in sys.argv:
			testOne()
		elif "2" in sys.argv:
			testTwo()
		elif "3" in sys.argv:
			testThree()
		elif "wine" in sys.argv:
			wineTest()
		else:
			irisTest()