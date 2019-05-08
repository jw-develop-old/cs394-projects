'''
Created on Apr 25, 2019

@author: James White
'''

import numpy as np
from cvxopt import matrix,solvers
import kernel
from termcolor import colored
import sys
import svm
from sklearn.model_selection import train_test_split

# File to test many possible orientations for a dataset and output an ideal model.
def run(data,n):

	# Trimming out iris objects that correspond to greater than 1 and converting data.
	targets = np.array([data['target'].tolist()[:n]]).T
	inputs = np.array(data['data'].tolist()[:n])

	# Turning 0's into -1's.
	for i in range(len(targets)):
		if targets[i][0] == 0:
			targets[i][0] = -1



	kernels = [kernel.make_poly_kernel(i) for i in range(2,3)]
	kernels.append(kernel.linear)

	thresholds = [10 ** -i for i in range(0,13)]

	randoms = 8
	splits = []
	for i in range(randoms):
		X_train, X_test, y_train, y_test = train_test_split(inputs,targets,random_state=i)
		splits.append((X_train, X_test, y_train, y_test))

	percent = 0.0
	i = 0
	best = 0
	
	for thrsh in thresholds:
		print("\nCurrent Threshold: ",thrsh)
		for it in range(randoms):
			da = splits[it]

			for k in kernels:			
				i += 1
				info = test(thrsh,k,i,da)

				if info[0] > percent:
					percent = info[0]
					y_pred = info[1]
					sums = info[2]
					best = (sums,y_test,thrsh,k,percent,i)


	print("thrsh:",best[2],"\nk:",best[3],"\n%:",best[4],"\niteration:",best[5])

# Display sums computed for each value before comparing using step function.				
def printSums(sums,y_test):
	print()
	print("Sums: Blue -1 | Green 1 | Red Wrong ")
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
	print()


def test(thrsh,k,i,da):

	print(i,end=" ")

	# Train and predict values.
	classifier = svm.train(da[0],da[2],k,threshold=thrsh)
	info = svm.classify(classifier, da[1],return_sums=True)

	y_pred = info[0]
	sums = info[1]

	# Print percentage success
	percent = 1 - np.mean(y_pred != da[3].T)
	if (percent > .5):
		print(colored("{:.2f}\t".format(percent),'green'),end=" ")
	elif (percent > .01):
		print(colored("{:.2f}\t".format(percent),'blue'),end=" ")	
	else:
		print(colored("{:.2f}\t".format(percent),'red'),end=" ")
	if i % 4 == 0:
		print()

	return percent,y_pred,sums