'''
Created on Mar 18, 2019

@author: James White
'''

import numpy as np
import collections
from sklearn.neural_network import MLPClassifier
from random import uniform
import perceptron as pc

#Primary classification unit
class Classifier:
	# Constructor
	def __init__(self,M,data,targets,pct):
		self.M = M
		self.data = data
		self.targets = targets
		self.pct = pct

	def predict(self,inputs):
		toReturn = []
		for y in self.pct[0](inputs):
			if y < .5:
				toReturn.append(0)
			else:
				toReturn.append(1)
		return toReturn

def printEachIteration(pct,data,i):
	# Projected values.
	print("Predictions: ",pct[0](data),"\n")

	# Iteration count.
	print("Iteration " + str(i) + ":")

	# Perceptron info.
	print("Perceptrons:")
	for p in pct:
		print(str(p))


def instantiate_Classifier(M,data,targets):

	# Data and targets.
	x = data
	y = targets

	# Data files.
	print("Data:")
	print(data.T)
	print(targets.T)

	pct = [pc.initialize_perceptron(len(data.T)) for i in range(M)]

	# Repeat until termination condition.
	for i in range(50):

		## May be commented out to prevent clutter ##
		#printEachIteration(pct,data,i)

		# Backpropogation: For each data point and corresponding target.
		# Previous layer, in this case, the data.

		if len(pct) == 1:
			l_0 = x

			# y val -> output.
			l_1 = pct[0](x)

			# Error.
			error = y - l_1

			# Delta.
			delta = error * pc.sigmoid_deriv(l_1)

			# Updating the weights.
			pct[0].weights += np.dot(l_0.T,delta)

		else:
			# Previous layer, in this case, the data.
			l_0 = x

			# y val -> output.
			l_1 = [pct[i](x) for i in range(len(pct))]

			# Error.
			error = [y - l_1[i] for i in range(len(pct))]

			# Delta.
			delta = [error[i] * pc.sigmoid_deriv(l_1[i]) for i in range(len(pct))]

			# Updating the weights.
			for i in range(len(pct)):
				pct[i].weights += np.dot(l_0.T,delta[i])

		# Output, originally used for print statements.
#		y_vals = [pct[i](x) for i in range(len(pct))]

			# 	# Compute all z
			# z_l = []
			# for p in self.pct:
			# 	z_l.append(np.dot([1]+self.data,p.weights))

			# # Compute all y
			# y_vals = self.pred(self.data)

			# # For each output unit, calculate the error.
			# d_y = []
			# for y in y_vals:
			# 	toAdd = y*(1-y)*(t-y)
			# 	d_y.append(toAdd)

			# # Calculate derivative of activation function.
			# d_z = []
			# for z in z_l:
			# 	toAdd = pc.sigmoid(z)*(1-pc.sigmoid(z))
			# 	d_z.append(toAdd)
				
			# # For each perceptron
			# for index in range(len(d_z)):

			# 	toAdd = .1*d_z[index]*z_l[index]
			# 	w_ylk.append(toAdd)

			# # Adjusting each weight of each perceptron.
			# for y in y_vals:
				
			# 	# For each perceptron
			# 	for index in len(d_z):

			# 		# For each weight
			# 		for w in self.pct[z]:
			# 			w += .1 * d_z[index]

	toReturn = Classifier(M,data,targets,pct)

	return toReturn

# Primary mlp function signatures.
def train(M,data,targets):
	print("\nTraining ...")
	return instantiate_Classifier(M,data,targets)

def classify(cfr,inputs):
	print("Classifying ...")
	toReturn = cfr.predict(inputs)
	return toReturn