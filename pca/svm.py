'''
Created on Apr 5, 2019

@author: James White
'''

import numpy as np
from cvxopt import matrix,solvers
import kernel
from termcolor import colored
import sys

class Classifier:

	# Constructor
	def __init__(self,a,l_m,b,t,k,data):

		# List of support vector indices.
		self.a = a
		# LaGrange Multipliers.
		self.l_m = l_m
		# Intercept
		self.b = b
		# Targets that correspond to support vectors.
		self.t = t
		# Kernel matrix from x to x.
		self.k = k
		# Training data references.
		self.data = data

	# Predictive method.
	def predict(self,inputs,return_sums):
		toReturn = []
		sums = []

		# To be sum over all nonzero support vectors.
		for x in inputs:
			sum = 0
			for i in self.a:
				sum += (self.l_m[i][0]*self.t[i][0]*self.k(self.data[i],x)+self.b)

			# Step function applied.
			toReturn.append(-1 if sum < 0 else 1)
			sums.append(sum)
		if return_sums:
			return (toReturn,sums)
		else:
			return toReturn

def train(data,t,k,C=None,threshold=1e-5):
	dim = len(data)
	dims = (dim,dim)

	# Compute kernel matrix K
	k_matr = []
	for i in range(dim):
		k_matr.append([])
		for j in range(dim):
			k_matr[i].append(k(data[i],data[j]))

	# Compute P = ttTK
	P = t*t.T*k_matr
	P = matrix(P,dims,'d')

	# Assemble q vector of -1s
	q = matrix(-1,(dim,1),'d')

	# Assemble A matrix of ti along diagonal
	A = np.array([t[i] for i in range(dim)])
	A = matrix(A,(1,dim),'d')

	# Assemble G matrix of -1s along diagonal.
	G = np.zeros(dims)
	for i in range(dim):
		G[i][i] = -1
	G = matrix(G,dims,'d')

	# Assemble h vector of 0s.
	h = matrix(0,(dim,1),'d')
	
	# Dummy zeroes for b.
	b = matrix(0.0)

	# Compute a vector by feeding P,q,G,h,A, and b=0 into QP solver.
	# Format: sol = solvers.qp(P,q,G,h,A,b)
	solvers.options['show_progress'] = False
	a = solvers.qp(P,q,G,h,A,b)

	# LaGrange Multipliers
	l_m = np.array(a['x'])

	# Find indices of support vector indices from a that are not zero.
	s_v = []
	# while threshold > 1e-15:
	s_v = np.where(l_m > threshold)[0]
		# if len(s_v) > 1:
			# break
		# threshold /= 10

	if len(s_v) == 0:
		return None
	print(colored(len(s_v),"yellow"),end=" ")

	# Compute b
	outer = 0
	for j in s_v:
		inner = sum([l_m[i][0] * t[i][0] * k(data[i],data[j]) for i in s_v])
		outer += t[j][0] - inner
	b = outer/len(s_v)

	toReturn = Classifier(s_v,l_m,b,t,k,data)

	return toReturn

def classify(svm,inputs,return_sums=False):
	if svm==None:
		return [[0],[0]]
	return svm.predict(inputs,return_sums)