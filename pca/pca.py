'''
Created on Apr 27, 2019

@author: James White
'''

import numpy as np
import sys

def pca(data,M):

	# Compute the covariance matrix.
	x_bar = np.mean(data)
	S = np.cov(data.T)

	# Compute the eigenvectors and corresponding eigenvalues.
	eigenvalues = np.linalg.eig(S)

	# Sort the eigenvectors by eigenvalues.
	indices = np.argsort(eigenvalues[0],axis=0)

	s_e = [eigenvalues[1][index] for index in indices]

	# Return the M eigenvectors with greatest eigenvalues (first occurring)
	components = s_e[:M]

	return components

def transform(data,components):

	# Making numpy arrays.
	t_data = np.array(data)
	t_c = np.array(components)

	# For all data points.
	toReturn = []
	for i in range(len(data)):

		# First part of the dot product.
		first = np.array(t_data[i])

		# Compute the dot products of x_i and each principal component.
		d = [np.dot(first,np.array(t_c[j])) for j in range(len(t_c))]

		toReturn.append(d)

	# Printing transformation results.
	print()
	print("Before -- After: (First five)")
	iter = 0
	for a,b in zip(data,toReturn):
		if (iter > 4):
			break
		print (a,"\t--\t",end="")
		nPrl(b)
		iter +=1

	return toReturn

def nPr(tPrint):
	print("{:.2f}".format(tPrint))

def nPrl(tPrint):
	print("[ ",end="")
	for p in tPrint:
		print("{:.2f} ".format(p),end="")
	print("]")