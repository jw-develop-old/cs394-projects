

'''

Inside of build:

else:

	# Compute all z
	z_l = []
	for p in self.pct:
		z_l.append(np.dot([1]+self.data,p.weights))

	# Compute all y
	y_vals = self.pred(self.data)

	# For each output unit, calculate the error.
	d_y = []
	for y in y_vals:
		toAdd = y*(1-y)*(t-y)
		d_y.append(toAdd)

	# Calculate derivative of activation function.
	d_z = []
	for z in z_l:
		toAdd = pc.sigmoid(z)*(1-pc.sigmoid(z))
		d_z.append(toAdd)
		
	# For each perceptron
	for index in range(len(d_z)):

		toAdd = .1*d_z[index]*z_l[index]
		w_ylk.append(toAdd)

	# Adjusting each weight of each perceptron.
	for y in y_vals:
		
		# For each perceptron
		for index in len(d_z):

			# For each weight
			for w in self.pct[z]:
				w += .1 * d_z[index]


	def toBinary(self,inputs):
		toReturn = []
		for i in range(len(inputs)):

			toReturn.append(0)
			for y in inputs:
				toReturn[i] += y[i]

			print("toReturn is ",toReturn[i])

			if toReturn[i] < .5:
				toReturn[i] = 0
			else:
				toReturn[i] = 1

		print(len(toReturn))
		return toReturn

	# Predict values based off of trained model.
	def pred(self,inputs):
		toReturn = []

		for i in range(len(inputs)):
			toReturn.append(0)

		# For each perceptron
		for p in self.pct:
			ys = pc.sigmoid(np.dot(inputs,p.weights))

			# For each prediction point
			print("y: ",ys)
			for i in range(len(ys)):
				toReturn[i] += ys[i] - .5
				
		print("toReturn: ",toReturn)

		return toReturn


	# # Training the classifier.
	# illegal = mlp.illegal_train(p_count, X_train, y_train,r)

	# # # The predicting done by the actual model.
	# # y_pred = mlp.illegal_classify(illegal, X_test)

	# percent = 1 - np.mean(y_pred != y_test)
	# print("\n{:.2f}".format(percent))



		elif len(pct) >1:

			

			################################################


#	wine_data = load_wine()
#	set1 = (wine_data['data'].tolist(),wine_data['target'].tolist())


##	print(type(set1[0]))
##	print(len(set1[0]))
	
##	print("--Importing/parsing scene data--")

#	load_iris = load_iris()
#	set2 = (load_iris['data'].tolist(),load_iris['target'].tolist())
# 	out2 = 'iris_results.csv'

#	print(type(set2[0]))
#	print(len(set2[0]))


	# print("--Test 2--")
	# runTest(set2,out2)

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


# 	print("--Importing wine data--")


class Writer:
	def __init__(self,a):
		self.file = open(a)

	def write(self,text):

	def close(self):
		file.close()
'''
