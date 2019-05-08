'''
Created on Feb 13, 2019

@author: sirjwhite
'''

import t_helper
import imgs

from sklearn.datasets import load_wine

# Main method. Imports data and sets up for the tests.
if __name__ == '__main__':

	print("--Importing wine data--")

	wine_data = load_wine()
	set1 = (wine_data['data'].tolist(),wine_data['target'].tolist())
	out1 = 'wine_results.csv'

#	print(type(set1[0]))
#	print(len(set1[0]))
	
#	print("--Importing/parsing scene data--")

#	set2 = imgs.load_scenes()
#	out2 = 'scene_results.csv'

#	print(type(set2[0]))
#	print(len(set2[0]))
	
	print("--Test 1--")
	t_helper.multiTest(set1,out1)
#	print("--Test 2--")
#	t_helper.multiTest(set2,out2)