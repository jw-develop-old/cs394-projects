'''
Created on Feb 13, 2019

@author: sirjwhite
'''

from module.knn import knn

if __name__ == '__main__':
    X_train = [1,3,4]
    y_train = [4,6,5]
    X_test = [2,3,4]
    k = knn()
    k.knn(X_train, y_train, knn.euclidean,5, X_test)