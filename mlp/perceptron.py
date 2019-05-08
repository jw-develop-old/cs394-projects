
import numpy as np
from random import uniform

# The step function
def step(x) :
    return -1 if x < 0 else 1

# The "logistic" function, often called "sigmoid"
def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x) :
    return x * (1-x)

# We can adjust the sigmoid to make it range from -1 to 1
def sigmoid_adjusted(x) :
    return 2 / (1 + np.exp(-x)) -1 

# A class that represents a single perceptron
class Perceptron :
    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = activation
    def dimension(self) :
        return len(self.weights)-1
    def __call__(self, inputs) :
        return self.activation(np.dot(inputs,self.weights))
    def __str__(self) :
        return ",".join([str(w) for w in self.weights])

def initialize_perceptron(n) :
    return Perceptron(2*np.random.random((n,1)) - 1, sigmoid)

# line = np.linspace(-5, 5, 100)
# import matplotlib.pyplot as plt

# and_gate = Perceptron([-1, 1, 1], sigmoid_adjusted)

# [and_gate([1,1]),and_gate([1,-1]),and_gate([-1,-1])]

# or_gate = Perceptron([1, 1, 1], sigmoid_adjusted)

# [or_gate([1,1]), or_gate([1,-1]), or_gate([-1,-1])]

# from random import uniform

# # train an and gate
# data = [([1,1],1), ([1,-1],-1), ([-1,1],-1), ([-1,-1],-1)]
# p = initialize_perceptron(3)
# for i in range(10) :
#     print("iteration " + str(i))
#     print(str(p))
#     print(",".join([str(p(d[0])) for d in data]))
#     for d in data:
#         perc_train_step(p, d[0], d[1])

# We can adjust the sigmoid to make it range from -1 to 1
# def sigmoid_adjusted(x) :
#     return 2 / (1 + np.exp(-x)) -1 

# The step function
# def step(x) :
#     return 0 if x < 0 else 1