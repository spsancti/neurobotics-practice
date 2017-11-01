# my implementation from memory
# of https://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

def f(x):
    return 1. / (1. + np.exp(-x))

def df(x):
    return x * (1 - x)
    
X = np.array([[0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1]])
                
Y = np.array([[1, 0, 0, 0, 0, 0, 0, 1]]).T

# randomly initialize our weights with mean 0
W0 = 2 * (np.random.random((3, 4)) - .5)
W1 = 2 * (np.random.random((4, 1)) - .5)

for j in range(1, 100000):

	# forward pass
    l0 = X
    l1 = f(np.dot(l0, W0))
    l2 = f(np.dot(l1, W1))

    # error
    l2_error = Y - l2    
    
    l2_delta = l2_error * df(l2)
    
    l1_error = np.dot(l2_delta, W1.T)
    
    l1_delta = l1_error * df(l1)

    W1 += np.dot(l1.T, l2_delta)
    W0 += np.dot(l0.T, l1_delta)

print ("Output :")
print (l2)
