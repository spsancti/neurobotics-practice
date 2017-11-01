# my implementation from memory
# of https://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

def f(x):
    return 1. / (1. + np.exp(-x))

def df(x):
    return x * (1 - x)

X = np.array([  [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1] ])
       
Y = np.array([[0, 1, 0, 1]]).T

W0 = 0.02 * (np.random.random((3, 1)) - .5)

rate = 1.
for iter in range(1, 10000):

    # forward pass
    l0 = X
    l1 = f(np.dot(l0,W0))

    # error
    l1_error = Y - l1

    # backward pass
    l1_delta = l1_error * df(l1)

    # update weights
    W0 += np.dot(l0.T,l1_delta)

print ("Output :")
print (l1)
