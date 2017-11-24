import numpy as np
from im2col import im2col_indices


def fc_forward(X, W, b):
    out = np.dot(X, W) + b
    return out


def conv_forward(X, K, b, padding=1, stride=1):
    n_x, d_x, h_x, w_x = X.shape
    n_k, d_k, h_k, w_k = K.shape

    X_col = im2col_indices(X, h_k, w_k, padding=padding, stride=stride)
    K_col = K.reshape(n_k, -1)

    Y_col = np.dot(K_col, X_col) + b

    h_out = int((h_x - h_k + 2 * padding) / stride + 1)
    w_out = int((w_x - w_k + 2 * padding) / stride + 1)

    Y = Y_col.reshape(n_k, h_out, w_out, n_x)
    Y = Y.transpose(3, 0, 1, 2)

    return Y


def relu_forward(X):
    return np.maximum(0, X)


def lrelu_forward(X, a=1e-3):
    out = np.maximum(a * X, X)
    return out


def sigmoid_forward(X):
    out = 1. / (1. + np.exp(-X))
    return out
