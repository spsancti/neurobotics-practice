import numpy as np
from im2col import im2col_indices
from im2col import col2im_indices


def fc_forward(X, W, b):
    out = np.dot(X, W) + b
    cache = (W, X)
    return out, cache


def fc_backward(dout, cache):
    W, h = cache

    dW = np.dot(h.T, dout)
    db = np.sum(dout, axis=0)
    dX = np.dot(dout, W.T)

    return dX, dW, db


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

    return Y, (X, K, b, stride, padding, X_col)


def conv_backward(dout, cache):
    X, K, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = K.shape

    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dK = dout_reshaped @ X_col.T
    dK = dK.reshape(K.shape)

    K_reshape = K.reshape(n_filter, -1)
    dX_col = np.dot(K_reshape.T, dout_reshaped)
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

    return dX, dK, db

def relu_forward(X):
    cache = X
    return np.maximum(0, X), cache


def relu_backward(dout, cache):
    dX = dout.copy()
    dX[cache <= 0] = 0
    return dX


def lrelu_forward(X, a=1e-3):
    out = np.maximum(a * X, X)
    cache = (X, a)
    return out, cache


def lrelu_backward(dout, cache):
    X, a = cache
    dX = dout.copy()
    dX[X < 0] *= a
    return dX


def sigmoid_forward(X):
    out = 1. / (1. + np.exp(-X))
    cache = out
    return out, cache


def sigmoid_backward(dout, cache):
    return cache * (1. - cache) * dout


def maxpool_forward(X, size=2, stride=2):
    n_x, d_x, h_x, w_x = X.shape

    # reshape the source to make K exactly size*size
    # without reshape, K would be d*size*size
    X_res = X.reshape(n_x * d_x, 1, h_x, w_x)
    X_col = im2col_indices(X_res, size, size, padding=0, stride=stride)

    # Next, at each possible patch location, i.e. at each column, we're taking the max index
    max_idx = np.argmax(X_col, axis=0)

    # get all the max value at each column
    out = X_col[max_idx, range(max_idx.size)]

    # the same formula as for convolution but padding is zero, so it's missed
    h_out = int((h_x - size) / stride + 1)
    w_out = int((w_x - size) / stride + 1)

    # reshape back
    out = out.reshape(h_out, w_out, n_x, d_x)
    out = out.transpose(2, 3, 0, 1)

    return out, (X, X_col, size, stride, max_idx)


def maxpool_backward(dout, cache):
    X, X_col, size, stride, max_idx = cache
    n_x, d_x, h_x, w_x = X.shape

    # reserve the exact shape as cached
    dX_col = np.zeros(X_col.shape)

    # ravel() is the same as flatten() but works inplace
    dout_flat = dout.transpose(2, 3, 0, 1).ravel()

    # fill reserved shapes with maximums at their locations from cache
    dX_col[max_idx, range(max_idx.size)] = dout_flat

    dX = col2im_indices(dX_col, (n_x * d_x, 1, h_x, w_x), size, size, padding=0, stride=stride)

    # Reshape back to match the input dimension
    dX = dX.reshape(X.shape)

    return dX

def dropout_forward(X, p_dropout):
    u = np.random.binomial(1, p_dropout, size=X.shape) / p_dropout
    out = X * u
    cache = u
    return out, cache


def dropout_backward(dout, cache):
    dX = dout * cache
    return dX







def softmax_forward(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T

def logistic_loss_forward(out, Y):
    m = out.shape[0]
    prob = softmax_forward(out)
    log_like = -np.log(prob[range(m), Y])
    loss = np.sum(log_like) / m

    return loss


def logistic_loss_backward(out, Y):
    m = out.shape[0]

    grad_y = softmax_forward(out)
    grad_y[range(m), Y] -= 1.
    grad_y /= m

    return grad_y


def onehot(labels):
    y = np.zeros([labels.size, np.max(labels) + 1])
    y[range(labels.size), labels] = 1.
    return y

def squared_loss(y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]

    data_loss = 0.5 * np.sum((onehot(y_train) - y_pred)**2) / m

    return data_loss


def dsquared_loss(y_pred, y_train):
    m = y_pred.shape[0]
    grad_y = y_pred - onehot(y_train)
    grad_y /= m

    return grad_y