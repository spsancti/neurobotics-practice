import numpy as np
from im2col import im2col_indices
from im2col import col2im_indices

class Layer(object):

    def forward(self, X, train=False):
        pass

    def backward(self, grad, train=False):
        pass

    def sgd_update(self, alpha=1e-3):
        pass


class Dense(Layer):
    def __init__(self, inDim, outDim):
        self.W = np.random.randn(inDim, outDim) * np.sqrt(2. / inDim)
        self.b = np.zeros((1, outDim))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache = None

    def forward(self, X, train=False):
        out = np.dot(X, self.W) + self.b
        self.cache = (self.W, X)
        return out

    def backward(self, grad, train=False):
        W, h = self.cache

        self.dW = np.dot(h.T, grad)
        self.db = np.sum(grad, axis=0)

        dX = np.dot(grad, W.T)
        return dX

    def sgd_update(self, alpha=1e-3):
        self.W -= alpha * self.dW
        self.b -= alpha * self.db


class BatchNorm(Layer):

    def forward(self, X, train=False):
        pass

    def backward(self, grad, train=False):
        pass

    def sgd_update(self, alpha=1e-3):
        pass


class Conv2D(Layer):
    def __init__(self, n_filters, d_filters, h_filter=3, w_filter=3, padding=1, stride=1):
        self.W = np.random.randn(n_filters, d_filters, h_filter, w_filter) * np.sqrt(2. / n_filters)
        self.b = np.zeros((n_filters, 1))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.padding = padding
        self.stride = stride
        self.cache = None

    def forward(self, X, train=False):
        n_x, d_x, h_x, w_x = X.shape
        n_w, d_w, h_w, w_w = self.W.shape

        X_col = im2col_indices(X, h_w, w_w, padding=self.padding, stride=self.stride)
        W_col = self.W.reshape(n_w, -1)

        Y_col = np.dot(W_col, X_col) + self.b

        h_out = int((h_x - h_w + 2 * self.padding) / self.stride + 1)
        w_out = int((w_x - w_w + 2 * self.padding) / self.stride + 1)

        Y = Y_col.reshape(n_w, h_out, w_out, n_x)
        Y = Y.transpose(3, 0, 1, 2)

        self.cache = (X, X_col)
        return Y

    def backward(self, grad, train=False):
        X, X_col = self.cache
        n_filter, d_filter, h_filter, w_filter = self.W.shape

        self.dB = np.sum(grad, axis=(0, 2, 3))
        self.dB = self.dB.reshape(n_filter, -1)

        grad_reshaped = grad.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        self.dW = np.dot(grad_reshaped, X_col.T)
        self.dW = self.dW.reshape(self.W.shape)

        W_reshape = self.W.reshape(n_filter, -1)
        dX_col = np.dot(W_reshape.T, grad_reshaped)
        dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=self.padding, stride=self.stride)

        return dX

    def sgd_update(self, alpha=1e-3):
        self.W -= alpha * self.dW
        self.b -= alpha * self.db


class ReLU(Layer):
    def __init__(self):
        self.cache = None

    def forward(self, X, train=False):
        self.cache = X
        return np.maximum(0, X)

    def backward(self, grad, train=False):
        X = self.cache
        dX = grad.copy()
        dX[X <= 0] = 0
        return dX

    def sgd_update(self, alpha=1e-3):
        # no parameters to update
        pass


class LReLU(Layer):
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.cache = None

    def forward(self, X, train=False):
        self.cache = X
        return np.maximum(self.alpha * X, X)

    def backward(self, grad, train=False):
        X = self.cache
        dX = grad.copy()
        dX[X < 0] *= self.alpha
        return dX

    def sgd_update(self, alpha=1e-3):
        # no parameters to update
        pass


class Sigmoid(Layer):
    def __init__(self):
        self.cache = None

    def forward(self, X, train=False):
        out = 1. / (1. + np.exp(-X))
        self.cache = out
        return out

    def backward(self, grad, train=False):
        out = self.cache
        return out * (1. - out) * grad

    def sgd_update(self, alpha=1e-3):
        # no parameters to update
        pass


class MaxPooling(Layer):
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride
        self.cache = None

    def forward(self, X, train=False):
        n_x, d_x, h_x, w_x = X.shape

        # reshape the source to make K exactly size*size
        # without reshape, K would be d*size*size
        X_res = X.reshape(n_x * d_x, 1, h_x, w_x)
        X_col = im2col_indices(X_res, self.size, self.size, padding=0, stride=self.stride)

        # Next, at each possible patch location, i.e. at each column, we're taking the max index
        max_idx = np.argmax(X_col, axis=0)

        # get all the max value at each column
        out = X_col[max_idx, range(max_idx.size)]

        # the same formula as for convolution but padding is zero, so it's missed
        h_out = int((h_x - self.size) / self.stride + 1)
        w_out = int((w_x - self.size) / self.stride + 1)

        # reshape back
        out = out.reshape(h_out, w_out, n_x, d_x)
        out = out.transpose(2, 3, 0, 1)

        self.cache = (X.shape, X_col.shape, max_idx)

        return out

    def backward(self, grad, train=False):
        X_shape, X_col_shape, max_idx = self.cache
        n_x, d_x, h_x, w_x = X_shape

        # reserve the exact shape as cached
        dX_col = np.zeros(X_col_shape)

        # ravel() is the same as flatten() but works inplace
        grad_flat = grad.transpose(2, 3, 0, 1).ravel()

        # fill reserved shapes with maximums at their locations from cache
        dX_col[max_idx, range(max_idx.size)] = grad_flat

        dX = col2im_indices(dX_col, (n_x * d_x, 1, h_x, w_x), self.size, self.size, padding=0, stride=self.stride)

        # Reshape back to match the input dimension
        dX = dX.reshape(X_shape)

        return dX

    def sgd_update(self, alpha=1e-3):
        # no parameters to update
        pass


class Dropout(Layer):

    def __init__(self, p):
        self.p = p
        self.cache = None

    def forward(self, X, train=False):
        if train == True:
            u = np.random.binomial(1, self.p, size=X.shape) / self.p
        else:
            u = np.ones(X.shape)

        out = X * u
        self.cache = u
        return out

    def backward(self, grad, train=False):
        u = self.cache
        dX = grad * u
        return dX

    def sgd_update(self, alpha=1e-3):
        pass


class Flatten(Layer):
    def __init__(self):
        self.cache = None

    def forward(self, X, train=False):
        self.cache = X.shape
        return X.ravel().reshape(X.shape[0], -1)

    def backward(self, grad, train=False):
        X_shape = self.cache
        return grad.ravel().reshape(X_shape)

    def sgd_update(self, alpha=1e-3):
        pass





def softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T

def logistic_loss_forward(out, Y):
    m = out.shape[0]
    prob = softmax(out)
    log_like = -np.log(prob[range(m), Y])
    loss = np.sum(log_like) / m

    return loss


def logistic_loss_backward(out, Y):
    m = out.shape[0]

    grad_y = softmax(out)

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