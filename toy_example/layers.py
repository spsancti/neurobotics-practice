import numpy as np


class Layer(object):

    def __init__(self):
        self.params = dict()
        self.dParams = dict()

    def forward(self, X, train=False):
        pass

    def backward(self, grad, train=False):
        pass

    def update(self, update_fun):
        for param in self.params:
            update_fun(self.params[param], self.dParams[param])


class Dense(Layer):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.params = dict(
            W = np.random.randn(inDim, outDim) * np.sqrt(1. / inDim),
            b = np.zeros((1, outDim))
        )
        self.dParams = dict(
            W = np.zeros_like(self.params["W"]),
            b = np.zeros_like(self.params["b"])
        )

        self.cache = None

    def forward(self, X, train=False):
        out = np.dot(X, self.params["W"]) + self.params["b"]
        self.cache = (self.params["W"], X)
        return out

    def backward(self, grad, train=False):
        W, h = self.cache

        self.dParams["W"] = np.dot(h.T, grad)
        self.dParams["b"] = np.sum(grad, axis=0)

        dX = np.dot(grad, W.T)
        return dX


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, X, train=False):
        self.cache = X
        return np.maximum(0, X)

    def backward(self, grad, train=False):
        X = self.cache
        dX = grad.copy()
        dX[X <= 0] = 0
        return dX


class LReLU(Layer):
    def __init__(self, alpha=1e-3):
        super().__init__()
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


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, X, train=False):
        out = 1. / (1. + np.exp(-X))
        self.cache = out
        return out

    def backward(self, grad, train=False):
        out = self.cache
        return out * (1. - out) * grad


class Dropout(Layer):

    def __init__(self, p):
        super().__init__()
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

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, X, train=False):
        self.cache = X.shape
        return X.ravel().reshape(X.shape[0], -1)

    def backward(self, grad, train=False):
        X_shape = self.cache
        return grad.ravel().reshape(X_shape)

def softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T


def onehot(labels):
    y = np.zeros([labels.size, np.max(labels) + 1])
    y[range(labels.size), labels] = 1.
    return y


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


def squared_loss(y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]

    data_loss = 0.5 * np.sum((onehot(y_train) - y_pred)**2) / m

    return data_loss


def dsquared_loss(y_pred, y_train):
    m = y_pred.shape[0]
    grad_y = y_pred - onehot(y_train)
    grad_y /= m

    return grad_y