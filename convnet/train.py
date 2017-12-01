import numpy as np
from sklearn.utils import shuffle as skshuffle


def get_minibatch(X, y, minibatch_size, shuffle=True):
    minibatches = []

    if shuffle:
        X, y = skshuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches

def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)

def sgd(nn, X_train, y_train, val_set=None, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    minibatches = get_minibatch(X_train, y_train, mb_size)

    if val_set:
        X_val, y_val = val_set

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        if iter % print_after == 0:
            val_acc = accuracy(y_val, nn.nn_forward(X_val))
            print('Iter-{} loss: {}'.format(iter, val_acc))

        for layer in grad:
            nn.model[layer] -= alpha * grad[layer]

    return nn