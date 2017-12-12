import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle as skshuffle
import pickle

def plot_layer(X, id):
    n, d, h, w = X.shape

    n_plots = int(np.ceil(np.sqrt(n)))
    f, ax = plt.subplots(n_plots, n_plots)

    for nx in range(n_plots):
        for ny in range(n_plots):
            if ny*n_plots + nx < n:
                ax[nx, ny].imshow(X[ny*n_plots + nx][id], cmap='magma')
            ax[nx, ny].axis('off')
    plt.show(block=False)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def get_minibatch(X, y, minibatch_size, shuffle=True):
    minibatches = []

    if shuffle:
        X, y = skshuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches

def sgd(nn, X_train, y_train, filename, val_set=None, alpha=1e-3, mb_size=128, n_epoch=10, print_after=100, show_after=100, save_after=100):

    if val_set:
        X_val, y_val = val_set

    for epoch in range(1, n_epoch):

        minibatches = get_minibatch(X_train, y_train, mb_size)

        print('Epoch: {}'.format(epoch))
        print('Iterations: {}'.format(len(minibatches)))
        print('Alpha: {}'.format(alpha))

        epoch_loss = 0.

        for iter in range(0, len(minibatches) - 1):

            X_mini, y_mini = minibatches[iter]
            grad, loss = nn.train_step(X_mini, y_mini)

            epoch_loss += loss

            if iter % print_after == 0:
                print('Iter-{} loss: {}'.format(iter, loss))

            if iter % save_after == 0:
                save_object(nn, filename + "_epoch_" + str(epoch) + "_iter_" + str(iter) + ".pickle")

            for layer in grad:
                nn.model[layer] -= alpha * grad[layer]

        alpha /= 2
        plot_layer(nn.model["conv1"], 0)
        print("Mean epoch loss: " + str(epoch_loss / len(minibatches)))

    return nn