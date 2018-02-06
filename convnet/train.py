import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle as skshuffle
import pickle

def plot_batch(X, id):
    n, d, h, w = X.shape

    n_plots = int(np.ceil(np.sqrt(n)))

    f, ax = plt.subplots(n_plots, n_plots)

    for nx in range(n_plots):
        for ny in range(n_plots):
            if ny*n_plots + nx < n:
                ax[nx, ny].imshow(X[ny*n_plots + nx][id], cmap='gray')
            ax[nx, ny].axis('off')
    plt.show(block=False)


def plot_layer(X, id):
    n, d, h, w = X.shape

    n_plots = int(np.ceil(np.sqrt(d)))

    f, ax = plt.subplots(n_plots, n_plots)

    for nx in range(n_plots):
        for ny in range(n_plots):
            if ny * n_plots + nx < d:
                ax[nx, ny].imshow(X[id][ny * n_plots + nx], cmap='viridis')
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

def test(nn, X_test, Y_test):
    minibatches = get_minibatch(X_test, Y_test, 128)

    overall_accuracy = 0.

    for i in range(0, len(minibatches)):
        x_i, y_i = minibatches[i]

        Y_predicted = nn.forward(x_i)
        labels_predicted = np.argmax(Y_predicted, axis=1)

        diff = (y_i == labels_predicted)
        overall_accuracy += sum(diff) / len(labels_predicted)

    overall_accuracy /= len(minibatches)

    return overall_accuracy



def sgd(nn, X_train, y_train, filename, val_set=None, alpha=1e-3, n_epoch=10, print_after=100):

    if val_set:
        X_val, y_val = val_set
    else:
        X_val, y_val = None, None

    for epoch in range(4, n_epoch + 4):

        mb_size = 2 ** epoch
        e_alpha = alpha * mb_size
        minibatches = get_minibatch(X_train, y_train, mb_size)

        print('Epoch: {}'.format(epoch))
        print('Iterations: {}'.format(len(minibatches)))
        print('Alpha: {}'.format(e_alpha))

        epoch_loss = 0.

        for iter in range(0, len(minibatches)):

            X_mini, y_mini = minibatches[iter]
            loss = nn.train_step(X_mini, y_mini)

            def upd(W, dW):
                W -= e_alpha * dW

            nn.update(upd)

            epoch_loss += loss

            if iter % print_after == 0:
                print('Iter-{} loss: {}'.format(iter, loss))

            if val_set and iter % (len(minibatches) / 10) == 0:
                print('Iter-{} accuracy: {}'.format(iter, test(nn, X_val, y_val)))

        save_object(nn, filename + "_epoch_" + str(epoch) + ".pickle")
        print("Mean epoch loss: " + str(epoch_loss / len(minibatches)))

        if val_set:
            print('Epoch accuracy: {}'.format(test(nn, X_val, y_val)))

    return nn
