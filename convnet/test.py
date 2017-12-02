from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
import neural_net
import mnist_loader as mnist
import train as tr
import pickle
import layers
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def plot_layer(X, id):
    n, d, h, w = X.shape

    n_plots = int(np.ceil(np.sqrt(n)))
    print(n_plots)
    f, ax = plt.subplots(n_plots, n_plots)

    for nx in range(n_plots):
        for ny in range(n_plots):
            if ny*n_plots + nx < n:
                ax[nx, ny].imshow(X[ny*n_plots + nx][id], cmap='gray')
            ax[nx, ny].axis('off')
    plt.show()


def __main():
    mndata = mnist.MNIST('data', return_type='numpy', mode='vanilla')
    train, train_labels = mndata.load_training()
    test, test_labels = mndata.load_testing()

    img_shape = (1, 28, 28)
    train = train.reshape(-1, *img_shape)
    test = test.reshape(-1, *img_shape)

    nn = load_object("/home/btymchenko/nn_mnist64.pickle")

    layer = nn.model["conv1"]

    n = 32

    image = np.asarray(test[:n]);

    plot_layer(layer, 0)

    out, cache = nn.nn_forward(image);
    out = layers.softmax_forward(out)
    out = np.argmax(out, axis=1)
    true = test_labels[:n]

    print(out)
    print(true)
    diff = out - true

    print(np.mean(diff))


def main():

    # image = misc.face()

    mndata = mnist.MNIST('data', return_type='numpy', mode='vanilla')
    train, train_labels = mndata.load_training()
    test, test_labels = mndata.load_testing()

    mean = np.mean(train)
    train = train -  mean
    mean = np.mean(test)
    test = test - mean

    img_shape = (1, 28, 28)
    train = train.reshape(-1, *img_shape)
    test = test.reshape(-1, *img_shape)

    net = neural_net.NeuralNet(16)

    tr.sgd(net, train, train_labels, val_set=(test, test_labels), print_after=1)

    save_object(net, "/home/btymchenko/nn_mnist128.pickle")

    # show this one!
    #plt.imshow(image, cmap='gray')
    #plt.show()


if __name__ == '__main__':
    main()
