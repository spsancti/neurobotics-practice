from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
import neural_net
import mnist_loader as mnist
import train as tr
import pickle
import layers

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def prepare_mnist():
    mndata = mnist.MNIST('data', return_type='numpy', mode='vanilla')
    train, train_labels = mndata.load_training()
    test, test_labels = mndata.load_testing()

    # normalize data to [-0.5, 0.5] as proposed by Hampel, 2015

    max_train = np.max(train)
    max_test = np.max(test)

    train = train / max_train
    test = test / max_test

    train = train - .5
    test  = test - .5

    img_shape = (1, 28, 28)
    train = train.reshape(-1, *img_shape)
    test = test.reshape(-1, *img_shape)

    return train, train_labels, test, test_labels


def test_nn():
    train, train_labels, test, test_labels = prepare_mnist()

    nn = load_object("/home/btymchenko/Discovery/Neurality/trained/new_mnist_keras__epoch_14_iter_400.pickle")

    n = 625

    X = np.asarray(test[:n])
    labels = test_labels[:n]

    Y_predicted = nn.forward(X)
    image_back  = nn.backward(Y_predicted)

    labels_predicted = np.argmax(Y_predicted, axis=1)

    tr.plot_batch(image_back, 0)

    diff = (labels == labels_predicted)

    print("Accuracy: ")
    print(sum(diff) / len(labels_predicted))
    plt.show()


def train_nn():
    train, train_labels, test, test_labels = prepare_mnist()

    net = neural_net.AdvancedNN()

    net.add(layers.Conv2D(32, 1, 3, 3))
    net.add(layers.LReLU())
    net.add(layers.MaxPooling(2, 2))

    net.add(layers.Flatten())

    net.add(layers.Dropout(0.25))
    net.add(layers.Dense(32*14*14, 128))
    net.add(layers.LReLU())

    net.add(layers.Dropout(0.5))
    net.add(layers.Dense(128, 10))
    net.add(layers.ReLU())

    net = load_object("/home/btymchenko/Discovery/Neurality/trained/new_mnist_keras__epoch_14_iter_400.pickle")

    tr.sgd(net,
           train,
           train_labels,
           "/home/btymchenko/Discovery/Neurality/trained/new_mnist_keras_",
           val_set=(test, test_labels),
           mb_size=128,
           n_epoch=15,
           alpha=1e-2,
           save_after=100,
           print_after=100,
           )

    # show this one!
    #plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    train_nn()
    # test_nn()
