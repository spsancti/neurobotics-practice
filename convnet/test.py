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

def __main():
    mndata = mnist.MNIST('data', return_type='numpy', mode='rounded_binarized')
    train, train_labels = mndata.load_training()
    test, test_labels = mndata.load_testing()

    img_shape = (1, 28, 28)
    train = train.reshape(-1, *img_shape)
    test = test.reshape(-1, *img_shape)

    nn = load_object("/home/btymchenko/Discovery/Neurality/trained/mnist_keras__epoch_5_iter_400.pickle")

    n = 36

    image = np.asarray(test[:n]);

    out, cache = nn.nn_forward(image);
    nout = layers.softmax_forward(out)
    nout = np.argmax(nout, axis=1)
    true = test_labels[:n]


    _, image = nn.nn_backward(out, cache)
    tr.plot_layer(image, 0)


    print(nout)
    print(true)
    diff = ((nout - true) == True)

    print(1. - np.mean(diff))
    plt.show()

def main():
    # plt.ion()
    # image = misc.face()

    mndata = mnist.MNIST('data', return_type='numpy', mode='rounded_binarized')
    train, train_labels = mndata.load_training()
    test, test_labels = mndata.load_testing()

    mean = np.mean(train)
    train = train - mean
    mean = np.mean(test)
    test = test - mean

    img_shape = (1, 28, 28)
    train = train.reshape(-1, *img_shape)
    test = test.reshape(-1, *img_shape)

    net = neural_net.NeuralNet()
    #net = load_object("/home/btymchenko/Discovery/Neurality/trained/mnist_keras__iter_400.pickle")

    tr.sgd(net,
           train,
           train_labels,
           "/home/btymchenko/Discovery/Neurality/trained/mnist_keras_",
           val_set=(test, test_labels),
           mb_size=128,
           n_epoch=10,
           alpha=1e-2,
           save_after=100,
           print_after=100,
           )

    # show this one!
    #plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
