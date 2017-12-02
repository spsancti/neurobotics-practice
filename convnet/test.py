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

    nn = load_object("/home/btymchenko/Discovery/Neurality/trained/mnist_32_32x3x3_iter_150.pickle")

    layer = nn.model["conv2"]
    tr.plot_layer(layer, 0)

    n = 640

    image = np.asarray(test[:n]);

    out, cache = nn.nn_forward(image);
    out = layers.softmax_forward(out)
    out = np.argmax(out, axis=1)
    true = test_labels[:n]

    print(out)
    print(true)
    diff = out - true

    print(np.mean(diff))
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

    net = neural_net.NeuralNet(32)
    #net = load_object("/home/btymchenko/Discovery/Neurality/trained/mnist_32_32x3x3_iter_150.pickle")

    tr.sgd(net,
           train,
           train_labels,
           "/home/btymchenko/Discovery/Neurality/trained/mnist_32_32x3x3_probe2",
           val_set=(test, test_labels),
           print_after=10,
           mb_size=256,
           n_iter=1000,
           alpha=1e-3,
           save_after=10,
           show_after=10
           )

    # show this one!
    #plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
