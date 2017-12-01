from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
import neural_net
import mnist_loader as mnist
import train as tr

from layers import maxpool_forward

def main():

    # image = misc.face()

    mndata = mnist.MNIST('data', return_type='numpy')
    train, train_labels = mndata.load_training()
    test, test_labels = mndata.load_testing()

    mean = np.mean(train)
    train = train -  mean
    mean = np.mean(test)
    test = test - mean

    img_shape = (1, 28, 28)
    train = train.reshape(-1, *img_shape)
    test = test.reshape(-1, *img_shape)

    net = neural_net.NeuralNet()

    tr.sgd(net, train, train_labels, val_set=(test, test_labels), print_after=5)



    # show this one!
    #plt.imshow(image, cmap='gray')
    #plt.show()


if __name__ == '__main__':
    main()
