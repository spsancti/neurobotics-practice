import layers
import numpy as np


class NeuralNet:

    def __init__(self):
        self.model = dict(
            conv1=np.random.randn(32, 1, 3, 3), # conv(32x3x3)
            conv2=np.random.randn(4, 32, 3, 3), # conv(64x3x3)
                                                # pool(2x2)
                                                # drop(0.25) not implemented :(
                                                # flat
            fc1=np.random.randn(784, 128),      # dense(128)
                                                # drop(0.25) not implemented :(
            fc2=np.random.randn(128, 10),       # dense(10)

            conv1b=np.zeros((32, 1)),
            conv2b=np.zeros((4, 1)),
            fc1b  =np.zeros((1, 128)),
            fc2b  =np.zeros((1, 10))
        )
        pass

    def nn_forward(self, X):
        conv1, conv1cache = layers.conv_forward(X, self.model["conv1"], self.model["conv1b"])
        relu1, relu1cache = layers.relu_forward(conv1)

        conv2, conv2cache = layers.conv_forward(relu1, self.model["conv2"], self.model["conv2b"])
        relu2, relu2cache = layers.relu_forward(conv2)

        pool, pool_cache = layers.maxpool_forward(relu2)

        # add drop later

        flat = pool.ravel().reshape(X.shape[0], -1) # keep batch together

        fc1, fc1cache = layers.fc_forward(flat, self.model["fc1"], self.model["fc1b"])
        relu3, relu3cache = layers.relu_forward(fc1)

        fc2, fc2cache = layers.fc_forward(relu3, self.model["fc2"], self.model["fc2b"])
        relu4, relu4cache = layers.relu_forward(fc2)

        out = relu4

        # oh fuck
        return out, (X, conv1, conv1cache, relu1, relu1cache, conv2, conv2cache, relu2, relu2cache, pool, pool_cache, flat, fc1, fc1cache, relu3, relu3cache, fc2, fc2cache, relu4, relu4cache)

    def nn_backward(self, y_out, y_true, cache):
        # oh fuck [2]
        X, conv1, conv1cache, relu1, relu1cache, conv2, conv2cache, relu2, relu2cache, pool, pool_cache, flat, fc1, fc1cache, relu3, relu3cache, fc2, fc2cache, relu4, relu4cache = cache

        grad_y = layers.logistic_loss_backward(y_out, y_true)

        dX_relu4 = layers.relu_backward(grad_y, relu4cache)
        dX_fc2, dW_fc2, dB_fc2 = layers.fc_backward(dX_relu4, fc2cache)

        dX_relu3 = layers.relu_backward(dX_fc2, relu3cache)
        dX_fc1, dW_fc1, dB_fc1 = layers.fc_backward(dX_relu3, fc1cache)

        unflat = dX_fc1.ravel().reshape(pool.shape)

        dX_pool = layers.maxpool_backward(unflat, pool_cache)

        dX_relu2 = layers.relu_backward(dX_pool, relu2cache)
        dX_conv2, dW_conv2, dB_conv2 = layers.conv_backward(dX_relu2, conv2cache)

        dX_relu1 = layers.relu_backward(dX_conv2, relu1cache)
        dX_conv1, dW_conv1, dB_conv1 = layers.conv_backward(dX_relu1, conv1cache)

        out = dict(conv1=dW_conv1,
                   conv2=dW_conv2,
                   fc1=dW_fc1,
                   fc2=dW_fc2,

                   conv1b=dB_conv1,
                   conv2b=dB_conv2,
                   fc1b=dB_fc1,
                   fc2b=dB_fc2
        )
        return out

    def train_step(self, X, Y):
        out, cache = self.nn_forward(X)
        loss = layers.logistic_loss_backward(out, Y)
        grad = self.nn_backward(out, Y, cache)
        return grad, loss
