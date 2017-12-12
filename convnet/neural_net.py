import layers
import numpy as np


class NeuralNet:

    def __init__(self):
        self.model = dict(
            conv1=np.random.randn(32, 1, 3, 3) / np.sqrt(32 / 2.),
            conv2=np.random.randn(64, 32, 3, 3) / np.sqrt(64 / 2.),
                                                # pool(2x2)
                                                # drop(0.25)
                                                # flat
            fc1=np.random.randn(64*14*14, 128) / np.sqrt(64 * 14 * 14 / 2.),      # dense(128)
                                                # drop(0.5)
            fc2=np.random.randn(128, 10)  / np.sqrt(128 / 2.),       # dense(10)

            conv1b=np.zeros((32, 1)),
            conv2b=np.zeros((64, 1)),
            fc1b  =np.zeros((1, 128)),
            fc2b  =np.zeros((1, 10))
        )
        pass

    def nn_forward(self, X, train=False):
        conv1, conv1cache = layers.conv_forward(X, self.model["conv1"], self.model["conv1b"], padding=1, stride=1)
        relu1, relu1cache = layers.relu_forward(conv1)

        conv2, conv2cache = layers.conv_forward(relu1, self.model["conv2"], self.model["conv2b"], padding=1, stride=1)
        relu2, relu2cache = layers.relu_forward(conv2)

        pool, pool_cache = layers.maxpool_forward(relu2)

        # add drop later
        if train==True:
            drop1, drop1cache = layers.dropout_forward(pool, 0.25)
        else:
            drop1, drop1cache = pool, pool_cache

        flat = drop1.ravel().reshape(X.shape[0], -1) # keep batch together

        fc1, fc1cache = layers.fc_forward(flat, self.model["fc1"], self.model["fc1b"])
        relu3, relu3cache = layers.relu_forward(fc1)

        if train==True:
            drop2, drop2_cache = layers.dropout_forward(relu3, 0.5)
        else:
            drop2, drop2_cache = relu3, fc1cache

        fc2, fc2cache = layers.fc_forward(drop2, self.model["fc2"], self.model["fc2b"])
        relu4, relu4cache = layers.relu_forward(fc2)

        out = relu4

        # oh fuck
        return out, (X, conv1cache, relu1cache, conv2cache, relu2cache, drop1cache, pool, pool_cache, fc1cache, relu3cache, drop2_cache, fc2cache, relu4cache)

    def nn_backward(self, grad_y, cache, train=False):
        # oh fuck [2]
        X, conv1cache, relu1cache, conv2cache, relu2cache, drop1cache, pool, pool_cache, fc1cache, relu3cache, drop2_cache, fc2cache, relu4cache = cache

        dX_relu4 = layers.relu_backward(grad_y, relu4cache)
        dX_fc2, dW_fc2, dB_fc2 = layers.fc_backward(dX_relu4, fc2cache)

        if train == True:
            dX_drop2 = layers.dropout_backward(dX_fc2, drop2_cache)
        else:
            dX_drop2 = dX_fc2

        dX_relu3 = layers.relu_backward(dX_drop2, relu3cache)
        dX_fc1, dW_fc1, dB_fc1 = layers.fc_backward(dX_relu3, fc1cache)

        unflat = dX_fc1.ravel().reshape(pool.shape)

        if train == True:
            dX_drop1 = layers.dropout_backward(unflat, drop1cache)
        else:
            dX_drop1 = unflat

        dX_pool = layers.maxpool_backward(dX_drop1, pool_cache)

        dX_relu2 = layers.relu_backward(dX_pool, relu2cache)
        dX_conv2, dW_conv2, dB_conv2 = layers.conv_backward(dX_relu2, conv2cache)

        dX_relu1 = layers.relu_backward(dX_conv2, relu1cache)
        dX_conv1, dW_conv1, dB_conv1 = layers.conv_backward(dX_relu1, conv1cache)

        out = dict(conv1=dW_conv1,
                   conv2= dW_conv2,
                   fc1=dW_fc1,
                   fc2=dW_fc2,

                   conv1b=dB_conv1,
                   conv2b=dB_conv2,
                   fc1b=dB_fc1,
                   fc2b=dB_fc2
        )
        return out, dX_conv1


    def train_step(self, X, Y):
        out, cache = self.nn_forward(X, train=True)

        error = layers.logistic_loss_backward(out, Y)
        grad, _ = self.nn_backward(error, cache, train=True)

        loss = layers.logistic_loss_forward(out, Y)
        return grad, loss
