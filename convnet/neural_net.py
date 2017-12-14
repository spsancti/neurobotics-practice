import layers


class AdvancedNN:

    def __init__(self):
        self.model = []

    def add(self, layer):
        self.model.append(layer)

    def forward(self, X, train=False):
        interlayer_result = X
        for layer in self.model:
            interlayer_result = layer.forward(interlayer_result, train)

        return interlayer_result

    def backward(self, grad, train=False):
        interlayer_result = grad
        for layer in reversed(self.model):
            interlayer_result = layer.backward(interlayer_result, train)

        return interlayer_result

    def sgd_update(self, alpha=1e-3):
        for layer in self.model:
            layer.sgd_update(alpha=alpha)

    def train_step(self, X, Y):
        out = self.forward(X, train=True)

        error = layers.logistic_loss_backward(out, Y)
        self.backward(error, train=True)

        loss = layers.logistic_loss_forward(out, Y)
        return loss

