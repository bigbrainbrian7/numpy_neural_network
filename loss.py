import numpy as np

class SoftmaxCrossEntropyLoss:
    def forward(self, inputs, y=None):
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        exp = np.exp(shifted)
        eps = 1e-9
        self.probs = exp / exp.sum(axis=1, keepdims=True)

        if y is not None: self.loss = -np.sum(y * np.log(self.probs + eps)) / inputs.shape[0]

        return self.probs
    
    def backward(self, y):
        return (self.probs - y) / y.shape[0]

