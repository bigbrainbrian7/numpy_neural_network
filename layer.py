import numpy as np

from typing import Protocol

class Layer(Protocol):
    def forward(self, inputs: np.ndarray) -> np.ndarray: ...
    def backward(self, upstream: np.ndarray) -> np.ndarray: ...
    def step(self, learning_rate: float = 0.01) -> None: ...
    def zero_grad(self) -> None: ...

class Linear:
    def __init__(self, input_size, output_size):
        # The weights for each node are column vectors
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)
        self.biases = np.zeros((1, output_size))
        self.input: np.ndarray = np.array([])
        self.grad_weights: np.ndarray = np.zeros_like(self.weights)
        self.grad_biases: np.ndarray = np.zeros_like(self.biases)


    def forward(self, inputs):
        self.input = inputs
        return inputs @ self.weights + self.biases
    
    def backward(self, upstream):
        self.grad_weights += self.input.T @ upstream
        self.grad_biases += np.sum(upstream, axis=0, keepdims=True)

        return upstream @ self.weights.T
    
    def step(self, learning_rate=0.01):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases
        
    def zero_grad(self):
        self.grad_weights = np.zeros_like(self.grad_weights)
        self.grad_biases = np.zeros_like(self.grad_biases)

class ReLU:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.input = inputs
        return np.maximum(0, inputs)
    
    def backward(self, upstream):
        return upstream * (self.input > 0)
    
    def zero_grad(self):
        pass

    def step(self, learning_rate=0.0):
        pass

class BatchNorm:
    def __init__(self, input_size, eps=1e-5, momentum=0.9):
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))

        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        self.gradient_cache = (np.array([]),np.array([]),np.array([]),np.array([]),)

    def forward(self, inputs, training=True):
        if training:
            mean = np.mean(inputs, axis=0, keepdims=True)
            var = np.var(inputs, axis=0, keepdims=True) #broadcasts across axis 1


            normed = (inputs - mean) / np.sqrt(var + self.eps)
            output = normed * self.gamma + self.beta

            self.running_mean = self.running_mean * self.momentum + (1 - self.momentum) * mean
            self.running_var = self.running_var * self.momentum + (1 - self.momentum) * var

            self.gradient_cache = (inputs, normed, mean, var)

        else:
            normed = (inputs - self.running_mean) / np.sqrt(self.running_var + self.eps)
            output = normed * self.gamma + self.beta

        return output
    
    def backward(self, upstream):
        self.grad_gamma += np.sum(upstream * self.gradient_cache[1], axis=0, keepdims=True) #broadcasts across axis 0
        self.grad_beta += np.sum(upstream, axis=0, keepdims=True)

        # shape of (1, num_features,), will broadcast down later
        # partial of x_hat wrt to x, negative wrt mean
        partial_x_hat = 1 / (np.sqrt(self.gradient_cache[3] + self.eps))

        # multivariate chain rule means that the effects of input into batch "node" propagate into every single output through the mean
        # since its sum of every upstream partial * partial of output wrt to mean, we can just add them together first
        # then multiply by partial of mean wrt to input
        summed_over_batch = np.sum(upstream, axis=0, keepdims=True)

        # direct wrt to x
        self.grad = upstream * self.gamma * partial_x_hat

        # wrt to mean
        self.grad -= self.gamma * partial_x_hat * summed_over_batch / self.gradient_cache[0].shape[0]

        normed = self.gradient_cache[0] - self.gradient_cache[2]

        dsigma_dx = normed / (self.gradient_cache[0].shape[0] * np.pow(self.gradient_cache[3] + self.eps, 1.5))
        assert self.grad.shape == dsigma_dx.shape == self.gradient_cache[0].shape

        sum_upstream_times_normed = np.sum(upstream * normed, axis=0, keepdims=True)
        assert sum_upstream_times_normed.shape == (1, self.grad.shape[1]) == self.gradient_cache[3].shape


        # wrt to var
        self.grad -= self.gamma * dsigma_dx * sum_upstream_times_normed 

        return self.grad

    def zero_grad(self):
        self.grad_gamma = np.zeros_like(self.grad_gamma)
        self.grad_beta = np.zeros_like(self.grad_beta)

    def step(self, learning_rate=0.01):
        self.gamma -= learning_rate * self.grad_gamma
        self.beta -= learning_rate * self.grad_beta



