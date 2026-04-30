import numpy as np

class Linear:
    def __init__(self, input_size, output_size):
        # The weights for each node are column vectors
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.biases = np.zeros((1, output_size))
        self.input: np.ndarray = np.array([])
        self.grad_weights: np.ndarray = np.zeros_like(self.weights)
        self.grad_biases: np.ndarray = np.zeros_like(self.biases)


    def forward(self, inputs):
        self.input = inputs
        return inputs @ self.weights + self.biases
    
    def backward(self, upstream):
        print(self.input.shape)
        print(upstream.shape)
        self.grad_weights += self.input.T @ upstream
        self.grad_biases += np.sum(upstream, axis=0, keepdims=True)

        return upstream @ self.weights.T
    
    def step(self, learning_rate=0.01):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases
        
    def zero_grad(self):
        self.grad_weights = np.zeros_like(self.grad_weights)
        self.grad_biases = np.zeros_like(self.grad_biases)