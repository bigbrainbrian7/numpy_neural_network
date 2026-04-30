import numpy as np

from typing import List
from layer import Layer

class Model:
    def __init__(self, *layers: Layer):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, upstream):
        for layer in reversed(self.layers):
            upstream = layer.backward(upstream)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def step(self, learning_rate=0.01):
        for layer in self.layers:
            layer.step(learning_rate)