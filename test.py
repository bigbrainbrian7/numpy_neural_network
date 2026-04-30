import numpy as np
import pytest

from layer import Linear

#----helpers------

def make_linear_layer(input_size, output_size, seed=0):
    np.random.seed(seed)
    return Linear(input_size, output_size)

#----forward pass------

def test__linear_forward_pass_shape():
    layer = make_linear_layer(30,20)
    input = np.ones((1,30))
    output = layer.forward(input)
    assert output.shape == (1,20)

#----backward pass------

def test_gradient_check():
    layer1 = make_linear_layer(30,20)
    layer2 = make_linear_layer(20,20)
    layer3 = make_linear_layer(20,1)

    input = np.random.randn(1,30)

    output = layer3.forward(layer2.forward(layer1.forward(input)))

    upstream = layer3.backward(np.ones_like(output))
    upstream = layer2.backward(upstream)
    layer1.backward(upstream)

    calculated_grad = layer1.grad_weights

    layer1.weights[0][0] += 1e-5
    output_plus = layer3.forward(layer2.forward(layer1.forward(input)))

    layer1.weights[0][0] -= 2e-5
    output_minus = layer3.forward(layer2.forward(layer1.forward(input)))

    numerical_grad = (output_plus - output_minus) / 2e-5

    backprop_grad = layer1.grad_weights[0][0]

    print(f"Numerical grad: {numerical_grad}, Backprop grad: {backprop_grad}")

    assert np.isclose(numerical_grad, backprop_grad, atol=1e-5)






