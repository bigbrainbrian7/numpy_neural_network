from typing import cast

import numpy as np
import pytest
from layer import Linear, ReLU
from model import Model
from loss import SoftmaxCrossEntropyLoss
 
# ── helpers ───────────────────────────────────────────────────────────────────
 
def make_linear(input_size, output_size, seed=0):
    np.random.seed(seed)
    return Linear(input_size, output_size)
 
def make_model(seed=0):
    np.random.seed(seed)
    return Model(Linear(4, 8), ReLU(), Linear(8, 3))
 
def make_batch(n_samples, n_features, n_classes, seed=0):
    np.random.seed(seed)
    x = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_classes, n_samples)
    y = np.zeros((n_samples, n_classes))
    y[np.arange(n_samples), labels] = 1
    return x, y, labels
 
def compute_loss(model, loss_fn, x, y):
    out = model.forward(x)
    probs = loss_fn.forward(out)
    eps = 1e-9
    return -np.sum(y * np.log(probs + eps)) / x.shape[0]
 
# ── linear forward ────────────────────────────────────────────────────────────
 
def test_linear_forward_shape():
    layer = make_linear(10, 5)
    x = np.random.randn(3, 10)
    out = layer.forward(x)
    assert out.shape == (3, 5)
 
def test_linear_forward_values():
    layer = make_linear(3, 2)
    x = np.ones((2, 3))
    out = layer.forward(x)
    expected = x @ layer.weights + layer.biases
    np.testing.assert_allclose(out, expected)
 
def test_linear_forward_stores_input():
    layer = make_linear(3, 2)
    x = np.random.randn(4, 3)
    layer.forward(x)
    np.testing.assert_array_equal(layer.input, x)
 
def test_linear_bias_shape():
    layer = make_linear(3, 2)
    assert layer.biases.shape == (1, 2)
 
# ── linear backward ───────────────────────────────────────────────────────────
 
def test_linear_backward_upstream_shape():
    layer = make_linear(4, 3)
    x = np.random.randn(5, 4)
    layer.forward(x)
    upstream = np.random.randn(5, 3)
    result = layer.backward(upstream)
    assert result.shape == (5, 4)
 
def test_linear_backward_grad_weights_shape():
    layer = make_linear(4, 3)
    x = np.random.randn(5, 4)
    layer.forward(x)
    layer.backward(np.random.randn(5, 3))
    assert layer.grad_weights.shape == (4, 3)
 
def test_linear_backward_grad_biases_shape():
    layer = make_linear(4, 3)
    x = np.random.randn(5, 4)
    layer.forward(x)
    layer.backward(np.random.randn(5, 3))
    assert layer.grad_biases.shape == (1, 3)
 
def test_linear_backward_grad_weights_values():
    layer = make_linear(3, 2)
    x = np.random.randn(4, 3)
    layer.forward(x)
    upstream = np.random.randn(4, 2)
    layer.backward(upstream)
    expected = x.T @ upstream
    np.testing.assert_allclose(layer.grad_weights, expected)
 
def test_linear_backward_grad_biases_values():
    layer = make_linear(3, 2)
    x = np.random.randn(4, 3)
    layer.forward(x)
    upstream = np.random.randn(4, 2)
    layer.backward(upstream)
    expected = upstream.sum(axis=0, keepdims=True)
    np.testing.assert_allclose(layer.grad_biases, expected)
 
def test_linear_backward_upstream_values():
    layer = make_linear(3, 2)
    x = np.random.randn(4, 3)
    layer.forward(x)
    upstream = np.random.randn(4, 2)
    result = layer.backward(upstream)
    expected = upstream @ layer.weights.T
    np.testing.assert_allclose(result, expected)
 
# ── gradient accumulation ─────────────────────────────────────────────────────
 
def test_grad_accumulates_across_backward_calls():
    layer = make_linear(3, 2)
    x = np.random.randn(2, 3)
    upstream = np.random.randn(2, 2)
 
    layer.forward(x)
    layer.backward(upstream)
    grad_after_first = layer.grad_weights.copy()
 
    layer.forward(x)
    layer.backward(upstream)
    grad_after_second = layer.grad_weights.copy()
 
    np.testing.assert_allclose(grad_after_second, 2 * grad_after_first)
 
def test_zero_grad_resets_weights():
    layer = make_linear(3, 2)
    x = np.random.randn(2, 3)
    layer.forward(x)
    layer.backward(np.random.randn(2, 2))
    layer.zero_grad()
    np.testing.assert_array_equal(layer.grad_weights, np.zeros_like(layer.weights))
    np.testing.assert_array_equal(layer.grad_biases, np.zeros_like(layer.biases))
 
# ── relu ──────────────────────────────────────────────────────────────────────
 
def test_relu_forward_zeros_negatives():
    relu = ReLU()
    x = np.array([[-1.0, 2.0, -3.0, 4.0]])
    out = relu.forward(x)
    np.testing.assert_array_equal(out, [[0.0, 2.0, 0.0, 4.0]])
 
def test_relu_forward_shape():
    relu = ReLU()
    x = np.random.randn(5, 10)
    assert relu.forward(x).shape == (5, 10)
 
def test_relu_backward_gates_gradient():
    relu = ReLU()
    x = np.array([[-1.0, 2.0, -3.0, 4.0]])
    relu.forward(x)
    upstream = np.ones((1, 4))
    result = relu.backward(upstream)
    np.testing.assert_array_equal(result, [[0.0, 1.0, 0.0, 1.0]])
 
def test_relu_backward_shape():
    relu = ReLU()
    x = np.random.randn(5, 10)
    relu.forward(x)
    result = relu.backward(np.random.randn(5, 10))
    assert result.shape == (5, 10)
 
# ── softmax cross entropy loss ────────────────────────────────────────────────
 
def test_softmax_probs_sum_to_one():
    loss_fn = SoftmaxCrossEntropyLoss()
    x = np.random.randn(5, 10)
    probs = loss_fn.forward(x)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-6)
 
def test_softmax_probs_all_positive():
    loss_fn = SoftmaxCrossEntropyLoss()
    x = np.random.randn(5, 10)
    probs = loss_fn.forward(x)
    assert np.all(probs > 0)
 
def test_softmax_output_shape():
    loss_fn = SoftmaxCrossEntropyLoss()
    x = np.random.randn(5, 10)
    probs = loss_fn.forward(x)
    assert probs.shape == (5, 10)
 
def test_softmax_numerical_stability():
    """large scores should not overflow"""
    loss_fn = SoftmaxCrossEntropyLoss()
    x = np.random.randn(3, 5) * 1000
    probs = loss_fn.forward(x)
    assert not np.any(np.isnan(probs))
    assert not np.any(np.isinf(probs))
 
def test_softmax_backward_shape():
    loss_fn = SoftmaxCrossEntropyLoss()
    x = np.random.randn(5, 3)
    y = np.zeros((5, 3)); y[np.arange(5), np.random.randint(0,3,5)] = 1
    loss_fn.forward(x)
    upstream = loss_fn.backward(y)
    assert upstream.shape == (5, 3)
 
def test_softmax_backward_correct_class_negative():
    """gradient for correct class should be negative (score needs to increase)"""
    loss_fn = SoftmaxCrossEntropyLoss()
    x = np.array([[1.0, 2.0, 3.0]])
    y = np.array([[0.0, 0.0, 1.0]])  # correct class is 2
    loss_fn.forward(x)
    grad = loss_fn.backward(y)
    assert grad[0, 2] < 0   # correct class gradient negative
    assert grad[0, 0] > 0   # wrong class gradient positive
    assert grad[0, 1] > 0
 
def test_softmax_backward_sums_to_zero():
    """gradients should sum to zero per sample"""
    loss_fn = SoftmaxCrossEntropyLoss()
    x = np.random.randn(4, 5)
    y = np.zeros((4, 5)); y[np.arange(4), np.random.randint(0,5,4)] = 1
    loss_fn.forward(x)
    grad = loss_fn.backward(y)
    np.testing.assert_allclose(grad.sum(axis=1), np.zeros(4), atol=1e-6)
 
# ── gradient check (numerical vs analytical) ──────────────────────────────────
 
def test_gradient_check_linear_single_layer():
    np.random.seed(42)
    layer = make_linear(3, 2)
    loss_fn = SoftmaxCrossEntropyLoss()
    x, y, _ = make_batch(4, 3, 2)
 
    # analytical
    loss_fn.forward(layer.forward(x))
    upstream = loss_fn.backward(y)
    layer.backward(upstream)
    analytical = layer.grad_weights[0, 0]
 
    # numerical
    eps = 1e-5
    layer.weights[0, 0] += eps
    loss_plus = compute_loss(Model(layer), loss_fn, x, y)
    layer.weights[0, 0] -= 2 * eps
    loss_minus = compute_loss(Model(layer), loss_fn, x, y)
    layer.weights[0, 0] += eps
    numerical = (loss_plus - loss_minus) / (2 * eps)
 
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4)
 
def test_gradient_check_full_model():
    """numerical gradient check through linear -> relu -> linear -> softmax"""
    np.random.seed(42)
    model = make_model()
    loss_fn = SoftmaxCrossEntropyLoss()
    x, y, _ = make_batch(8, 4, 3)
 
    # analytical gradient for first layer weight
    loss_fn.forward(model.forward(x))
    upstream = loss_fn.backward(y)
    model.backward(upstream)
    first_layer = cast(Linear, model.layers[0])
    analytical = first_layer.grad_weights[0, 0]
 
    # numerical gradient
    eps = 1e-5
    first_layer.weights[0, 0] += eps
    loss_plus = compute_loss(model, loss_fn, x, y)
    first_layer.weights[0, 0] -= 2 * eps
    loss_minus = compute_loss(model, loss_fn, x, y)
    first_layer.weights[0, 0] += eps
    numerical = (loss_plus - loss_minus) / (2 * eps)
 
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4,
        err_msg=f"analytical={analytical:.8f} numerical={numerical:.8f}")
 
def test_gradient_check_bias():
    np.random.seed(42)
    layer = make_linear(3, 2)
    loss_fn = SoftmaxCrossEntropyLoss()
    x, y, _ = make_batch(4, 3, 2)
 
    loss_fn.forward(layer.forward(x))
    upstream = loss_fn.backward(y)
    layer.backward(upstream)
    analytical = layer.grad_biases[0, 0]
 
    eps = 1e-5
    layer.biases[0, 0] += eps
    loss_plus = compute_loss(Model(layer), loss_fn, x, y)
    layer.biases[0, 0] -= 2 * eps
    loss_minus = compute_loss(Model(layer), loss_fn, x, y)
    layer.biases[0, 0] += eps
    numerical = (loss_plus - loss_minus) / (2 * eps)
 
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4)
 
# ── model ─────────────────────────────────────────────────────────────────────
 
def test_model_forward_shape():
    model = make_model()
    x = np.random.randn(5, 4)
    out = model.forward(x)
    assert out.shape == (5, 3)
 
def test_model_zero_grad_clears_all_layers():
    model = make_model()
    x, y, _ = make_batch(4, 4, 3)
    loss_fn = SoftmaxCrossEntropyLoss()
    loss_fn.forward(model.forward(x))
    model.backward(loss_fn.backward(y))
    model.zero_grad()
    for layer in model.layers:
        if isinstance(layer, Linear):
            np.testing.assert_array_equal(layer.grad_weights, np.zeros_like(layer.weights))
 
def test_model_step_updates_weights():
    model = make_model()
    x, y, _ = make_batch(4, 4, 3)
    loss_fn = SoftmaxCrossEntropyLoss()
 
    first_layer = cast(Linear, model.layers[0])
    weights_before = first_layer.weights.copy()
    loss_fn.forward(model.forward(x))
    model.backward(loss_fn.backward(y))
    model.step(learning_rate=0.1)
 
    assert not np.allclose(first_layer.weights, weights_before)
 
# ── training sanity check ─────────────────────────────────────────────────────
 
def test_loss_decreases_after_training():
    """loss should go down over several gradient steps"""
    np.random.seed(0)
    model = make_model()
    loss_fn = SoftmaxCrossEntropyLoss()
    x, y, _ = make_batch(16, 4, 3)
 
    losses = []
    for _ in range(50):
        model.zero_grad()
        probs = loss_fn.forward(model.forward(x))
        eps = 1e-9
        losses.append(-np.sum(y * np.log(probs + eps)) / x.shape[0])
        model.backward(loss_fn.backward(y))
        model.step(learning_rate=0.1)
 
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
 
def test_model_can_overfit_small_batch():
    """model should be able to get near perfect accuracy on a tiny dataset"""
    np.random.seed(0)
    model = Model(Linear(4, 16), ReLU(), Linear(16, 3))
    loss_fn = SoftmaxCrossEntropyLoss()
    x, y, labels = make_batch(8, 4, 3)
 
    for _ in range(500):
        model.zero_grad()
        loss_fn.forward(model.forward(x))
        model.backward(loss_fn.backward(y))
        model.step(learning_rate=0.05)
 
    probs = loss_fn.forward(model.forward(x))
    preds = probs.argmax(axis=1)
    accuracy = (preds == labels).mean()
    assert accuracy > 0.9, f"expected >90% on tiny batch, got {accuracy:.2f}"
 
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
