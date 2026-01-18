import numpy as np
from src.automatic_diff_engine.tensor import Tensor


def test_sigmoid_forward():
    # Test with negative, zero, and positive values
    x = Tensor(np.array([-100.0, 0.0, 100.0]))
    result = x.sigmoid()

    # Expected approx: [0.0, 0.5, 1.0]
    assert np.allclose(result.value, [0.0, 0.5, 1.0], atol=1e-7)


def test_sigmoid_backward_basic():
    # Gradient of sigmoid at x=0 is 0.25
    x = Tensor(np.array([0.0]), requires_grad=True)
    result = x.sigmoid()
    result.backward()

    # grad = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    assert np.allclose(x.grad, [0.25])


def test_sigmoid_chain_rule():
    # y = sigmoid(x * 2)
    x = Tensor(np.array([0.0]), requires_grad=True)
    w = Tensor(np.array([2.0]), requires_grad=False)

    z = x * w
    y = z.sigmoid()
    y.backward()

    # Forward: 0 * 2 = 0 -> sigmoid(0) = 0.5
    # Backward: dy/dz = 0.25, dz/dx = 2 -> dy/dx = 0.5
    assert np.allclose(x.grad, [0.5])


def test_sigmoid_matrix():
    # Multi-dimensional shape test
    data = np.array([[0.0, 1.0], [-1.0, 2.0]])
    x = Tensor(data, requires_grad=True)

    result = x.sigmoid()
    result.backward()

    # Calculate expected values using numpy for comparison
    def sig(v): return 1 / (1 + np.exp(-v))

    expected_forward = sig(data)
    expected_grad = expected_forward * (1 - expected_forward)

    assert np.allclose(result.value, expected_forward)
    assert np.allclose(x.grad, expected_grad)