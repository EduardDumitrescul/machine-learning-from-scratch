import numpy as np
from src.automatic_diff_engine.tensor import Tensor


def test_log_forward():
    # Test natural log values
    x = Tensor(np.array([1.0, np.e, np.exp(2)]))
    result = x.log()

    # Expected: [ln(1)=0, ln(e)=1, ln(e^2)=2]
    assert np.allclose(result.value, [0.0, 1.0, 2.0])


def test_log_backward_basic():
    # Gradient of ln(x) is 1/x
    x = Tensor(np.array([2.0, 10.0]), requires_grad=True)
    result = x.log()
    result.backward()

    # Expected gradients: [1/2, 1/10]
    assert np.allclose(x.grad, [0.5, 0.1])


def test_log_chain_rule():
    # y = ln(x^2) -> dy/dx = (1/x^2) * 2x = 2/x
    # For x = 4, dy/dx = 2/4 = 0.5
    x = Tensor(np.array([4.0]), requires_grad=True)

    # Simulating x^2 as x * x
    z = x * x
    y = z.log()
    y.backward()

    assert np.allclose(x.grad, [0.5])


def test_log_matrix():
    # Ensure element-wise log and grad work on matrices
    data = np.array([[1.0, 2.0], [4.0, 8.0]])
    x = Tensor(data, requires_grad=True)

    result = x.log()
    result.backward()

    expected_forward = np.log(data)
    expected_grad = 1.0 / data

    assert np.allclose(result.value, expected_forward)
    assert np.allclose(x.grad, expected_grad)