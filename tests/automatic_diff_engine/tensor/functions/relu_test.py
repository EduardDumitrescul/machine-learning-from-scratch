import numpy as np
from src.automatic_diff_engine.tensor import Tensor


def test_relu_forward():
    # Test with a mix of positive, zero, and negative values
    x = Tensor(np.array([-2.0, 0.0, 3.0]))
    result = x.relu()

    # Expected: [-2 -> 0, 0 -> 0, 3 -> 3]
    assert result.value.tolist() == [0.0, 0.0, 3.0]


def test_relu_backward_basic():
    # Create input and run through relu
    x = Tensor(np.array([-10.0, 2.0]), requires_grad=True)
    result = x.relu()

    # We call backward on a sum or mean to get a scalar gradient of 1
    # or just call it on the tensor itself.
    # If result is [0, 2], and we backward, grad_output is [1, 1]
    result.backward()

    # For x = -10, grad should be 0
    # For x = 2, grad should be 1
    assert x.grad.tolist() == [0.0, 1.0]


def test_relu_chain_rule():
    # Test ReLU combined with another operation to ensure chain rule works
    # y = ReLU(x * 2)
    x = Tensor(np.array([-1.0, 2.0]), requires_grad=True)
    w = Tensor(np.array([2.0]), requires_grad=False)

    z = x * w
    y = z.relu()
    y.backward()

    # Forward: x*w = [-2, 4] -> relu([-2, 4]) = [0, 4]
    # Backward:
    # dy/dz = [0, 1] (ReLU derivative)
    # dz/dx = 2      (Multiplication derivative)
    # dy/dx = [0*2, 1*2] = [0, 2]
    assert x.grad.tolist() == [0.0, 2.0]


def test_relu_matrix():
    # Ensure it works on multi-dimensional shapes
    data = np.array([[-1, 1],
                     [2, -2]])
    x = Tensor(data, requires_grad=True)

    result = x.relu()
    result.backward()

    expected_forward = [[0, 1], [2, 0]]
    expected_grad = [[0, 1], [1, 0]]

    assert result.value.tolist() == expected_forward
    assert x.grad.tolist() == expected_grad