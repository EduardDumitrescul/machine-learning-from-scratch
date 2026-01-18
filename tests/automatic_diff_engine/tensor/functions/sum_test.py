import numpy as np
from src.automatic_diff_engine.tensor import Tensor


def test_sum_forward():
    # Test summing a 1D vector
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    result = x.sum()
    assert result.value.item() == 6.0

    # Test summing a 2D matrix
    x_mat = Tensor(np.array([[1, 2], [3, 4]]))
    result_mat = x_mat.sum()
    assert result_mat.value.item() == 10.0


def test_sum_backward_vector():
    # If y = x1 + x2 + x3, then dy/dx1 = 1, dy/dx2 = 1, dy/dx3 = 1
    x = Tensor(np.array([10.0, 20.0, 30.0]), requires_grad=True)
    y = x.sum()
    y.backward()

    # The gradient of the sum with respect to any input element is 1
    assert x.grad.shape == (3,)
    assert x.grad.tolist() == [1.0, 1.0, 1.0]


def test_sum_backward_matrix():
    # Ensure gradient broadcasts to the original matrix shape
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    y = x.sum()
    y.backward()

    assert x.grad.shape == (2, 3)
    assert np.all(x.grad == 1.0)