import numpy as np

from automatic_diff_engine.tensor.tensor import Tensor


def test_forward_scalar():
    lhs = Tensor(np.array([4]))
    rhs = Tensor(np.array([1]))
    result = lhs - rhs

    assert result.value.tolist() == [3]

def test_forward_scalar_not_tensor():
    lhs = Tensor(np.array([4]))
    rhs = 4
    result = lhs - rhs
    assert (result.value == 0)

def test_forward_vector():
    lhs = Tensor(np.array([4, 5, 6]))
    rhs = Tensor(np.array([1, 2, 3]))
    result = lhs - rhs

    assert result.value.tolist() == [3, 3, 3]

def test_forward_matrices():
    lhs = Tensor(np.array([[2, 2], [2, 2]]))
    rhs = Tensor(np.array([[1, 1], [1, 1]]))
    result = lhs - rhs

    assert result.value.tolist() == [[1, 1], [1, 1]]

def test_backward_scalar():
    a = Tensor(1)
    b = Tensor(4)

    result = a - b
    result.backward()

    assert result.grad.tolist() == [1]
    assert a.grad.tolist() == [1]
    assert b.grad.tolist() == [-1]

def test_backwards_vector():
    a = Tensor([4, 5, 6])
    b = Tensor([1, 2, 3])

    result = a - b
    result.backward()

    assert result.grad.tolist() == [1, 1, 1]
    assert a.grad.tolist() == [1, 1, 1]
    assert b.grad.tolist() == [-1, -1, -1]