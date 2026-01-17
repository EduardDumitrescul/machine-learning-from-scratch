import numpy as np

from automatic_diff_engine.tensor.tensor import Tensor


def test_forward_scalar():
    lfs = Tensor(np.array([1]))
    rhs = Tensor(np.array([4]))
    result = lfs + rhs

    assert result.value.tolist() == [5]

def test_forward_vector():
    lhs = Tensor(np.array([1, 2, 3]))
    rhs = Tensor(np.array([4, 5, 6]))
    result = lhs + rhs

    assert result.value.tolist() == [5, 7, 9]

def test_forward_matrices():
    lhs = Tensor(np.array([[1, 1], [1, 1]]))
    rhs = Tensor(np.array([[2, 2], [2, 2]]))
    result = lhs + rhs

    assert result.value.tolist() == [[3, 3], [3, 3]]


def test_backwards_scalar():
    a = Tensor(np.array([1]))
    b = Tensor(np.array([4]))

    result = a + b
    result.backward()

    assert result.grad.tolist() == [1]
    assert a.grad.tolist() == [1]
    assert a.grad.tolist() == [1]

def test_backwards_vector():
    a = Tensor(np.array([4, 5, 6]))
    b = Tensor(np.array([1, 2, 3]))

    result = a + b
    result.backward()

    assert result.grad.tolist() == [1, 1, 1]
    assert a.grad.tolist() == [1, 1, 1]
    assert a.grad.tolist() == [1, 1, 1]