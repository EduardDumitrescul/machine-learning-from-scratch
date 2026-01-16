import numpy as np

from automatic_diff_engine.tensor.tensor import Tensor


def test_forward_scalar():
    lhs = Tensor(np.array([4]))
    rhs = Tensor(np.array([1]))
    result = lhs * rhs

    assert result.value.tolist() == [4]

def test_forward_vector():
    lhs = Tensor(np.array([4, 5, 6]))
    rhs = Tensor(np.array([1, 2, 3]))

    result = lhs * rhs

    assert result.value.tolist() == [4, 10, 18]


def test_forward_matrices():
    lhs = Tensor(np.array([[2, 2], [2, 2]]))
    rhs = Tensor(np.array([[1, 1], [1, 1]]))

    result = lhs * rhs

    assert result.value.tolist() == [[2, 2], [2, 2]]


def test_backwards_scalar():
    a = Tensor(np.array([2]))
    b = Tensor(np.array([4]))

    result = a * b
    result.backward()

    assert result.grad.tolist() == [1]
    assert a.grad.tolist() == [4]
    assert b.grad.tolist() == [2]


def test_backwards_vector():
    a = Tensor(np.array([2, 2, 2]))
    b = Tensor(np.array([4, 4, 4]))

    result = a * b
    result.backward()

    assert result.grad.tolist() == [1, 1, 1]
    assert a.grad.tolist() == [4, 4, 4]
    assert b.grad.tolist() == [2, 2, 2]