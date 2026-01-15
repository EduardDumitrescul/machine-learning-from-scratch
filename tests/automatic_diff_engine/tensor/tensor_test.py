import numpy as np

from automatic_diff_engine.tensor.tensor import Tensor

def test_create_tensor():
    tensor = Tensor(np.array([1, 2, 3]))

    assert tensor.shape() == (3, )
    assert tensor.data.tolist() == [1, 2, 3]

def test_addition_scalar():
    lfs = Tensor(np.array([1]))
    rhs = Tensor(np.array([4]))
    result = lfs + rhs

    assert result.shape() == (1, )
    assert result.data.tolist() == [5]

def test_addition_vector():
    lhs = Tensor(np.array([1, 2, 3]))
    rhs = Tensor(np.array([4, 5, 6]))
    result = lhs + rhs

    assert result.shape() == (3, )
    assert result.data.tolist() == [5, 7, 9]

def test_addition_matrices():
    lhs = Tensor(np.array([[1, 1], [1, 1]]))
    rhs = Tensor(np.array([[2, 2], [2, 2]]))
    result = lhs + rhs

    assert result.shape() == (2, 2, )
    assert result.data.tolist() == [[3, 3], [3, 3]]

def test_substraction_scalar():
    lhs = Tensor(np.array([4]))
    rhs = Tensor(np.array([1]))
    result = lhs - rhs

    assert result.shape() == (1, )
    assert result.data.tolist() == [3]

def test_subtraction_vector():
    lhs = Tensor(np.array([4, 5, 6]))
    rhs = Tensor(np.array([1, 2, 3]))
    result = lhs - rhs

    assert result.shape() == (3, )
    assert result.data.tolist() == [3, 3, 3]

def test_substraction_matrices():
    lhs = Tensor(np.array([[2, 2], [2, 2]]))
    rhs = Tensor(np.array([[1, 1], [1, 1]]))
    result = lhs - rhs

    assert result.shape() == (2, 2, )
    assert result.data.tolist() == [[1, 1], [1, 1]]

def test_multiplication_scalar():
    lhs = Tensor(np.array([4]))
    rhs = Tensor(np.array([1]))
    result = lhs * rhs

    assert result.shape() == (1, )
    assert result.data.tolist() == [4]

def test_multiplication_vector():
    lhs = Tensor(np.array([4, 5, 6]))
    rhs = Tensor(np.array([1, 2, 3]))

    result = lhs * rhs

    assert result.shape() == (3, )
    assert result.data.tolist() == [4, 10, 18]

def test_multiplication_matrices():
    lhs = Tensor(np.array([[2, 2], [2, 2]]))
    rhs = Tensor(np.array([[1, 1], [1, 1]]))

    result = lhs * rhs

    assert result.shape() == (2, 2, )
    assert result.data.tolist() == [[2, 2], [2, 2]]

