import numpy as np

from automatic_diff_engine.tensor.tensor import Tensor

def test_create_tensor():
    tensor = Tensor(np.array([1, 2, 3]))

    assert tensor.value.tolist() == [1, 2, 3]

def test_addition_scalar():
    lfs = Tensor(np.array([1]))
    rhs = Tensor(np.array([4]))
    result = lfs + rhs

    assert result.value.tolist() == [5]

def test_addition_vector():
    lhs = Tensor(np.array([1, 2, 3]))
    rhs = Tensor(np.array([4, 5, 6]))
    result = lhs + rhs

    assert result.value.tolist() == [5, 7, 9]

def test_addition_matrices():
    lhs = Tensor(np.array([[1, 1], [1, 1]]))
    rhs = Tensor(np.array([[2, 2], [2, 2]]))
    result = lhs + rhs

    assert result.value.tolist() == [[3, 3], [3, 3]]

def test_substraction_scalar():
    lhs = Tensor(np.array([4]))
    rhs = Tensor(np.array([1]))
    result = lhs - rhs

    assert result.value.tolist() == [3]

def test_subtraction_vector():
    lhs = Tensor(np.array([4, 5, 6]))
    rhs = Tensor(np.array([1, 2, 3]))
    result = lhs - rhs

    assert result.value.tolist() == [3, 3, 3]

def test_substraction_matrices():
    lhs = Tensor(np.array([[2, 2], [2, 2]]))
    rhs = Tensor(np.array([[1, 1], [1, 1]]))
    result = lhs - rhs

    assert result.value.tolist() == [[1, 1], [1, 1]]

def test_multiplication_scalar():
    lhs = Tensor(np.array([4]))
    rhs = Tensor(np.array([1]))
    result = lhs * rhs

    assert result.value.tolist() == [4]

def test_multiplication_vector():
    lhs = Tensor(np.array([4, 5, 6]))
    rhs = Tensor(np.array([1, 2, 3]))

    result = lhs * rhs

    assert result.value.tolist() == [4, 10, 18]

def test_multiplication_matrices():
    lhs = Tensor(np.array([[2, 2], [2, 2]]))
    rhs = Tensor(np.array([[1, 1], [1, 1]]))

    result = lhs * rhs

    assert result.value.tolist() == [[2, 2], [2, 2]]

def test_equation():
    a = Tensor(np.array([1]))
    b = Tensor(np.array([4]))
    c = Tensor(np.array([7]))

    result = a + b * c

    assert result.value.tolist() == [29]

def test_addition_backwards_scalar():
    a = Tensor(np.array([1]))
    b = Tensor(np.array([4]))

    result = a + b
    result.backward()

    assert result.grad.tolist() == [1]
    assert a.grad.tolist() == [1]
    assert a.grad.tolist() == [1]

def test_addition_backwards_vector():
    a = Tensor(np.array([4, 5, 6]))
    b = Tensor(np.array([1, 2, 3]))

    result = a + b
    result.backward()

    assert result.grad.tolist() == [1, 1, 1]
    assert a.grad.tolist() == [1, 1, 1]
    assert a.grad.tolist() == [1, 1, 1]

def test_multiplication_backwards_scalar():
    a = Tensor(np.array([2]))
    b = Tensor(np.array([4]))

    result = a * b
    result.backward()

    assert result.grad.tolist() == [1]
    assert a.grad.tolist() == [4]
    assert b.grad.tolist() == [2]

def test_multiplication_backwards_vector():
    a = Tensor(np.array([2, 2, 2]))
    b = Tensor(np.array([4, 4, 4]))

    result = a * b
    result.backward()

    assert result.grad.tolist() == [1, 1, 1]
    assert a.grad.tolist() == [4, 4, 4]
    assert b.grad.tolist() == [2, 2, 2]

def test_first_grade_equation_backwards_scalar():
    a = Tensor(np.array([1]))
    b = Tensor(np.array([4]))
    x = Tensor(np.array([7]))

    y = a * x + b
    y.backward()
    assert y.grad.tolist() == [1]
    assert x.grad.tolist() == a.value.tolist()
    assert a.grad.tolist() == x.value.tolist()
    assert b.grad.tolist() == [1]

def test_first_grade_equation_backwards_vector():
    a = Tensor(np.array([1, 1]))
    b = Tensor(np.array([4, 4]))
    x = Tensor(np.array([7, 7]))

    y = a * (x + b)
    y.backward()
    assert y.grad.tolist() == [1, 1]
    assert x.grad.tolist() == a.value.tolist()
    assert a.grad.tolist() == (x+b).value.tolist()
    assert b.grad.tolist() == a.value.tolist()

def test_exponentiation_vector():
    a = Tensor(np.array([1, 2, 3]))

    result = a**2
    assert result.value.tolist() == [1, 4, 9]

def test_exponentiation_backwards_vector():
    a = Tensor(np.array([1, 2, 3]))

    result = a**2
    result.backward()

    assert a.grad.tolist() == [2, 4, 6]


