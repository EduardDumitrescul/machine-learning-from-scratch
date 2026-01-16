import numpy as np

from automatic_diff_engine.tensor.tensor import Tensor

def test_create_tensor():
    tensor = Tensor(np.array([1, 2, 3]))

    assert tensor.value.tolist() == [1, 2, 3]

def test_forward_equation():
    a = Tensor(np.array([1]))
    b = Tensor(np.array([4]))
    c = Tensor(np.array([7]))

    result = a + b * c

    assert result.value.tolist() == [29]

def test_forward_matrix_equation():
    matrix1 = np.array([[1, 2], [3, 4], [5, 6]])
    matrix2 = np.array([[7, 8], [9, 10], [11, 11]])
    matrix3 = np.array([[12, 13, 14], [15, 16, 17]])
    tensor1 = Tensor(matrix1)
    tensor2 = Tensor(matrix2)
    tensor3 = Tensor(matrix3)

    result = (tensor1 + tensor2) @ tensor3
    assert (result.value == (matrix1 + matrix2) @ matrix3).all()

def test_backward_scalar_equation():
    a = Tensor(np.array([1]))
    b = Tensor(np.array([4]))
    x = Tensor(np.array([7]))

    y = a * x + b
    y.backward()
    assert y.grad.tolist() == [1]
    assert x.grad.tolist() == a.value.tolist()
    assert a.grad.tolist() == x.value.tolist()
    assert b.grad.tolist() == [1]

def test_backward_vector_equation():
    a = Tensor(np.array([1, 1]))
    b = Tensor(np.array([4, 4]))
    x = Tensor(np.array([7, 7]))

    y = a * (x + b)
    y.backward()
    assert y.grad.tolist() == [1, 1]
    assert x.grad.tolist() == a.value.tolist()
    assert a.grad.tolist() == (x+b).value.tolist()
    assert b.grad.tolist() == a.value.tolist()

def test_backward_matrix_equation():
    matrix1 = np.array([[1, 2], [3, 4], [5, 6]])
    matrix2 = np.array([[7, 8], [9, 10], [11, 11]])
    matrix3 = np.array([[12, 13, 14], [15, 16, 17]])
    tensor1 = Tensor(matrix1)
    tensor2 = Tensor(matrix2)
    tensor3 = Tensor(matrix3)

    result = (tensor1 + tensor2) @ tensor3

    assert (tensor1.grad == result.grad @ matrix3.T).all()
    assert (tensor2.grad == result.grad @ matrix3.T).all()
    assert (tensor3.grad == (matrix1.T + matrix2.T) @ result.grad).all()






