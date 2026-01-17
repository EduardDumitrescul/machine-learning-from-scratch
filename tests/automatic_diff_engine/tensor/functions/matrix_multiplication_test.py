import numpy as np

from automatic_diff_engine.tensor import Tensor


def test_forward():
    matrix1 = np.array([[1, 2], [4, 5], [7, 8]])
    matrix2 = np.array([[4, 5, 6], [7, 8, 9]])

    tensor1 = Tensor(matrix1)
    tensor2 = Tensor(matrix2)

    result = tensor1 @ tensor2
    assert (result.value == matrix1 @ matrix2).all()

def test_forward_triple():
    matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrix2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrix3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    tensor1 = Tensor(matrix1)
    tensor2 = Tensor(matrix2)
    tensor3 = Tensor(matrix3)

    result = tensor1 @ tensor2 @ tensor3
    assert (result.value == matrix1 @ matrix2 @ matrix3).all()

def test_backward():
    matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
    matrix2 = np.array([[1, 2], [4, 5], [7, 8]])

    tensor1 = Tensor(matrix1)
    tensor2 = Tensor(matrix2)

    result = tensor1 @ tensor2
    result.backward()

    assert (result.grad == np.ones_like(result.grad)).all()
    assert (tensor1.grad == np.ones_like(result.grad) @ matrix2.T).all()
    assert (tensor2.grad == matrix1.T @ np.ones_like(result.grad)).all()


def test_backward_triple():
    # Setup sequential matrices
    A_mat = np.arange(6).reshape(2, 3)  # [[0, 1, 2], [3, 4, 5]]
    B_mat = np.arange(12).reshape(3, 4)  # [[0..3], [4..7], [8..11]]
    C_mat = np.arange(8).reshape(4, 2)  # [[0, 1], [2, 3], [4, 5], [6, 7]]

    A = Tensor(A_mat)
    B = Tensor(B_mat)
    C = Tensor(C_mat)

    # Forward pass: Y = (A @ B) @ C
    res = A @ B @ C
    res.backward()

    # Outgoing gradient G (dL/dY)
    G = res.grad

    # Pre-compute intermediate gradient dL/dZ where Z = A @ B
    # Since Y = Z @ C, dL/dZ = G @ C.T
    grad_Z = G @ C_mat.T

    # Manual Gradient Calculation
    expected_grad_A = grad_Z @ B_mat.T
    expected_grad_B = A_mat.T @ grad_Z
    expected_grad_C = (A_mat @ B_mat).T @ G

    assert np.allclose(A.grad, expected_grad_A)
    assert np.allclose(B.grad, expected_grad_B)
    assert np.allclose(C.grad, expected_grad_C)