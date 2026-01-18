import numpy as np
from automatic_diff_engine.tensor import Tensor


def test_reshape_forward():
    # Arrange: Create a 1D vector of 4 elements
    a = Tensor(np.array([1, 2, 3, 4]))

    # Act: Reshape to (2, 2)
    result = a.reshape((2, 2))

    # Assert
    assert result.shape == (2, 2)
    assert result.value.tolist() == [[1, 2], [3, 4]]


def test_reshape_backward():
    # Arrange: Create a vector that will be reshaped
    # We use a simple operation after reshape to ensure gradients flow back
    a = Tensor(np.array([1, 2, 3, 4]))  # Shape (4,)

    # Act
    b = a.reshape((2, 2))  # Shape (2, 2)
    # Perform an operation that returns a scalar (sum) to start backward
    result = b.sum()
    result.backward()

    # Assert
    # The gradient of sum() is all ones in the shape of (2, 2)
    assert b.grad.shape == (2, 2)
    assert b.grad.tolist() == [[1, 1], [1, 1]]

    # The gradient of a should be 'un-reshaped' back to (4,)
    assert a.grad.shape == (4,)
    assert a.grad.tolist() == [1, 1, 1, 1]


def test_reshape_complex_chain():
    # Testing: a -> reshape -> multiply by constant -> sum
    a = Tensor(np.array([[1, 2], [3, 4]]))  # Shape (2, 2)

    # Flatten to (4,)
    b = a.reshape((4,))

    # Multiply by a constant to scale gradients
    # Assuming MultiplicationFunction is implemented
    c = b * Tensor(np.array([10, 20, 30, 40]), requires_grad=False)

    result = c.sum()
    result.backward()

    # Assert
    # Gradient at 'c' is [1, 1, 1, 1]
    # Gradient at 'b' (after mul) is [10, 20, 30, 40]
    # Gradient at 'a' should be [[10, 20], [30, 40]]
    assert a.grad.shape == (2, 2)
    assert a.grad.tolist() == [[10.0, 20.0], [30.0, 40.0]]