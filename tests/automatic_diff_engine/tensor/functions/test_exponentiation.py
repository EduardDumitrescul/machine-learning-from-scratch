import numpy as np

from automatic_diff_engine.tensor import Tensor


def test_forward_vector():
    a = Tensor(np.array([1, 2, 3]))

    result = a**2
    assert result.value.tolist() == [1, 4, 9]


def test_backward_vector():
    a = Tensor(np.array([1, 2, 3]))

    result = a**2
    result.backward()

    assert a.grad.tolist() == [2, 4, 6]