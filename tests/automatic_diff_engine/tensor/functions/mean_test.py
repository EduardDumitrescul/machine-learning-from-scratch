import numpy as np

from automatic_diff_engine.tensor import Tensor


def test_mean_matrix():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tensor = Tensor(matrix)
    result = tensor.mean()
    assert isinstance(result, Tensor)
    assert result.value == 5
    assert result.creator_operands == [tensor]