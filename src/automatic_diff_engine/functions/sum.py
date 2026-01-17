import numpy as np

from automatic_diff_engine.function import Function
from automatic_diff_engine.tensor_data import TensorData


class Sum(Function):
    @staticmethod
    def forward(tensor_data):
        return np.sum(tensor_data.value)

    @staticmethod
    def backward(grad, *operands):
        assert len(operands) == 1
        assert isinstance(operands[0], TensorData)

        return Function.unbroadcast(grad, operands[0].value.shape)
