import numpy as np

from automatic_diff_engine.function import Function
from automatic_diff_engine.tensor_data import TensorData


class Log(Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: TensorData):
        return np.log(x.value)

    @staticmethod
    def backward(grad, *operands):
        assert len(operands) == 1
        grad = Function.unbroadcast(grad / operands[0].value, operands[0].value.shape)
        return grad
