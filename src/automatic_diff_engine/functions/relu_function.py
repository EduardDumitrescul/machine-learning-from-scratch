import numpy as np

from automatic_diff_engine.function import Function
from automatic_diff_engine.tensor_data import TensorData


class ReLU(Function):
    def __init__(self):
        super(ReLU, self).__init__()

    @staticmethod
    def forward(x: TensorData):
        return np.maximum(0, x.value)

    @staticmethod
    def backward(grad, *operands):
        assert len(operands) == 1
        assert isinstance(operands[0], TensorData)

        mask = operands[0].value > 0
        grad = Function.unbroadcast(grad * mask, operands[0].value.shape)
        return grad
