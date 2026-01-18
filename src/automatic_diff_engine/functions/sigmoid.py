import numpy as np

from automatic_diff_engine.function import Function
from automatic_diff_engine.tensor_data import TensorData


class Sigmoid(Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: TensorData):
        return 1.0 / (1 + np.exp(-x.value))

    @staticmethod
    def backward(grad, *operands):
        assert len(operands) == 1
        assert isinstance(operands[0], TensorData)

        sigmoid = Sigmoid.forward(operands[0])

        grad = Function.unbroadcast(grad * sigmoid * (1 - sigmoid), operands[0].value.shape)
        return grad
