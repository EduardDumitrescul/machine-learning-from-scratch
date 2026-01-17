import numpy as np

from automatic_diff_engine.function import Function
from automatic_diff_engine.tensor_data import TensorData


class ExponentialFunction(Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(base: TensorData, power: TensorData):
        return base.value ** power.value

    @staticmethod
    def backward(grad: np.typing.NDArray, *operands):
        assert len(operands) == 2
        assert type(operands[0]) == TensorData
        assert type(operands[1]) == TensorData

        base = operands[0]
        power = operands[1]
        base_grad = Function.unbroadcast(grad * power.value * (base.value ** (power.value - 1)), base.value.shape)
        return base_grad, None