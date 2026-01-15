import numpy as np

from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.tensor_data import TensorData


class ExponentialFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, base: TensorData, power: TensorData):
        return base.value ** power.value

    def backward(self, grad: np.typing.NDArray, *operands):
        assert len(operands) == 2
        assert type(operands[0]) == TensorData
        assert type(operands[1]) == TensorData

        base = operands[0]
        power = operands[1]
        return grad * power.value * (base.value ** (power.value - 1)), None