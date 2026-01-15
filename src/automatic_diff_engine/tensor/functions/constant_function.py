from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.tensor_data import TensorData


class ConstantFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, value):
        # /self._save_operators(value)
        return value

    def backward(self, grad, *operands):
        return 1