from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.tensor_data import TensorData


class AdditionFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: TensorData, b: TensorData):
        return a.value + b.value

    def backward(self, grad, *operands):
        return grad, grad