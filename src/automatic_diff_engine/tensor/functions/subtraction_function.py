from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.tensor_data import TensorData


class SubtractionFunction(Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a: TensorData, b: TensorData):
        return a.value - b.value

    @staticmethod
    def backward(grad, *operands):
        return grad, -grad