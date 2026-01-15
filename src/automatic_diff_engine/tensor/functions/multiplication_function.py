from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.tensor_data import TensorData


class MultiplicationFunction(Function):
    def __init__(self):
        super().__init__()

    def forward(self, a: TensorData, b: TensorData):
        return a.value * b.value

    def backward(self, grad, *operands):
        assert len(operands) == 2
        assert isinstance(operands[0], TensorData)
        assert isinstance(operands[1], TensorData)

        return grad * operands[1].value, grad * operands[0].value