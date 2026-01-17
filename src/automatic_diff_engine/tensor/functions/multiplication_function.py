from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.tensor_data import TensorData


class MultiplicationFunction(Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a: TensorData, b: TensorData):
        return a.value * b.value

    @staticmethod
    def backward(grad, *operands):
        assert len(operands) == 2
        assert isinstance(operands[0], TensorData)
        assert isinstance(operands[1], TensorData)

        grad1 = Function.unbroadcast(grad * operands[1].value, operands[0].value.shape)
        grad2 = Function.unbroadcast(grad * operands[0].value, operands[1].value.shape)

        return grad1, grad2