from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.tensor_data import TensorData


class AdditionFunction(Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(a: TensorData, b: TensorData):
        return a.value + b.value

    @staticmethod
    def backward(grad, *operands):
        grad1 = Function.unbroadcast(grad, operands[0].value.shape)
        grad2 = Function.unbroadcast(grad, operands[1].value.shape)
        return grad1, grad2