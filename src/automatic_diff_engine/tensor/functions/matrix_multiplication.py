from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.tensor_data import TensorData


class MatrixMultiplication(Function):
    @staticmethod
    def forward(a: TensorData, b: TensorData) -> TensorData:
        return a.value @ b.value

    @staticmethod
    def backward(grad, *operands):
        assert len(operands) == 2
        assert isinstance(operands[0], TensorData)
        assert isinstance(operands[1], TensorData)
        return grad @ operands[1].value.T, operands[0].value.T @ grad
