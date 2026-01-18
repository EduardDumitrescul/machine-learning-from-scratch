from automatic_diff_engine.function import Function
from automatic_diff_engine.tensor_data import TensorData


class Reshape(Function):
    @staticmethod
    def forward(x: TensorData, shape: tuple):
        return x.value.reshape(shape)

    @staticmethod
    def backward(grad_out, *operands):
        assert len(operands) == 1
        target_shape = operands[0].value.shape
        return grad_out.reshape(target_shape)