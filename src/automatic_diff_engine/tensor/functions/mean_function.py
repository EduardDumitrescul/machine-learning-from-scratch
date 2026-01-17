import numpy as np

from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.tensor_data import TensorData


class MeanFunction(Function):
    def __init__(self):
        super(MeanFunction, self).__init__()

    @staticmethod
    def forward(tensor):
        assert isinstance(tensor, TensorData)
        return tensor.value.mean()

    @staticmethod
    def backward(grad, tensor):
        assert isinstance(tensor, TensorData)
        size = tensor.value.size
        grad = Function.unbroadcast(grad * np.ones_like(tensor.value) / size, tensor.value.shape)
        return grad