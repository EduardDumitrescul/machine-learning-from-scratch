from abc import abstractmethod, ABC

import numpy as np

from automatic_diff_engine.tensor.tensor_data import TensorData


class Function(ABC):
    def __init__(self):
        pass

    @staticmethod
    def forward(*operands):
        raise NotImplementedError

    @staticmethod
    def backward(grad: np.typing.NDArray, operands: list[TensorData]) -> list[np.typing.NDArray]:
        raise NotImplementedError

    @staticmethod
    def unbroadcast(grad, target_shape):
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

