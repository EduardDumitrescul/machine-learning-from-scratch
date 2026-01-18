from abc import ABC

import numpy as np

from automatic_diff_engine.tensor_data import TensorData


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
        # 1. Sum out leading extra dimensions
        while grad.ndim > len(target_shape):
            grad = np.sum(grad, axis=0)

        # 2. Sum out dimensions that were broadcasted from 1 to N
        for axis, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[axis] > 1:
                grad = np.sum(grad, axis=axis, keepdims=True)

        return grad

