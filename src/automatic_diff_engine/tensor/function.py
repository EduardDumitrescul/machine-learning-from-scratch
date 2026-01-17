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

