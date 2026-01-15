from abc import abstractmethod, ABC

import numpy as np

from automatic_diff_engine.tensor.tensor_data import TensorData


class Function(ABC):
    def __init__(self):
        # self._operators = []
        pass
    # def _save_operators(self, *operators):
    #     self._operators.extend(operators)

    @abstractmethod
    def forward(self, *operands):
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: np.typing.NDArray, operands: list[TensorData]) -> list[np.typing.NDArray]:
        raise NotImplementedError

