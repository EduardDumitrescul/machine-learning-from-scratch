from typing import Self, Union

import numpy as np


class TensorData:
    def __init__(self, value, grad, requires_grad):
        self.value = value
        self.grad = grad
        self.requires_grad = requires_grad

    def __eq__(self, other: Union[Self, np.ndarray]) -> bool:
        if isinstance(other, TensorData):
            return np.array_equal(self.value, other.value)
        return np.array_equal(self.value, other)