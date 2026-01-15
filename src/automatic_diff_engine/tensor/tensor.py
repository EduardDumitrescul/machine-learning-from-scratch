import numpy as np

class Tensor:
    def __init__(self, data: np.typing.NDArray):
        self.data = data

    def shape(self):
        return np.shape(self.data)
    
    def __repr__(self):
        return f"Tensor: {self.data}"

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)