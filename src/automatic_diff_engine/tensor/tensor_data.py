import numpy as np


class TensorData:
    def __init__(self, value, grad, requires_grad):
        self.value = np.array(value)
        self.grad = np.array(grad)
        self.requires_grad = requires_grad