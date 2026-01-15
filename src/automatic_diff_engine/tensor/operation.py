from abc import abstractmethod, ABC

import numpy as np


class Operation(ABC):
    def __init__(self):
        self._operators = []

    def _save_operators(self, *operators):
        self._operators.extend(operators)

    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad):
        raise NotImplementedError

class Constant(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, value):
        self._save_operators(value)
        return value

    def backward(self, grad):
        return [1]

class AdditionFunction(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        self._save_operators(a, b)
        return a + b

    def backward(self, grad):
        return [grad, grad]

class MultiplicationFunction(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        self._save_operators(a, b)
        return a * b

    def backward(self, grad):
        a = self._operators[0]
        b = self._operators[1]
        return [grad * b, grad * a]


class ExponentialFunction(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, a, power):
        self._save_operators(a, power)
        return a ** power

    def backward(self, grad):
        a = self._operators[0]
        power = self._operators[1]
        return [grad * power * (a ** (power - 1))]
